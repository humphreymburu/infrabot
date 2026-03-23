"""
Simplified analysis pipeline.
Flow: compress → 4 specialists (parallel) → evaluator → optional revision → synthesis.

~300 lines vs ~2800 in the original.
"""
import asyncio
import json
import logging
import re
import uuid
import datetime
from typing import Any, AsyncIterator

from app.llm import llm_text, route_for_agent
from app.prompts import (
    AGENT_PROMPTS,
    AGENT_BRIEF_KEYS,
    COMPRESS_PROMPT,
    EVALUATOR_PROMPT,
    SYNTHESIS_PROMPT,
    IMPLEMENTATION_PROMPT,
)
from app.search import research, search_available, format_search_context

logger = logging.getLogger(__name__)


# ── JSON parsing ────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict[str, Any]:
    """Parse JSON from LLM output, stripping markdown fences."""
    if not raw or not raw.strip():
        return {"_error": "empty_response"}
    clean = re.sub(r"```json\s*|```\s*", "", raw).strip()
    # Try direct parse
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    # Try extracting the outermost {...}
    start = clean.find("{")
    end = clean.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(clean[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {"_error": "parse_failed", "_raw_preview": clean[:500]}


def _slim(obj: Any, max_str: int = 400, max_list: int = 5) -> Any:
    """Recursively trim large payloads to stay within context budgets."""
    if isinstance(obj, str):
        return obj[:max_str] + "…" if len(obj) > max_str else obj
    if isinstance(obj, list):
        return [_slim(x, max_str, max_list) for x in obj[:max_list]]
    if isinstance(obj, dict):
        return {k: _slim(v, max_str, max_list) for k, v in obj.items()}
    return obj


# ── Single agent call ───────────────────────────────────────────────

async def _run_agent(
    agent_key: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int = 3500,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Run one agent. Returns parsed JSON dict or error envelope."""
    try:
        text, route = await llm_text(
            agent_key=agent_key,
            system_prompt=system_prompt,
            user_message=user_message,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as e:
        return {"_error": "llm_failed", "_detail": str(e)[:300]}

    if not text:
        return {"_error": "empty_output"}

    parsed = _parse_json(text)

    # One repair attempt if parse failed
    if parsed.get("_error") == "parse_failed":
        try:
            repair_text, _ = await llm_text(
                agent_key=agent_key,
                system_prompt="Fix this malformed JSON. Return ONLY valid JSON.",
                user_message=f"Malformed output:\n{text[:4000]}",
                max_tokens=2000,
                temperature=0.0,
            )
            repaired = _parse_json(repair_text)
            if not repaired.get("_error"):
                return repaired
        except Exception:
            pass

    return parsed


# ── Event helper ────────────────────────────────────────────────────

def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _ev(event_type: str, **data: Any) -> dict[str, Any]:
    return {"type": event_type, "ts": _now(), **data}


# ── Main pipeline (streaming) ──────────────────────────────────────

async def run_analysis_stream(
    user_message: str,
    intake_context: dict[str, Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """
    Stream analysis events as dicts.

    Events:
      {type: "phase", phase: "compressing|researching|evaluating|revising|synthesizing|done"}
      {type: "agent", agent: "cost|arch|...", status: "working|done|error"}
      {type: "assistant_delta", delta: "text chunk"}
      {type: "final", brief: {...}}
      {type: "error", detail: "..."}
    """
    run_id = uuid.uuid4().hex

    # ── 1. Compress input ───────────────────────────────────────────
    yield _ev("phase", phase="compressing", run_id=run_id)
    compressed = user_message
    try:
        comp_text, _ = await llm_text(
            agent_key="compress",
            system_prompt=COMPRESS_PROMPT,
            user_message=user_message,
            max_tokens=300,
            temperature=0.3,
        )
        if comp_text and len(comp_text.split()) < len(user_message.split()):
            compressed = comp_text.strip()
    except Exception:
        pass  # Fall back to original message

    # Add intake constraints if present
    agent_input = compressed
    if intake_context:
        constraints = []
        for key in ("budget", "timeline", "riskAppetite"):
            v = str(intake_context.get(key) or "").strip()
            if v:
                constraints.append(f"- {key}: {v}")
        compliance = intake_context.get("compliance") or []
        if compliance:
            constraints.append(f"- compliance: {', '.join(str(c) for c in compliance)}")
        if constraints:
            agent_input += "\n\nCONSTRAINTS:\n" + "\n".join(constraints)

    # ── 2. Run specialists in parallel (with multi-source research) ──
    yield _ev("phase", phase="researching", run_id=run_id)
    can_search = search_available()
    research_stats_by_agent: dict[str, dict[str, int]] = {}
    research_sources_by_agent: dict[str, list[dict[str, str]]] = {}

    async def _run_specialist(key: str, prompt: str) -> dict[str, Any]:
        """Run one specialist with research context (pricing pages + web search)."""
        enriched_input = agent_input

        if can_search:
            results, stats = await research(user_message, agent_key=key)
            research_stats_by_agent[key] = stats
            # Track source URLs for the brief
            research_sources_by_agent[key] = [
                {
                    "type": r.get("type", "web_search"),
                    "title": r.get("title") or r.get("service") or "",
                    "url": r.get("url", ""),
                    "provider": r.get("provider", ""),
                }
                for r in results if r.get("url")
            ]
            # Cost agent gets a larger context budget (more sources to cite)
            char_budget = 6000 if key == "cost" else 3000
            context = format_search_context(results, max_chars=char_budget)
            if context:
                enriched_input = (
                    agent_input
                    + "\n\nRESEARCH DATA (use these sources to ground your analysis — cite URLs for any pricing or factual claims):\n"
                    + context
                )

        return await _run_agent(key, prompt, enriched_input)

    if can_search:
        yield _ev("search_status", run_id=run_id, enabled=True, provider="tavily")
    else:
        yield _ev("search_status", run_id=run_id, enabled=False, provider=None)

    tasks = {
        key: asyncio.create_task(_run_specialist(key, prompt))
        for key, prompt in AGENT_PROMPTS.items()
    }

    # Emit status as each finishes
    specialist_results: dict[str, dict[str, Any]] = {}
    for coro in asyncio.as_completed(list(tasks.values())):
        result = await coro
        # Find which agent this was
        for key, task in tasks.items():
            if task.done() and key not in specialist_results:
                specialist_results[key] = result
                status = "error" if result.get("_error") else "done"
                stats = research_stats_by_agent.get(key, {})
                total_sources = stats.get("pricing_pages", 0) + stats.get("web_results", 0)
                yield _ev("agent", agent=key, status=status, run_id=run_id,
                          sources=total_sources,
                          pricing_pages=stats.get("pricing_pages", 0),
                          web_results=stats.get("web_results", 0),
                          queries_run=stats.get("queries_run", 0))
                break

    # Build preliminary results keyed by perspective (financial, architecture, operations, strategy)
    prelim: dict[str, Any] = {}
    for key in AGENT_PROMPTS:
        brief_key = AGENT_BRIEF_KEYS[key]
        res = specialist_results.get(key, {})
        prelim[brief_key] = res if not res.get("_error") else None

    successful = sum(1 for v in prelim.values() if v is not None)
    if successful == 0:
        yield _ev("error", detail="All specialist agents failed. Please retry.", run_id=run_id)
        return

    # ── 3. Evaluator (devil's advocate) ─────────────────────────────
    yield _ev("phase", phase="evaluating", run_id=run_id)
    yield _ev("agent", agent="evaluator", status="working", run_id=run_id)

    eval_input = (
        "Review this preliminary analysis and find weaknesses:\n\n"
        + json.dumps(_slim(prelim), indent=2)
    )
    eval_result = await _run_agent("evaluator", EVALUATOR_PROMPT, eval_input, max_tokens=2500)

    if eval_result.get("_error"):
        eval_result = {"assessment": "Evaluator unavailable", "revision_needed": {}}
    yield _ev("agent", agent="evaluator", status="done", run_id=run_id)

    # ── 4. Revision pass (if evaluator flagged issues) ──────────────
    revision_needed = eval_result.get("revision_needed") or {}
    revisions = {k: v for k, v in revision_needed.items() if v and v != "null"}

    if revisions:
        yield _ev("phase", phase="revising", run_id=run_id)
        for agent_key, feedback in revisions.items():
            if agent_key not in AGENT_PROMPTS:
                continue
            yield _ev("agent", agent=agent_key, status="working", run_id=run_id)
            revision_input = (
                agent_input
                + f"\n\nCRITICAL FEEDBACK — address this in your revised analysis:\n{feedback}"
            )
            revised = await _run_agent(agent_key, AGENT_PROMPTS[agent_key], revision_input)
            status = "error" if revised.get("_error") else "done"
            yield _ev("agent", agent=agent_key, status=status, run_id=run_id)
            if not revised.get("_error"):
                prelim[AGENT_BRIEF_KEYS[agent_key]] = revised

    # ── 5. Synthesis ────────────────────────────────────────────────
    yield _ev("phase", phase="synthesizing", run_id=run_id)
    yield _ev("agent", agent="synthesis", status="working", run_id=run_id)

    synthesis_input = (
        "Synthesize these specialist outputs into a final executive decision brief:\n\n"
        + json.dumps(_slim({
            **prelim,
            "devils_advocate": _slim(eval_result),
        }), indent=2)
    )

    synthesis_result = await _run_agent(
        "synthesis", SYNTHESIS_PROMPT, synthesis_input,
        max_tokens=3500, temperature=0.2,
    )

    if synthesis_result.get("_error"):
        # Fallback: return raw specialist outputs
        yield _ev("agent", agent="synthesis", status="error", run_id=run_id)
        synthesis_result = {
            "decision_statement": "Synthesis failed — specialist outputs provided as-is.",
            "cost_analysis": prelim.get("cost"),
            "architecture_review": prelim.get("architecture"),
            "operations_assessment": prelim.get("operations"),
            "strategic_assessment": prelim.get("strategy"),
            "devils_advocate": eval_result,
            "recommendation": {
                "decision": "RETRY",
                "rationale": "Synthesis model failed. Review specialist outputs directly or retry.",
            },
            "confidence": 2,
        }
    else:
        yield _ev("agent", agent="synthesis", status="done", run_id=run_id)

    # Attach metadata
    synthesis_result["_run_id"] = run_id
    synthesis_result["_timestamp"] = _now()
    synthesis_result["_specialists_available"] = successful
    synthesis_result["devils_advocate"] = eval_result

    # Fallback: if synthesis dropped pricing_tiers, recover from financial specialist
    syn_tiers = synthesis_result.get("pricing_tiers")
    has_real_tiers = (
        isinstance(syn_tiers, list)
        and len(syn_tiers) > 0
        and all(
            isinstance(t, dict) and t.get("monthly_cost") and t.get("monthly_cost") != "—"
            for t in syn_tiers
        )
    )
    if not has_real_tiers:
        cost_output = prelim.get("financial")
        if isinstance(cost_output, dict):
            fallback_tiers = cost_output.get("pricing_tiers")
            if isinstance(fallback_tiers, list) and len(fallback_tiers) > 0:
                synthesis_result["pricing_tiers"] = fallback_tiers

    # Attach research sources so the frontend can show provenance
    all_sources: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for agent_key in ["cost", "arch", "ops", "strategy"]:
        for src in research_sources_by_agent.get(agent_key, []):
            url = src.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_sources.append({**src, "agent": agent_key})
    synthesis_result["research_sources"] = all_sources

    # ── 6. Implementation planning ──────────────────────────────────
    yield _ev("phase", phase="planning", run_id=run_id)
    yield _ev("agent", agent="planner", status="working", run_id=run_id)

    # Build context for the planner: user question + synthesized recommendation
    rec = synthesis_result.get("recommendation") or {}
    rec_decision = str(rec.get("decision") or "").upper()
    plan_input = json.dumps(_slim({
        "user_question": user_message,
        "decision_statement": synthesis_result.get("decision_statement"),
        "recommended_decision": rec_decision,
        "recommendation": rec,
        "options": synthesis_result.get("options"),
        "risk_register": synthesis_result.get("risk_register"),
        "analysis_highlights": synthesis_result.get("analysis_highlights"),
        "confidence": synthesis_result.get("confidence"),
        "what_flips_this": synthesis_result.get("what_flips_this"),
    }), indent=2)

    impl_result = await _run_agent(
        "planner", IMPLEMENTATION_PROMPT, plan_input,
        max_tokens=3500, temperature=0.2,
    )

    if impl_result.get("_error"):
        yield _ev("agent", agent="planner", status="error", run_id=run_id)
        synthesis_result["implementation_plan"] = None
    else:
        yield _ev("agent", agent="planner", status="done", run_id=run_id)
        synthesis_result["implementation_plan"] = impl_result

    # Stream the executive summary as text deltas for chat UX
    summary = str(synthesis_result.get("executive_summary") or "").strip()
    if summary:
        words = summary.split()
        for i in range(0, len(words), 6):
            chunk = " ".join(words[i:i + 6])
            yield _ev("assistant_delta", delta=chunk + " ", run_id=run_id)

    yield _ev("phase", phase="done", run_id=run_id)
    yield _ev("final", brief=synthesis_result, run_id=run_id)


# ── Non-streaming wrapper ──────────────────────────────────────────

async def run_analysis(
    user_message: str,
    intake_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run full pipeline synchronously, return final brief."""
    async for ev in run_analysis_stream(user_message, intake_context):
        if ev.get("type") == "final":
            return ev["brief"]
    raise RuntimeError("Analysis produced no output")