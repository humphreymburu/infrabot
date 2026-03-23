"""
Simplified LLM router. One provider at a time, automatic fallback, no cooldown state.
"""
import os
import asyncio
import random
import time
import logging
from dataclasses import dataclass
from typing import Any

from litellm import acompletion

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LlmRoute:
    provider: str
    model: str


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _has_key(name: str) -> bool:
    v = _env(name)
    return bool(v) and not v.upper().startswith("YOUR_")


# ── Provider detection ──────────────────────────────────────────────
PROVIDERS: list[dict[str, Any]] = [
    {"name": "gemini",    "key_env": "GEMINI_API_KEY",    "fast": "gemini/gemini-2.5-flash",         "pro": "gemini/gemini-2.5-pro"},
    {"name": "anthropic", "key_env": "ANTHROPIC_API_KEY", "fast": "anthropic/claude-3-5-haiku-latest","pro": "anthropic/claude-sonnet-4-5-latest"},
    {"name": "openai",    "key_env": "OPENAI_API_KEY",    "fast": "openai/gpt-4o-mini",              "pro": "openai/gpt-4.1"},
]


def _available_providers() -> list[dict[str, Any]]:
    return [p for p in PROVIDERS if _has_key(p["key_env"])]


def _pick_route(lane: str = "fast") -> LlmRoute:
    """Pick the first available provider. Lane is 'fast' or 'pro'."""
    for p in _available_providers():
        model = _env(f"{p['name'].upper()}_MODEL_{lane.upper()}", p.get(lane, p["fast"]))
        return LlmRoute(provider=p["name"], model=model)
    raise RuntimeError("No LLM API keys configured. Set GEMINI_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY.")


# ── Agent routing table ─────────────────────────────────────────────
# Specialists use fast models; synthesis uses pro.
AGENT_LANE = {
    "cost": "fast", "arch": "fast", "ops": "fast", "strategy": "fast",
    "evaluator": "fast", "synthesis": "pro", "compress": "fast",
    "planner": "fast", "verifier": "fast",
}


def route_for_agent(agent_key: str) -> LlmRoute:
    lane = AGENT_LANE.get(agent_key, "fast")
    return _pick_route(lane)


# ── Core LLM call with retry ────────────────────────────────────────
async def llm_text(
    *,
    agent_key: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int = 3500,
    temperature: float = 0.2,
    timeout_seconds: float = 30.0,
) -> tuple[str, LlmRoute]:
    """
    Call an LLM and return (text, route_used).
    Tries each available provider in order with 2 retries per provider.
    """
    providers = _available_providers()
    if not providers:
        raise RuntimeError("No LLM API keys configured.")

    lane = AGENT_LANE.get(agent_key, "fast")
    last_err: Exception | None = None

    for p in providers:
        model = _env(f"{p['name'].upper()}_MODEL_{lane.upper()}", p.get(lane, p["fast"]))
        route = LlmRoute(provider=p["name"], model=model)

        for attempt in range(3):
            try:
                start = time.perf_counter()
                resp = await acompletion(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout_seconds,
                )
                elapsed = (time.perf_counter() - start) * 1000
                content = resp.choices[0].message.get("content", "")
                text = content if isinstance(content, str) else str(content or "")
                logger.info("llm_call agent=%s provider=%s model=%s ms=%.0f", agent_key, p["name"], model, elapsed)
                return text, route

            except Exception as e:
                last_err = e
                err = str(e).lower()
                # Don't retry auth or model-not-found errors
                if any(s in err for s in ["invalid api key", "unauthorized", "not_found"]):
                    break
                if attempt < 2:
                    await asyncio.sleep(0.5 * (2 ** attempt) * (0.5 + random.random()))

    raise last_err or RuntimeError("All LLM providers failed")