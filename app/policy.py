from __future__ import annotations

import json
import os
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.llm import LlmRoute, route_for_agent


PolicyMode = Literal["shadow", "enforced_partial", "enforced_full"]
RiskTier = Literal["low", "medium", "high"]
PathName = Literal["fast_path", "standard_path", "high_assurance_path"]


def _env(name: str, default: str = "") -> str:
    """Read and trim environment configuration values with fallback."""
    return os.environ.get(name, default).strip()


def _is_placeholder(value: str) -> bool:
    """Detect placeholder/non-real API key values."""
    v = (value or "").strip()
    if not v:
        return True
    upper = v.upper()
    return (
        upper.startswith("YOUR_")
        or "YOUR_" in upper
        or upper in {"CHANGE_ME", "CHANGEME", "DUMMY", "PLACEHOLDER"}
    )


def _has(name: str) -> bool:
    """Check if env var is set with a real (non-placeholder) value."""
    return not _is_placeholder(_env(name))


def _csv_env(name: str, default: str = "") -> list[str]:
    """Parse comma-separated env values into normalized, deduped lowercase list."""
    raw = _env(name, default)
    out: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        v = part.strip().lower()
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _default_retrieval_preferred_domains() -> list[str]:
    """Return curated default preferred domains for implementation-grade research."""
    defaults = ",".join(
        [
            "learn.microsoft.com",
            "azure.microsoft.com",
            "aws.amazon.com",
            "docs.aws.amazon.com",
            "cloud.google.com",
            "developers.google.com",
            "kubernetes.io",
            "cncf.io",
            "opensearch.org",
            "elastic.co",
            "postgresql.org",
            "redis.io",
            "grafana.com",
            "prometheus.io",
            "www.cncf.io",
            "finops.org",
            "opensource.googleblog.com",
            "netflixtechblog.com",
            "blog.cloudflare.com",
            "stripe.com",
            "developer.hashicorp.com",
            "martinfowler.com",
            "trustradius.com",
            "g2.com",
        ]
    )
    return _csv_env("RETRIEVAL_PREFERRED_DOMAINS", defaults)


def _default_retrieval_allowed_domains() -> list[str]:
    """Return optional allow-list domains; empty means open web with preferred ranking."""
    defaults = ",".join(
        [
            "learn.microsoft.com",
            "azure.microsoft.com",
            "aws.amazon.com",
            "docs.aws.amazon.com",
            "cloud.google.com",
            "developers.google.com",
            "opensearch.org",
            "elastic.co",
            "postgresql.org",
            "redis.io",
            "mongodb.com",
            "kubernetes.io",
            "cncf.io",
            "opensource.googleblog.com",
            "netflixtechblog.com",
            "blog.cloudflare.com",
            "stripe.com",
            "developer.hashicorp.com",
            "grafana.com",
            "prometheus.io",
            "datadoghq.com",
            "newrelic.com",
            "confluent.io",
            "snowflake.com",
            "vercel.com",
            "cloudflare.com",
            "martinfowler.com",
            "thoughtworks.com",
            "finops.org",
            "trustradius.com",
            "g2.com",
        ]
    )
    return _csv_env("RETRIEVAL_ALLOWED_DOMAINS", defaults)


def _evidence_policy(*, min_citations: int, block_without_evidence: bool, require_freshness: bool) -> EvidencePolicy:
    """Build evidence policy with curated domain defaults and env overrides."""
    return EvidencePolicy(
        min_citations_for_recommend=min_citations,
        high_assurance_block_recommend_without_evidence=block_without_evidence,
        retrieval_allowed_domains=_default_retrieval_allowed_domains(),
        retrieval_preferred_domains=_default_retrieval_preferred_domains(),
        retrieval_require_freshness=require_freshness,
    )


class ModelRoutePolicy(BaseModel):
    preferred: LlmRoute
    fallbacks: list[LlmRoute] = Field(default_factory=list)


class TokenBudget(BaseModel):
    specialist_input_max: int = 2200
    synthesis_max: int = 1800


class ToolBudget(BaseModel):
    max_search_calls: int = 1
    enable_web_search: bool = True


class TimeoutBudget(BaseModel):
    analysis_max_seconds: int = 180
    llm_timeout_seconds: int = 25


class RequiredStages(BaseModel):
    planner: bool = True
    compress: bool = True
    cost: bool = True
    arch: bool = True
    ops: bool = True
    strategy: bool = True
    evaluator: bool = False
    verifier: bool = False
    synthesis: bool = True


class ConfidenceFloors(BaseModel):
    evaluator: int = 0
    verifier: int = 0


class EvidencePolicy(BaseModel):
    min_citations_for_recommend: int = 0
    high_assurance_block_recommend_without_evidence: bool = False
    retrieval_allowed_domains: list[str] = Field(default_factory=list)
    retrieval_preferred_domains: list[str] = Field(default_factory=list)
    retrieval_require_freshness: bool = False


class ReviewPolicy(BaseModel):
    require_blocking_on_confidence_drop: bool = False
    require_review_on_primary_source_failure: bool = False
    require_review_on_unresolved_contradiction: bool = False
    require_review_after_evaluator: bool = False
    require_review_on_low_evidence: bool = True
    require_review_before_final: bool = True
    confidence_drop_threshold: int = 5


class AgentCapability(BaseModel):
    tools_allowed: list[str] = Field(default_factory=list)
    domains_allowed: list[str] = Field(default_factory=list)
    domains_preferred: list[str] = Field(default_factory=list)
    providers_allowed: list[str] = Field(default_factory=list)
    max_tokens: int | None = None
    llm_timeout_seconds: int | None = None
    target_latency_ms: int | None = None
    max_cost_usd: float | None = None


class WorkflowPolicy(BaseModel):
    mode: PolicyMode = "shadow"
    risk_tier: RiskTier = "medium"
    path: PathName = "standard_path"
    model_routes: dict[str, ModelRoutePolicy] = Field(default_factory=dict)
    token_budget: TokenBudget = Field(default_factory=TokenBudget)
    tool_budget: ToolBudget = Field(default_factory=ToolBudget)
    timeout_budget: TimeoutBudget = Field(default_factory=TimeoutBudget)
    required_stages: RequiredStages = Field(default_factory=RequiredStages)
    confidence_floors: ConfidenceFloors = Field(default_factory=ConfidenceFloors)
    evidence_policy: EvidencePolicy = Field(default_factory=EvidencePolicy)
    review_policy: ReviewPolicy = Field(default_factory=ReviewPolicy)
    agent_capabilities: dict[str, AgentCapability] = Field(default_factory=dict)
    max_refinement_cycles: int = 1
    allow_auto_resume_on_synthesis_failure: bool = True
    selection_reason: str = ""
    enforced_constraints: list[str] = Field(default_factory=list)


class TeamTopology(BaseModel):
    name: str = "default"
    lead_agent: str = "planner"
    specialist_agents: list[str] = Field(default_factory=lambda: ["cost", "arch", "ops", "strategy"])
    critic_agent: str | None = "evaluator"
    verifier_agent: str | None = "verifier"
    synthesis_agent: str = "synthesis"


_TOPOLOGY_SPECIALISTS = {"cost", "arch", "ops", "strategy"}


def _capability_registry(
    *,
    token_budget: TokenBudget,
    timeout_budget: TimeoutBudget,
    evidence_policy: EvidencePolicy,
) -> dict[str, AgentCapability]:
    """Build per-agent capability registry used for dispatch and route constraints."""
    allowed_domains = evidence_policy.retrieval_allowed_domains or []
    preferred_domains = evidence_policy.retrieval_preferred_domains or []
    search_tools = ["web_search", "pricing_api_search"]
    no_tools: list[str] = []
    return {
        "planner": AgentCapability(
            tools_allowed=no_tools,
            providers_allowed=["openai", "gemini", "qwen", "anthropic"],
            max_tokens=min(800, token_budget.specialist_input_max),
            llm_timeout_seconds=timeout_budget.llm_timeout_seconds,
            target_latency_ms=7000,
            max_cost_usd=0.02,
        ),
        "compress": AgentCapability(
            tools_allowed=no_tools,
            providers_allowed=["openai", "gemini", "qwen", "anthropic"],
            max_tokens=min(800, token_budget.specialist_input_max),
            llm_timeout_seconds=timeout_budget.llm_timeout_seconds,
            target_latency_ms=7000,
            max_cost_usd=0.02,
        ),
        "distill": AgentCapability(
            tools_allowed=no_tools,
            providers_allowed=["openai", "gemini", "qwen", "anthropic"],
            max_tokens=1200,
            llm_timeout_seconds=timeout_budget.llm_timeout_seconds,
            target_latency_ms=9000,
            max_cost_usd=0.03,
        ),
        "cost": AgentCapability(
            tools_allowed=search_tools,
            domains_allowed=allowed_domains,
            domains_preferred=preferred_domains,
            providers_allowed=["qwen", "gemini", "anthropic", "openai"],
            max_tokens=token_budget.specialist_input_max,
            llm_timeout_seconds=timeout_budget.llm_timeout_seconds,
            target_latency_ms=20000,
            max_cost_usd=0.25,
        ),
        "arch": AgentCapability(
            tools_allowed=["web_search"],
            domains_allowed=allowed_domains,
            domains_preferred=preferred_domains,
            providers_allowed=["openai", "gemini", "qwen", "anthropic"],
            max_tokens=token_budget.specialist_input_max,
            llm_timeout_seconds=timeout_budget.llm_timeout_seconds,
            target_latency_ms=20000,
            max_cost_usd=0.25,
        ),
        "ops": AgentCapability(
            tools_allowed=["web_search"],
            domains_allowed=allowed_domains,
            domains_preferred=preferred_domains,
            providers_allowed=["qwen", "gemini", "anthropic", "openai"],
            max_tokens=token_budget.specialist_input_max,
            llm_timeout_seconds=timeout_budget.llm_timeout_seconds,
            target_latency_ms=20000,
            max_cost_usd=0.25,
        ),
        "strategy": AgentCapability(
            tools_allowed=["web_search"],
            domains_allowed=allowed_domains,
            domains_preferred=preferred_domains,
            providers_allowed=["qwen", "gemini", "anthropic", "openai"],
            max_tokens=token_budget.specialist_input_max,
            llm_timeout_seconds=timeout_budget.llm_timeout_seconds,
            target_latency_ms=20000,
            max_cost_usd=0.25,
        ),
        "evaluator": AgentCapability(
            tools_allowed=["web_search"],
            domains_allowed=allowed_domains,
            domains_preferred=preferred_domains,
            providers_allowed=["anthropic", "gemini", "qwen", "openai"],
            max_tokens=token_budget.specialist_input_max,
            llm_timeout_seconds=timeout_budget.llm_timeout_seconds,
            target_latency_ms=22000,
            max_cost_usd=0.25,
        ),
        "verifier": AgentCapability(
            tools_allowed=no_tools,
            providers_allowed=["openai", "gemini", "qwen", "anthropic"],
            max_tokens=min(1800, token_budget.synthesis_max),
            llm_timeout_seconds=timeout_budget.llm_timeout_seconds,
            target_latency_ms=12000,
            max_cost_usd=0.08,
        ),
        "synthesis": AgentCapability(
            tools_allowed=["web_search"],
            domains_allowed=allowed_domains,
            domains_preferred=preferred_domains,
            providers_allowed=["anthropic", "gemini", "qwen", "openai"],
            max_tokens=token_budget.synthesis_max,
            llm_timeout_seconds=timeout_budget.llm_timeout_seconds,
            target_latency_ms=26000,
            max_cost_usd=0.40,
        ),
    }


def refresh_agent_capabilities(policy: WorkflowPolicy) -> None:
    """Rebuild capability registry from current policy budgets/evidence settings."""
    policy.agent_capabilities = _capability_registry(
        token_budget=policy.token_budget,
        timeout_budget=policy.timeout_budget,
        evidence_policy=policy.evidence_policy,
    )


def _parse_json_env(name: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """Parse JSON env var into dict with safe fallback."""
    raw = _env(name, "").strip()
    if not raw:
        return dict(default or {})
    try:
        parsed = json.loads(raw)
    except Exception:
        return dict(default or {})
    return parsed if isinstance(parsed, dict) else dict(default or {})


def _default_team_topology(policy: WorkflowPolicy) -> TeamTopology:
    """Build default topology from required stage policy."""
    specialists: list[str] = []
    if policy.required_stages.cost:
        specialists.append("cost")
    if policy.required_stages.arch:
        specialists.append("arch")
    if policy.required_stages.ops:
        specialists.append("ops")
    if policy.required_stages.strategy:
        specialists.append("strategy")
    return TeamTopology(
        name=f"{policy.path}_default",
        lead_agent="planner",
        specialist_agents=specialists,
        critic_agent="evaluator" if policy.required_stages.evaluator else None,
        verifier_agent="verifier" if policy.required_stages.verifier else None,
        synthesis_agent="synthesis",
    )


def _apply_topology_variant(base: TeamTopology, variant: dict[str, Any]) -> TeamTopology:
    """Apply topology variant object onto base topology with validation."""
    out = base.model_copy(deep=True)
    if not isinstance(variant, dict):
        return out
    name = variant.get("name")
    if isinstance(name, str) and name.strip():
        out.name = name.strip()[:80]
    lead = variant.get("lead_agent")
    if isinstance(lead, str) and lead == "planner":
        out.lead_agent = lead
    specs = variant.get("specialist_agents")
    if isinstance(specs, list):
        cleaned = [str(s).strip() for s in specs if str(s).strip() in _TOPOLOGY_SPECIALISTS]
        out.specialist_agents = cleaned
    critic = variant.get("critic_agent")
    if critic is None or (isinstance(critic, str) and critic == "evaluator"):
        out.critic_agent = critic
    verifier = variant.get("verifier_agent")
    if verifier is None or (isinstance(verifier, str) and verifier == "verifier"):
        out.verifier_agent = verifier
    synthesis = variant.get("synthesis_agent")
    if isinstance(synthesis, str) and synthesis == "synthesis":
        out.synthesis_agent = synthesis
    return out


def resolve_team_topology(
    policy: WorkflowPolicy,
    *,
    tenant_id: str | None = None,
    intake_context: dict[str, Any] | None = None,
) -> TeamTopology:
    """Resolve effective team graph from policy + optional tenant/use-case overrides."""
    base = _default_team_topology(policy)
    variants = _parse_json_env("TEAM_TOPOLOGY_VARIANTS_JSON", {})
    tenant_map = _parse_json_env("TEAM_TOPOLOGY_TENANT_MAP_JSON", {})
    use_case_map = _parse_json_env("TEAM_TOPOLOGY_USE_CASE_MAP_JSON", {})

    ctx = intake_context or {}
    explicit_variant = str(ctx.get("team_topology") or "").strip()
    use_case = str(ctx.get("use_case") or "").strip()
    variant_key = ""
    if explicit_variant and explicit_variant in variants:
        variant_key = explicit_variant
    elif tenant_id and str(tenant_id).strip() in tenant_map:
        candidate = str(tenant_map.get(str(tenant_id).strip()) or "").strip()
        if candidate in variants:
            variant_key = candidate
    elif use_case and use_case in use_case_map:
        candidate = str(use_case_map.get(use_case) or "").strip()
        if candidate in variants:
            variant_key = candidate

    topo = _apply_topology_variant(base, variants.get(variant_key) if variant_key else {})

    # Hard safety alignment with required stage policy.
    if not policy.required_stages.evaluator:
        topo.critic_agent = None
    if not policy.required_stages.verifier:
        topo.verifier_agent = None
    allowed_specialists = {"cost", "arch", "ops", "strategy"}
    filtered_specs: list[str] = []
    for s in topo.specialist_agents:
        if s not in allowed_specialists:
            continue
        if s == "cost" and not policy.required_stages.cost:
            continue
        if s == "arch" and not policy.required_stages.arch:
            continue
        if s == "ops" and not policy.required_stages.ops:
            continue
        if s == "strategy" and not policy.required_stages.strategy:
            continue
        filtered_specs.append(s)
    topo.specialist_agents = filtered_specs
    if not topo.specialist_agents:
        topo.specialist_agents = [s for s in ("cost", "arch", "ops", "strategy") if getattr(policy.required_stages, s if s != "arch" else "arch", False)]
    return topo


def _route_chain(agent_key: str) -> ModelRoutePolicy:
    """Build preferred and fallback model routes for a policy stage/agent."""
    preferred = route_for_agent(agent_key)
    fallbacks: list[LlmRoute] = []
    lane = "pro" if agent_key == "synthesis" else "fast"
    # Keep architecture fallback tight and stable to reduce timeout/cooldown churn.
    if agent_key == "arch":
        provider_order = ["openai", "gemini"]
    else:
        provider_order = ["gemini", "qwen", "anthropic", "openai"]

    for provider in provider_order:
        if provider == preferred.provider.lower():
            continue
        if provider == "gemini" and _has("GEMINI_API_KEY"):
            model = _env("GEMINI_MODEL_PRO", "gemini/gemini-2.5-pro") if lane == "pro" else _env(
                "GEMINI_MODEL_FAST", "gemini/gemini-2.5-flash"
            )
            fallbacks.append(LlmRoute(provider="gemini", model=model))
        elif provider == "qwen" and _has("QWEN_API_KEY"):
            fallbacks.append(LlmRoute(provider="qwen", model=_env("QWEN_MODEL", "qwen-plus")))
        elif provider == "anthropic" and _has("ANTHROPIC_API_KEY"):
            model = _env("ANTHROPIC_MODEL_PRO", "anthropic/claude-sonnet-4-5-20250929") if lane == "pro" else _env(
                "ANTHROPIC_MODEL_FAST", "anthropic/claude-sonnet-4-5-20250929"
            )
            fallbacks.append(LlmRoute(provider="anthropic", model=model))
        elif provider == "openai" and _has("OPENAI_API_KEY"):
            model = _env("OPENAI_MODEL_PRO", "openai/gpt-4.1") if lane == "pro" else _env(
                "OPENAI_MODEL_FAST", "openai/gpt-4o-mini"
            )
            fallbacks.append(LlmRoute(provider="openai", model=model))
    return ModelRoutePolicy(preferred=preferred, fallbacks=fallbacks)


def _base_policy(mode: PolicyMode, risk_tier: RiskTier, path: PathName) -> WorkflowPolicy:
    """Create a baseline policy object before tier-specific overrides are applied."""
    model_routes = {
        "planner": _route_chain("planner"),
        "compress": _route_chain("compress"),
        "cost": _route_chain("cost"),
        "arch": _route_chain("arch"),
        "ops": _route_chain("ops"),
        "strategy": _route_chain("strategy"),
        "evaluator": _route_chain("evaluator"),
        "verifier": _route_chain("verifier"),
        "synthesis": _route_chain("synthesis"),
    }
    p = WorkflowPolicy(mode=mode, risk_tier=risk_tier, path=path, model_routes=model_routes)
    refresh_agent_capabilities(p)
    return p


def resolve_policy(
    user_message: str,
    intake_context: dict | None,
    *,
    force_tier: str | None = None,
    mode_override: str | None = None,
) -> WorkflowPolicy:
    """Resolve policy profile from input text + structured intake context."""
    mode = (mode_override or _env("POLICY_MODE", "enforced_partial")).lower()
    if mode not in ("shadow", "enforced_partial", "enforced_full"):
        mode = "enforced_partial"
    mode_t = mode  # type: ignore[assignment]

    msg = (user_message or "").lower()
    ctx = intake_context or {}
    compliance = ctx.get("compliance") or []
    risk = str(ctx.get("riskAppetite") or "").lower()
    high_keywords = ["hipaa", "pci", "soc2", "gdpr", "compliance", "security", "mission critical", "payments"]
    forced_tier = (force_tier or _env("POLICY_FORCE_TIER", "")).lower()

    is_high = forced_tier == "high" or bool(compliance) or risk == "aggressive" or any(k in msg for k in high_keywords)
    is_low = forced_tier == "low" or (not is_high and len(msg.split()) < 40 and "migration" not in msg and "replatform" not in msg)

    if is_high:
        p = _base_policy(mode_t, "high", "high_assurance_path")
        p.required_stages = RequiredStages(
            planner=True,
            compress=True,
            cost=True,
            arch=True,
            ops=True,
            strategy=True,
            evaluator=True,
            verifier=True,
            synthesis=True,
        )
        p.token_budget = TokenBudget(specialist_input_max=2800, synthesis_max=2200)
        p.tool_budget = ToolBudget(max_search_calls=4, enable_web_search=True)
        p.timeout_budget = TimeoutBudget(analysis_max_seconds=240, llm_timeout_seconds=50)
        p.confidence_floors = ConfidenceFloors(evaluator=6, verifier=6)
        p.evidence_policy = _evidence_policy(
            min_citations=3,
            block_without_evidence=True,
            require_freshness=True,
        )
        p.review_policy = ReviewPolicy(
            require_blocking_on_confidence_drop=True,
            require_review_on_primary_source_failure=True,
            require_review_on_unresolved_contradiction=True,
            require_review_after_evaluator=True,
            require_review_on_low_evidence=True,
            require_review_before_final=True,
            confidence_drop_threshold=6,
        )
        p.max_refinement_cycles = 2
        p.allow_auto_resume_on_synthesis_failure = True
        p.selection_reason = (
            "Policy tier forced to HIGH by POLICY_FORCE_TIER."
            if forced_tier == "high"
            else "Compliance/risk-sensitive request detected from intake or query terms."
        )
        p.enforced_constraints = [
            "evaluator_required",
            "verifier_required",
            "citation_threshold_enforced",
            "search_enabled_with_budget",
            "refinement_cycles_capped",
            "wall_clock_deadline_enforced",
        ]
        refresh_agent_capabilities(p)
        return p

    if is_low:
        p = _base_policy(mode_t, "low", "fast_path")
        p.required_stages = RequiredStages(
            planner=True,
            compress=True,
            cost=True,
            arch=True,
            ops=True,
            strategy=False,
            evaluator=False,
            verifier=False,
            synthesis=True,
        )
        p.token_budget = TokenBudget(specialist_input_max=1600, synthesis_max=1200)
        p.tool_budget = ToolBudget(max_search_calls=0, enable_web_search=False)
        p.timeout_budget = TimeoutBudget(analysis_max_seconds=120, llm_timeout_seconds=30)
        p.confidence_floors = ConfidenceFloors(evaluator=0, verifier=0)
        p.evidence_policy = _evidence_policy(
            min_citations=0,
            block_without_evidence=False,
            require_freshness=False,
        )
        p.review_policy = ReviewPolicy(
            require_blocking_on_confidence_drop=False,
            require_review_on_primary_source_failure=False,
            require_review_on_unresolved_contradiction=False,
            require_review_after_evaluator=False,
            require_review_on_low_evidence=False,
            require_review_before_final=False,
            confidence_drop_threshold=0,
        )
        p.max_refinement_cycles = 0
        p.allow_auto_resume_on_synthesis_failure = True
        p.selection_reason = "Low-risk short request without migration/compliance indicators."
        p.enforced_constraints = [
            "strategy_stage_disabled",
            "evaluator_skipped",
            "verifier_skipped",
            "web_search_disabled",
            "tight_token_and_time_budget",
        ]
        refresh_agent_capabilities(p)
        return p

    p = _base_policy(mode_t, "medium", "standard_path")
    p.required_stages = RequiredStages(
        planner=True,
        compress=True,
        cost=True,
        arch=True,
        ops=True,
        strategy=True,
        evaluator=True,
        verifier=True,
        synthesis=True,
    )
    p.token_budget = TokenBudget(specialist_input_max=2200, synthesis_max=1800)
    p.tool_budget = ToolBudget(max_search_calls=2, enable_web_search=True)
    p.timeout_budget = TimeoutBudget(analysis_max_seconds=180, llm_timeout_seconds=40)
    p.confidence_floors = ConfidenceFloors(evaluator=0, verifier=0)
    p.evidence_policy = _evidence_policy(
        min_citations=1,
        block_without_evidence=False,
        require_freshness=False,
    )
    p.review_policy = ReviewPolicy(
        require_blocking_on_confidence_drop=False,
        require_review_on_primary_source_failure=False,
        require_review_on_unresolved_contradiction=True,
        require_review_after_evaluator=False,
        require_review_on_low_evidence=True,
        require_review_before_final=True,
        confidence_drop_threshold=4,
    )
    p.max_refinement_cycles = 1
    p.allow_auto_resume_on_synthesis_failure = True
    p.selection_reason = (
        "Policy tier forced to MEDIUM by POLICY_FORCE_TIER."
        if forced_tier == "medium"
        else "Default balanced path for mixed-risk architecture decisions."
    )
    p.enforced_constraints = [
        "verifier_enabled",
        "limited_search_budget",
        "single_refinement_cycle",
        "wall_clock_deadline_enforced",
    ]
    refresh_agent_capabilities(p)
    return p
