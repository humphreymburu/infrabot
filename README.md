# Tech Advisor Backend

FastAPI backend for the Atlas AI tech-decision advisor. Runs the multi-agent pipeline on the server and can route agents across **Gemini** and **Qwen**.

## Run

```bash
uv sync
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Environment

Set keys in `.env` (or environment):

- `GEMINI_API_KEY`: required if using Gemini routes
- `QWEN_API_KEY`: required if using Qwen routes
- `QWEN_BASE_URL`: recommended (gateway endpoint). For DashScope Intl use `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`
- `QWEN_MODEL`: default Qwen model name (default: `qwen-plus`)
- `TAVILY_API_KEY`: optional, enables server-side web search snippets
- `ENABLE_WEB_SEARCH`: set to `0` to disable web search (default: enabled)
- `TAVILY_SEARCH_DEPTH`: `basic` (default) or `advanced`
- `EVALUATOR_ALWAYS_ON`: set to `1` to always run Devil's Advocate evaluator stage (even on low-risk fast-path requests)
- `COST_WEB_SEARCH_REQUIRED`: set to `1` (default) to reserve search budget for cost agent and force cost grounding search
- `COST_PRICING_API_FIRST`: set to `1` (default) to fetch cost evidence from official pricing endpoints first (Azure Retail Prices + AWS OpenSearch public offer files)
- `COST_REQUIRE_EXPLICIT_FOUNDATION`: set to `1` to block recommendation until explicit stack configuration is provided for cost basis (default `0` for context-driven mode)
- `COST_REQUIRE_EXPLICIT_FOUNDATION_AT_INTAKE`: set to `1` to fail API requests immediately (422) when cost foundation fields are missing (default `0`)
- `COST_AUTOFOUNDATION_ENABLED`: set to `1` (default) to infer missing foundation fields from prompt context/workload hints
- `COST_PRICING_API_ENRICHMENT`: set to `1` (default) to enrich unit rates from Azure Retail Prices API + AWS OpenSearch published pricing offer files
- `RETRIEVAL_ALLOWED_DOMAINS`: comma-separated domain allowlist for web evidence
- `RETRIEVAL_PREFERRED_DOMAINS`: comma-separated high-authority domains to prioritize in ranking
- `EVIDENCE_ENFORCE_SOURCE_MIX`: set to `1` (default) to enforce official-vendor + independent source mix before recommendation passes
- `EVIDENCE_MIN_OFFICIAL_VENDOR_SOURCES_LOW|MEDIUM|HIGH`: per-tier minimum distinct official vendor domains (defaults: `1|2|2`)
- `EVIDENCE_MIN_INDEPENDENT_SOURCES_LOW|MEDIUM|HIGH`: per-tier minimum distinct independent domains (defaults: `0|1|1`)

### DashScope note

You mentioned `dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'`.
That `/api/v1` is DashScope's native API base; LiteLLM works best with DashScope's **OpenAI-compatible** base:

- `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`

If you set `QWEN_BASE_URL` to the `/api/v1` value anyway, the backend will automatically rewrite it to `/compatible-mode/v1`.

### Evidence Source-Mix Gate

When `EVIDENCE_ENFORCE_SOURCE_MIX=1`, governance blocks recommendation if source composition is weak.

- Official vendor sources: provider docs/pricing/product pages (AWS/Azure/GCP/docs + major vendor docs).
- Independent sources: analyst/benchmark/community sources (for example TrustRadius, G2, Gartner, Forrester, Thoughtworks, Martin Fowler, FinOps, CNCF).

Recommended production defaults:

- `EVIDENCE_MIN_OFFICIAL_VENDOR_SOURCES_LOW=1`
- `EVIDENCE_MIN_OFFICIAL_VENDOR_SOURCES_MEDIUM=2`
- `EVIDENCE_MIN_OFFICIAL_VENDOR_SOURCES_HIGH=2`
- `EVIDENCE_MIN_INDEPENDENT_SOURCES_LOW=0`
- `EVIDENCE_MIN_INDEPENDENT_SOURCES_MEDIUM=1`
- `EVIDENCE_MIN_INDEPENDENT_SOURCES_HIGH=1`

### Per-agent routing (optional)

Override which provider/model each agent uses:

- `ROUTE_COST_PROVIDER`, `ROUTE_COST_MODEL`
- `ROUTE_ARCH_PROVIDER`, `ROUTE_ARCH_MODEL`
- `ROUTE_OPS_PROVIDER`, `ROUTE_OPS_MODEL`
- `ROUTE_STRATEGY_PROVIDER`, `ROUTE_STRATEGY_MODEL`
- `ROUTE_EVALUATOR_PROVIDER`, `ROUTE_EVALUATOR_MODEL`
- `ROUTE_SYNTHESIS_PROVIDER`, `ROUTE_SYNTHESIS_MODEL`

### Explicit Cost Foundation (Decision-Grade TCO)

For non-speculative cost comparison, send explicit config for both current and proposed stacks in `/api/analyze`:

- `current_stack_config`: `instance_type`, `node_count`, `storage_gb`, `region`
- `proposed_stack_config`: `tier` (or `sku_or_tier`), `search_units` (or `replicas`/`partitions`), `storage_gb`, `region`
- `workload_assumptions`: shared workload assumptions used for both sides (for example `doc_count`, `qps`, `growth_3y_multiplier`)

If `COST_REQUIRE_EXPLICIT_FOUNDATION=1`, recommendation gate blocks with `missing_explicit_cost_foundation` until this is present.

## Task Queue Control Plane (New)

The backend now supports durable async analysis tasks in addition to direct `/api/analyze`.

### Queue APIs

- `POST /api/tasks`: enqueue a task
- `POST /api/analyze/async`: alias for enqueueing analysis tasks
- `GET /api/tasks`: list tasks (tenant-scoped unless admin)
- `GET /api/tasks/{task_id}`: fetch one task
- `POST /api/tasks/{task_id}/cancel`: cancel queued/running task
- `POST /api/tasks/{task_id}/retry`: requeue a task
- `GET /api/review-templates`: list reusable human gate templates (`financial_risk_gate`, `compliance_gate`, `source_trust_gate`)
- `GET /api/telemetry/fleet`: fleet ops dashboard metrics (queue depth, stuck tasks, retries, fallback rate, approval latency, cost/decision)
- `POST /api/policy/simulate/replay`: projected/full replay simulation with diff risk scorecard before policy rollout
- `GET /api/agents/contracts`: inspect internal Agent SDK contracts (schema, quality checks, tool budgets)

### Worker settings

- `TASK_WORKER_ENABLED`: `1` (default) to run background queue worker
- `TASK_WORKER_ID`: worker id label (default: `worker-1`)
- `TASK_WORKER_POLL_MS`: queue poll interval in ms (default: `1200`)
- `TASK_WORKER_OWNER_AGENT`: optional owner-agent filter for this worker
- `TASK_RETRY_BACKOFF_SECONDS`: retry delay when a task attempt fails (default: `20`)
- `TASK_LEASE_SECONDS`: worker lease duration per claimed task (default: `90`)
- `TASK_HEARTBEAT_INTERVAL_SECONDS`: lease heartbeat interval while task is running (default: `15`)
- `TASK_DIR`: task storage directory (default: `/tmp/tech_advisor_tasks`)

### Task model

Task records include first-class queue metadata:

- `goal`
- `owner_agent`
- `priority` (`low|normal|high|urgent`)
- `sla_due_at`
- `state` (`queued|claimed|running|retrying|done|failed|cancelled`)

## Team Topology (New)

Specialist orchestration is now topology-driven instead of hardcoded specialist wiring.
You can configure per-tenant/per-use-case specialist team composition without code changes.

- `TEAM_TOPOLOGY_VARIANTS_JSON`: JSON object of named topology variants
- `TEAM_TOPOLOGY_TENANT_MAP_JSON`: JSON object mapping tenant id -> topology variant name
- `TEAM_TOPOLOGY_USE_CASE_MAP_JSON`: JSON object mapping intake `use_case` -> topology variant name

Variant shape:

```json
{
  "finops_light": {
    "name": "finops_light",
    "lead_agent": "planner",
    "specialist_agents": ["cost", "strategy"],
    "critic_agent": "evaluator",
    "verifier_agent": "verifier",
    "synthesis_agent": "synthesis"
  }
}
```
