"""
Multi-source research layer.

Sources (in priority order):
1. Direct pricing page fetch — hit known vendor pricing URLs, extract structured data
2. Tavily web search — broad web search for context, comparisons, real-world reports
3. LLM training data — fallback when no external sources available (handled by prompts)

Each source is optional and degrades gracefully.
"""
import os
import re
import asyncio
import logging
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ── Tavily client (lazy init) ──────────────────────────────────────

_tavily = None


def _get_tavily():
    global _tavily
    if _tavily is not None:
        return _tavily
    key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not key or key.upper().startswith("YOUR_"):
        return None
    try:
        from tavily import TavilyClient
        _tavily = TavilyClient(api_key=key)
        return _tavily
    except ImportError:
        logger.warning("tavily-python not installed. Run: pip install tavily-python")
        return None


def search_available() -> bool:
    return _get_tavily() is not None


# ── Known pricing page URLs by stack keyword ───────────────────────

PRICING_URLS: dict[str, list[dict[str, str]]] = {
    "opensearch": [
        {"url": "https://aws.amazon.com/opensearch-service/pricing/", "provider": "AWS", "service": "OpenSearch Service"},
    ],
    "elasticsearch": [
        {"url": "https://www.elastic.co/pricing/", "provider": "Elastic", "service": "Elasticsearch"},
        {"url": "https://aws.amazon.com/opensearch-service/pricing/", "provider": "AWS", "service": "OpenSearch Service"},
    ],
    "azure search": [
        {"url": "https://azure.microsoft.com/en-us/pricing/details/search/", "provider": "Azure", "service": "AI Search"},
    ],
    "azure cognitive search": [
        {"url": "https://azure.microsoft.com/en-us/pricing/details/search/", "provider": "Azure", "service": "AI Search"},
    ],
    "rds": [
        {"url": "https://aws.amazon.com/rds/pricing/", "provider": "AWS", "service": "RDS"},
    ],
    "aurora": [
        {"url": "https://aws.amazon.com/rds/aurora/pricing/", "provider": "AWS", "service": "Aurora"},
    ],
    "planetscale": [
        {"url": "https://planetscale.com/pricing", "provider": "PlanetScale", "service": "PlanetScale"},
    ],
    "neon": [
        {"url": "https://neon.tech/pricing", "provider": "Neon", "service": "Neon Postgres"},
    ],
    "redis": [
        {"url": "https://aws.amazon.com/elasticache/pricing/", "provider": "AWS", "service": "ElastiCache"},
        {"url": "https://redis.io/pricing/", "provider": "Redis", "service": "Redis Cloud"},
    ],
    "kafka": [
        {"url": "https://www.confluent.io/confluent-cloud/pricing/", "provider": "Confluent", "service": "Confluent Cloud"},
        {"url": "https://aws.amazon.com/msk/pricing/", "provider": "AWS", "service": "MSK"},
    ],
    "redpanda": [
        {"url": "https://redpanda.com/pricing", "provider": "Redpanda", "service": "Redpanda Cloud"},
    ],
    "datadog": [
        {"url": "https://www.datadoghq.com/pricing/", "provider": "Datadog", "service": "Datadog"},
    ],
    "auth0": [
        {"url": "https://auth0.com/pricing", "provider": "Auth0", "service": "Auth0"},
    ],
    "clerk": [
        {"url": "https://clerk.com/pricing", "provider": "Clerk", "service": "Clerk"},
    ],
    "supabase": [
        {"url": "https://supabase.com/pricing", "provider": "Supabase", "service": "Supabase"},
    ],
    "vercel": [
        {"url": "https://vercel.com/pricing", "provider": "Vercel", "service": "Vercel"},
    ],
    "ec2": [
        {"url": "https://aws.amazon.com/ec2/pricing/on-demand/", "provider": "AWS", "service": "EC2"},
    ],
    "s3": [
        {"url": "https://aws.amazon.com/s3/pricing/", "provider": "AWS", "service": "S3"},
    ],
    "eks": [
        {"url": "https://aws.amazon.com/eks/pricing/", "provider": "AWS", "service": "EKS"},
    ],
    "aks": [
        {"url": "https://azure.microsoft.com/en-us/pricing/details/kubernetes-service/", "provider": "Azure", "service": "AKS"},
    ],
    "gke": [
        {"url": "https://cloud.google.com/kubernetes-engine/pricing", "provider": "GCP", "service": "GKE"},
    ],
}


def _detect_stacks(text: str) -> list[str]:
    """Detect technology/vendor keywords in user message."""
    lower = text.lower()
    found = []
    for keyword in PRICING_URLS:
        if keyword in lower:
            found.append(keyword)
    return found


def _pricing_urls_for(text: str) -> list[dict[str, str]]:
    """Return relevant pricing URLs based on detected stacks."""
    stacks = _detect_stacks(text)
    seen: set[str] = set()
    urls: list[dict[str, str]] = []
    for stack in stacks:
        for entry in PRICING_URLS.get(stack, []):
            if entry["url"] not in seen:
                seen.add(entry["url"])
                urls.append(entry)
    return urls


# ── Direct pricing page fetch via Tavily extract ───────────────────

async def _fetch_pricing_pages(urls: list[dict[str, str]]) -> list[dict[str, Any]]:
    """
    Fetch pricing page content via Tavily extract API.
    Returns structured results with provider/service metadata.
    Falls back gracefully if Tavily extract is unavailable.
    """
    client = _get_tavily()
    if not client or not urls:
        return []

    url_list = [u["url"] for u in urls]
    url_meta = {u["url"]: u for u in urls}

    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.extract(urls=url_list),
        )
        results = response.get("results", [])
        out: list[dict[str, Any]] = []
        for r in results:
            if not isinstance(r, dict):
                continue
            url = str(r.get("url", "")).strip()
            raw = str(r.get("raw_content", "")).strip()
            meta = url_meta.get(url, {})
            if raw:
                out.append({
                    "type": "pricing_page",
                    "provider": meta.get("provider", ""),
                    "service": meta.get("service", ""),
                    "url": url,
                    "content": raw[:3000],
                })
        return out
    except Exception as e:
        logger.warning("pricing_page_fetch failed: %s", str(e)[:200])
        return []


# ── Tavily web search ──────────────────────────────────────────────

async def web_search(query: str, *, max_results: int = 5, agent: str = "") -> list[dict[str, Any]]:
    """Search the web via Tavily. Returns list of {title, url, content} dicts."""
    client = _get_tavily()
    if not client:
        return []

    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.search(
                query=query,
                max_results=max_results,
                include_answer=False,
                include_raw_content=False,
            ),
        )
        results = response.get("results", [])
        return [
            {
                "type": "web_search",
                "title": str(r.get("title", "")).strip(),
                "url": str(r.get("url", "")).strip(),
                "content": str(r.get("content", "")).strip()[:800],
            }
            for r in results
            if isinstance(r, dict)
        ]
    except Exception as e:
        logger.warning("web_search failed agent=%s query=%s error=%s", agent, query[:80], e)
        return []


# ── Combined research function ─────────────────────────────────────

def _build_cost_queries(user_message: str, stacks: list[str]) -> list[str]:
    """
    Build multiple targeted search queries for financial research.
    Mirrors how Claude searches: per-stack pricing, comparison queries,
    tier/SKU queries, cost optimization queries.
    """
    base = " ".join(user_message.strip().split()[:30])
    queries: list[str] = []

    # Per-stack pricing queries
    for stack in stacks[:4]:
        queries.append(f"{stack} pricing tiers monthly cost 2026")
        queries.append(f"{stack} pricing calculator instance types SKU")

    # Head-to-head comparison if 2+ stacks detected
    if len(stacks) >= 2:
        vs = " vs ".join(stacks[:3])
        queries.append(f"{vs} pricing comparison cost")
        queries.append(f"{vs} total cost of ownership TCO migration")

    # Cost optimization / real-world cost reports
    for stack in stacks[:2]:
        queries.append(f"{stack} cost optimization tips real world cost")

    # Fallback if no stacks detected
    if not queries:
        queries.append(f"{base} pricing cost comparison monthly estimate per tier")
        queries.append(f"{base} total cost of ownership migration cost")

    # Deduplicate
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        key = " ".join(q.lower().split())
        if key not in seen:
            seen.add(key)
            out.append(q)
    return out


def _build_agent_queries(agent_key: str, user_message: str) -> list[str]:
    """Build multiple search queries for non-financial agents."""
    base = " ".join(user_message.strip().split()[:35])
    queries_by_agent: dict[str, list[str]] = {
        "arch": [
            f"{base} architecture best practices",
            f"{base} scalability reliability failure modes",
        ],
        "ops": [
            f"{base} operations production deployment SRE",
            f"{base} observability monitoring incident response",
        ],
        "strategy": [
            f"{base} vendor comparison strategic assessment",
            f"{base} lock-in migration risk market",
        ],
    }
    return queries_by_agent.get(agent_key, [f"{base} best practices"])


async def _run_searches_parallel(
    queries: list[str],
    *,
    max_results_per: int = 8,
    agent: str = "",
) -> list[dict[str, Any]]:
    """Run multiple search queries in parallel, deduplicate results."""
    client = _get_tavily()
    if not client:
        return []

    async def _single(query: str) -> list[dict[str, Any]]:
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda q=query: client.search(
                    query=q,
                    max_results=max_results_per,
                    include_answer=False,
                    include_raw_content=False,
                ),
            )
            return [
                {
                    "type": "web_search",
                    "title": str(r.get("title", "")).strip(),
                    "url": str(r.get("url", "")).strip(),
                    "content": str(r.get("content", "")).strip()[:800],
                    "query": query,
                }
                for r in response.get("results", [])
                if isinstance(r, dict)
            ]
        except Exception as e:
            logger.warning("search failed agent=%s query=%s error=%s", agent, query[:60], e)
            return []

    # Run all queries in parallel
    tasks = [_single(q) for q in queries]
    all_results = await asyncio.gather(*tasks)

    # Flatten and deduplicate by URL
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for batch in all_results:
        for r in batch:
            url = r.get("url", "").lower().split("#")[0].rstrip("/")
            if url and url not in seen:
                seen.add(url)
                deduped.append(r)

    return deduped


async def research(
    user_message: str,
    agent_key: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Run deep multi-source research for an agent.

    Financial agent gets:
    - Direct pricing page extraction (vendor pages)
    - 6-10 parallel web searches across pricing, comparison, optimization queries
    - Deduplication across all results

    Other agents get:
    - 2 parallel web searches with domain focus

    Total Tavily calls per analysis run:
    - Cost: ~1 extract + 6-10 searches
    - Arch: 2 searches
    - Ops: 2 searches
    - Strategy: 2 searches
    - Total: ~13-17 Tavily API calls
    """
    results: list[dict[str, Any]] = []
    stats: dict[str, int] = {"pricing_pages": 0, "web_results": 0, "queries_run": 0}

    if agent_key == "cost":
        stacks = _detect_stacks(user_message)
        pricing_urls = _pricing_urls_for(user_message)

        # Layer 1: Direct pricing page extraction
        if pricing_urls:
            logger.info("cost_research: extracting %d pricing pages: %s",
                        len(pricing_urls), [u["url"] for u in pricing_urls])
            pages = await _fetch_pricing_pages(pricing_urls)
            results.extend(pages)
            stats["pricing_pages"] = len(pages)
            if not pages:
                logger.warning("cost_research: extract returned 0 — falling back to search only")

        # Layer 2: Multi-round parallel web search
        queries = _build_cost_queries(user_message, stacks)
        stats["queries_run"] = len(queries)
        logger.info("cost_research: running %d search queries: %s",
                    len(queries), [q[:60] for q in queries])
        web_results = await _run_searches_parallel(queries, max_results_per=8, agent=agent_key)
        results.extend(web_results)
        stats["web_results"] = len(web_results)

        logger.info("cost_research: done — %d pricing pages, %d web results from %d queries",
                    stats["pricing_pages"], stats["web_results"], stats["queries_run"])
        return results, stats

    # Other agents: 2 parallel searches
    queries = _build_agent_queries(agent_key, user_message)
    stats["queries_run"] = len(queries)
    web_results = await _run_searches_parallel(queries, max_results_per=5, agent=agent_key)
    results.extend(web_results)
    stats["web_results"] = len(web_results)

    return results, stats


# ── Format for prompt injection ────────────────────────────────────

def format_search_context(results: list[dict[str, Any]], *, max_chars: int = 5000) -> str:
    """
    Format research results into a text block for agent context injection.
    Prioritizes: pricing pages first, then web results sorted by relevance.
    """
    if not results:
        return ""

    # Group by type — pricing pages first (highest value), then web search
    pricing = [r for r in results if r.get("type") == "pricing_page"]
    web = [r for r in results if r.get("type") == "web_search"]

    lines: list[str] = []
    total = 0

    if pricing:
        lines.append("── OFFICIAL PRICING DATA (from vendor pricing pages) ──")
        total += 60
        for r in pricing:
            provider = r.get("provider", "")
            service = r.get("service", "")
            url = r.get("url", "")
            content = r.get("content", "")
            entry = f"\n[{provider} — {service}]({url})\n{content}\n"
            if total + len(entry) > max_chars:
                # Truncate content to fit
                remaining = max_chars - total - 100
                if remaining > 200:
                    entry = f"\n[{provider} — {service}]({url})\n{content[:remaining]}...\n"
                else:
                    break
            lines.append(entry)
            total += len(entry)

    if web:
        lines.append("\n── WEB SEARCH RESULTS ──")
        total += 30
        for r in web:
            title = r.get("title", "")
            url = r.get("url", "")
            content = r.get("content", "")
            entry = f"\n[{title}]({url})\n{content}\n"
            if total + len(entry) > max_chars:
                break
            lines.append(entry)
            total += len(entry)

    return "\n".join(lines).strip()


def build_search_query(agent_key: str, user_message: str) -> str:
    """Build a focused search query for an agent's domain."""
    base = " ".join(user_message.strip().split()[:40])
    focus = {
        "cost": "pricing cost comparison monthly estimate per tier",
        "arch": "architecture best practices scalability reliability",
        "ops": "operations SRE observability deployment production",
        "strategy": "strategic assessment vendor comparison market",
    }.get(agent_key, "")
    return f"{base} {focus}".strip()[:300]