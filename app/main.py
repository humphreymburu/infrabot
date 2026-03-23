"""
Simplified FastAPI backend for the tech decision advisor.
Three endpoints: health, analyze (JSON), chat (streaming text).
"""
import json
import uuid
import logging
import os

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.analysis import run_analysis, run_analysis_stream

logger = logging.getLogger(__name__)

app = FastAPI(title="Tech Decision Advisor API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Run-Id"],
)


# ── Models ──────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    user_message: str = Field(..., description="The decision question")
    budget: str | None = None
    timeline: str | None = None
    risk_appetite: str | None = None
    compliance: list[str] | None = None


class AnalyzeResponse(BaseModel):
    brief: dict


# ── Endpoints ───────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """Run analysis and return final brief as JSON."""
    if not req.user_message.strip():
        raise HTTPException(400, "user_message is required")

    intake = {
        "budget": req.budget or "",
        "timeline": req.timeline or "",
        "riskAppetite": req.risk_appetite or "",
        "compliance": req.compliance or [],
    }
    try:
        brief = await run_analysis(req.user_message, intake)
        return AnalyzeResponse(brief=brief)
    except RuntimeError as e:
        raise HTTPException(502, str(e))
    except Exception as e:
        logger.exception("analyze_failed: %s", e)
        raise HTTPException(500, f"Analysis failed: {e}")


def _extract_user_text(message: dict) -> str:
    """Extract plain text from AI SDK UIMessage (supports both v4 content and v5 parts)."""
    # v5+: parts array
    parts = message.get("parts")
    if isinstance(parts, list):
        chunks = []
        for p in parts:
            if isinstance(p, dict) and p.get("type") == "text":
                t = str(p.get("text") or "").strip()
                if t:
                    chunks.append(t)
        if chunks:
            return "\n".join(chunks)
    # v4 fallback: content string
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    return ""


@app.post("/api/chat")
async def chat(request: Request):
    """
    Streaming chat endpoint compatible with AI SDK useChat + TextStreamChatTransport.
    Receives AI SDK message format, streams plain text back.
    """
    payload = await request.json()
    messages = payload.get("messages", [])
    if not messages:
        raise HTTPException(400, "messages required")

    # Extract latest user message (AI SDK sends full history)
    user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and str(msg.get("role")) == "user":
            user_message = _extract_user_text(msg)
            if user_message:
                break
    if not user_message.strip():
        raise HTTPException(400, "No user message found")

    run_id = uuid.uuid4().hex
    intake = {
        "budget": str(payload.get("budget") or "").strip(),
        "timeline": str(payload.get("timeline") or "").strip(),
        "riskAppetite": str(payload.get("risk_appetite") or "").strip(),
        "compliance": payload.get("compliance") or [],
    }

    async def stream():
        async for ev in run_analysis_stream(user_message, intake):
            if ev.get("type") == "assistant_delta":
                yield str(ev.get("delta") or "")

    return StreamingResponse(
        stream(),
        media_type="text/plain; charset=utf-8",
        headers={"X-Run-Id": run_id},
    )


@app.post("/api/analyze/stream")
async def analyze_stream(req: AnalyzeRequest):
    """Stream full analysis events as NDJSON."""
    if not req.user_message.strip():
        raise HTTPException(400, "user_message is required")

    intake = {
        "budget": req.budget or "",
        "timeline": req.timeline or "",
        "riskAppetite": req.risk_appetite or "",
        "compliance": req.compliance or [],
    }

    async def stream():
        async for ev in run_analysis_stream(req.user_message, intake):
            yield json.dumps(ev, ensure_ascii=False) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")