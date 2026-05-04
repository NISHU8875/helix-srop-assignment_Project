"""
POST /v1/chat/{session_id} — send a user message, get assistant reply.
"""
import structlog
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.srop import pipeline
from app.api.errors import SessionNotFoundError, UpstreamTimeoutError

router = APIRouter(tags=["chat"])
log = structlog.get_logger()


class ChatRequest(BaseModel):
    content: str


class ChatResponse(BaseModel):
    reply: str
    routed_to: str   # which sub-agent handled this turn
    trace_id: str


@router.post("/chat/{session_id}", response_model=ChatResponse)
async def chat(
    session_id: str,
    body: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    """
    Run one turn of the SROP pipeline.

    Error cases:
    - Session not found → 404 (SessionNotFoundError)
    - LLM timeout → 504 (UpstreamTimeoutError)
    """
    structlog.contextvars.bind_contextvars(session_id=session_id)
    log.info("chat_request", message_len=len(body.content))

    try:
        result = await pipeline.run(session_id, body.content, db)
        log.info("chat_complete", routed_to=result.routed_to)
        return ChatResponse(
            reply=result.content,
            routed_to=result.routed_to,
            trace_id=result.trace_id,
        )
    except (SessionNotFoundError, UpstreamTimeoutError):
        # Re-raise framework errors (they have proper status codes)
        raise
    except Exception as e:
        log.error("chat_error", error=str(e), exc_info=True)
        raise

