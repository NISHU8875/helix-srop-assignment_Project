"""
SROP entrypoint — called by the message route.

This is the core of the assignment. It ties together:
  - Loading session state from DB
  - Running the ADK orchestrator with that state as context
  - Extracting routing decision and tool calls from ADK events
  - Recording the trace
  - Persisting updated session state to DB

The route calls: result = await pipeline.run(session_id, user_message, db)
It receives: PipelineResult(content, routed_to, trace_id)

Design questions answered:
  1. SessionState injection: Pattern 3 — store state in DB, inject as system context
  2. Routing extraction: event.author from final response event
  3. Tool captures: event.type == "tool_call" and "tool_result"
  4. Timeout: asyncio.wait_for wraps the runner
  5. Failure handling: Trace is recorded before returning; state saved after

State persistence: SessionState(user_id, plan_tier, last_agent, turn_count) lives in sessions.state JSON column.
"""
from __future__ import annotations

import asyncio
import uuid
import time
import structlog
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.orchestrator import create_orchestrator
from app.agents.tools.search_docs import search_docs
from app.db.models import Session as DBSession, Message, AgentTrace, User
from app.srop.state import SessionState
from app.api.errors import SessionNotFoundError, UpstreamTimeoutError
from app.settings import settings

from google.adk.runners import InMemoryRunner


log = structlog.get_logger()


@dataclass
class PipelineResult:
    content: str
    routed_to: str
    trace_id: str


async def _load_session_state(session_id: str, db: AsyncSession) -> tuple[SessionState, DBSession]:
    """Load SessionState from DB. Raise SessionNotFoundError if not found."""
    stmt = select(DBSession).where(DBSession.session_id == session_id)
    db_session = await db.scalar(stmt)

    if not db_session:
        raise SessionNotFoundError(f"Session {session_id} not found")

    # Deserialize state
    state_data = db_session.state or {}
    state = SessionState.from_db_dict(state_data)

    return state, db_session


async def _load_message_history(session_id: str, db: AsyncSession) -> list[Message]:
    """Load all prior messages for this session, ordered by creation time."""
    stmt = select(Message).where(Message.session_id == session_id).order_by(Message.created_at)
    messages = await db.scalars(stmt)
    return list(messages)


async def _extract_routing_and_tools(response_iter) -> tuple[str, list[dict], list[str], str]:
    """
    Parse ADK event stream to extract:
    - routed_to: which agent handled this turn
    - tool_calls: list of {tool_name, args, result}
    - retrieved_chunk_ids: chunk IDs from search_docs calls
    - final_text: the final response text

    The event stream is consumed and buffered here.
    """
    routed_to = "unknown"
    tool_calls: list[dict] = []
    retrieved_chunk_ids: list[str] = []
    current_tool_call: dict | None = None
    final_text = ""

    async for event in response_iter:
        # Tool call initiated
        if hasattr(event, "type") and event.type == "tool_call":
            current_tool_call = {
                "tool_name": getattr(event, "tool_name", "unknown"),
                "args": getattr(event, "tool_args", {}),
                "result": None,
            }

        # Tool call result
        if hasattr(event, "type") and event.type == "tool_result":
            if current_tool_call:
                current_tool_call["result"] = getattr(event, "tool_result", None)
                tool_calls.append(current_tool_call)
                current_tool_call = None

                # Extract chunk IDs if this was a search_docs call
                if tool_calls[-1]["tool_name"] == "search_docs":
                    result = tool_calls[-1]["result"]
                    if isinstance(result, list):
                        for r in result:
                            if isinstance(r, dict) and "chunk_id" in r:
                                retrieved_chunk_ids.append(r["chunk_id"])

        # Final response event indicates which agent responded
        if hasattr(event, "is_final_response") and event.is_final_response():
            routed_to = getattr(event, "author", "unknown")
            # Extract final text
            if hasattr(event, "content") and hasattr(event.content, "parts"):
                parts = event.content.parts
                if parts and hasattr(parts[0], "text"):
                    final_text = parts[0].text

    return routed_to, tool_calls, retrieved_chunk_ids, final_text


async def run(session_id: str, user_message: str, db: AsyncSession) -> PipelineResult:
    """
    Run one turn of the SROP pipeline.

    1. Load session state from DB
    2. Inject state into orchestrator via instruction context
    3. Build message history for ADK
    4. Run orchestrator with timeout
    5. Extract routing + tool calls from events
    6. Record trace to DB
    7. Save updated state + new messages to DB
    8. Return result

    Errors:
    - SessionNotFoundError if session doesn't exist
    - UpstreamTimeoutError if LLM times out
    """
    trace_id = str(uuid.uuid4())
    start_time = time.time()
    structlog.contextvars.bind_contextvars(session_id=session_id, trace_id=trace_id)

    try:
        # Step 1: Load state
        state, db_session = await _load_session_state(session_id, db)
        log.info("session_loaded", user_id=state.user_id, plan_tier=state.plan_tier)

        # Step 2: Build context injection
        context_str = f"""
Current User Context:
- user_id: {state.user_id}
- plan_tier: {state.plan_tier}
- turn_count: {state.turn_count}
- last_agent: {state.last_agent or "none"}

Use this context to provide personalized responses. If the user asks about their account, include their plan tier in the answer.
"""

        # Step 3: Load prior messages and inject into context
        prior_messages = await _load_message_history(session_id, db)
        
        # Build conversation history string for context injection
        history_lines = []
        for m in prior_messages:
            history_lines.append(f"{m.role}: {m.content}")
        history_str = "\n".join(history_lines)
        
        # Enhance context with conversation history
        full_context = context_str
        if history_str:
            full_context += f"\n\nPrior Conversation:\n{history_str}"

        # Step 4: Create orchestrator with enhanced context
        orchestrator = create_orchestrator(full_context)

        # Create runner (ADK will manage sessions internally)
        runner = InMemoryRunner(agent=orchestrator)

        # Create new message wrapper for ADK compatibility
        class _Message:
            def __init__(self, role: str, parts: list):
                self.role = role
                self.parts = parts

        new_message = _Message(role="user", parts=[{"text": user_message}])

        # Run orchestrator with timeout
        try:
            # Generate a session ID for ADK (it will auto-create the session)
            adk_session_id = str(uuid.uuid4())
            response_iter = runner.run_async(
                user_id=state.user_id,
                session_id=adk_session_id,
                new_message=new_message,
            )
            # Extract routing + tool calls + retrieved chunks with timeout
            routed_to, tool_calls, retrieved_chunk_ids, final_text = await asyncio.wait_for(
                _extract_routing_and_tools(response_iter),
                timeout=settings.llm_timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise UpstreamTimeoutError(
                f"LLM did not respond within {settings.llm_timeout_seconds}s"
            )

        # Step 6: Record trace to DB
        latency_ms = int((time.time() - start_time) * 1000)
        trace = AgentTrace(
            trace_id=trace_id,
            session_id=session_id,
            routed_to=routed_to,
            tool_calls=tool_calls,
            retrieved_chunk_ids=retrieved_chunk_ids,
            latency_ms=latency_ms,
        )
        db.add(trace)

        # Step 7: Save new messages + update state
        user_msg = Message(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role="user",
            content=user_message,
            trace_id=trace_id,
        )
        assistant_msg = Message(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role="assistant",
            content=final_text,
            trace_id=trace_id,
        )
        db.add(user_msg)
        db.add(assistant_msg)

        # Update state
        state.last_agent = routed_to
        state.turn_count += 1
        db_session.state = state.to_db_dict()

        # Commit everything
        await db.commit()

        log.info(
            "pipeline_complete",
            routed_to=routed_to,
            tool_calls=len(tool_calls),
            latency_ms=latency_ms,
        )

        return PipelineResult(content=final_text, routed_to=routed_to, trace_id=trace_id)

    except Exception as e:
        log.error("pipeline_error", error=str(e), exc_info=True)
        raise

