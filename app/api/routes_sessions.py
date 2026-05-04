"""
POST /v1/sessions — create a session.
"""
import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.models import User, Session as DBSession
from app.db.session import get_db
from app.srop.state import SessionState

router = APIRouter(tags=["sessions"])


class CreateSessionRequest(BaseModel):
    user_id: str
    plan_tier: str = "free"


class CreateSessionResponse(BaseModel):
    session_id: str
    user_id: str


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(
    body: CreateSessionRequest,
    db: AsyncSession = Depends(get_db),
) -> CreateSessionResponse:
    """
    Create a new session. Upsert the user if not seen before.
    Initialize SessionState and persist to DB.
    """
    session_id = str(uuid.uuid4())

    # Upsert user
    stmt = select(User).where(User.user_id == body.user_id)
    user = await db.scalar(stmt)

    if not user:
        user = User(user_id=body.user_id, plan_tier=body.plan_tier)
        db.add(user)
    else:
        # Update plan_tier if provided
        user.plan_tier = body.plan_tier

    # Create initial session state
    state = SessionState(
        user_id=body.user_id,
        plan_tier=body.plan_tier,
        last_agent=None,
        turn_count=0,
    )

    # Create session record
    session = DBSession(
        session_id=session_id,
        user_id=body.user_id,
        state=state.to_db_dict(),
    )
    db.add(session)

    # Commit transaction
    await db.commit()

    return CreateSessionResponse(session_id=session_id, user_id=body.user_id)

