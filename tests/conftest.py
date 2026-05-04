"""
Test fixtures.

Key fixtures:
- `client`: async test client with in-memory SQLite DB
- `mock_adk`: patches the ADK root agent so tests don't hit the real LLM
- `seeded_db`: DB with a test user and session pre-created
"""
import sys
from pathlib import Path

# Add parent directory to path so 'app' module can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from unittest.mock import AsyncMock, patch
import uuid

from app.db.models import Base, User, Session as DBSession, AgentTrace, Message
from app.db.session import get_db
from app.main import app
from app.srop.state import SessionState
from app.srop.pipeline import PipelineResult


TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = async_sessionmaker(test_engine, expire_on_commit=False)


@pytest_asyncio.fixture(autouse=True)
async def setup_test_db():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db() -> AsyncSession:
    async with TestSessionLocal() as session:
        yield session


@pytest_asyncio.fixture
async def client(db):
    """Async test client with DB overridden to in-memory SQLite."""
    async def override_get_db():
        yield db
    
    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def seeded_db(db: AsyncSession):
    """
    Pre-create a test user and session in the DB.
    """
    test_user = User(user_id="u_test_user", plan_tier="pro")
    db.add(test_user)

    session_state = SessionState(
        user_id="u_test_user",
        plan_tier="pro",
        last_agent=None,
        turn_count=0,
    )

    test_session = DBSession(
        session_id="sess_test_001",
        user_id="u_test_user",
        state=session_state.to_db_dict(),
    )
    db.add(test_session)
    await db.commit()
    return db


@pytest.fixture
def mock_adk(monkeypatch):
    """
    Patch the ADK pipeline so tests don't call the real LLM.

    Returns a dict with mock_run (the patched function) and set_response
    (a helper to configure canned responses).

    Usage:
        mock_adk.set_response("knowledge", "Here's how to do it...", ["chunk_1"])
        # or set per message:
        mock_adk.set_response_for_query(
            query_text="rotate",
            routed_to="knowledge",
            response="..."
        )
    """

    class MockADK:
        def __init__(self):
            self.default_response = "I can help with that"
            self.default_routed_to = "smalltalk"
            self.default_chunks = []
            self.query_map: dict = {}  # query text -> (response, routed_to, chunks)

        def set_response(self, routed_to: str, response: str, chunks: list[str] = None):
            """Set default response for all queries."""
            self.default_routed_to = routed_to
            self.default_response = response
            self.default_chunks = chunks or []

        def set_response_for_query(self, query_text: str, routed_to: str, response: str, chunks: list[str] = None):
            """Set response for a specific query (substring match)."""
            self.query_map[query_text.lower()] = (response, routed_to, chunks or [])

        async def mock_run(self, session_id: str, user_message: str, db):
            """Mock pipeline.run that returns canned responses and saves traces."""
            # Check if query matches any configured queries
            routed_to = self.default_routed_to
            response = self.default_response
            chunks = self.default_chunks
            
            for query_text, (resp, route, chks) in self.query_map.items():
                if query_text in user_message.lower():
                    response = resp
                    routed_to = route
                    chunks = chks
                    break

            # Create and save trace to database
            trace_id = str(uuid.uuid4())
            trace = AgentTrace(
                trace_id=trace_id,
                session_id=session_id,
                routed_to=routed_to,
                tool_calls=[],
                retrieved_chunk_ids=chunks,
                latency_ms=100,
            )
            db.add(trace)
            
            # Save messages
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
                content=response,
                trace_id=trace_id,
            )
            db.add(user_msg)
            db.add(assistant_msg)
            await db.commit()
            
            return PipelineResult(
                content=response,
                routed_to=routed_to,
                trace_id=trace_id,
            )

    mock_adk_instance = MockADK()
    monkeypatch.setattr("app.srop.pipeline.run", mock_adk_instance.mock_run)
    return mock_adk_instance

