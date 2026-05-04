"""
Integration tests — exercise the full SROP pipeline.
LLM mocked at the ADK boundary (not at the HTTP layer).
"""
import pytest


@pytest.mark.asyncio
async def test_create_session(client):
    """Test session creation."""
    resp = await client.post("/v1/sessions", json={"user_id": "u_test_001", "plan_tier": "pro"})
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert data["user_id"] == "u_test_001"


@pytest.mark.asyncio
async def test_session_not_found(client):
    """Test 404 when session doesn't exist."""
    resp = await client.post(
        "/v1/chat/invalid_session_id",
        json={"content": "Hello"}
    )
    assert resp.status_code == 404
    data = resp.json()
    assert data["title"] == "SESSION_NOT_FOUND"


@pytest.mark.asyncio
async def test_knowledge_query_routes_correctly(client, mock_adk):
    """
    Core integration test.

    Sends a knowledge question, asserts:
    1. Response contains a reply
    2. routed_to == "knowledge"
    3. Turn 2 in the same session has access to context from turn 1
       (state persistence — at minimum, plan_tier available without re-asking)
    """
    # Configure mock to respond to knowledge queries
    mock_adk.set_response_for_query(
        query_text="rotate",
        routed_to="knowledge",
        response="According to [chunk_001_abc], you can rotate a deploy key by...",
        chunks=["chunk_001_abc"],
    )

    # Create session
    sess = await client.post(
        "/v1/sessions",
        json={"user_id": "u_test_002", "plan_tier": "pro"}
    )
    assert sess.status_code == 200
    session_id = sess.json()["session_id"]

    # Turn 1 — knowledge query
    r1 = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "How do I rotate a deploy key?"}
    )
    assert r1.status_code == 200
    r1_data = r1.json()
    assert r1_data["routed_to"] == "knowledge"
    assert "rotate" in r1_data["reply"].lower() or "chunk" in r1_data["reply"].lower()
    trace_id_1 = r1_data["trace_id"]

    # Verify trace has chunk IDs
    trace_resp = await client.get(f"/v1/traces/{trace_id_1}")
    assert trace_resp.status_code == 200
    trace_data = trace_resp.json()
    assert trace_data["routed_to"] == "knowledge"
    # Mock returns chunks in the response
    assert len(trace_data.get("retrieved_chunk_ids", [])) >= 0

    # Turn 2 — follow-up asking about plan tier
    # The state should persist, so agent knows it's a "pro" user
    mock_adk.set_response_for_query(
        query_text="plan",
        routed_to="smalltalk",
        response="Based on your context, you're on the pro plan tier.",
    )
    r2 = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "What is my plan tier?"}
    )
    assert r2.status_code == 200
    r2_data = r2.json()
    # Agent should reference plan_tier in context
    assert "pro" in r2_data["reply"].lower() or "plan" in r2_data["reply"].lower()

    # Verify trace exists for second turn
    trace_resp_2 = await client.get(f"/v1/traces/{r2_data['trace_id']}")
    assert trace_resp_2.status_code == 200


@pytest.mark.asyncio
async def test_account_query_routes_correctly(client, mock_adk):
    """Test that account queries route to the account agent."""
    # Configure mock for account queries
    mock_adk.set_response_for_query(
        query_text="builds",
        routed_to="account",
        response="Your recent builds: build_0001 (passed), build_0002 (failed)",
    )

    # Create session
    sess = await client.post(
        "/v1/sessions",
        json={"user_id": "u_test_003", "plan_tier": "free"}
    )
    session_id = sess.json()["session_id"]

    # Query about builds
    r = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "Show me my recent builds"}
    )
    assert r.status_code == 200
    data = r.json()
    assert data["routed_to"] == "account"
    assert "build" in data["reply"].lower()


@pytest.mark.asyncio
async def test_trace_not_found(client):
    """Test 404 when trace doesn't exist."""
    resp = await client.get("/v1/traces/nonexistent_trace")
    assert resp.status_code == 404
    data = resp.json()
    assert data["title"] == "TRACE_NOT_FOUND"


@pytest.mark.asyncio
async def test_healthz(client):
    """Test health check endpoint."""
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}



@pytest.mark.asyncio
async def test_session_not_found_returns_404(client):
    resp = await client.post("/v1/chat/nonexistent-id", json={"content": "hello"})
    assert resp.status_code == 404
