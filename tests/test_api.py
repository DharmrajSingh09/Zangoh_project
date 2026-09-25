from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.models import ChatResponse
from src.utils.errors import AgentProcessingError, ComponentNotReadyError


@pytest.fixture
def client():
    from src.api.server import app, pipeline
    async def mock_init():
        pipeline.ready = True
        
        mock_workflow = AsyncMock()
        mock_workflow.ainvoke = AsyncMock(return_value={
            "response_text": "Standard delivery takes 3-5 business days.",
            "sources": ["shipping.md"],
            "ticket_id": None,
        })
        pipeline.workflow = mock_workflow

        if hasattr(pipeline.retriever, 'search'):
            pipeline.retriever.search = AsyncMock(return_value=[])
            pipeline.retriever._store = MagicMock()

    with patch.object(pipeline, 'initialize', side_effect=mock_init):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c, pipeline
    
    # Clean up after test
    pipeline.sessions._sessions.clear()
    pipeline.tickets._tickets.clear()
    pipeline.tickets._session_ticket.clear()


def test_health_ready(client):
    c, pipeline = client
    assert c.get("/health").status_code == 200
    data = c.get("/health").json()
    assert data["status"] == "ready"


def test_health_not_ready(client):
    c, pipeline = client
    pipeline.ready = False
    resp = c.get("/health")
    assert resp.status_code == 503
    pipeline.ready = True


def test_chat_grounded_answer(client):
    c, pipeline = client
    resp = c.post("/chat", json={"session_id": "s1", "message": "How long does shipping take?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "shipping.md" in data["sources"]
    assert data["response"] != ""


def test_chat_missing_message(client):
    c, pipeline = client
    resp = c.post("/chat", json={"session_id": "s1", "message": ""})
    assert resp.status_code == 422


def test_chat_missing_session_id(client):
    c, pipeline = client
    resp = c.post("/chat", json={"session_id": "", "message": "hello"})
    assert resp.status_code == 422


def test_ticket_not_found(client):
    c, pipeline = client
    resp = c.get("/tickets/CST-2026-9999")
    assert resp.status_code == 404


def test_chat_unanswerable_no_fabrication(client):
    c, pipeline = client
    pipeline.workflow.ainvoke = AsyncMock(return_value={
        "response_text": "I don't have information about that topic in my knowledge base. I'd be happy to open a support ticket if you need further assistance.",
        "sources": [],
        "ticket_id": None,
    })
    resp = c.post("/chat", json={"session_id": "s2", "message": "What is the weather today?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["sources"] == []


def test_multi_turn_ticket_creation(client):
    c, pipeline = client

    call_count = 0

    async def mock_invoke(state):
        nonlocal call_count
        call_count += 1
        session_id = state.get("session_id", "")

        if call_count == 1:
            conv = pipeline.sessions.get_or_create(session_id)
            conv.issue_description = "My order arrived damaged"
            conv.category = "order"
            return {
                "response_text": "I'd be happy to help you create a support ticket. Could you please provide your full name?",
                "sources": [],
                "ticket_id": None,
            }
        elif call_count == 2:
            conv = pipeline.sessions.get_or_create(session_id)
            conv.customer_name = "John Doe"
            return {
                "response_text": "Thank you, John! Could you please provide your email address?",
                "sources": [],
                "ticket_id": None,
            }
        else:
            conv = pipeline.sessions.get_or_create(session_id)
            conv.customer_email = "john@example.com"

            from src.models import TicketCreate
            req = TicketCreate(
                customer_name=conv.customer_name,
                customer_email=conv.customer_email,
                issue_description=conv.issue_description,
                category=conv.category,
                summary=conv.issue_description[:100],
            )
            ticket = pipeline.tickets.create(session_id, req)
            conv.ticket_id = ticket.ticket_id

            return {
                "response_text": f"Your ticket {ticket.ticket_id} has been created successfully!",
                "sources": [],
                "ticket_id": ticket.ticket_id,
            }

    pipeline.workflow.ainvoke = mock_invoke

    sid = "multi-turn-session"
    resp1 = c.post("/chat", json={"session_id": sid, "message": "My order arrived damaged"})
    assert resp1.status_code == 200
    assert resp1.json()["ticket_id"] is None

    resp2 = c.post("/chat", json={"session_id": sid, "message": "John Doe"})
    assert resp2.status_code == 200
    assert resp2.json()["ticket_id"] is None

    resp3 = c.post("/chat", json={"session_id": sid, "message": "john@example.com"})
    assert resp3.status_code == 200
    data3 = resp3.json()
    assert data3["ticket_id"] is not None
    ticket_id = data3["ticket_id"]
    assert ticket_id.startswith("CST-2026-")

    ticket_resp = c.get(f"/tickets/{ticket_id}")
    assert ticket_resp.status_code == 200
    ticket_data = ticket_resp.json()
    assert ticket_data["ticket_id"] == ticket_id
    assert ticket_data["customer_name"] == "John Doe"
    assert ticket_data["customer_email"] == "john@example.com"


def test_retry_same_session_returns_same_ticket(client):
    c, pipeline = client

    from src.models import TicketCreate
    req = TicketCreate(
        customer_name="Retry User",
        customer_email="retry@example.com",
        issue_description="Duplicate charge on my account",
        category="payment",
        summary="Duplicate charge issue",
    )
    ticket = pipeline.tickets.create("retry-session", req)
    tid = ticket.ticket_id

    ticket2 = pipeline.tickets.create("retry-session", req)
    assert ticket2.ticket_id == tid
    assert len(list(pipeline.tickets.all())) >= 1

    t1 = pipeline.tickets.get(tid)
    assert t1 is not None
    assert t1.ticket_id == tid


def test_backend_failure_returns_502(client):
    c, pipeline = client

    async def failing_invoke(state):
        raise RuntimeError("LLM is down")

    pipeline.workflow.ainvoke = failing_invoke

    resp = c.post("/chat", json={"session_id": "fail-session", "message": "help"})
    assert resp.status_code == 502


def test_no_filesystem_paths_in_response(client):
    c, pipeline = client

    pipeline.workflow.ainvoke = AsyncMock(return_value={
        "response_text": "Here is the info.",
        "sources": ["shipping.md"],
        "ticket_id": None,
    })

    resp = c.post("/chat", json={"session_id": "path-check", "message": "shipping"})
    data = resp.json()
    for source in data["sources"]:
        assert "/" not in source or source.endswith(".md")
        assert "\\" not in source
        assert ":" not in source
