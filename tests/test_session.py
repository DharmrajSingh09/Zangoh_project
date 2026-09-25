from src.sessions.store import ConversationState, SessionStore


def test_session_reports_only_missing_ticket_fields() -> None:
    state = ConversationState(customer_name="Asha", customer_email="asha@example.com")
    assert state.missing_ticket_fields() == ["issue_description", "category"]


def test_new_session_all_fields_missing() -> None:
    state = ConversationState()
    assert state.missing_ticket_fields() == [
        "customer_name",
        "customer_email",
        "issue_description",
        "category",
    ]


def test_complete_session_no_missing_fields() -> None:
    state = ConversationState(
        customer_name="Asha",
        customer_email="asha@example.com",
        issue_description="My order arrived damaged",
        category="order",
    )
    assert state.missing_ticket_fields() == []


def test_sessions_are_isolated() -> None:
    store = SessionStore()
    s1 = store.get_or_create("session-1")
    s2 = store.get_or_create("session-2")
    s1.customer_name = "Alice"
    assert s2.customer_name is None


def test_same_session_returns_same_state() -> None:
    store = SessionStore()
    s1 = store.get_or_create("session-1")
    s1.customer_name = "Alice"
    s2 = store.get_or_create("session-1")
    assert s2.customer_name == "Alice"


def test_history_preserves_order() -> None:
    state = ConversationState()
    state.history.append({"role": "user", "content": "Hello"})
    state.history.append({"role": "assistant", "content": "Hi there!"})
    state.history.append({"role": "user", "content": "I need help"})
    assert len(state.history) == 3
    assert state.history[0]["role"] == "user"
    assert state.history[1]["role"] == "assistant"
    assert state.history[2]["role"] == "user"


def test_completed_session_retains_ticket_id() -> None:
    state = ConversationState(
        customer_name="Test",
        customer_email="test@example.com",
        issue_description="Problem with order",
        category="order",
        ticket_id="CST-2026-0001",
    )
    assert state.ticket_id == "CST-2026-0001"
    assert state.missing_ticket_fields() == []
