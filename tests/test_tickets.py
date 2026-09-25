from src.models import TicketCreate
from src.tools.ticket_tool import TicketRepository


def test_repository_prevents_duplicate_ticket_per_session() -> None:
    repository = TicketRepository()
    request = TicketCreate(
        customer_name="Test User",
        customer_email="test@example.com",
        issue_description="Payment was charged twice",
        category="payment",
        summary="Possible duplicate payment charge",
    )

    first = repository.create("session-1", request)
    second = repository.create("session-1", request)

    assert first.ticket_id == second.ticket_id
    assert len(list(repository.all())) == 1


def test_unique_ids_across_sessions() -> None:
    repository = TicketRepository()
    request = TicketCreate(
        customer_name="Test User",
        customer_email="test@example.com",
        issue_description="Payment was charged twice",
        category="payment",
        summary="Possible duplicate payment charge",
    )

    t1 = repository.create("session-1", request)
    t2 = repository.create("session-2", request)

    assert t1.ticket_id != t2.ticket_id
    assert len(list(repository.all())) == 2


def test_ticket_preserves_fields() -> None:
    repository = TicketRepository()
    request = TicketCreate(
        customer_name="Jane Doe",
        customer_email="jane@example.com",
        issue_description="My package was damaged",
        category="order",
        summary="Damaged package received",
    )
    ticket = repository.create("session-1", request)
    assert ticket.customer_name == "Jane Doe"
    assert ticket.customer_email == "jane@example.com"
    assert ticket.issue_description == "My package was damaged"
    assert ticket.category == "order"
    assert ticket.summary == "Damaged package received"


def test_get_existing_ticket() -> None:
    repository = TicketRepository()
    request = TicketCreate(
        customer_name="Test User",
        customer_email="test@example.com",
        issue_description="Order issue",
        category="order",
        summary="Order issue summary",
    )
    created = repository.create("session-1", request)
    fetched = repository.get(created.ticket_id)
    assert fetched is not None
    assert fetched.ticket_id == created.ticket_id


def test_get_nonexistent_ticket() -> None:
    repository = TicketRepository()
    assert repository.get("CST-2026-9999") is None


def test_ticket_id_format() -> None:
    repository = TicketRepository()
    request = TicketCreate(
        customer_name="Test",
        customer_email="test@example.com",
        issue_description="Issue description here",
        category="technical",
        summary="Technical issue summary",
    )
    ticket = repository.create("session-1", request)
    assert ticket.ticket_id.startswith("CST-2026-")
