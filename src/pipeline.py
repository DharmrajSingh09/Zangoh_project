from __future__ import annotations

from pathlib import Path

from src.config import Settings
from src.llm.client import build_chat_model
from src.llm.workflow import build_support_workflow
from src.models import ChatResponse
from src.rag.retriever import KnowledgeRetriever
from src.sessions.store import SessionStore
from src.tools.ticket_tool import TicketRepository
from src.utils.errors import AgentProcessingError, ComponentNotReadyError


class SupportPipeline:
    """Top-level binding for model, RAG, workflow, sessions, and ticket tool."""

    def __init__(self, settings: Settings, documents_dir: Path) -> None:
        self.settings = settings
        self.model = build_chat_model(settings)
        self.retriever = KnowledgeRetriever(settings, documents_dir)
        self.sessions = SessionStore()
        self.tickets = TicketRepository()
        self.workflow = None
        self.ready = False

    async def initialize(self) -> None:
        """Initialize shared components once during FastAPI startup."""
        await self.retriever.initialize()
        self.workflow = build_support_workflow(self.model)
        self.ready = True

    async def process(self, session_id: str, message: str) -> ChatResponse:
        if not self.ready or self.workflow is None:
            raise ComponentNotReadyError("Support pipeline is not ready")

        session_id = session_id.strip()
        message = message.strip()

        if not session_id:
            raise AgentProcessingError("session_id must not be empty")
        if not message:
            raise AgentProcessingError("message must not be empty")

        conv = self.sessions.get_or_create(session_id)
        conv.history.append({"role": "user", "content": message})

        initial_state = {
            "session_id": session_id,
            "customer_message": message,
            "messages": [],
            "retrieved_chunks": [],
            "route": "",
            "extracted_fields": {},
            "response_text": "",
            "sources": [],
            "ticket_id": None,
            "retriever": self.retriever,
            "session_store": self.sessions,
            "ticket_repo": self.tickets,
        }

        try:
            result = await self.workflow.ainvoke(initial_state)
        except Exception as exc:
            raise AgentProcessingError(f"Workflow processing failed: {exc}") from exc

        response_text = result.get("response_text", "")
        if not response_text:
            response_text = "I apologize, but I was unable to process your request. Please try again."

        sources = result.get("sources", [])
        sources = [Path(s).name for s in sources]

        ticket_id = result.get("ticket_id")

        conv.history.append({"role": "assistant", "content": response_text})

        return ChatResponse(
            success=True,
            session_id=session_id,
            response=response_text,
            sources=sources,
            ticket_id=ticket_id,
        )
