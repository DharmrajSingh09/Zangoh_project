from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import Settings
from src.pipeline import SupportPipeline
from src.utils.errors import AgentProcessingError, ComponentNotReadyError


def make_settings() -> Settings:
    return Settings(
        llm_base_url="http://localhost:11434/v1",
        llm_api_key="not-required",
        llm_model="test-model",
        vector_db_path=".data/test-vector-db",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        rag_collection="test-support",
        rag_top_k=3,
        api_host="127.0.0.1",
        api_port=8000,
        streamlit_host="127.0.0.1",
        streamlit_port=8501,
    )


@pytest.mark.asyncio
async def test_uninitialized_pipeline_rejects_chat(tmp_path: Path) -> None:
    pipeline = SupportPipeline(make_settings(), tmp_path)
    with pytest.raises(ComponentNotReadyError):
        await pipeline.process("session-1", "hello")


@pytest.mark.asyncio
async def test_empty_message_raises_error(tmp_path: Path) -> None:
    pipeline = SupportPipeline(make_settings(), tmp_path)
    pipeline.ready = True
    pipeline.workflow = MagicMock()
    with pytest.raises(AgentProcessingError):
        await pipeline.process("session-1", "   ")


@pytest.mark.asyncio
async def test_empty_session_id_raises_error(tmp_path: Path) -> None:
    pipeline = SupportPipeline(make_settings(), tmp_path)
    pipeline.ready = True
    pipeline.workflow = MagicMock()
    with pytest.raises(AgentProcessingError):
        await pipeline.process("   ", "hello")


@pytest.mark.asyncio
async def test_successful_answer_response(tmp_path: Path) -> None:
    pipeline = SupportPipeline(make_settings(), tmp_path)
    pipeline.ready = True

    mock_workflow = AsyncMock()
    mock_workflow.ainvoke = AsyncMock(return_value={
        "response_text": "Returns are accepted within 30 days.",
        "sources": ["returns.md"],
        "ticket_id": None,
    })
    pipeline.workflow = mock_workflow

    result = await pipeline.process("session-1", "What is the return policy?")
    assert result.success is True
    assert result.session_id == "session-1"
    assert "30 days" in result.response or "Returns" in result.response
    assert "returns.md" in result.sources
    assert result.ticket_id is None


@pytest.mark.asyncio
async def test_workflow_failure_raises_processing_error(tmp_path: Path) -> None:
    pipeline = SupportPipeline(make_settings(), tmp_path)
    pipeline.ready = True

    mock_workflow = AsyncMock()
    mock_workflow.ainvoke = AsyncMock(side_effect=RuntimeError("LLM failed"))
    pipeline.workflow = mock_workflow

    with pytest.raises(AgentProcessingError):
        await pipeline.process("session-1", "hello")


@pytest.mark.asyncio
async def test_history_is_preserved(tmp_path: Path) -> None:
    pipeline = SupportPipeline(make_settings(), tmp_path)
    pipeline.ready = True

    mock_workflow = AsyncMock()
    mock_workflow.ainvoke = AsyncMock(return_value={
        "response_text": "Hello! How can I help?",
        "sources": [],
        "ticket_id": None,
    })
    pipeline.workflow = mock_workflow

    await pipeline.process("session-1", "Hi")
    conv = pipeline.sessions.get_or_create("session-1")
    assert len(conv.history) == 2
    assert conv.history[0]["role"] == "user"
    assert conv.history[0]["content"] == "Hi"
    assert conv.history[1]["role"] == "assistant"
