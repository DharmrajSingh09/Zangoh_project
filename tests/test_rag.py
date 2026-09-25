import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import Settings
from src.rag.retriever import KnowledgeRetriever
from src.rag.document_loader import load_support_documents, split_support_documents
from src.utils.errors import ComponentNotReadyError


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        llm_base_url="http://localhost:11434/v1",
        llm_api_key="not-required",
        llm_model="test-model",
        vector_db_path=str(tmp_path / "vector_db"),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        rag_collection="test-support",
        rag_top_k=3,
        api_host="127.0.0.1",
        api_port=8000,
        streamlit_host="127.0.0.1",
        streamlit_port=8501,
    )


def test_loader_preserves_source_names(knowledge_dir) -> None:
    documents = load_support_documents(knowledge_dir)
    assert {item.metadata["source"] for item in documents} == {
        "accounts.md",
        "payments.md",
        "returns.md",
        "shipping.md",
    }


def test_splitter_keeps_source_metadata(knowledge_dir) -> None:
    chunks = split_support_documents(load_support_documents(knowledge_dir))
    assert chunks
    assert all(chunk.metadata.get("source", "").endswith(".md") for chunk in chunks)


@pytest.mark.asyncio
async def test_search_before_initialization(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    retriever = KnowledgeRetriever(settings, tmp_path / "nonexistent")
    with pytest.raises(ComponentNotReadyError):
        await retriever.search("test query")


class FakeEmbeddings:
    def embed_documents(self, texts):
        results = []
        for text in texts:
            h = hashlib.md5(text.encode()).hexdigest()
            vec = [int(c, 16) / 15.0 for c in h[:32]]
            results.append(vec)
        return results

    def embed_query(self, text):
        h = hashlib.md5(text.encode()).hexdigest()
        return [int(c, 16) / 15.0 for c in h[:32]]


@pytest.mark.asyncio
async def test_initialize_idempotency(knowledge_dir, tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    retriever = KnowledgeRetriever(settings, knowledge_dir)

    with patch("src.rag.retriever.build_embeddings", return_value=FakeEmbeddings()):
        await retriever.initialize()
        count_1 = retriever._store._collection.count()

        await retriever.initialize()
        count_2 = retriever._store._collection.count()

    assert count_1 == count_2
    assert count_1 > 0


@pytest.mark.asyncio
async def test_search_relevant_query(knowledge_dir, tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    retriever = KnowledgeRetriever(settings, knowledge_dir)

    with patch("src.rag.retriever.build_embeddings", return_value=FakeEmbeddings()):
        await retriever.initialize()
        results = await retriever.search("return policy")

    for r in results:
        assert "content" in r
        assert "source" in r
        assert not r["source"].startswith("/")
        assert not r["source"].startswith("C:")
        assert r["source"].endswith(".md")


@pytest.mark.asyncio
async def test_search_blank_query(knowledge_dir, tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    retriever = KnowledgeRetriever(settings, knowledge_dir)

    with patch("src.rag.retriever.build_embeddings", return_value=FakeEmbeddings()):
        await retriever.initialize()

    result = await retriever.search("")
    assert result == []

    result2 = await retriever.search("   ")
    assert result2 == []
