from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_chroma import Chroma

from src.config import Settings
from src.rag.document_loader import load_support_documents, split_support_documents
from src.rag.embeddings import build_embeddings
from src.utils.errors import ComponentNotReadyError


class KnowledgeRetriever:
    """Persistent Chroma retrieval scaffold with stable output contracts."""

    RELEVANCE_THRESHOLD = 0.3

    def __init__(self, settings: Settings, documents_dir: Path) -> None:
        self.settings = settings
        self.documents_dir = documents_dir
        self._store: Chroma | None = None

    async def initialize(self) -> None:
        documents = split_support_documents(load_support_documents(self.documents_dir))
        embeddings = build_embeddings(self.settings)

        persist_dir = str(Path(self.settings.vector_db_path).resolve())
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self._store = Chroma(
            collection_name=self.settings.rag_collection,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )

        ids = []
        texts = []
        metadatas = []
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            content = doc.page_content
            chunk_id = hashlib.sha256(f"{source}:{content}".encode()).hexdigest()
            ids.append(chunk_id)
            texts.append(content)
            metadatas.append({"source": source})

        self._store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    async def search(self, query: str, limit: int | None = None) -> list[dict[str, str]]:
        if self._store is None:
            raise ComponentNotReadyError("Retriever has not been initialized")

        if not query or not query.strip():
            return []

        k = limit if limit is not None else self.settings.rag_top_k

        results = self._store.similarity_search_with_relevance_scores(query, k=k)

        output: list[dict[str, str]] = []
        for doc, score in results:
            if score < self.RELEVANCE_THRESHOLD:
                continue
            source = doc.metadata.get("source", "unknown")
            source = Path(source).name
            output.append({"content": doc.page_content, "source": source})

        return output
