# Customer Support Ticket Agent

A guided starter project for building a text-based support agent using FastAPI, Streamlit, an open-source LLM, RAG, and a mock ticket tool.

Read the separate Assignment Implementation Guide before changing the starter.

## Architecture

```text
Streamlit -> FastAPI -> SupportPipeline -> LangGraph workflow
                                      |-> RAG/Chroma knowledge
                                      \-> session-bound ticket tool
```

## What Is Provided

- Request, response, and ticket models.
- A LangChain `ChatOpenAI` binding for an OpenAI-compatible open-source model.
- A typed LangGraph state, node skeleton, and routing graph.
- Document loading, splitting, embeddings, and Chroma component bindings.
- Session-state data structure.
- In-memory ticket repository with duplicate protection.
- Four support-policy documents.
- FastAPI and Streamlit scaffolding.
- RAG and agent integration TODOs.
- Initial repository and session tests.

## Candidate Work

Complete the TODOs in:

- `src/rag/retriever.py`
- `src/llm/workflow.py`
- `src/pipeline.py`
- `streamlit_app.py`

You may add or reorganize files when the resulting design remains clear and testable.

## Requirements

- Python 3.11 or newer.
- An OpenAI-compatible endpoint serving an open-source instruction model, or an equivalent open-source model integration.
- Enough local space for the selected embedding model and vector index.

## Setup

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

### Linux

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

### macOS

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Configure `.env` for the selected model endpoint. Do not commit credentials.

## Run

Start the API:

```sh
uvicorn src.api.server:app --reload --host "${API_HOST:-127.0.0.1}" --port "${API_PORT:-8000}"
```

In another terminal, start the UI:

```sh
streamlit run streamlit_app.py --server.address "${STREAMLIT_HOST:-127.0.0.1}" --server.port "${STREAMLIT_PORT:-8501}"
```

The default UI is `http://localhost:8501`; FastAPI documentation is `http://localhost:8000/docs`. Override both ports through `.env` and the corresponding command-line values when necessary.

## Test

```sh
pytest -q
```

Useful manual requests:

```sh
curl http://localhost:8000/health

curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo-1","message":"How long does standard shipping take?"}'
```

For Windows PowerShell, use `Invoke-RestMethod` or place the equivalent JSON request in FastAPI's `/docs` interface.

## Completion Checklist

- [x] RAG answers use the supplied documents and return source names.
- [x] Unknown answers are not fabricated.
- [x] Ticket details are collected over multiple turns.
- [x] Tickets are created only after validation.
- [x] Repeated requests in one session do not create duplicate tickets.
- [x] API and Streamlit failures are displayed clearly.
- [x] Automated tests cover the principal success and failure paths.
- [x] This README is updated with the candidate's final architecture, provider choices, platform notes, and troubleshooting guidance.

## Architecture and Implementation Notes

### Architecture overview
The architecture is fundamentally a graph-based state machine driven by `langgraph`.
- **RAG/Retrieval**: `KnowledgeRetriever` leverages a persistent `Chroma` instance with `sentence-transformers/all-MiniLM-L6-v2` embeddings, ensuring chunks are deterministically identified to avoid duplication.
- **Workflow / Routing**:
  1. `retrieve`: Embeds the customer message and fetches relevant docs using relevance scores (threshold > 0.3).
  2. `decide`: Uses structured output to classify user intent ("answer" vs "ticket"), capturing missing fields incrementally if the customer wants a ticket or reporting an issue.
  3. `answer`: Synthesizes a grounded response based on `retrieve` chunks and clearly states when the KB doesn't have the answer.
  4. `collect_or_create` (ticket): Collects missing ticket fields (name, email, issue, category) interactively. Once complete, it calls the mocked `create_support_ticket` tool.
- **API (FastAPI)**: Connects standard REST HTTP POST to LangGraph's orchestration state via `SupportPipeline`, gracefully wrapping pipeline exceptions (502/503/422).
- **UI (Streamlit)**: Robustly handles session-bound conversations, HTTP errors, ticket banners, and RAG attribution rendering.

### Provider Choices
- **LLM**: Local model `qwen2.5:3b` run via Ollama, accessible through `ChatOpenAI`.
- **Embeddings**: Local HuggingFace model `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector DB**: `Chroma` persistence in local file system (`.data/vector_db`).

### Troubleshooting
- **API 502 Errors**: Check if the LLM backend (Ollama) is running and the model (`qwen2.5:3b`) is pulled.
- **Missing modules**: Ensure you ran `pip install -r requirements.txt`.
- **Duplicate embeddings**: Handled smoothly using deterministic SHA256 hashes of the file content in the retriever.
