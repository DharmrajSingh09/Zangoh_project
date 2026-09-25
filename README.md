# Customer Support Ticket Agent

A guided starter project for building a text-based support agent using FastAPI, Streamlit, an open-source LLM, RAG, and a mock ticket tool — now extended with **voice input (STT) and spoken response playback (TTS)** as a mid-session requirement.

Read the separate Assignment Implementation Guide before changing the starter.

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI                            │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ 🎤 Mic   │  │ 📝 Editable  │  │ 💬 Chat + 🔊 Speaker  │ │
│  │ Record   │→ │ Transcript   │→ │ History    Icons       │ │
│  └──────────┘  └──────────────┘  └────────────────────────┘ │
│       ↓ (audio)     ↓ (confirmed text)     ↑ (text)  ↓ (text) │
└───────┼─────────────┼──────────────────────┼──────────┼─────┘
        │             │                      │          │
   POST /voice/    POST /chat          POST /chat   POST /voice/
   transcribe      (unchanged)         response     synthesize
        │             │                      │          │
┌───────┼─────────────┼──────────────────────┼──────────┼─────┐
│       ↓             ↓                      ↑          ↓     │
│  ┌──────────┐  ┌──────────────┐      ┌──────────┐ ┌──────┐ │
│  │ Voice    │  │ Support      │      │ Support  │ │Voice │ │
│  │ Pipeline │  │ Pipeline     │─────→│ Pipeline │ │Pipe- │ │
│  │ (STT)    │  │ (LangGraph)  │      │ response │ │line  │ │
│  └──────────┘  │  ├─ retrieve │      └──────────┘ │(TTS) │ │
│                │  ├─ decide   │                   └──────┘ │
│                │  ├─ answer   │                            │
│                │  └─ collect  │                            │
│                │     /create  │                            │
│                └──────────────┘                            │
│                       FastAPI Server                       │
└────────────────────────────────────────────────────────────┘
```

**Key principle**: Audio is converted to/from text **only at the edges**. The agent always receives plain confirmed text through the existing `POST /chat` → `SupportPipeline` → LangGraph workflow. Voice never bypasses the session, RAG, or ticket logic.

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
- Voice service contracts (abstract `STTService` / `TTSService`), models, and `VoicePipeline` scaffold.

## Candidate Work

Complete the TODOs in:

- `src/rag/retriever.py`
- `src/llm/workflow.py`
- `src/pipeline.py`
- `streamlit_app.py`

Mid-session requirement (voice):

- `src/voice/stt_whisper.py` — Concrete STT adapter
- `src/voice/tts_edge.py` — Concrete TTS adapter
- `src/api/server.py` — New `/voice/*` endpoints
- `streamlit_app.py` — Mic input, editable transcript, speaker icons
- `tests/test_voice_pipeline.py` — Extended voice pipeline tests
- `tests/test_voice_api.py` — Voice API endpoint + regression tests


## Requirements

- Python 3.11 or newer.
- An OpenAI-compatible endpoint serving an open-source instruction model, or an equivalent open-source model integration.
- Enough local space for the selected embedding model, vector index, and **Whisper STT model** (~150 MB for `base`).

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

### Voice-Specific Setup

The voice feature requires **faster-whisper** (STT) and **edge-tts** (TTS), both installed automatically via `requirements.txt`. No API keys are needed — both run locally or use free cloud services.

On first startup, faster-whisper will download the Whisper `base` model (~150 MB). This happens automatically and requires an internet connection only once.

**New environment variables** (all have safe defaults — see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `STT_MODEL_SIZE` | `base` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`) |
| `STT_DEVICE` | `cpu` | Compute device (`cpu` or `cuda`) |
| `STT_COMPUTE_TYPE` | `int8` | Quantization type (`int8`, `float16`, `float32`) |
| `TTS_VOICE` | `en-US-AriaNeural` | Edge TTS voice name (see `edge-tts --list-voices`) |

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

### Using the Voice Feature

1. **Record**: Click the 🎤 microphone button to record your message.
2. **Review**: The transcript appears in an editable text box — correct any STT errors.
3. **Submit**: Click ✅ Submit to send the edited text through the normal chat pipeline.
4. **Listen**: Click 🔊 Listen on any assistant response to hear it spoken aloud.
5. **Typed chat** continues to work exactly as before, side-by-side with voice.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Pipeline readiness check |
| `POST` | `/chat` | Text chat (unchanged from original) |
| `GET` | `/tickets/{ticket_id}` | Retrieve a created ticket |
| `POST` | `/voice/transcribe` | Upload audio file → get transcript JSON |
| `POST` | `/voice/synthesize` | Submit text → receive playable audio bytes |

### Voice Endpoint Details

**POST /voice/transcribe** — Accepts `multipart/form-data` with a `file` field:
```sh
curl -X POST http://localhost:8000/voice/transcribe \
  -F "file=@recording.wav;type=audio/wav"
```
Response: `{ "success": true, "transcript": "...", "processing_time_ms": 820 }`

**POST /voice/synthesize** — Accepts JSON:
```sh
curl -X POST http://localhost:8000/voice/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"message_id": "msg-0", "text": "Hello, how can I help?"}'
```
Response: Raw audio bytes (`audio/mpeg`) with `X-Processing-Time-Ms` header.

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

### Original Requirements
- [x] RAG answers use the supplied documents and return source names.
- [x] Unknown answers are not fabricated.
- [x] Ticket details are collected over multiple turns.
- [x] Tickets are created only after validation.
- [x] Repeated requests in one session do not create duplicate tickets.
- [x] API and Streamlit failures are displayed clearly.
- [x] Automated tests cover the principal success and failure paths.
- [x] This README is updated with the candidate's final architecture, provider choices, platform notes, and troubleshooting guidance.

### Mid-Session Voice Requirement
- [x] Microphone input with editable transcript (no auto-submit).
- [x] Voice input flows through existing `POST /chat` pipeline.
- [x] Speaker icon on every assistant response for TTS playback (no autoplay).
- [x] TTS audio cached per message within the session.
- [x] TTS failure does not hide the original text response.
- [x] Typed chat remains fully functional.
- [x] `POST /chat` contract unchanged.
- [x] New voice tests added; existing tests still pass.
- [x] No API keys or secrets committed.

## Architecture and Implementation Notes

### Architecture overview
The architecture is fundamentally a graph-based state machine driven by `langgraph`.
- **RAG/Retrieval**: `KnowledgeRetriever` leverages a persistent `Chroma` instance with `sentence-transformers/all-MiniLM-L6-v2` embeddings, ensuring chunks are deterministically identified to avoid duplication.
- **Workflow / Routing**:
  1. `retrieve`: Embeds the customer message and fetches relevant docs using relevance scores (threshold > 0.3).
  2. `decide`: Uses structured output to classify user intent ("answer" vs "ticket"), capturing missing fields incrementally if the customer wants a ticket or reporting an issue.
  3. `answer`: Synthesizes a grounded response based on `retrieve` chunks and clearly states when the KB doesn't have the answer.
  4. `collect_or_create` (ticket): Collects missing ticket fields (name, email, issue, category) interactively. Once complete, it calls the mocked `create_support_ticket` tool.
- **Voice Pipeline** (mid-session addition): Wraps STT and TTS adapters behind the existing abstract contracts (`STTService` / `TTSService`). Audio is converted to/from text at the API boundary; the agent workflow never sees audio directly.
- **API (FastAPI)**: Connects standard REST HTTP POST to LangGraph's orchestration state via `SupportPipeline`, gracefully wrapping pipeline exceptions (502/503/422). New `/voice/*` endpoints handle audio upload and synthesis without touching `POST /chat`.
- **UI (Streamlit)**: Robustly handles session-bound conversations, HTTP errors, ticket banners, RAG attribution rendering, mic recording, editable transcripts, and per-message TTS playback with caching.

### Provider Choices
- **LLM**: Local model `qwen2.5:3b` run via Ollama, accessible through `ChatOpenAI`.
- **Embeddings**: Local HuggingFace model `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector DB**: `Chroma` persistence in local file system (`.data/vector_db`).
- **STT**: `faster-whisper` — Chosen because it is **fully local, offline-capable, and free**. It uses CTranslate2-optimized Whisper models with int8 quantization for fast CPU inference. Consistent with the project's "local, open-source" philosophy.
- **TTS**: `edge-tts` — Chosen because it is **free, requires no API key**, and produces natural-sounding neural voices. While it calls a Microsoft Edge cloud endpoint, it needs no account or credentials. As a trade-off vs. a fully local option like Coqui TTS, it avoids the ~1 GB model download and CUDA dependency for comparable quality.

### Mid-Session Change Summary
The voice feature was added as a **mid-session requirement** on top of the fully-working text agent. What changed:
- **Added**: `src/voice/stt_whisper.py`, `src/voice/tts_edge.py` (concrete adapter implementations)
- **Added**: `POST /voice/transcribe`, `POST /voice/synthesize` in `src/api/server.py`
- **Modified**: `streamlit_app.py` (mic input, editable transcript, speaker icons, TTS caching)
- **Modified**: `src/config.py` (4 new env vars for voice)
- **Added**: `tests/test_voice_api.py` (13 tests), extended `tests/test_voice_pipeline.py` (12 tests)
- **Unchanged**: `POST /chat` contract, `SupportPipeline`, LangGraph workflow, RAG, sessions, tickets

### Troubleshooting
- **API 502 Errors**: Check if the LLM backend (Ollama) is running and the model (`qwen2.5:3b`) is pulled.
- **Missing modules**: Ensure you ran `pip install -r requirements.txt`.
- **Duplicate embeddings**: Handled smoothly using deterministic SHA256 hashes of the file content in the retriever.
- **Voice transcription errors**: Ensure faster-whisper is installed and the Whisper model has been downloaded (happens automatically on first run).
- **TTS silence / no audio**: Edge TTS requires internet access. Check network connectivity.
- **Microphone not working**: Streamlit's `audio_input` requires HTTPS in production; it works on `localhost` in development.

