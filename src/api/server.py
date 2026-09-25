from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import Response

from src.config import load_settings
from src.models import ChatRequest, ChatResponse, Ticket
from src.pipeline import SupportPipeline
from src.utils.errors import AgentProcessingError, ComponentNotReadyError
from src.voice.contracts import STTService, TTSService
from src.voice.models import TranscriptionResponse, SynthesisRequest
from src.voice.pipeline import VoicePipeline
from src.voice.stt_whisper import WhisperSTTService
from src.voice.tts_edge import EdgeTTSService


settings = load_settings()
pipeline = SupportPipeline(settings, Path(__file__).resolve().parents[2] / "knowledge_base")

# Voice pipeline constructed from settings; initialized during lifespan
_stt: STTService = WhisperSTTService(
    model_size=settings.stt_model_size,
    device=settings.stt_device,
    compute_type=settings.stt_compute_type,
)
_tts: TTSService = EdgeTTSService(voice=settings.tts_voice)
voice_pipeline = VoicePipeline(_stt, _tts)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Startup owns expensive shared initialization. Request handlers reuse the
    # resulting components, while shutdown makes readiness false immediately.
    await pipeline.initialize()
    await voice_pipeline.initialize()
    yield
    pipeline.ready = False
    await voice_pipeline.cleanup()


app = FastAPI(title="Customer Support Ticket Agent", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    if not pipeline.ready:
        raise HTTPException(status_code=503, detail="Support pipeline is not ready")
    return {"status": "ready"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    # Keep this transport boundary thin: Pydantic validates the public request,
    # the pipeline owns orchestration, and known service errors are translated
    # to stable HTTP responses here.
    try:
        return await pipeline.process(request.session_id, request.message)
    except ComponentNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AgentProcessingError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/tickets/{ticket_id}", response_model=Ticket)
async def get_ticket(ticket_id: str) -> Ticket:
    # Keep this endpoint read-only. It should return the exact repository record,
    # not ask the model to reconstruct ticket details. Test both the successful
    # lookup and unknown-ID response through FastAPI's test client.
    ticket = pipeline.tickets.get(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket


# ---------------------------------------------------------------------------
# Voice endpoints (mid-session requirement)
# ---------------------------------------------------------------------------

@app.post("/voice/transcribe")
async def voice_transcribe(file: UploadFile = File(...)):
    """Accept an audio upload, return the transcript as JSON."""
    if file.size is not None and file.size == 0:
        raise HTTPException(status_code=422, detail="Audio file is empty")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="Audio file is empty")

    # Determine media type from the upload header, fall back to wav
    media_type = file.content_type or "audio/wav"

    try:
        transcript, processing_time_ms = await voice_pipeline.transcribe(
            audio_bytes, media_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Transcription failed: {exc}",
        ) from exc

    return TranscriptionResponse(
        success=True,
        transcript=transcript,
        processing_time_ms=processing_time_ms,
    )


@app.post("/voice/synthesize")
async def voice_synthesize(request: SynthesisRequest):
    """Accept a text payload, return playable audio bytes."""
    try:
        audio_bytes, media_type, processing_time_ms = (
            await voice_pipeline.synthesize(request.text)
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Speech synthesis failed: {exc}",
        ) from exc

    return Response(
        content=audio_bytes,
        media_type=media_type,
        headers={
            "X-Processing-Time-Ms": str(processing_time_ms),
        },
    )
