import os
import uuid
import hashlib

import httpx
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Customer Support", page_icon="🎧")
st.title("Customer Support")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
# Cache for synthesized audio: msg_key -> audio_bytes
if "tts_cache" not in st.session_state:
    st.session_state.tts_cache = {}

# Voice transcript state
if "voice_transcript" not in st.session_state:
    st.session_state.voice_transcript = None
if "voice_processing" not in st.session_state:
    st.session_state.voice_processing = False


def _msg_key(idx: int) -> str:
    """Stable per-message key for TTS cache/state."""
    return f"msg-{idx}"


def _send_chat(prompt: str):
    """Send a message through the existing POST /chat flow and handle the response."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                with httpx.Client(timeout=60.0) as client:
                    resp = client.post(
                        f"{API_BASE_URL}/chat",
                        json={
                            "session_id": st.session_state.session_id,
                            "message": prompt,
                        },
                    )

                if resp.status_code == 422:
                    detail = resp.json().get("detail", "Invalid input")
                    st.error(f"Validation error: {detail}")
                elif resp.status_code == 503:
                    st.warning("The system is still warming up. Please try again in a moment.")
                elif resp.status_code == 502:
                    st.error("The agent encountered an error processing your request. Please try again.")
                elif resp.status_code >= 400:
                    st.error(f"Server error (HTTP {resp.status_code}). Please try again.")
                else:
                    try:
                        data = resp.json()
                    except Exception:
                        st.error("Received an invalid response from the server. Please try again.")
                        st.stop()

                    response_text = data.get("response", "")
                    sources = data.get("sources", [])
                    ticket_id = data.get("ticket_id")

                    st.write(response_text)
                    if sources:
                        st.caption("Sources: " + ", ".join(sources))
                    if ticket_id:
                        st.success(f"🎫 Ticket created: {ticket_id}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "sources": sources,
                        "ticket_id": ticket_id,
                    })

            except httpx.ConnectError:
                st.error("Could not connect to the backend server. Please make sure the API is running.")
            except httpx.TimeoutException:
                st.error("The request timed out. Please try again.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")


def _synthesize_and_play(msg_key: str, text: str):
    """Call POST /voice/synthesize and cache/play the resulting audio."""
    # Already cached?
    if msg_key in st.session_state.tts_cache:
        st.audio(st.session_state.tts_cache[msg_key], format="audio/mpeg")
        return

    with st.spinner("Generating audio..."):
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    f"{API_BASE_URL}/voice/synthesize",
                    json={"message_id": msg_key, "text": text},
                )

            if resp.status_code >= 400:
                try:
                    detail = resp.json().get("detail", "Synthesis failed")
                except Exception:
                    detail = f"HTTP {resp.status_code}"
                st.warning(f"🔇 Could not generate audio: {detail}")
            else:
                audio_bytes = resp.content
                st.session_state.tts_cache[msg_key] = audio_bytes
                st.rerun()
        except httpx.ConnectError:
            st.warning("🔇 Could not connect to the voice service.")
        except httpx.TimeoutException:
            st.warning("🔇 Voice synthesis timed out.")
        except Exception as e:
            st.warning(f"🔇 Voice error: {str(e)}")


# ---------------------------------------------------------------------------
# Render chat history (with speaker icons on assistant messages)
# ---------------------------------------------------------------------------
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            sources = message.get("sources", [])
            if sources:
                st.caption("Sources: " + ", ".join(sources))
            tid = message.get("ticket_id")
            if tid:
                st.success(f"🎫 Ticket created: {tid}")

            # Speaker icon for TTS playback
            key = _msg_key(idx)
            # Show cached audio if available
            if key in st.session_state.tts_cache:
                st.audio(st.session_state.tts_cache[key], format="audio/mpeg")
            else:
                # Speaker button
                if st.button(
                    "🔊 Listen",
                    key=f"tts-btn-{key}",
                ):
                    _synthesize_and_play(key, message["content"])


# ---------------------------------------------------------------------------
# Voice input section (microphone)
# ---------------------------------------------------------------------------
st.divider()

col_mic, col_status = st.columns([1, 2])

with col_mic:
    audio_input = st.audio_input("🎤 Record voice message", key="mic_input")

with col_status:
    if audio_input is not None and not st.session_state.voice_processing:
        st.session_state.voice_processing = True
        with st.spinner("🎙️ Transcribing..."):
            try:
                audio_bytes = audio_input.read()
                with httpx.Client(timeout=30.0) as client:
                    resp = client.post(
                        f"{API_BASE_URL}/voice/transcribe",
                        files={"file": ("recording.wav", audio_bytes, "audio/wav")},
                    )

                if resp.status_code >= 400:
                    try:
                        detail = resp.json().get("detail", "Transcription failed")
                    except Exception:
                        detail = f"HTTP {resp.status_code}"
                    st.error(f"Transcription error: {detail}")
                    st.session_state.voice_processing = False
                else:
                    data = resp.json()
                    st.session_state.voice_transcript = data.get("transcript", "")
                    st.session_state.voice_processing = False
                    st.rerun()
            except httpx.ConnectError:
                st.error("Could not connect to the voice service. Is the API running?")
                st.session_state.voice_processing = False
            except httpx.TimeoutException:
                st.error("Transcription timed out. Please try again.")
                st.session_state.voice_processing = False
            except Exception as e:
                st.error(f"Transcription error: {str(e)}")
                st.session_state.voice_processing = False

# Editable transcript + submit
if st.session_state.voice_transcript is not None:
    st.info("📝 Review and edit your transcription below, then click Submit:")
    edited = st.text_area(
        "Transcript",
        value=st.session_state.voice_transcript,
        key="transcript_editor",
        label_visibility="collapsed",
    )

    col_submit, col_cancel = st.columns([1, 1])
    with col_submit:
        if st.button("✅ Submit", key="voice_submit", type="primary"):
            if edited.strip():
                st.session_state.voice_transcript = None
                _send_chat(edited.strip())
                st.rerun()
            else:
                st.warning("Cannot submit an empty message.")
    with col_cancel:
        if st.button("❌ Cancel", key="voice_cancel"):
            st.session_state.voice_transcript = None
            st.rerun()

# ---------------------------------------------------------------------------
# Typed chat input (existing behavior preserved exactly)
# ---------------------------------------------------------------------------
if prompt := st.chat_input("How can we help?"):
    _send_chat(prompt)
    st.rerun()
