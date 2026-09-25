import os
import uuid

import httpx
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Customer Support", page_icon="🎧")
st.title("Customer Support")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            sources = message.get("sources", [])
            if sources:
                st.caption("Sources: " + ", ".join(sources))
            tid = message.get("ticket_id")
            if tid:
                st.success(f"🎫 Ticket created: {tid}")

if prompt := st.chat_input("How can we help?"):
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
