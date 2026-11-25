from __future__ import annotations

import json
import os
from typing import Any, Iterator
from uuid import uuid4

import requests
import streamlit as st


API_BASE_URL = os.getenv(
    "UNIBOT_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

CHAT_URL = f"{API_BASE_URL}/chat"
HEALTH_URL = f"{API_BASE_URL}/health"

st.set_page_config(
    page_title="Uni-Bot | JECRC University",
    page_icon="🎓",
    layout="centered",
)


# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


# ------------------------------------------------------------
# API HELPERS
# ------------------------------------------------------------

@st.cache_data(ttl=10, show_spinner=False)
def check_api_health() -> tuple[bool, str]:
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        response.raise_for_status()
        body = response.json()
        return True, body.get("status", "ok")
    except requests.RequestException as exc:
        return False, str(exc)
    except ValueError:
        return False, "Health endpoint returned invalid JSON."


def iter_sse_events(response: requests.Response) -> Iterator[tuple[str, Any]]:
    """Parse Server-Sent Events from a requests streaming response."""
    event_name = "message"
    data_lines: list[str] = []

    def emit_current():
        if not data_lines:
            return None

        raw_data = "\n".join(data_lines)
        try:
            payload = json.loads(raw_data)
        except json.JSONDecodeError:
            payload = {"raw": raw_data}

        return event_name, payload

    for raw_line in response.iter_lines(chunk_size=1, decode_unicode=True):
        if raw_line is None:
            continue

        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="replace")
        else:
            line = raw_line

        if line == "":
            event = emit_current()
            if event is not None:
                yield event
            event_name = "message"
            data_lines = []
            continue

        if line.startswith(":"):
            # SSE comment/heartbeat; ignore it.
            continue

        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())

    # Process a final event even if the stream did not end with a blank line.
    event = emit_current()
    if event is not None:
        yield event



# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

with st.sidebar:
    st.title("🎓 Uni-Bot")
    st.caption("JECRC University Knowledge Assistant")

    api_ok, api_status = check_api_health()
    if api_ok:
        st.success("Backend connected")
    else:
        st.error("Backend unavailable")
        st.caption(api_status)

    st.caption(f"API: {API_BASE_URL}")
    st.divider()

    if st.button("＋ New conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid4())
        st.session_state.messages = []
        st.rerun()

    with st.expander("Conversation details"):
        st.code(st.session_state.thread_id, language=None)


# ------------------------------------------------------------
# CHAT UI
# ------------------------------------------------------------

st.title("Ask Uni-Bot")
st.caption(
    "Ask about JECRC University or get help understanding academic and technical topics."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask Uni-Bot a question...")

if question:
    question = question.strip()

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            answer_placeholder.markdown("Connecting to Uni-Bot...")

            answer_parts: list[str] = []
            response_metadata: dict[str, Any] = {}
            error_message = None

            try:
                payload = {
                    "query": question,
                    "thread_id": st.session_state.thread_id,
                }

                with requests.post(
                    CHAT_URL,
                    json=payload,
                    headers={"Accept": "text/event-stream"},
                    stream=True,
                    timeout=(10, 300),
                ) as response:
                    response.raise_for_status()
                    response.encoding = "utf-8"

                    content_type = response.headers.get("content-type", "")
                    if "text/event-stream" not in content_type.lower():
                        raise RuntimeError(
                            f"Expected an SSE response, received: {content_type or 'unknown content type'}"
                        )

                    for event_name, data in iter_sse_events(response):
                        if not isinstance(data, dict):
                            continue

                        if event_name == "start":
                            received_thread_id = data.get("thread_id")
                            if received_thread_id:
                                st.session_state.thread_id = received_thread_id

                        elif event_name == "token":
                            text = data.get("text", "")
                            if text:
                                answer_parts.append(str(text))
                                answer_placeholder.markdown("".join(answer_parts))

                        elif event_name == "done":
                            response_metadata.update(data)

                        elif event_name == "error":
                            error_message = data.get(
                                "detail",
                                "Uni-Bot could not process your request.",
                            )

                answer = "".join(answer_parts).strip()

                if error_message:
                    if answer:
                        answer_placeholder.markdown(answer)
                        st.warning(error_message)
                    else:
                        answer = error_message
                        answer_placeholder.error(answer)

                elif not answer:
                    answer = "Uni-Bot returned an empty response. Please try again."
                    answer_placeholder.warning(answer)

                else:
                    answer_placeholder.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                })

            except requests.HTTPError as exc:
                response_text = ""
                if exc.response is not None:
                    response_text = exc.response.text[:1000]
                message = (
                    f"API request failed ({exc.response.status_code if exc.response else 'HTTP error'}). "
                    f"{response_text}"
                )
                answer_placeholder.error(message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": message,
                })

            except requests.RequestException as exc:
                message = f"Could not connect to Uni-Bot API: {exc}"
                answer_placeholder.error(message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": message,
                })

            except Exception as exc:
                message = f"Unexpected frontend error: {exc}"
                answer_placeholder.error(message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": message,
                })
