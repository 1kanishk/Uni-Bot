
from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from graph.workflow import uni_bot_graph


# ============================================================
# 1. LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("unibot.api")


# ============================================================
# 2. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Uni-Bot API",
    description="University Information and Academic Learning Assistant",
    version="1.0.0",
)


# ============================================================
# 3. CORS CONFIGURATION
# ============================================================

# Read allowed frontend origins from environment variables.
origins = [
    origin.strip()
    for origin in os.getenv(
        "UNIBOT_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8501,http://127.0.0.1:8501",
    ).split(",")
    if origin.strip()
]

# Allow configured frontends to communicate with FastAPI.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# ============================================================
# 4. REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=4000,
        description="Student question",
    )

    thread_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Conversation thread identifier",
    )


# ============================================================
# 5. REQUEST PREPARATION
# ============================================================

def prepare_request(
    payload: ChatRequest,
) -> tuple[dict[str, Any], dict[str, Any], str]:

    # Clean and validate the student's question.
    query = payload.query.strip()

    if not query:
        raise HTTPException(
            status_code=422,
            detail="Query cannot be empty.",
        )

    # Validate the supplied thread ID.
    if payload.thread_id is not None and not payload.thread_id.strip():
        raise HTTPException(
            status_code=422,
            detail="Thread ID cannot be empty.",
        )

    # Reuse an existing thread ID or generate a new one.
    thread_id = (
        payload.thread_id.strip()
        if payload.thread_id
        else str(uuid4())
    )

    # Prepare the input expected by LangGraph.
    graph_input = {
        "query": query,
        "messages": [HumanMessage(content=query)],
    }

    # Pass the thread ID to LangGraph.
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    return graph_input, config, thread_id


# ============================================================
# 6. SSE EVENT HELPER
# ============================================================

def sse_event(event_name: str, data: dict[str, Any]) -> str:

    # Convert Python data into the Server-Sent Events format.
    return (
        f"event: {event_name}\n"
        f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    )


# ============================================================
# 7. ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Welcome to Uni-Bot API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
    }


# ============================================================
# 8. HEALTH CHECK ENDPOINT
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "ok",
        "service": "uni-bot-api",
    }


# ============================================================
# 9. STREAMING CHAT ENDPOINT
# ============================================================


@app.post("/chat")
def chat_stream(payload: ChatRequest):

    graph_input, config, thread_id = prepare_request(payload)

    def generate_events() -> Iterator[str]:

        streamed = False
        final_answer = None

        intent = None
        intent_confidence = None

        try:

            # Send conversation ID immediately.
            yield sse_event(
                "start",
                {
                    "thread_id": thread_id
                },
            )

            # Execute LangGraph.
            for mode, chunk in uni_bot_graph.stream(
                graph_input,
                config=config,
                stream_mode=["custom", "updates"],
            ):

                if mode == "custom":

                    if isinstance(chunk, dict):
                        text = chunk.get("text", "")
                    else:
                        text = ""

                    if text:
                        streamed = True

                        yield sse_event(
                            "token",
                            {"text": text},
                        )

                elif mode == "updates" and isinstance(chunk, dict):

                    for node_name, node_update in chunk.items():

                        if not isinstance(node_update, dict):
                            continue

                        if "intent" in node_update:
                            intent = node_update["intent"]

                        if "intent_confidence" in node_update:
                            intent_confidence = node_update[
                                "intent_confidence"
                            ]

                        if node_update.get("answer") is not None:
                            final_answer = node_update["answer"]

            # Fallback if custom streaming produced no tokens.
            if not streamed:

                if not final_answer:
                    raise RuntimeError(
                        "Workflow did not generate an answer."
                    )

                yield sse_event(
                    "token",
                    {"text": final_answer},
                )

            # Completion event.
            yield sse_event(
                "done",
                {
                    "thread_id": thread_id,
                },
            )

            logger.info(
                "Streaming completed. Thread ID: %s",
                thread_id,
            )

        except Exception:

            logger.exception("Uni-Bot streaming request failed")

            yield sse_event(
                "error",
                {
                    "thread_id": thread_id,
                    "detail": "Uni-Bot could not process your request."
                },
            )

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
