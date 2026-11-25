import json
import os
from pathlib import Path
from uuid import uuid4

import requests


API_URL = os.getenv(
    "UNIBOT_API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

RESULTS_PATH = Path("evaluation/api_streaming_results.json")


def read_sse(response):
    event_name = "message"
    data_lines = []
    events = []

    for line in response.iter_lines(
        chunk_size=1,
        decode_unicode=True
    ):
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")

        if not line:
            if data_lines:
                raw = "\n".join(data_lines)

                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    data = {"raw": raw}

                events.append((event_name, data))

            event_name = "message"
            data_lines = []
            continue

        if line.startswith("event:"):
            event_name = line[6:].strip()

        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    if data_lines:
        events.append((event_name, json.loads("\n".join(data_lines))))

    return events


def send_message(query, thread_id):
    with requests.post(
        f"{API_URL}/chat",
        json={
            "query": query,
            "thread_id": thread_id
        },
        headers={"Accept": "text/event-stream"},
        stream=True,
        timeout=(10, 300)
    ) as response:

        response.raise_for_status()

        content_type = response.headers.get("content-type", "")

        if "text/event-stream" not in content_type.lower():
            raise RuntimeError(f"Unexpected content type: {content_type}")

        response.encoding = "utf-8"
        events = read_sse(response)

    errors = [data for name, data in events if name == "error"]

    if errors:
        raise RuntimeError(f"Streaming error: {errors}")

    starts = [data for name, data in events if name == "start"]
    tokens = [data for name, data in events if name == "token"]
    dones = [data for name, data in events if name == "done"]

    answer = "".join(
        str(data.get("text", ""))
        for data in tokens
    )

    return {
        "events": events,
        "answer": answer,
        "start": starts[0] if len(starts) == 1 else {},
        "done": dones[0] if len(dones) == 1 else {},
        "valid_structure": (
            len(starts) == 1
            and len(dones) == 1
            and len(tokens) > 0
            and bool(answer.strip())
        )
    }


def normalized(text):
    return "".join(
        char.lower()
        for char in str(text)
        if char.isalnum()
    )


def main():
    checks = {}
    results = {}

    health = requests.get(
        f"{API_URL}/health",
        timeout=10
    )
    health.raise_for_status()

    checks["api_health"] = health.json().get("status") == "ok"

    thread_id = str(uuid4())

    first = send_message(
        "What percentage of internships are paid at JECRC University?",
        thread_id
    )

    checks["first_stream"] = first["valid_structure"]

    checks["first_thread_id"] = (
        first["start"].get("thread_id") == thread_id
        and first["done"].get("thread_id") == thread_id
    )

    checks["first_answer"] = (
        "61" in normalized(first["answer"])
    )

    followup = send_message(
        "And what about the monthly stipend?",
        thread_id
    )

    checks["followup_stream"] = followup["valid_structure"]

    checks["followup_thread_id"] = (
        followup["start"].get("thread_id") == thread_id
        and followup["done"].get("thread_id") == thread_id
    )

    followup_text = normalized(followup["answer"])

    checks["followup_answer"] = (
        ("5000" in followup_text or "5k" in followup_text)
        and ("60000" in followup_text or "60k" in followup_text)
    )

    fallback = send_message(
        "Recommend a smartphone under 25000 rupees.",
        str(uuid4())
    )

    checks["fallback_stream"] = fallback["valid_structure"]

    clarification = send_message(
        "What are the requirements?",
        str(uuid4())
    )

    checks["clarification_stream"] = clarification["valid_structure"]

    results = {
        "first_question": {
            "answer": first["answer"],
            "events": [name for name, _ in first["events"]]
        },
        "followup": {
            "answer": followup["answer"],
            "events": [name for name, _ in followup["events"]]
        },
        "fallback": {
            "answer": fallback["answer"],
            "events": [name for name, _ in fallback["events"]]
        },
        "clarification": {
            "answer": clarification["answer"],
            "events": [name for name, _ in clarification["events"]]
        }
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "checks": checks,
                "passed": all(checks.values()),
                "results": results
            },
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 55)
    print("UNI-BOT FASTAPI STREAMING TEST")
    print("=" * 55)

    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    print("\n" + "=" * 55)
    print(f"TOTAL: {sum(checks.values())}/{len(checks)}")
    print("RESULT:", "PASS" if all(checks.values()) else "FAIL")
    print("=" * 55)

    print("\nFIRST ANSWER:\n", first["answer"])
    print("\nFOLLOW-UP ANSWER:\n", followup["answer"])

    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()