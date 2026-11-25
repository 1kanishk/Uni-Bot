
import requests

url = "http://localhost:11434/api/chat"

payload = {
    "model": "qwen3.5:9b",
    "messages": [
        {
            "role": "user",
            "content": "What is RAG? Explain briefly."
        }
    ],
    "think": False,
    "stream": False,
    "keep_alive": "10m",
    "options": {
        "temperature": 0.2,
        "num_ctx": 4096,
        "num_predict": 256
    }
}

response = requests.post(url, json=payload, timeout=300)
result = response.json()

print(result["message"]["content"])
print("\nThinking:", result["message"].get("thinking", "Disabled"))

