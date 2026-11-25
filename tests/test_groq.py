from src.llm import get_groq_llm

llm = get_groq_llm()

for chunk in llm.stream(
    "Explain the difference between RAG and fine-tuning."
):
    if isinstance(chunk.content, str):
        print(chunk.content, end="", flush=True)
