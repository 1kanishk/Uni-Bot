from src.llm import get_groq_llm

llm = get_groq_llm()

response = llm.invoke(
    "What is Retrieval Augmented Generation? Explain in 3 points."
)

print("\nMODEL RESPONSE:\n")
print(response.content)

print("\nMODEL:", response.response_metadata.get("model_name"))
