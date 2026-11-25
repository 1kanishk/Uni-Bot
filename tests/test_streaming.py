from graph.main import stream_uni_bot


question = "Explain binary search in Python."

print("\nUni-Bot: ", end="", flush=True)

for token in stream_uni_bot(question):
    print(token, end="", flush=True)

print()
