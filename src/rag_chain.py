# from src.llm import get_llm
# from src.prompt import get_prompt
# import re

# class RAGChain:

#     def __init__(self, retriever, llm):

#         self.retriever = retriever
#         self.llm = llm
#         self.prompt = get_prompt()

#     def ask(self, question):

#         documents = self.retriever.retrieve(
#             question
#         )

#         context = "\n\n".join(
#             f"[Context {i+1}]\n{doc.page_content}"
#             for i, doc in enumerate(documents)
#         )

#         messages = self.prompt.format_messages(
#             context=context,
#             question=question
#         )

#         response = self.llm.invoke(messages)

#         sources = [
#             {
#                 "source": doc.metadata.get("source"),
#                 "page": doc.metadata.get("page")
#             }
#             for doc in documents
#         ]   

#         return response.content, documents ,sources

import re

from src.prompt import get_prompt
from src.llm import get_llm, get_groq_llm

def clean_context(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class RAGChain:

    llm = get_groq_llm()

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.prompt = get_prompt()

    def ask(self, question):

        documents = self.retriever.retrieve(question)

        context = "\n\n".join(
            f"[Context {i+1}]\n{clean_context(doc.page_content)}"
            for i, doc in enumerate(documents)
        )

        messages = self.prompt.format_messages(
            context=context,
            question=question
        )

        response = self.llm.invoke(messages)

        sources = [
            {
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page")
            }
            for doc in documents
        ]

        return response.content, sources