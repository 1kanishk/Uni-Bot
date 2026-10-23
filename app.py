# from pathlib import Path

# import streamlit as st

# from src.config import DATA_DIR
# from src.loader import load_documents, split_documents
# from src.vector_store import (
#     create_vector_store,
#     load_documents_from_vectorstore,
#     load_vector_store,
# )
# from src.retriever import AdvancedRetriever
# from src.rag_chain import RAGChain
# from src.llm import get_llm


# st.set_page_config(
#     page_title="Uni-Bot | JECRC University",
#     page_icon="🎓",
#     layout="centered",
# )

# DATA_PATH = Path(DATA_DIR)
# DATA_PATH.mkdir(parents=True, exist_ok=True)

# st.session_state.setdefault("messages", [])
# st.session_state.setdefault("vectorstore", None)
# st.session_state.setdefault("documents", None)
# st.session_state.setdefault("rag", None)
# st.session_state.setdefault("kb_checked", False)
# st.session_state.setdefault("kb_error", None)
# st.session_state.setdefault("kb_notice", None)


# def load_existing_knowledge_base():
#     try:
#         vectorstore = load_vector_store()
#         documents = load_documents_from_vectorstore()

#         if vectorstore is not None and documents:
#             st.session_state.vectorstore = vectorstore
#             st.session_state.documents = documents
#     except Exception as exc:
#         st.session_state.kb_error = str(exc)
#         print(f"Knowledge-base loading error: {exc}")
#     finally:
#         st.session_state.kb_checked = True


# def rebuild_knowledge_base():
#     documents = load_documents(DATA_PATH)

#     if not documents:
#         raise RuntimeError(
#             f"No supported PDF or DOCX files were found in {DATA_PATH}."
#         )

#     chunks = split_documents(documents)

#     if not chunks:
#         raise RuntimeError("Document loading succeeded, but no chunks were created.")

#     vectorstore = create_vector_store(chunks)

#     st.session_state.vectorstore = vectorstore
#     st.session_state.documents = chunks
#     st.session_state.rag = None
#     st.session_state.kb_error = None
#     st.session_state.kb_checked = True
#     st.session_state.kb_notice = (
#         f"Knowledge base rebuilt: {len(documents)} documents, {len(chunks)} chunks."
#     )


# def get_rag():
#     if st.session_state.rag is None:
#         if st.session_state.vectorstore is None or not st.session_state.documents:
#             raise RuntimeError("The knowledge base is not ready. Rebuild it from the sidebar.")

#         retriever = AdvancedRetriever(
#             st.session_state.documents,
#             st.session_state.vectorstore,
#         )
#         st.session_state.rag = RAGChain(retriever, get_llm())

#     return st.session_state.rag


# def show_sources(sources):
#     if not sources:
#         return

#     source = sources[0]
#     filename = source.get("source") or "University knowledge base"
#     page = source.get("page")

#     with st.expander("Source", expanded=False):
#         st.write(filename)
#         if page is not None:
#             st.caption(f"Page {page}")


# if not st.session_state.kb_checked:
#     load_existing_knowledge_base()

# with st.sidebar:
#     st.title("Uni-Bot")
#     st.caption("JECRC University Knowledge Assistant")

#     if st.session_state.documents:
#         st.success(f"Knowledge base ready · {len(st.session_state.documents):,} chunks")
#     else:
#         st.warning("Knowledge base not loaded")

#     if st.session_state.kb_error:
#         st.caption("The saved knowledge base could not be loaded. Rebuild it below.")

#     if st.session_state.kb_notice:
#         st.success(st.session_state.kb_notice)
#         st.session_state.kb_notice = None

#     st.divider()
#     st.subheader("Documents")
#     uploaded_files = st.file_uploader(
#         "Add PDF or DOCX files",
#         type=["pdf", "docx"],
#         accept_multiple_files=True,
#     )

#     if st.button(
#         "Save uploaded files",
#         use_container_width=True,
#         disabled=not uploaded_files,
#     ):
#         saved = 0
#         for uploaded_file in uploaded_files:
#             filename = Path(uploaded_file.name).name
#             if Path(filename).suffix.lower() not in {".pdf", ".docx"}:
#                 continue
#             (DATA_PATH / filename).write_bytes(uploaded_file.getvalue())
#             saved += 1

#         st.session_state.kb_notice = f"Saved {saved} file(s). Rebuild the knowledge base to index them."
#         st.rerun()

#     if st.button("Rebuild knowledge base", use_container_width=True):
#         try:
#             with st.spinner("Loading documents and rebuilding FAISS..."):
#                 rebuild_knowledge_base()
#             st.success(st.session_state.kb_notice)
#             st.session_state.kb_notice = None
#         except Exception as exc:
#             st.session_state.kb_error = str(exc)
#             print(f"Knowledge-base rebuild error: {exc}")
#             st.error("Rebuild failed. Check the terminal for the error details.")

#     st.divider()
#     if st.button("Clear chat", use_container_width=True):
#         st.session_state.messages = []
#         st.rerun()

# st.title("Ask Uni-Bot")
# st.caption("Ask questions about JECRC University. Answers are based on the loaded university documents.")

# if not st.session_state.documents:
#     st.info("Upload PDF/DOCX files and select **Rebuild knowledge base** in the sidebar to get started.")

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
#         if message["role"] == "assistant":
#             show_sources(message.get("sources", []))

# question = st.chat_input(
#     "Ask about courses, fees, placements, admissions...",
#     disabled=not bool(st.session_state.documents),
# )

# if question:
#     st.session_state.messages.append({"role": "user", "content": question})

#     with st.chat_message("user"):
#         st.write(question)

#     with st.chat_message("assistant"):
#         with st.spinner("Searching the university documents..."):
#             try:
#                 answer, sources = get_rag().ask(question)
#                 st.markdown(answer)
#                 show_sources(sources)
#                 st.session_state.messages.append(
#                     {
#                         "role": "assistant",
#                         "content": answer,
#                         "sources": sources[:1] if sources else [],
#                     }
#                 )
#             except Exception as exc:
#                 print(f"RAG response error: {exc}")
#                 message = "Sorry, I couldn't process that question. Check the terminal for the error details."
#                 st.error(message)
#                 st.session_state.messages.append(
#                     {"role": "assistant", "content": message, "sources": []}
#                 )


from pathlib import Path
from time import perf_counter
import streamlit as st

from src.config import DATA_DIR, VECTOR_DB_DIR
from src.loader import load_documents, split_documents
from src.vector_store import (
    create_vector_store,
    load_documents_from_vectorstore,
    load_vector_store,
)
from src.retriever import AdvancedRetriever
from src.rag_chain import RAGChain
from src.llm import get_llm

st.set_page_config(
    page_title="Uni-Bot | JECRC University",
    page_icon="🎓",
    layout="centered",
)

DATA_PATH = Path(DATA_DIR)
VECTOR_PATH = Path(VECTOR_DB_DIR)
DATA_PATH.mkdir(parents=True, exist_ok=True)

st.session_state.setdefault("messages", [])
st.session_state.setdefault("kb_error", None)
st.session_state.setdefault("kb_notice", None)


def get_kb_signature():
    """Change the cache key whenever the saved FAISS index/chunks change."""
    names = ("index.faiss", "index.pkl", "chunks.pkl")
    signature = []
    for name in names:
        path = VECTOR_PATH / name
        if path.exists():
            stat = path.stat()
            signature.append((name, stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


@st.cache_resource(show_spinner=False)
def load_cached_knowledge_base(signature):
    if not signature:
        return None, None
    vectorstore = load_vector_store()
    documents = load_documents_from_vectorstore()
    if vectorstore is None or not documents:
        return None, None
    return vectorstore, documents


@st.cache_resource(show_spinner=False)
def load_llm_once():
    started = perf_counter()
    llm = get_llm()
    print(f"[Uni-Bot timing] LLM initialization: {perf_counter() - started:.2f}s")
    return llm


@st.cache_resource(show_spinner=False)
def load_retriever_once(signature):
    vectorstore, documents = load_cached_knowledge_base(signature)
    if vectorstore is None or not documents:
        raise RuntimeError("The knowledge base is not ready. Rebuild it from the sidebar.")
    started = perf_counter()
    retriever = AdvancedRetriever(documents, vectorstore)
    print(f"[Uni-Bot timing] Retriever initialization: {perf_counter() - started:.2f}s")
    return retriever


@st.cache_resource(show_spinner=False)
def load_rag_once(signature):
    started = perf_counter()
    rag = RAGChain(load_retriever_once(signature), load_llm_once())
    print(f"[Uni-Bot timing] RAG chain initialization: {perf_counter() - started:.2f}s")
    return rag


def rebuild_knowledge_base():
    documents = load_documents(DATA_PATH)
    if not documents:
        raise RuntimeError(f"No supported PDF or DOCX files were found in {DATA_PATH}.")

    chunks = split_documents(documents)
    if not chunks:
        raise RuntimeError("Document loading succeeded, but no chunks were created.")

    create_vector_store(chunks)

    # Invalidate only KB-dependent resources. Keep the loaded LLM in memory.
    load_rag_once.clear()
    load_retriever_once.clear()
    load_cached_knowledge_base.clear()

    st.session_state.kb_error = None
    st.session_state.kb_notice = (
        f"Knowledge base rebuilt: {len(documents)} documents, {len(chunks)} chunks."
    )


def show_sources(sources):
    if not sources:
        return
    source = sources[0]
    filename = source.get("source") or "University knowledge base"
    page = source.get("page")
    with st.expander("Source", expanded=False):
        st.write(filename)
        if page is not None:
            st.caption(f"Page {page}")


def show_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_sources(message.get("sources", []))
            timing = message.get("timing")
            if timing:
                if timing.get("setup", 0) >= 2:
                    st.caption(
                        f"Assistant setup: {timing['setup']:.1f}s · "
                        f"Answer generation: {timing['answer']:.1f}s"
                    )
                else:
                    st.caption(f"Answer generation: {timing['answer']:.1f}s")


signature = get_kb_signature()
try:
    vectorstore, documents = load_cached_knowledge_base(signature) if signature else (None, None)
except Exception as exc:
    vectorstore, documents = None, None
    st.session_state.kb_error = str(exc)
    print(f"Knowledge-base loading error: {exc}")

with st.sidebar:
    st.title("Uni-Bot")
    st.caption("JECRC University Knowledge Assistant")

    if documents:
        st.success(f"Knowledge base ready · {len(documents):,} chunks")
    else:
        st.warning("Knowledge base not loaded")

    if st.session_state.kb_error:
        st.caption("The saved knowledge base could not be loaded. Rebuild it below.")

    if st.session_state.kb_notice:
        st.success(st.session_state.kb_notice)
        st.session_state.kb_notice = None

    st.divider()
    st.subheader("Documents")
    uploaded_files = st.file_uploader(
        "Add PDF or DOCX files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )

    if st.button("Save uploaded files", use_container_width=True, disabled=not uploaded_files):
        saved = 0
        for uploaded_file in uploaded_files:
            filename = Path(uploaded_file.name).name
            if Path(filename).suffix.lower() in {".pdf", ".docx"}:
                (DATA_PATH / filename).write_bytes(uploaded_file.getvalue())
                saved += 1
        st.session_state.kb_notice = f"Saved {saved} file(s). Rebuild the knowledge base to index them."
        st.rerun()

    if st.button("Rebuild knowledge base", use_container_width=True):
        try:
            with st.spinner("Loading documents and rebuilding FAISS..."):
                rebuild_knowledge_base()
            st.rerun()
        except Exception as exc:
            st.session_state.kb_error = str(exc)
            print(f"Knowledge-base rebuild error: {exc}")
            st.error("Rebuild failed. Check the terminal for details.")

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("Ask Uni-Bot")
st.caption("Ask questions about JECRC University. Answers are based on the loaded university documents.")

if not documents:
    st.info("Upload PDF/DOCX files and select **Rebuild knowledge base** in the sidebar to get started.")

for saved_message in st.session_state.messages:
    show_message(saved_message)

question = st.chat_input(
    "Ask about courses, fees, placements, admissions...",
    disabled=not bool(documents),
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        setup_seconds = 0.0
        try:
            init_started = perf_counter()
            with st.spinner("Preparing Uni-Bot for the first question (model and retriever will be reused)..."):
                rag = load_rag_once(signature)
            setup_seconds = perf_counter() - init_started

            answer_started = perf_counter()
            with st.spinner("Searching university documents and generating your answer..."):
                answer, sources = rag.ask(question)
            answer_seconds = perf_counter() - answer_started

            st.markdown(answer)
            show_sources(sources)
            timing = {"setup": setup_seconds, "answer": answer_seconds}
            if setup_seconds >= 2:
                st.caption(
                    f"Assistant setup: {setup_seconds:.1f}s · "
                    f"Answer generation: {answer_seconds:.1f}s"
                )
            else:
                st.caption(f"Answer generation: {answer_seconds:.1f}s")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources[:1] if sources else [],
                    "timing": timing,
                }
            )
            print(
                f"[Uni-Bot timing] total setup={setup_seconds:.2f}s, "
                f"answer={answer_seconds:.2f}s"
            )
        except Exception as exc:
            print(f"RAG response error: {exc}")
            message = "Sorry, I couldn't process that question. Check the terminal for the error details."
            st.error(message)
            st.session_state.messages.append(
                {"role": "assistant", "content": message, "sources": []}
            )
