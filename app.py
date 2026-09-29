from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from ingest import build_index
from rag import DuoRAG

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
VECTOR_DIR = BASE_DIR / "vectorstore"

st.set_page_config(
    page_title="DuoRAG Assistant",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 DuoRAG Assistant")
st.caption("Personal Academic & Portfolio RAG Chatbot for Uzair Bin Ahmad and Muhammad Zain")

st.markdown(
    """
Ask about **education, technical skills, projects, experience, NLP/ML coursework**, or compare the two profiles.

Examples:
- What projects has Uzair built?
- What is Zain's professional experience?
- Compare Uzair and Zain's frontend skills.
- Which person has worked on an NLP-related project?
"""
)

# Streamlit Cloud stores secrets in st.secrets rather than a local .env file.
try:
    if not os.getenv("GEMINI_API_KEY"):
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
except Exception:
    pass

if not (VECTOR_DIR / "index.faiss").exists():
    with st.spinner("Preparing the personal knowledge base for the first time..."):
        build_index()


@st.cache_resource(show_spinner="Loading embeddings and RAG pipeline...")
def load_rag() -> DuoRAG:
    return DuoRAG()


try:
    rag = load_rag()
except Exception as exc:
    st.error(str(exc))
    st.info("Add GEMINI_API_KEY, then restart the app.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Retrieved sources"):
                for src in message["sources"]:
                    st.write(f"• {src}")

question = st.chat_input("Ask about Uzair or Zain...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    history_for_rag = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant personal data..."):
            answer, retrieved = rag.answer(question, history_for_rag)
        st.markdown(answer)

        sources = sorted({r["source"] for r in retrieved})
        with st.expander("Retrieved sources"):
            for src in sources:
                st.write(f"• {src}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

with st.sidebar:
    st.header("About this project")
    st.write("RAG pipeline: Documents → Chunks → Embeddings → FAISS → Retrieval → Gemini → Answer")
    st.write("Embeddings: all-MiniLM-L6-v2")
    st.write("Vector DB: FAISS")
    st.write("LLM: Gemini 2.5 Flash")
    st.write("History: Streamlit session state")

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()
