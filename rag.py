from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
VECTOR_DIR = BASE_DIR / "vectorstore"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

load_dotenv()


class DuoRAG:
    def __init__(self) -> None:
        index_path = VECTOR_DIR / "index.faiss"
        chunks_path = VECTOR_DIR / "chunks.json"

        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                "Vector store not found. Run `python ingest.py` first."
            )

        self.index = faiss.read_index(str(index_path))
        self.records = json.loads(
            chunks_path.read_text(encoding="utf-8")
        )
        self.embedder = SentenceTransformer(MODEL_NAME)

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing. "
                "Add it to .env locally or Streamlit Secrets when deployed."
            )

        self.client = genai.Client(api_key=api_key)

    def retrieve(
        self,
        question: str,
        k: int = TOP_K,
    ) -> list[dict[str, Any]]:

        query = self.embedder.encode(
            [question],
            normalize_embeddings=True,
        )

        query = np.asarray(query, dtype="float32")

        scores, indices = self.index.search(query, k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            record = dict(self.records[idx])
            record["score"] = float(score)
            results.append(record)

        return results

    @staticmethod
    def _history_to_text(
        history: list[dict[str, str]],
        limit: int = 6,
    ) -> str:

        if not history:
            return "No previous conversation."

        recent = history[-limit:]

        return "\n".join(
            f"{m['role'].title()}: {m['content']}"
            for m in recent
        )

    def answer(
        self,
        question: str,
        history: list[dict[str, str]],
    ) -> tuple[str, list[dict[str, Any]]]:

        retrieved = self.retrieve(question)

        context = "\n\n".join(
            f"[Source: {r['source']}]\n{r['text']}"
            for r in retrieved
        )

        prompt = f"""
You are DuoRAG Assistant, a personal academic and portfolio chatbot
for two Software Engineering students:
Uzair Bin Ahmad and Muhammad Zain.

Your job is to answer questions ONLY from the retrieved context below.

Rules:
1. Use the retrieved context as the factual source of truth.
2. If the context does not contain enough information, clearly say:
   "I don't have enough information in the provided personal knowledge base to answer that."
3. Do not invent personal facts, dates, experience, skills, contact details, or projects.
4. If the user asks to compare Uzair and Zain, compare only facts supported by the context.
5. Be concise but helpful.
6. You may use recent chat history only to understand follow-up references;
   do not treat unsupported history as new factual data.

RECENT CHAT HISTORY:
{self._history_to_text(history)}

RETRIEVED CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
""".strip()

        try:
            response = self.client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
            )

        except Exception:
            response = self.client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt,
            )

        answer = (response.text or "").strip()

        if not answer:
            answer = (
                "I don't have enough information in the provided "
                "personal knowledge base to answer that."
            )

        return answer, retrieved