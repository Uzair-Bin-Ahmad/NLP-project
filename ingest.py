from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DIR = BASE_DIR / "vectorstore"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return ""


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        # Try to finish on a sentence or whitespace boundary.
        if end < len(text):
            boundary = max(chunk.rfind(". "), chunk.rfind("; "), chunk.rfind(" "))
            if boundary > chunk_size * 0.6:
                end = start + boundary + 1
                chunk = text[start:end]

        chunks.append(chunk.strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def build_index() -> None:
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for path in sorted(DATA_DIR.iterdir()):
        if path.suffix.lower() not in {".txt", ".pdf"}:
            continue
        text = read_file(path)
        for i, chunk in enumerate(chunk_text(text)):
            records.append({
                "source": path.name,
                "chunk_id": i,
                "text": chunk,
            })

    if not records:
        raise RuntimeError("No .txt or .pdf documents found in the data folder.")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        [r["text"] for r in records],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(VECTOR_DIR / "index.faiss"))
    (VECTOR_DIR / "chunks.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Indexed {len(records)} chunks from {len(list(DATA_DIR.iterdir()))} data files.")
    print(f"Vector store saved to: {VECTOR_DIR}")


if __name__ == "__main__":
    build_index()
