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

ROOT_DATA_FILES = [
    BASE_DIR / "uzair_profile.txt",
    BASE_DIR / "zain_profile.txt",
]

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8")

    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(
            (page.extract_text() or "")
            for page in reader.pages
        )

    return ""


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:

    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        if end < len(text):
            boundary = max(
                chunk.rfind(". "),
                chunk.rfind("; "),
                chunk.rfind(" "),
            )

            if boundary > chunk_size * 0.6:
                end = start + boundary + 1
                chunk = text[start:end]

        chunks.append(chunk.strip())

        if end >= len(text):
            break

        start = max(0, end - overlap)

    return chunks


def get_data_files() -> list[Path]:
    files = []

    if DATA_DIR.exists():
        for path in sorted(DATA_DIR.iterdir()):
            if path.suffix.lower() in {".txt", ".pdf"}:
                files.append(path)

    if not files:
        for path in ROOT_DATA_FILES:
            if path.exists():
                files.append(path)

    return files


def build_index() -> None:
    VECTOR_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data_files = get_data_files()

    if not data_files:
        raise RuntimeError(
            "No .txt or .pdf data files found. "
            "Expected files inside data/ folder or "
            "uzair_profile.txt and zain_profile.txt in project root."
        )

    records = []

    for path in data_files:
        text = read_file(path)

        for i, chunk in enumerate(chunk_text(text)):
            records.append({
                "source": path.name,
                "chunk_id": i,
                "text": chunk,
            })

    if not records:
        raise RuntimeError(
            "Data files were found, but no readable text was extracted."
        )

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        [r["text"] for r in records],
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32",
    )

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    faiss.write_index(
        index,
        str(VECTOR_DIR / "index.faiss"),
    )

    (
        VECTOR_DIR / "chunks.json"
    ).write_text(
        json.dumps(
            records,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Indexed {len(records)} chunks "
        f"from {len(data_files)} data files."
    )

    print(
        f"Vector store saved to: {VECTOR_DIR}"
    )


if __name__ == "__main__":
    build_index()
