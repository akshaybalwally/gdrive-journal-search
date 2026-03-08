"""Chunking, embedding, and vector store management via ChromaDB."""

from functools import lru_cache
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from .config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
)


@lru_cache(maxsize=1)
def _get_collection() -> chromadb.Collection:
    """Return the ChromaDB collection, creating it on first call."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return client.get_or_create_collection(name=CHROMA_COLLECTION, embedding_function=ef)


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping character-level chunks."""
    if not text.strip():
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


def upsert_doc(doc: dict[str, Any]) -> int:
    """Chunk a document and upsert all chunks into ChromaDB.

    Returns the number of chunks written.
    Expected doc keys: id, name, created_at, modified_at, text.
    """
    chunks = chunk_text(doc["text"])
    if not chunks:
        return 0

    collection = _get_collection()
    collection.upsert(
        ids=[f"{doc['id']}__chunk{i}" for i in range(len(chunks))],
        documents=chunks,
        metadatas=[
            {
                "doc_id": doc["id"],
                "doc_name": doc["name"],
                "created_at": doc.get("created_at") or "",
                "modified_at": doc.get("modified_at") or "",
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            for i in range(len(chunks))
        ],
    )
    return len(chunks)


def delete_doc(doc_id: str) -> None:
    """Remove all chunks belonging to *doc_id*."""
    _get_collection().delete(where={"doc_id": doc_id})


def query(text: str, n_results: int) -> list[dict[str, Any]]:
    """Return the top-n most relevant chunks for a natural-language query."""
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return []

    results = collection.query(
        query_texts=[text],
        n_results=min(n_results, count),
    )

    return [
        {
            "text": doc_text,
            "doc_name": meta.get("doc_name", ""),
            "created_at": meta.get("created_at", ""),
            "modified_at": meta.get("modified_at", ""),
            "distance": distance,
        }
        for doc_text, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def collection_count() -> int:
    """Total number of chunks currently stored."""
    return _get_collection().count()
