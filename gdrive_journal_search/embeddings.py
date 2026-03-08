"""Chunking, embedding, and vector store management via ChromaDB."""

from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from .config import CHROMA_COLLECTION, CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL


def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return client.get_or_create_collection(name=CHROMA_COLLECTION, embedding_function=ef)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-level chunks."""
    if not text.strip():
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


def upsert_doc(doc: dict[str, Any]) -> int:
    """
    Chunk a doc and upsert all chunks into ChromaDB.
    Returns the number of chunks written.

    doc keys: id, name, created_at, modified_at, text
    """
    collection = _get_collection()
    chunks = chunk_text(doc["text"])
    if not chunks:
        return 0

    ids = [f"{doc['id']}__chunk{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "doc_id": doc["id"],
            "doc_name": doc["name"],
            "created_at": doc.get("created_at") or "",
            "modified_at": doc.get("modified_at") or "",
            "chunk_index": i,
            "total_chunks": len(chunks),
        }
        for i in range(len(chunks))
    ]

    collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
    return len(chunks)


def delete_doc(doc_id: str) -> None:
    """Remove all chunks for a given doc_id from the collection."""
    collection = _get_collection()
    collection.delete(where={"doc_id": doc_id})


def query(text: str, n_results: int) -> list[dict[str, Any]]:
    """
    Retrieve the top-n most relevant chunks for a query string.
    Returns list of dicts with keys: text, doc_name, created_at, modified_at, distance.
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []

    results = collection.query(query_texts=[text], n_results=min(n_results, collection.count()))

    docs_out = []
    for doc_text, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        docs_out.append(
            {
                "text": doc_text,
                "doc_name": meta.get("doc_name", ""),
                "created_at": meta.get("created_at", ""),
                "modified_at": meta.get("modified_at", ""),
                "distance": distance,
            }
        )
    return docs_out


def collection_count() -> int:
    return _get_collection().count()
