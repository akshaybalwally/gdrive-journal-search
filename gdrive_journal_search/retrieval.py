"""Hybrid retrieval pipeline: vector search + BM25 + cross-encoder reranking."""

import re
from functools import lru_cache

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from .config import RERANK_MODEL, TOP_K_FINAL, TOP_K_RETRIEVE
from .embeddings import get_all_chunks, vector_search


# -- BM25 index ---------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """In-memory BM25 index built from all ChromaDB chunks."""

    def __init__(self) -> None:
        docs, metas, ids = get_all_chunks()
        self._docs = docs
        self._metas = metas
        self._ids = ids

        if docs:
            tokenized = [_tokenize(d) for d in docs]
            self._index = BM25Okapi(tokenized)
        else:
            self._index = None

    @property
    def empty(self) -> bool:
        return self._index is None

    def search(self, query: str, n_results: int) -> list[dict]:
        """Return top-n chunks by BM25 score."""
        if self.empty:
            return []

        scores = self._index.get_scores(_tokenize(query))

        # Get top-n indices sorted by descending score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]

        results = []
        for i in top_indices:
            if scores[i] <= 0:
                break
            meta = self._metas[i]
            results.append({
                "id": self._ids[i],
                "text": self._docs[i],
                "doc_name": meta.get("doc_name", ""),
                "created_at": meta.get("created_at", ""),
                "modified_at": meta.get("modified_at", ""),
                "bm25_score": float(scores[i]),
            })
        return results


# -- Reranker -----------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANK_MODEL)


def _rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Re-score chunks with a cross-encoder and return the top-k."""
    if not chunks:
        return []

    reranker = _get_reranker()
    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = reranker.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]


# -- Hybrid query pipeline ----------------------------------------------------

def _reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion.

    Each result is scored as sum(1 / (k + rank)) across all lists it
    appears in. Higher is better.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for results in result_lists:
        for rank, chunk in enumerate(results):
            chunk_id = chunk["id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            chunk_map[chunk_id] = chunk

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [chunk_map[cid] for cid in sorted_ids]


def query(
    text: str,
    bm25_index: BM25Index,
    top_k_retrieve: int = TOP_K_RETRIEVE,
    top_k_final: int = TOP_K_FINAL,
) -> list[dict]:
    """Full retrieval pipeline: vector + BM25 → RRF merge → cross-encoder rerank.

    Returns the top_k_final most relevant chunks.
    """
    # 1. Retrieve candidates from both methods
    vector_results = vector_search(text, n_results=top_k_retrieve)
    bm25_results = bm25_index.search(text, n_results=top_k_retrieve)

    # 2. Merge via Reciprocal Rank Fusion
    merged = _reciprocal_rank_fusion([vector_results, bm25_results])

    # 3. Rerank the merged candidates with cross-encoder
    reranked = _rerank(text, merged, top_k=top_k_final)

    return reranked
