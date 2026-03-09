"""Tests for the retrieval module (BM25, reranking, hybrid pipeline)."""

import pytest

from gdrive_journal_search.retrieval import _tokenize, _reciprocal_rank_fusion, BM25Index


class TestTokenize:
    def test_basic(self):
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_punctuation(self):
        assert _tokenize("it's a test!") == ["it", "s", "a", "test"]

    def test_empty(self):
        assert _tokenize("") == []

    def test_numbers(self):
        assert _tokenize("March 2024") == ["march", "2024"]

    def test_lowercased(self):
        tokens = _tokenize("BIG small MiXeD")
        assert tokens == ["big", "small", "mixed"]


class TestReciprocalRankFusion:
    def _chunk(self, id: str, text: str = "") -> dict:
        return {"id": id, "text": text, "doc_name": "", "created_at": "", "modified_at": ""}

    def test_empty_lists(self):
        assert _reciprocal_rank_fusion([[], []]) == []

    def test_single_list(self):
        chunks = [self._chunk("a"), self._chunk("b")]
        result = _reciprocal_rank_fusion([chunks])
        assert [r["id"] for r in result] == ["a", "b"]

    def test_merge_two_lists(self):
        list1 = [self._chunk("a"), self._chunk("b"), self._chunk("c")]
        list2 = [self._chunk("b"), self._chunk("d"), self._chunk("a")]
        result = _reciprocal_rank_fusion([list1, list2])
        ids = [r["id"] for r in result]
        # "a" and "b" appear in both lists, so they should rank higher
        assert "a" in ids[:3]
        assert "b" in ids[:3]

    def test_deduplication(self):
        list1 = [self._chunk("a"), self._chunk("b")]
        list2 = [self._chunk("a"), self._chunk("b")]
        result = _reciprocal_rank_fusion([list1, list2])
        ids = [r["id"] for r in result]
        assert len(ids) == 2  # no duplicates

    def test_preserves_chunk_data(self):
        chunk = self._chunk("x", text="hello")
        result = _reciprocal_rank_fusion([[chunk]])
        assert result[0]["text"] == "hello"


class TestBM25Index:
    """BM25Index requires ChromaDB data. Test with an empty index."""

    def test_empty_index(self, tmp_path, monkeypatch):
        """An empty collection should produce an empty BM25 index."""
        monkeypatch.setattr(
            "gdrive_journal_search.retrieval.get_all_chunks",
            lambda: ([], [], []),
        )
        idx = BM25Index()
        assert idx.empty
        assert idx.search("test query", n_results=5) == []

    def test_search_with_data(self, monkeypatch):
        """BM25 should find documents containing the query term."""
        # Use enough docs that IDF doesn't go negative for the target term
        docs = [
            "alpha bravo charlie delta echo foxtrot",
            "golf hotel india juliet kilo lima",
            "mike november oscar papa quebec romeo",
            "sierra tango uniform victor whiskey xray",
            "alpha zulu yankee washington virginia utah",
        ]
        metas = [{"doc_name": f"d{i}", "created_at": "", "modified_at": ""} for i in range(5)]
        ids = [f"chunk{i}" for i in range(5)]

        monkeypatch.setattr(
            "gdrive_journal_search.retrieval.get_all_chunks",
            lambda: (docs, metas, ids),
        )
        idx = BM25Index()
        assert not idx.empty

        results = idx.search("alpha", n_results=3)
        result_ids = [r["id"] for r in results]
        # "alpha" appears in chunk0 and chunk4
        assert "chunk0" in result_ids
        assert "chunk4" in result_ids

    def test_search_respects_n_results(self, monkeypatch):
        docs = [
            "the cat sat on the mat",
            "the dog ran in the park",
            "the cat and the dog played together",
        ]
        metas = [{"doc_name": f"d{i}", "created_at": "", "modified_at": ""} for i in range(3)]
        ids = [f"id{i}" for i in range(3)]

        monkeypatch.setattr(
            "gdrive_journal_search.retrieval.get_all_chunks",
            lambda: (docs, metas, ids),
        )
        idx = BM25Index()
        results = idx.search("cat", n_results=1)
        assert len(results) == 1
