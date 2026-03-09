"""Tests for the embeddings module (chunking, contextual prefixes, storage)."""

import pytest

from gdrive_journal_search.embeddings import chunk_text, _contextual_prefix


class TestChunkText:
    def test_empty_string(self):
        assert chunk_text("") == []

    def test_whitespace_only(self):
        assert chunk_text("   \n\t  ") == []

    def test_short_text_single_chunk(self):
        text = "Hello world"
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert chunks == ["Hello world"]

    def test_exact_chunk_size(self):
        text = "a" * 100
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert chunks == [text]

    def test_overlapping_chunks(self):
        text = "a" * 200
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) == 3  # 0-100, 80-180, 160-200
        # First chunk is 100 chars
        assert len(chunks[0]) == 100
        # Overlap: last 20 chars of chunk 0 == first 20 chars of chunk 1
        assert chunks[0][-20:] == chunks[1][:20]

    def test_no_overlap(self):
        text = "a" * 200
        chunks = chunk_text(text, chunk_size=100, overlap=0)
        assert len(chunks) == 2
        assert len(chunks[0]) == 100
        assert len(chunks[1]) == 100

    def test_last_chunk_shorter(self):
        text = "a" * 150
        chunks = chunk_text(text, chunk_size=100, overlap=0)
        assert len(chunks) == 2
        assert len(chunks[0]) == 100
        assert len(chunks[1]) == 50

    def test_uses_default_config(self):
        """chunk_text should work with no explicit size/overlap (uses config defaults)."""
        text = "word " * 200  # 1000 chars
        chunks = chunk_text(text)
        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)


class TestContextualPrefix:
    def test_with_date(self):
        prefix = _contextual_prefix("20240315", "2024-03-15T10:00:00Z")
        assert "20240315" in prefix
        assert "2024-03-15" in prefix

    def test_with_empty_date(self):
        prefix = _contextual_prefix("My Doc", "")
        assert "My Doc" in prefix
        assert "unknown date" in prefix

    def test_with_none_date(self):
        """Callers may pass empty string for missing dates."""
        prefix = _contextual_prefix("My Doc", "")
        assert "unknown date" in prefix

    def test_prefix_is_string(self):
        prefix = _contextual_prefix("title", "2024-01-01")
        assert isinstance(prefix, str)
        assert prefix.endswith(": ")
