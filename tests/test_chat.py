"""Tests for the chat module (context formatting, debug output)."""

import pytest

from gdrive_journal_search.chat import _format_context


class TestFormatContext:
    def _chunk(self, doc_name="test", created_at="2024-03-15T10:00:00Z", text="hello"):
        return {"doc_name": doc_name, "created_at": created_at, "text": text}

    def test_single_chunk(self):
        ctx = _format_context([self._chunk()])
        assert "test" in ctx
        assert "2024-03-15" in ctx
        assert "hello" in ctx

    def test_multiple_chunks_separated(self):
        chunks = [self._chunk(text="one"), self._chunk(text="two")]
        ctx = _format_context(chunks)
        assert "---" in ctx
        assert "one" in ctx
        assert "two" in ctx

    def test_empty_date(self):
        ctx = _format_context([self._chunk(created_at="")])
        assert "unknown date" in ctx

    def test_empty_list(self):
        ctx = _format_context([])
        assert ctx == ""

    def test_doc_name_in_output(self):
        ctx = _format_context([self._chunk(doc_name="My Journal 2024")])
        assert "My Journal 2024" in ctx
