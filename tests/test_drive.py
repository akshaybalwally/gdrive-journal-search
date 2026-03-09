"""Tests for the drive module (credential checking, doc dict construction)."""

import pytest

from gdrive_journal_search.drive import _fetch_one


class TestFetchOne:
    def test_returns_none_on_403(self, monkeypatch):
        """Should return None (not raise) for permission errors."""
        from googleapiclient.errors import HttpError
        from unittest.mock import MagicMock

        def fake_export(service, file_id):
            resp = MagicMock()
            resp.status = 403
            raise HttpError(resp, b"forbidden")

        monkeypatch.setattr("gdrive_journal_search.drive._build_service", lambda: None)
        monkeypatch.setattr("gdrive_journal_search.drive._export_doc_text", fake_export)

        meta = {"id": "abc", "name": "Test Doc", "createdTime": "2024-01-01", "modifiedTime": "2024-01-02"}
        result = _fetch_one(meta)
        assert result is None

    def test_returns_doc_dict_on_success(self, monkeypatch):
        monkeypatch.setattr("gdrive_journal_search.drive._build_service", lambda: None)
        monkeypatch.setattr("gdrive_journal_search.drive._export_doc_text", lambda svc, fid: "Hello world")

        meta = {"id": "abc", "name": "Test Doc", "createdTime": "2024-01-01", "modifiedTime": "2024-01-02"}
        result = _fetch_one(meta)
        assert result is not None
        assert result["id"] == "abc"
        assert result["name"] == "Test Doc"
        assert result["text"] == "Hello world"
        assert result["created_at"] == "2024-01-01"
        assert result["modified_at"] == "2024-01-02"
