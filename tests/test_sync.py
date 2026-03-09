"""Tests for the sync module (state management, atomic writes)."""

import json

import pytest

from gdrive_journal_search.sync import _load_state, _save_state, _last_sync_time
from gdrive_journal_search.config import SYNC_STATE_FILE


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    """Point SYNC_STATE_FILE at a temp directory for every test."""
    fake_path = tmp_path / "sync_state.json"
    monkeypatch.setattr("gdrive_journal_search.sync.SYNC_STATE_FILE", fake_path)
    monkeypatch.setattr("gdrive_journal_search.config.SYNC_STATE_FILE", fake_path)
    yield fake_path


class TestStateIO:
    def test_load_missing_file(self, _isolate_state):
        assert _load_state() == {}

    def test_save_and_load_roundtrip(self, _isolate_state):
        state = {"last_sync": "2024-03-15T10:00:00+00:00", "foo": "bar"}
        _save_state(state)
        loaded = _load_state()
        assert loaded == state

    def test_save_is_atomic(self, _isolate_state):
        """After save, no .tmp file should remain."""
        _save_state({"x": 1})
        tmp_file = _isolate_state.with_suffix(".tmp")
        assert not tmp_file.exists()
        assert _isolate_state.exists()


class TestLastSyncTime:
    def test_returns_none_when_missing(self):
        assert _last_sync_time({}) is None

    def test_parses_iso_timestamp(self):
        from datetime import datetime, timezone
        state = {"last_sync": "2024-03-15T10:00:00+00:00"}
        result = _last_sync_time(state)
        assert result.year == 2024
        assert result.month == 3
        assert result.day == 15
        assert result.tzinfo is not None
