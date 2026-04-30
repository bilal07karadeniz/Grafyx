"""find_related_code degrades gracefully when fastembed is missing or download fails."""
from __future__ import annotations

from grafyx.server import _state
from grafyx.server._tools_search import find_related_code


class _DegradedSearcher:
    """Simulates encoder-unavailable state."""

    def search(self, q, max_results=10):
        return [{"name": "fallback_match", "type": "function", "file": "x.py",
                 "score": 0.5, "low_confidence": True}]

    @property
    def encoder_meta(self):
        return {"model": "none", "version": ""}

    @property
    def degraded(self):
        return True


class _FakeGraph:
    initialized = True


def _call_tool(name, *args, **kwargs):
    fn = getattr(name, "fn", name)
    return fn(*args, **kwargs)


def _patch_state(monkeypatch):
    monkeypatch.setattr(_state, "_searcher", _DegradedSearcher())
    monkeypatch.setattr(_state, "_graph", _FakeGraph())
    monkeypatch.setattr(_state, "_init_ready", True)
    monkeypatch.setattr(_state, "_init_error", None)


def test_degraded_flag_in_response(monkeypatch):
    _patch_state(monkeypatch)
    resp = _call_tool(find_related_code, "query")
    assert resp.get("degraded") is True
    assert "action_hint" in resp
