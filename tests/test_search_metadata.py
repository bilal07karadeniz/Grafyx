"""find_related_code response includes model + latency metadata."""
from __future__ import annotations

from grafyx.server import _state
from grafyx.server._tools_search import find_related_code


class _FakeSearcher:
    def search(self, q, max_results=10):
        return [{"name": "foo", "type": "function", "file": "f.py", "score": 0.9}]

    @property
    def encoder_meta(self):
        return {"model": "coderankembed", "version": "v1"}


class _FakeGraph:
    initialized = True


def _call_tool(name, *args, **kwargs):
    """Call an MCP tool whether decorated as a FunctionTool or kept plain."""
    fn = getattr(name, "fn", name)
    return fn(*args, **kwargs)


def _patch_state(monkeypatch):
    monkeypatch.setattr(_state, "_searcher", _FakeSearcher())
    monkeypatch.setattr(_state, "_graph", _FakeGraph())
    monkeypatch.setattr(_state, "_init_ready", True)
    monkeypatch.setattr(_state, "_init_error", None)


def test_response_includes_model(monkeypatch):
    _patch_state(monkeypatch)
    resp = _call_tool(find_related_code, "anything")
    assert "model" in resp, resp
    assert resp["model"]["model"] == "coderankembed"


def test_response_includes_latency(monkeypatch):
    _patch_state(monkeypatch)
    resp = _call_tool(find_related_code, "anything")
    assert "latency_ms" in resp
    assert isinstance(resp["latency_ms"], (int, float))
    assert resp["latency_ms"] >= 0
