"""Tests for the encoder model registry in EmbeddingSearcher."""
from __future__ import annotations

import pytest

from grafyx.search import _embeddings


def test_resolve_model_default(monkeypatch):
    monkeypatch.delenv("GRAFYX_ENCODER", raising=False)
    cfg = _embeddings.resolve_model_config()
    assert cfg["id"] in {"coderankembed", "jina-v2"}
    assert "model_name" in cfg
    assert "query_prefix" in cfg


def test_resolve_model_jina_v2(monkeypatch):
    monkeypatch.setenv("GRAFYX_ENCODER", "jina-v2")
    cfg = _embeddings.resolve_model_config()
    assert cfg["id"] == "jina-v2"
    assert cfg["model_name"] == "jinaai/jina-embeddings-v2-base-code"
    assert cfg["query_prefix"] == ""


def test_resolve_model_coderankembed(monkeypatch):
    monkeypatch.setenv("GRAFYX_ENCODER", "coderankembed")
    cfg = _embeddings.resolve_model_config()
    assert cfg["id"] == "coderankembed"
    assert cfg["query_prefix"].startswith("Represent this query")


def test_resolve_model_unknown_raises(monkeypatch):
    monkeypatch.setenv("GRAFYX_ENCODER", "totally-fake")
    with pytest.raises(ValueError, match="Unknown encoder"):
        _embeddings.resolve_model_config()


def test_searcher_uses_resolved_config(monkeypatch):
    """EmbeddingSearcher picks up the env-driven config without fastembed installed."""
    monkeypatch.setenv("GRAFYX_ENCODER", "jina-v2")

    class _FakeGraph:
        def iter_functions_with_source(self):
            return iter([])

        def get_all_classes(self, **_):
            return []

    s = _embeddings.EmbeddingSearcher(_FakeGraph(), cache_dir="/tmp/grafyx-test-cache")
    assert s._cfg["id"] == "jina-v2"
    assert s._model_name == "jinaai/jina-embeddings-v2-base-code"
    assert s._query_prefix == ""
