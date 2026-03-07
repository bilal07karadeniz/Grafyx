"""Tests for grafyx.utils module."""

import json
from pathlib import Path

from grafyx.utils import (
    EXTENSION_TO_LANGUAGE,
    LANGUAGE_EXTENSIONS,
    DEFAULT_IGNORE_PATTERNS,
    detect_languages,
    truncate_response,
    build_directory_tree,
    safe_get_attr,
    split_tokens,
)


class TestLanguageMappings:
    def test_python_extensions(self):
        assert ".py" in EXTENSION_TO_LANGUAGE
        assert EXTENSION_TO_LANGUAGE[".py"] == "python"

    def test_typescript_extensions(self):
        assert ".ts" in EXTENSION_TO_LANGUAGE
        assert EXTENSION_TO_LANGUAGE[".ts"] == "typescript"

    def test_javascript_extensions(self):
        assert ".js" in EXTENSION_TO_LANGUAGE
        assert EXTENSION_TO_LANGUAGE[".js"] == "javascript"

    def test_all_extensions_mapped(self):
        for lang, exts in LANGUAGE_EXTENSIONS.items():
            for ext in exts:
                assert ext in EXTENSION_TO_LANGUAGE
                assert EXTENSION_TO_LANGUAGE[ext] == lang


class TestDetectLanguages:
    def test_detect_python_project(self, python_project_path):
        langs = detect_languages(str(python_project_path))
        assert "python" in langs

    def test_detect_empty_dir(self, tmp_path):
        langs = detect_languages(str(tmp_path))
        assert langs == []

    def test_detect_single_file_not_enough(self, tmp_path):
        (tmp_path / "single.py").write_text("x = 1")
        langs = detect_languages(str(tmp_path))
        assert "python" not in langs  # needs >= 2 files

    def test_detect_two_files_enough(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1")
        (tmp_path / "b.py").write_text("y = 2")
        langs = detect_languages(str(tmp_path))
        assert "python" in langs

    def test_ignores_node_modules(self, tmp_path):
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "a.js").write_text("x")
        (nm / "b.js").write_text("y")
        langs = detect_languages(str(tmp_path))
        assert "javascript" not in langs


class TestTruncateResponse:
    def test_small_response_unchanged(self):
        data = {"key": "value"}
        result = truncate_response(data, max_chars=1000)
        assert result == data

    def test_large_dict_truncated(self):
        data = {"items": list(range(1000))}
        result = truncate_response(data, max_chars=100)
        assert result.get("_truncated") is True

    def test_large_string_truncated(self):
        data = "x" * 1000
        result = truncate_response(data, max_chars=100)
        assert "[... truncated]" in result
        assert len(result) <= 120

    def test_large_list_truncated(self):
        data = list(range(1000))
        result = truncate_response(data, max_chars=100)
        assert result["_truncated"] is True
        assert result["_total"] == 1000


class TestSplitTokens:
    def test_snake_case(self):
        assert split_tokens("get_user_data") == ["get", "user", "data"]

    def test_camel_case(self):
        assert split_tokens("getUserData") == ["get", "user", "data"]

    def test_pascal_case(self):
        assert split_tokens("UserData") == ["user", "data"]

    def test_acronym(self):
        tokens = split_tokens("HTTPResponse")
        assert "http" in tokens
        assert "response" in tokens

    def test_single_word(self):
        assert split_tokens("hello") == ["hello"]

    def test_empty_string(self):
        assert split_tokens("") == []

    def test_path_splitting(self):
        tokens = split_tokens("src/utils/helpers")
        assert "src" in tokens
        assert "utils" in tokens
        assert "helpers" in tokens


class TestSafeGetAttr:
    def test_existing_attr(self):
        class Obj:
            name = "test"
        assert safe_get_attr(Obj(), "name") == "test"

    def test_missing_attr(self):
        class Obj:
            pass
        assert safe_get_attr(Obj(), "name", "default") == "default"

    def test_none_attr(self):
        class Obj:
            name = None
        assert safe_get_attr(Obj(), "name", "default") == "default"


class TestBuildDirectoryTree:
    def test_simple_tree(self, tmp_path):
        files = ["src/main.py", "src/utils.py", "README.md"]
        tree = build_directory_tree(files, str(tmp_path))
        assert "src" in tree
        assert "main.py" in tree

    def test_empty_files(self, tmp_path):
        tree = build_directory_tree([], str(tmp_path))
        assert tree == "(empty project)"
