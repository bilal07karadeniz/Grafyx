"""Tests for grafyx.conventions module."""

from unittest.mock import MagicMock
from grafyx.conventions import ConventionDetector


def _make_mock_graph():
    graph = MagicMock()
    graph.get_all_functions.return_value = [
        {"name": "process_data", "signature": "def process_data(data: list) -> dict", "file": "main.py", "language": "python", "line": 10, "docstring": ""},
        {"name": "validate_input", "signature": "def validate_input(data: list) -> list", "file": "main.py", "language": "python", "line": 20, "docstring": ""},
        {"name": "get_user", "signature": "def get_user(user_id: int) -> dict", "file": "users.py", "language": "python", "line": 5, "docstring": ""},
        {"name": "format_output", "signature": "def format_output(data)", "file": "utils.py", "language": "python", "line": 8, "docstring": ""},
    ]
    graph.get_all_classes.return_value = [
        {"name": "DataProcessor", "base_classes": [], "file": "main.py", "language": "python", "line": 30, "docstring": "", "method_count": 3},
        {"name": "UserManager", "base_classes": [], "file": "users.py", "language": "python", "line": 15, "docstring": "", "method_count": 5},
    ]
    graph.get_all_files.return_value = [
        {"path": "main.py", "function_count": 2, "class_count": 1, "import_count": 3, "language": "python"},
        {"path": "utils.py", "function_count": 3, "class_count": 0, "import_count": 1, "language": "python"},
        {"path": "models.py", "function_count": 0, "class_count": 2, "import_count": 2, "language": "python"},
    ]
    return graph


class TestConventionDetector:
    def test_detect_all_returns_list(self):
        detector = ConventionDetector(_make_mock_graph())
        result = detector.detect_all()
        assert isinstance(result, list)
        for item in result:
            assert "category" in item
            assert "pattern" in item
            assert "confidence" in item
            assert "examples" in item

    def test_detect_snake_case_functions(self):
        detector = ConventionDetector(_make_mock_graph())
        conventions = detector.detect_naming_conventions()
        naming = [c for c in conventions if c.category == "naming" and "function" in c.pattern.lower()]
        assert len(naming) > 0
        assert "snake_case" in naming[0].pattern

    def test_detect_pascal_case_classes(self):
        detector = ConventionDetector(_make_mock_graph())
        conventions = detector.detect_naming_conventions()
        naming = [c for c in conventions if c.category == "naming" and "class" in c.pattern.lower()]
        assert len(naming) > 0
        assert "PascalCase" in naming[0].pattern

    def test_detect_typing_conventions(self):
        detector = ConventionDetector(_make_mock_graph())
        conventions = detector.detect_typing_conventions()
        # 3 out of 4 functions have return type -> 75%
        typing_convs = [c for c in conventions if c.category == "typing"]
        assert len(typing_convs) > 0

    def test_detect_structure(self):
        detector = ConventionDetector(_make_mock_graph())
        conventions = detector.detect_structure_conventions()
        structure = [c for c in conventions if c.category == "structure"]
        assert len(structure) > 0

    def test_confidence_range(self):
        detector = ConventionDetector(_make_mock_graph())
        result = detector.detect_all()
        for item in result:
            assert 0 <= item["confidence"] <= 1.0
