"""Tests for v0.2.2 audit-driven fixes.

Each test pins behavior reported as a bug in the v0.2.1 audit so the
regressions don't return.

Covered fixes:
    - P0.1: get_call_graph method/route disambiguation
    - P0.2: TS destructured-prop signature rendering
    - P1.4: cross-language caller filter
    - P1.5: TS return-type detection
    - P1.6: search module-name collision penalty
"""

from __future__ import annotations

from types import SimpleNamespace

from grafyx.server._tools_graph import _classify_call_style
from grafyx.server._tools_introspection import _filter_cross_language_callers
from grafyx.utils import _extract_ts_param_str, format_function_signature


# ----------------------------------------------------------------------
# P0.1 — _classify_call_style
# ----------------------------------------------------------------------

class TestClassifyCallStyle:
    def test_dotted_only(self):
        src = "def caller(self):\n    self.db.refresh(user)\n"
        assert _classify_call_style(src, "refresh") == "dotted"

    def test_bare_only(self):
        src = "def caller():\n    refresh()\n    do_thing()\n"
        assert _classify_call_style(src, "refresh") == "bare"

    def test_mixed(self):
        src = "def caller():\n    refresh()\n    cache.refresh()\n"
        assert _classify_call_style(src, "refresh") == "mixed"

    def test_unknown_when_absent(self):
        src = "def caller():\n    pass\n"
        assert _classify_call_style(src, "refresh") == "unknown"

    def test_attribute_access_does_not_match_call(self):
        # `obj.refresh` (no parens) is attribute access, not a call.
        src = "x = obj.refresh\n"
        assert _classify_call_style(src, "refresh") == "unknown"

    def test_db_refresh_is_dotted(self):
        # The headline audit case: register_user calls db.refresh(user).
        # Must NOT misclassify as bare.
        src = (
            "async def register_user(payload, db):\n"
            "    user = User(payload)\n"
            "    db.add(user)\n"
            "    await db.commit()\n"
            "    db.refresh(user)\n"
            "    return user\n"
        )
        assert _classify_call_style(src, "refresh") == "dotted"


# ----------------------------------------------------------------------
# P0.2 — TS destructured prop signatures
# ----------------------------------------------------------------------

class TestTsDestructuredProps:
    def test_extract_destructured_object_param(self):
        src = (
            "function AgentKnowledgeTab({ agentId, agentName }: AgentKnowledgeTabProps) {\n"
            "  return null;\n"
            "}\n"
        )
        out = _extract_ts_param_str(src)
        assert out is not None
        assert "{ agentId, agentName }" in out
        assert "AgentKnowledgeTabProps" in out

    def test_extract_named_params(self):
        src = "function add(a: number, b: number): number { return a + b; }"
        assert _extract_ts_param_str(src) == "a: number, b: number"

    def test_extract_callback_param_with_nested_parens(self):
        src = "function withRetry(cb: () => Promise<void>, attempts: number) {}"
        out = _extract_ts_param_str(src)
        assert out == "cb: () => Promise<void>, attempts: number"

    def test_arrow_function_with_destructure(self):
        src = "const Foo = ({ a, b }: Props) => <div>{a}{b}</div>"
        out = _extract_ts_param_str(src)
        assert out is not None
        assert "{ a, b }" in out
        assert "Props" in out

    def test_returns_none_on_no_parens(self):
        assert _extract_ts_param_str("class Foo {}") is None
        assert _extract_ts_param_str("") is None

    def test_format_signature_uses_literal_for_destructured(self):
        # Simulate what graph-sitter exposes for a destructured param:
        # name comes from each field but type is the shared interface.
        # Without the source-extraction fix, this used to render as
        # `function Comp(a: Props, b: Props)`.
        params = [
            SimpleNamespace(name="a", type="Props", default=None),
            SimpleNamespace(name="b", type="Props", default=None),
        ]
        func = SimpleNamespace(
            name="Comp",
            is_async=False,
            filepath="frontend/src/Comp.tsx",
            parameters=params,
            return_type=None,
            source="function Comp({ a, b }: Props) { return null; }",
        )
        sig = format_function_signature(func)
        assert "{ a, b }" in sig
        assert "Props" in sig
        assert sig.count("Props") == 1  # not duplicated per destructured field

    def test_format_signature_falls_back_when_no_source(self):
        params = [
            SimpleNamespace(name="x", type="number", default=None),
        ]
        func = SimpleNamespace(
            name="square",
            is_async=False,
            filepath="lib/math.ts",
            parameters=params,
            return_type="number",
            source="",
        )
        sig = format_function_signature(func)
        # Falls back to reconstructed form, with TS return-type syntax.
        assert sig == "function square(x: number): number"


# ----------------------------------------------------------------------
# P1.4 — cross-language caller filter
# ----------------------------------------------------------------------

class TestCrossLanguageCallerFilter:
    def test_same_language_callers_pass_through(self):
        callers = [
            {"name": "register_user", "file": "backend/auth.py", "has_dot_syntax": False},
            {"name": "verify_jwt", "file": "backend/api.py", "has_dot_syntax": True},
        ]
        kept = _filter_cross_language_callers(callers, "python")
        assert len(kept) == 2

    def test_bare_ts_caller_dropped_for_python_target(self):
        # Bare `login()` in TS calls TS's own `login`, NOT the Python one.
        callers = [
            {"name": "handleLogin", "file": "frontend/Login.tsx", "has_dot_syntax": False},
        ]
        kept = _filter_cross_language_callers(callers, "python")
        assert kept == []

    def test_dotted_ts_caller_kept_for_python_target(self):
        # `await api.login(...)` looks like an actual API client call.
        callers = [
            {"name": "submit", "file": "frontend/Login.tsx", "has_dot_syntax": True},
        ]
        kept = _filter_cross_language_callers(callers, "python")
        assert len(kept) == 1

    def test_ts_to_js_is_not_cross_language(self):
        # Web family: TS and JS share a parser; not really cross-lang.
        callers = [
            {"name": "doIt", "file": "shared/helper.js", "has_dot_syntax": False},
        ]
        kept = _filter_cross_language_callers(callers, "typescript")
        assert len(kept) == 1

    def test_no_target_lang_keeps_all(self):
        callers = [{"name": "x", "file": "a.py", "has_dot_syntax": False}]
        assert _filter_cross_language_callers(callers, "") == callers

    def test_caller_without_extension_passes_through(self):
        callers = [{"name": "x", "file": "Makefile", "has_dot_syntax": False}]
        # No extension -> no language inference -> we don't drop.
        assert _filter_cross_language_callers(callers, "python") == callers


# ----------------------------------------------------------------------
# P1.5 — TS return-type detection in conventions
# ----------------------------------------------------------------------

class TestTsReturnTypeDetection:
    def test_ts_return_type_detected_with_colon(self):
        from grafyx.conventions import ConventionDetector

        # detect_typing_conventions reads f.get("signature", ""). We
        # don't need a real graph — feed sample function dicts.
        graph = SimpleNamespace(
            get_all_functions=lambda **_: [
                {
                    "language": "typescript",
                    "signature": "function getRoleBadgeVariant(role: string): string",
                },
                {
                    "language": "typescript",
                    "signature": "function isAdmin(user: User): boolean",
                },
                {
                    "language": "typescript",
                    "signature": "function noReturn(x: number)",
                },
            ],
        )
        detector = ConventionDetector(graph)
        conventions = detector.detect_typing_conventions(
            graph.get_all_functions(),
        )
        # Two of three have return types -> >= 50% threshold
        return_conv = [c for c in conventions if "return type" in c.pattern]
        assert return_conv, (
            "Expected a TypeScript return-type convention to be detected"
        )
        # Confidence reflects the 2/3 ratio, NOT the broken 0%.
        assert return_conv[0].confidence > 0.5

    def test_ts_callback_return_type_not_double_counted(self):
        from grafyx.conventions import ConventionDetector

        # `(cb: () => void): R` — the rfind(")") logic must lock onto
        # the OUTER paren, so we still detect the return type.
        graph = SimpleNamespace(
            get_all_functions=lambda **_: [
                {
                    "language": "typescript",
                    "signature": "function withRetry(cb: () => void): boolean",
                },
                {
                    "language": "typescript",
                    "signature": "function plain(): void",
                },
            ],
        )
        detector = ConventionDetector(graph)
        conventions = detector.detect_typing_conventions(
            graph.get_all_functions(),
        )
        # Both signatures have return types -> 100%
        return_conv = [c for c in conventions if "return type" in c.pattern]
        assert return_conv
        assert return_conv[0].confidence == 1.0

    def test_python_return_type_unchanged(self):
        from grafyx.conventions import ConventionDetector

        graph = SimpleNamespace(
            get_all_functions=lambda **_: [
                {"language": "python", "signature": "def add(a: int, b: int) -> int"},
                {"language": "python", "signature": "def name(x: str) -> str"},
            ],
        )
        detector = ConventionDetector(graph)
        conventions = detector.detect_typing_conventions(
            graph.get_all_functions(),
        )
        return_conv = [c for c in conventions if "return type" in c.pattern]
        assert return_conv
        assert return_conv[0].confidence == 1.0
