"""Tests to verify hardcoded truncation caps have been removed from server tools.

These tests ensure that:
1. MAX_HINTS is 5 (raised from 3)
2. depended_on_by no longer has a [:20] cap
3. imported_by no longer has a [:30] cap
4. cross_file_usages no longer has a [:20] cap
"""

import inspect


def test_max_hints_is_five():
    """MAX_HINTS should be 5, not 3."""
    from grafyx.server._hints import MAX_HINTS

    assert MAX_HINTS == 5, f"MAX_HINTS should be 5, got {MAX_HINTS}"


def test_dep_graph_no_20_cap():
    """The [:20] cap on depended_on_by should be removed from _tools_graph.py."""
    from grafyx.server import _tools_graph

    source = inspect.getsource(_tools_graph)
    assert "depended_on_by" in source, "depended_on_by not found in source"
    # The old code had sorted(by_file.items(), ...)[:20] in the dep graph builder.
    # After removal, there should be no [:20] slice in _tools_graph at all.
    assert "[:20]" not in source, (
        "[:20] cap still present in _tools_graph.py"
    )


def test_imported_by_no_30_cap():
    """The [:30] cap on imported_by should be removed from _tools_introspection.py."""
    from grafyx.server import _tools_introspection

    source = inspect.getsource(_tools_introspection)
    # Find all occurrences of 'imported_by' and check none have [:30]
    assert "imported_by" in source, "imported_by not found in source"
    # Check that nowhere in the source do we have importers[:30]
    assert "importers[:30]" not in source, (
        "[:30] cap still present on importers in _tools_introspection.py"
    )


def test_cross_file_usages_no_20_cap():
    """The [:20] cap on cross_file_usages should be removed from _tools_introspection.py."""
    from grafyx.server import _tools_introspection

    source = inspect.getsource(_tools_introspection)
    assert "cross_file_usages" in source, "cross_file_usages not found in source"
    # Check that .items())[:20] pattern is gone
    assert ".items())[:20]" not in source, (
        "[:20] cap still present on cross_file_usages in _tools_introspection.py"
    )
