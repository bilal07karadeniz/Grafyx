"""Tests for __init__.py re-export resolution in import index.

When a consumer does `from package import X` and package/__init__.py
re-exports X from a submodule, the import index should link the
consumer to BOTH the __init__.py AND the actual source module.
"""

import re
from unittest.mock import MagicMock

from grafyx.graph import CodebaseGraph


class TestInitReExportResolution:
    """Verify that _resolve_init_reexports adds actual source file links."""

    def _make_graph(self):
        graph = MagicMock(spec=CodebaseGraph)
        graph._project_path = "/project"
        graph.original_path = "/project"
        graph._lock = MagicMock()
        graph._external_packages = set()
        graph.translate_path = lambda p: str(p) if p else ""
        graph._build_import_index = CodebaseGraph._build_import_index.__get__(graph)
        graph._extract_module_from_import = CodebaseGraph._extract_module_from_import
        graph._extract_symbol_names_from_import = CodebaseGraph._extract_symbol_names_from_import
        graph._resolve_init_reexports = CodebaseGraph._resolve_init_reexports.__get__(graph)
        graph._is_ignored_file_path = lambda p: False
        return graph

    # ------------------------------------------------------------------ #
    # Pattern 1: Explicit relative re-exports (from .module import X)
    # ------------------------------------------------------------------ #

    def test_explicit_reexport_links_consumer_to_source(self):
        """from .impl import ServiceClass in __init__.py should link
        consumers that import ServiceClass from the package to impl.py."""
        graph = self._make_graph()

        # __init__.py re-exports from .impl
        init_file = MagicMock()
        init_file.filepath = "/project/package/__init__.py"
        init_file.path = "/project/package/__init__.py"
        init_imp = MagicMock()
        init_imp.source = "from .impl import ServiceClass"
        init_file.imports = [init_imp]
        init_file.functions = []

        impl_file = MagicMock()
        impl_file.filepath = "/project/package/impl.py"
        impl_file.path = "/project/package/impl.py"
        impl_file.imports = []
        impl_file.functions = []

        # Consumer imports ServiceClass from the package
        consumer = MagicMock()
        consumer.filepath = "/project/app/main.py"
        consumer.path = "/project/app/main.py"
        consumer_imp = MagicMock()
        consumer_imp.source = "from package import ServiceClass"
        consumer.imports = [consumer_imp]
        consumer.functions = []

        codebase = MagicMock()
        codebase.files = [init_file, impl_file, consumer]
        graph._codebases = {"python": codebase}

        graph._build_import_index()

        # Consumer should STILL import __init__.py (not removed)
        init_importers = graph._import_index.get("/project/package/__init__.py", [])
        assert "/project/app/main.py" in init_importers

        # Consumer should ALSO be linked to impl.py (the actual source)
        impl_importers = graph._import_index.get("/project/package/impl.py", [])
        assert "/project/app/main.py" in impl_importers, (
            f"Consumer should be linked to impl.py, got importers: {impl_importers}"
        )

        # Forward index should also include impl.py for the consumer
        consumer_fwd = graph._forward_import_index.get("/project/app/main.py", [])
        assert "/project/package/impl.py" in consumer_fwd

    def test_explicit_reexport_multiple_symbols_from_same_module(self):
        """Multiple symbols re-exported from the same submodule."""
        graph = self._make_graph()

        init_file = MagicMock()
        init_file.filepath = "/project/auth/__init__.py"
        init_file.path = "/project/auth/__init__.py"
        init_imp = MagicMock()
        init_imp.source = "from .core import authenticate, verify_token, create_session"
        init_file.imports = [init_imp]
        init_file.functions = []

        core_file = MagicMock()
        core_file.filepath = "/project/auth/core.py"
        core_file.path = "/project/auth/core.py"
        core_file.imports = []
        core_file.functions = []

        consumer = MagicMock()
        consumer.filepath = "/project/api/routes.py"
        consumer.path = "/project/api/routes.py"
        consumer_imp = MagicMock()
        consumer_imp.source = "from auth import authenticate"
        consumer.imports = [consumer_imp]
        consumer.functions = []

        codebase = MagicMock()
        codebase.files = [init_file, core_file, consumer]
        graph._codebases = {"python": codebase}

        graph._build_import_index()

        # Consumer imports "authenticate" which is re-exported from core.py
        core_importers = graph._import_index.get("/project/auth/core.py", [])
        assert "/project/api/routes.py" in core_importers

    def test_explicit_reexport_from_multiple_submodules(self):
        """Re-exports from different submodules resolve to correct files."""
        graph = self._make_graph()

        init_file = MagicMock()
        init_file.filepath = "/project/services/__init__.py"
        init_file.path = "/project/services/__init__.py"
        imp1 = MagicMock()
        imp1.source = "from .auth import login"
        imp2 = MagicMock()
        imp2.source = "from .users import get_user"
        init_file.imports = [imp1, imp2]
        init_file.functions = []

        auth_file = MagicMock()
        auth_file.filepath = "/project/services/auth.py"
        auth_file.path = "/project/services/auth.py"
        auth_file.imports = []
        auth_file.functions = []

        users_file = MagicMock()
        users_file.filepath = "/project/services/users.py"
        users_file.path = "/project/services/users.py"
        users_file.imports = []
        users_file.functions = []

        consumer1 = MagicMock()
        consumer1.filepath = "/project/app/api.py"
        consumer1.path = "/project/app/api.py"
        c1_imp = MagicMock()
        c1_imp.source = "from services import login"
        consumer1.imports = [c1_imp]
        consumer1.functions = []

        consumer2 = MagicMock()
        consumer2.filepath = "/project/app/views.py"
        consumer2.path = "/project/app/views.py"
        c2_imp = MagicMock()
        c2_imp.source = "from services import get_user"
        consumer2.imports = [c2_imp]
        consumer2.functions = []

        codebase = MagicMock()
        codebase.files = [init_file, auth_file, users_file, consumer1, consumer2]
        graph._codebases = {"python": codebase}

        graph._build_import_index()

        # consumer1 imports login -> should link to auth.py
        auth_importers = graph._import_index.get("/project/services/auth.py", [])
        assert "/project/app/api.py" in auth_importers

        # consumer2 imports get_user -> should link to users.py
        users_importers = graph._import_index.get("/project/services/users.py", [])
        assert "/project/app/views.py" in users_importers

        # consumer1 should NOT link to users.py (imports login, not get_user)
        assert "/project/app/api.py" not in users_importers

    # ------------------------------------------------------------------ #
    # Pattern 2: __getattr__ lazy loading
    # ------------------------------------------------------------------ #

    def test_getattr_lazy_reexport(self):
        """__getattr__ lazy loading pattern should resolve re-exports."""
        graph = self._make_graph()

        init_file = MagicMock()
        init_file.filepath = "/project/package/__init__.py"
        init_file.path = "/project/package/__init__.py"
        init_file.imports = []

        # Simulate __getattr__ function with lazy import dict
        getattr_func = MagicMock()
        getattr_func.name = "__getattr__"
        getattr_func.source = '''def __getattr__(name):
    _lazy = {
        "ServiceClass": ".impl",
        "Helper": ".utils",
    }
    if name in _lazy:
        import importlib
        mod = importlib.import_module(_lazy[name], __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")'''
        init_file.functions = [getattr_func]

        impl_file = MagicMock()
        impl_file.filepath = "/project/package/impl.py"
        impl_file.path = "/project/package/impl.py"
        impl_file.imports = []
        impl_file.functions = []

        utils_file = MagicMock()
        utils_file.filepath = "/project/package/utils.py"
        utils_file.path = "/project/package/utils.py"
        utils_file.imports = []
        utils_file.functions = []

        consumer = MagicMock()
        consumer.filepath = "/project/app/main.py"
        consumer.path = "/project/app/main.py"
        consumer_imp = MagicMock()
        consumer_imp.source = "from package import ServiceClass"
        consumer.imports = [consumer_imp]
        consumer.functions = []

        codebase = MagicMock()
        codebase.files = [init_file, impl_file, utils_file, consumer]
        graph._codebases = {"python": codebase}

        graph._build_import_index()

        # Consumer imports ServiceClass -> should link to impl.py
        impl_importers = graph._import_index.get("/project/package/impl.py", [])
        assert "/project/app/main.py" in impl_importers

    # ------------------------------------------------------------------ #
    # Pattern 3: __all__ combined with explicit imports
    # ------------------------------------------------------------------ #

    def test_all_with_explicit_imports(self):
        """__all__ combined with from .module import X should work."""
        graph = self._make_graph()

        init_file = MagicMock()
        init_file.filepath = "/project/mylib/__init__.py"
        init_file.path = "/project/mylib/__init__.py"
        init_imp = MagicMock()
        init_imp.source = "from .core import process"
        init_file.imports = [init_imp]
        init_file.functions = []

        core_file = MagicMock()
        core_file.filepath = "/project/mylib/core.py"
        core_file.path = "/project/mylib/core.py"
        core_file.imports = []
        core_file.functions = []

        consumer = MagicMock()
        consumer.filepath = "/project/main.py"
        consumer.path = "/project/main.py"
        c_imp = MagicMock()
        c_imp.source = "from mylib import process"
        consumer.imports = [c_imp]
        consumer.functions = []

        codebase = MagicMock()
        codebase.files = [init_file, core_file, consumer]
        graph._codebases = {"python": codebase}

        graph._build_import_index()

        core_importers = graph._import_index.get("/project/mylib/core.py", [])
        assert "/project/main.py" in core_importers

    # ------------------------------------------------------------------ #
    # Edge cases
    # ------------------------------------------------------------------ #

    def test_init_defined_symbols_not_resolved_elsewhere(self):
        """Symbols defined IN __init__.py (not re-exported) should NOT
        be redirected to a submodule."""
        graph = self._make_graph()

        init_file = MagicMock()
        init_file.filepath = "/project/pkg/__init__.py"
        init_file.path = "/project/pkg/__init__.py"
        # No re-export imports
        init_file.imports = []
        init_file.functions = []

        consumer = MagicMock()
        consumer.filepath = "/project/main.py"
        consumer.path = "/project/main.py"
        c_imp = MagicMock()
        c_imp.source = "from pkg import VERSION"
        consumer.imports = [c_imp]
        consumer.functions = []

        codebase = MagicMock()
        codebase.files = [init_file, consumer]
        graph._codebases = {"python": codebase}

        graph._build_import_index()

        # Consumer should only link to __init__.py
        init_importers = graph._import_index.get("/project/pkg/__init__.py", [])
        assert "/project/main.py" in init_importers

    def test_subpackage_reexport_resolution(self):
        """Re-export from a sub-package (directory) should resolve to
        sub-package/__init__.py if the .py file doesn't exist."""
        graph = self._make_graph()

        init_file = MagicMock()
        init_file.filepath = "/project/pkg/__init__.py"
        init_file.path = "/project/pkg/__init__.py"
        init_imp = MagicMock()
        init_imp.source = "from .sub import handler"
        init_file.imports = [init_imp]
        init_file.functions = []

        # sub is a sub-package, not a .py file
        sub_init = MagicMock()
        sub_init.filepath = "/project/pkg/sub/__init__.py"
        sub_init.path = "/project/pkg/sub/__init__.py"
        sub_init.imports = []
        sub_init.functions = []

        consumer = MagicMock()
        consumer.filepath = "/project/main.py"
        consumer.path = "/project/main.py"
        c_imp = MagicMock()
        c_imp.source = "from pkg import handler"
        consumer.imports = [c_imp]
        consumer.functions = []

        codebase = MagicMock()
        codebase.files = [init_file, sub_init, consumer]
        graph._codebases = {"python": codebase}

        graph._build_import_index()

        # Should resolve to pkg/sub/__init__.py
        sub_importers = graph._import_index.get("/project/pkg/sub/__init__.py", [])
        assert "/project/main.py" in sub_importers

    def test_no_duplicate_entries_in_indexes(self):
        """Resolving re-exports should not create duplicate entries."""
        graph = self._make_graph()

        init_file = MagicMock()
        init_file.filepath = "/project/pkg/__init__.py"
        init_file.path = "/project/pkg/__init__.py"
        init_imp = MagicMock()
        init_imp.source = "from .core import func"
        init_file.imports = [init_imp]
        init_file.functions = []

        core_file = MagicMock()
        core_file.filepath = "/project/pkg/core.py"
        core_file.path = "/project/pkg/core.py"
        core_file.imports = []
        core_file.functions = []

        consumer = MagicMock()
        consumer.filepath = "/project/main.py"
        consumer.path = "/project/main.py"
        c_imp = MagicMock()
        c_imp.source = "from pkg import func"
        consumer.imports = [c_imp]
        consumer.functions = []

        codebase = MagicMock()
        codebase.files = [init_file, core_file, consumer]
        graph._codebases = {"python": codebase}

        graph._build_import_index()

        # No duplicates in reverse index
        core_importers = graph._import_index.get("/project/pkg/core.py", [])
        assert core_importers.count("/project/main.py") == 1

        # No duplicates in forward index
        consumer_fwd = graph._forward_import_index.get("/project/main.py", [])
        assert consumer_fwd.count("/project/pkg/core.py") == 1

    def test_symbol_imports_updated_for_resolved_source(self):
        """_file_symbol_imports should include entries for the resolved
        source file, not just __init__.py."""
        graph = self._make_graph()

        init_file = MagicMock()
        init_file.filepath = "/project/pkg/__init__.py"
        init_file.path = "/project/pkg/__init__.py"
        init_imp = MagicMock()
        init_imp.source = "from .core import MyClass"
        init_file.imports = [init_imp]
        init_file.functions = []

        core_file = MagicMock()
        core_file.filepath = "/project/pkg/core.py"
        core_file.path = "/project/pkg/core.py"
        core_file.imports = []
        core_file.functions = []

        consumer = MagicMock()
        consumer.filepath = "/project/main.py"
        consumer.path = "/project/main.py"
        c_imp = MagicMock()
        c_imp.source = "from pkg import MyClass"
        consumer.imports = [c_imp]
        consumer.functions = []

        codebase = MagicMock()
        codebase.files = [init_file, core_file, consumer]
        graph._codebases = {"python": codebase}

        graph._build_import_index()

        # Symbol imports should include core.py with MyClass
        sym_imports = graph._file_symbol_imports.get("/project/main.py", {})
        core_symbols = sym_imports.get("/project/pkg/core.py", set())
        assert "MyClass" in core_symbols, (
            f"Expected MyClass in core.py symbol imports, got: {sym_imports}"
        )

    def test_getattr_with_dotted_submodule(self):
        """__getattr__ with dotted submodule path like '.sub.module'."""
        graph = self._make_graph()

        init_file = MagicMock()
        init_file.filepath = "/project/pkg/__init__.py"
        init_file.path = "/project/pkg/__init__.py"
        init_file.imports = []

        getattr_func = MagicMock()
        getattr_func.name = "__getattr__"
        getattr_func.source = '''def __getattr__(name):
    _lazy = {
        "deep_func": ".sub.deep",
    }
    if name in _lazy:
        import importlib
        mod = importlib.import_module(_lazy[name], __name__)
        return getattr(mod, name)
    raise AttributeError(name)'''
        init_file.functions = [getattr_func]

        deep_file = MagicMock()
        deep_file.filepath = "/project/pkg/sub/deep.py"
        deep_file.path = "/project/pkg/sub/deep.py"
        deep_file.imports = []
        deep_file.functions = []

        consumer = MagicMock()
        consumer.filepath = "/project/main.py"
        consumer.path = "/project/main.py"
        c_imp = MagicMock()
        c_imp.source = "from pkg import deep_func"
        consumer.imports = [c_imp]
        consumer.functions = []

        codebase = MagicMock()
        codebase.files = [init_file, deep_file, consumer]
        graph._codebases = {"python": codebase}

        graph._build_import_index()

        deep_importers = graph._import_index.get("/project/pkg/sub/deep.py", [])
        assert "/project/main.py" in deep_importers

    def test_consumer_with_no_matching_symbols_not_linked(self):
        """If consumer imports a symbol NOT in the re-export map, don't
        add spurious links."""
        graph = self._make_graph()

        init_file = MagicMock()
        init_file.filepath = "/project/pkg/__init__.py"
        init_file.path = "/project/pkg/__init__.py"
        init_imp = MagicMock()
        init_imp.source = "from .core import Alpha"
        init_file.imports = [init_imp]
        init_file.functions = []

        core_file = MagicMock()
        core_file.filepath = "/project/pkg/core.py"
        core_file.path = "/project/pkg/core.py"
        core_file.imports = []
        core_file.functions = []

        # Consumer imports Beta, which is NOT re-exported from .core
        consumer = MagicMock()
        consumer.filepath = "/project/main.py"
        consumer.path = "/project/main.py"
        c_imp = MagicMock()
        c_imp.source = "from pkg import Beta"
        consumer.imports = [c_imp]
        consumer.functions = []

        codebase = MagicMock()
        codebase.files = [init_file, core_file, consumer]
        graph._codebases = {"python": codebase}

        graph._build_import_index()

        # Consumer should NOT be linked to core.py (Beta is not from core)
        core_importers = graph._import_index.get("/project/pkg/core.py", [])
        assert "/project/main.py" not in core_importers
