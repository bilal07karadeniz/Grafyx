"""Convention and pattern detection for Grafyx MCP server."""

import re
from dataclasses import dataclass, field

from grafyx.graph import CodebaseGraph


@dataclass
class Convention:
    """A detected coding convention."""

    category: str  # "naming", "imports", "error_handling", "structure", "typing", "async"
    pattern: str  # Human-readable description
    confidence: float  # 0.0 to 1.0 (how consistent this pattern is)
    examples: list[str] = field(default_factory=list)  # Up to 3 concrete examples


class ConventionDetector:
    """Analyzes the codebase graph to detect coding conventions."""

    def __init__(self, graph: CodebaseGraph):
        self._graph = graph

    def detect_all(self) -> list[dict]:
        """Run all detection methods and return combined results.

        Returns list of dicts: [{category, pattern, confidence, examples}]
        """
        # Fetch function list ONCE and share across detectors.
        # get_all_functions(include_methods=True) is expensive on large
        # codebases (iterates all graph-sitter objects under lock).
        # With caching (single call), 3000 is safe — covers both Python
        # and TS/JS without one language starving the other.
        all_functions = self._graph.get_all_functions(
            max_results=3000, include_methods=True,
        )

        conventions: list[Convention] = []
        conventions.extend(self.detect_naming_conventions(all_functions))
        conventions.extend(self.detect_structure_conventions())
        conventions.extend(self.detect_typing_conventions(all_functions))
        conventions.extend(self.detect_async_patterns(all_functions))
        conventions.extend(self.detect_docstring_conventions(all_functions))
        conventions.extend(self.detect_import_conventions())
        conventions.extend(self.detect_decorator_patterns())

        conventions.sort(key=lambda c: c.confidence, reverse=True)

        return [
            {
                "category": c.category,
                "pattern": c.pattern,
                "confidence": round(c.confidence, 2),
                "examples": c.examples,
            }
            for c in conventions
        ]

    # ------------------------------------------------------------------
    # Naming conventions
    # ------------------------------------------------------------------

    def detect_naming_conventions(
        self, all_functions: list[dict] | None = None,
    ) -> list[Convention]:
        """Detect naming patterns for functions, classes, and files."""
        conventions: list[Convention] = []
        self._detect_function_naming(conventions, all_functions)
        self._detect_class_naming(conventions)
        self._detect_file_naming(conventions)
        return conventions

    def _detect_function_naming(
        self, out: list[Convention], all_functions: list[dict] | None = None,
    ) -> None:
        functions = all_functions or self._graph.get_all_functions(
            max_results=3000, include_methods=True,
        )
        if not functions:
            return

        # Group by language
        by_lang: dict[str, list[dict]] = {}
        for f in functions:
            lang = f.get("language", "unknown")
            by_lang.setdefault(lang, []).append(f)

        for lang, lang_funcs in by_lang.items():
            lang_label = lang.capitalize()
            true_total = len(lang_funcs)

            # Sample first 500 for pattern detection (sufficient for statistics)
            names = [f.get("name", "") for f in lang_funcs[:500]]
            names = [n for n in names if n]

            # Strip leading/trailing underscores before checking style.
            # This ensures __init__, _private_method, __repr__ are recognized as snake_case.
            snake = [
                n for n in names
                if (s := n.strip("_")) and re.match(r"^[a-z][a-z0-9]*(_[a-z0-9]+)+$", s)
            ]
            camel = [
                n for n in names
                if re.match(r"^[a-z][a-zA-Z0-9]*$", n) and any(c.isupper() for c in n)
            ]
            total = len(names)
            if total == 0:
                continue

            if len(snake) >= len(camel):
                pct = len(snake) / total
                out.append(Convention(
                    category="naming",
                    pattern=f"{lang_label} functions use snake_case ({int(pct * 100)}% of {true_total} functions)",
                    confidence=pct,
                    examples=snake[:3],
                ))
            else:
                pct = len(camel) / total
                out.append(Convention(
                    category="naming",
                    pattern=f"{lang_label} functions use camelCase ({int(pct * 100)}% of {true_total} functions)",
                    confidence=pct,
                    examples=camel[:3],
                ))

    def _detect_class_naming(self, out: list[Convention]) -> None:
        classes = self._graph.get_all_classes(max_results=5000)
        if not classes:
            return

        # Group by language
        by_lang: dict[str, list[dict]] = {}
        for c in classes:
            lang = c.get("language", "unknown")
            by_lang.setdefault(lang, []).append(c)

        for lang, lang_classes in by_lang.items():
            lang_label = lang.capitalize()
            true_total = len(lang_classes)

            names = [c.get("name", "") for c in lang_classes]
            names = [n for n in names if n]

            pascal = [
                n for n in names
                if (s := n.lstrip("_")) and re.match(r"^[A-Z][a-zA-Z0-9]*$", s)
            ]
            total = len(names)
            if total == 0:
                continue

            pct = len(pascal) / total
            out.append(Convention(
                category="naming",
                pattern=f"{lang_label} classes use PascalCase ({int(pct * 100)}% of {true_total} classes)",
                confidence=pct,
                examples=pascal[:3],
            ))

    def _detect_file_naming(self, out: list[Convention]) -> None:
        files = self._graph.get_all_files(max_results=500)
        if not files:
            return

        stems: list[str] = []
        for f in files:
            path = f.get("path", "")
            if not path:
                continue
            # Extract filename only (not directory names)
            name = path.replace("\\", "/").rsplit("/", 1)[-1]
            stem = name.rsplit(".", 1)[0] if "." in name else name
            # Skip index files and dunder files — they don't reflect naming convention
            if stem and not stem.startswith("__") and stem not in ("index",):
                stems.append(stem)

        if not stems:
            return

        # Require at least one separator to classify style.
        # Single-word lowercase files (e.g., "utils", "types") are ambiguous
        # and inflate snake_case counts if included.
        snake = [s for s in stems if re.match(r"^[a-z][a-z0-9]*(_[a-z0-9]+)+$", s)]
        kebab = [s for s in stems if re.match(r"^[a-z][a-z0-9]*(-[a-z0-9]+)+$", s)]
        pascal = [s for s in stems if re.match(r"^[A-Z][a-zA-Z0-9]*$", s)]
        camel = [s for s in stems if re.match(r"^[a-z][a-zA-Z0-9]*$", s) and any(c.isupper() for c in s)]

        # Count files that have a clear naming signal (separator or casing)
        styled_count = len(snake) + len(kebab) + len(pascal) + len(camel)
        total = styled_count if styled_count > 0 else len(stems)

        # Find the dominant style
        styles = [
            ("PascalCase", pascal),
            ("camelCase", camel),
            ("snake_case", snake),
            ("kebab-case", kebab),
        ]
        styles.sort(key=lambda x: len(x[1]), reverse=True)
        best_name, best_list = styles[0]

        if best_list and total > 0:
            pct = len(best_list) / total
            out.append(Convention(
                category="naming",
                pattern=f"Files use {best_name} naming ({int(pct * 100)}% of {total} files)",
                confidence=pct,
                examples=best_list[:3],
            ))
            # If there's a significant secondary style, report it too
            if len(styles) > 1:
                sec_name, sec_list = styles[1]
                if sec_list and len(sec_list) / total >= 0.2:
                    sec_pct = len(sec_list) / total
                    out.append(Convention(
                        category="naming",
                        pattern=f"Files also use {sec_name} naming ({int(sec_pct * 100)}% of {total} files)",
                        confidence=sec_pct,
                        examples=sec_list[:3],
                    ))

    # ------------------------------------------------------------------
    # Structure conventions
    # ------------------------------------------------------------------

    def detect_structure_conventions(self) -> list[Convention]:
        """Detect project structure patterns."""
        conventions: list[Convention] = []
        files = self._graph.get_all_files(max_results=500)
        if not files:
            return conventions

        # Average functions per file
        func_counts = [f.get("function_count", 0) for f in files]
        if func_counts:
            avg_funcs = sum(func_counts) / len(func_counts)
            top = sorted(files, key=lambda x: x.get("function_count", 0), reverse=True)
            conventions.append(Convention(
                category="structure",
                pattern=f"Average {avg_funcs:.1f} functions per file across {len(files)} files",
                confidence=0.9,
                examples=[
                    f"{f.get('path', '?')}: {f.get('function_count', 0)} functions"
                    for f in top[:3]
                ],
            ))

        # One-class-per-file pattern
        class_counts = [f.get("class_count", 0) for f in files]
        files_with_classes = sum(1 for c in class_counts if c > 0)
        if files_with_classes > 0:
            one_class = sum(1 for c in class_counts if c == 1)
            if one_class > files_with_classes * 0.6:
                conventions.append(Convention(
                    category="structure",
                    pattern=f"One class per file pattern ({one_class}/{files_with_classes} files with classes)",
                    confidence=one_class / max(files_with_classes, 1),
                    examples=[
                        f.get("path", "?")
                        for f in files
                        if f.get("class_count", 0) == 1
                    ][:3],
                ))

        # Test file detection
        test_files = [f for f in files if "test" in f.get("path", "").lower()]
        if test_files:
            test_prefix = sum(
                1 for f in test_files
                if f.get("path", "").rsplit("/", 1)[-1].rsplit("\\", 1)[-1].startswith("test_")
            )
            test_suffix = sum(
                1 for f in test_files
                if f.get("path", "").rsplit("/", 1)[-1].rsplit("\\", 1)[-1].endswith(
                    ("_test.py", ".test.ts", ".test.js", ".spec.ts", ".spec.js")
                )
            )

            if test_prefix > test_suffix:
                conventions.append(Convention(
                    category="structure",
                    pattern=f"Test files use 'test_' prefix ({test_prefix} files)",
                    confidence=0.85,
                    examples=[
                        f.get("path", "?") for f in test_files
                        if f.get("path", "").rsplit("/", 1)[-1].rsplit("\\", 1)[-1].startswith("test_")
                    ][:3],
                ))
            elif test_suffix > 0:
                conventions.append(Convention(
                    category="structure",
                    pattern=f"Test files use '_test' / '.test' / '.spec' suffix ({test_suffix} files)",
                    confidence=0.85,
                    examples=[f.get("path", "?") for f in test_files[:3]],
                ))

        return conventions

    # ------------------------------------------------------------------
    # Typing conventions
    # ------------------------------------------------------------------

    def detect_typing_conventions(
        self, all_functions: list[dict] | None = None,
    ) -> list[Convention]:
        """Detect type annotation usage patterns."""
        conventions: list[Convention] = []
        functions = all_functions or self._graph.get_all_functions(
            max_results=3000, include_methods=True,
        )
        if not functions:
            return conventions

        # Group by language
        by_lang: dict[str, list[dict]] = {}
        for f in functions:
            lang = f.get("language", "unknown")
            by_lang.setdefault(lang, []).append(f)

        for lang, lang_funcs in by_lang.items():
            lang_label = lang.capitalize()
            true_total = len(lang_funcs)

            signatures = [f.get("signature", "") for f in lang_funcs[:500]]
            total = len(signatures)
            if total == 0:
                continue

            # Return type annotations — use regex to ensure '->' follows
            # the closing paren, not just appears anywhere in the signature
            # (e.g. dict literals with '->' in default values)
            with_return = [s for s in signatures if re.search(r'\)\s*->', s)]
            pct = len(with_return) / total

            if pct > 0.5:
                conventions.append(Convention(
                    category="typing",
                    pattern=f"{lang_label}: {int(pct * 100)}% of functions have return type annotations ({len(with_return)}/{true_total})",
                    confidence=pct,
                    examples=with_return[:3],
                ))
            elif pct < 0.2:
                without_return = [s for s in signatures if "->" not in s]
                conventions.append(Convention(
                    category="typing",
                    pattern=f"{lang_label}: Minimal type annotations ({int(pct * 100)}% of functions have return types)",
                    confidence=1 - pct,
                    examples=without_return[:3],
                ))

            # Parameter type annotations
            with_param_types = []
            for sig in signatures:
                paren_start = sig.find("(")
                paren_end = sig.find(")")
                if paren_start >= 0 and paren_end >= 0:
                    params_str = sig[paren_start + 1:paren_end]
                    parts = [p.strip() for p in params_str.split(",") if p.strip()]
                    typed = [
                        p for p in parts
                        if ":" in p.split("=")[0]
                        and p.split("=")[0].split(":")[0].strip() not in ("self", "cls")
                    ]
                    if typed:
                        with_param_types.append(sig)

            if total > 0:
                pct_params = len(with_param_types) / total
                if pct_params > 0.3:
                    conventions.append(Convention(
                        category="typing",
                        pattern=f"{lang_label}: {int(pct_params * 100)}% of functions have parameter type annotations",
                        confidence=pct_params,
                        examples=with_param_types[:3],
                    ))

        return conventions

    # ------------------------------------------------------------------
    # Async patterns
    # ------------------------------------------------------------------

    def detect_async_patterns(
        self, all_functions: list[dict] | None = None,
    ) -> list[Convention]:
        """Detect async/await usage patterns."""
        conventions: list[Convention] = []
        functions = all_functions or self._graph.get_all_functions(
            max_results=3000, include_methods=True,
        )
        if not functions:
            return conventions

        # Group by language
        by_lang: dict[str, list[dict]] = {}
        for f in functions:
            lang = f.get("language", "unknown")
            by_lang.setdefault(lang, []).append(f)

        for lang, lang_funcs in by_lang.items():
            lang_label = lang.capitalize()
            true_total = len(lang_funcs)
            async_names = [
                f.get("name", "?") for f in lang_funcs
                if f.get("signature", "").startswith("async ")
            ]

            if async_names and true_total > 0:
                pct = len(async_names) / true_total
                conventions.append(Convention(
                    category="async",
                    pattern=f"{lang_label}: {int(pct * 100)}% of functions are async ({len(async_names)}/{true_total})",
                    confidence=max(pct, 0.5),
                    examples=async_names[:3],
                ))

        return conventions

    # ------------------------------------------------------------------
    # Docstring conventions
    # ------------------------------------------------------------------

    def detect_docstring_conventions(
        self, all_functions: list[dict] | None = None,
    ) -> list[Convention]:
        """Detect docstring style patterns (Google, NumPy, Sphinx, or none)."""
        conventions: list[Convention] = []
        # Include methods to get accurate coverage — excluding them inflates
        # the stat because __init__, __repr__, helpers often lack docstrings.
        functions = all_functions or self._graph.get_all_functions(
            max_results=3000, include_methods=True,
        )
        if not functions:
            return conventions

        # Group by language
        by_lang: dict[str, list[dict]] = {}
        for f in functions:
            lang = f.get("language", "unknown")
            by_lang.setdefault(lang, []).append(f)

        for lang, lang_funcs in by_lang.items():
            lang_label = lang.capitalize()
            true_total = len(lang_funcs)
            total = true_total
            docstrings = [f.get("docstring") or "" for f in lang_funcs]
            docstrings = [d for d in docstrings if d and d.strip()]

            if not docstrings:
                conventions.append(Convention(
                    category="docstrings",
                    pattern=f"{lang_label}: No docstrings found across {true_total} functions",
                    confidence=0.9,
                    examples=[],
                ))
                continue

            # Coverage
            coverage = len(docstrings) / total

            # Detect style by scanning docstring bodies
            google_count = 0  # "Args:", "Returns:", "Raises:"
            numpy_count = 0   # "Parameters\n----------"
            sphinx_count = 0  # ":param", ":returns:", ":type"
            plain_count = 0   # No structured sections

            google_examples: list[str] = []
            numpy_examples: list[str] = []
            sphinx_examples: list[str] = []

            for doc in docstrings:
                if re.search(r"^\s*(Args|Returns|Raises|Yields|Attributes)\s*:", doc, re.MULTILINE):
                    google_count += 1
                    if len(google_examples) < 3:
                        google_examples.append(doc.strip().split("\n")[0][:80])
                elif re.search(r"^\s*Parameters\s*\n\s*-{3,}", doc, re.MULTILINE):
                    numpy_count += 1
                    if len(numpy_examples) < 3:
                        numpy_examples.append(doc.strip().split("\n")[0][:80])
                elif re.search(r":(param|type|returns|rtype|raises)\s", doc):
                    sphinx_count += 1
                    if len(sphinx_examples) < 3:
                        sphinx_examples.append(doc.strip().split("\n")[0][:80])
                else:
                    plain_count += 1

            # Report coverage
            conventions.append(Convention(
                category="docstrings",
                pattern=f"{lang_label}: {int(coverage * 100)}% of functions have docstrings ({len(docstrings)}/{true_total})",
                confidence=coverage,
                examples=docstrings[:3],
            ))

            # Report dominant style
            style_counts = {
                "Google style (Args/Returns sections)": (google_count, google_examples),
                "NumPy style (Parameters/underline sections)": (numpy_count, numpy_examples),
                "Sphinx style (:param/:returns: tags)": (sphinx_count, sphinx_examples),
            }

            best_style = max(style_counts.items(), key=lambda x: x[1][0])
            if best_style[1][0] > 0:
                style_name, (count, examples) = best_style
                pct = count / len(docstrings)
                conventions.append(Convention(
                    category="docstrings",
                    pattern=f"{lang_label}: Docstrings use {style_name} ({int(pct * 100)}% of {len(docstrings)} docstrings)",
                    confidence=pct,
                    examples=examples,
                ))

            if plain_count > 0 and plain_count == len(docstrings):
                conventions.append(Convention(
                    category="docstrings",
                    pattern=f"{lang_label}: Docstrings are plain text (no structured format)",
                    confidence=0.8,
                    examples=[d.strip().split("\n")[0][:80] for d in docstrings[:3]],
                ))

        return conventions

    # ------------------------------------------------------------------
    # Import conventions
    # ------------------------------------------------------------------

    def detect_import_conventions(self) -> list[Convention]:
        """Detect import style patterns (absolute vs relative, ordering)."""
        conventions: list[Convention] = []

        # Use pre-cached import sources from index building (collected
        # during _build_import_index which already iterates all files).
        # This avoids re-iterating graph-sitter objects (slow V8 bridge).
        all_imports: list[dict] = getattr(
            self._graph, "_convention_import_sources", None,
        ) or []

        if not all_imports:
            return conventions

        # Group by language for per-language reporting
        imports_by_lang: dict[str, list[dict]] = {}
        for imp in all_imports:
            lang = imp.get("language", "unknown")
            imports_by_lang.setdefault(lang, []).append(imp)

        for lang, lang_imports in imports_by_lang.items():
            lang_label = lang.capitalize()

            relative_imports = []
            absolute_imports = []
            for imp in lang_imports:
                src = imp["source"].strip()
                # Extract the module path for relative/absolute classification.
                # TS/JS: `import { Foo } from "./bar"` → module is "./bar"
                # Python: `from .utils import helper` → module is ".utils"
                # Python: `from app.core import X` → module is "app.core"
                m = re.search(r'''from\s+['"](.+?)['"]''', src)
                if m:
                    # TS/JS style: from "..." or from '...'
                    module = m.group(1)
                else:
                    # Python style: from X import Y  or  import X
                    module = src.replace("from ", "").split(" import ")[0].strip()
                if module.startswith(".") or module.startswith(".."):
                    relative_imports.append(src)
                elif src:
                    absolute_imports.append(src)

            total = len(relative_imports) + len(absolute_imports)
            if total == 0:
                continue

            if relative_imports and absolute_imports:
                rel_pct = len(relative_imports) / total
                abs_pct = len(absolute_imports) / total
                if rel_pct > abs_pct:
                    conventions.append(Convention(
                        category="imports",
                        pattern=f"{lang_label}: Predominantly relative imports ({int(rel_pct * 100)}% of {total} imports)",
                        confidence=rel_pct,
                        examples=relative_imports[:3],
                    ))
                else:
                    conventions.append(Convention(
                        category="imports",
                        pattern=f"{lang_label}: Predominantly absolute imports ({int(abs_pct * 100)}% of {total} imports)",
                        confidence=abs_pct,
                        examples=absolute_imports[:3],
                    ))
            elif relative_imports:
                conventions.append(Convention(
                    category="imports",
                    pattern=f"{lang_label}: Uses relative imports exclusively ({len(relative_imports)} imports)",
                    confidence=0.95,
                    examples=relative_imports[:3],
                ))
            elif absolute_imports:
                conventions.append(Convention(
                    category="imports",
                    pattern=f"{lang_label}: Uses absolute imports exclusively ({len(absolute_imports)} imports)",
                    confidence=0.95,
                    examples=absolute_imports[:3],
                ))

        # Detect common third-party packages
        # TypeScript/JavaScript keywords that appear in import statements
        # but are NOT package names (e.g., "import type { Foo }", "export default")
        _IMPORT_NOISE = {
            "type", "export", "default", "from", "as", "import", "const",
            "let", "var", "function", "class", "interface", "enum",
            "async", "await", "return", "if", "else", "for", "while",
            "new", "this", "super", "extends", "implements",
        }
        third_party: dict[str, int] = {}
        for imp in all_imports:
            src = imp["source"].strip()
            # Extract top-level package name
            pkg = src.replace("from ", "").replace("import ", "").split(".")[0].split(" ")[0].strip()
            if pkg and not pkg.startswith(".") and len(pkg) > 1 and pkg not in _IMPORT_NOISE:
                third_party[pkg] = third_party.get(pkg, 0) + 1

        # Filter to likely third-party (not stdlib-obvious and used multiple times)
        popular = sorted(third_party.items(), key=lambda x: x[1], reverse=True)[:10]
        if popular:
            pkg_list = [f"{name} ({count}x)" for name, count in popular[:5]]
            conventions.append(Convention(
                category="imports",
                pattern=f"Most imported packages: {', '.join(pkg_list)}",
                confidence=0.85,
                examples=[name for name, _ in popular[:3]],
            ))

        return conventions

    # ------------------------------------------------------------------
    # Decorator patterns
    # ------------------------------------------------------------------

    def detect_decorator_patterns(self) -> list[Convention]:
        """Detect common decorator usage patterns (@property, @classmethod, etc.)."""
        conventions: list[Convention] = []

        # Use pre-cached decorator info from index building (collected
        # during _build_caller_index which already iterates all classes/methods).
        # This avoids re-iterating graph-sitter objects (slow V8 bridge).
        conv_decorators: dict[str, dict[str, tuple[int, list[str]]]] = getattr(
            self._graph, "_convention_decorator_info", None,
        ) or {}
        conv_method_counts: dict[str, int] = getattr(
            self._graph, "_convention_method_counts", None,
        ) or {}

        # Reshape cached data into the format expected below
        lang_decorator_counts: dict[str, dict[str, int]] = {}
        lang_decorator_examples: dict[str, dict[str, list[str]]] = {}
        lang_total_methods: dict[str, int] = conv_method_counts

        for lang, dec_map in conv_decorators.items():
            lang_decorator_counts[lang] = {}
            lang_decorator_examples[lang] = {}
            for d_str, (count, examples) in dec_map.items():
                lang_decorator_counts[lang][d_str] = count
                lang_decorator_examples[lang][d_str] = examples

        for lang, decorator_counts in lang_decorator_counts.items():
            lang_label = lang.capitalize()
            total_methods = lang_total_methods.get(lang, 0)
            if not decorator_counts or total_methods == 0:
                continue

            for dec_name, count in sorted(decorator_counts.items(), key=lambda x: -x[1]):
                if count < 3:
                    continue
                pct = count / total_methods
                examples = lang_decorator_examples.get(lang, {}).get(dec_name, [])
                conventions.append(Convention(
                    category="decorators",
                    pattern=f"{lang_label}: @{dec_name} used on {count} methods ({int(pct * 100)}% of {total_methods} class methods)",
                    confidence=min(0.95, pct + 0.5) if count >= 5 else 0.6,
                    examples=examples,
                ))

        return conventions
