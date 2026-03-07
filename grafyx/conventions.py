"""Convention and pattern detection for Grafyx MCP server."""

import re
from dataclasses import dataclass, field

from grafyx.graph import CodebaseGraph
from grafyx.utils import safe_get_attr, safe_str


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
        conventions: list[Convention] = []
        conventions.extend(self.detect_naming_conventions())
        conventions.extend(self.detect_structure_conventions())
        conventions.extend(self.detect_typing_conventions())
        conventions.extend(self.detect_async_patterns())
        conventions.extend(self.detect_docstring_conventions())
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

    def detect_naming_conventions(self) -> list[Convention]:
        """Detect naming patterns for functions, classes, and files."""
        conventions: list[Convention] = []
        self._detect_function_naming(conventions)
        self._detect_class_naming(conventions)
        self._detect_file_naming(conventions)
        return conventions

    def _detect_function_naming(self, out: list[Convention]) -> None:
        functions = self._graph.get_all_functions(max_results=5000)
        if not functions:
            return
        true_total = len(functions)

        # Sample first 500 for pattern detection (sufficient for statistics)
        names = [f.get("name", "") for f in functions[:500]]
        names = [n for n in names if n]

        # Strip leading/trailing underscores before checking style.
        # This ensures __init__, _private_method, __repr__ are recognized as snake_case.
        snake = [
            n for n in names
            if (s := n.strip("_")) and re.match(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$", s)
        ]
        camel = [
            n for n in names
            if re.match(r"^[a-z][a-zA-Z0-9]*$", n) and any(c.isupper() for c in n)
        ]
        total = len(names)
        if total == 0:
            return

        if len(snake) >= len(camel):
            pct = len(snake) / total
            out.append(Convention(
                category="naming",
                pattern=f"Functions use snake_case ({int(pct * 100)}% of {true_total} functions)",
                confidence=pct,
                examples=snake[:3],
            ))
        else:
            pct = len(camel) / total
            out.append(Convention(
                category="naming",
                pattern=f"Functions use camelCase ({int(pct * 100)}% of {true_total} functions)",
                confidence=pct,
                examples=camel[:3],
            ))

    def _detect_class_naming(self, out: list[Convention]) -> None:
        classes = self._graph.get_all_classes(max_results=5000)
        if not classes:
            return
        true_total = len(classes)

        names = [c.get("name", "") for c in classes]
        names = [n for n in names if n]

        pascal = [
            n for n in names
            if (s := n.lstrip("_")) and re.match(r"^[A-Z][a-zA-Z0-9]*$", s)
        ]
        total = len(names)
        if total == 0:
            return

        pct = len(pascal) / total
        out.append(Convention(
            category="naming",
            pattern=f"Classes use PascalCase ({int(pct * 100)}% of {true_total} classes)",
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

        total = len(stems)
        snake = [s for s in stems if re.match(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$", s)]
        kebab = [s for s in stems if re.match(r"^[a-z][a-z0-9]*(-[a-z0-9]+)+$", s)]
        pascal = [s for s in stems if re.match(r"^[A-Z][a-zA-Z0-9]*$", s)]
        camel = [s for s in stems if re.match(r"^[a-z][a-zA-Z0-9]*$", s) and any(c.isupper() for c in s)]

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

    def detect_typing_conventions(self) -> list[Convention]:
        """Detect type annotation usage patterns."""
        conventions: list[Convention] = []
        functions = self._graph.get_all_functions(max_results=5000)
        if not functions:
            return conventions
        true_total = len(functions)

        signatures = [f.get("signature", "") for f in functions[:500]]
        total = len(signatures)
        if total == 0:
            return conventions

        # Return type annotations
        with_return = [s for s in signatures if "->" in s]
        pct = len(with_return) / total

        if pct > 0.5:
            conventions.append(Convention(
                category="typing",
                pattern=f"{int(pct * 100)}% of functions have return type annotations ({len(with_return)}/{true_total})",
                confidence=pct,
                examples=with_return[:3],
            ))
        elif pct < 0.2:
            without_return = [s for s in signatures if "->" not in s]
            conventions.append(Convention(
                category="typing",
                pattern=f"Minimal type annotations ({int(pct * 100)}% of functions have return types)",
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
                    if ":" in p and p.split(":")[0].strip() not in ("self", "cls")
                ]
                if typed:
                    with_param_types.append(sig)

        if total > 0:
            pct_params = len(with_param_types) / total
            if pct_params > 0.3:
                conventions.append(Convention(
                    category="typing",
                    pattern=f"{int(pct_params * 100)}% of functions have parameter type annotations",
                    confidence=pct_params,
                    examples=with_param_types[:3],
                ))

        return conventions

    # ------------------------------------------------------------------
    # Async patterns
    # ------------------------------------------------------------------

    def detect_async_patterns(self) -> list[Convention]:
        """Detect async/await usage patterns."""
        conventions: list[Convention] = []
        functions = self._graph.get_all_functions(max_results=5000)
        if not functions:
            return conventions

        true_total = len(functions)
        total = len(functions)
        async_names = [
            f.get("name", "?") for f in functions
            if f.get("signature", "").startswith("async ")
        ]

        if async_names and total > 0:
            pct = len(async_names) / total
            conventions.append(Convention(
                category="async",
                pattern=f"{int(pct * 100)}% of functions are async ({len(async_names)}/{true_total})",
                confidence=max(pct, 0.5),
                examples=async_names[:3],
            ))

        return conventions

    # ------------------------------------------------------------------
    # Docstring conventions
    # ------------------------------------------------------------------

    def detect_docstring_conventions(self) -> list[Convention]:
        """Detect docstring style patterns (Google, NumPy, Sphinx, or none)."""
        conventions: list[Convention] = []
        # Include methods to get accurate coverage — excluding them inflates
        # the stat because __init__, __repr__, helpers often lack docstrings.
        functions = self._graph.get_all_functions(max_results=5000, include_methods=True)
        if not functions:
            return conventions

        true_total = len(functions)
        total = len(functions)
        docstrings = [f.get("docstring") or "" for f in functions]
        docstrings = [d for d in docstrings if d and d.strip()]

        if not docstrings:
            conventions.append(Convention(
                category="docstrings",
                pattern=f"No docstrings found across {true_total} functions",
                confidence=0.9,
                examples=[],
            ))
            return conventions

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
            pattern=f"{int(coverage * 100)}% of functions have docstrings ({len(docstrings)}/{true_total})",
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
                pattern=f"Docstrings use {style_name} ({int(pct * 100)}% of {len(docstrings)} docstrings)",
                confidence=pct,
                examples=examples,
            ))

        if plain_count > 0 and plain_count == len(docstrings):
            conventions.append(Convention(
                category="docstrings",
                pattern="Docstrings are plain text (no structured format)",
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

        # Get raw import data from graph-sitter codebases
        all_imports: list[dict] = []
        with self._graph._lock:
            for lang, codebase in self._graph._codebases.items():
                try:
                    for f in codebase.files:
                        imports = safe_get_attr(f, "imports", [])
                        if not imports:
                            continue
                        fpath = str(safe_get_attr(f, "filepath", safe_get_attr(f, "path", "")))
                        seen_stmts: set[tuple[str, str]] = set()
                        for imp in imports:
                            source = safe_str(safe_get_attr(imp, "source", ""))
                            if not source:
                                source = safe_str(imp)
                            # Deduplicate: count import STATEMENTS not names.
                            # "from X import a, b, c" may appear as 3 objects
                            # but is one statement. Key on (file, module).
                            module = source.replace("from ", "").split(" import ")[0].strip()
                            if not module:
                                module = source
                            stmt_key = (fpath, module)
                            if stmt_key in seen_stmts:
                                continue
                            seen_stmts.add(stmt_key)
                            all_imports.append({
                                "source": source,
                                "file": fpath,
                                "language": lang,
                            })
                except Exception:
                    continue

        if not all_imports:
            return conventions

        # Analyze import sources
        relative_imports = []
        absolute_imports = []

        for imp in all_imports:
            src = imp["source"].strip()
            if src.startswith(".") or src.startswith("from ."):
                relative_imports.append(src)
            elif src:
                absolute_imports.append(src)

        total = len(relative_imports) + len(absolute_imports)
        if total == 0:
            return conventions

        # Relative vs absolute
        if relative_imports and absolute_imports:
            rel_pct = len(relative_imports) / total
            abs_pct = len(absolute_imports) / total
            if rel_pct > abs_pct:
                conventions.append(Convention(
                    category="imports",
                    pattern=f"Predominantly relative imports ({int(rel_pct * 100)}% of {total} imports)",
                    confidence=rel_pct,
                    examples=relative_imports[:3],
                ))
            else:
                conventions.append(Convention(
                    category="imports",
                    pattern=f"Predominantly absolute imports ({int(abs_pct * 100)}% of {total} imports)",
                    confidence=abs_pct,
                    examples=absolute_imports[:3],
                ))
        elif relative_imports:
            conventions.append(Convention(
                category="imports",
                pattern=f"Uses relative imports exclusively ({len(relative_imports)} imports)",
                confidence=0.95,
                examples=relative_imports[:3],
            ))
        elif absolute_imports:
            conventions.append(Convention(
                category="imports",
                pattern=f"Uses absolute imports exclusively ({len(absolute_imports)} imports)",
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

        # Scan class methods for decorators
        decorator_counts: dict[str, int] = {}
        decorator_examples: dict[str, list[str]] = {}
        total_methods = 0

        with self._graph._lock:
            for lang, codebase in self._graph._codebases.items():
                try:
                    for cls in codebase.classes:
                        cls_name = safe_get_attr(cls, "name", "")
                        methods = safe_get_attr(cls, "methods", [])
                        if not methods:
                            continue
                        for m in methods:
                            total_methods += 1
                            decorators = safe_get_attr(m, "decorators", [])
                            if not decorators:
                                continue
                            m_name = safe_get_attr(m, "name", "?")
                            for d in decorators:
                                d_str = safe_str(d).strip("@").split("(")[0]
                                if not d_str:
                                    continue
                                decorator_counts[d_str] = decorator_counts.get(d_str, 0) + 1
                                if d_str not in decorator_examples:
                                    decorator_examples[d_str] = []
                                if len(decorator_examples[d_str]) < 3:
                                    label = f"{cls_name}.{m_name}" if cls_name else m_name
                                    decorator_examples[d_str].append(label)
                except Exception:
                    continue

        if not decorator_counts or total_methods == 0:
            return conventions

        # Report significant decorator patterns (used 3+ times)
        for dec_name, count in sorted(decorator_counts.items(), key=lambda x: -x[1]):
            if count < 3:
                continue
            pct = count / total_methods
            examples = decorator_examples.get(dec_name, [])
            conventions.append(Convention(
                category="decorators",
                pattern=f"@{dec_name} used on {count} methods ({int(pct * 100)}% of {total_methods} class methods)",
                confidence=min(0.95, pct + 0.5) if count >= 5 else 0.6,
                examples=examples,
            ))

        return conventions
