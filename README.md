# Grafyx

[![PyPI](https://img.shields.io/pypi/v/grafyx-mcp.svg?v=0.2.0)](https://pypi.org/project/grafyx-mcp/)
[![CI](https://github.com/bilal07karadeniz/Grafyx/actions/workflows/ci.yml/badge.svg)](https://github.com/bilal07karadeniz/Grafyx/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/protocol-MCP-green.svg)](https://modelcontextprotocol.io)

**Real-time codebase understanding for AI coding assistants.**

> Latest release: **v0.2.0** — semantic encoder (jina-v2) is now the default for `find_related_code` (+135% nDCG@10 over token search). See [CHANGELOG](CHANGELOG.md) and [0.2.0 benchmark](docs/benchmarks/0.2.0/summary.md).

---

## What is Grafyx?

AI coding tools read raw files with zero architectural understanding -- they don't know what calls what, which classes inherit from where, or how your modules connect. Grafyx fixes this by parsing your entire codebase into a full relationship graph using [Graph-sitter](https://github.com/getsentry/graph-sitter) (built on tree-sitter), then exposing that graph to any AI assistant through the [Model Context Protocol (MCP)](https://modelcontextprotocol.io). Your assistant can trace call chains, map dependencies, find related code by description, detect conventions, and understand your project's architecture -- all in real time, with a file watcher that keeps the graph current as you edit.

---

## Quick Start

### Claude Code

```bash
# Zero-install (recommended)
claude mcp add --scope user grafyx -- uvx --from grafyx-mcp grafyx

# Or install with pip first
pip install grafyx-mcp
claude mcp add --scope user grafyx -- grafyx
```

### Cursor / Windsurf / Cline

Add to your MCP config file:

- **Cursor**: `.cursor/mcp.json` (project) or `~/.cursor/mcp.json` (global)
- **Windsurf**: `~/.codeium/windsurf/mcp_config.json`
- **Cline**: Cline MCP settings in VS Code

```json
{
  "mcpServers": {
    "grafyx": {
      "command": "uvx",
      "args": ["--from", "grafyx-mcp", "grafyx"]
    }
  }
}
```

### VS Code (GitHub Copilot)

Add to `.vscode/mcp.json`:

```json
{
  "servers": {
    "grafyx": {
      "command": "uvx",
      "args": ["--from", "grafyx-mcp", "grafyx"]
    }
  }
}
```

> **Using pip instead of uvx?** Replace the command with: `"command": "grafyx"` (no args needed).

---

## Available Tools

| Tool | Description |
|------|-------------|
| `get_project_skeleton` | Full project structure with stats per module |
| `get_function_context` | Everything about a function: callers, callees, deps |
| `get_file_context` | File contents, imports, dependencies |
| `get_class_context` | Class methods, inheritance, usages |
| `find_related_code` | Natural language search across the codebase |
| `find_related_files` | Find files relevant to a feature by matching symbols |
| `get_dependency_graph` | Impact analysis: what depends on what |
| `get_conventions` | Detected coding patterns and conventions |
| `get_call_graph` | Call chain tracing upstream and downstream |
| `refresh_graph` | Force re-parse of the codebase |
| `get_module_context` | Symbols in a directory/package (intermediate zoom) |
| `get_subclasses` | Inheritance tree for a base class |
| `get_unused_symbols` | Dead code detection |
| `set_project` | Switch the served project at runtime |

---

## How It Works

```
Your AI Assistant
       |
       | MCP Protocol (stdio)
       v
  +-----------+
  |  Grafyx   |  FastMCP server with 14 tools
  |  Server   |
  +-----------+
       |
  +-----------+     +-----------+     +-------------+
  |  Graph    |---->|  Search   |     | Convention  |
  |  Engine   |---->|  Engine   |     | Detector    |
  +-----------+     +-----------+     +-------------+
       |
       v
  +-----------+
  |  Graph-   |  Tree-sitter based parsing
  |  sitter   |
  +-----------+
       |
  +-----------+
  |  Watchdog |  File watcher for live updates
  +-----------+
```

1. **Startup** -- Grafyx detects languages in your project and parses all source files into a semantic graph via Graph-sitter.
2. **Serving** -- The FastMCP server exposes 14 tools over stdio. Your AI assistant calls them as needed.
3. **Live updates** -- Watchdog monitors file changes. When you save, the graph is automatically re-parsed after a short debounce.

### ML-augmented search

Grafyx's `find_related_code` uses a pretrained code embedding model (default:
`jinaai/jina-embeddings-v2-base-code`, Apache-2.0, 161M params) running on CPU
via ONNX through [`fastembed`](https://github.com/qdrant/fastembed). The model
is downloaded on first use and cached locally — no GPU, no daemon, no cloud
calls. Install with `pip install grafyx-mcp[embeddings]`.

**Benchmark (0.2.0, 278 docstring→function queries across FastAPI + Django):**

| Encoder | nDCG@10 | MRR@10 | p50 latency |
|---|---:|---:|---:|
| jina-v2 (default) | **0.787** | **0.741** | ~1.5 s |
| coderankembed | 0.663 | 0.623 | ~1.3 s |
| tokens-only (no fastembed) | 0.335 | 0.297 | ~0.9 s |

The default encoder more than doubles retrieval quality over plain
source-token search (+135% nDCG@10).

Full breakdown + per-query JSONL: [`docs/benchmarks/0.2.0/`](docs/benchmarks/0.2.0/).

Switch encoders via the `GRAFYX_ENCODER` env var:

- `jina-v2` (default) — Apache-2.0, fastembed-native, ~150 MB. Wins on
  accuracy; recommended unless you have a specific reason to switch.
- `coderankembed` — MIT, 137M, ONNX-int8, ~140 MB. Lower latency but
  ~12 nDCG@10 points behind jina-v2 in our eval. Hosted at
  [`Bilal7Dev/grafyx-coderankembed-onnx`](https://huggingface.co/Bilal7Dev/grafyx-coderankembed-onnx).

Supporting numpy-only MLPs (~5 MB total weights, bundled in the wheel):

- **M1 Relevance ranker** — 33-feature MLP that re-ranks the encoder's top
  candidates using structural signals (caller count, name overlap, exports).
- **M3 Source token filter** — suppresses noise tokens (imports, strings,
  magic methods) from full-text search.
- **M4 Symbol importance** — weights symbols by caller count, exports, and
  structural signals.
- **Gibberish detector** — character-bigram MLP that blocks nonsense queries
  before they hit the index.

Reproducible benchmarks against FastAPI, Django, and Home Assistant ship in
[`benchmarks/`](benchmarks/) (`python -m scripts.run_all`).

---

## Supported Languages

| Language | Extensions |
|----------|------------|
| Python | `.py`, `.pyi` |
| TypeScript | `.ts`, `.tsx` |
| JavaScript | `.js`, `.jsx` |

Languages are auto-detected. To specify manually:

```bash
grafyx --languages python,typescript
```

---

## Options

```
grafyx [OPTIONS]

  --project PATH       Project to analyze (default: current directory)
  --languages LANGS    Comma-separated languages (default: auto-detect)
  --ignore PATTERNS    Additional directories to ignore
  --no-watch           Disable file watching
  --verbose, -v        Debug logging
  --version            Show version
```

Default ignored: `node_modules`, `.git`, `__pycache__`, `.venv`, `venv`, `.env`, `dist`, `build`, `.tox`, `.mypy_cache`, `.pytest_cache`, `.ruff_cache`, `egg-info`, `.eggs`, `.next`, `.nuxt`, `coverage`, `.coverage`, `.nyc_output`

---

## Multi-Agent Support

Grafyx works with agent teams. A single Grafyx instance serves all agents connected to the same project. When one agent modifies code, the file watcher updates the graph automatically, so other agents immediately see the changes.

---

## Contributing

```bash
git clone https://github.com/bilal07karadeniz/Grafyx.git
cd Grafyx
pip install -e ".[dev]"
pytest
```

---

## Troubleshooting

**Windows**: Graph-sitter requires Linux. Use WSL and configure your MCP client to launch via `wsl`:

```json
{
  "mcpServers": {
    "grafyx": {
      "command": "wsl",
      "args": ["-e", "bash", "-c", "source ~/your-venv/bin/activate && grafyx"]
    }
  }
}
```

---

## License

MIT -- see [LICENSE](LICENSE) for details.
