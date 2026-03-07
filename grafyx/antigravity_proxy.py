"""Stdio proxy for Antigravity IDE — strips \\r from WSL stdout pipe.

Antigravity's MCP parser chokes on \\r\\n line endings that wsl.exe injects
when piping stdout from Linux to Windows.  This proxy runs on Windows Python,
spawns grafyx inside WSL, and forwards stdin/stdout in binary mode with \\r
characters stripped.

Usage (mcp_config.json):
    {
        "mcpServers": {
            "grafyx": {
                "command": "python",
                "args": ["-u", "C:\\path\\to\\grafyx\\grafyx\\antigravity_proxy.py"]
            }
        }
    }
"""

import os
import subprocess
import sys
import threading

# Ensure Windows binary mode — prevent Python itself from adding \r
if sys.platform == "win32":
    import msvcrt

    msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
    msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)


def forward_stdin(proc: subprocess.Popen) -> None:
    """Forward Antigravity → grafyx stdin (binary, untouched)."""
    try:
        assert proc.stdin is not None
        while True:
            # MCP sends JSON-RPC messages exactly one per line.
            data = sys.stdin.buffer.readline()
            if not data:
                break
            proc.stdin.write(data)
            proc.stdin.flush()
            proc.stdin.flush()
    except (OSError, BrokenPipeError):
        pass
    finally:
        try:
            proc.stdin.close()
        except OSError:
            pass


def forward_stdout(proc: subprocess.Popen) -> None:
    """Forward grafyx → Antigravity stdout, stripping \\r bytes."""
    try:
        assert proc.stdout is not None
        while True:
            data = proc.stdout.read(4096)
            if not data:
                break
            # Strip \r injected by wsl.exe line-ending translation
            clean = data.replace(b"\r", b"")
            sys.stdout.buffer.write(clean)
            sys.stdout.buffer.flush()
    except (OSError, BrokenPipeError):
        pass


def main() -> None:
    # Accept optional --project arg, default to "."
    project = "."
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--project" and i < len(sys.argv) - 1:
            project = sys.argv[i + 1]

    proc = subprocess.Popen(
        [
            "wsl", "-e",
            "/home/bilal/grafyx-venv/bin/python3",
            "-W", "ignore", "-u", "-c",
            (
                "import os,sys,warnings;"
                "os.environ['NO_COLOR']='1';"
                "os.environ['TERM']='dumb';"
                "warnings.filterwarnings('ignore');"
                "import logging;"
                "logging.basicConfig(level=logging.INFO,"
                "format='[grafyx] %(levelname)s %(message)s',stream=sys.stderr,force=True);"
                f"from grafyx.server import create_server;"
                f"s=create_server('{project}',lazy=True);"
                "s.run(transport='stdio',show_banner=False)"
            ),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    t_in = threading.Thread(target=forward_stdin, args=(proc,), daemon=True)
    t_out = threading.Thread(target=forward_stdout, args=(proc,), daemon=True)
    t_in.start()
    t_out.start()

    proc.wait()


if __name__ == "__main__":
    main()
