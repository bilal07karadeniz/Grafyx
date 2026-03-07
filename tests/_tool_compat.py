"""Compatibility helper for calling @mcp.tool functions across fastmcp versions.

Old fastmcp wraps decorated functions in FunctionTool objects (call via .fn()).
New fastmcp 3.0+ returns plain functions with __fastmcp__ metadata.
"""


def call_tool(tool, *args, **kwargs):
    """Call an @mcp.tool decorated function across fastmcp versions."""
    fn = getattr(tool, "fn", tool)
    return fn(*args, **kwargs)
