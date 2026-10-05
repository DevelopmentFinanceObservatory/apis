"""The optional Search engine package. Install with `uv sync --extra engine` or
`pip install 'search[engine]'`. Importing `search` alone does not load it.
"""

# Fail at import when the optional engine dependencies are missing.
try:
    import httpx as _httpx  # noqa: F401
except ImportError as _exc:
    raise ImportError(
        "search.engine requires the 'engine' extra. "
        "Install with: uv sync --extra engine (or pip install 'search[engine]')."
    ) from _exc

from search.engine.core import resolve, resolve_async

__all__ = ["resolve", "resolve_async"]
