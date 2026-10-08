"""Pin each route's surface, method, path, name, operation ID, and summary across mounted surfaces.
The fixture records description presence, not text, so wording changes do not churn it.
Run `uv run python -m tests.conformance.test_route_metadata` to regenerate the fixture after route changes.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from fastapi.routing import APIRoute

from dfo_apis.app import all_surfaces
from dfo_platform.app import create_app

# Search surface construction loads coverage, so these defaults support standalone runs.
_COVERAGE_SLICE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "coverage_slice"
os.environ.setdefault("COVERAGE_ARTIFACT_URI", f"file://{_COVERAGE_SLICE_DIR}")
os.environ.setdefault("COVERAGE_RECHECK_SECONDS", "0")

_FIXTURE_PATH = Path(__file__).with_name("route_metadata.json")

_GRAPH_ENV_VARS = (
    "GRAPH_DCP_BASE_URL",
    "GRAPH_DCP_TOKEN",
    "GRAPH_NODE_POOL_SIZE",
    "GRAPH_RESOLVE_TIMEOUT_SECONDS",
    "GRAPH_NODE_TIMEOUT_SECONDS",
)

# Starlette keeps converters on routes, while OpenAPI strips `{id:dfo_spec_id}` to `{id}`.
_CONVERTER_SUFFIX_RE = re.compile(r"\{([^:}]+):[^}]+\}")


def _schema_path(path: str) -> str:
    return _CONVERTER_SUFFIX_RE.sub(r"{\1}", path)


def _current_route_rows() -> list[dict[str, Any]]:
    """Return route metadata as clients see it in the composed `/openapi.json`.

    Schema-excluded routes have no operation ID or summary and `has_description=False`.
    """
    for var in _GRAPH_ENV_VARS:
        os.environ.pop(var, None)
    surfaces = all_surfaces()
    app = create_app(surfaces)
    schema = app.openapi()

    rows: list[dict[str, Any]] = []
    for surface in surfaces:
        for route in surface.router.routes:
            if not isinstance(route, APIRoute):
                continue
            schema_path = _schema_path(route.path)
            for method in sorted(route.methods):
                operation = (
                    schema["paths"].get(schema_path, {}).get(method.lower(), {})
                    if route.include_in_schema
                    else {}
                )
                rows.append(
                    {
                        "surface": surface.name,
                        "method": method,
                        "path": schema_path,
                        "name": route.name,
                        "operation_id": operation.get("operationId"),
                        "summary": operation.get("summary"),
                        "has_description": bool(operation.get("description")),
                    }
                )
    rows.sort(key=lambda row: (row["surface"], row["path"], row["method"]))
    return rows


def _load_fixture() -> list[dict[str, Any]]:
    return json.loads(_FIXTURE_PATH.read_text())


def _write_fixture(rows: list[dict[str, Any]]) -> None:
    _FIXTURE_PATH.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")


def _route_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    # `refuse_malformed_id` and the real options route share a normalised path.
    # Include `name` to distinguish their converters.
    return (row["surface"], row["method"], row["path"], row["name"])


def test_route_metadata_matches_fixture():
    current = _current_route_rows()
    fixture = _load_fixture()

    current_by_key = {_route_key(row): row for row in current}
    fixture_by_key = {_route_key(row): row for row in fixture}
    assert len(current_by_key) == len(current), (
        "two current routes share a (surface, method, path, name) key"
    )
    assert len(fixture_by_key) == len(fixture), (
        "two fixture routes share a (surface, method, path, name) key"
    )

    added = current_by_key.keys() - fixture_by_key.keys()
    removed = fixture_by_key.keys() - current_by_key.keys()
    assert not added, f"routes served but not in the fixture: {sorted(added)}"
    assert not removed, f"routes in the fixture but no longer served: {sorted(removed)}"

    changed = {
        key: (fixture_by_key[key], current_by_key[key])
        for key in current_by_key
        if current_by_key[key] != fixture_by_key[key]
    }
    assert not changed, (
        "route metadata changed for: "
        f"{ {k: {'fixture': v[0], 'current': v[1]} for k, v in changed.items()} }"
    )


def test_fixture_route_count_per_surface():
    fixture = _load_fixture()
    counts: dict[str, int] = {}
    for row in fixture:
        counts[row["surface"]] = counts.get(row["surface"], 0) + 1
    assert counts == {
        "search": 6,
        "datasets": 54,
        "graph": 7,
        "presentation": 10,
        "skills": 4,
    }


if __name__ == "__main__":
    _write_fixture(_current_route_rows())
    print(f"wrote {_FIXTURE_PATH}")
