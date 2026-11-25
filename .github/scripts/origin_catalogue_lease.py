"""Prints where the origin catalogue job keeps its build lease, and the job's task timeout.

Reads a `gcloud run jobs describe --format=json` document of the origin catalogue job on
stdin and writes two `name=value` lines for $GITHUB_OUTPUT:

- `lease`: the lease object `origin_catalogue.job` takes, `lease.json` under the format root
  of the checked-out code, joined onto the job's ORIGIN_CATALOGUE_URI as `join_uri` joins it.
- `task_timeout`: the job's task timeout in seconds, which `origin_catalogue.job` requires to
  equal `_MAX_RUN_SECONDS`, the lease's lifetime.

The deploy runner has no project environment, so the format root and the lease file name
are read from the source of `origin_catalogue.schema` and `artifact_publication.jobs` with
`ast`, which needs no dependencies. Exits 1 with an `::error::` line when the document
carries no ORIGIN_CATALOGUE_URI value or no task timeout.
"""

import ast
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "packages" / "dfo-data-tools" / "src"
_SCHEMA = _SRC / "origin_catalogue" / "schema.py"
_JOBS = _SRC / "artifact_publication" / "jobs.py"


def _constants(path: Path) -> dict[str, str | int]:
    """The module's top-level constants: literals, and f-strings over earlier constants."""
    found: dict[str, str | int] = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if not isinstance(target, ast.Name):
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, str | int):
            found[target.id] = value.value
        elif isinstance(value, ast.JoinedStr):
            found[target.id] = "".join(
                str(found[part.value.id])
                if isinstance(part, ast.FormattedValue) and isinstance(part.value, ast.Name)
                else str(part.value)
                for part in value.values
                if isinstance(part, ast.Constant | ast.FormattedValue)
            )
    return found


def lease_uri(base_uri: str) -> str:
    """The lease object under `base_uri` for the checked-out format."""
    format_root = _constants(_SCHEMA)["FORMAT_ROOT"]
    lease_file = _constants(_JOBS)["LEASE_FILE"]
    return f"{base_uri.rstrip('/')}/{format_root}/{lease_file}"


def main() -> int:
    job = json.load(sys.stdin)
    task = job["spec"]["template"]["spec"]["template"]["spec"]
    env = task["containers"][0].get("env") or []
    base_uri = next(
        (item.get("value") for item in env if item.get("name") == "ORIGIN_CATALOGUE_URI"), None
    )
    if not base_uri:
        print(
            "::error::The origin catalogue job has no ORIGIN_CATALOGUE_URI value.", file=sys.stderr
        )
        return 1
    timeout = task.get("timeoutSeconds")
    if not timeout:
        print("::error::The origin catalogue job reports no task timeout.", file=sys.stderr)
        return 1
    print(f"lease={lease_uri(base_uri)}")
    print(f"task_timeout={int(timeout)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
