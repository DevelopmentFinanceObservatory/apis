"""A PATH shim standing in for the `gcloud run` calls the deploy scripts make, over a state file.

The state holds the apis service as `gcloud run services describe --format=json` prints it,
each revision's image, and each job's image. `jobs update --image` writes the job's image back,
so a test can run one script after another against the state the first one left.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

_SHIM = """#!/usr/bin/env python3
import json, sys
from pathlib import Path

state_path = Path({state!r})
state = json.loads(state_path.read_text())
args = sys.argv[1:]
state["calls"].append(" ".join(a for a in args if not a.startswith(("--region", "--project"))))
state_path.write_text(json.dumps(state))
if state.get("fail"):
    sys.exit(1)
kind, verb, name = args[1], args[2], args[3]
if (kind, verb) == ("services", "describe"):
    service = state["service"]
    print(service if isinstance(service, str) else json.dumps(service))
elif (kind, verb) == ("revisions", "describe"):
    print(state["revisions"].get(name, ""))
elif (kind, verb) == ("jobs", "describe"):
    if name not in state["jobs"]:
        sys.exit(1)
    print(state["jobs"][name])
elif (kind, verb) == ("jobs", "update"):
    image = next(a.split("=", 1)[1] for a in args if a.startswith("--image="))
    state["jobs"][name] = image
    state_path.write_text(json.dumps(state))
else:
    sys.exit(2)
"""


def install(
    bin_dir: Path,
    *,
    service: dict,
    revisions: dict[str, str],
    jobs: dict[str, str],
    fail: bool = False,
) -> Path:
    """Writes the state and a `gcloud` shim reading it into `bin_dir`, and returns the state
    file."""
    bin_dir.mkdir(exist_ok=True)
    state = bin_dir / "state.json"
    state.write_text(
        json.dumps(
            {"service": service, "revisions": revisions, "jobs": jobs, "fail": fail, "calls": []}
        )
    )
    shim = bin_dir / "gcloud"
    shim.write_text(_SHIM.format(state=str(state)))
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    return state


def read(state: Path) -> dict:
    return json.loads(state.read_text())
