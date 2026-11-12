"""Where the tests find the spec-kit catalogue: `DFO_SPEC_KIT_DIR`, else a sibling checkout,
and a release whose line chart series cap is another (`with_line_chart_cap`)."""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from assembler.catalogue import CatalogueRelease

_BUNDLE_RELATIVE_PATH = Path("python/dfo-spec-kit-components/src/dfo_spec_kit_components/catalogue")


def _spec_kit_dir() -> Path:
    env_dir = os.environ.get("DFO_SPEC_KIT_DIR")
    if env_dir:
        return Path(env_dir)
    # Sibling checkout: <repo root>/../dfo-spec-kit, resolved from this file's
    # own path so it works regardless of the invoking cwd.
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root.parent / "dfo-spec-kit"


def resolved_catalogue_dir() -> Path:
    return _spec_kit_dir() / _BUNDLE_RELATIVE_PATH


def with_line_chart_cap(release: CatalogueRelease, cap: int) -> CatalogueRelease:
    """`release` with its `line_chart` select array capped at `cap` in both intent schemas,
    as an earlier release with another cap holds it, so `line_chart_max_series` is `cap`."""

    def capped(schema: Mapping[str, Any]) -> dict[str, Any]:
        select = schema["properties"]["select"]
        values = select["additionalProperties"]
        branches = [
            {**branch, "maxItems": cap} if branch.get("type") == "array" else branch
            for branch in values["anyOf"]
        ]
        return {
            **schema,
            "properties": {
                **schema["properties"],
                "select": {**select, "additionalProperties": {**values, "anyOf": branches}},
            },
        }

    return dataclasses.replace(
        release,
        intent_schemas={
            **release.intent_schemas,
            "line_chart": capped(release.intent_schemas["line_chart"]),
        },
        intent_schemas_tolerant={
            **release.intent_schemas_tolerant,
            "line_chart": capped(release.intent_schemas_tolerant["line_chart"]),
        },
    )
