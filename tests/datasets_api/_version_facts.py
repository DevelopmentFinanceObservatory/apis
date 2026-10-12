"""The facts migration `0031_version_record_facts_backfill.sql` backfills from a stored version body.

`facts_from_body` projects a body onto the fact columns key by key, as the migration's
`UPDATE` does: a body holding `prepared_at` is a dataset version's, and one holding
`input_content_versions` a published frame's. `version_facts_golden.json` holds the bodies the
golden test projects, and the live backfill test applies the migration to the same bodies.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from dataset_store.snapshot import FrameLineage, ManifestFacts, VersionFacts

GOLDEN_PATH = Path(__file__).with_name("version_facts_golden.json")


def golden_cases() -> dict[str, Any]:
    return json.loads(GOLDEN_PATH.read_text())


def facts_from_body(body: Mapping[str, object]) -> VersionFacts:
    if "prepared_at" in body:
        return ManifestFacts(
            prepared_at=datetime.fromisoformat(cast(str, body["prepared_at"])).astimezone(UTC),
            definition=cast("Mapping[str, object]", body["definition"]),
            coverage=cast("Mapping[str, object]", body["coverage"]),
            citation=cast("Mapping[str, object]", body["citation"]),
            trace=cast("Mapping[str, object]", body["trace"]),
            warnings=cast("list[Mapping[str, object]]", body.get("warnings") or []),
            series_count=cast("int | None", body.get("series_count")),
            series_coverage_stamp=cast("str | None", body.get("series_coverage_stamp")),
            output_profile=cast("Mapping[str, object] | None", body.get("output_profile")),
            series_conversions=cast(
                "list[Mapping[str, object]] | None", body.get("series_conversions")
            ),
            default_scope_plural=cast("bool | None", body.get("default_scope_plural")),
        )
    if "input_content_versions" in body:
        return FrameLineage(
            input_content_versions=cast("list[str]", body["input_content_versions"]),
            composed_stat_vars=cast(
                "list[Mapping[str, object]] | None", body.get("composed_stat_vars")
            ),
        )
    raise ValueError(f"a body with keys {sorted(body)} holds no facts")
