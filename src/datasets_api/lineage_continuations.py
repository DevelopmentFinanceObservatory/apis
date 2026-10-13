"""The continuations a computed version inherits from the versions it was computed from.

A computed (`ops_`) frame keeps its roots' method cells, so a series a root's source
estimates continue reaches it as two methods. `derived_continuations` gives the version the
fact that groups them back into one series, stored with its lineage
(`dataset_store.snapshot.FrameLineage.continuations`) by the publish door and by the
`backfill-descriptions` task:

- a dataset input gives what its trace records, scoped to the method and source its series
  were held under (`materialise.continued_series.continued_series_of_trace`);
- a computed input gives the entries it stores, unchanged, so a continuation keeps its root's
  scope through any chain of computations;
- a computed input recorded before versions stored them gives what its own stored lineage
  derives, by the same rule.

The inputs are the versions a computed version's `retrieve` steps name, each with its id, or,
for a version without steps, the content versions its lineage lists, under any subject. An
input the store does not hold gives nothing. Entries keep their first appearance, in input
order, without repeats.
"""

from __future__ import annotations

from collections.abc import Sequence

from dataset_store.snapshot import FrameLineage, ManifestFacts, SnapshotReader, StoredVersion
from description.facts import InputVersion
from materialise.continued_series import (
    ContinuedSeries,
    continued_series_of_entries,
    continued_series_of_trace,
)

__all__ = ["derived_continuations", "lineage_inputs"]


def lineage_inputs(
    *, steps_inputs: Sequence[InputVersion] | None, lineage: FrameLineage
) -> list[InputVersion]:
    """The inputs a computed version's continuations derive from: those its `retrieve` steps
    name (`description_record.computed_inputs`), or, without steps, each content version its
    lineage lists, with an unknown id."""
    if steps_inputs is not None:
        return list(steps_inputs)
    return [InputVersion(id=None, content_version=cv) for cv in lineage.input_content_versions]


def derived_continuations(
    inputs: Sequence[InputVersion], *, reader: SnapshotReader
) -> list[dict[str, object]]:
    """The stored entries (`ContinuedSeries.as_entry`) of the continuations `inputs` carry,
    read through `reader` one batch per level of lineage a pre-continuation computed input
    makes it walk."""
    found: dict[ContinuedSeries, None] = {}
    seen: set[str] = set()
    pending = list(inputs)
    while pending:
        seen.update(one.content_version for one in pending)
        held = reader.read_versions_of([one.content_version for one in pending])
        walk: list[InputVersion] = []
        for one in pending:
            for version in _matching(held, one):
                series, deeper = _carried(version)
                found.update(dict.fromkeys(series))
                walk.extend(cv for cv in deeper if cv.content_version not in seen)
        pending = list({cv.content_version: cv for cv in walk}.values())
    return [series.as_entry() for series in found]


def _matching(held: Sequence[StoredVersion], one: InputVersion) -> list[StoredVersion]:
    """The stored versions `one` names: its pair, or every subject at its content version
    when its id is unknown. Sorted by subject so the order is the store's own."""
    return sorted(
        (
            version
            for version in held
            if version.content_version == one.content_version
            and (one.id is None or version.subject_id == one.id)
        ),
        key=lambda version: version.subject_id,
    )


def _carried(version: StoredVersion) -> tuple[tuple[ContinuedSeries, ...], list[InputVersion]]:
    """What `version` carries, and the inputs still to read for a computed version that stores
    no continuations."""
    facts = version.facts
    if isinstance(facts, ManifestFacts):
        return continued_series_of_trace(facts.trace), []
    if facts.continuations is not None:
        return continued_series_of_entries(facts.continuations), []
    return (), lineage_inputs(steps_inputs=None, lineage=facts)
