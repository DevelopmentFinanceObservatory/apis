"""Checks between the artifacts a request binds: the origin catalogue, the coverage build
and the packaged code index.

`pair_stale_warning` compares the ingestion the catalogue and coverage builds read.
`index_drift_warning` compares the catalogue's arcs with the code index's rows for the
StatVars an answer serves. Each logs one WARNING line, `artifact_pair_stale` or
`catalogue_index_drift` followed by a JSON object, whenever it finds a disagreement, and
returns the response warning unless `CatalogueSettings` silences it. Neither fails a
request or readiness: the code index's compatibility pointer stays the one hard gate.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from typing import Final, Literal

from code_index.query import CodeIndex
from code_index.types import StatVarRow
from dc_graph import IMPORT_ADDED_CONSTRAINT_PROPERTIES
from origin_catalogue import Catalogue
from origin_catalogue.un_mirror import UnArc
from search.engine.region.codes import ARTIFACT_PAIR_STALE, CATALOGUE_INDEX_DRIFT
from search.models import (
    CandidatesPart,
    DefinitePart,
    MembersNotStated,
    Resolution,
    ResolveResponse,
    Spec,
    Warning,
)
from series_coverage.schema import CoverageManifest

logger = logging.getLogger(__name__)

type DriftAspect = Literal["five_tuple", "constraints", "roles"]

_FIVE_TUPLE_FIELDS: Final[tuple[tuple[str, str], ...]] = (
    ("populationType", "population_type"),
    ("measuredProperty", "measured_property"),
    ("statType", "stat_type"),
    ("measurementQualifier", "measurement_qualifier"),
    ("measurementDenominator", "measurement_denominator"),
)
"""Each five-tuple arc beside the `StatVarRow` field the code index export reads it into."""

_NAMED_IN_MESSAGE: Final[int] = 5
"""How many drifted StatVars a warning's message names before counting the rest."""


def pair_stale_warning(
    catalogue: Catalogue, coverage: CoverageManifest, *, warn: bool
) -> Warning | None:
    """The `artifact_pair_stale` warning when `catalogue` and `coverage` record different
    ingestions, else `None`. A build that records none is unknown and never warns.

    A mismatch logs one WARNING line whatever `warn` says; `warn=False` returns `None`.
    """
    ours, theirs = catalogue.ingestion, coverage.ingestion
    if ours is None or theirs is None or ours.workflow_execution_id == theirs.workflow_execution_id:
        return None
    logger.warning(
        "artifact_pair_stale %s",
        json.dumps(
            {
                "catalogue_build": catalogue.build_id,
                "catalogue_ingestion": ours.workflow_execution_id,
                "catalogue_ingested_at": ours.creation_timestamp,
                "coverage_build": coverage.build_id,
                "coverage_ingestion": theirs.workflow_execution_id,
                "coverage_ingested_at": theirs.creation_timestamp,
            }
        ),
    )
    if not warn:
        return None
    return Warning(
        code=ARTIFACT_PAIR_STALE,
        severity="warn",
        message=(
            f"The origin catalogue build {catalogue.build_id!r} read ingestion "
            f"{ours.workflow_execution_id!r} of {ours.creation_timestamp}, and the coverage "
            f"build {coverage.build_id!r} read ingestion {theirs.workflow_execution_id!r} of "
            f"{theirs.creation_timestamp}. Until both rebuild from one ingestion, this answer "
            "may miss data the newer one added."
        ),
    )


def index_drift(row: StatVarRow, arcs: Iterable[UnArc]) -> tuple[DriftAspect, ...]:
    """Where `arcs`, a StatVar's arcs in the catalogue, disagree with its code index
    `row`: its five-tuple, its constraint properties and their values, or its roles
    (`observationProperties`). Arc values compare as sets, and the constraint list leaves
    out `IMPORT_ADDED_CONSTRAINT_PROPERTIES`, as the graph client and the export do."""
    values: dict[str, set[str]] = {}
    for arc in arcs:
        target = arc.dcid or arc.value
        if target is not None:
            values.setdefault(arc.property, set()).add(target)

    def held(property: str) -> set[str]:
        return values.get(property, set())

    aspects: list[DriftAspect] = []
    if any(
        held(property) != ({value} if (value := getattr(row, field)) else set())
        for property, field in _FIVE_TUPLE_FIELDS
    ):
        aspects.append("five_tuple")
    constraints = held("constraintProperties") - IMPORT_ADDED_CONSTRAINT_PROPERTIES
    if constraints != set(row.constraint_properties) or any(
        held(property) != {value} for property, value in row.constraint_properties.items()
    ):
        aspects.append("constraints")
    if held("observationProperties") != set(row.observation_properties):
        aspects.append("roles")
    return tuple(aspects)


def _served_specs(response: ResolveResponse) -> Iterator[Spec]:
    if not isinstance(response, Resolution):
        return
    for part in response.parts:
        if isinstance(part, DefinitePart):
            yield part.spec
        elif isinstance(part, CandidatesPart):
            yield from part.candidates.specs


def _served_variables(response: ResolveResponse) -> tuple[str, ...]:
    """The StatVars `response`'s definite and candidate Specs serve, in first-seen order. A
    Spec rebuilt from its wire form, as a cached response's are, states none."""
    variables: dict[str, None] = {}
    for spec in _served_specs(response):
        try:
            members = spec.resolved.members
        except MembersNotStated:
            continue
        variables.update((member.ref.ref, None) for member in members)
    return tuple(variables)


def index_drift_warning(
    response: ResolveResponse, *, catalogue: Catalogue, index: CodeIndex, warn: bool
) -> Warning | None:
    """The `catalogue_index_drift` warning when, for a StatVar `response` serves that both
    `index` lists and `catalogue`'s arc tables hold, the two disagree (`index_drift`), else
    `None`. A StatVar the arc tables do not hold is out of their scope and proves nothing.

    Drift logs one WARNING line whatever `warn` says; `warn=False` returns `None`.
    """
    drifted: dict[str, tuple[DriftAspect, ...]] = {}
    for variable in _served_variables(response):
        row = index.statvar_by_dcid(variable)
        arcs = catalogue.held_arcs(variable)
        if row is None or arcs is None:
            continue
        if aspects := index_drift(row, arcs):
            drifted[variable] = aspects
    if not drifted:
        return None
    logger.warning(
        "catalogue_index_drift %s",
        json.dumps(
            {
                "catalogue_build": catalogue.build_id,
                "index_hash": index.stamp.content_hash,
                "variables": drifted,
            }
        ),
    )
    if not warn:
        return None
    named = [
        f"{variable} ({', '.join(aspect.replace('_', '-') for aspect in aspects)})"
        for variable, aspects in list(drifted.items())[:_NAMED_IN_MESSAGE]
    ]
    rest = len(drifted) - len(named)
    return Warning(
        code=CATALOGUE_INDEX_DRIFT,
        severity="warn",
        message=(
            f"The origin catalogue build {catalogue.build_id!r} and the code index "
            f"{index.stamp.content_hash!r} disagree on {len(drifted)} StatVar(s) this answer "
            f"serves: {', '.join(named)}{f' and {rest} more' if rest else ''}."
        ),
    )


def with_warning(response: ResolveResponse, warning: Warning | None) -> ResolveResponse:
    """`response` with `warning` appended to its diagnostics, or `response` itself when
    `warning` is `None`."""
    if warning is None:
        return response
    diagnostics = response.diagnostics
    return response.model_copy(
        update={
            "diagnostics": diagnostics.model_copy(
                update={"warnings": [*diagnostics.warnings, warning]}
            )
        }
    )
