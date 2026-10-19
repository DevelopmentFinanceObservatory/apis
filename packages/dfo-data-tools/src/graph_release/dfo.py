"""The DFO layer: built from the platform parts, `Declarations` and the base parties.

`build_dfo_parts` returns each DFO part's bytes. `declarations` is `declarations_json`.
`code_index_blocks` is the code index's graph blocks, built by `code_index.export` over
`ReleaseGraphReader`, a reader over the `edges` and `nodes` parts. Where the export script joins
a published coverage build, these blocks take each variable's provenances, the listed datasets'
variables and the parties from `coverage_rows` over `series`, so the code index and coverage
read one release. The blocks carry no stamp; a reader stamps them from the manifest.

The parties block needs each donor's and recipient's type and name. DFO's Spanner holds them for
organisations but not for the countries of the base Data Commons graph, so the builder reads the
parties the release does not answer (`base_party_ids`) from the mixer once
(`read_base_parties`) and stores the answer as the `base_parties` part. `export_parties` reads
the release first and `base_parties` for the rest; every other block reads the release alone.
Every part serialises deterministically, so a rebuild from the stored parts gives the same
bytes with no mixer call.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from code_index.export import (
    ExportRefused,
    PartyPublication,
    export_graph_blocks,
    export_parties,
    index_bytes,
    listed_dcids,
    verify_complete_systems,
)
from dc_graph.visible_arcs import mixer_target, visible_mask
from graph_release.coverage import coverage_rows, role_map
from graph_release.declarations import Declarations
from graph_release.format import StagedPart, write_blob_part
from graph_release.platform import PlatformParts, PlatformRefusal
from series_coverage.schema import ABSENT_ENTITY, DONOR_COLUMN, FACET_COLUMN, RECIPIENT_COLUMN

if TYPE_CHECKING:
    from dc_graph import SubjectPage
    from dfo_platform.roles import RoleAxis

_TYPE_OF: Final[str] = "typeOf"
_NAME: Final[str] = "name"
_PARTY_PROPERTIES: Final[tuple[str, ...]] = (_NAME, _TYPE_OF)
_PARTY_AXES: Final[Mapping[str, RoleAxis]] = {
    DONOR_COLUMN: "origin",
    RECIPIENT_COLUMN: "destination",
}
_SOURCES: Final[Mapping[str, str]] = {
    "declarations": "graph_release.dfo.declarations",
    "code_index_blocks": "graph_release.dfo.code_index_blocks",
    "base_parties": "mixer.node",
}


class ArcReader(Protocol):
    """The one read `read_base_parties` makes; `dc_graph.GraphClient` implements it."""

    def node_arcs_many(
        self, dcids: Sequence[str], *, properties: Sequence[str]
    ) -> dict[str, dict[str, tuple[str, ...]] | None]: ...


@dataclass(frozen=True)
class DfoParts:
    """The bytes of each DFO part."""

    declarations: bytes
    code_index_blocks: bytes
    base_parties: bytes

    def blobs(self) -> Mapping[str, bytes]:
        """Each part's bytes by part name, as `graph_release.format.JSON_PARTS` names them."""
        return {
            "declarations": self.declarations,
            "code_index_blocks": self.code_index_blocks,
            "base_parties": self.base_parties,
        }


@dataclass(frozen=True)
class _Coverage:
    """The coverage view of a release and each facet's provenance."""

    rows: pa.Table
    facet_provenances: Mapping[str, str]


def base_party_ids(platform: PlatformParts, declarations: Declarations) -> tuple[str, ...]:
    """The parties of the declared and listed datasets whose type the release does not answer,
    sorted: the ids `read_base_parties` reads."""
    coverage = _coverage(platform, declarations)
    parties = _parties(coverage, _party_datasets(declarations))
    reader = ReleaseGraphReader(edges=platform.edges, nodes=platform.nodes)
    answered = _answered(reader, sorted(parties))
    return tuple(party for party in sorted(parties) if party not in answered)


def read_base_parties(party_ids: Sequence[str], *, graph: ArcReader) -> bytes:
    """Each party's `typeOf` and `name` as `graph` answers them, as JSON with sorted keys and
    sorted values: `null` for a party the graph does not hold. A read error propagates."""
    ids = sorted(set(party_ids))
    arcs = graph.node_arcs_many(ids, properties=list(_PARTY_PROPERTIES))
    document = {
        party: None
        if (found := arcs.get(party)) is None
        else {prop: sorted(set(found.get(prop, ()))) for prop in _PARTY_PROPERTIES}
        for party in ids
    }
    return (json.dumps(document, sort_keys=True, ensure_ascii=False, indent=1) + "\n").encode(
        "utf-8"
    )


def build_dfo_parts(
    platform: PlatformParts, declarations: Declarations, base_parties: bytes
) -> DfoParts:
    """The DFO parts of `platform` under `declarations`, with `base_parties` (`read_base_parties`)
    answering the parties the release does not.

    Raises `ValueError` for declarations with no role axes, code systems or datasets,
    and `PlatformRefusal` where `role_map` or `coverage_rows` refuses, or with code
    `code_index_export_refused` where the code-index export refuses the graph, as for a party
    neither the release nor `base_parties` answers.
    """
    empty = [
        name
        for name in ("role_axes", "code_systems", "datasets")
        if not getattr(declarations, name)
    ]
    if empty:
        raise ValueError(f"declarations hold no {', '.join(empty)}; the DFO layer needs them")

    coverage = _coverage(platform, declarations)
    try:
        blocks = _code_index_blocks(platform, declarations, coverage, json.loads(base_parties))
    except ExportRefused as exc:
        raise PlatformRefusal("code_index_export_refused", str(exc)) from exc
    return DfoParts(
        declarations=declarations.declarations_json(),
        code_index_blocks=index_bytes(blocks),
        base_parties=base_parties,
    )


def _code_index_blocks(
    platform: PlatformParts,
    declarations: Declarations,
    coverage: _Coverage,
    base_parties: Mapping[str, dict | None],
) -> dict:
    """The code index's blocks: the graph blocks over the release, provenances and parties from
    `coverage`, party types and names from the release or `base_parties`."""
    provenances = _variable_provenances(coverage)
    reader = ReleaseGraphReader(edges=platform.edges, nodes=platform.nodes)
    blocks = export_graph_blocks(
        reader,
        code_systems=declarations.code_systems,
        measure_axes=declarations.measure_axes,
        measure_partition_labels=declarations.measure_partition_labels,
        measure_nouns=declarations.measure_nouns,
        datasets=declarations.datasets,
        party_type_kinds=declarations.party_type_kinds,
        listed=listed_dcids(provenances, declarations.listed_datasets),
        shape_slot_defaults=declarations.shape_slot_defaults,
        partly_stated_values=declarations.partly_stated_values,
        measure_classes=declarations.measure_classes,
        variable_aliases=declarations.variable_aliases,
        stat_type_words=declarations.stat_type_words,
    )
    for block in ("statvars", "listed_statvars"):
        for row in blocks[block]:
            row["provenances"] = list(provenances.get(row["dcid"], ()))
    verify_complete_systems(blocks["statvars"], declarations.code_systems)
    blocks["parties"] = export_parties(
        _PartyReader(reader, base=base_parties),
        _parties(coverage, _party_datasets(declarations)),
        type_kinds=declarations.party_type_kinds,
        kind_overrides=declarations.party_kind_overrides,
        total_groups=declarations.total_groups,
    )
    return blocks


def stage_dfo_parts(parts: DfoParts, *, staging_dir: Path, execution_id: str) -> list[StagedPart]:
    """Writes every DFO part to `staging_dir`."""
    return [
        write_blob_part(
            data,
            staging_dir=staging_dir,
            name=name,
            source=_SOURCES[name],
            produced_by_execution=execution_id,
        )
        for name, data in parts.blobs().items()
    ]


def _coverage(platform: PlatformParts, declarations: Declarations) -> _Coverage:
    return _Coverage(
        rows=coverage_rows(platform.series, role_map(platform.edges, declarations.role_axes)),
        facet_provenances=_facet_provenances(platform.facets),
    )


def _party_datasets(declarations: Declarations) -> set[str]:
    return {dataset.provenance for dataset in declarations.datasets} | set(
        declarations.listed_datasets
    )


def _answered(reader: ReleaseGraphReader, dcids: Sequence[str]) -> set[str]:
    """The dcids whose type `reader` answers."""
    arcs = reader.node_arcs_many(dcids, properties=[_TYPE_OF])
    return {dcid for dcid, found in arcs.items() if found and found[_TYPE_OF]}


def _facet_provenances(facets: pa.Table) -> dict[str, str]:
    return dict(
        zip(
            cast("list[str]", _strings(facets["facet_id"]).to_pylist()),
            cast("list[str]", _strings(facets["provenance"]).to_pylist()),
            strict=True,
        )
    )


def _variable_provenances(coverage: _Coverage) -> dict[str, tuple[str, ...]]:
    """Each covered variable's sorted provenances, from its rows' facets."""
    pairs = coverage.rows.group_by(["variable", FACET_COLUMN], use_threads=False).aggregate([])
    found: dict[str, set[str]] = {}
    for variable, facet_id in zip(
        cast("list[str]", _strings(pairs["variable"]).to_pylist()),
        cast("list[str]", _strings(pairs[FACET_COLUMN]).to_pylist()),
        strict=True,
    ):
        found.setdefault(variable, set()).add(coverage.facet_provenances[facet_id])
    return {variable: tuple(sorted(provenances)) for variable, provenances in found.items()}


def _parties(coverage: _Coverage, datasets: set[str]) -> dict[str, PartyPublication]:
    """Every donor and recipient on a row of `datasets`, with the datasets that publish it and
    the flow axes they publish it on: `origin` as a donor, `destination` as a recipient."""
    publishers: dict[str, set[str]] = {}
    axes: dict[str, set[RoleAxis]] = {}
    for column, axis in _PARTY_AXES.items():
        pairs = coverage.rows.group_by([column, FACET_COLUMN], use_threads=False).aggregate([])
        for party, facet_id in zip(
            cast("list[str]", _strings(pairs[column]).to_pylist()),
            cast("list[str]", _strings(pairs[FACET_COLUMN]).to_pylist()),
            strict=True,
        ):
            provenance = coverage.facet_provenances[facet_id]
            if party == ABSENT_ENTITY or provenance not in datasets:
                continue
            publishers.setdefault(party, set()).add(provenance)
            axes.setdefault(party, set()).add(axis)
    return {
        party: PartyPublication(datasets=frozenset(found), axes=frozenset(axes[party]))
        for party, found in publishers.items()
    }


class ReleaseGraphReader:
    """`code_index.export.ExportReader` over the `edges` and `nodes` parts, answering as the
    mixer's `/v2/node` reads do after `dc_graph.visible_arcs.visible_mask`.

    A node is held when it has an edge or its `Node` row carries a name, value or types. A
    target answers as `visible_arcs.mixer_target` names it. A node whose `typeOf` or `name` has
    no edge answers from its `Node` row's types or name, as the mixer does for the nodes DFO
    authors. Values come distinct and sorted.
    """

    def __init__(self, *, edges: pa.Table, nodes: pa.Table) -> None:
        self._ids = _strings(nodes["subject_id"])
        self._id_list = cast("list[str]", self._ids.to_pylist())
        self._code = {dcid: code for code, dcid in enumerate(self._id_list)}
        names = cast("list[str | None]", nodes["name"].to_pylist())
        values = cast("list[str | None]", nodes["value"].to_pylist())
        types = cast("list[list[str] | None]", nodes["types"].to_pylist())
        self._names = names
        self._types = [tuple(sorted(found or ())) for found in types]
        self._targets = [
            mixer_target(dcid, value=value, types=found)
            for dcid, value, found in zip(self._id_list, values, self._types, strict=True)
        ]
        self._described = [
            name is not None or value is not None or bool(found)
            for name, value, found in zip(names, values, self._types, strict=True)
        ]

        visible = edges.filter(
            visible_mask(_strings(edges["predicate"]), _strings(edges["object_id"]))
        )
        predicates = _strings(visible["predicate"])
        provenances = _strings(visible["provenance"])
        self._predicates = cast("list[str]", pc.unique(predicates).sort().to_pylist())
        self._provenances = cast("list[str]", pc.unique(provenances).sort().to_pylist())
        self._predicate_code = {name: code for code, name in enumerate(self._predicates)}
        s = self._codes(_strings(visible["subject_id"]), self._ids)
        p = self._codes(predicates, pa.array(self._predicates, pa.string()))
        o = self._codes(_strings(visible["object_id"]), self._ids)
        v = self._codes(provenances, pa.array(self._provenances, pa.string()))
        order = np.lexsort((v, o, p, s))
        self._s, self._p, self._o, self._v = s[order], p[order], o[order], v[order]
        self._starts = np.searchsorted(self._s, np.arange(len(self._id_list) + 1))
        reverse_key = self._p * len(self._id_list) + self._o
        reverse = np.argsort(reverse_key, kind="stable")
        self._reverse_key = reverse_key[reverse]
        self._reverse_subject = self._s[reverse]

    @staticmethod
    def _codes(column: pa.Array, value_set: pa.Array) -> np.ndarray:
        codes = pc.index_in(column, value_set=value_set)
        if codes.null_count:
            missing = column.filter(pc.is_null(codes)).unique().to_pylist()[:5]
            raise ValueError(f"edges name nodes the nodes part lacks: {missing}")
        return codes.to_numpy().astype(np.int64)

    def _held(self, dcid: str) -> int | None:
        code = self._code.get(dcid)
        if code is None:
            return None
        if self._starts[code + 1] > self._starts[code] or self._described[code]:
            return code
        return None

    def _subjects(self, property: str, dcid: str) -> np.ndarray:
        predicate, code = self._predicate_code.get(property), self._code.get(dcid)
        if predicate is None or code is None:
            return np.empty(0, np.int64)
        key = predicate * len(self._id_list) + code
        low, high = np.searchsorted(self._reverse_key, [key, key + 1])
        return self._reverse_subject[low:high]

    def node_subjects(
        self,
        dcid: str,
        *,
        property: str,
        filters: Mapping[str, str] | None = None,
        max_results: int,
    ) -> SubjectPage:
        from dc_graph import SubjectPage

        subjects = np.unique(self._subjects(property, dcid))
        for key, value in sorted((filters or {}).items()):
            subjects = np.intersect1d(subjects, self._subjects(key, value))
        dcids = sorted(self._id_list[int(code)] for code in subjects)
        return SubjectPage(dcids=tuple(dcids[:max_results]), truncated=len(dcids) > max_results)

    def node_arcs_many(
        self, dcids: Sequence[str], *, properties: Sequence[str]
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        out: dict[str, dict[str, tuple[str, ...]] | None] = {}
        for dcid in dcids:
            code = self._held(dcid)
            if code is None:
                out[dcid] = None
                continue
            low, high = self._starts[code], self._starts[code + 1]
            by_predicate: dict[str, set[str]] = {}
            for predicate, target in zip(
                self._p[low:high].tolist(), self._o[low:high].tolist(), strict=True
            ):
                by_predicate.setdefault(self._predicates[predicate], set()).add(
                    self._targets[target]
                )
            arcs = {}
            for prop in properties:
                found = tuple(sorted(by_predicate.get(prop, ())))
                if not found and prop == _TYPE_OF:
                    found = self._types[code]
                if not found and prop == _NAME and self._names[code]:
                    found = (cast("str", self._names[code]),)
                arcs[prop] = found
            out[dcid] = arcs
        return out

    def node_arcs_batch(self, dcids: list[str]) -> Mapping[str, dict | None]:
        out: dict[str, dict | None] = {}
        for dcid in dcids:
            code = self._held(dcid)
            if code is None:
                out[dcid] = None
                continue
            low, high = self._starts[code], self._starts[code + 1]
            arcs: dict[str, dict] = {}
            for predicate, target, provenance in zip(
                self._p[low:high].tolist(),
                self._o[low:high].tolist(),
                self._v[low:high].tolist(),
                strict=True,
            ):
                answer = self._targets[target]
                entry = {"dcid": answer} if answer == self._id_list[target] else {"value": answer}
                entry["provenanceId"] = self._provenances[provenance]
                arcs.setdefault(self._predicates[predicate], {"nodes": []})["nodes"].append(entry)
            out[dcid] = arcs
        return out


class _PartyReader:
    """The reader `export_parties` reads: the release for a dcid whose type it answers, the
    stored `base_parties` answer for any other."""

    def __init__(self, release: ReleaseGraphReader, *, base: Mapping[str, dict | None]) -> None:
        self._release = release
        self._base = base

    def node_subjects(
        self,
        dcid: str,
        *,
        property: str,
        filters: Mapping[str, str] | None = None,
        max_results: int,
    ) -> SubjectPage:
        return self._release.node_subjects(
            dcid, property=property, filters=filters, max_results=max_results
        )

    def node_arcs_many(
        self, dcids: Sequence[str], *, properties: Sequence[str]
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        answered = _answered(self._release, dcids)
        out = self._release.node_arcs_many(
            [dcid for dcid in dcids if dcid in answered], properties=properties
        )
        for dcid in dcids:
            if dcid in answered:
                continue
            found = self._base.get(dcid)
            out[dcid] = (
                None if found is None else {prop: tuple(found.get(prop, ())) for prop in properties}
            )
        return out

    def node_arcs_batch(self, dcids: list[str]) -> Mapping[str, dict | None]:
        return self._release.node_arcs_batch(dcids)


def _strings(column: pa.ChunkedArray) -> pa.Array:
    """`column` as one string array, decoded where it is dictionary-encoded."""
    array = column.combine_chunks()
    return array.cast(pa.string()) if pa.types.is_dictionary(array.type) else array


__all__ = [
    "ArcReader",
    "DfoParts",
    "ReleaseGraphReader",
    "base_party_ids",
    "build_dfo_parts",
    "read_base_parties",
    "stage_dfo_parts",
]
