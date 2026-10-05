"""`CodeIndex`: the in-memory, queryable catalogue built from parsed rows and
declarations. No graph access and no file I/O here -- `loader.load_index`
is the file-reading entry point that constructs one of these from a
committed JSON export.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

from code_index.types import (
    AbsentObservation,
    Bound,
    BoundSet,
    CodeRow,
    CodeSystemDeclaration,
    ConstraintQuery,
    DatasetDeclaration,
    IndexStamp,
    LevelOption,
    StatVarRow,
)
from statvar_members import FiveTuple


class DuplicateDcid(ValueError):
    """Two rows -- code or StatVar -- in the same index share one dcid, or
    two code rows in the same code system share one code string."""


class CyclicParent(ValueError):
    """A code row's `parent` chain revisits a row already in that chain --
    a self-parent or a longer cycle. `descendants`'s breadth-first walk
    follows this same relation, so a cycle here would otherwise grow its
    result without bound."""


class CrossSystemParent(ValueError):
    """A code row's `parent` names a row in a different `code_system`.
    `specializationOf` orders one code system's own codes, so a parent from
    another system would walk `descendants`/`covering_roots` outside the
    vocabulary and roll that other system's members up as this one's
    siblings."""


class InvalidDeclaredDefault(ValueError):
    """A default code belongs outside its declaring system, or a `when_bound_under` code is
    missing or belongs to that system. Defaults scope over another system's slot."""


def row_five_tuple(row: StatVarRow) -> FiveTuple:
    """Returns the row's five fields as a resolved `FiveTuple`.

    Exported rows set `denominator_unresolved=False`."""
    return FiveTuple(
        population_type=row.population_type,
        measured_property=row.measured_property,
        stat_type=row.stat_type,
        measurement_qualifier=row.measurement_qualifier,
        measurement_denominator=row.measurement_denominator,
    )


def _datasets_by_five_tuple(
    statvars: Iterable[StatVarRow], datasets: Mapping[str, DatasetDeclaration]
) -> dict[FiveTuple, tuple[str, ...]]:
    """Maps each five-tuple to the datasets publishing any of its rows, lowest
    `DatasetDeclaration.rank` first. A provenance no dataset declares follows the declared
    ones, and provenance breaks a tie. A row with no provenance belongs to no dataset."""
    publishers: dict[FiveTuple, set[str]] = defaultdict(set)
    for row in statvars:
        publishers[row_five_tuple(row)].update(row.provenances)

    def serving_order(provenance: str) -> tuple[bool, int, str]:
        declaration = datasets.get(provenance)
        return (declaration is None, declaration.rank if declaration else 0, provenance)

    return {
        five_tuple: tuple(sorted(provenances, key=serving_order))
        for five_tuple, provenances in publishers.items()
        if provenances
    }


def _dataset_pins(
    statvars: Iterable[StatVarRow], code_systems: Iterable[str]
) -> dict[tuple[str, FiveTuple], dict[str, str]]:
    """Maps each dataset and five-tuple it publishes to the code-system properties that
    every one of the dataset's rows of the five-tuple carries with one value, and that
    value. Measure axes are not declared code systems, so no measure axis is pinned."""
    rows_by_key: dict[tuple[str, FiveTuple], list[StatVarRow]] = defaultdict(list)
    for row in statvars:
        five_tuple = row_five_tuple(row)
        for provenance in row.provenances:
            rows_by_key[provenance, five_tuple].append(row)
    pins: dict[tuple[str, FiveTuple], dict[str, str]] = {}
    for key, rows in rows_by_key.items():
        pinned: dict[str, str] = {}
        for code_system in sorted(code_systems):
            held = {row.constraint_properties.get(code_system) for row in rows}
            if len(held) == 1 and (value := held.pop()) is not None:
                pinned[code_system] = value
        pins[key] = pinned
    return pins


def _check_consistent_parents(codes_by_dcid: dict[str, CodeRow]) -> None:
    """Raises `CrossSystemParent` for a parent in another code system.

    Missing parent rows are left for `_root_of` to treat as chain roots."""
    for row in codes_by_dcid.values():
        if row.parent is None:
            continue
        parent_row = codes_by_dcid.get(row.parent)
        if parent_row is not None and parent_row.code_system != row.code_system:
            raise CrossSystemParent(
                f"code row {row.dcid!r} in code system {row.code_system!r} has parent "
                f"{row.parent!r} in code system {parent_row.code_system!r}"
            )


def _check_acyclic_parents(codes_by_dcid: dict[str, CodeRow]) -> None:
    """Raises `CyclicParent` when a chain revisits a dcid. Each row has at most one parent.

    Chains reaching a resolved dcid or missing row terminate."""
    resolved: set[str] = set()
    for start_dcid in codes_by_dcid:
        if start_dcid in resolved:
            continue
        chain: list[str] = []
        node: str | None = start_dcid
        while node is not None and node not in resolved:
            if node in chain:
                cycle = [*chain[chain.index(node) :], node]
                raise CyclicParent(f"code parent relation forms a cycle: {' -> '.join(cycle)}")
            chain.append(node)
            row = codes_by_dcid.get(node)
            node = row.parent if row is not None else None
        resolved.update(chain)


def _check_declared_defaults(
    declarations: Mapping[str, CodeSystemDeclaration], codes_by_dcid: dict[str, CodeRow]
) -> None:
    """Raises `InvalidDeclaredDefault` for a default code outside its system or an invalid
    `when_bound_under` dcid."""
    for declaration in declarations.values():
        default = declaration.default
        if default is None:
            continue
        code_row = codes_by_dcid.get(default.code)
        if code_row is None or code_row.code_system != declaration.code_system:
            raise InvalidDeclaredDefault(
                f"code system {declaration.code_system!r} declares default code "
                f"{default.code!r}, which is not a row of that code system"
            )
        for scoping_dcid in default.when_bound_under:
            scoping_row = codes_by_dcid.get(scoping_dcid)
            if scoping_row is None:
                raise InvalidDeclaredDefault(
                    f"code system {declaration.code_system!r} declares a default "
                    f"when_bound_under code {scoping_dcid!r}, which this index holds no "
                    "row for"
                )
            if scoping_row.code_system == declaration.code_system:
                raise InvalidDeclaredDefault(
                    f"code system {declaration.code_system!r} declares a default "
                    f"when_bound_under code {scoping_dcid!r} in its own code system -- a "
                    "default cannot condition on the slot it defaults"
                )


def dedupe_first_seen(groups: Iterable[Iterable[str]]) -> tuple[str, ...]:
    """Flattens resolved groups, keeping each dcid's first occurrence.

    Descendant expansion can return the same dcid for multiple input refs."""
    seen: set[str] = set()
    result: list[str] = []
    for group in groups:
        for item in group:
            if item not in seen:
                seen.add(item)
                result.append(item)
    return tuple(result)


def _normalize_text(text: str) -> str:
    """Strips surrounding whitespace and casefolds text for normalized lookups."""
    return text.strip().casefold()


class CodeIndex:
    """Read-only index. Lookup tables are built once in `__init__`."""

    __slots__ = (
        "__weakref__",
        "_children_by_parent",
        "_codes_by_dcid",
        "_codes_by_normalized_text",
        "_codes_by_system_code",
        "_codes_by_text",
        "codes",
        "dataset_pins",
        "datasets",
        "datasets_by_five_tuple",
        "declarations",
        "stamp",
        "statvars",
    )

    def __init__(
        self,
        *,
        stamp: IndexStamp,
        declarations: Iterable[CodeSystemDeclaration],
        codes: Iterable[CodeRow],
        statvars: Iterable[StatVarRow],
        datasets: Iterable[DatasetDeclaration] = (),
    ) -> None:
        self.stamp = stamp
        self.codes = tuple(codes)
        self.statvars = tuple(statvars)
        self.declarations: dict[str, CodeSystemDeclaration] = {
            d.code_system: d for d in declarations
        }
        self.datasets: dict[str, DatasetDeclaration] = {d.provenance: d for d in datasets}

        codes_by_dcid: dict[str, CodeRow] = {}
        codes_by_system_code: dict[tuple[str, str], CodeRow] = {}
        children_by_parent: dict[str, list[str]] = defaultdict(list)
        text_index: dict[str, list[CodeRow]] = defaultdict(list)
        normalized_text_index: dict[str, list[CodeRow]] = defaultdict(list)
        for row in self.codes:
            if row.dcid in codes_by_dcid:
                raise DuplicateDcid(f"code dcid {row.dcid!r} appears more than once")
            codes_by_dcid[row.dcid] = row
            system_code = (row.code_system, row.code)
            if system_code in codes_by_system_code:
                raise DuplicateDcid(
                    f"code system {row.code_system!r} code {row.code!r} appears more than once"
                )
            codes_by_system_code[system_code] = row
            if row.parent is not None:
                children_by_parent[row.parent].append(row.dcid)
            for text in (row.label, *row.synonyms):
                text_index[text].append(row)
            # A label and a synonym on the same row can normalize to the
            # same key ("action relating to debt" / "Action Relating to
            # Debt" on CRSPurposeCode600 both do) -- dedupe per row before
            # indexing so that row appears once per normalized key it earns.
            normalized_texts = {_normalize_text(text) for text in (row.label, *row.synonyms)}
            for normalized in normalized_texts:
                normalized_text_index[normalized].append(row)

        _check_consistent_parents(codes_by_dcid)
        _check_acyclic_parents(codes_by_dcid)
        _check_declared_defaults(self.declarations, codes_by_dcid)

        seen_statvar_dcids: set[str] = set()
        for row in self.statvars:
            if row.dcid in seen_statvar_dcids:
                raise DuplicateDcid(f"StatVar dcid {row.dcid!r} appears more than once")
            seen_statvar_dcids.add(row.dcid)

        self.datasets_by_five_tuple: dict[FiveTuple, tuple[str, ...]] = _datasets_by_five_tuple(
            self.statvars, self.datasets
        )
        """The provenances of the datasets publishing each five-tuple's rows, in serving
        order (`DatasetDeclaration.rank`)."""
        self.dataset_pins: dict[tuple[str, FiveTuple], dict[str, str]] = _dataset_pins(
            self.statvars, self.declarations
        )
        """For each dataset and five-tuple it publishes, every code-system property the
        dataset's rows of the five-tuple all carry with one value, mapped to that value. An
        identity whose family that dataset serves binds it."""

        self._codes_by_dcid = codes_by_dcid
        self._codes_by_system_code = codes_by_system_code
        self._children_by_parent = {k: tuple(v) for k, v in children_by_parent.items()}
        self._codes_by_text = {k: tuple(v) for k, v in text_index.items()}
        self._codes_by_normalized_text = {k: tuple(v) for k, v in normalized_text_index.items()}

    # --- Lookup ---

    def code_by_dcid(self, dcid: str) -> CodeRow | None:
        """The code row for `dcid`, or `None` if this index holds no such
        row."""
        return self._codes_by_dcid.get(dcid)

    def code_by_code(self, code_system: str, code: str) -> CodeRow | None:
        """Returns the row for `code` in `code_system`. Code strings may recur across systems,
        so pass both fields to identify the row."""
        return self._codes_by_system_code.get((code_system, code))

    def codes_by_text(self, text: str) -> tuple[CodeRow, ...]:
        """Every code row whose label or synonym exactly matches `text`,
        across every code system this index carries. Each row appears at
        most once per distinct text it carries, and the same text naming
        codes in different systems returns more than one row. The caller
        disambiguates. This method performs lookup only."""
        return self._codes_by_text.get(text, ())

    def codes_by_normalized_text(self, text: str) -> tuple[CodeRow, ...]:
        """Returns rows whose labels or synonyms match after whitespace stripping and casefolding.

        A row appears once when its own label and synonym normalize alike.
        Distinct matching rows remain distinct."""
        return self._codes_by_normalized_text.get(_normalize_text(text), ())

    # --- Dataset declarations ---

    def absent_observation(self, provenance: str) -> AbsentObservation:
        """Returns the dataset's declared absent-observation rule, or `unknown` when no dataset
        declares `provenance`."""
        dataset = self.datasets.get(provenance)
        return dataset.absent_observation if dataset is not None else "unknown"

    # --- Descendants ---

    def descendants(self, dcid: str) -> tuple[str, ...]:
        """Returns `dcid` and its descendants breadth-first.

        Flat or unknown codes return `(dcid,)`. The walk tracks visited nodes to
        terminate on cyclic parent data."""
        row = self._codes_by_dcid.get(dcid)
        if row is None:
            return (dcid,)
        declaration = self.declarations.get(row.code_system)
        if declaration is None or not declaration.hierarchical:
            return (dcid,)
        result = [dcid]
        visited = {dcid}
        frontier = [dcid]
        while frontier:
            next_frontier = [
                child
                for node in frontier
                for child in self._children_by_parent.get(node, ())
                if child not in visited
            ]
            visited.update(next_frontier)
            result.extend(next_frontier)
            frontier = next_frontier
        return tuple(result)

    # --- Rollup ---

    def covering_roots(self, values: Iterable[str]) -> tuple[str, ...]:
        """Returns each value's root ancestor, deduplicated in input order.

        Flat-system and unknown values roll up to themselves."""
        result: list[str] = []
        for value in values:
            root = self._root_of(value)
            if root not in result:
                result.append(root)
        return tuple(result)

    def ancestor_at_depth(self, dcid: str, depth: int) -> str | None:
        """Returns the ancestor of `dcid` at `depth` on its parent chain, or `None` when `dcid`
        sits shallower than `depth`.

        Depth 0 is the chain's root, and `dcid`'s own depth returns `dcid`. Depth counts parent
        steps, so codes of one length can sit at different depths. Flat-system and unknown codes
        are roots. A dangling parent is the root, as in `covering_roots`."""
        chain = [dcid]
        row = self._codes_by_dcid.get(dcid)
        while row is not None and row.parent is not None:
            chain.append(row.parent)
            row = self._codes_by_dcid.get(row.parent)
        if depth >= len(chain):
            return None
        return chain[-1 - depth]

    def _root_of(self, dcid: str) -> str:
        """Returns the topmost ancestor, or `dcid` when unknown. A dangling parent returns
        that missing dcid as the root."""
        row = self._codes_by_dcid.get(dcid)
        if row is None:
            return dcid
        while row.parent is not None:
            parent_row = self._codes_by_dcid.get(row.parent)
            if parent_row is None:
                return row.parent
            row = parent_row
        return row.dcid

    # --- Filter ---

    def filter(self, query: ConstraintQuery) -> tuple[StatVarRow, ...]:
        """Returns matching StatVar rows in index order.

        `Bound` and `BoundSet` require a declared matching property. `Open`
        and omitted properties are ignored."""
        return tuple(row for row in self.statvars if self._matches(row, query))

    @staticmethod
    def _matches(row: StatVarRow, query: ConstraintQuery) -> bool:
        for prop, binding in query.items():
            if isinstance(binding, Bound):
                if row.constraint_properties.get(prop) != binding.value:
                    return False
            elif (
                isinstance(binding, BoundSet)
                and row.constraint_properties.get(prop) not in binding.values
            ):
                return False
        return True

    # --- Level-scoped menus ---

    def value_counts(self, code_system: str) -> dict[str, int]:
        """Counts StatVar rows by declared value in one linear scan."""
        counts: dict[str, int] = defaultdict(int)
        for row in self.statvars:
            value = row.constraint_properties.get(code_system)
            if value is not None:
                counts[value] += 1
        return dict(counts)

    def level_scoped_options(
        self, code_system: str, bound_dcid: str | None
    ) -> tuple[LevelOption, ...]:
        """Returns the menu of values for one code-system slot.

        A flat system returns every value as a root. A valid hierarchical binding returns
        its immediate children, then its parent. An unbound hierarchical slot returns each
        root followed by that root's children. `hierarchical` selects the branch, so a flat
        system's parent fields are ignored. Every hierarchical option with a parent names it
        in `parent`.

        Unknown or foreign `bound_dcid` values act as unbound. Set bindings pass `None`.
        Siblings sort by member count descending, then dcid ascending."""
        counts = self.value_counts(code_system)
        declaration = self.declarations.get(code_system)
        hierarchical = declaration is not None and declaration.hierarchical

        def _ordered(dcids: Iterable[str]) -> list[str]:
            return sorted(dcids, key=lambda d: (-counts.get(d, 0), d))

        def _children(parent: str) -> list[LevelOption]:
            return [
                LevelOption(dcid=d, relation="child", member_count=counts.get(d, 0), parent=parent)
                for d in _ordered(self._children_by_parent.get(parent, ()))
            ]

        if not hierarchical:
            all_dcids = (row.dcid for row in self.codes if row.code_system == code_system)
            return tuple(
                LevelOption(dcid=d, relation="root", member_count=counts.get(d, 0))
                for d in _ordered(all_dcids)
            )

        row = self._codes_by_dcid.get(bound_dcid) if bound_dcid is not None else None
        if row is not None and row.code_system == code_system:
            options = _children(row.dcid)
            if row.parent is not None:
                parent_row = self._codes_by_dcid.get(row.parent)
                options.append(
                    LevelOption(
                        dcid=row.parent,
                        relation="parent",
                        member_count=counts.get(row.parent, 0),
                        parent=parent_row.parent if parent_row is not None else None,
                    )
                )
            return tuple(options)

        roots = (r.dcid for r in self.codes if r.code_system == code_system and r.parent is None)
        options = []
        for root in _ordered(roots):
            options.append(
                LevelOption(dcid=root, relation="root", member_count=counts.get(root, 0))
            )
            options.extend(_children(root))
        return tuple(options)
