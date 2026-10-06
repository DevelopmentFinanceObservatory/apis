"""The Search eval case model: one YAML case format, its loader, and its eligibility rules.

A case file under ``corpus/`` is a YAML list of cases. Measure families live under
``corpus/families/<source>.yaml`` as a mapping from family id to family; a family id is
unique across files. Loading validates every case against ``Case`` with the families in
the validation context, so each measure arrives with its families expanded into
``Measure.forms`` and evaluators read one form whether the case named a family or wrote
an inline shape.

Under ``expect``, the status keys (``definite``, ``candidates``, ``no_data``,
``unsupported``, ``mixed``) are the accepted statuses. Fields under a status key apply only
when that status returns. Fields beside them (``measure``, ``entities``, ``provenance``,
``window``, ``unit``, ``operations``, ``warnings``) apply to every accepted status, and
``Expect.for_status`` merges the two. ``{gl: "..."}`` marks a value the grounding pass has
yet to read from the graph, and a case holding one is ineligible.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Iterator, Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, Self, get_args

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_serializer,
    model_validator,
)
from yaml.constructor import ConstructorError

from search.models import (
    DataCommonsId,
    DeclinedReason,
    EntityRoleKey,
    EntryPath,
    IdentityCurrency,
    IdentityFacetBinding,
    IdentityShapeSchema,
    NoDataCategory,
    NoDataGate,
    NoDataReason,
    OperationAxis,
    OperationVerb,
    ParsedPart,
    Slot,
)

DEFAULT_CORPUS_DIR = Path(__file__).resolve().parents[3] / "corpus"

CheckName = Literal[
    "disposition",
    "measure",
    "entities",
    "provenance",
    "request",
    "operations",
    "refusal",
    "warnings",
    "candidates",
    "parts",
    "skills",
    "entry_path",
    "structural",
    "groundedness",
    "materialisation",
    "task_outcome",
]
"""Every check name the scorer reports and ``known_failure.checks`` may cite."""
CHECK_NAMES: tuple[CheckName, ...] = get_args(CheckName)

StatusKey = Literal["definite", "candidates", "no_data", "unsupported", "mixed"]
ShapeKey = Literal[
    "population_type",
    "measured_property",
    "stat_type",
    "measurement_qualifier",
    "measurement_denominator",
]

_NonEmpty = Annotated[str, StringConstraints(min_length=1)]
_FamilyId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9_]+(\.[a-z0-9_]+)*$")]


class _CaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Gl(_CaseModel):
    """An unconfirmed value: the text says what the grounding pass must read from the graph."""

    gl: _NonEmpty


Pending = _NonEmpty | Gl
"""A string value the case asserts, or a ``Gl`` marker standing in for it."""
BoundValue = Pending | Annotated[list[Pending], Field(min_length=1)]
"""A slot binding: one value, or a list for a set binding."""
Shape = dict[ShapeKey, Pending | None]
"""A partial five-tuple. An omitted key is unasserted, and ``None`` asserts absence."""
Bound = dict[_NonEmpty, BoundValue]
PendingBound = Annotated[Gl | Bound, Field(union_mode="left_to_right")]
"""Slot bindings, or one ``Gl`` for slots not yet named. ``{gl: ...}`` reads as the marker."""


class Provenance(_CaseModel):
    """The Data Commons and provenance dcid an answer's facets name, and whether its
    identity binds the provenance (the part names a dataset, or bind elects a
    shape only an opt-in dataset publishes, which names that table)."""

    origin: DataCommonsId | None = None
    dataset: Pending | None = Field(default=None, description="the provenance dcid")
    bound: bool | None = Field(
        default=None, description="whether the identity's provenance facet is a value binding"
    )

    @model_validator(mode="after")
    def _asserts_something(self) -> Self:
        if self.origin is None and self.dataset is None and self.bound is None:
            raise ValueError("provenance needs origin, dataset or bound")
        return self

    @model_serializer(mode="wrap")
    def _serialise(self, handler: SerializerFunctionWrapHandler) -> Any:
        # An unasserted bound stays out of the dump, so the expectation hash of a case that
        # asserts none matches the one it had before the field existed.
        dumped = handler(self)
        if self.bound is None:
            dumped.pop("bound", None)
        return dumped

    @property
    def names_source(self) -> bool:
        """Whether it names an origin or dataset, which replaces a family's."""
        return self.origin is not None or self.dataset is not None


class Family(_CaseModel):
    """A measure family: what every query of one kind must resolve to."""

    origin: DataCommonsId
    dataset: Pending = Field(description="the provenance dcid the family's data comes from")
    shape: Shape | Gl = Field(default_factory=dict)
    fixed: PendingBound = Field(default_factory=dict)
    stat_vars: list[Pending] | None = None


class MeasureForm(_CaseModel):
    """One acceptable measure after family expansion. A family list yields one form each.

    ``unconfirmed`` holds a whole-map ``Gl`` from the family's ``fixed`` or the case's
    ``bound``, standing for slots the grounding pass has yet to name.
    """

    family: str | None
    shape: Shape | Gl
    bound: Bound
    unconfirmed: list[Gl]
    open: list[Pending]
    stat_vars: list[Pending] | None
    provenance: Provenance | None


def _families(info: ValidationInfo) -> Mapping[str, Family]:
    return (info.context or {}).get("families", {})


def _family_form(
    family_id: str,
    family: Family,
    *,
    bound: PendingBound,
    open_slots: list[Pending],
    stat_vars: list[Pending] | None,
) -> MeasureForm:
    """Expand one family under a case's own bound and open slots."""
    unconfirmed = [side for side in (family.fixed, bound) if isinstance(side, Gl)]
    fixed = {} if isinstance(family.fixed, Gl) else family.fixed
    case_bound = {} if isinstance(bound, Gl) else bound
    for slot, value in case_bound.items():
        if slot in fixed and fixed[slot] != value:
            raise ValueError(f"bound {slot} conflicts with family {family_id}'s fixed value")
    open_in_fixed = {s for s in open_slots if isinstance(s, str)} & set(fixed)
    if open_in_fixed:
        raise ValueError(f"open slot(s) {sorted(open_in_fixed)} are fixed by family {family_id}")
    return MeasureForm(
        family=family_id,
        shape=family.shape,
        bound={**fixed, **case_bound},
        unconfirmed=unconfirmed,
        open=open_slots,
        stat_vars=stat_vars if stat_vars is not None else family.stat_vars,
        provenance=Provenance(origin=family.origin, dataset=family.dataset),
    )


class Measure(_CaseModel):
    """The measure a part resolves to: a family (or list, any matches) or an inline shape."""

    family: _FamilyId | Annotated[list[_FamilyId], Field(min_length=1)] | None = None
    shape: Shape | None = None
    bound: PendingBound = Field(default_factory=dict)
    open: list[Pending] = Field(default_factory=list)
    stat_vars: list[Pending] | None = Field(
        default=None, description="exact StatVars, only where the StatVar is the answer"
    )

    _forms: list[MeasureForm] = PrivateAttr(default_factory=list)

    @property
    def forms(self) -> list[MeasureForm]:
        """The acceptable measures with families expanded; any one matching passes."""
        return self._forms

    @model_validator(mode="after")
    def _expand(self, info: ValidationInfo) -> Self:
        if self.family is not None and self.shape is not None:
            raise ValueError("family and shape are exclusive")
        bound_slots = set() if isinstance(self.bound, Gl) else set(self.bound)
        both = {s for s in self.open if isinstance(s, str)} & bound_slots
        if both:
            raise ValueError(f"slot(s) {sorted(both)} are both bound and open")
        if self.family is None:
            if self.shape is None and not self.bound and not self.open and self.stat_vars is None:
                raise ValueError("measure asserts nothing")
            self._forms = [
                MeasureForm(
                    family=None,
                    shape=self.shape or {},
                    bound={} if isinstance(self.bound, Gl) else self.bound,
                    unconfirmed=[self.bound] if isinstance(self.bound, Gl) else [],
                    open=self.open,
                    stat_vars=self.stat_vars,
                    provenance=None,
                )
            ]
            return self
        families = _families(info)
        family_ids = [self.family] if isinstance(self.family, str) else self.family
        unknown = [f for f in family_ids if f not in families]
        if unknown:
            raise ValueError(f"unknown family {', '.join(unknown)}")
        self._forms = [
            _family_form(
                f, families[f], bound=self.bound, open_slots=self.open, stat_vars=self.stat_vars
            )
            for f in family_ids
        ]
        return self

    @model_serializer(mode="wrap")
    def _serialise(self, handler: SerializerFunctionWrapHandler, info: SerializationInfo) -> Any:
        if _expanding(info):
            return {"forms": [form.model_dump(mode="json") for form in self._forms]}
        return handler(self)


def _expanding(info: SerializationInfo) -> bool:
    return bool(info.context and info.context.get("expand"))


class ExpectedEntity(_CaseModel):
    """One entity the answer binds, on one role. ``expands`` marks a group read as members.

    A group the graph holds no node for (G20, the African Union) is named by ``label``
    instead of ``dcid``, matching the label the answer's ``entity_groups`` records.
    """

    dcid: Pending | None = None
    label: _NonEmpty | None = Field(
        default=None, description="an expanding group's name, for a group with no graph node"
    )
    role: EntityRoleKey | Gl
    expands: bool = False

    @model_validator(mode="after")
    def _names_the_entity_once(self) -> Self:
        if self.expands:
            if (self.dcid is None) == (self.label is None):
                raise ValueError("an expanding entity needs exactly one of dcid and label")
        elif self.dcid is None or self.label is not None:
            raise ValueError("an entity that does not expand needs dcid and takes no label")
        return self


class Window(_CaseModel):
    """The requested window: every year, the latest year, or a range open at either end."""

    kind: Literal["all", "latest", "range"]
    start: int | None = None
    end: int | None = None

    @model_validator(mode="after")
    def _bounds_match_kind(self) -> Self:
        if self.kind != "range":
            if self.start is not None or self.end is not None:
                raise ValueError(f"window kind {self.kind} takes no start or end")
            return self
        if self.start is None and self.end is None:
            raise ValueError("window kind range needs start or end")
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("window start is after its end")
        return self


class ExpectedOperation(_CaseModel):
    verb: OperationVerb
    over: OperationAxis | None = None
    target_year: int | None = None


class ExpectedUnit(_CaseModel):
    """The unit request the identity carries. An omitted field is unasserted."""

    currency: IdentityCurrency | None = None
    prices: Literal["current", "constant"] | None = None
    base_year: int | None = None

    @model_validator(mode="after")
    def _asserts_something_consistent(self) -> Self:
        if not self.model_fields_set:
            raise ValueError("unit asserts nothing; omit it or write none")
        if self.prices == "current" and self.base_year is not None:
            raise ValueError("base_year needs prices constant")
        return self


class Skills(_CaseModel):
    """Whether the first definite spec carries skills, and optionally which."""

    present: bool
    ids: list[_NonEmpty] | None = None


class _Assertions(_CaseModel):
    """The fields every status key, and the shared level beside them, may carry."""

    measure: Measure | None = None
    entities: list[ExpectedEntity] | None = None
    provenance: Provenance | None = Field(
        default=None, description="overrides the provenance the measure's family names"
    )
    window: Window | None = None
    unit: ExpectedUnit | Literal["none"] | None = None
    operations: list[ExpectedOperation] | None = None
    warnings: list[Pending] | None = Field(
        default=None, description="warning codes the answer must carry; extra warnings pass"
    )

    @field_validator("entities")
    @classmethod
    def _entities_are_a_set(
        cls, entities: list[ExpectedEntity] | None
    ) -> list[ExpectedEntity] | None:
        keys = [e.model_dump_json(include={"dcid", "label", "role"}) for e in entities or []]
        if len(set(keys)) != len(keys):
            raise ValueError("entities repeat a (dcid or label, role) pair")
        return entities


SHARED_FIELDS: tuple[str, ...] = tuple(_Assertions.model_fields)


class DefiniteExpect(_Assertions):
    skills: Skills | None = None


class CandidatesExpect(_Assertions):
    include_families: Annotated[list[_FamilyId], Field(min_length=1)] | None = Field(
        default=None, description="families the offered candidates must include"
    )

    _family_forms: dict[str, MeasureForm] = PrivateAttr(default_factory=dict)

    @property
    def family_forms(self) -> dict[str, MeasureForm]:
        """``include_families`` expanded, keyed by family id."""
        return self._family_forms

    @model_validator(mode="after")
    def _expand(self, info: ValidationInfo) -> Self:
        families = _families(info)
        unknown = [f for f in self.include_families or [] if f not in families]
        if unknown:
            raise ValueError(f"unknown family {', '.join(unknown)}")
        self._family_forms = {
            f: _family_form(f, families[f], bound={}, open_slots=[], stat_vars=None)
            for f in self.include_families or []
        }
        return self

    @model_serializer(mode="wrap")
    def _serialise(self, handler: SerializerFunctionWrapHandler, info: SerializationInfo) -> Any:
        dumped = handler(self)
        if _expanding(info) and "include_families" in dumped:
            dumped["include_families"] = {
                f: form.model_dump(mode="json") for f, form in self._family_forms.items()
            }
        return dumped


class NoDataExpect(_Assertions):
    reason: NoDataReason | Annotated[list[NoDataReason], Field(min_length=1)] | None = None
    category: NoDataCategory | None = None
    gate: NoDataGate | None = None


class UnsupportedExpect(_Assertions):
    code: _NonEmpty | None = None


class MixedExpect(_Assertions):
    pass


type StatusExpect = (
    DefiniteExpect | CandidatesExpect | NoDataExpect | UnsupportedExpect | MixedExpect
)


class _StatusKeyed(_Assertions):
    definite: DefiniteExpect | None = None
    candidates: CandidatesExpect | None = None
    no_data: NoDataExpect | None = None
    unsupported: UnsupportedExpect | None = None

    @property
    def accepted_statuses(self) -> tuple[StatusKey, ...]:
        """The status keys this expectation accepts, in declaration order."""
        return tuple(s for s in get_args(StatusKey) if getattr(self, s, None) is not None)

    def for_status(self, status: StatusKey) -> StatusExpect | None:
        """The status key's own fields merged with the shared ones, or None when not accepted."""
        block = getattr(self, status, None)
        if block is None:
            return None
        shared = {name: getattr(self, name) for name in self._shared_fields_set()}
        return block.model_copy(update=shared)

    def _shared_fields_set(self) -> set[str]:
        return self.model_fields_set & set(SHARED_FIELDS)

    @model_validator(mode="after")
    def _status_keys_do_not_repeat_shared_fields(self) -> Self:
        shared = self._shared_fields_set()
        for status in self.accepted_statuses:
            repeated = shared & getattr(self, status).model_fields_set
            if repeated:
                raise ValueError(f"{status} repeats shared field(s) {sorted(repeated)}")
        return self


class PartExpect(_StatusKeyed):
    """One part's expectation, in query order."""

    @model_validator(mode="after")
    def _needs_a_status(self) -> Self:
        if not self.accepted_statuses:
            raise ValueError("a part needs at least one status key")
        return self


class DeclinedExpect(_CaseModel):
    reason: DeclinedReason


ExpectedKind = Literal["resolution", "catalogue", "declined", "problem"]
"""The answer kinds a case may accept: the response kinds, plus ``problem`` for an RFC 9457
problem document a door answers with in place of a response."""


class ProblemExpect(_CaseModel):
    """The problem document a refused request returns."""

    code: _NonEmpty = Field(description="the problem's registry code")


class Expect(_StatusKeyed):
    """What a correct answer looks like."""

    kind: ExpectedKind | Annotated[list[ExpectedKind], Field(min_length=1)] = "resolution"
    scope_place: str | None = Field(
        default=None,
        description="the scope.place a catalogue or declined answer names; null asserts none",
    )
    declined: DeclinedExpect | None = None
    problem: ProblemExpect | None = None
    mixed: MixedExpect | None = None
    parts: Annotated[list[PartExpect], Field(min_length=2)] | None = None

    @property
    def kinds(self) -> frozenset[ExpectedKind]:
        return frozenset([self.kind] if isinstance(self.kind, str) else self.kind)

    @model_validator(mode="after")
    def _kinds_match_fields(self) -> Self:
        kinds = self.kinds
        statuses = self.accepted_statuses
        shared = self._shared_fields_set()
        if self.parts is not None:
            if statuses:
                raise ValueError("parts and top-level status keys are exclusive")
            if shared:
                raise ValueError(f"set {sorted(shared)} on each part, not beside parts")
        if "resolution" in kinds:
            if not statuses and self.parts is None:
                raise ValueError("a resolution case needs a status key or parts")
        elif statuses or self.parts is not None or shared:
            raise ValueError("status keys, parts and shared fields need kind resolution")
        if ("declined" in kinds) != (self.declined is not None):
            raise ValueError("declined is required exactly when kind includes declined")
        if ("problem" in kinds) != (self.problem is not None):
            raise ValueError("problem is required exactly when kind includes problem")
        if "scope_place" in self.model_fields_set and not kinds & {"catalogue", "declined"}:
            raise ValueError("scope_place needs kind catalogue or declined")
        return self


class KnownFailure(_CaseModel):
    """Checks expected to fail on the current engine, tracked by an issue (0 until filed)."""

    issue: int = Field(ge=0)
    checks: Annotated[list[CheckName], Field(min_length=1)]

    @field_validator("checks", mode="before")
    @classmethod
    def _not_answer(cls, checks: object) -> object:
        # The driver derives `answer` from the other checks and judges it beside them.
        if isinstance(checks, list) and "answer" in checks:
            raise ValueError(
                "answer is derived from the other checks; list the checks that fail instead"
            )
        return checks


class Resubmit(_CaseModel):
    """A dataset_resubmit identity, built by ``search.eval.dataset.resubmit_identity``."""

    variable_text: _NonEmpty
    shape: IdentityShapeSchema
    slots: list[Slot]
    entity_dcids: list[_NonEmpty] | None = None
    origin_dcids: Annotated[list[_NonEmpty], Field(min_length=1)] | None = None
    facets: dict[_NonEmpty, IdentityFacetBinding] | None = None


class Case(_CaseModel):
    """One eval case: a request, the reason it exists, and what a correct answer looks like."""

    id: Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]*$")]
    query: _NonEmpty
    door: EntryPath = "raw_text"
    parsed: Annotated[list[ParsedPart], Field(min_length=1)] | None = None
    resubmit: Resubmit | None = None
    include_observations: bool = Field(
        default=False, description="ask for the observation preview on the request"
    )
    slice: Literal["main", "holdout"] = "main"
    activation: Annotated[str, StringConstraints(pattern=r"^(live|pending_data:[a-z0-9_]+)$")] = (
        "live"
    )
    why: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    reuse: _NonEmpty | None = Field(default=None, description="a sentence-panel id")
    tags: dict[_NonEmpty, _NonEmpty | list[_NonEmpty]] = Field(default_factory=dict)
    known_failure: KnownFailure | None = None
    expect: Expect

    @field_validator("why")
    @classmethod
    def _why_is_one_line(cls, why: str) -> str:
        if "\n" in why:
            raise ValueError("why must be one line")
        return why

    @model_validator(mode="after")
    def _door_payload_matches_door(self) -> Self:
        if (self.door == "parsed") != (self.parsed is not None):
            raise ValueError("parsed is required exactly when door is parsed")
        if (self.door == "dataset_resubmit") != (self.resubmit is not None):
            raise ValueError("resubmit is required exactly when door is dataset_resubmit")
        return self


class CorpusError(Exception):
    """The corpus failed to load; ``errors`` lists every problem found."""

    def __init__(self, errors: list[str]):
        super().__init__("\n".join(errors))
        self.errors = errors


class _UniqueKeyLoader(yaml.SafeLoader):
    """A safe loader that refuses a mapping repeating a key, which plain YAML overwrites."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise ConstructorError(None, None, f"duplicate key {key!r}", key_node.start_mark)
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _read_yaml(path: Path, errors: list[str]) -> Any:
    try:
        return yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        errors.append(f"{path.name}: invalid YAML: {exc}")
        return None


def _validation_errors(prefix: str, exc: ValidationError) -> Iterator[str]:
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"])
        yield f"{prefix}: {loc}: {err['msg']}" if loc else f"{prefix}: {err['msg']}"


_FAMILY_ID = TypeAdapter(_FamilyId)


def _load_families(families_dir: Path) -> tuple[dict[str, Family], list[str]]:
    families: dict[str, Family] = {}
    errors: list[str] = []
    if not families_dir.is_dir():
        return families, errors
    for path in sorted(families_dir.glob("*.yaml")):
        raw = _read_yaml(path, errors) or {}
        if not isinstance(raw, dict):
            errors.append(f"families/{path.name}: expected a mapping of family id to family")
            continue
        for family_id, body in raw.items():
            prefix = f"families/{path.name}: {family_id}"
            if family_id in families:
                errors.append(f"{prefix}: duplicate family id")
                continue
            try:
                _FAMILY_ID.validate_python(family_id)
                families[family_id] = Family.model_validate(body)
            except ValidationError as exc:
                errors.extend(_validation_errors(prefix, exc))
    return families, errors


def load_families(families_dir: Path) -> dict[str, Family]:
    """Load every family under ``families_dir``; a missing directory holds none.

    Raises:
        CorpusError: a file fails to parse, a family fails validation, or an id repeats.
    """
    families, errors = _load_families(families_dir)
    if errors:
        raise CorpusError(errors)
    return families


def validate_case(raw: object, *, families: Mapping[str, Family]) -> Case:
    """Validate one case with ``families`` available for expansion."""
    return Case.model_validate(raw, context={"families": families})


def load_case_files(corpus_dir: Path = DEFAULT_CORPUS_DIR) -> dict[str, list[Case]]:
    """Load ``corpus_dir/*.yaml`` keyed by file stem, with ``corpus_dir/families`` expanded.

    Raises:
        CorpusError: listing every parse error, schema error, unknown family, and id
            repeated across files.
    """
    families, errors = _load_families(corpus_dir / "families")
    by_file: dict[str, list[Case]] = {}
    seen: dict[str, str] = {}
    for path in sorted(corpus_dir.glob("*.yaml")):
        raw = _read_yaml(path, errors) or []
        if not isinstance(raw, list):
            errors.append(f"{path.name}: expected a list of cases")
            continue
        cases = by_file.setdefault(path.stem, [])
        for index, item in enumerate(raw):
            case_id = item.get("id") if isinstance(item, dict) else None
            prefix = f"{path.name}: {case_id or f'[{index}]'}"
            if isinstance(case_id, str):
                if case_id in seen:
                    errors.append(f"{prefix}: duplicate id, first in {seen[case_id]}")
                seen.setdefault(case_id, path.name)
            try:
                cases.append(validate_case(item, families=families))
            except ValidationError as exc:
                errors.extend(_validation_errors(prefix, exc))
    if errors:
        raise CorpusError(errors)
    return by_file


def load_cases(corpus_dir: Path = DEFAULT_CORPUS_DIR) -> list[Case]:
    """Every case under ``corpus_dir``, in file then list order. See ``load_case_files``."""
    return [case for cases in load_case_files(corpus_dir).values() for case in cases]


def _normalised(case: Case) -> dict[str, Any]:
    """The request and expectation as authored, with every family expanded."""
    return {
        "query": case.query,
        "door": case.door,
        "parsed": [p.model_dump(mode="json", exclude_unset=True) for p in case.parsed or []],
        "resubmit": case.resubmit.model_dump(mode="json", exclude_unset=True)
        if case.resubmit
        else None,
        "include_observations": case.include_observations,
        "expect": case.expect.model_dump(mode="json", exclude_unset=True, context={"expand": True}),
    }


def expectation_hash(case: Case) -> str:
    """sha256 over the normalised expectation (families expanded), query, door and payload.

    Editing ``why``, ``tags``, ``slice``, ``activation``, ``reuse`` or ``known_failure``
    leaves it unchanged. Editing a family changes it for every case citing the family.
    """
    canonical = json.dumps(
        _normalised(case), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _gl_paths(node: Any, path: str) -> Iterator[str]:
    if isinstance(node, dict):
        if set(node) == {"gl"}:
            yield path
            return
        for key, value in node.items():
            yield from _gl_paths(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _gl_paths(value, f"{path}[{index}]")


def unconfirmed_paths(case: Case) -> list[str]:
    """Paths of every ``gl`` value the case asserts, including those its families carry."""
    return list(_gl_paths(_normalised(case)["expect"], "expect"))


def _assertion_checks(block: _Assertions) -> set[CheckName]:
    fields = block.model_fields_set
    checks: set[CheckName] = set()
    if block.measure is not None:
        checks.add("measure")
        if any(form.provenance is not None for form in block.measure.forms):
            checks.add("provenance")
    if "provenance" in fields:
        checks.add("provenance")
    if "entities" in fields:
        checks.add("entities")
    if fields & {"window", "unit"}:
        checks.add("request")
    if "operations" in fields:
        checks.add("operations")
    if "warnings" in fields:
        checks.add("warnings")
    if isinstance(block, DefiniteExpect) and "skills" in fields:
        checks.add("skills")
    if isinstance(block, CandidatesExpect) and "include_families" in fields:
        checks.add("candidates")
    if isinstance(block, NoDataExpect) and fields & {"reason", "category", "gate"}:
        checks.add("refusal")
    if isinstance(block, UnsupportedExpect) and "code" in fields:
        checks.add("refusal")
    return checks


def _status_keyed_checks(expect: _StatusKeyed) -> set[CheckName]:
    checks: set[CheckName] = set()
    for status in expect.accepted_statuses:
        merged = expect.for_status(status)
        if merged is not None:
            checks |= _assertion_checks(merged)
    return checks


def asserted_checks(case: Case) -> frozenset[CheckName]:
    """The field-driven checks a case's ``expect`` gives something to assert.

    ``disposition`` reads kind, accepted statuses, ``scope_place`` and the problem code.
    ``refusal`` reads the ``no_data``, ``unsupported`` and ``declined`` details. ``request``
    reads ``window`` and ``unit``. ``provenance`` reads an explicit override or the
    measure's family. The checks that read no case field (``entry_path``, ``structural``,
    ``groundedness``, ``materialisation``, ``task_outcome``) are left out.
    """
    expect = case.expect
    checks: set[CheckName] = {"disposition"}
    if expect.declined is not None:
        checks.add("refusal")
    checks |= _status_keyed_checks(expect)
    for part in expect.parts or []:
        checks |= {"parts"} | _status_keyed_checks(part)
    return frozenset(checks)


def ineligibility(case: Case, *, implemented_checks: Collection[str]) -> list[str]:
    """Why the case cannot gate; empty when it is eligible.

    A case is eligible when it sits in the main slice, its activation is live, it holds no
    ``gl`` value, and every check it asserts is in ``implemented_checks``.
    """
    reasons = []
    if case.slice != "main":
        reasons.append(f"slice {case.slice}")
    if case.activation != "live":
        reasons.append(f"activation {case.activation}")
    reasons.extend(f"unconfirmed value at {path}" for path in unconfirmed_paths(case))
    missing = sorted(set(asserted_checks(case)) - set(implemented_checks))
    reasons.extend(f"no evaluator for check {name}" for name in missing)
    return reasons


def corpus_schema() -> dict[str, Any]:
    """The JSON schema of one case file: a list of cases."""
    return TypeAdapter(list[Case]).json_schema()
