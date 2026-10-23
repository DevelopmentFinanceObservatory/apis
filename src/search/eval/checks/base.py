"""The scoring core: verdicts, the wire schema, path reads, and the check registry.

A check reads named JSON paths from the raw response dict a Search app returned. Whether a
path exists is decided by that app's own ``GET /api/search/schema`` document, parsed into a
``WireSchema``: a path the schema lacks is UNVERIFIABLE (the field was renamed or removed on
that side), and a path the schema has but the body omits reads as None, because the resolve
endpoint serialises with ``exclude_none``. ``read_path`` raises ``Unverifiable`` for the
first case, and the scorer turns it into the check's verdict.

Two kinds of check register here. A case check (``register``) scores a whole response. A
part check (``register_part``) scores one response part against the status block that
applies to it; registering one also registers a case check that runs it over every part
(``over_parts``), and the ``parts`` check runs every part check on each part in order.

``expected_parts`` pairs each expected part with the response in one pass, and every check
reads the pairing from it. A part whose status the expectation does not accept pairs with
no block, and so does an expected part the response did not return: one past the last part
returned, or any part of an answer whose kind the case does not accept. No check scores an
unpaired part. A check reads UNREACHED for it when a block the expectation holds for that
part asserts the check (``cases.assertion_checks``), and n/a when none does. The declined
reason, which no status block holds, reads UNREACHED when the answer leaves every expected
part unpaired (``refusal``). With ``expect.parts`` the result names the parts behind it
(``Result.parts``), so the gate can tell a failure on one part from an unreached part
elsewhere.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol, get_args

import httpx

from dc_graph import GraphReadError
from search.engine.errors import GraphInfraError
from search.eval.cases import (
    CHECK_NAMES,
    Case,
    CheckName,
    Expect,
    Gl,
    PartExpect,
    StatusExpect,
    StatusKey,
    assertion_checks,
)
from series_coverage.loader import CoverageArtifactError

if TYPE_CHECKING:
    from search.engine.clients.sdmx import EngineSdmxClient
    from search.eval.graph import GraphClient
    from series_coverage.shared import ServedCoverage


STATUS_KEYS: tuple[StatusKey, ...] = get_args(StatusKey)


class Verdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNVERIFIABLE = "unverifiable"
    UNREACHED = "unreached"
    """The expectation asserts the check, but the response's status or kind is one the case
    does not accept, or an expected part is missing, so the check never ran."""
    NA = "n/a"


@dataclass(frozen=True)
class Result:
    """A check's verdict with the reason a report shows beside it."""

    verdict: Verdict
    detail: str = ""
    infra: bool = False
    """The check could not read the graph or coverage it scores against (``infra_error``).
    The verdict is UNVERIFIABLE, and the driver reports the sample as infrastructure."""
    parts: Mapping[int, Verdict] | None = field(default=None, hash=False)
    """On a case with ``expect.parts``, the verdict of each expected part that did not pass,
    by position, as a read-only copy. None when the verdict concerns the whole answer, and
    on a pass or n/a. Hashing leaves it out, so equal results still hash alike."""

    def __post_init__(self) -> None:
        if self.parts is None:
            return
        held = None if self.verdict in (Verdict.PASS, Verdict.NA) else dict(self.parts)
        object.__setattr__(self, "parts", None if held is None else MappingProxyType(held))

    def at(self, prefix: str) -> Result:
        """The same verdict with ``prefix`` leading the detail of a failure."""
        if self.verdict in (Verdict.PASS, Verdict.NA):
            return self
        detail = f"{prefix}: {self.detail}" if self.detail else prefix
        return replace(self, detail=detail)


PASS = Result(Verdict.PASS)
NA = Result(Verdict.NA)


def fail(detail: str) -> Result:
    return Result(Verdict.FAIL, detail)


def unverifiable(detail: str) -> Result:
    return Result(Verdict.UNVERIFIABLE, detail)


def infra_error(detail: str) -> Result:
    """A read of the graph or coverage failed, so the check scored nothing."""
    return Result(Verdict.UNVERIFIABLE, detail, infra=True)


READ_ERRORS: tuple[type[Exception], ...] = (
    GraphInfraError,
    GraphReadError,
    httpx.HTTPError,
    CoverageArtifactError,
)
"""What a failed graph or coverage read raises. A check scores these as ``infra_error``
and lets anything else propagate, so a bug in a check stops the run."""


def all_of(results: Iterable[Result]) -> Result:
    """Every assertion must hold: the first failure, else the first unverifiable, else the
    first unreached, else a pass when anything passed, else n/a."""
    collected = list(results)
    for verdict in (Verdict.FAIL, Verdict.UNVERIFIABLE, Verdict.UNREACHED):
        found = next((r for r in collected if r.verdict == verdict), None)
        if found is not None:
            return found
    return PASS if any(r.verdict == Verdict.PASS for r in collected) else NA


def any_of(results: Iterable[Result]) -> Result:
    """One alternative must hold: a pass, else the first unverifiable (it may pass), else
    the first failure, else the first unreached, else n/a."""
    collected = list(results)
    for verdict in (Verdict.PASS, Verdict.UNVERIFIABLE, Verdict.FAIL, Verdict.UNREACHED):
        found = next((r for r in collected if r.verdict == verdict), None)
        if found is not None:
            return found
    return NA


_ANY: Any = True
"""The JSON schema ``true``: every path below it exists."""


@dataclass
class WireSchema:
    """The response half of one app's ``GET /api/search/schema`` document.

    Paths are dotted property names from the response root, with ``[]`` after a property
    whose value is an array to step into its items (``parts[].spec.definition``). A key
    under an object whose ``additionalProperties`` names a schema is any key, so
    ``resolved.facets.provenance`` exists wherever ``facets`` maps axes to facets.
    """

    response: Mapping[str, Any]
    _known: dict[str, bool] = field(default_factory=dict, repr=False)

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> WireSchema:
        """Parse the JSON body ``GET /api/search/schema`` returns."""
        return cls(response=document["response"])

    def has(self, path: str) -> bool:
        """Whether some branch of the response schema declares ``path``."""
        if path not in self._known:
            self._known[path] = self._walk(path)
        return self._known[path]

    def _walk(self, path: str) -> bool:
        nodes = self._branches(self.response)
        for segment in path.split("."):
            name, items = (segment[:-2], True) if segment.endswith("[]") else (segment, False)
            nodes = [child for node in nodes for child in self._property(node, name)]
            if items:
                nodes = [child for node in nodes for child in self._items(node)]
            if any(node is _ANY for node in nodes):
                return True
            if not nodes:
                return False
        return True

    def _branches(self, node: Any) -> list[Any]:
        """``node`` with every ``$ref`` followed and every union arm listed."""
        if node is True or node == {}:
            return [_ANY]
        if not isinstance(node, Mapping):
            return []
        if "$ref" in node:
            name = node["$ref"].removeprefix("#/$defs/")
            return self._branches(self.response["$defs"][name])
        arms = [arm for key in ("anyOf", "oneOf", "allOf") for arm in node.get(key, ())]
        if arms:
            return [branch for arm in arms for branch in self._branches(arm)]
        return [node]

    def _property(self, node: Any, name: str) -> list[Any]:
        if node is _ANY:
            return [_ANY]
        properties = node.get("properties", {})
        if name in properties:
            return self._branches(properties[name])
        extra = node.get("additionalProperties")
        if extra is True or isinstance(extra, Mapping):
            return self._branches(extra)
        return []

    def _items(self, node: Any) -> list[Any]:
        if node is _ANY:
            return [_ANY]
        return self._branches(node["items"]) if "items" in node else []


class MembersReader(Protocol):
    """Reads the StatVars a Datasets mint of a Spec serves, as bare dcids."""

    def __call__(self, spec_id: str, definition: Mapping[str, Any]) -> list[str]: ...


class MembersRefused(Exception):
    """The members door refused a Spec's definition itself: the app answered with a Spec its
    own Datasets door will not mint. A check scores it as a failure of the answer, naming the
    problem ``code`` and ``detail``."""

    def __init__(self, *, code: str, detail: str) -> None:
        super().__init__(f"the members door refuses the definition: {code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class ScoreContext:
    schema: WireSchema
    """The responding app's ``GET /api/search/schema``, parsed."""
    graph: GraphClient | None = None
    sdmx: EngineSdmxClient | None = None
    coverage: ServedCoverage | None = None
    side: Literal["head", "base"] = "head"
    members: MembersReader | None = None
    """The responding app's members door (``runner.members_door_reader``): the wire states
    how many StatVars a Spec serves, and the door names them (``served_stat_vars``). None
    leaves a check that needs them unverifiable on an app whose wire does not list them."""


class SpecRead(Protocol):
    """A Spec body in a response, read at its own schema path ``at``."""

    @property
    def at(self) -> str: ...

    def read(self, path: str, ctx: ScoreContext) -> Any: ...


# Where an app built before the members door lists the StatVars a Spec serves.
_WIRE_MEMBERS = "resolved.stat_vars.items"


def served_stat_vars(spec: SpecRead, ctx: ScoreContext) -> list[str] | None:
    """The StatVars a mint of ``spec`` serves, as the responding app states them, or None
    when it states them nowhere the check can read.

    An app whose schema declares ``resolved.stat_vars.items`` lists them in the answer, and
    its ``ref``s are read there. Any other app names them through its members door
    (``ScoreContext.members``), so each side of a paired run is scored by its own wire.
    A failed door read raises one of ``READ_ERRORS``, and a refusal of the definition
    ``MembersRefused``.
    """
    items_at = join(spec.at, _WIRE_MEMBERS)
    if ctx.schema.has(items_at):
        items = spec.read(_WIRE_MEMBERS, ctx) or []
        return [read_path(item, "ref", ctx, at=f"{items_at}[]") for item in items]
    if ctx.members is None:
        return None
    return ctx.members(spec.read("spec_id", ctx), spec.read("definition", ctx))


class Unverifiable(Exception):
    """A check read a path the responding app's schema does not declare."""

    def __init__(self, path: str, side: str):
        super().__init__(f"{path} is not in the {side} schema")
        self.path = path


def join(at: str, path: str) -> str:
    return f"{at}.{path}" if at else path


def read_path(node: Any, path: str, ctx: ScoreContext, *, at: str = "") -> Any:
    """The value at dotted ``path`` under ``node``, where ``at`` is ``node``'s own schema path.

    Returns None when the body omits the field or one of its parents, since the wire
    omits None fields.

    Raises:
        Unverifiable: the schema declares no ``at.path``.
    """
    full = join(at, path)
    if not ctx.schema.has(full):
        raise Unverifiable(full, ctx.side)
    for key in path.split("."):
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return node


def guarded(assertion: Callable[[], Result]) -> Result:
    """Run ``assertion``, scoring a read of an undeclared path as unverifiable."""
    try:
        return assertion()
    except Unverifiable as exc:
        return unverifiable(str(exc))


def unconfirmed(*values: object) -> bool:
    """Whether any expected value is a ``gl`` marker the grounding pass has yet to fill."""
    return any(
        isinstance(v, Gl) or (isinstance(v, list) and any(isinstance(x, Gl) for x in v))
        for v in values
    )


def pairs_one_to_one(rows: int, cols: int, admits: Callable[[int, int], bool]) -> bool:
    """Whether every row pairs with a distinct column it admits (augmenting paths)."""
    owner: dict[int, int] = {}

    def claim(row: int, seen: set[int]) -> bool:
        for col in range(cols):
            if col in seen or not admits(row, col):
                continue
            seen.add(col)
            if col not in owner or claim(owner[col], seen):
                owner[col] = row
                return True
        return False

    return all(claim(row, set()) for row in range(rows))


@dataclass(frozen=True)
class PartView:
    """One response part, its position, and its status."""

    index: int
    body: Mapping[str, Any]
    status: str

    path = "parts[]"
    """The schema path of every part."""


type CheckFn = Callable[[Case, Mapping[str, Any], ScoreContext], Result]
"""Scores one case against one raw response."""
type PartCheckFn = Callable[[StatusExpect, PartView, Mapping[str, Any], ScoreContext], Result]
"""Scores one response part against the status block that applies to it."""

_CHECKS: dict[CheckName, CheckFn] = {}
_PART_CHECKS: dict[CheckName, PartCheckFn] = {}
_MODAL_ONLY: set[CheckName] = set()

CHECKS: Mapping[CheckName, CheckFn] = MappingProxyType(_CHECKS)
"""Every registered case check by name. ``scripts/eval_corpus.py`` reads its keys."""
PART_CHECKS: Mapping[CheckName, PartCheckFn] = MappingProxyType(_PART_CHECKS)
"""Every registered part check by name. The ``parts`` check runs them on each part."""
MODAL_ONLY_CHECKS: AbstractSet[CheckName] = _MODAL_ONLY
"""Checks the driver runs on a case's modal sample alone, registered with ``modal_only``;
other samples pass them to ``score(..., skip=...)``."""


def register(name: CheckName, *, modal_only: bool = False) -> Callable[[CheckFn], CheckFn]:
    """Register a case check under ``name``, once per name. A ``modal_only`` check
    joins ``MODAL_ONLY_CHECKS``: it reads the graph or coverage, so the driver runs it on
    the modal sample alone."""

    def decorate(fn: CheckFn) -> CheckFn:
        if name in _CHECKS:
            raise ValueError(f"check {name} is already registered")
        _CHECKS[name] = fn
        if modal_only:
            _MODAL_ONLY.add(name)
        return fn

    return decorate


def register_part(
    name: CheckName, *, case_check: bool = True
) -> Callable[[PartCheckFn], PartCheckFn]:
    """Register a part check under ``name``, and with ``case_check`` the case check that
    runs it over every part (``over_parts``). A check that also reads the response root
    passes ``case_check=False`` and registers its own case check."""

    def decorate(fn: PartCheckFn) -> PartCheckFn:
        if name in _PART_CHECKS:
            raise ValueError(f"part check {name} is already registered")
        _PART_CHECKS[name] = fn
        if case_check:
            register(name)(over_parts(fn, name=name))
        return fn

    return decorate


def response_parts(response: Mapping[str, Any], ctx: ScoreContext) -> list[PartView]:
    """A resolution's parts; a catalogue or declined answer has none."""
    if read_path(response, "kind", ctx) != "resolution":
        return []
    parts = read_path(response, "parts", ctx) or []
    return [
        PartView(index=i, body=part, status=read_path(part, "status", ctx, at=PartView.path))
        for i, part in enumerate(parts)
    ]


def problem_code(response: Mapping[str, Any]) -> str | None:
    """The registry code of a problem document, or None for any other body."""
    code = response.get("code")
    if "kind" in response or not isinstance(code, str):
        return None
    return code if isinstance(response.get("status"), int) else None


def answer_kind(response: Mapping[str, Any], ctx: ScoreContext) -> object:
    """The root ``kind``, or ``problem`` for a problem document (``problem_code``)."""
    return "problem" if problem_code(response) is not None else read_path(response, "kind", ctx)


@dataclass(frozen=True)
class ExpectedPart:
    """One expected part paired with the response: the expectation that applies to it, its
    position, the response part there, and the status block that part's status picks."""

    expectation: Expect | PartExpect
    index: int
    part: PartView | None
    block: StatusExpect | None
    """None when the part pairs with no block, for the reason ``missed`` gives."""
    missed: str = ""
    """Why the part pairs with no block: ``kind: got 'problem'`` for an answer of a kind the
    case does not accept, else the status the response returned or the part it did not."""


def expected_parts(
    case: Case, response: Mapping[str, Any], ctx: ScoreContext
) -> list[ExpectedPart]:
    """Each expected part paired with the response, in one pass.

    Without ``expect.parts``, each response part reads the root status against the case's
    expectation, so a ``mixed`` block applies to each part of a mixed answer; with them,
    each expected part reads the part at its position. An answer of another kind the case
    accepts has no expected part. One the case does not accept leaves every expected part
    unpaired."""
    expect = case.expect
    kind = answer_kind(response, ctx)
    if kind != "resolution":
        expected = [] if kind in expect.kinds else (expect.parts or [expect])
        missed = f"kind: got {kind!r}"
        return [ExpectedPart(e, i, None, None, missed) for i, e in enumerate(expected)]
    parts = response_parts(response, ctx)
    if expect.parts is None:
        status = read_path(response, "status", ctx)
        block = status_block(expect, status)
        missed = "" if block is not None else f"status: got {status!r}"
        return [ExpectedPart(expect, part.index, part, block, missed) for part in parts]
    return [
        _at_position(e, i, parts[i] if i < len(parts) else None) for i, e in enumerate(expect.parts)
    ]


def _at_position(expectation: PartExpect, index: int, part: PartView | None) -> ExpectedPart:
    if part is None:
        return ExpectedPart(expectation, index, None, None, f"parts[{index}]: not returned")
    block = status_block(expectation, part.status)
    missed = "" if block is not None else f"parts[{index}] status: got {part.status!r}"
    return ExpectedPart(expectation, index, part, block, missed)


def part_pairs(expected: Iterable[ExpectedPart]) -> list[tuple[StatusExpect, PartView]]:
    """Each response part with the status block that applies to it, leaving out the parts
    that pair with none: ``disposition`` and ``parts`` fail those."""
    return [(e.block, e.part) for e in expected if e.block is not None and e.part is not None]


def asserted(expectation: Expect | PartExpect, check: CheckName) -> bool:
    """Whether a block ``expectation`` holds for an accepted status asserts ``check``."""
    blocks = (expectation.for_status(s) for s in expectation.accepted_statuses)
    return any(block is not None and check in assertion_checks(block) for block in blocks)


def unreached(case: Case, expected: Iterable[ExpectedPart], *, check: CheckName) -> Result:
    """UNREACHED when an expected part pairs with no block while a block the expectation
    holds for it asserts ``check``, else n/a. The detail says why the first such part
    missed, and with ``expect.parts`` the result names them all."""
    missed = {
        e.index: e.missed for e in expected if e.block is None and asserted(e.expectation, check)
    }
    if not missed:
        return NA
    detail = next(iter(missed.values()))
    named = dict.fromkeys(missed, Verdict.UNREACHED) if case.expect.parts is not None else None
    return Result(Verdict.UNREACHED, detail, parts=named)


def by_part(case: Case, scored: Iterable[tuple[int, Result]], missed: Result) -> Result:
    """``all_of`` over each scored part's result and the ``unreached`` result, naming the
    parts that did not pass when the case has ``expect.parts``."""
    scored = list(scored)
    combined = all_of([*(result for _index, result in scored), missed])
    if case.expect.parts is None:
        return combined
    parts = {i: r.verdict for i, r in scored if r.verdict not in (Verdict.PASS, Verdict.NA)}
    return replace(combined, parts={**parts, **(missed.parts or {})})


def status_block(expectation: Expect | PartExpect, status: object) -> StatusExpect | None:
    """The block ``expectation`` holds for a returned ``status``, or None when it does not
    accept that status."""
    key = next((key for key in STATUS_KEYS if key == status), None)
    return expectation.for_status(key) if key is not None else None


def over_parts(fn: PartCheckFn, *, name: CheckName) -> CheckFn:
    """The case check running ``fn``, registered as ``name``, on every part that has a status
    block, and reading UNREACHED for a part that has none where a block it could have
    asserts ``name``."""

    def check(case: Case, response: Mapping[str, Any], ctx: ScoreContext) -> Result:
        expected = expected_parts(case, response, ctx)
        scored = (
            (part.index, run_part(fn, block, part, response, ctx))
            for block, part in part_pairs(expected)
        )
        return by_part(case, scored, unreached(case, expected, check=name))

    check.__doc__ = fn.__doc__
    return check


def run_part(
    fn: PartCheckFn,
    block: StatusExpect,
    part: PartView,
    response: Mapping[str, Any],
    ctx: ScoreContext,
) -> Result:
    """``fn`` on one part, its failure detail led by the part's position."""
    return guarded(lambda: fn(block, part, response, ctx)).at(f"parts[{part.index}]")


def score_results(
    case: Case,
    response: Mapping[str, Any],
    ctx: ScoreContext,
    *,
    skip: Collection[CheckName] = (),
) -> dict[str, Result]:
    """Every registered check's result with its detail, in ``CHECK_NAMES`` order, leaving
    out the checks in ``skip``."""
    return {
        name: _run(_CHECKS[name], case, response, ctx)
        for name in CHECK_NAMES
        if name in _CHECKS and name not in skip
    }


def _run(fn: CheckFn, case: Case, response: Mapping[str, Any], ctx: ScoreContext) -> Result:
    return guarded(lambda: fn(case, response, ctx))


ANSWER = "answer"
"""The derived check ``answer_result`` scores. It has no registered function, and no case may
list it in ``known_failure.checks``."""


def answer_result(results: Mapping[str, Result], *, known: Collection[str]) -> Result:
    """The derived ``answer`` check on one sample, from the check results it holds.

    Leaving out the checks in ``known``: a failure when any fails, else unverifiable when
    any is, else unreached when any is, else a pass. n/a counts as a pass. The detail names
    the first check behind the verdict, and the result is infra when any check's read failed.
    The parts behind it are those the checks that did not pass name, or None when one of
    them concerns the whole answer.
    """
    counted = {name: r for name, r in results.items() if name not in known}
    infra = any(r.infra for r in counted.values())
    for verdict in (Verdict.FAIL, Verdict.UNVERIFIABLE, Verdict.UNREACHED):
        first = next((name for name, r in counted.items() if r.verdict == verdict), None)
        if first is not None:
            return Result(verdict, first, infra=infra, parts=_merged_parts(counted.values()))
    return Result(Verdict.PASS, infra=infra)


def _merged_parts(results: Iterable[Result]) -> Mapping[int, Verdict] | None:
    merged: dict[int, Verdict] = {}
    for result in results:
        if result.verdict in (Verdict.PASS, Verdict.NA):
            continue
        if result.parts is None:
            return None
        for index, verdict in result.parts.items():
            merged[index] = all_of((Result(merged.get(index, Verdict.NA)), Result(verdict))).verdict
    return merged


def score_with_answer(
    case: Case,
    response: Mapping[str, Any],
    ctx: ScoreContext,
    *,
    skip: Collection[CheckName] = (),
) -> dict[str, Result]:
    """``score_results`` with the derived ``answer`` check last, leaving out the case's
    ``known_failure.checks``."""
    results = score_results(case, response, ctx, skip=skip)
    known = case.known_failure.checks if case.known_failure else ()
    results[ANSWER] = answer_result(results, known=known)
    return results


def score(
    case: Case,
    response: Mapping[str, Any],
    ctx: ScoreContext,
    *,
    skip: Collection[CheckName] = (),
) -> dict[str, Verdict]:
    """Every registered check's verdict for ``case`` on one raw response, leaving out the
    checks in ``skip``."""
    results = score_results(case, response, ctx, skip=skip)
    return {name: result.verdict for name, result in results.items()}
