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
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, get_args

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
)
from series_coverage.loader import CoverageArtifactError

if TYPE_CHECKING:
    from search.engine.clients.sdmx import EngineSdmxClient
    from search.eval.graph import GraphClient
    from series_coverage.shared import CoverageService


STATUS_KEYS: tuple[StatusKey, ...] = get_args(StatusKey)


class Verdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNVERIFIABLE = "unverifiable"
    NA = "n/a"


@dataclass(frozen=True)
class Result:
    """A check's verdict with the reason a report shows beside it."""

    verdict: Verdict
    detail: str = ""
    infra: bool = False
    """The check could not read the graph or coverage it scores against (``infra_error``).
    The verdict is UNVERIFIABLE, and the driver reports the sample as infrastructure."""

    def at(self, prefix: str) -> Result:
        """The same verdict with ``prefix`` leading the detail of a failure."""
        if self.verdict in (Verdict.PASS, Verdict.NA):
            return self
        detail = f"{prefix}: {self.detail}" if self.detail else prefix
        return Result(self.verdict, detail, self.infra)


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
    """Every assertion must hold: the first failure, else the first unverifiable, else a
    pass when anything passed, else n/a."""
    collected = list(results)
    for verdict in (Verdict.FAIL, Verdict.UNVERIFIABLE):
        found = next((r for r in collected if r.verdict == verdict), None)
        if found is not None:
            return found
    return PASS if any(r.verdict == Verdict.PASS for r in collected) else NA


def any_of(results: Iterable[Result]) -> Result:
    """One alternative must hold: a pass, else the first unverifiable (it may pass), else
    the first failure, else n/a."""
    collected = list(results)
    for verdict in (Verdict.PASS, Verdict.UNVERIFIABLE, Verdict.FAIL):
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


@dataclass(frozen=True)
class ScoreContext:
    schema: WireSchema
    """The responding app's ``GET /api/search/schema``, parsed."""
    graph: GraphClient | None = None
    sdmx: EngineSdmxClient | None = None
    coverage: CoverageService | None = None
    side: Literal["head", "base"] = "head"


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
            register(name)(over_parts(fn))
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


def part_pairs(
    case: Case, response: Mapping[str, Any], ctx: ScoreContext
) -> list[tuple[StatusExpect, PartView]]:
    """Each response part with the status block that applies to it.

    Without ``expect.parts``, the block for the root status applies to every part, so a
    ``mixed`` block applies to each part of a mixed answer. With ``expect.parts``, each
    expected part pairs with the response part at its position. A part whose status the
    expectation does not accept pairs with nothing: ``disposition`` and ``parts`` fail it.
    """
    parts = response_parts(response, ctx)
    expect = case.expect
    if expect.parts is None:
        block = status_block(expect, read_path(response, "status", ctx)) if parts else None
        return [(block, part) for part in parts] if block is not None else []
    pairs = []
    for part_expect, part in zip(expect.parts, parts, strict=False):
        block = status_block(part_expect, part.status)
        if block is not None:
            pairs.append((block, part))
    return pairs


def status_block(expectation: Expect | PartExpect, status: object) -> StatusExpect | None:
    """The block ``expectation`` holds for a returned ``status``, or None when it does not
    accept that status."""
    key = next((key for key in STATUS_KEYS if key == status), None)
    return expectation.for_status(key) if key is not None else None


def over_parts(fn: PartCheckFn) -> CheckFn:
    """The case check running ``fn`` on every part that has a status block."""

    def check(case: Case, response: Mapping[str, Any], ctx: ScoreContext) -> Result:
        return all_of(
            run_part(fn, block, part, response, ctx)
            for block, part in part_pairs(case, response, ctx)
        )

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
    any is, else a pass. n/a counts as a pass. The detail names the first check behind the
    verdict, and the result is infra when any check's read failed.
    """
    counted = {name: r for name, r in results.items() if name not in known}
    infra = any(r.infra for r in counted.values())
    for verdict in (Verdict.FAIL, Verdict.UNVERIFIABLE):
        first = next((name for name, r in counted.items() if r.verdict == verdict), None)
        if first is not None:
            return Result(verdict, first, infra=infra)
    return Result(Verdict.PASS, infra=infra)


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
