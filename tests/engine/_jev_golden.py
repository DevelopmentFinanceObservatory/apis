"""Golden items of the Jev binder: stored Jev answers and the output they give.

Each file under tests/fixtures/jev/golden is one dev item of the evaluation set: the
arguments of its bind call and its quantity read, the answers Jev gave to the questions of
both requests, and the expected output (`expected`). Values the JSON cannot hold carry a
tag (`decode`): ``$tuple``, ``$frozenset``, ``$dataclass``, ``$namedtuple``, ``$pydantic``
and ``$dict``, each naming its class by module and qualified name.

`replay` runs an item through the Jev request builders and `jev_assemble`, then through the
steps the engine applies after a binder returns (`bind.split_stray_bindings`,
`bind.rank_elected`, `_validate_requested_all_bindings`, `_validate_sole_option_bindings`),
and reads the result in the form `expected` holds (`canonical`).
"""

from __future__ import annotations

import dataclasses
import importlib
import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from code_index.shared import get_index
from search.engine.bind import (
    SlotBindingDraft,
    _UniversalBindOutput,
    rank_elected,
    split_stray_bindings,
)
from search.engine.binders.base import BindRequest
from search.engine.binders.jev_assemble import (
    Confidence,
    JevAnswers,
    assemble_quantity,
    bind_result,
)
from search.engine.binders.jev_menu import jev_menu
from search.engine.binders.jev_request import bind_request, quantity_request
from search.engine.quantity_read import QuantityOutput
from search.engine.region.phase_election import _validate_sole_option_bindings
from search.engine.region.phase_rank_bind import (
    _HousePolicyStep,
    _names_base,
    _validate_requested_all_bindings,
)
from search.engine.shape import ShapeDraft

GOLDEN = Path(__file__).parents[1] / "fixtures" / "jev" / "golden"
_BIND_FIELDS = (
    "variable",
    "shapes",
    "admitted_qualifiers",
    "admitted_ties",
    "withheld_measure_axes",
    "dropped_qualifier_phrases",
    "candidate_menus",
    "frame",
    "refused",
    "availability",
)
# Stored answers to questions the bind request no longer asks.
_UNASKED_PREFIXES = ("topic", "within|")


def _resolve(name: str) -> Any:
    module, _, qualname = name.rpartition(".")
    while module:
        try:
            obj: Any = importlib.import_module(module)
        except ModuleNotFoundError:
            module, _, head = module.rpartition(".")
            qualname = f"{head}.{qualname}"
            continue
        for part in qualname.split("."):
            obj = getattr(obj, part)
        return obj
    raise ModuleNotFoundError(name)


def decode(data: Any) -> Any:
    """The object a tagged JSON value encodes, built from the current engine classes."""
    if isinstance(data, list):
        return [decode(v) for v in data]
    if not isinstance(data, dict):
        return data
    if "$pydantic" in data:
        return _resolve(data["$pydantic"]).model_validate(data["data"])
    if "$dataclass" in data:
        return _resolve(data["$dataclass"])(**{k: decode(v) for k, v in data["fields"].items()})
    if "$namedtuple" in data:
        return _resolve(data["$namedtuple"])(*(decode(v) for v in data["items"]))
    if "$tuple" in data:
        return tuple(decode(v) for v in data["$tuple"])
    if "$frozenset" in data:
        return frozenset(decode(v) for v in data["$frozenset"])
    if "$dict" in data:
        return {decode(k): decode(v) for k, v in data["$dict"]}
    return {k: decode(v) for k, v in data.items()}


def golden_paths(directory: Path = GOLDEN) -> list[Path]:
    """The item files of `directory`, in name order."""
    return sorted(directory.glob("*.json"))


def answers_of(stored: dict[str, Any]) -> JevAnswers:
    """Stored answers as the Jev client returns them: each noul's probability and each
    choice's probability per option."""
    return JevAnswers(
        nouls={k: float(v["noul"]) for k, v in stored.items() if v["type"] == "noul"},
        choices={
            k: {o: float(p) for o, p in v["probabilities"].items()}
            for k, v in stored.items()
            if v["type"] == "choice"
        },
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class GoldenItem:
    """One golden item: the bind request, the quantity read's phrase and denominators, the
    stored answers, and the stored expected output."""

    item: str
    request: BindRequest
    policies: _HousePolicyStep
    quantity_variable: str
    denominators: list[Any]
    stored: dict[str, Any]
    expected: dict[str, Any]

    @property
    def answers(self) -> JevAnswers:
        return answers_of(self.stored)


def load_item(path: Path) -> GoldenItem:
    """The item `path` holds, its request asked with a query, as the stored answers were."""
    record = json.loads(path.read_text())
    args = {name: decode(record["args"][name]) for name in _BIND_FIELDS}
    step = _HousePolicyStep(
        shapes=args["shapes"],
        frame=args["frame"],
        admitted_qualifiers=args["admitted_qualifiers"],
        availability=args["availability"],
        code_index=get_index(),
    )
    request = BindRequest(**args, query_text="", apply_policies=step)
    quantity = {k: decode(v) for k, v in record["quantity_args"].items()}
    return GoldenItem(
        item=record["item"],
        request=request,
        policies=step,
        quantity_variable=quantity["variable"],
        denominators=list(quantity["denominators"]),
        stored=record["answers"],
        expected=record["expected"],
    )


def ground(output: QuantityOutput, variable: str, shapes: Sequence[ShapeDraft]) -> QuantityOutput:
    """`output` as the engine keeps it (`_read_quantity_answer`): the index, and a rate or
    share on the quote, drop when the quote names no base in `variable` (`_names_base`)."""
    state = SimpleNamespace(variable=variable, entities=(), entity_roles={})
    # The stub holds the three fields `_names_base` reads.
    if _names_base(state, output.base_words, shapes):  # ty: ignore[invalid-argument-type]
        return output
    stated_form = output.stated_form
    if output.base_words and stated_form in ("rate", "share"):
        stated_form = None
    return output.model_copy(update={"denominator_index": None, "stated_form": stated_form})


def asked_ids(item: GoldenItem) -> tuple[list[str], list[str]]:
    """The question ids of the bind request and of the quantity request."""
    _state, bind = bind_request(jev_menu(item.request))
    _qstate, quantity = quantity_request(
        item.quantity_variable,
        item.denominators,
        bind_variable=item.request.variable,
        query=item.request.query_text,
    )
    return list(bind), list(quantity)


def stored_ids(item: GoldenItem) -> list[str]:
    """The stored answer ids the requests ask: every one but ``topic`` and ``within|i``."""
    return sorted(k for k in item.stored if not k.startswith(_UNASKED_PREFIXES))


def canonical(
    output: _UniversalBindOutput,
    strays: Sequence[SlotBindingDraft],
    elected: int | None,
    requested_all: frozenset[str],
) -> dict[str, Any]:
    """The fields downstream reads, each in a form two outputs compare by equality, as the
    golden files hold them."""
    return {
        "shape": elected if elected is not None else "ask",
        "bindings": sorted(
            [d.axis, d.property_dcid or "", d.kind, sorted(d.value_dcids), d.evidence]
            for d in output.bindings
            if d.axis != "source"
        ),
        "also_plausible": sorted(
            set(output.also_plausible or ()) - {output.chosen_shape_index, elected}
        ),
        "ask": bool(output.ask),
        "nearest_refused": output.nearest_refused_index if output.ask else None,
        "strays": sorted([d.property_dcid or "", sorted(d.value_dcids)] for d in strays),
        "requested_all": sorted(requested_all),
    }


def after_bind(output: _UniversalBindOutput, request: BindRequest) -> dict[str, Any]:
    """`canonical` of a binder's output after the steps the engine applies once the binder
    returns: the stray split, the menu-rank election, and the ``requested_all`` and
    ``sole_option`` checks on the elected shape."""
    shapes = request.shapes
    output, strays = split_stray_bindings(output, shapes)
    if output.ask or not 0 <= output.chosen_shape_index < len(shapes):
        return canonical(output, strays, None, frozenset())
    elected = rank_elected(shapes, output, availability=request.availability)
    slot_properties = frozenset(
        s.property_dcid for s in shapes[elected].slot_keys if s.property_dcid is not None
    )
    validated, applied = _validate_requested_all_bindings(
        list(output.bindings),
        list(request.dropped_qualifier_phrases),
        slot_properties=slot_properties,
    )
    settled = _validate_sole_option_bindings(validated, shapes[elected])
    return canonical(output.model_copy(update={"bindings": settled}), strays, elected, applied)


def replay(
    item: GoldenItem, *, answers: JevAnswers | None = None
) -> tuple[dict[str, Any], Confidence]:
    """The item's output from its stored answers, or from `answers` when given, in the form
    `expected` holds: the bind output, the quantity read and the election rules."""
    answers = answers or item.answers
    shapes = item.request.shapes
    quantity = ground(assemble_quantity(answers, item.denominators), item.quantity_variable, shapes)
    result, conf = bind_result(jev_menu(item.request), answers, stated_form=quantity.stated_form)
    out = {
        "bind": after_bind(result.output, item.request),
        "quantity": quantity.model_dump(mode="json"),
        "rules": conf.rules,
    }
    return out, conf
