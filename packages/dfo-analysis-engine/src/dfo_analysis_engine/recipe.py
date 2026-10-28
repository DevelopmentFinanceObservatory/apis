"""A derived frame's recipe: the tree of op-specs that computed it, down to the
root datasets, and the two hashes apis indexes it by.

This module holds the part of a recipe that needs no store: the error types, the
normal form and the two hashes. The walk of the materialisation store from a
derived frame's pair, which builds the recipe, is outside this package. Each
derived frame's stored trace carries the canonical op-spec that produced it
(`execution_record.content`, the RFC 8785 text `identity.canonicalize_op_spec`
wrote), and the op-spec names its input frames. An input with an `ops_` id is
derived and is followed to its own trace. An input with a `spec_` id is a root,
described by its mint record. An op-spec is read as the parsed JSON object.

The recipe is the JSON object the apis service's publish door stores:

- `tree`: a node per frame. A derived node is `{"frame", "op_spec", "inputs"}`,
  where `op_spec` is the parsed canonical op-spec and `inputs` maps each of its
  frame slots to the node of the frame in that slot. The slots are `frame_ref`,
  and every parameter whose name ends `_frame_ref` and is set, keyed by that
  name. A root node is `{"frame", "root"}`, with `root` an index into `roots`.
- `roots`: each root's mint record, once per root frame, in the order a
  depth-first walk of `tree` first reaches them (slots in the order above,
  parameters by name).
- `normal_form_hash`, `replay_key`, `execution_identity`, and `anchor`.

The normal form (`normal_form`) is the recipe with everything a replay on newer
data changes taken out. A derived node becomes its verb, its parameters and its
inputs' normal forms, so no frame id appears. A root becomes its dataset's
`spec_id` and the mint `options` that select its rendering, without its
`content_version`. A literal a rolling overlay governs is dropped, since a replay
rewrites it: on an op-spec whose `rolling` parameter is an object, each member
that is a rolling reference (an object with an `anchor`) names the sibling
parameter it governs, and a predicate whose `rolling` is set has its `value`
governed. The overlay itself stays. Two frames computed by one recipe from
different versions of the same datasets therefore share a normal form. A derived
node that names `anchor_roots`, the roots its tree's latest period is taken over
(`dfo_analysis_engine.evaluate`), keeps them as their roots' normal forms, in order,
and each is a root of `roots` in the walk order, after the node's inputs.

`normal_form_hash` is `rnf_` and 32 hex digits of the SHA-256 of the RFC 8785
form of `{"identity_version": "rnf1", "recipe": <normal form>}`.

`replay_key` is `rpk_` and 32 hex digits of the SHA-256 of the RFC 8785 form of
`{"identity_version": "rpk1", "normal_form_hash", "roots", "execution_identities"}`.
`roots` holds each root's identity (`root_identity`) in `roots` order: its apis
pair, and the enrichment stamp its frame was named under when it has one. A root
with no stamp is identified by its pair alone. `execution_identities` holds every
distinct `ExecutionIdentity.canonical_string()` in the tree, sorted. A replay
computes the key before it runs: the hash from the stored recipe, the roots'
current identities, and the one identity of the engine replaying. A frame whose
ancestors were computed by different engine builds lists
each of them, so its key is not the key of a replay under any single build.

`execution_identity` is the identity of the frame the recipe is for.

`anchor` is null for a tree with no rolling op-spec. For a tree with one it is
`{"period", "rule", "left_out"}`, read from the anchor record
the rolling calls were checked against:
the latest period, the rule that chose it, and each series that holds no value
at it as `{"root", "coordinates"}`, `root` indexing the populations the rolling
op-specs carry. `recipe_anchor` finds the record by recomputing its key from
those populations and the root frames the tree's rolling periods are taken over
(`dfo_analysis_engine.rolling.rolling_roots`), which are the roots under its rolling op-specs. A
dataset joined beside a rolling step is no part of the key.

A frame can be part of a wider plan than its own tree: its rolling op-specs then
carry populations naming a dataset none of its rolling periods reads, and the
period was taken over that dataset too. Such a frame has no anchor of its own.
`build_recipe` raises `RecipeAnchorError` with reason
`populations_outside_reach` and the rest of the recipe on the error, so a caller
can replay it over the populations of its own tree
(`dfo_analysis_engine.rolling.anchor_populations`), which gives a frame with an anchor.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

import rfc8785

from dfo_analysis_engine.frame_ref import FrameRef

__all__ = [
    "RECIPE_SCHEMA_VERSION",
    "RecipeAncestorMissingError",
    "RecipeAnchorError",
    "RecipeError",
    "RecipeRootUnrecoverableError",
    "RecipeTraceUnusableError",
    "derived_nodes",
    "input_slots",
    "normal_form",
    "normal_form_hash",
    "pair",
    "replay_key",
    "rolling_literals",
    "root_identity",
]

RECIPE_SCHEMA_VERSION = 1

NORMAL_FORM_IDENTITY_VERSION = "rnf1"
NORMAL_FORM_HASH_PREFIX = "rnf_"
REPLAY_KEY_IDENTITY_VERSION = "rpk1"
REPLAY_KEY_PREFIX = "rpk_"

_DIGEST_HEX_LENGTH = 32
_PRIMARY_SLOT = "frame_ref"
_SLOT_SUFFIX = "_frame_ref"
_ROLLING = "rolling"
_ANCHOR_ROOTS = "anchor_roots"


class RecipeError(Exception):
    """No recipe can be built. `ref` is the frame the walk stopped at."""

    def __init__(self, ref: FrameRef, message: str) -> None:
        self.ref = ref
        super().__init__(message)


class RecipeAncestorMissingError(RecipeError):
    """The store holds no trace for the derived frame `ref`, which is the frame the
    recipe was asked for or one of its ancestors."""

    def __init__(self, ref: FrameRef) -> None:
        super().__init__(
            ref, f"no trace is held for the derived frame {ref.spec_id}, {ref.content_version}"
        )


class RecipeTraceUnusableError(RecipeError):
    """The store holds something under the derived frame `ref` that no op-spec can
    be read from: the object fails a store check, its trace carries no canonical
    op-spec, or the op-spec does not hash to `ref.spec_id`."""

    def __init__(self, ref: FrameRef, *, reason: str) -> None:
        self.reason = reason
        super().__init__(
            ref,
            f"the trace of the derived frame {ref.spec_id}, {ref.content_version} "
            f"is unusable: {reason}",
        )


class RecipeRootUnrecoverableError(RecipeError):
    """The root frame `ref` has no mint record and none could be rebuilt. `reason` is
    `no_mint_record` when no recovery routine was given, else the routine's own."""

    def __init__(self, ref: FrameRef, *, reason: str) -> None:
        self.reason = reason
        super().__init__(
            ref,
            f"no mint record for the root frame {ref.spec_id}, {ref.content_version}: {reason}",
        )


class RecipeAnchorError(RecipeError):
    """The tree of the derived frame `ref` holds a rolling op-spec and its anchor
    cannot be stated. `reason` is `populations_differ` or `rules_differ` when two
    rolling op-specs of the tree disagree, `populations_outside_reach` when their
    populations name a dataset no rolling period of the tree reads,
    `no_anchor_record` when the store holds no record under the key the tree
    gives, `anchor_record_unreadable` when it holds one that fails its check, and
    `period_differs` when a rolling literal is not the record's period at its
    offset.

    `recipe` is set by `build_recipe` for `populations_outside_reach` alone: the
    recipe it would have returned, with a null `anchor`."""

    def __init__(self, ref: FrameRef, *, reason: str) -> None:
        self.reason = reason
        self.recipe: dict[str, Any] | None = None
        super().__init__(
            ref,
            f"no anchor for the rolling recipe of {ref.spec_id}, {ref.content_version}: {reason}",
        )


def _digest(preimage: Mapping[str, Any], *, prefix: str) -> str:
    return f"{prefix}{hashlib.sha256(rfc8785.dumps(preimage)).hexdigest()[:_DIGEST_HEX_LENGTH]}"


def pair(ref: FrameRef) -> dict[str, str]:
    """`ref` as the `{spec_id, content_version}` object a tree node holds."""
    return {"spec_id": ref.spec_id, "content_version": ref.content_version}


def input_slots(op_spec: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    """Each frame slot `op_spec` fills, with the pair in it: `frame_ref`, then every
    set parameter named `*_frame_ref`, by name."""
    parameters = op_spec["parameters"]
    return [
        (_PRIMARY_SLOT, op_spec[_PRIMARY_SLOT]),
        *(
            (name, parameters[name])
            for name in sorted(parameters)
            if name.endswith(_SLOT_SUFFIX) and parameters[name] is not None
        ),
    ]


def _is_rolling_ref(value: object) -> bool:
    return isinstance(value, Mapping) and "anchor" in value


def _normal_predicates(predicates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """`predicates` without the `value` of each one a rolling reference governs, in
    the order of their RFC 8785 bytes, which is the order an op-spec keeps its
    predicates in before any literal is dropped."""
    normal = [
        {
            key: value
            for key, value in predicate.items()
            if not (key == "value" and predicate.get(_ROLLING) is not None)
        }
        for predicate in predicates
    ]
    return sorted(normal, key=rfc8785.dumps)


def _normal_parameters(parameters: Mapping[str, Any], *, slots: Iterable[str]) -> dict[str, Any]:
    """`parameters` without its filled frame slots and without every literal a
    rolling overlay governs."""
    overlay = parameters.get(_ROLLING)
    governed = (
        {name for name, value in overlay.items() if _is_rolling_ref(value)}
        if isinstance(overlay, Mapping)
        else set()
    )
    dropped = governed | set(slots)
    normal = {name: value for name, value in parameters.items() if name not in dropped}
    if isinstance(normal.get("predicates"), list):
        normal["predicates"] = _normal_predicates(normal["predicates"])
    return normal


def normal_form(tree: Mapping[str, Any], roots: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The normal form of a recipe's `tree` over its `roots`. See the module docstring."""
    if "root" in tree:
        root = roots[tree["root"]]
        return {"spec_id": root["dataset"]["spec_id"], "rendering": root["options"]}
    op_spec = tree["op_spec"]
    inputs = tree["inputs"]
    normal = {
        "identity_version": op_spec["identity_version"],
        "verb": op_spec["verb"],
        "parameters": _normal_parameters(op_spec["parameters"], slots=inputs),
        "inputs": {slot: normal_form(node, roots) for slot, node in inputs.items()},
    }
    if anchors := tree.get(_ANCHOR_ROOTS):
        normal[_ANCHOR_ROOTS] = [normal_form(node, roots) for node in anchors]
    return normal


def normal_form_hash(normal: Mapping[str, Any]) -> str:
    """`rnf_` and 32 hex digits naming the normal form `normal`."""
    return _digest(
        {"identity_version": NORMAL_FORM_IDENTITY_VERSION, "recipe": normal},
        prefix=NORMAL_FORM_HASH_PREFIX,
    )


def root_identity(dataset: Mapping[str, str], *, enrichment_stamp: str | None) -> dict[str, str]:
    """A root's identity: the apis `{spec_id, content_version}` its frame was
    minted from, and the enrichment stamp the frame was named under
    (`MintRecord.enrichment_stamp`) when it has one. apis keeps a version's
    `content_version` across a change of party kinds, so the stamp is what tells two
    frames of one version apart.
    Without a stamp the identity is the pair alone."""
    identity = {"spec_id": dataset["spec_id"], "content_version": dataset["content_version"]}
    if enrichment_stamp is not None:
        identity["enrichment_stamp"] = enrichment_stamp
    return identity


def replay_key(
    *,
    normal_form_hash: str,
    root_datasets: Sequence[Mapping[str, str]],
    execution_identities: Iterable[str],
) -> str:
    """`rpk_` and 32 hex digits naming one replay of a recipe.

    Args:
        normal_form_hash: The recipe's `normal_form_hash`.
        root_datasets: Each root's identity (`root_identity`), in the recipe's
            `roots` order. A bare `{spec_id, content_version}` is a root with no stamp.
        execution_identities: The `ExecutionIdentity.canonical_string()` of every
            frame in the tree. Order and repeats do not matter.
    """
    return _digest(
        {
            "identity_version": REPLAY_KEY_IDENTITY_VERSION,
            "normal_form_hash": normal_form_hash,
            "roots": [
                root_identity(root, enrichment_stamp=root.get("enrichment_stamp"))
                for root in root_datasets
            ],
            "execution_identities": sorted(set(execution_identities)),
        },
        prefix=REPLAY_KEY_PREFIX,
    )


def rolling_literals(parameters: Mapping[str, Any]) -> Iterator[tuple[str, Mapping[str, Any]]]:
    """Each literal period of `parameters` a rolling reference governs, with the
    reference: a window bound named in `rolling`, and a rolling predicate's `value`."""
    overlay = parameters.get(_ROLLING)
    if isinstance(overlay, Mapping):
        for name, ref in overlay.items():
            if _is_rolling_ref(ref):
                yield parameters[name], ref
    for predicate in parameters.get("predicates", ()):
        if _is_rolling_ref(predicate.get(_ROLLING)):
            yield predicate["value"], predicate[_ROLLING]


def derived_nodes(tree: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    """Every derived node of `tree`, depth first, `tree` itself first."""
    if "root" in tree:
        return
    yield tree
    for node in tree["inputs"].values():
        yield from derived_nodes(node)
