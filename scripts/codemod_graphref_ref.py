"""Rename the dcid of every graph reference Search serves or accepts from `dcid` to `ref`.

A graph reference on the Search wire is `{ref, label}`, the shape the Graph API serves. The
script rewrites the repository in place and is idempotent: a second run changes nothing.

Models. The `dcid` field becomes `ref` on the models whose instances are a graph reference
with a display name, in `src/search/models`:

- `GraphRef` (grammar.py) and, by inheritance, `ProvenanceRef`;
- `AmbiguousEntityCandidate` (outcomes.py), one candidate of `NoData.ambiguous_entities`;
- `EntityGroup` (entity.py), served on `ResolvedSpec.entity_groups` and posted on
  `ResubmitIdentityEntry.entity_groups` and `ResubmitCandidate.entity_groups`.

Inside those class bodies a `"dcid"` subscript (the `documentation` default factory reads
`data["dcid"]`) becomes `"ref"`.

Python readers. ty reports every attribute read and constructor keyword the field rename
breaks, in `src`, `tests`, `scripts` and `promptlab`: an `unresolved-attribute` on `dcid`
for an object whose type is one of the models above, and an `unknown-argument` `dcid` in
a call to one of them. The script renames the attribute or keyword at each reported
position and runs ty again until it reports none. A `dcid` read on a reference field whose
receiver ty cannot type (`slot.key.property.dcid` with `slot` unannotated) is renamed too,
where the field name holds a renamed model, or a list of them, on every Search model that
declares it (`property`, `ref`, `population_type`, `refs`, `resolved_sources`, ...;
`value` is excluded, since it also names `SlotBinding.value` and pytest's
`ExceptionInfo.value`): a read on the field, on an item of the list, or on a loop
variable over the list. A rename ty then types as a miss stops the run. A dict literal whose string keys
include `dcid` and `label` and fall within the renamed models' fields (a `model_validate`
payload or a posted body) has its `dcid` key renamed too.

JSON. Each fixture below is walked beside the model type it records, and a `dcid` key
becomes `ref` only in an object the walk places at one of the models above:

- `tests/render/heading_words_specs.json` (spec key to recorded Spec) and
  `tests/render/mls_001_spec.json` (one recorded Spec), each Spec carrying its
  `SentenceFacts` under `sentence_facts`;
- `tests/fixtures/first_answer_place_scenarios.json`, whose `answer.specs`,
  `answer.flagged_match` and `answer.alternate_match` hold `ResolvedSpec` dumps;
- `tests/fixtures/sentence_panel/*.jsonl.gz`, one golden line per query, whose `parts` are
  `Part` and `sentence_plan` is a list of `SentenceToken`.

Each file is written back in the formatting it was read in; a file whose formatting the
script cannot reproduce stops the run.

YAML. Each evaluation corpus (`corpus` and
`tests/promptlab/fixtures/structured_door_corpus`) is walked the same way: a top-level
file as a list of `search.eval.cases.Case`, whose `resubmit` slots are a posted Search
request, and a file under `families/` as `Family` by id. The script renames each `dcid`
key in place in the text, so comments and layout stay as written.

Excluded. `ParsedEntity.dcid` (a dcid pinned on a parsed entity's `text`, no display name);
the flat `ref` of the catalogue and the Datasets manifests; `stat_var` columns; the
recorded graph request logs under `tests/fixtures/requests`, the graph node fixtures and
every other Data Commons payload, whose `dcid` is the graph's own key; and the expected
entities of an evaluation case (`search.eval.cases.ExpectedEntity`), which name what an
answer must bind and are never served or posted.

Every Python file the script changes is formatted with ruff afterwards.

Untyped readers of the wire JSON (a test indexing a response body, an evaluation check
reading a dumped body) are outside ty's reach and are edited by hand.

Usage:
    uv run python scripts/codemod_graphref_ref.py
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import gzip
import itertools
import json
import os
import re
import subprocess
import sys
import types
import typing
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel

_REPO = Path(__file__).resolve().parent.parent
_OLD, _NEW = "dcid", "ref"

_MODEL_FILES: dict[str, tuple[str, ...]] = {
    "src/search/models/grammar.py": ("GraphRef", "ProvenanceRef"),
    "src/search/models/outcomes.py": ("AmbiguousEntityCandidate",),
    "src/search/models/entity.py": ("EntityGroup",),
}
# The names ty prints for the renamed models' instances.
_TYPE_NAMES = frozenset({"GraphRef", "ProvenanceRef", "AmbiguousEntityCandidate", "EntityGroup"})
_PY_ROOTS = ("src", "tests", "scripts", "promptlab")
# The keys a renamed model's dump can carry: a dict literal holding `dcid` and `label` and no
# key outside these is one of them.
_REF_KEYS = frozenset(
    {_OLD, _NEW, "label", "source", "data_commons", "documentation", "kind", "entity_type", "total"}
)
_RENAMED_TYPE = re.compile(rf"\b(?:{'|'.join(sorted(_TYPE_NAMES))})\b")
_TY_DIAGNOSTIC = re.compile(
    r"^(?P<path>[^:]+):(?P<line>\d+):(?P<col>\d+): \w+\[(?P<rule>[\w-]+)\] (?P<message>.*)$"
)


# ---------------------------------------------------------------------------
# Python source
# ---------------------------------------------------------------------------


type _Edit = tuple[int, int, int, str]
"""A replacement: 1-based line, start and end byte columns, new text."""


def _apply(path: Path, edits: Iterable[_Edit]) -> bool:
    lines = path.read_bytes().splitlines(keepends=True)
    changed = False
    for line, start, end, text in sorted(set(edits), reverse=True):
        row = lines[line - 1]
        new = row[:start] + text.encode() + row[end:]
        changed |= new != row
        lines[line - 1] = new
    if changed:
        path.write_bytes(b"".join(lines))
    return changed


def _name_end(node: ast.Attribute) -> _Edit:
    """The edit renaming `node`'s attribute, which ends the node."""
    assert node.end_lineno is not None and node.end_col_offset is not None
    return (node.end_lineno, node.end_col_offset - len(_OLD), node.end_col_offset, _NEW)


def _keyword_edit(keyword: ast.keyword) -> _Edit:
    return (keyword.lineno, keyword.col_offset, keyword.col_offset + len(_OLD), _NEW)


def _constant_edit(node: ast.Constant) -> _Edit:
    """The edit renaming a quoted `dcid` string literal, keeping its quotes."""
    assert node.end_col_offset is not None
    return (node.lineno, node.col_offset + 1, node.end_col_offset - 1, _NEW)


def _rename_model_fields() -> list[Path]:
    """Rename the `dcid` field, and `"dcid"` subscripts, inside each renamed model's body."""
    changed = []
    for relative, classes in _MODEL_FILES.items():
        path = _REPO / relative
        edits: list[_Edit] = []
        for node in ast.walk(ast.parse(path.read_bytes())):
            if not (isinstance(node, ast.ClassDef) and node.name in classes):
                continue
            for item in node.body:
                target = item.target if isinstance(item, ast.AnnAssign) else None
                if isinstance(target, ast.Name) and target.id == _OLD:
                    edits.append(
                        (target.lineno, target.col_offset, target.col_offset + len(_OLD), _NEW)
                    )
            edits.extend(
                _constant_edit(inner.slice)
                for inner in ast.walk(node)
                if isinstance(inner, ast.Subscript)
                and isinstance(inner.slice, ast.Constant)
                and inner.slice.value == _OLD
            )
        if _apply(path, edits):
            changed.append(path)
    return changed


def _ty_diagnostics() -> list[tuple[Path, int, int, str, str]]:
    result = subprocess.run(
        # pyproject's ty include leaves the tests out; name every root to check them too.
        [
            *("uv", "run", "--no-sync", "ty", "check", "--output-format", "concise"),
            *("--config", f"src.include={json.dumps(list(_PY_ROOTS))}"),
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    diagnostics = []
    for row in result.stdout.splitlines():
        match = _TY_DIAGNOSTIC.match(row)
        if match:
            diagnostics.append(
                (
                    _REPO / match["path"],
                    int(match["line"]),
                    int(match["col"]),
                    match["rule"],
                    match["message"],
                )
            )
    return diagnostics


def _names_renamed_type(message: str) -> bool:
    """Whether a type ty prints in `message` (`GraphRef`, `GraphRef | None`,
    `GraphRef & ~AlwaysFalsy`) names a renamed model."""
    return any(_RENAMED_TYPE.search(printed) for printed in re.findall(r"`([^`]*)`", message))


def _ty_edits() -> dict[Path, list[_Edit]]:
    """The renames ty's diagnostics point at, by file. ty places an attribute diagnostic at
    the start of the attribute expression and an argument diagnostic at the keyword."""
    wanted: dict[Path, set[tuple[str, int, int]]] = defaultdict(set)
    for path, line, col, rule, message in _ty_diagnostics():
        if rule == "unresolved-attribute" and f"ttribute `{_OLD}`" in message:
            if _names_renamed_type(message):
                wanted[path].add(("attribute", line, col))
        elif rule == "unknown-argument" and f"`{_OLD}`" in message:
            wanted[path].add(("keyword", line, col))
    edits: dict[Path, list[_Edit]] = defaultdict(list)
    for path, positions in wanted.items():
        source = path.read_bytes()
        rows = source.splitlines()

        def position(node: ast.AST, rows: list[bytes] = rows) -> tuple[int, int]:
            """`node`'s start as ty prints it: 1-based line and character column."""
            line, offset = node.lineno, node.col_offset  # type: ignore[attr-defined]
            return line, len(rows[line - 1][:offset].decode()) + 1

        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Attribute) and node.attr == _OLD:
                if ("attribute", *position(node)) in positions:
                    edits[path].append(_name_end(node))
            elif isinstance(node, ast.Call) and _calls_renamed_model(node):
                edits[path].extend(
                    _keyword_edit(keyword)
                    for keyword in node.keywords
                    if keyword.arg == _OLD and ("keyword", *position(keyword)) in positions
                )
    return edits


def _calls_renamed_model(call: ast.Call) -> bool:
    func = call.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    return name in _TYPE_NAMES


def _python_files() -> list[Path]:
    files = []
    for root in _PY_ROOTS:
        files.extend(
            path
            for path in sorted((_REPO / root).rglob("*.py"))
            if ".venv" not in path.parts and path.resolve() != Path(__file__).resolve()
        )
    return files


def _dict_literal_edits(path: Path) -> list[_Edit]:
    """`dcid` keys of dict literals shaped as a renamed model's dump."""
    edits = []
    for node in ast.walk(ast.parse(path.read_bytes())):
        if not isinstance(node, ast.Dict):
            continue
        keys = [key for key in node.keys if isinstance(key, ast.Constant)]
        if len(keys) != len(node.keys) or not all(isinstance(key.value, str) for key in keys):
            continue
        names = {key.value for key in keys}
        if {_OLD, "label"} <= names and _NEW not in names and names <= _REF_KEYS:
            edits.extend(_constant_edit(key) for key in keys if key.value == _OLD)
    return edits


def _rewrite_python() -> list[Path]:
    changed = set(_rename_model_fields())
    for path in _python_files():
        if _apply(path, _dict_literal_edits(path)):
            changed.add(path)
    while edits := _ty_edits():
        moved = {path for path, file_edits in edits.items() if _apply(path, file_edits)}
        if not moved:
            break
        changed |= moved
    chained = {path: _chain_edits(path) for path in _python_files()}
    changed |= {path for path, edits in chained.items() if _apply(path, edits)}
    _refuse_chain_misreads(chained)
    if changed:
        # A shorter key can let ruff join a wrapped line; format as the repository does.
        subprocess.run(
            ["uv", "run", "--no-sync", "ruff", "format", "--quiet", *map(str, sorted(changed))],
            cwd=_REPO,
            check=True,
        )
    return sorted(changed)


# `ExceptionInfo.value` and `SlotBinding.value` share the name with `OpenAxisFact.value`.
_AMBIGUOUS_FIELDS = frozenset({"value"})


@functools.cache
def _reference_fields() -> tuple[frozenset[str], frozenset[str]]:
    """The Search fields (`_search_fields`) that hold renamed models: those holding one, and
    those holding a list of them. A name another Search field uses for other models is left
    out, since a read through it may reach either."""
    single, listed, other = set(), set(), set()
    for field_name, annotation in _search_fields():
        args = _flat_args(annotation)
        elements = [
            element
            for arg in args
            if typing.get_origin(arg) in (list, tuple)
            for item in typing.get_args(arg)
            for element in _flat_args(item)
        ]
        kinds = {
            arg
            for arg in (elements or args)
            if isinstance(arg, type)
            and (issubclass(arg, BaseModel) or dataclasses.is_dataclass(arg))
        }
        if kinds and all(issubclass(kind, _renamed_models()) for kind in kinds):
            (listed if elements else single).add(field_name)
        elif kinds:
            other.add(field_name)
    return (
        frozenset(single - other - _AMBIGUOUS_FIELDS),
        frozenset(listed - other - _AMBIGUOUS_FIELDS),
    )


def _search_fields() -> list[tuple[str, Any]]:
    """Each field of the Search wire models and of the engine's results
    (`search.engine.region.result`, which `RegionResult.elected_stat_vars` and the Specs it
    carries pass through), with its annotation."""
    # `search.engine.config` builds its settings at import and requires a graph base; the
    # script reads only class annotations and makes no request.
    os.environ.setdefault("SEARCH_GRAPH_BASE", "https://graph.invalid")
    import search.engine.region.result as results
    import search.models as models

    fields = []
    for module in (models, results):
        for name in dir(module):
            owner = getattr(module, name)
            if isinstance(owner, type) and issubclass(owner, BaseModel):
                fields += [(key, field.annotation) for key, field in owner.model_fields.items()]
            elif isinstance(owner, type) and dataclasses.is_dataclass(owner):
                fields += [
                    (field.name, _annotation(field.type, vars(sys.modules[owner.__module__])))
                    for field in dataclasses.fields(owner)
                ]
    return fields


def _annotation(annotation: Any, namespace: dict[str, Any]) -> Any:
    """A postponed dataclass annotation evaluated in its module, or None when it names a type
    imported only for type checking, which holds no Search model."""
    if not isinstance(annotation, str):
        return annotation
    try:
        return eval(annotation, namespace)
    except NameError:
        return None


def _flat_args(tp: Any) -> list[Any]:
    tp = _bare(tp)
    if typing.get_origin(tp) in (typing.Union, types.UnionType):
        return [arg for branch in typing.get_args(tp) for arg in _flat_args(branch)]
    return [tp]


def _iterated_list(iterable: ast.expr) -> str | None:
    """The field a loop iterates, `spec.trace.refs` or `spec.trace.refs or []`."""
    if isinstance(iterable, ast.BoolOp) and isinstance(iterable.op, ast.Or):
        iterable = iterable.values[0]
    return iterable.attr if isinstance(iterable, ast.Attribute) else None


def _chain_edits(path: Path) -> list[_Edit]:
    """`dcid` reads ty cannot type, such as through an unannotated parameter, on a reference
    field (`slot.key.property.dcid`), on an item of a reference list
    (`filter.refs[0].dcid`), and on a loop variable over one (`ref.dcid for ref in
    filter.refs`)."""
    single, listed = _reference_fields()
    tree = ast.parse(path.read_bytes())
    reads: list[ast.Attribute] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == _OLD:
            receiver = node.value
            if (isinstance(receiver, ast.Attribute) and receiver.attr in single) or (
                isinstance(receiver, ast.Subscript)
                and isinstance(receiver.value, ast.Attribute)
                and receiver.value.attr in listed
            ):
                reads.append(node)
        loops = (
            node.generators
            if isinstance(node, ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp)
            else [node]
            if isinstance(node, ast.For)
            else []
        )
        for loop in loops:
            if _iterated_list(loop.iter) in listed and isinstance(loop.target, ast.Name):
                reads.extend(
                    inner
                    for inner in ast.walk(node)
                    if isinstance(inner, ast.Attribute)
                    and inner.attr == _OLD
                    and isinstance(inner.value, ast.Name)
                    and inner.value.id == loop.target.id
                )
    return [_name_end(read) for read in reads]


def _refuse_chain_misreads(chained: dict[Path, list[_Edit]]) -> None:
    """Stop when ty types a renamed chain read's receiver as something without `ref`."""
    edited = {(path, line) for path, edits in chained.items() for line, *_ in edits}
    misreads = [
        f"{path.relative_to(_REPO)}:{line}: {message}"
        for path, line, _col, rule, message in _ty_diagnostics()
        if rule == "unresolved-attribute"
        and f"ttribute `{_NEW}`" in message
        and not _names_renamed_type(message)
        and (path, line) in edited
    ]
    if misreads:
        raise SystemExit("a chained rename reached a non-reference:\n" + "\n".join(misreads))


# ---------------------------------------------------------------------------
# JSON fixtures
# ---------------------------------------------------------------------------


def _renamed_models() -> tuple[type[BaseModel], ...]:
    from search.models import AmbiguousEntityCandidate, EntityGroup, GraphRef

    return (GraphRef, AmbiguousEntityCandidate, EntityGroup)


def _walk(tp: Any, value: Any, extras: dict[str, Any]) -> None:
    """Rename `dcid` in every object of `value` the type `tp` places at a renamed model.
    `extras` types keys a recording stores beside a model's own fields."""
    origin = typing.get_origin(tp)
    if origin is Annotated:
        _walk(typing.get_args(tp)[0], value, extras)
    elif isinstance(tp, typing.TypeAliasType):
        _walk(tp.__value__, value, extras)
    elif origin in (typing.Union, types.UnionType):
        for branch in _branches(typing.get_args(tp), value):
            _walk(branch, value, extras)
    elif origin in (list, tuple, set, frozenset, typing.Sequence) or _is_abc(origin, "Sequence"):
        if isinstance(value, list):
            args = [arg for arg in typing.get_args(tp) if arg is not Ellipsis]
            for index, item in enumerate(value):
                item_tp = args[index] if origin is tuple and len(args) > 1 else args[0]
                _walk(item_tp, item, extras)
    elif origin is dict or _is_abc(origin, "Mapping"):
        if isinstance(value, dict):
            for item in value.values():
                _walk(typing.get_args(tp)[1], item, extras)
    elif isinstance(tp, _Record):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in tp.fields:
                    _walk(tp.fields[key], item, extras)
    elif isinstance(tp, type) and issubclass(tp, BaseModel) and isinstance(value, dict):
        if issubclass(tp, _renamed_models()) and _OLD in value and _NEW not in value:
            if isinstance(value, _YamlMapping):
                value.edits.append(value.key_marks[_OLD])
            renamed = {(_NEW if key == _OLD else key): item for key, item in value.items()}
            value.clear()
            value.update(renamed)
        fields = {field.alias or name: field for name, field in tp.model_fields.items()}
        for key, item in value.items():
            if key in fields:
                _walk(fields[key].annotation, item, extras)
            elif key in extras:
                _walk(extras[key], item, extras)


def _is_abc(origin: Any, name: str) -> bool:
    import collections.abc

    return origin is getattr(collections.abc, name)


def _branches(args: tuple[Any, ...], value: Any) -> list[Any]:
    """The union branches `value` can be: for an object, the models whose fields hold its
    keys and whose literal discriminators it matches; otherwise every branch."""
    if not isinstance(value, dict):
        return [arg for arg in args if arg is not type(None)]
    branches = []
    for arg in args:
        model = _bare(arg)
        if not (isinstance(model, type) and issubclass(model, BaseModel)):
            if arg is not type(None):
                branches.append(arg)
            continue
        fields = {field.alias or name: field for name, field in model.model_fields.items()}
        keys = set(value) - {_OLD, _NEW, "sentence_facts"}
        if not keys <= set(fields):
            continue
        if any(
            typing.get_origin(fields[key].annotation) is Literal
            and value[key] not in typing.get_args(fields[key].annotation)
            for key in keys
        ):
            continue
        branches.append(arg)
    return branches


def _bare(tp: Any) -> Any:
    while typing.get_origin(tp) is Annotated:
        tp = typing.get_args(tp)[0]
    return tp


@dataclasses.dataclass(frozen=True, eq=False)
class _Record:
    """A recording's own object around model dumps: the type of each key it walks."""

    fields: dict[str, Any]


def _fixture_types() -> dict[str, Any]:
    from search.models import Part, SentenceFacts, SentenceToken, Spec

    resolved = Spec.model_fields["resolved"].annotation

    recorded_spec = _Record({"resolved": resolved})
    projection = _Record(
        {
            "specs": list[recorded_spec],
            "flagged_match": list[recorded_spec],
            "alternate_match": recorded_spec,
        }
    )
    golden_line = _Record({"parts": list[Part], "sentence_plan": list[SentenceToken] | None})

    return {
        "spec": Spec,
        "specs": dict[str, Spec],
        "scenarios": dict[str, _Record({"answer": projection})],
        "golden": golden_line,
        "extras": {"sentence_facts": SentenceFacts},
    }


_JSON_FIXTURES: tuple[tuple[str, str], ...] = (
    ("tests/render/heading_words_specs.json", "specs"),
    ("tests/render/mls_001_spec.json", "spec"),
    ("tests/fixtures/first_answer_place_scenarios.json", "scenarios"),
)
_GOLDEN_GLOB = "tests/fixtures/sentence_panel/*.jsonl.gz"


def _json_dumper(text: str, data: Any) -> Callable[[Any], str]:
    """The `json.dumps` call that reproduces `text` from `data`. A file whose keys are sorted
    keeps them sorted, as the recorders that write sorted fixtures do."""
    for indent, ensure_ascii, sort_keys, trailer in itertools.product(
        (None, 1, 2, 4), (False, True), (True, False), ("\n", "")
    ):

        def dump(value: Any, *, indent=indent, ascii=ensure_ascii, sort=sort_keys, end=trailer):
            return json.dumps(value, indent=indent, ensure_ascii=ascii, sort_keys=sort) + end

        if dump(data) == text:
            return dump
    raise SystemExit("cannot reproduce the formatting of a JSON fixture; stopping")


def _rewrite_json() -> list[Path]:
    fixture_types = _fixture_types()
    extras = fixture_types["extras"]
    changed = []
    for relative, kind in _JSON_FIXTURES:
        path = _REPO / relative
        text = path.read_text()
        data = json.loads(text)
        dump = _json_dumper(text, data)
        _walk(fixture_types[kind], data, extras)
        new = dump(data)
        if new != text:
            path.write_text(new)
            changed.append(path)
    for path in sorted(_REPO.glob(_GOLDEN_GLOB)):
        raw = path.read_bytes()
        lines = [json.loads(line) for line in gzip.decompress(raw).splitlines()]
        if _golden_text(lines) != gzip.decompress(raw).decode():
            raise SystemExit(f"cannot reproduce the formatting of {path}; stopping")
        for line in lines:
            _walk(fixture_types["golden"], line, extras)
        text = _golden_text(lines)
        if text != gzip.decompress(raw).decode():
            # As scripts/sentence_golden.py `write_golden` writes them.
            new = gzip.compress(text.encode(), compresslevel=9, mtime=0)
            path.write_bytes(new)
            changed.append(path)
    return changed


def _golden_text(lines: list[Any]) -> str:
    return "".join(json.dumps(line, separators=(",", ":")) + "\n" for line in lines)


# ---------------------------------------------------------------------------
# YAML evaluation corpora
# ---------------------------------------------------------------------------


class _YamlMapping(dict):
    """A YAML mapping read for `_walk`, with the position of each key, so a rename edits the
    key in the file's text and leaves every other byte as written."""

    def __init__(self, items: dict[str, Any], key_marks: dict[str, yaml.Mark], edits: list):
        super().__init__(items)
        self.key_marks = key_marks
        self.edits = edits


def _yaml_value(node: yaml.Node, loader: yaml.SafeLoader, edits: list[yaml.Mark]) -> Any:
    if isinstance(node, yaml.MappingNode):
        items, marks = {}, {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            items[key] = _yaml_value(value_node, loader, edits)
            marks[key] = key_node.start_mark
        return _YamlMapping(items, marks, edits)
    if isinstance(node, yaml.SequenceNode):
        return [_yaml_value(item, loader, edits) for item in node.value]
    return loader.construct_object(node)


def _corpus_types() -> dict[str, Any]:
    from search.eval.cases import Case, Family

    return {"cases": list[Case], "families": dict[str, Family]}


_CORPUS_DIRS = ("corpus", "tests/promptlab/fixtures/structured_door_corpus")


def _rewrite_yaml() -> list[Path]:
    """Rename `dcid` keys in each evaluation corpus file: its cases (`Case`), whose
    `resubmit` slots are posted as Search requests, and its families (`Family`)."""
    corpus_types = _corpus_types()
    changed = []
    for directory in _CORPUS_DIRS:
        root = _REPO / directory
        files = [(path, "cases") for path in sorted(root.glob("*.yaml"))]
        files += [(path, "families") for path in sorted(root.glob("families/*.yaml"))]
        for path, kind in files:
            text = path.read_text()
            loader = yaml.SafeLoader(text)
            node = loader.get_single_node()
            edits: list[yaml.Mark] = []
            if node is not None:
                _walk(corpus_types[kind], _yaml_value(node, loader, edits), {})
            lines = text.splitlines(keepends=True)
            for mark in edits:
                row = lines[mark.line]
                if row[mark.column : mark.column + len(_OLD)] != _OLD:
                    raise SystemExit(f"{path}:{mark.line + 1}: no `{_OLD}` key at the mark")
                lines[mark.line] = row[: mark.column] + _NEW + row[mark.column + len(_OLD) :]
            if edits:
                path.write_text("".join(lines))
                changed.append(path)
    return changed


def main() -> int:
    python = _rewrite_python()
    fixtures = [*_rewrite_json(), *_rewrite_yaml()]
    for path in [*python, *fixtures]:
        print(path.relative_to(_REPO))
    print(f"{len(python)} Python files and {len(fixtures)} fixtures changed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
