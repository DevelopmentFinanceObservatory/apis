"""No served contract document names a removed field.

Search states how many StatVars an answer holds and serves, and the members door
(`POST /api/datasets/{id}/members`) names them, so no request takes a member list and no
response carries one. The composed app's OpenAPI document and Search's schema document
(`GET /api/search/schema`) are scanned property by property. The Datasets manifest keeps
`trace.resolved_stat_vars`, the record of what a version served, so that name is checked on
Search's trace alone.

The documents also omit the request's `include_sentence` and `ResolveOptions.max_candidates`
(the candidate cap is a server setting, and `max_candidates` stays on `CandidateSet` and the
capability caps), and the response's `restatement_reason`, `measure_label_named_values`,
`NoData.unit_alternatives_truncated`, and the trace's `applied_window`, `date_source`,
`pipeline_trace`, `n_recalled`, `unit_lexicon_version` and `declaration_version`. The
Datasets manifest trace keeps `applied_window`, so that name is checked on Search's trace
alone.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from dfo_apis.app import all_surfaces
from dfo_platform.app import create_app
from search.engine.app import _schema_payload

_GRAPH_ENV_VARS = (
    "GRAPH_DCP_BASE_URL",
    "GRAPH_DCP_TOKEN",
    "GRAPH_NODE_POOL_SIZE",
    "GRAPH_RESOLVE_TIMEOUT_SECONDS",
    "GRAPH_NODE_TIMEOUT_SECONDS",
)

# Gone from every model of every document.
_REMOVED_EVERYWHERE = frozenset(
    {
        "resolved_stat_var_dcids",
        "elected_stat_vars",
        "base_spec_id",
        "base_identity",
        "resolved_stat_vars_truncated",
        "resolved_stat_vars_max",
        "pipeline_trace",
        "n_recalled",
        "unit_lexicon_version",
        "declaration_version",
        "date_source",
        "restatement_reason",
        "measure_label_named_values",
        "unit_alternatives_truncated",
        "include_sentence",
    }
)
# Gone from one Search model, whose name other models still use for their own fields.
_REMOVED_FROM_MODEL = {
    "ResolvedStatVars": frozenset({"items", "truncated"}),
    "ResolutionTrace": frozenset({"resolved_stat_vars", "applied_window"}),
    "ResolveOptions": frozenset({"max_candidates"}),
    "RelatedMeasure": frozenset({"label"}),
    "ResubmitIdentityEntry": frozenset({"removed"}),
    "ResubmitCandidate": frozenset({"removed"}),
}


def _documents() -> dict[str, Mapping[str, Any]]:
    for var in _GRAPH_ENV_VARS:
        os.environ.pop(var, None)
    return {
        "openapi": create_app(all_surfaces()).openapi(),
        "search_schema": _schema_payload(),
    }


def _models(node: object, name: str = "") -> Iterator[tuple[str, frozenset[str]]]:
    """Each object schema under `node` with its property names, named by the component or
    `$defs` key that holds it, else by the nearest one above it."""
    if isinstance(node, Mapping):
        properties = node.get("properties")
        if isinstance(properties, Mapping):
            yield name, frozenset(properties)
        for key, child in node.items():
            if key in ("schemas", "$defs") and isinstance(child, Mapping):
                for model, schema in child.items():
                    yield from _models(schema, model)
            else:
                yield from _models(child, name)
    elif isinstance(node, list):
        for child in node:
            yield from _models(child, name)


@pytest.fixture(scope="module")
def documents() -> dict[str, Mapping[str, Any]]:
    return _documents()


@pytest.mark.parametrize("document", ["openapi", "search_schema"])
def test_no_model_names_a_removed_field(documents, document):
    found = {
        (model, field)
        for model, fields in _models(documents[document])
        for field in fields & _REMOVED_EVERYWHERE
    }
    assert found == set()


@pytest.mark.parametrize("document", ["openapi", "search_schema"])
def test_the_search_models_drop_their_removed_fields(documents, document):
    models = dict(_models(documents[document]))
    assert "ResolvedStatVar" not in models
    held = {model: models[model] & removed for model, removed in _REMOVED_FROM_MODEL.items()}
    assert held == dict.fromkeys(_REMOVED_FROM_MODEL, frozenset())
    assert {"served_count", "data_date_range"} <= models["ResolvedStatVars"]
    assert {"stat_var", "definition"} <= models["RelatedMeasure"]
    assert {"include_observations", "include_option_coverage"} <= models["ResolveOptions"]
