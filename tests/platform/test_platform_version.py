"""PLATFORM_CAPABILITY_VERSION: the token's exact value, determinism, its
distinctness from a mounted surface's own token, and sensitivity to every
input _compute_platform_capability_version actually reads.
"""

from __future__ import annotations

import ast
import asyncio
import copy
import dataclasses
import inspect
import logging
import os
import subprocess
import sys
from collections.abc import Mapping
from typing import cast

import pytest
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from dfo_platform import codes, refusal, version
from dfo_platform.codes import PLATFORM_CODE_CATALOGUE
from dfo_platform.problem import ProblemDetails, ProblemSpec
from dfo_platform.trace import _current as _trace_current
from dfo_platform.validation import validation_problem
from dfo_platform.version import PLATFORM_CAPABILITY_VERSION, _compute_platform_capability_version
from search.engine.config import settings


def test_platform_capability_version_is_pinned():
    """Pins the token so changes to the platform wire contract require an explicit version update."""
    # The digest includes the `cause` extension and registry mappings rendered by `registry_entry_body`.
    assert PLATFORM_CAPABILITY_VERSION == "plat_d0d03f7f48032ca9"
    assert PLATFORM_CAPABILITY_VERSION.startswith("plat_")
    assert settings().capability_version != PLATFORM_CAPABILITY_VERSION


def test_platform_capability_version_is_stable_across_processes():
    """A fresh process with a different `PYTHONHASHSEED` checks digest determinism."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from dfo_platform.version import _compute_platform_capability_version; "
            "print(_compute_platform_capability_version())",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONHASHSEED": "12345"},
        check=True,
    )
    assert result.stdout.strip() == _compute_platform_capability_version()


def test_importing_version_leaves_no_trace_id_bound():
    """Import-time exemplar rendering must leave the trace context unbound."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import dfo_platform.version\n"
            "from dfo_platform.trace import _current\n"
            "print(_current.get())\n",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "None"


def test_no_trace_id_bound_after_calling_compute_directly():
    """Direct token computation must leave the caller's trace context unbound."""
    token = _trace_current.set(None)
    try:
        _compute_platform_capability_version()
        _compute_platform_capability_version()
        assert _trace_current.get() is None
    finally:
        _trace_current.reset(token)


def test_digest_moves_when_a_status_code_entry_changes():
    """A change to the status-to-code policy changes the digest."""
    baseline = _compute_platform_capability_version()
    perturbed_policy = copy.deepcopy(refusal.as_canonical_data())
    status_code_map = cast(dict[str, str], perturbed_policy["status_code_map"])
    status_code_map["404"] = "something_else"
    perturbed = _compute_platform_capability_version(refusal_policy=perturbed_policy)
    assert perturbed != baseline


def test_digest_moves_when_a_retryable_value_flips():
    baseline = _compute_platform_capability_version()
    perturbed_policy = copy.deepcopy(refusal.as_canonical_data())
    timeout_default = cast(dict[str, object], perturbed_policy["timeout_default"])
    timeout_default["retryable"] = False
    perturbed = _compute_platform_capability_version(refusal_policy=perturbed_policy)
    assert perturbed != baseline


def test_digest_moves_when_a_catalogue_entry_title_changes():
    baseline = _compute_platform_capability_version()
    perturbed_codes = dict(PLATFORM_CODE_CATALOGUE)
    title, detail = perturbed_codes["internal_error"]
    perturbed_codes["internal_error"] = (f"{title} (reworded)", detail)
    perturbed = _compute_platform_capability_version(platform_codes=perturbed_codes)
    assert perturbed != baseline


def test_digest_is_unaffected_by_a_reworded_field_description():
    """Field descriptions appear in JSON Schema, so rewording one leaves the wire digest unchanged."""
    baseline = _compute_platform_capability_version()
    reworded_schema = copy.deepcopy(ProblemDetails.model_json_schema())
    reworded_schema["properties"]["detail"]["description"] = "a totally different description"
    perturbed = _compute_platform_capability_version(problem_details_schema=reworded_schema)
    assert perturbed == baseline


def test_digest_moves_when_the_code_not_found_refusal_status_changes():
    """The `/codes/{code}` miss response contributes its status to the platform digest."""
    baseline = _compute_platform_capability_version()
    perturbed_policy = copy.deepcopy(refusal.as_canonical_data())
    code_not_found_default = cast(dict[str, object], perturbed_policy["code_not_found_default"])
    code_not_found_default["status"] = 400
    perturbed = _compute_platform_capability_version(refusal_policy=perturbed_policy)
    assert perturbed != baseline


def test_digest_moves_when_code_for_status_logic_changes(monkeypatch):
    """Changing `code_for_status` alone must change the digest, alongside its policy data."""
    baseline = _compute_platform_capability_version()
    monkeypatch.setattr(refusal, "code_for_status", lambda status: "internal_error")
    perturbed = _compute_platform_capability_version()
    assert perturbed != baseline


def test_digest_moves_when_render_timeout_refusal_logic_changes(monkeypatch):
    """Changing timeout rendering logic alone must change the digest."""
    baseline = _compute_platform_capability_version()

    def _different_timeout_refusal():
        return ProblemSpec(status=504, code="timeout", retryable=False)

    monkeypatch.setattr(refusal, "render_timeout_refusal", _different_timeout_refusal)
    perturbed = _compute_platform_capability_version()
    assert perturbed != baseline


def test_digest_moves_when_render_http_exception_refusal_detail_handling_changes(monkeypatch):
    """Changing HTTP-exception detail handling alone must change the digest."""
    baseline = _compute_platform_capability_version()

    def _always_none_detail(*, status, detail, headers):
        return ProblemSpec(
            status=status,
            code=refusal.code_for_status(status),
            detail=None,
            retryable=refusal.HTTP_EXCEPTION_DEFAULT_RETRYABLE,
            headers=headers,
        )

    monkeypatch.setattr(refusal, "render_http_exception_refusal", _always_none_detail)
    perturbed = _compute_platform_capability_version()
    assert perturbed != baseline


def test_digest_moves_when_a_default_exemplar_is_added(monkeypatch):
    """Adding a default refusal exemplar must change the digest."""
    baseline = _compute_platform_capability_version()
    monkeypatch.setattr(
        refusal,
        "_DEFAULT_EXEMPLARS",
        (
            *refusal._DEFAULT_EXEMPLARS,
            StarletteHTTPException(status_code=418, detail=None, headers=None),
        ),
    )
    perturbed = _compute_platform_capability_version()
    assert perturbed != baseline


def test_computing_the_token_emits_no_refusal_logger_records(caplog):
    """A recompute suppresses refusal logger records even when the logger is set to DEBUG."""
    caplog.set_level(logging.DEBUG, logger=refusal.__name__)
    _compute_platform_capability_version()
    assert [r for r in caplog.records if r.name == refusal.__name__] == []


def test_refusal_logger_level_restored_after_a_normal_recompute():
    """A normal recompute restores the refusal logger's prior level."""
    refusal_logger = logging.getLogger(refusal.__name__)
    prior_level = refusal_logger.level
    _compute_platform_capability_version()
    assert refusal_logger.level == prior_level


def test_refusal_logger_level_restored_to_notset_after_a_normal_recompute():
    """A recompute preserves the refusal logger's `NOTSET` level."""
    refusal_logger = logging.getLogger(refusal.__name__)
    refusal_logger.setLevel(logging.NOTSET)
    _compute_platform_capability_version()
    assert refusal_logger.level == logging.NOTSET


def test_refusal_logger_level_restored_after_an_exemplar_render_raises(monkeypatch):
    """A rendering error propagates and the logger level is restored."""
    refusal_logger = logging.getLogger(refusal.__name__)
    prior_level = refusal_logger.level

    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(refusal, "render_timeout_refusal", _raise)
    with pytest.raises(RuntimeError, match="boom"):
        _compute_platform_capability_version()
    assert refusal_logger.level == prior_level


def test_every_default_exception_type_has_a_matching_default_exemplar():
    """Each declared default exception type has a matching exemplar."""
    for exception_type in refusal._DEFAULT_EXCEPTION_TYPES:
        assert any(isinstance(exemplar, exception_type) for exemplar in refusal._DEFAULT_EXEMPLARS)


def test_every_default_exemplar_matches_a_declared_default_exception_type():
    for exemplar in refusal._DEFAULT_EXEMPLARS:
        if isinstance(exemplar, refusal._DEFAULT_EXCEPTION_TYPES):
            continue
        # `RuntimeError` represents the catch-all branch.
        assert type(exemplar) is RuntimeError


def _isinstance_checked_types(func) -> set[type]:
    """Returns exception types checked by `isinstance` in `func`."""
    tree = ast.parse(inspect.getsource(func))
    checked: set[type] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "isinstance"
            and len(node.args) == 2
        ):
            expr = ast.fix_missing_locations(ast.Expression(node.args[1]))
            checked.add(eval(compile(expr, "<isinstance-arg>", "eval"), vars(refusal)))
    return checked


def test_default_exception_types_matches_classify_defaults_real_source():
    """The exception types in `classify_default` must match `_DEFAULT_EXCEPTION_TYPES`."""
    assert _isinstance_checked_types(refusal.classify_default) == set(
        refusal._DEFAULT_EXCEPTION_TYPES
    )


def test_digest_moves_when_classify_default_branch_order_changes(monkeypatch):
    """The digest changes when refusal dispatch order changes and stays fixed for the same order."""

    def _same_order(exc: BaseException) -> ProblemSpec:
        if isinstance(exc, RequestValidationError):
            return validation_problem(exc)
        if isinstance(exc, asyncio.TimeoutError):
            return refusal.render_timeout_refusal()
        if isinstance(exc, StarletteHTTPException):
            return refusal.render_http_exception_refusal(
                status=exc.status_code, detail=exc.detail, headers=exc.headers
            )
        return refusal.render_unhandled_refusal()

    def _reordered(exc: BaseException) -> ProblemSpec:
        if isinstance(exc, StarletteHTTPException):
            return refusal.render_http_exception_refusal(
                status=exc.status_code, detail=exc.detail, headers=exc.headers
            )
        if isinstance(exc, RequestValidationError):
            return validation_problem(exc)
        if isinstance(exc, asyncio.TimeoutError):
            return refusal.render_timeout_refusal()
        return refusal.render_unhandled_refusal()

    baseline = _compute_platform_capability_version()
    monkeypatch.setattr(refusal, "classify_default", _same_order)
    assert _compute_platform_capability_version() == baseline
    monkeypatch.setattr(refusal, "classify_default", _reordered)
    assert _compute_platform_capability_version() != baseline


def test_digest_moves_when_the_codes_success_body_shape_changes(monkeypatch):
    """The successful `/codes/{code}` body contributes to the platform digest."""
    baseline = _compute_platform_capability_version()

    class _PerturbedEntry:
        def __init__(self, *, code, registry_entry, catalogue_entry):
            self._code = code

        def model_dump(self):
            return {"code": self._code, "extra": "field"}

    monkeypatch.setattr(
        codes,
        "registry_entry_body",
        lambda **kwargs: _PerturbedEntry(**kwargs),
    )
    perturbed = _compute_platform_capability_version()
    assert perturbed != baseline


@pytest.mark.parametrize(
    "branch_function_name",
    [
        "_render_registered_code_exemplar",
        "_render_warning_code_exemplar",
        "_render_no_data_reason_refusal_code_exemplar",
        "_render_multi_channel_code_exemplar",
        "_render_unregistered_code_exemplar",
    ],
)
def test_digest_moves_when_one_codes_success_body_branch_changes(monkeypatch, branch_function_name):
    """Each successful-code response branch contributes to the platform digest."""
    baseline = _compute_platform_capability_version()
    monkeypatch.setattr(
        version,
        branch_function_name,
        lambda catalogue: {"perturbed_branch": branch_function_name},
    )
    perturbed = _compute_platform_capability_version()
    assert perturbed != baseline


def _leaf_values(value: object) -> set[object]:
    """Returns all non-container values in dataclasses and nested containers."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        leaves: set[object] = set()
        for f in dataclasses.fields(value):
            leaves |= _leaf_values(getattr(value, f.name))
        return leaves
    if isinstance(value, Mapping):
        leaves = set()
        for key, val in value.items():
            leaves |= _leaf_values(key)
            leaves |= _leaf_values(val)
        return leaves
    if isinstance(value, list | tuple | set | frozenset):
        leaves = set()
        for item in value:
            leaves |= _leaf_values(item)
        return leaves
    return {value}


def _is_represented(leaf: object, canonical_leaves: set[object]) -> bool:
    """Matches a leaf directly, or matches an integer by its string form."""
    if leaf in canonical_leaves:
        return True
    if isinstance(leaf, int) and not isinstance(leaf, bool):
        return str(leaf) in canonical_leaves
    return False


def test_every_public_refusal_constant_is_represented_in_canonical_data():
    """Every public refusal constant's values must appear in `as_canonical_data()`.

    Add missing constants to the canonical projection; skipping them weakens the digest contract.
    """
    canonical_leaves = _leaf_values(refusal.as_canonical_data())
    for name, value in vars(refusal).items():
        if name.startswith("_"):
            continue
        if inspect.ismodule(value) or inspect.isclass(value) or callable(value):
            continue
        if type(value).__module__ == "__future__":
            # The `annotations` feature is interpreter metadata.
            continue
        missing = {
            leaf for leaf in _leaf_values(value) if not _is_represented(leaf, canonical_leaves)
        }
        assert not missing, (
            f"dfo_platform.refusal.{name}'s own value(s) {missing!r} do not appear "
            "anywhere inside as_canonical_data()'s output, so "
            "PLATFORM_CAPABILITY_VERSION would not move if this constant changed. "
            "Add it into as_canonical_data() rather than skipping it here."
        )
