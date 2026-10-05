"""Runs `dfo_spec_kit.identity` against the language-neutral dfo-spec-kit identity vectors."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

import pytest
from dfo_spec_kit.identity import (
    IdentityError,
    build_projection,
    canonicalize,
    mint_spec_id,
    parse_identity_bytes,
    spec_id_from_value,
)


def _spec_kit_dir() -> Path | None:
    env_dir = os.environ.get("DFO_SPEC_KIT_DIR")
    if env_dir:
        return Path(env_dir)
    # Sibling checkout: <repo root>/../dfo-spec-kit, resolved from this test
    # file's own path so it works regardless of the invoking cwd.
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root.parent / "dfo-spec-kit"


def _vector_paths() -> list[Path]:
    spec_kit_dir = _spec_kit_dir()
    if spec_kit_dir is None or not spec_kit_dir.is_dir():
        return []
    return sorted((spec_kit_dir / "vectors" / "identity").glob("*.json"))


_VECTOR_PATHS = _vector_paths()

if not _VECTOR_PATHS:
    # CI sets DFO_SPEC_KIT_REQUIRED, where a missing suite means the checkout
    # step broke or the upstream layout moved. Skipping there would report
    # green with zero vectors exercised, so fail instead.
    if os.environ.get("DFO_SPEC_KIT_REQUIRED"):
        raise RuntimeError(
            "DFO_SPEC_KIT_REQUIRED is set but no identity vectors were found "
            f"under {_spec_kit_dir()} -- the vector suite checkout is missing "
            "or has moved"
        )
    pytest.skip(
        "dfo-spec-kit identity vector suite not found -- set DFO_SPEC_KIT_DIR "
        "or check out dfo-spec-kit as a sibling of dfo-apis",
        allow_module_level=True,
    )


def _load_vector(path: Path) -> dict:
    with path.open("rb") as f:
        return json.load(f)


_VECTORS = [_load_vector(path) for path in _VECTOR_PATHS]


def _build_projection_for(vector: dict) -> dict:
    """Run the vector's input through the pipeline stage its form implies:
    an input_bytes vector starts at parse_identity_bytes (the strict parse
    stage), an input vector is already-parsed JSON and starts at
    build_projection (the schema-validation stage onward)."""
    if "input_bytes" in vector:
        value = parse_identity_bytes(bytes.fromhex(vector["input_bytes"]))
    else:
        value = vector["input"]
    return build_projection(value)


@pytest.mark.parametrize("vector", _VECTORS, ids=[v["name"] for v in _VECTORS])
def test_identity_vector(vector: dict) -> None:
    if vector["classification"] == "negative":
        with pytest.raises(IdentityError) as excinfo:
            _build_projection_for(vector)
        assert excinfo.value.category == vector["failure_category"]
        return

    projection = _build_projection_for(vector)
    assert projection == vector["expected_projection"]

    canonical = canonicalize(projection)
    assert canonical == bytes.fromhex(vector["canonical_bytes"])

    spec_id = mint_spec_id(canonical)
    assert spec_id == vector["spec_id"]


# ---------------------------------------------------------------------------
# Non-vector test: a worked example from the identity spec, not in the
# vector set, so it has no pinned expected id -- only the shape of the
# output.
# ---------------------------------------------------------------------------


def test_worked_example_round_trips_to_a_valid_spec_id() -> None:
    identity_core = {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": "DevelopmentFinance",
            "measured_property": "DevelopmentFinanceFlow",
            "stat_type": "measuredValue",
        },
        "slots": [
            {
                "property": "DevelopmentFinancePurpose",
                "binding": {"state": "value", "ref": "Malaria"},
            },
            {
                "property": "DevelopmentFinanceScheme",
                "binding": {"state": "value", "ref": "GrantsScheme"},
            },
        ],
        "entities": {
            "provider": ["country/FRA"],
            "recipient": ["country/TGO"],
        },
        "facets": {
            "provenance": {"state": "value", "ref": "OECD_DAC2A"},
            "measurementMethod": {"state": "value", "ref": "GrantEquivalent"},
            "unit": {
                "state": "value",
                "request": {"base_year": 2023, "currency": "EUR", "prices": "constant"},
            },
        },
        "requested_window": {"kind": "range", "start_year": 2015, "end_year": 2024},
    }

    spec_id = spec_id_from_value(identity_core)

    assert spec_id.startswith("spec_")
    assert len(spec_id) == len("spec_") + 32
    hex_part = spec_id[len("spec_") :]
    assert len(hex_part) == 32
    int(hex_part, 16)  # must be valid lowercase hex
    assert hex_part == hex_part.lower()


# ---------------------------------------------------------------------------
# Non-vector tests: edge cases not covered by the vector suite.
# ---------------------------------------------------------------------------


def _minimal_document(**overrides: object) -> dict:
    doc: dict = {
        "identity_version": "spec1",
        "shape": {"kind": "statvar_schema", "population_type": "test_shape"},
        "slots": [],
        "entities": {},
        "requested_window": {"kind": "all"},
    }
    doc.update(overrides)
    return doc


class TestOutOfRangeYearBeyondSafeInteger:
    """Both entry paths refuse years above the I-JSON safe-integer limit, 2**53 - 1, as `number_out_of_range`."""

    def test_bytes_path_float_spelled_out_of_range_year(self) -> None:
        doc = _minimal_document(requested_window={"kind": "range", "start_year": 1e16})
        data = json.dumps(doc).encode("utf-8")
        with pytest.raises(IdentityError) as excinfo:
            spec_id_from_value(parse_identity_bytes(data))
        assert excinfo.value.category == "number_out_of_range"

    def test_value_path_plain_int_out_of_range_year(self) -> None:
        doc = _minimal_document(requested_window={"kind": "range", "start_year": 10**20})
        with pytest.raises(IdentityError) as excinfo:
            spec_id_from_value(doc)
        assert excinfo.value.category == "number_out_of_range"

    def test_value_path_float_spelled_out_of_range_year(self) -> None:
        doc = _minimal_document(requested_window={"kind": "range", "start_year": 1e16})
        with pytest.raises(IdentityError) as excinfo:
            spec_id_from_value(doc)
        assert excinfo.value.category == "number_out_of_range"

    def test_at_limit_year_mints_and_float_and_int_spellings_converge(self) -> None:
        limit = 2**53 - 1
        doc_int = _minimal_document(requested_window={"kind": "range", "start_year": limit})
        doc_float = _minimal_document(
            requested_window={"kind": "range", "start_year": float(limit)}
        )
        id_from_int = spec_id_from_value(doc_int)
        id_from_float = spec_id_from_value(doc_float)
        assert id_from_int == id_from_float

        # A float LITERAL (as opposed to a Python float object) spelling this
        # same boundary value on the bytes path is a different case under the
        # published max_number_length limit (16 characters): Python's default
        # JSON writer spells float(2**53 - 1) as "9007199254740991.0", 18
        # characters -- the 16 significant digits plus the ".0" float marker
        # -- which exceeds the limit. This particular value has no shorter
        # float spelling: it isn't a round decimal, so all 16 significant
        # digits are needed, and the ".0" marker always adds at least one
        # character past them. The number-length check is token-level and
        # runs before magnitude, so this refuses as input_limit_exceeded
        # rather than converging with id_from_int.
        data = json.dumps(doc_float).encode("utf-8")
        with pytest.raises(IdentityError) as excinfo:
            spec_id_from_value(parse_identity_bytes(data))
        assert excinfo.value.category == "input_limit_exceeded"


class TestNFCCollidingEntityRoleKeysRefuse:
    """NFC-colliding entity-role keys raise `duplicate_keys` instead of overwriting a role."""

    _DECOMPOSED_KEY = "róle"  # "role" with a combining acute accent (NFD-style)
    _PRECOMPOSED_KEY = "róle"  # the same word, NFC-composed

    def test_bytes_path(self) -> None:
        doc = _minimal_document(
            entities={self._DECOMPOSED_KEY: ["a"], self._PRECOMPOSED_KEY: ["b"]}
        )
        data = json.dumps(doc, ensure_ascii=False).encode("utf-8")
        with pytest.raises(IdentityError) as excinfo:
            spec_id_from_value(parse_identity_bytes(data))
        assert excinfo.value.category == "duplicate_keys"

    def test_value_path(self) -> None:
        doc = _minimal_document(
            entities={self._DECOMPOSED_KEY: ["a"], self._PRECOMPOSED_KEY: ["b"]}
        )
        with pytest.raises(IdentityError) as excinfo:
            build_projection(doc)
        assert excinfo.value.category == "duplicate_keys"


class TestSchemaGuardsSurviveOptimizedMode:
    """Schema checks must still raise `IdentityError` under `python -O`, which strips assertions."""

    _SCRIPT = (
        "from dfo_spec_kit.identity import IdentityError, build_projection\n"
        "doc = {\n"
        "    'identity_version': 'spec1',\n"
        "    'shape': 5,\n"
        "    'slots': [],\n"
        "    'entities': {},\n"
        "    'requested_window': {'kind': 'all'},\n"
        "}\n"
        "try:\n"
        "    build_projection(doc)\n"
        "except IdentityError as exc:\n"
        "    assert exc.category == 'schema_violation', exc.category\n"
        "    print('IdentityError-raised')\n"
        "else:\n"
        "    raise SystemExit('build_projection did not raise for a non-object shape')\n"
    )

    def test_malformed_shape_still_refuses_under_dash_o(self) -> None:
        result = subprocess.run(
            [sys.executable, "-O", "-c", self._SCRIPT],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "IdentityError-raised" in result.stdout


class TestLoneSurrogateInParsedValueRefuses:
    """Already-parsed values with lone surrogates raise `lone_surrogate` on both value entry points."""

    _LONE_SURROGATE = "\ud800"

    def test_ref_string(self) -> None:
        doc = _minimal_document(
            slots=[
                {
                    "property": "p",
                    "binding": {"state": "value", "ref": self._LONE_SURROGATE},
                }
            ]
        )
        with pytest.raises(IdentityError) as excinfo:
            build_projection(doc)
        assert excinfo.value.category == "lone_surrogate"

        with pytest.raises(IdentityError) as excinfo:
            spec_id_from_value(doc)
        assert excinfo.value.category == "lone_surrogate"

    def test_object_key(self) -> None:
        doc = _minimal_document(entities={self._LONE_SURROGATE: ["a"]})
        with pytest.raises(IdentityError) as excinfo:
            build_projection(doc)
        assert excinfo.value.category == "lone_surrogate"

        with pytest.raises(IdentityError) as excinfo:
            spec_id_from_value(doc)
        assert excinfo.value.category == "lone_surrogate"


class TestNestingDepthCheckedOnBothEntryPaths:
    """Both bytes and parsed-value entry paths reject over-limit nesting depth."""

    _OVER_LIMIT_SLOTS: ClassVar[list[dict]] = [
        {"property": "p", "binding": {"state": "set", "refs": [["nested"]]}}
    ]

    def test_bytes_path(self) -> None:
        doc = _minimal_document(slots=self._OVER_LIMIT_SLOTS)
        data = json.dumps(doc).encode("utf-8")
        with pytest.raises(IdentityError) as excinfo:
            parse_identity_bytes(data)
        assert excinfo.value.category == "input_limit_exceeded"

    def test_value_path(self) -> None:
        doc = _minimal_document(slots=self._OVER_LIMIT_SLOTS)
        with pytest.raises(IdentityError) as excinfo:
            build_projection(doc)
        assert excinfo.value.category == "input_limit_exceeded"


class TestStringLengthCheckedOnBothEntryPaths:
    """Both entry paths reject strings longer than the 1,024-character limit."""

    _OVER_LIMIT_POPULATION_TYPE = "x" * 1025

    def test_bytes_path(self) -> None:
        doc = _minimal_document(
            shape={"kind": "statvar_schema", "population_type": self._OVER_LIMIT_POPULATION_TYPE}
        )
        data = json.dumps(doc).encode("utf-8")
        with pytest.raises(IdentityError) as excinfo:
            parse_identity_bytes(data)
        assert excinfo.value.category == "input_limit_exceeded"

    def test_value_path(self) -> None:
        doc = _minimal_document(
            shape={"kind": "statvar_schema", "population_type": self._OVER_LIMIT_POPULATION_TYPE}
        )
        with pytest.raises(IdentityError) as excinfo:
            build_projection(doc)
        assert excinfo.value.category == "input_limit_exceeded"


class TestNestingDepthGuardIsRecursionSafe:
    """Iterative checks reject 32,000 byte-path levels under the 65,536-byte cap and
    100,000 parsed levels. JSON parsing itself recurses near 1,000 levels."""

    def test_deep_bytes_payload_refuses_via_the_prescan(self) -> None:
        depth = 32_000
        data = b"[" * depth + b"1" + b"]" * depth
        assert len(data) < 65_536
        with pytest.raises(IdentityError) as excinfo:
            parse_identity_bytes(data)
        assert excinfo.value.category == "input_limit_exceeded"

    def test_deep_parsed_value_refuses_via_the_value_level_check(self) -> None:
        # No byte-size cap applies on the value path, so this can go far
        # deeper than any bytes payload could -- 100,000 levels, built
        # iteratively (a Python loop, not recursion).
        value: object = 1
        for _ in range(100_000):
            value = [value]
        with pytest.raises(IdentityError) as excinfo:
            build_projection(value)
        assert excinfo.value.category == "input_limit_exceeded"

    def test_bracket_lookalike_in_a_string_does_not_trip_the_prescan(self) -> None:
        doc = _minimal_document(
            slots=[{"property": "p", "binding": {"state": "value", "ref": "[[[[[["}}]
        )
        data = json.dumps(doc).encode("utf-8")
        value = parse_identity_bytes(data)
        spec_id = spec_id_from_value(value)
        assert spec_id.startswith("spec_")

    def test_ordinary_depth_six_input_still_refuses_alongside_a_bracket_lookalike(
        self,
    ) -> None:
        # Regression guard for the prescan's string-awareness: a bracket
        # lookalike sitting next to genuine over-depth nesting must not make
        # the scan under-count the real structural depth.
        doc = _minimal_document(
            slots=[
                {
                    "property": "p",
                    "binding": {"state": "set", "refs": [["nested", "[[[["]]},
                }
            ]
        )
        data = json.dumps(doc).encode("utf-8")
        with pytest.raises(IdentityError) as excinfo:
            parse_identity_bytes(data)
        assert excinfo.value.category == "input_limit_exceeded"
