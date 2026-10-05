"""Parse SDMX-JSON 2.0.0 availability fixtures from the mixer."""

from __future__ import annotations

import json
import re

import pytest

from dc_sdmx.errors import SdmxResponseError
from dc_sdmx.results import parse_availability_json

from .conftest import read_fixture_text


def _load(name: str) -> dict:
    return json.loads(read_fixture_text(name))


def test_single_component_golden():
    result = parse_availability_json(_load("availability_observation_about.json"))
    assert result.components == {"observationAbout": ("country/USA", "geoId/06")}
    assert result.raw["data"]["dataConstraints"][0]["id"] == "DF_OBS_AVAILABILITY"


def test_multiple_components_golden():
    result = parse_availability_json(_load("availability_multiple_components.json"))
    assert result.components == {
        "observationAbout": ("country/USA", "geoId/06"),
        "TIME_PERIOD": ("2020", "2021"),
    }


def test_empty_cube_regions_golden_yields_no_components():
    result = parse_availability_json(_load("availability_time_period_empty.json"))
    assert result.components == {}
    # raw is preserved verbatim even though this view has nothing to show.
    assert result.raw["data"]["dataConstraints"][0]["role"] == "Actual"


def test_duplicate_values_across_regions_are_deduplicated():
    body = {
        "data": {
            "dataConstraints": [
                {
                    "cubeRegions": [
                        {"keyValues": [{"id": "observationAbout", "values": ["country/USA"]}]},
                        {
                            "keyValues": [
                                {"id": "observationAbout", "values": ["country/USA", "geoId/06"]}
                            ]
                        },
                    ]
                }
            ]
        }
    }
    result = parse_availability_json(body)
    assert result.components == {"observationAbout": ("country/USA", "geoId/06")}


def test_member_objects_are_unwrapped_and_deduplicated_with_strings():
    body = {
        "data": {
            "dataConstraints": [
                {
                    "cubeRegions": [
                        {
                            "keyValues": [
                                {
                                    "id": "observationAbout",
                                    "values": [
                                        {"value": "country/USA"},
                                        "geoId/06",
                                        {"value": "country/USA"},
                                    ],
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    }

    result = parse_availability_json(body)

    assert result.components == {"observationAbout": ("country/USA", "geoId/06")}


@pytest.mark.parametrize("body", [["not", "an", "object"], "a bare string", 42, None])
def test_non_object_json_body_raises_response_error(body):
    with pytest.raises(SdmxResponseError, match="not a JSON object"):
        parse_availability_json(body)


@pytest.mark.parametrize("data", [["not", "an", "object"], "a bare string", 42])
def test_non_object_data_field_raises_response_error(data):
    # This malformed `data` field would otherwise raise an unwrapped AttributeError.
    with pytest.raises(SdmxResponseError, match="'data' is not a JSON object"):
        parse_availability_json({"data": data})


# The mixer requires `data` and `dataConstraints`; nested cube-region fields may be omitted.


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({}, id="missing-data"),
        pytest.param({"message": "upstream failure"}, id="error-envelope"),
    ],
)
def test_missing_data_key_raises_response_error(body):
    with pytest.raises(SdmxResponseError, match="missing required field 'data'"):
        parse_availability_json(body)


def test_missing_data_constraints_key_raises_response_error():
    with pytest.raises(
        SdmxResponseError, match=re.escape("missing required field 'data.dataConstraints'")
    ):
        parse_availability_json({"data": {}})


# The mixer emits `include=true`; the client rejects unsupported exclusions.


def test_region_level_include_false_raises_response_error():
    body = {
        "data": {
            "dataConstraints": [
                {
                    "cubeRegions": [
                        {
                            "include": False,
                            "keyValues": [{"id": "observationAbout", "values": ["country/USA"]}],
                        }
                    ]
                }
            ]
        }
    }
    with pytest.raises(SdmxResponseError, match="include=false") as excinfo:
        parse_availability_json(body)
    assert "observationAbout" in str(excinfo.value)


def test_key_value_level_include_false_raises_response_error():
    body = {
        "data": {
            "dataConstraints": [
                {
                    "cubeRegions": [
                        {
                            "keyValues": [
                                {
                                    "id": "observationAbout",
                                    "include": False,
                                    "values": ["country/USA"],
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    }
    with pytest.raises(SdmxResponseError, match=r"observationAbout.*include=false"):
        parse_availability_json(body)


@pytest.mark.parametrize(
    "region_include, key_value_include",
    [(True, True), (True, None), (None, True), (None, None)],
)
def test_include_true_or_missing_never_raises(region_include, key_value_include):
    # A missing "include" key defaults to true (the schema's own default),
    # matching an explicit include=true -- neither should raise.
    key_value = {"id": "observationAbout", "values": ["country/USA"]}
    if key_value_include is not None:
        key_value["include"] = key_value_include
    region = {"keyValues": [key_value]}
    if region_include is not None:
        region["include"] = region_include
    body = {"data": {"dataConstraints": [{"cubeRegions": [region]}]}}
    result = parse_availability_json(body)
    assert result.components == {"observationAbout": ("country/USA",)}


@pytest.mark.parametrize(
    "body",
    [
        {"data": {"dataConstraints": None}},
        {"data": {"dataConstraints": "not-a-list"}},
        {"data": {"dataConstraints": [None]}},
        {"data": {"dataConstraints": [42]}},
        {"data": {"dataConstraints": [{"cubeRegions": None}]}},
        {"data": {"dataConstraints": [{"cubeRegions": "not-a-list"}]}},
        {"data": {"dataConstraints": [{"cubeRegions": [None]}]}},
        {"data": {"dataConstraints": [{"cubeRegions": [{"keyValues": None}]}]}},
        {"data": {"dataConstraints": [{"cubeRegions": [{"keyValues": "not-a-list"}]}]}},
        {"data": {"dataConstraints": [{"cubeRegions": [{"keyValues": [None]}]}]}},
        {
            "data": {
                "dataConstraints": [
                    {"cubeRegions": [{"keyValues": [{"id": "observationAbout", "values": None}]}]}
                ]
            }
        },
        {
            "data": {
                "dataConstraints": [
                    {
                        "cubeRegions": [
                            {"keyValues": [{"id": "observationAbout", "values": "not-a-list"}]}
                        ]
                    }
                ]
            }
        },
    ],
)
def test_malformed_nested_shape_raises_response_error_not_a_builtin_exception(body):
    # These malformed levels would otherwise raise unwrapped TypeError or AttributeError.
    with pytest.raises(SdmxResponseError):
        parse_availability_json(body)


# Mixer `id` and `values` fields are strings; reject malformed scalar types.


def test_null_key_values_inside_excluded_region_raises_response_error():
    # Validate the shape before the include=false diagnostic iterates `keyValues`.
    body = {"data": {"dataConstraints": [{"cubeRegions": [{"include": False, "keyValues": None}]}]}}
    with pytest.raises(SdmxResponseError, match="'keyValues' is not a JSON array"):
        parse_availability_json(body)


def test_non_string_truthy_id_raises_response_error():
    body = {
        "data": {
            "dataConstraints": [
                {"cubeRegions": [{"keyValues": [{"id": ["x"], "values": ["country/USA"]}]}]}
            ]
        }
    }
    with pytest.raises(SdmxResponseError, match=r"keyValues\[\]\.id is not a string"):
        parse_availability_json(body)


@pytest.mark.parametrize(
    "values",
    [
        pytest.param([{}], id="unhashable-value"),
        pytest.param([{"value": 42}], id="member-object-with-non-string-value"),
        pytest.param(["ok", 42], id="hashable-non-string-value"),
    ],
)
def test_non_string_values_raise_response_error(values):
    body = {
        "data": {
            "dataConstraints": [
                {"cubeRegions": [{"keyValues": [{"id": "observationAbout", "values": values}]}]}
            ]
        }
    }
    with pytest.raises(SdmxResponseError, match="non-string value"):
        parse_availability_json(body)


def test_missing_or_empty_id_still_silently_skipped():
    # Missing or empty ids skip the entry.
    body = {
        "data": {
            "dataConstraints": [
                {
                    "cubeRegions": [
                        {
                            "keyValues": [
                                {"values": ["should-be-skipped"]},
                                {"id": "", "values": ["also-skipped"]},
                                {"id": "observationAbout", "values": ["country/USA"]},
                            ]
                        }
                    ]
                }
            ]
        }
    }
    result = parse_availability_json(body)
    assert result.components == {"observationAbout": ("country/USA",)}
