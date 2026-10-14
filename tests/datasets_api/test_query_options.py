"""Tests that dotted GET options parse or refuse like the POST body options, including
matching validation pointers.
"""

from __future__ import annotations

from urllib.parse import parse_qsl

import pytest
from pydantic import ValidationError

from datasets_api.mint_models import MintOptions
from datasets_api.query_options import (
    QueryOptionsInvalid,
    encode_query_options,
    parse_query_options,
)


def _items(query: str) -> list[tuple[str, str]]:
    # Preserve `format=` as an explicit empty value for the parser.
    return parse_qsl(query, keep_blank_values=True)


def _body_fault_pointer(body: dict[str, object]) -> str:
    """Returns the validation pointer raised by `MintOptions` for `body`."""
    try:
        MintOptions.model_validate(body)
    except ValidationError as exc:
        first = exc.errors()[0]
        return "/options/" + "/".join(str(segment) for segment in first["loc"])
    raise AssertionError(f"body {body!r} was expected to be refused by MintOptions")


# (query, body options, expected pointer; None means both parse identically).
GOLDEN_TABLE: list[tuple[str, dict[str, object], str | None]] = [
    ("", {}, None),
    ("format=parquet", {"format": "parquet"}, None),
    ("bundle=false", {"bundle": False}, None),
    ("format=xlsx&bundle=false", {"format": "xlsx", "bundle": False}, None),
    (
        "groupby.by=provider&groupby.op=sum",
        {"groupby": {"by": ["provider"], "op": "sum"}},
        None,
    ),
    (
        "groupby.by=provider,recipient&groupby.op=sum",
        {"groupby": {"by": ["provider", "recipient"], "op": "sum"}},
        None,
    ),
    (
        "pivot.index=year&pivot.columns=recipient",
        {"pivot": {"index": "year", "columns": "recipient"}},
        None,
    ),
    (
        "columns=year,recipient,value,unit",
        {"columns": ["year", "recipient", "value", "unit"]},
        None,
    ),
    (
        "format=parquet&bundle=false&groupby.by=provider&groupby.op=sum"
        "&pivot.index=year&pivot.columns=recipient&columns=year,recipient,value,unit",
        {
            "format": "parquet",
            "bundle": False,
            "groupby": {"by": ["provider"], "op": "sum"},
            "pivot": {"index": "year", "columns": "recipient"},
            "columns": ["year", "recipient", "value", "unit"],
        },
        None,
    ),
    ("format=json", {"format": "json"}, "/options/format"),
    ("bundle=maybe", {"bundle": "maybe"}, "/options/bundle"),
    ("format=", {"format": ""}, "/options/format"),
    ("bundle=", {"bundle": ""}, "/options/bundle"),
    (
        "groupby.by=&groupby.op=sum",
        {"groupby": {"by": [""], "op": "sum"}},
        "/options/groupby/by/0",
    ),
    (
        "groupby.by=time_period,crsPurpose&groupby.op=sum&groupby.levels=crsPurpose:0",
        {
            "groupby": {
                "by": ["time_period", "crsPurpose"],
                "op": "sum",
                "levels": {"crsPurpose": 0},
            }
        },
        None,
    ),
    (
        "groupby.by=time_period&groupby.op=sum&groupby.levels=crsPurpose:0",
        {"groupby": {"by": ["time_period"], "op": "sum", "levels": {"crsPurpose": 0}}},
        "/options/groupby/levels",
    ),
    (
        "groupby.by=crsPurpose&groupby.op=sum&groupby.levels=crsPurpose:-1",
        {"groupby": {"by": ["crsPurpose"], "op": "sum", "levels": {"crsPurpose": -1}}},
        "/options/groupby/levels/crsPurpose",
    ),
    (
        "groupby.by=crsPurpose&groupby.op=sum&groupby.levels=crsPurpose:100",
        {"groupby": {"by": ["crsPurpose"], "op": "sum", "levels": {"crsPurpose": 100}}},
        "/options/groupby/levels/crsPurpose",
    ),
    (
        "format=parquet&bundle=false&layout=frame",
        {"format": "parquet", "bundle": False, "layout": "frame"},
        None,
    ),
    (
        "format=csv&bundle=false&layout=frame",
        {"format": "csv", "bundle": False, "layout": "frame"},
        "/options/layout",
    ),
    ("format=parquet&layout=frame", {"format": "parquet", "layout": "frame"}, "/options/layout"),
    ("without_party_totals=true", {"without_party_totals": True}, None),
    (
        "groupby.by=recipient&groupby.op=sum&without_party_totals=true",
        {"groupby": {"by": ["recipient"], "op": "sum"}, "without_party_totals": True},
        None,
    ),
    (
        "without_party_totals=maybe",
        {"without_party_totals": "maybe"},
        "/options/without_party_totals",
    ),
    # Both validators report the same pointer for unknown keys.
    ("foo=bar", {"foo": "bar"}, "/options/foo"),
]


@pytest.mark.parametrize(
    "query, body, fault_pointer", GOLDEN_TABLE, ids=[row[0] or "empty" for row in GOLDEN_TABLE]
)
def test_query_and_body_agree(
    query: str, body: dict[str, object], fault_pointer: str | None
) -> None:
    items = _items(query)

    if fault_pointer is None:
        parsed = parse_query_options(items)
        assert parsed == MintOptions.model_validate(body)
        return

    with pytest.raises(QueryOptionsInvalid) as exc_info:
        parse_query_options(items)
    assert exc_info.value.pointer == fault_pointer
    assert _body_fault_pointer(body) == fault_pointer


class TestNamespaceIsClosed:
    @pytest.mark.parametrize(
        ("key", "expected_pointer"),
        [
            pytest.param("bogus", "/options/bogus", id="simple-key"),
            pytest.param("groupby.limit", "/options/groupby/limit", id="dotted-key"),
            pytest.param("foo/bar", "/options/foo~1bar", id="slash-rfc6901-escape"),
            pytest.param("foo~bar", "/options/foo~0bar", id="tilde-rfc6901-escape"),
            pytest.param("a.b", "/options/a/b", id="dotted-key-splits-before-escaping"),
        ],
    )
    def test_unknown_key_refuses_with_its_rfc6901_pointer(self, key: str, expected_pointer: str):
        with pytest.raises(QueryOptionsInvalid) as exc_info:
            parse_query_options([(key, "1")])

        assert exc_info.value.pointer == expected_pointer


class TestRepeatedKeyRefuses:
    """Repeated query keys have no JSON body equivalent."""

    @pytest.mark.parametrize(
        ("items", "expected_pointer"),
        [
            pytest.param(
                [("format", "csv"), ("format", "parquet")],
                "/options/format",
                id="top-level-key",
            ),
            pytest.param(
                [("groupby.by", "a"), ("groupby.by", "b")],
                "/options/groupby/by",
                id="dotted-key",
            ),
        ],
    )
    def test_repeated_key_refuses_at_its_own_pointer(
        self, items: list[tuple[str, str]], expected_pointer: str
    ):
        with pytest.raises(QueryOptionsInvalid) as exc_info:
            parse_query_options(items)

        assert exc_info.value.pointer == expected_pointer


class TestLevelsGrammar:
    @pytest.mark.parametrize(
        "value", ["crsPurpose", "crsPurpose:", "crsPurpose:one", "a:01", "a:1,a:2"]
    )
    def test_a_malformed_levels_entry_refuses_at_the_levels_pointer(self, value: str):
        with pytest.raises(QueryOptionsInvalid) as exc_info:
            parse_query_options(
                [("groupby.by", "a"), ("groupby.op", "sum"), ("groupby.levels", value)]
            )

        assert exc_info.value.pointer == "/options/groupby/levels"

    @pytest.mark.parametrize("sign", ["", "-"], ids=["positive", "negative"])
    def test_a_depth_past_the_integer_conversion_limit_refuses_like_the_body(self, sign: str):
        digits = "1" + "0" * 4300
        with pytest.raises(QueryOptionsInvalid) as exc_info:
            parse_query_options(
                [
                    ("groupby.by", "crsPurpose"),
                    ("groupby.op", "sum"),
                    ("groupby.levels", f"crsPurpose:{sign}{digits}"),
                ]
            )

        depth = int(f"{sign}1") * 10**4300
        body = {"groupby": {"by": ["crsPurpose"], "op": "sum", "levels": {"crsPurpose": depth}}}
        assert exc_info.value.pointer == "/options/groupby/levels/crsPurpose"
        assert _body_fault_pointer(body) == exc_info.value.pointer

    def test_a_boolean_depth_in_the_body_refuses(self):
        assert (
            _body_fault_pointer(
                {"groupby": {"by": ["crsPurpose"], "op": "sum", "levels": {"crsPurpose": True}}}
            )
            == "/options/groupby/levels/crsPurpose"
        )


class TestDefaults:
    def test_no_items_parses_to_mint_options_defaults(self):
        assert parse_query_options([]) == MintOptions()


_ROUND_TRIP_OPTIONS: list[MintOptions] = [
    MintOptions(),
    MintOptions(format="xlsx"),
    MintOptions(bundle=True),
    MintOptions(bundle=False),
    MintOptions(groupby={"by": ["stat_var"], "op": "sum"}),
    MintOptions(groupby={"by": ["stat_var", "time_period"], "op": "sum"}),
    MintOptions(pivot={"index": "year", "columns": "recipient"}),
    MintOptions(
        groupby={
            "by": ["time_period", "crsPurpose", "crsTypeOfFinance"],
            "op": "sum",
            "levels": {"crsTypeOfFinance": 1, "crsPurpose": 0},
        }
    ),
    MintOptions(columns=["year", "recipient", "value", "unit"]),
    MintOptions(
        format="parquet",
        bundle=False,
        groupby={"by": ["recipient"], "op": "sum"},
        pivot={"index": "year", "columns": "recipient"},
        columns=["year", "recipient", "value", "unit"],
    ),
    MintOptions(
        format="parquet",
        bundle=False,
        groupby={"by": ["time_period"], "op": "sum"},
        columns=["observationAbout_type"],
        layout="frame",
    ),
    MintOptions(
        format="parquet",
        bundle=False,
        groupby={"by": ["time_period"], "op": "sum"},
        without_party_totals=True,
        layout="frame",
    ),
]


@pytest.mark.parametrize("options", _ROUND_TRIP_OPTIONS, ids=[str(o) for o in _ROUND_TRIP_OPTIONS])
def test_encode_query_options_round_trips_through_parse_query_options(options: MintOptions):
    assert parse_query_options(list(encode_query_options(options).items())) == options
