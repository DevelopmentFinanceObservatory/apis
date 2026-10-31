"""Pin component defaults, values, source metadata, and schema validation."""

from __future__ import annotations

import dataclasses
import datetime
from typing import get_args

import pytest

from assembler.bind import BoundSeries, bind_all
from assembler.errors import BindingInvalid, SchemaInvalid
from assembler.limits import LABEL_MAX
from assembler.mint import default_window, mint_component, stamp_envelope, warned_refs
from assembler.mint.provenance import (
    _REF_FIELDS,
    _WARNING_KIND_BY_CODE,
    DESCRIPTION_PROVENANCE_MEMBERS,
)
from assembler.ports import FrameRef
from assembler.resolve import ResolvedSeries, resolve_selectors
from assembler.select import normalize_select
from assembler.series_id import mint_series_id
from assembler.testing import (
    build_resolved_frame,
    malaria_ethiopia_frame_ref,
    malaria_ethiopia_resolved_frame,
    malaria_ethiopia_rows,
)
from datasets_api.manifest_models import CountedEntityYearsModel, CountedRefsModel, WarningModel
from datasets_api.warnings import DATASETS_WARNING_CATALOGUE
from presentation_api.refusals import binding_invalid_spec


def _fixed_clock():
    moment = datetime.datetime(2026, 7, 14, 10, 2, 0, tzinfo=datetime.UTC)
    return lambda: moment


@pytest.fixture
def resolved_frame():
    return malaria_ethiopia_resolved_frame()


def _resolve_and_bind(*, resolved_frame, selectors, window=None):
    series = resolve_selectors(resolved_frame=resolved_frame, selectors=selectors, window=window)
    effective = window or default_window(resolved_frame=resolved_frame)
    bound = bind_all(resolved_frame=resolved_frame, series=series, window=effective)
    return series, bound


def _malaria_selectors():
    return [
        {
            "bindings": [
                {"dimension": "stat_var", "values": ["dcid:MalariaGrantDisbursements"]},
                {"dimension": "recipient", "values": ["country/ETH"]},
            ]
        }
    ]


def _malaria_select():
    return {"stat_var": "dcid:MalariaGrantDisbursements", "recipient": "country/ETH"}


# -- line_chart --------------------------------------------------------------


def test_line_chart_mints_a_schema_valid_envelope(release, resolved_frame):
    series, bound = _resolve_and_bind(
        resolved_frame=resolved_frame,
        selectors=_malaria_selectors(),
        window={"start": "2015", "end": "2024"},
    )
    intent = {
        "type": "line_chart",
        "frame_ref": {
            "spec_id": resolved_frame.frame_ref.spec_id,
            "content_version": resolved_frame.frame_ref.content_version,
        },
        "select": _malaria_select(),
        "window": {"start": "2015", "end": "2024"},
        "title": "Malaria grant disbursements to Ethiopia",
    }
    minted = mint_component(
        member="line_chart",
        merged_intent=intent,
        resolved_frame=resolved_frame,
        series=series,
        bound_series=bound,
        release=release,
        clock=_fixed_clock(),
    )
    envelope = stamp_envelope(
        minted=minted, component_id="cmp_" + "a" * 20, revision=1, release=release
    )
    assert envelope["component"]["type"] == "line_chart"
    assert envelope["component"]["title"]["text"] == "Malaria grant disbursements to Ethiopia"


def test_line_chart_defaults_to_linear_scale_and_unbounded_range_when_the_intent_omits_them(
    release, resolved_frame
):
    """A line_chart mints `scale: "linear"` and an unbounded `range` when
    the intent carries neither field itself."""
    series, bound = _resolve_and_bind(resolved_frame=resolved_frame, selectors=_malaria_selectors())
    intent = {
        "type": "line_chart",
        "frame_ref": {
            "spec_id": resolved_frame.frame_ref.spec_id,
            "content_version": resolved_frame.frame_ref.content_version,
        },
        "select": _malaria_select(),
    }
    minted = mint_component(
        member="line_chart",
        merged_intent=intent,
        resolved_frame=resolved_frame,
        series=series,
        bound_series=bound,
        release=release,
        clock=_fixed_clock(),
    )
    assert minted.component["y_axis"]["scale"] == "linear"
    assert minted.component["y_axis"]["range"] == {"min": None, "max": None}


def test_line_chart_refuses_an_authored_range_with_min_above_max(release, resolved_frame):
    """An inverted `range` refuses at `/range` before any constraint runs,
    since `range_contains_visible_data`'s own detail names an excluded
    value, not the inverted bound itself."""
    series, bound = _resolve_and_bind(resolved_frame=resolved_frame, selectors=_malaria_selectors())
    intent = {
        "type": "line_chart",
        "frame_ref": {
            "spec_id": resolved_frame.frame_ref.spec_id,
            "content_version": resolved_frame.frame_ref.content_version,
        },
        "select": _malaria_select(),
        "range": {"min": 460, "max": 390},
    }
    with pytest.raises(BindingInvalid) as excinfo:
        mint_component(
            member="line_chart",
            merged_intent=intent,
            resolved_frame=resolved_frame,
            series=series,
            bound_series=bound,
            release=release,
            clock=_fixed_clock(),
        )
    assert excinfo.value.pointer == "/range"


def test_line_chart_default_title_when_intent_omits_it(release, resolved_frame):
    series, bound = _resolve_and_bind(resolved_frame=resolved_frame, selectors=_malaria_selectors())
    intent = {
        "type": "line_chart",
        "frame_ref": {
            "spec_id": resolved_frame.frame_ref.spec_id,
            "content_version": resolved_frame.frame_ref.content_version,
        },
        "select": _malaria_select(),
    }
    minted = mint_component(
        member="line_chart",
        merged_intent=intent,
        resolved_frame=resolved_frame,
        series=series,
        bound_series=bound,
        release=release,
        clock=_fixed_clock(),
    )
    assert minted.component["title"]["text"]
    assert minted.component["title"]["authorship"] == "model"


def test_y_axis_label_derived_from_stat_var_graph_label(release, resolved_frame):
    series, bound = _resolve_and_bind(resolved_frame=resolved_frame, selectors=_malaria_selectors())
    intent = {
        "type": "line_chart",
        "frame_ref": {
            "spec_id": resolved_frame.frame_ref.spec_id,
            "content_version": resolved_frame.frame_ref.content_version,
        },
        "select": _malaria_select(),
    }
    minted = mint_component(
        member="line_chart",
        merged_intent=intent,
        resolved_frame=resolved_frame,
        series=series,
        bound_series=bound,
        release=release,
        clock=_fixed_clock(),
    )
    # `malaria_ethiopia_resolved_frame` carries no label for the stat_var
    # ref itself, so the fallback (the raw ref) is what should surface.
    assert minted.component["y_axis"]["label"] == "dcid:MalariaGrantDisbursements"


def test_source_ref_reuses_a_raw_ref_already_shaped_as_a_source_id(release, resolved_frame):
    series, bound = _resolve_and_bind(resolved_frame=resolved_frame, selectors=_malaria_selectors())
    intent = {
        "type": "line_chart",
        "frame_ref": {
            "spec_id": resolved_frame.frame_ref.spec_id,
            "content_version": resolved_frame.frame_ref.content_version,
        },
        "select": _malaria_select(),
    }
    minted = mint_component(
        member="line_chart",
        merged_intent=intent,
        resolved_frame=resolved_frame,
        series=series,
        bound_series=bound,
        release=release,
        clock=_fixed_clock(),
    )
    assert minted.component["series"][0]["source_ref"] == "src_oecd_dac"
    assert minted.provenance["sources"][0]["id"] == "src_oecd_dac"


def test_source_id_derivation_is_deterministic_for_a_non_conforming_raw_ref(release):
    from assembler.ports import FrameRef
    from assembler.testing import build_resolved_frame

    rf = build_resolved_frame(
        frame_ref=FrameRef(spec_id="spec_" + "e" * 32, content_version="cv_" + "f" * 32),
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": "dcid:X",
                "recipient": "country/ETH",
                "time_period": "2020",
                "value": 1.0,
                "unit": None,
                "measurementMethod": None,
                "provenance": "https://example.org/not-a-source-id",
            }
        ],
    )
    selectors = [{"bindings": [{"dimension": "recipient", "values": ["country/ETH"]}]}]
    series, bound = _resolve_and_bind(resolved_frame=rf, selectors=selectors)
    intent = {
        "type": "line_chart",
        "frame_ref": {
            "spec_id": rf.frame_ref.spec_id,
            "content_version": rf.frame_ref.content_version,
        },
        "select": {"recipient": "country/ETH"},
    }
    first = mint_component(
        member="line_chart",
        merged_intent=intent,
        resolved_frame=rf,
        series=series,
        bound_series=bound,
        release=release,
        clock=_fixed_clock(),
    )
    second = mint_component(
        member="line_chart",
        merged_intent=intent,
        resolved_frame=rf,
        series=series,
        bound_series=bound,
        release=release,
        clock=_fixed_clock(),
    )
    source_id = first.component["series"][0]["source_ref"]
    assert source_id.startswith("src_")
    assert source_id != "https://example.org/not-a-source-id"
    assert source_id == second.component["series"][0]["source_ref"]


# -- bar_chart -----------------------------------------------------------


def _multi_recipient_rows(*, role="recipient"):
    """Build data where Kenya's last observation (2021) precedes the window end (2022)."""
    data = {
        "country/ETH": {"2020": 100.0, "2021": 110.0, "2022": 120.0},
        "country/GHA": {"2020": 200.0, "2021": 210.0, "2022": 220.0},
        "country/KEN": {"2020": 300.0, "2021": 310.0},
    }
    return tuple(
        {
            "stat_var": "dcid:MalariaGrantDisbursements",
            role: recipient,
            "time_period": period,
            "value": value,
            "unit": None,
            "measurementMethod": None,
            "provenance": "src_oecd_dac",
        }
        for recipient, values in data.items()
        for period, value in values.items()
    )


def _multi_recipient_resolved_frame(*, role="recipient"):
    return build_resolved_frame(
        frame_ref=malaria_ethiopia_frame_ref(),
        role_keys=(role,),
        rows=_multi_recipient_rows(role=role),
        labels={"country/ETH": "Ethiopia", "country/GHA": "Ghana", "country/KEN": "Kenya"},
    )


def _multi_recipient_selectors(
    *, recipients, stat_var="dcid:MalariaGrantDisbursements", role="recipient"
):
    return [
        {
            "bindings": [
                {"dimension": "stat_var", "values": [stat_var]},
                {"dimension": role, "values": list(recipients)},
            ]
        }
    ]


def test_bar_chart_entity_mode_mints_the_shared_latest_observed_period(release):
    resolved_frame = _multi_recipient_resolved_frame()
    recipients = ["country/ETH", "country/GHA", "country/KEN"]
    series, bound = _resolve_and_bind(
        resolved_frame=resolved_frame,
        selectors=_multi_recipient_selectors(recipients=recipients),
        window={"start": "2020", "end": "2022"},
    )
    intent = {
        "type": "bar_chart",
        "frame_ref": {
            "spec_id": resolved_frame.frame_ref.spec_id,
            "content_version": resolved_frame.frame_ref.content_version,
        },
        "select": {
            "stat_var": "dcid:MalariaGrantDisbursements",
            "recipient": recipients,
        },
        "window": {"start": "2020", "end": "2022"},
    }
    minted = mint_component(
        member="bar_chart",
        merged_intent=intent,
        resolved_frame=resolved_frame,
        series=series,
        bound_series=bound,
        release=release,
        clock=_fixed_clock(),
    )
    # Kenya's last observation is 2021, so all three series share that period.
    assert minted.component["category_axis"] == {
        "mode": "entity",
        "category_dimension": "recipient",
        "period": "2021",
    }
    assert minted.component["value_axis"]["scale"] == "linear"
    assert minted.component["value_axis"]["range"] == {"min": None, "max": None}
    assert len(minted.component["series"]) == 3
    for one in minted.component["series"]:
        assert [point["period"] for point in one["points"]] == ["2021"]
    assert minted.observed_periods == frozenset({"2021"})

    envelope = stamp_envelope(
        minted=minted, component_id="cmp_" + "c" * 20, revision=1, release=release
    )
    assert envelope["component"]["type"] == "bar_chart"


def test_bar_chart_entity_mode_refuses_when_entities_share_no_observed_period(release):
    rows = [
        {
            "stat_var": "dcid:X",
            "recipient": "country/ETH",
            "time_period": "2020",
            "value": 1.0,
            "unit": None,
            "measurementMethod": None,
            "provenance": "src_oecd_dac",
        },
        {
            "stat_var": "dcid:X",
            "recipient": "country/GHA",
            "time_period": "2021",
            "value": 2.0,
            "unit": None,
            "measurementMethod": None,
            "provenance": "src_oecd_dac",
        },
    ]
    resolved_frame = build_resolved_frame(
        frame_ref=malaria_ethiopia_frame_ref(), role_keys=("recipient",), rows=rows
    )
    selectors = _multi_recipient_selectors(
        recipients=["country/ETH", "country/GHA"], stat_var="dcid:X"
    )
    series, bound = _resolve_and_bind(
        resolved_frame=resolved_frame, selectors=selectors, window={"start": "2020", "end": "2021"}
    )
    intent = {
        "type": "bar_chart",
        "frame_ref": {
            "spec_id": resolved_frame.frame_ref.spec_id,
            "content_version": resolved_frame.frame_ref.content_version,
        },
        "select": {"stat_var": "dcid:X", "recipient": ["country/ETH", "country/GHA"]},
        "window": {"start": "2020", "end": "2021"},
    }
    with pytest.raises(BindingInvalid) as excinfo:
        mint_component(
            member="bar_chart",
            merged_intent=intent,
            resolved_frame=resolved_frame,
            series=series,
            bound_series=bound,
            release=release,
            clock=_fixed_clock(),
        )
    assert excinfo.value.pointer == "/select"
    assert "share no observed period" in excinfo.value.detail


def test_bar_chart_period_mode_mints_one_series_across_the_window(release, resolved_frame):
    series, bound = _resolve_and_bind(
        resolved_frame=resolved_frame,
        selectors=_malaria_selectors(),
        window={"start": "2015", "end": "2024"},
    )
    intent = {
        "type": "bar_chart",
        "frame_ref": {
            "spec_id": resolved_frame.frame_ref.spec_id,
            "content_version": resolved_frame.frame_ref.content_version,
        },
        "select": _malaria_select(),
        "window": {"start": "2015", "end": "2024"},
    }
    minted = mint_component(
        member="bar_chart",
        merged_intent=intent,
        resolved_frame=resolved_frame,
        series=series,
        bound_series=bound,
        release=release,
        clock=_fixed_clock(),
    )
    assert minted.component["category_axis"] == {
        "mode": "period",
        "window": {"start": "2015", "end": "2024"},
    }
    assert minted.component["value_axis"]["scale"] == "linear"
    assert minted.component["value_axis"]["range"] == {"min": None, "max": None}
    assert len(minted.component["series"]) == 1
    assert len(minted.component["series"][0]["points"]) == 10

    envelope = stamp_envelope(
        minted=minted, component_id="cmp_" + "d" * 20, revision=1, release=release
    )
    assert envelope["component"]["type"] == "bar_chart"


_PROVIDERS = {f"org/P{index}": f"Provider {index}" for index in range(10)}


def _provider_resolved_frame(*, extra_roles=(), periods=("2022",), gaps=None):
    """One stat_var and ten providers, each observed in every one of `periods`.
    `extra_roles` adds a second varying dimension as `(role, values)` cycled across the
    providers. A provider's value rises with its index, so the last is the largest.

    `gaps` maps a provider to the status its row in the last period carries, or to `None`
    to drop that row."""
    gaps = gaps or {}
    rows = []
    for index, provider in enumerate(_PROVIDERS):
        for period in periods:
            if period == periods[-1] and provider in gaps and gaps[provider] is None:
                continue
            row = {
                "stat_var": "dcid:MalariaGrantDisbursements",
                "financingProvider": provider,
                "time_period": period,
                "value": 100.0 + index,
                "unit": None,
                "measurementMethod": None,
                "provenance": "src_oecd_dac",
            }
            for role, values in extra_roles:
                row[role] = values[index % len(values)]
            rows.append(row)
    resolved_frame = build_resolved_frame(
        frame_ref=malaria_ethiopia_frame_ref(),
        role_keys=("financingProvider", *(role for role, _ in extra_roles)),
        rows=rows,
        labels=_PROVIDERS,
    )
    statuses = {provider: status for provider, status in gaps.items() if status is not None}
    if not statuses:
        return resolved_frame
    # `Frame.build` writes a null status, so a flagged row is set on the built document.
    columns = resolved_frame.frame["columns"]
    provider_at = columns.index("financingProvider")
    period_at = columns.index("time_period")
    status_at = columns.index("status")
    value_at = columns.index("value")
    frame_rows = [list(row) for row in resolved_frame.frame["rows"]]
    for row in frame_rows:
        if row[period_at] == periods[-1] and row[provider_at] in statuses:
            row[status_at] = statuses[row[provider_at]]
            row[value_at] = None
    return dataclasses.replace(resolved_frame, frame={**resolved_frame.frame, "rows": frame_rows})


def _mint_bar_chart(*, release, resolved_frame, select, window=None):
    selectors = normalize_select(select=select)
    series, bound = _resolve_and_bind(
        resolved_frame=resolved_frame, selectors=selectors, window=window
    )
    intent = {
        "type": "bar_chart",
        "frame_ref": {
            "spec_id": resolved_frame.frame_ref.spec_id,
            "content_version": resolved_frame.frame_ref.content_version,
        },
        "select": select,
    }
    if window is not None:
        intent["window"] = window
    return mint_component(
        member="bar_chart",
        merged_intent=intent,
        resolved_frame=resolved_frame,
        series=series,
        bound_series=bound,
        release=release,
        clock=_fixed_clock(),
    )


def test_bar_chart_draws_every_value_of_the_one_unbound_dimension_largest_first(release):
    """Ten providers in one period and a select that binds only the stat_var leave the
    provider as the one dimension to compare, so the chart draws all ten, largest first."""
    resolved_frame = _provider_resolved_frame()
    minted = _mint_bar_chart(
        release=release,
        resolved_frame=resolved_frame,
        select={"stat_var": "dcid:MalariaGrantDisbursements"},
    )
    assert minted.component["category_axis"] == {
        "mode": "entity",
        "category_dimension": "financingProvider",
        "period": "2022",
    }
    assert [one["label"] for one in minted.component["series"]] == list(
        reversed(_PROVIDERS.values())
    )


def test_bar_chart_listing_values_keeps_the_list_order(release):
    resolved_frame = _provider_resolved_frame()
    providers = list(_PROVIDERS)
    minted = _mint_bar_chart(
        release=release,
        resolved_frame=resolved_frame,
        select={"stat_var": "dcid:MalariaGrantDisbursements", "financingProvider": providers},
    )
    assert [one["label"] for one in minted.component["series"]] == list(_PROVIDERS.values())


def _kenya_aid_resolved_frame(*, recipients=("country/KEN",), donor=None):
    """Health and education aid to each of `recipients` in 2021 and 2022, one series per
    stat_var and recipient. A `donor` adds a leading `donor` role every row shares."""
    return build_resolved_frame(
        frame_ref=malaria_ethiopia_frame_ref(),
        role_keys=("recipient",) if donor is None else ("donor", "recipient"),
        rows=[
            {
                **({} if donor is None else {"donor": donor}),
                "stat_var": stat_var,
                "recipient": recipient,
                "time_period": period,
                "value": value,
                "unit": None,
                "measurementMethod": None,
                "provenance": "src_oecd_dac",
            }
            for stat_var, base in (("dcid:HealthAid", 20.0), ("dcid:EducationAid", 10.0))
            for recipient in recipients
            for period, value in (("2021", base - 1), ("2022", base))
        ],
        labels={
            "country/KEN": "Kenya",
            "country/UGA": "Uganda",
            "dcid:HealthAid": "Health aid",
            "dcid:EducationAid": "Education aid",
        },
    )


@pytest.mark.parametrize(
    ("select", "window"),
    [
        ({"recipient": "country/KEN"}, {"start": "2022", "end": "2022"}),
        (
            {"stat_var": ["dcid:HealthAid", "dcid:EducationAid"], "recipient": "country/KEN"},
            None,
        ),
    ],
    ids=["default", "listed"],
)
def test_bars_compared_on_stat_var_carry_the_stat_var_labels(release, select, window):
    """Bars that differ only on stat_var are told apart by its labels, and the value axis
    names no single stat_var."""
    minted = _mint_bar_chart(
        release=release, resolved_frame=_kenya_aid_resolved_frame(), select=select, window=window
    )
    component = minted.component
    assert component["category_axis"]["category_dimension"] == "stat_var"
    assert [one["label"] for one in component["series"]] == ["Health aid", "Education aid"]
    assert component["value_axis"]["label"] == "Value"
    assert set(component["title"]["text"].split(", ")) == {"Health aid", "Education aid"}


def test_bar_chart_listing_values_refuses_when_two_bars_would_share_a_value(release):
    """Listing the stat_vars while recipient stays unbound would draw two bars per stat_var
    under one label, so the chart refuses naming the recipients to bind."""
    with pytest.raises(BindingInvalid) as excinfo:
        _mint_bar_chart(
            release=release,
            resolved_frame=_kenya_aid_resolved_frame(recipients=("country/KEN", "country/UGA")),
            select={"stat_var": ["dcid:HealthAid", "dcid:EducationAid"]},
        )

    spec = binding_invalid_spec(excinfo.value)
    assert spec.pointer == "/select"
    assert excinfo.value.dimension == "recipient"
    assert [one.ref for one in spec.alternatives] == ["country/KEN", "country/UGA"]


@pytest.mark.parametrize(
    ("stat_var_labels", "expected"),
    [
        pytest.param({}, ["Health aid / Kenya", "Education aid / Kenya"], id="short-labels"),
        pytest.param(
            {"dcid:HealthAid": "H" * 200, "dcid:EducationAid": "E" * 200},
            ["H" * 192 + " / Kenya", "E" * 192 + " / Kenya"],
            id="a-200-character-stat-var-label-keeps-the-series-label",
        ),
    ],
)
def test_lines_that_differ_only_on_stat_var_lead_with_its_label(release, stat_var_labels, expected):
    resolved_frame = _kenya_aid_resolved_frame()
    resolved_frame = dataclasses.replace(
        resolved_frame, labels={**resolved_frame.labels, **stat_var_labels}
    )
    select = {"stat_var": ["dcid:HealthAid", "dcid:EducationAid"], "recipient": "country/KEN"}
    series, bound = _resolve_and_bind(
        resolved_frame=resolved_frame, selectors=normalize_select(select=select)
    )
    minted = mint_component(
        member="line_chart",
        merged_intent={
            "type": "line_chart",
            "frame_ref": {
                "spec_id": resolved_frame.frame_ref.spec_id,
                "content_version": resolved_frame.frame_ref.content_version,
            },
            "select": select,
        },
        resolved_frame=resolved_frame,
        series=series,
        bound_series=bound,
        release=release,
        clock=_fixed_clock(),
    )
    component = minted.component
    assert [one["label"] for one in component["series"]] == expected
    assert component["y_axis"]["label"] == "Value"


def test_line_labels_over_a_stat_var_and_two_roles_keep_every_part(release):
    """Two StatVars, one donor and two recipients, with three 99-character labels, draw four
    lines that read apart."""
    resolved_frame = _kenya_aid_resolved_frame(
        recipients=("country/KEN", "country/UGA"), donor="country/USA"
    )
    resolved_frame = dataclasses.replace(
        resolved_frame,
        labels={
            **resolved_frame.labels,
            "dcid:HealthAid": "H" * 99,
            "dcid:EducationAid": "E" * 99,
            "country/USA": "D" * 99,
        },
    )
    select = {"stat_var": ["dcid:HealthAid", "dcid:EducationAid"], "donor": "country/USA"}
    series, bound = _resolve_and_bind(
        resolved_frame=resolved_frame, selectors=normalize_select(select=select)
    )
    minted = mint_component(
        member="line_chart",
        merged_intent={
            "type": "line_chart",
            "frame_ref": {
                "spec_id": resolved_frame.frame_ref.spec_id,
                "content_version": resolved_frame.frame_ref.content_version,
            },
            "select": select,
        },
        resolved_frame=resolved_frame,
        series=series,
        bound_series=bound,
        release=release,
        clock=_fixed_clock(),
    )
    labels = [one["label"] for one in minted.component["series"]]
    assert len(set(labels)) == 4
    assert all(len(label) <= LABEL_MAX for label in labels)
    assert sorted(label.rsplit(" / ", 1)[1] for label in labels) == [
        "Kenya",
        "Kenya",
        "Uganda",
        "Uganda",
    ]


def test_bar_chart_over_several_periods_refuses_naming_both_next_calls(release):
    resolved_frame = _provider_resolved_frame(periods=("2021", "2022"))
    with pytest.raises(BindingInvalid) as excinfo:
        _mint_bar_chart(
            release=release,
            resolved_frame=resolved_frame,
            select={"stat_var": "dcid:MalariaGrantDisbursements"},
        )

    spec = binding_invalid_spec(excinfo.value)
    assert spec.pointer == "/select"
    assert [(one.ref, one.label) for one in spec.alternatives] == list(_PROVIDERS.items())
    assert "2 periods (2021..2022)" in spec.detail
    assert "Bind 'financingProvider' to one value" in spec.detail

    retry = {
        "stat_var": "dcid:MalariaGrantDisbursements",
        "financingProvider": [one.ref for one in spec.alternatives],
    }
    minted = _mint_bar_chart(release=release, resolved_frame=resolved_frame, select=retry)
    assert minted.component["category_axis"]["period"] == "2022"


@pytest.mark.parametrize(
    ("periods", "gaps", "window"),
    [
        (("2022",), {"org/P3": "suppressed", "org/P7": "suppressed"}, None),
        (("2021", "2022"), {"org/P3": None, "org/P7": None}, {"start": "2022", "end": "2022"}),
    ],
    ids=["status_flagged", "no_row_in_window"],
)
def test_bar_chart_refuses_the_default_when_a_value_lacks_the_one_period(
    release, periods, gaps, window
):
    """Two providers hold no value in the one period with data, so the default refuses naming
    them, offers the eight observed there, and listing those draws them."""
    resolved_frame = _provider_resolved_frame(periods=periods, gaps=gaps)
    with pytest.raises(BindingInvalid) as excinfo:
        _mint_bar_chart(
            release=release,
            resolved_frame=resolved_frame,
            select={"stat_var": "dcid:MalariaGrantDisbursements"},
            window=window,
        )

    spec = binding_invalid_spec(excinfo.value)
    observed = [provider for provider in _PROVIDERS if provider not in gaps]
    assert spec.pointer == "/select"
    assert excinfo.value.dimension == "financingProvider"
    assert [one.ref for one in spec.alternatives] == observed
    assert "Provider 3 (org/P3), Provider 7 (org/P7)" in spec.detail

    retry = {"stat_var": "dcid:MalariaGrantDisbursements", "financingProvider": observed}
    minted = _mint_bar_chart(
        release=release, resolved_frame=resolved_frame, select=retry, window=window
    )
    assert minted.component["category_axis"]["period"] == "2022"
    assert len(minted.component["series"]) == 8


def test_bar_chart_with_two_unbound_dimensions_refuses_listing_the_one_to_bind(release):
    resolved_frame = _provider_resolved_frame(
        extra_roles=(("recipient", ["country/ETH", "country/GHA"]),)
    )
    with pytest.raises(BindingInvalid) as excinfo:
        _mint_bar_chart(
            release=release,
            resolved_frame=resolved_frame,
            select={"stat_var": "dcid:MalariaGrantDisbursements"},
        )

    refusal = excinfo.value
    assert refusal.dimension == "financingProvider"
    assert set(refusal.legal_values) == set(_PROVIDERS)
    assert "'financingProvider' (10 values), 'recipient' (2 values)" in refusal.detail
    assert "bind each of 'recipient' to one value" in refusal.detail
    assert "'recipient' holds country/ETH, country/GHA" in refusal.detail


# -- table -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("role", "header"),
    [
        ("recipient", "Recipient"),
        ("financingRecipient", "Recipient"),
        # A creditor shares the provider's flow axis and keeps its own noun.
        ("creditor", "Creditor"),
    ],
    ids=["recipient", "crs", "ids-creditor"],
)
def test_table_builds_one_row_per_coordinate_by_period(release, role, header):
    resolved_frame = _multi_recipient_resolved_frame(role=role)
    recipients = ["country/ETH", "country/GHA"]
    series, bound = _resolve_and_bind(
        resolved_frame=resolved_frame,
        selectors=_multi_recipient_selectors(recipients=recipients, role=role),
        window={"start": "2020", "end": "2021"},
    )
    intent = {
        "type": "table",
        "frame_ref": {
            "spec_id": resolved_frame.frame_ref.spec_id,
            "content_version": resolved_frame.frame_ref.content_version,
        },
        "select": {"stat_var": "dcid:MalariaGrantDisbursements", role: recipients},
        "window": {"start": "2020", "end": "2021"},
    }
    minted = mint_component(
        member="table",
        merged_intent=intent,
        resolved_frame=resolved_frame,
        series=series,
        bound_series=bound,
        release=release,
        clock=_fixed_clock(),
    )
    component = minted.component
    assert len(component["rows"]) == 4
    assert "total_rows" not in component
    assert [(col["key"], col["label"]) for col in component["columns"]] == [
        ("stat_var", "Variable"),
        (role, header),
        ("period", "Period"),
        ("value", "Value"),
    ]
    row = next(
        r
        for r in component["rows"]
        if r["dimensions"][role] == "Ethiopia" and r["period"] == "2020"
    )
    assert row["value"] == pytest.approx(100.0)
    assert row["source_ref"] == "src_oecd_dac"
    ethiopia_id = next(b.id for b in bound if b.label == "Ethiopia")
    assert row["series_id"] == ethiopia_id
    assert {r["series_id"] for r in component["rows"]} == {b.id for b in bound}

    envelope = stamp_envelope(
        minted=minted, component_id="cmp_" + "e" * 20, revision=1, release=release
    )
    assert envelope["component"]["type"] == "table"


def test_table_select_naming_one_stat_var_over_multiple_entities_distinguishes_rows(release):
    """An unbound entity role expands to separate series, which table rows must preserve.

    `recipient` is an entity role, so its values expand into series without a binding.
    """
    resolved_frame = _multi_recipient_resolved_frame()
    selectors = [
        {"bindings": [{"dimension": "stat_var", "values": ["dcid:MalariaGrantDisbursements"]}]}
    ]
    series, bound = _resolve_and_bind(
        resolved_frame=resolved_frame, selectors=selectors, window={"start": "2020", "end": "2020"}
    )
    intent = {
        "type": "table",
        "frame_ref": {
            "spec_id": resolved_frame.frame_ref.spec_id,
            "content_version": resolved_frame.frame_ref.content_version,
        },
        "select": {"stat_var": "dcid:MalariaGrantDisbursements"},
        "window": {"start": "2020", "end": "2020"},
    }
    minted = mint_component(
        member="table",
        merged_intent=intent,
        resolved_frame=resolved_frame,
        series=series,
        bound_series=bound,
        release=release,
        clock=_fixed_clock(),
    )
    component = minted.component
    assert [col["key"] for col in component["columns"]] == [
        "stat_var",
        "recipient",
        "period",
        "value",
    ]
    recipients = {row["dimensions"]["recipient"] for row in component["rows"]}
    assert recipients == {"Ethiopia", "Ghana", "Kenya"}
    eth_row = next(r for r in component["rows"] if r["dimensions"]["recipient"] == "Ethiopia")
    gha_row = next(r for r in component["rows"] if r["dimensions"]["recipient"] == "Ghana")
    assert eth_row["value"] == pytest.approx(100.0)
    assert gha_row["value"] == pytest.approx(200.0)
    assert eth_row["series_id"] == next(b.id for b in bound if b.label == "Ethiopia")
    assert gha_row["series_id"] == next(b.id for b in bound if b.label == "Ghana")
    assert {r["series_id"] for r in component["rows"]} == {b.id for b in bound}

    envelope = stamp_envelope(
        minted=minted, component_id="cmp_" + "g" * 20, revision=1, release=release
    )
    assert envelope["component"]["type"] == "table"


def test_table_row_unit_reflects_its_own_bound_series(release):
    """Each table row carries its bound series' unit.

    The series are built directly to give each a unit the frame does not carry.
    """
    resolved_frame = _multi_recipient_resolved_frame()
    series = (
        ResolvedSeries(id="eth", coordinates=(("recipient", "country/ETH"),), rows=()),
        ResolvedSeries(id="gha", coordinates=(("recipient", "country/GHA"),), rows=()),
    )
    bound = (
        BoundSeries(
            id="eth",
            label="Ethiopia",
            unit="unit/USD",
            source_ref="src_oecd_dac",
            points=({"period": "2020", "value": 100.0},),
        ),
        BoundSeries(
            id="gha",
            label="Ghana",
            unit="unit/EUR",
            source_ref="src_oecd_dac",
            points=({"period": "2020", "value": 200.0},),
        ),
    )
    intent = {
        "type": "table",
        "frame_ref": {
            "spec_id": resolved_frame.frame_ref.spec_id,
            "content_version": resolved_frame.frame_ref.content_version,
        },
        "select": {"recipient": ["country/ETH", "country/GHA"]},
        "window": {"start": "2020", "end": "2020"},
    }
    minted = mint_component(
        member="table",
        merged_intent=intent,
        resolved_frame=resolved_frame,
        series=series,
        bound_series=bound,
        release=release,
        clock=_fixed_clock(),
    )
    rows = minted.component["rows"]
    eth_row = next(r for r in rows if r["dimensions"]["recipient"] == "Ethiopia")
    gha_row = next(r for r in rows if r["dimensions"]["recipient"] == "Ghana")
    assert eth_row["unit"] == "unit/USD"
    assert gha_row["unit"] == "unit/EUR"
    assert "unit" not in minted.component
    assert eth_row["series_id"] == "eth"
    assert gha_row["series_id"] == "gha"

    envelope = stamp_envelope(
        minted=minted, component_id="cmp_" + "f" * 20, revision=1, release=release
    )
    assert envelope["component"]["type"] == "table"


def test_table_preserves_status_for_a_missing_period(release):
    resolved_frame = _multi_recipient_resolved_frame()
    series, bound = _resolve_and_bind(
        resolved_frame=resolved_frame,
        selectors=_multi_recipient_selectors(recipients=["country/KEN"]),
        window={"start": "2021", "end": "2022"},
    )
    intent = {
        "type": "table",
        "frame_ref": {
            "spec_id": resolved_frame.frame_ref.spec_id,
            "content_version": resolved_frame.frame_ref.content_version,
        },
        "select": {"stat_var": "dcid:MalariaGrantDisbursements", "recipient": "country/KEN"},
        "window": {"start": "2021", "end": "2022"},
    }
    minted = mint_component(
        member="table",
        merged_intent=intent,
        resolved_frame=resolved_frame,
        series=series,
        bound_series=bound,
        release=release,
        clock=_fixed_clock(),
    )
    rows = minted.component["rows"]
    assert len(rows) == 2
    missing_row = next(r for r in rows if r["period"] == "2022")
    assert missing_row["status"] == "missing"
    assert "value" not in missing_row
    observed_row = next(r for r in rows if r["period"] == "2021")
    assert observed_row["value"] == pytest.approx(310.0)
    assert "2022" not in minted.observed_periods
    assert "2021" in minted.observed_periods
    assert missing_row["series_id"] == bound[0].id
    assert observed_row["series_id"] == bound[0].id


_STAT_VAR_UNIT_SERIES = (
    ("dcid:GDP", "USDollar", {"2020": 77.6, "2021": 86.0}),
    ("dcid:Population", None, {"2020": 16.9, "2021": 17.1}),
)


def _mixed_unit_resolved_frame(*, unit_series=_STAT_VAR_UNIT_SERIES, facet="unit"):
    """One recipient with one series per `(stat_var, facet value, values)` entry,
    the value set on `facet` (`unit` or `measurementMethod`) and the other facet
    null. By default `GDP` reports in `USDollar` and `Population` carries a null
    `unit`."""
    rows = tuple(
        {
            "stat_var": stat_var,
            "recipient": "country/GTM",
            "time_period": period,
            "value": value,
            "unit": None,
            "measurementMethod": None,
            "provenance": "src_oecd_dac",
            facet: facet_value,
        }
        for stat_var, facet_value, values in unit_series
        for period, value in values.items()
    )
    return build_resolved_frame(
        frame_ref=malaria_ethiopia_frame_ref(),
        role_keys=("recipient",),
        rows=rows,
        labels={
            "country/GTM": "Guatemala",
            "dcid:GDP": "GDP",
            "dcid:Population": "Population",
            "USDollar": "US Dollar",
            "Survey": "Survey",
        },
    )


def _mint_gtm_table(*, release, resolved_frame, stat_vars, selectors=None):
    selectors = selectors or [
        {
            "bindings": [
                {"dimension": "stat_var", "values": stat_vars},
                {"dimension": "recipient", "values": ["country/GTM"]},
            ]
        }
    ]
    series, bound = _resolve_and_bind(
        resolved_frame=resolved_frame,
        selectors=selectors,
        window={"start": "2020", "end": "2021"},
    )
    intent = {
        "type": "table",
        "frame_ref": {
            "spec_id": resolved_frame.frame_ref.spec_id,
            "content_version": resolved_frame.frame_ref.content_version,
        },
        "select": {"stat_var": stat_vars, "recipient": "country/GTM"},
        "window": {"start": "2020", "end": "2021"},
    }
    minted = mint_component(
        member="table",
        merged_intent=intent,
        resolved_frame=resolved_frame,
        series=series,
        bound_series=bound,
        release=release,
        clock=_fixed_clock(),
    )
    return minted, bound


def _dimension_maps_by_series(component):
    maps: dict[str, set[tuple[tuple[str, str], ...]]] = {}
    for row in component["rows"]:
        maps.setdefault(row["series_id"], set()).add(tuple(sorted(row["dimensions"].items())))
    return maps


def test_table_labels_a_dimension_one_series_leaves_null(release):
    """Issue #992: `unit` is set on one series and null on the other, so it
    varies across the table's series and becomes a column. The null series'
    rows carry a defined label for it, and the two series keep distinct
    `dimensions` maps."""
    minted, _ = _mint_gtm_table(
        release=release,
        resolved_frame=_mixed_unit_resolved_frame(),
        stat_vars=["dcid:GDP", "dcid:Population"],
    )
    component = minted.component
    assert [col["key"] for col in component["columns"]] == [
        "stat_var",
        "recipient",
        "unit",
        "period",
        "value",
    ]
    rows = component["rows"]
    assert len(rows) == 4
    gdp_dimensions = {
        tuple(sorted(r["dimensions"].items())) for r in rows if r["dimensions"]["stat_var"] == "GDP"
    }
    population_dimensions = {
        tuple(sorted(r["dimensions"].items()))
        for r in rows
        if r["dimensions"]["stat_var"] == "Population"
    }
    assert gdp_dimensions == {
        (("recipient", "Guatemala"), ("stat_var", "GDP"), ("unit", "US Dollar"))
    }
    assert population_dimensions == {
        (("recipient", "Guatemala"), ("stat_var", "Population"), ("unit", "Not specified"))
    }

    envelope = stamp_envelope(
        minted=minted, component_id="cmp_" + "h" * 20, revision=1, release=release
    )
    assert envelope["component"]["type"] == "table"


@pytest.mark.parametrize(
    ("facet", "value", "label"),
    [("unit", "USDollar", "US Dollar"), ("measurementMethod", "Survey", "Survey")],
)
def test_table_labels_the_facet_of_two_facets_of_one_stat_var(release, facet, value, label):
    """Two facets of one stat_var share provenance and differ only in `facet` (set
    on one, null on the other). `facet` is the only unselected column, and the two
    series differ in `dimensions` by that key. The selectors name the series by id,
    since bindings refuse a facet that splits one series."""
    gdp_ids = [
        mint_series_id(
            coordinates=tuple(
                sorted(
                    {
                        "stat_var": "dcid:GDP",
                        "recipient": "country/GTM",
                        "unit": None,
                        "measurementMethod": None,
                        "provenance": "src_oecd_dac",
                        facet: facet_value,
                    }.items()
                )
            )
        )
        for facet_value in (value, None)
    ]
    minted, bound = _mint_gtm_table(
        release=release,
        resolved_frame=_mixed_unit_resolved_frame(
            unit_series=(
                ("dcid:GDP", value, {"2020": 77.6, "2021": 86.0}),
                ("dcid:GDP", None, {"2020": 16.9, "2021": 17.1}),
            ),
            facet=facet,
        ),
        stat_vars=["dcid:GDP"],
        selectors=[{"id": series_id} for series_id in gdp_ids],
    )
    component = minted.component
    assert [col["key"] for col in component["columns"]] == [
        "stat_var",
        "recipient",
        facet,
        "period",
        "value",
    ]
    assert len(component["rows"]) == 4
    assert len({b.id for b in bound}) == 2

    maps = _dimension_maps_by_series(component)
    assert set(maps) == {b.id for b in bound}
    assert all(len(series_maps) == 1 for series_maps in maps.values())
    first, second = (dict(next(iter(series_maps))) for series_maps in maps.values())
    assert {k for k in first if first[k] != second[k]} == {facet}
    assert {first[facet], second[facet]} == {label, "Not specified"}

    envelope = stamp_envelope(
        minted=minted, component_id="cmp_" + "i" * 20, revision=1, release=release
    )
    assert envelope["component"]["type"] == "table"


# -- stat_card ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "route"),
    [
        pytest.param(None, "via", id="headline"),
        pytest.param(
            {"basis": ["2023", "2024"], "derivation": "difference"}, "processed by", id="change"
        ),
    ],
)
def test_stat_card_mints_a_schema_valid_envelope(release, resolved_frame, change, route):
    """The citation routes through the DFO as the card's figures say: a headline as the
    dataset publishes it reads "via", and a change the card computes reads "processed by"."""
    series, bound = _resolve_and_bind(resolved_frame=resolved_frame, selectors=_malaria_selectors())
    intent = {
        "type": "stat_card",
        "frame_ref": {
            "spec_id": resolved_frame.frame_ref.spec_id,
            "content_version": resolved_frame.frame_ref.content_version,
        },
        "select": _malaria_select(),
    }
    if change is not None:
        intent["change"] = change
    minted = mint_component(
        member="stat_card",
        merged_intent=intent,
        resolved_frame=resolved_frame,
        series=series,
        bound_series=bound,
        release=release,
        clock=_fixed_clock(),
    )
    envelope = stamp_envelope(
        minted=minted, component_id="cmp_" + "b" * 20, revision=1, release=release
    )
    assert envelope["component"]["type"] == "stat_card"
    assert envelope["component"]["value"] == pytest.approx(441.9)
    assert envelope["component"]["period"] == "2024"
    assert f"{route} the Development Finance Observatory" in envelope["provenance"]["citation"]


def test_stat_card_headline_value_is_the_latest_observed_point(release, resolved_frame):
    """The headline uses the latest observed point; absent `change` stays absent."""
    series, bound = _resolve_and_bind(resolved_frame=resolved_frame, selectors=_malaria_selectors())
    intent = {
        "type": "stat_card",
        "frame_ref": {
            "spec_id": resolved_frame.frame_ref.spec_id,
            "content_version": resolved_frame.frame_ref.content_version,
        },
        "select": _malaria_select(),
    }
    minted = mint_component(
        member="stat_card",
        merged_intent=intent,
        resolved_frame=resolved_frame,
        series=series,
        bound_series=bound,
        release=release,
        clock=_fixed_clock(),
    )
    assert minted.component["value"] == pytest.approx(441.9)
    assert minted.component["period"] == "2024"
    assert "change" not in minted.component


def test_stat_card_period_tracks_the_headline_point_not_the_window_end(release, resolved_frame):
    """The 2023 headline value carries period 2023 although the window ends in 2024."""
    bound = (
        BoundSeries(
            id="eth",
            label="Ethiopia",
            unit=None,
            source_ref="src_oecd_dac",
            points=(
                {"period": "2023", "value": 433.7},
                {"period": "2024", "status": "missing"},
            ),
        ),
    )
    intent = {
        "type": "stat_card",
        "frame_ref": {
            "spec_id": resolved_frame.frame_ref.spec_id,
            "content_version": resolved_frame.frame_ref.content_version,
        },
        "select": _malaria_select(),
    }
    minted = mint_component(
        member="stat_card",
        merged_intent=intent,
        resolved_frame=resolved_frame,
        series=(),
        bound_series=bound,
        release=release,
        clock=_fixed_clock(),
    )
    assert minted.component["value"] == pytest.approx(433.7)
    assert minted.component["period"] == "2023"


def test_stat_card_with_no_valued_point_refuses_binding_invalid(release, resolved_frame):
    """A status-only series has no headline value and raises `BindingInvalid`."""
    bound = (
        BoundSeries(
            id="eth",
            label="Ethiopia",
            unit=None,
            source_ref="src_oecd_dac",
            points=(
                {"period": "2023", "status": "missing"},
                {"period": "2024", "status": "missing"},
            ),
        ),
    )
    intent = {
        "type": "stat_card",
        "frame_ref": {
            "spec_id": resolved_frame.frame_ref.spec_id,
            "content_version": resolved_frame.frame_ref.content_version,
        },
        "select": _malaria_select(),
    }
    with pytest.raises(BindingInvalid) as excinfo:
        mint_component(
            member="stat_card",
            merged_intent=intent,
            resolved_frame=resolved_frame,
            series=(),
            bound_series=bound,
            release=release,
            clock=_fixed_clock(),
        )
    assert excinfo.value.pointer == "/select"
    assert "no observed point" in excinfo.value.detail


def test_observed_periods_excludes_missing_points(release, resolved_frame):
    series, bound = _resolve_and_bind(
        resolved_frame=resolved_frame,
        selectors=_malaria_selectors(),
        window={"start": "2015", "end": "2024"},
    )
    intent = {
        "type": "line_chart",
        "frame_ref": {
            "spec_id": resolved_frame.frame_ref.spec_id,
            "content_version": resolved_frame.frame_ref.content_version,
        },
        "select": _malaria_select(),
        "window": {"start": "2015", "end": "2024"},
    }
    minted = mint_component(
        member="line_chart",
        merged_intent=intent,
        resolved_frame=resolved_frame,
        series=series,
        bound_series=bound,
        release=release,
        clock=_fixed_clock(),
    )
    assert "2018" not in minted.observed_periods
    assert "2015" in minted.observed_periods


def test_default_window_falls_back_to_frame_coverage(resolved_frame):
    window = default_window(resolved_frame=resolved_frame)
    assert window == {"start": "2015", "end": "2024"}


# -- note_narration --------------------------------------------------------


def test_note_narration_mints_a_schema_valid_envelope(release):
    intent = {"type": "note_narration", "text": "Disbursements rose each year after 2020."}
    minted = mint_component(
        member="note_narration",
        merged_intent=intent,
        resolved_frame=None,
        series=(),
        bound_series=(),
        release=release,
        clock=_fixed_clock(),
    )
    assert minted.component == {
        "type": "note_narration",
        "text": "Disbursements rose each year after 2020.",
        "authorship": "model",
    }
    envelope = stamp_envelope(
        minted=minted, component_id="cmp_" + "g" * 20, revision=1, release=release
    )
    assert envelope["component"]["type"] == "note_narration"


def test_note_narration_provenance_names_no_inputs_and_no_sources(release):
    """A note narration keeps lineage alone."""
    intent = {"type": "note_narration", "text": "No frame is resolved for this member."}
    minted = mint_component(
        member="note_narration",
        merged_intent=intent,
        resolved_frame=None,
        series=(),
        bound_series=(),
        release=release,
        clock=_fixed_clock(),
    )
    assert minted.provenance["inputs"] == []
    assert minted.observed_periods == frozenset()
    assert not set(DESCRIPTION_PROVENANCE_MEMBERS) & set(minted.provenance)


# -- source_card -------------------------------------------------------------


def test_source_card_mints_a_schema_valid_envelope(release, resolved_frame):
    intent = {
        "type": "source_card",
        "frame_ref": {
            "spec_id": resolved_frame.frame_ref.spec_id,
            "content_version": resolved_frame.frame_ref.content_version,
        },
    }
    minted = mint_component(
        member="source_card",
        merged_intent=intent,
        resolved_frame=resolved_frame,
        series=(),
        bound_series=(),
        release=release,
        clock=_fixed_clock(),
    )
    assert minted.component == {"type": "source_card", "source_refs": ["src_oecd_dac"]}
    envelope = stamp_envelope(
        minted=minted, component_id="cmp_" + "h" * 20, revision=1, release=release
    )
    assert envelope["component"]["type"] == "source_card"
    assert envelope["provenance"]["sources"][0]["id"] == "src_oecd_dac"


def test_stamp_envelope_refuses_a_source_ref_no_listed_source_names(release, resolved_frame):
    """`src_other` passes the envelope schema, and `provenance.sources` lists no such id."""
    minted = mint_component(
        member="source_card",
        merged_intent={
            "type": "source_card",
            "frame_ref": {
                "spec_id": resolved_frame.frame_ref.spec_id,
                "content_version": resolved_frame.frame_ref.content_version,
            },
        },
        resolved_frame=resolved_frame,
        series=(),
        bound_series=(),
        release=release,
        clock=_fixed_clock(),
    )
    unlisted = dataclasses.replace(
        minted, component={"type": "source_card", "source_refs": ["src_oecd_dac", "src_other"]}
    )

    with pytest.raises(SchemaInvalid) as refused:
        stamp_envelope(minted=unlisted, component_id="cmp_" + "h" * 20, revision=1, release=release)

    assert refused.value.pointer == "/component/source_refs/1"


def test_source_card_names_every_source_the_frame_carries_not_just_a_series_subset(release):
    """`source_card` lists distinct provenance refs from all frame rows."""
    rows = [
        {
            "stat_var": "dcid:X",
            "recipient": "country/ETH",
            "time_period": "2020",
            "value": 1.0,
            "unit": None,
            "measurementMethod": None,
            "provenance": "src_oecd_dac",
        },
        {
            "stat_var": "dcid:X",
            "recipient": "country/GHA",
            "time_period": "2020",
            "value": 2.0,
            "unit": None,
            "measurementMethod": None,
            "provenance": "src_worldbank",
        },
    ]
    rf = build_resolved_frame(
        frame_ref=malaria_ethiopia_frame_ref(), role_keys=("recipient",), rows=rows
    )
    intent = {
        "type": "source_card",
        "frame_ref": {
            "spec_id": rf.frame_ref.spec_id,
            "content_version": rf.frame_ref.content_version,
        },
    }
    minted = mint_component(
        member="source_card",
        merged_intent=intent,
        resolved_frame=rf,
        series=(),
        bound_series=(),
        release=release,
        clock=_fixed_clock(),
    )
    assert set(minted.component["source_refs"]) == {"src_oecd_dac", "src_worldbank"}
    assert len(minted.provenance["sources"]) == 2


# -- provenance sources and freshness --------------------------------------

_WDI_REF = "dc/base/WorldDevelopmentIndicators"
_UNCITED_REF = "dc/base/NotInTheCitation"


def _two_source_frame(*, manifest):
    rows = [
        {
            "stat_var": "dcid:X",
            "recipient": "country/ETH",
            "time_period": "2020",
            "value": 1.0,
            "unit": None,
            "measurementMethod": None,
            "provenance": _WDI_REF,
        },
        {
            "stat_var": "dcid:X",
            "recipient": "country/GHA",
            "time_period": "2020",
            "value": 2.0,
            "unit": None,
            "measurementMethod": None,
            "provenance": _UNCITED_REF,
        },
    ]
    return build_resolved_frame(
        frame_ref=malaria_ethiopia_frame_ref(),
        role_keys=("recipient",),
        rows=rows,
        manifest=manifest,
        retrieved_on=datetime.date(2026, 9, 12),
    )


def _mint_source_card(*, resolved_frame, release):
    intent = {
        "type": "source_card",
        "frame_ref": {
            "spec_id": resolved_frame.frame_ref.spec_id,
            "content_version": resolved_frame.frame_ref.content_version,
        },
    }
    return mint_component(
        member="source_card",
        merged_intent=intent,
        resolved_frame=resolved_frame,
        series=(),
        bound_series=(),
        release=release,
        clock=_fixed_clock(),
    )


@pytest.mark.parametrize(
    ("frame_ref", "route"),
    [
        pytest.param(malaria_ethiopia_frame_ref(), "via", id="dataset-frame"),
        pytest.param(
            FrameRef(spec_id="ops_" + "c" * 32, content_version="ocv_" + "d" * 32),
            "processed by",
            id="computed-frame",
        ),
    ],
)
def test_sources_carry_the_manifest_citation_and_the_frame_s_retrieval_date(
    release, frame_ref, route
):
    """Citation metadata comes from the manifest. `retrieved` uses the frame date, 2026-09-12.
    The mint clock date is 2026-07-14. Unlisted refs use their raw ref as name, an empty URL, and
    `Not stated` for licence. The citation credits the sources, "processed by" the DFO where
    the frame is a computed version.
    """
    manifest = {
        "schema_version": 1,
        "citation": {
            "text": "World Development Indicators",
            "sources": [
                {
                    "ref": _WDI_REF,
                    "label": "World Development Indicators",
                    "license": "Creative Commons Attribution License",
                    "url": "https://datacatalog.worldbank.org/search/dataset/0037712",
                }
            ],
        },
    }
    resolved_frame = dataclasses.replace(_two_source_frame(manifest=manifest), frame_ref=frame_ref)
    minted = _mint_source_card(resolved_frame=resolved_frame, release=release)

    names = {source["name"]: source for source in minted.provenance["sources"]}
    wdi = names["World Development Indicators"]
    assert wdi["url"] == "https://datacatalog.worldbank.org/search/dataset/0037712"
    assert wdi["licence"] == "Creative Commons Attribution License"
    assert wdi["retrieved"] == "2026-09-12"
    assert wdi["ref"] == _WDI_REF
    uncited = names[_UNCITED_REF]
    assert uncited["ref"] == _UNCITED_REF
    assert (uncited["url"], uncited["licence"], uncited["retrieved"]) == (
        "",
        "Not stated",
        "2026-09-12",
    )
    assert minted.provenance["citation"] == (
        f"World Development Indicators and {_UNCITED_REF} (2026), {route} the Development"
        " Finance Observatory"
    )
    envelope = stamp_envelope(
        minted=minted, component_id="cmp_" + "s" * 20, revision=1, release=release
    )
    assert envelope["provenance"]["sources"] == minted.provenance["sources"]


@pytest.mark.parametrize(
    ("raw_ref", "kept"),
    [
        pytest.param(None, False, id="unattributed"),
        pytest.param("dc/base/" + "x" * 192, True, id="at-the-schema-maximum"),
        pytest.param("dc/base/" + "x" * 193, False, id="over-the-schema-maximum"),
    ],
)
def test_source_keeps_ref_only_when_the_raw_ref_can_stand_for_a_dcid(release, raw_ref, kept):
    rows = [
        {
            "stat_var": "dcid:X",
            "recipient": "country/ETH",
            "time_period": "2020",
            "value": 1.0,
            "unit": None,
            "measurementMethod": None,
            "provenance": raw_ref,
        }
    ]
    resolved_frame = build_resolved_frame(
        frame_ref=malaria_ethiopia_frame_ref(), role_keys=("recipient",), rows=rows
    )
    minted = _mint_source_card(resolved_frame=resolved_frame, release=release)

    (source,) = minted.provenance["sources"]
    assert (source.get("ref"), "ref" in source) == ((raw_ref, True) if kept else (None, False))
    # Stamping validates the envelope against the schema.
    stamp_envelope(minted=minted, component_id="cmp_" + "s" * 20, revision=1, release=release)


_DAC2A_REF = "dc/base/OECD_DAC2a"
_DERIVED_FRAME_REF = FrameRef(spec_id="ops_" + "c" * 32, content_version="ocv_" + "d" * 32)
_ROOTS_CITATION = {
    "text": "OECD DAC2a.\n\nWorld Development Indicators.",
    "sources": [
        {"ref": _DAC2A_REF, "label": "OECD DAC2a", "license": None, "url": "https://oecd.org"},
        {"ref": _WDI_REF, "label": "World Development Indicators", "license": None, "url": ""},
    ],
}
"""The citation `datasets_api.derived.recipe_citation` gives a quotient of an OECD DAC2a root
over a WDI root."""


def _mint_on_a_quotient_frame(member, *, release, frame_ref=_DERIVED_FRAME_REF):
    """`member` drawn on a frame at `frame_ref` whose rows name no provenance, as a quotient of
    two roots' figures has no one source, under a citation of two sources."""
    rows = [
        {
            "stat_var": "dcid:ops_quotient",
            "recipient": "country/KEN",
            "time_period": period,
            "value": value,
            "unit": None,
            "measurementMethod": None,
            "provenance": None,
        }
        for period, value in (("2020", 1.5), ("2021", 2.5))
    ]
    resolved_frame = build_resolved_frame(
        frame_ref=frame_ref,
        role_keys=("recipient",),
        rows=rows,
        manifest={"schema_version": 1, "citation": _ROOTS_CITATION},
    )
    if member == "source_card":
        return _mint_source_card(resolved_frame=resolved_frame, release=release)
    selectors = [
        {
            "bindings": [
                {"dimension": "stat_var", "values": ["dcid:ops_quotient"]},
                {"dimension": "recipient", "values": ["country/KEN"]},
            ]
        }
    ]
    series, bound = _resolve_and_bind(resolved_frame=resolved_frame, selectors=selectors)
    intent = {
        "type": member,
        "frame_ref": {
            "spec_id": frame_ref.spec_id,
            "content_version": frame_ref.content_version,
        },
        "select": {"stat_var": "dcid:ops_quotient", "recipient": "country/KEN"},
    }
    return mint_component(
        member=member,
        merged_intent=intent,
        resolved_frame=resolved_frame,
        series=series,
        bound_series=bound,
        release=release,
        clock=_fixed_clock(),
    )


@pytest.mark.parametrize("member", ["table", "line_chart", "source_card"])
def test_a_card_on_a_derived_frame_cites_its_roots_sources(release, member):
    """A row naming no provenance stands for every source its frame cites, so the card lists
    the roots' sources and names one of them on each value, where it once named an
    unattributed source."""
    minted = _mint_on_a_quotient_frame(member, release=release)

    sources = minted.provenance["sources"]
    assert [(source["name"], source["ref"]) for source in sources] == [
        ("OECD DAC2a", _DAC2A_REF),
        ("World Development Indicators", _WDI_REF),
    ]
    listed = {source["id"] for source in sources}
    component = minted.component
    named = {
        *(entry["source_ref"] for key in ("rows", "series") for entry in component.get(key, ())),
        *component.get("source_refs", ()),
    }
    assert named and named <= listed
    assert minted.provenance["citation"].startswith("OECD DAC2a and World Development Indicators")
    stamp_envelope(minted=minted, component_id="cmp_" + "s" * 20, revision=1, release=release)


@pytest.mark.parametrize("member", ["table", "line_chart", "source_card"])
def test_a_card_on_a_dataset_frame_leaves_a_row_with_no_provenance_unattributed(release, member):
    """A dataset row with no provenance names none of its version's sources, so the card
    names the unattributed source for it rather than the first source cited."""
    dataset_frame = FrameRef(spec_id="spec_" + "c" * 32, content_version="cv_" + "d" * 32)

    minted = _mint_on_a_quotient_frame(member, release=release, frame_ref=dataset_frame)

    assert [source["name"] for source in minted.provenance["sources"]] == ["Unattributed source"]


def test_sources_carry_the_citation_s_source_and_data_commons_as_id_and_name(release):
    """A citation entry's `source` and `data_commons` become the envelope source's
    `{id, name}` members. A ref with no entry, like a version minted before the origin
    catalogue, carries neither."""
    manifest = {
        "schema_version": 1,
        "citation": {
            "text": "World Development Indicators",
            "sources": [
                {
                    "ref": _WDI_REF,
                    "label": "World Development Indicators",
                    "license": None,
                    "url": "https://datacatalog.worldbank.org/search/dataset/0037712",
                    "source": {"ref": "dc/s/WorldBank", "label": "World Bank"},
                    "data_commons": {"id": "base", "label": "Data Commons by Google"},
                }
            ],
        },
    }
    minted = _mint_source_card(resolved_frame=_two_source_frame(manifest=manifest), release=release)

    names = {source["name"]: source for source in minted.provenance["sources"]}
    wdi = names["World Development Indicators"]
    assert wdi["source"] == {"id": "dc/s/WorldBank", "name": "World Bank"}
    assert wdi["data_commons"] == {"id": "base", "name": "Data Commons by Google"}
    assert "source" not in names[_UNCITED_REF]
    assert "data_commons" not in names[_UNCITED_REF]
    envelope = stamp_envelope(
        minted=minted, component_id="cmp_" + "s" * 20, revision=1, release=release
    )
    assert envelope["provenance"]["sources"] == minted.provenance["sources"]


def test_freshness_data_through_is_the_manifest_window_s_end_year(release):
    manifest = {
        "schema_version": 1,
        "coverage": {
            "rows": 2,
            "dimensions": [],
            "window": {"kind": "all", "start_year": 2015, "end_year": 2020, "origin": "default"},
        },
    }
    minted = _mint_source_card(resolved_frame=_two_source_frame(manifest=manifest), release=release)

    assert minted.provenance["freshness"] == {
        "data_through": "2020",
        "last_refreshed": None,
        "next_expected": None,
        "partial_period": False,
    }


def test_freshness_data_through_is_null_when_the_manifest_records_no_window(release):
    minted = _mint_source_card(
        resolved_frame=_two_source_frame(manifest={"schema_version": 1}), release=release
    )

    assert minted.provenance["freshness"]["data_through"] is None


# -- provenance warnings -------------------------------------------------


def _resolved_frame_with_warnings(warnings, *, labels=()):
    return build_resolved_frame(
        frame_ref=malaria_ethiopia_frame_ref(),
        role_keys=("recipient",),
        rows=malaria_ethiopia_rows(),
        labels={"country/ETH": "Ethiopia", **dict(labels)},
        manifest={"schema_version": 1, "warnings": list(warnings)},
    )


def _mint_line_chart_provenance(*, resolved_frame, release):
    series, bound = _resolve_and_bind(resolved_frame=resolved_frame, selectors=_malaria_selectors())
    intent = {
        "type": "line_chart",
        "frame_ref": {
            "spec_id": resolved_frame.frame_ref.spec_id,
            "content_version": resolved_frame.frame_ref.content_version,
        },
        "select": _malaria_select(),
    }
    minted = mint_component(
        member="line_chart",
        merged_intent=intent,
        resolved_frame=resolved_frame,
        series=series,
        bound_series=bound,
        release=release,
        clock=_fixed_clock(),
    )
    return minted.provenance


@pytest.mark.parametrize(
    ("code", "expected_kind"),
    [
        ("revision_unavailable", "provisional"),
        ("facet_default_selected", "comparability_break"),
        ("enrichment_unavailable", "coverage_gap"),
        ("processing_level_undeclared", "provisional"),
        ("coverage_below_scope", "coverage_gap"),
        ("empty_result", "coverage_gap"),
        ("deflator_projected", "estimated"),
    ],
)
def test_provenance_warnings_translate_each_mapped_code_to_its_envelope_kind(
    release, code, expected_kind
):
    resolved_frame = _resolved_frame_with_warnings(
        [{"code": code, "severity": "info", "message": "a manifest-authored detail"}]
    )
    provenance = _mint_line_chart_provenance(resolved_frame=resolved_frame, release=release)
    assert provenance["warnings"] == [
        {"kind": expected_kind, "detail": "a manifest-authored detail"}
    ]


def test_every_datasets_warning_catalogue_code_has_an_envelope_kind_entry():
    """Every datasets warning code needs a `_WARNING_KIND_BY_CODE` entry.

    Translation raises `AssertionError` when a manifest carries an unknown code.
    """
    missing = sorted(set(DATASETS_WARNING_CATALOGUE) - set(_WARNING_KIND_BY_CODE))
    assert missing == []


def test_provenance_warnings_translate_independently_and_preserve_order(release):
    resolved_frame = _resolved_frame_with_warnings(
        [
            {"code": "revision_unavailable", "severity": "info", "message": "no live revision"},
            {"code": "coverage_below_scope", "severity": "warn", "message": "short of scope"},
        ]
    )
    provenance = _mint_line_chart_provenance(resolved_frame=resolved_frame, release=release)
    assert provenance["warnings"] == [
        {"kind": "provisional", "detail": "no live revision"},
        {"kind": "coverage_gap", "detail": "short of scope"},
    ]


def _long_label(index: int) -> str:
    return f"{index}".ljust(LABEL_MAX, "x")


_LONG = {f"dcid:Long{i}": _long_label(i) for i in range(10)}


@pytest.mark.parametrize(
    ("members", "expected"),
    [
        pytest.param(
            {"off_cadence_members": {"count": 2, "refs": ["dcid:Health", "dcid:Water"]}},
            "Short of scope. At a different frequency: Basic health, Water.",
            id="other-frequency",
        ),
        pytest.param(
            {"unserved_members": {"count": 1, "refs": ["dcid:Water"]}},
            "Short of scope. Under another source, unit or method: Water.",
            id="unserved-members",
        ),
        # A member the labels hold no name for reads as its bare dcid, and the members the
        # field counts past its refs as a count.
        pytest.param(
            {"out_of_window_members": {"count": 7, "refs": ["dcid:Health", "dcid:Roads"]}},
            "Short of scope. No observations in the requested years: "
            "Basic health, Roads and 5 more.",
            id="out-of-window",
        ),
        # Long names share the room the message leaves: each field names the whole labels
        # that fit and counts the rest, and a field fitting none states its count alone.
        pytest.param(
            {
                "off_cadence_members": {"count": 6, "refs": [f"dcid:Long{i}" for i in range(5)]},
                "out_of_window_members": {
                    "count": 6,
                    "refs": [f"dcid:Long{i}" for i in range(5, 10)],
                },
            },
            f"Short of scope. At a different frequency: {_long_label(0)} and 5 more. "
            "No observations in the requested years: 6 series.",
            id="both-long",
        ),
        # Entities lead the members, as the message counts them first, and a field fitting
        # no name counts them in its own noun.
        pytest.param(
            {
                "unserved_entities": {"count": 1, "refs": ["dcid:country/KEN"]},
                "off_cadence_members": {"count": 1, "refs": ["dcid:Health"]},
            },
            "Short of scope. Not served: Kenya. At a different frequency: Basic health.",
            id="unserved-entities",
        ),
        pytest.param(
            {
                "unserved_entities": {"count": 2, "refs": ["dcid:Long0", "dcid:Long1"]},
                "off_cadence_members": {
                    "count": 6,
                    "refs": [f"dcid:Long{i}" for i in range(2, 7)],
                },
                "out_of_window_members": {"count": 1, "refs": ["dcid:Long7"]},
            },
            "Short of scope. Not served: 2 entities. At a different frequency: 6 series. "
            f"No observations in the requested years: {_long_label(7)}.",
            id="entities-long",
        ),
    ],
)
def test_a_coverage_warning_names_its_members_by_label(release, members, expected):
    resolved_frame = _resolved_frame_with_warnings(
        [
            {
                "code": "coverage_below_scope",
                "severity": "warn",
                "message": "Short of scope.",
                "member_count": 9,
                **members,
            }
        ],
        labels={
            "dcid:Health": "Basic health",
            "dcid:Water": "Water",
            "dcid:country/KEN": "Kenya",
            **_LONG,
        },
    )

    provenance = _mint_line_chart_provenance(resolved_frame=resolved_frame, release=release)

    assert provenance["warnings"] == [{"kind": "coverage_gap", "detail": expected}]
    assert warned_refs(manifest=resolved_frame.manifest) == [
        ref for field in members.values() for ref in field["refs"]
    ]


@pytest.mark.parametrize(
    ("code", "field", "lead", "named", "refs"),
    [
        pytest.param(
            "currency_unmapped_omitted",
            {"omitted_entities": {"count": 3, "refs": ["dcid:country/KEN", "dcid:XYZ"]}},
            "Left out",
            "Kenya, XYZ",
            ["dcid:country/KEN", "dcid:XYZ"],
            id="currency",
        ),
        # Each entity's name carries the years its rows were left out.
        pytest.param(
            "deflator_missing_omitted",
            {
                "omitted_entity_years": {
                    "count": 3,
                    "entities": [
                        {"ref": "dcid:country/KEN", "years": ["1990-2001", "2004"]},
                        {"ref": "dcid:XYZ", "years": ["2020"]},
                    ],
                }
            },
            "Left out",
            "Kenya (1990-2001, 2004), XYZ (2020)",
            ["dcid:country/KEN", "dcid:XYZ"],
            id="deflator",
        ),
        pytest.param(
            "withheld_by_registry",
            {"withheld_members": {"count": 3, "refs": ["dcid:Water", "dcid:XYZ"]}},
            "Withheld series",
            "Water, XYZ",
            ["dcid:Water", "dcid:XYZ"],
            id="withheld-members",
        ),
        # Provenances of one Dataset share its name, which a card gives once.
        pytest.param(
            "withheld_by_registry",
            {
                "withheld_provenances": {
                    "count": 3,
                    "refs": ["dcid:dc/base/WHO_GHO", "dcid:dc/base/WHO_GHO_Archive"],
                }
            },
            "Withheld sources",
            "WHO Global Health Observatory",
            [],
            id="withheld-provenances",
        ),
    ],
)
def test_a_counted_field_warning_names_what_it_counts_by_label(
    release, code, field, lead, named, refs
):
    resolved_frame = _resolved_frame_with_warnings(
        [{"code": code, "severity": "warn", "message": "The message.", **field}],
        labels={
            "dcid:country/KEN": "Kenya",
            "dcid:Water": "Water",
            "dcid:dc/base/WHO_GHO": "WHO Global Health Observatory",
            "dcid:dc/base/WHO_GHO_Archive": "WHO Global Health Observatory",
        },
    )

    provenance = _mint_line_chart_provenance(resolved_frame=resolved_frame, release=release)

    assert provenance["warnings"] == [
        {"kind": "coverage_gap", "detail": f"The message. {lead}: {named} and 1 more."}
    ]
    assert warned_refs(manifest=resolved_frame.manifest) == refs


def test_a_card_names_every_counted_field_a_stored_warning_holds():
    """Each `WarningModel` field holding counted refs has a `_REF_FIELDS` row, so the frame
    store labels its refs and a card names them; a field without one would go unnamed."""
    counted = {
        name
        for name, field in WarningModel.model_fields.items()
        if {CountedRefsModel, CountedEntityYearsModel} & set(get_args(field.annotation))
    }

    assert counted == {field.key for field in _REF_FIELDS}


def test_provenance_warnings_refuses_a_code_absent_from_the_translation_table(release):
    """Unknown manifest codes raise `AssertionError` before minting an invalid envelope."""
    resolved_frame = _resolved_frame_with_warnings(
        [{"code": "not_a_real_code", "severity": "info", "message": "unexpected"}]
    )
    with pytest.raises(AssertionError, match="not_a_real_code"):
        _mint_line_chart_provenance(resolved_frame=resolved_frame, release=release)
