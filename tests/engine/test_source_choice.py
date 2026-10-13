"""A part's source scope through the region pipeline: the scoped read, the provenance pin,
the restatement, the UN gate, the source slot's narrowing, recall's completion under a scope,
and the no-data part a source refusal answers."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import Callable
from types import SimpleNamespace
from typing import Literal, cast

import pytest

from code_index.shared import get_index
from origin_catalogue import BASE, UN_SYSTEM, Catalogue, ProvenanceRow, SourceRef
from origin_catalogue.context import current_source_scope
from search.engine.clients.graph import (
    DetectedSV,
    EngineGraphClient,
    Facet,
    with_rival_families,
)
from search.engine.conjoin import _build_part, _unbound_terms
from search.engine.core import resolve_async
from search.engine.entry.parsed import _dedupe_parts_with_indices
from search.engine.fanout import restated_sources
from search.engine.options import surface as surface_module
from search.engine.options.resolve import ResolvedOptionsTarget
from search.engine.quantity_read import QuantityOutput
from search.engine.region.arcs import standard_bindings_from_arcs
from search.engine.region.facets import _resolved_facets
from search.engine.region.named_source import with_named_source
from search.engine.region.pipeline import resolve_variable
from search.engine.region.result import RegionResult, source_refusal_result
from search.engine.region.source_choice import (
    allows_un_system,
    idle_exclusion_warnings,
    with_idle_exclusions,
)
from search.engine.retrieval.bound_values import identity_query, shape_for_part, shape_query
from search.engine.role_text import DoorRole
from search.engine.shape import ShapeDraft, identity_shape_for
from search.engine.source_scope import (
    AmbiguousSource,
    SourceReference,
    SourceRefusal,
    SourceScope,
    compile_sources,
)
from search.engine.spec_build import build_slot, pair_slot_drafts, with_served_provenance
from search.engine.spec_id import (
    bound_facet_axes,
    build_identity_projection,
    build_spec_identity,
)
from search.models import (
    BindingUnbound,
    GraphRef,
    IdentityBindingValue,
    OptionsRequest,
    ParsedInput,
    ParsedPart,
    ResolvedQualifier,
    ResolveRequest,
    SeriesFacetRef,
    Slot,
    SlotKey,
    SourceMention,
)
from series_coverage.shared import CoverageSnapshot
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._quantity import NO_QUANTITY
from tests.engine.test_admissibility_table import _FACETS, _GDP_SOLE_OPTION, _node, _SoleOptionLLM
from tests.engine.test_ids_index_sourcing import _crs_arcs, _derive_one, _ids_shape
from tests.engine.test_parsed_pipeline import _fertility_kenya_part
from tests.fixtures import FakeGraph, _load_json, make_resolver
from tests.origin_catalogue.fake_catalogue import FakeCatalogue

_WDI = "dc/base/WorldDevelopmentIndicators"
_CENSUS = "dc/base/Kenya_Census"
_UN_ROW = "undata/p/WHO"


def _row(provenance: str, *, data_commons=BASE) -> ProvenanceRow:
    name = provenance.rsplit("/", 1)[-1]
    return ProvenanceRow(
        id=provenance,
        data_commons=data_commons,
        source=SourceRef(ref=f"dc/s/{name}", label=name),
        name=name,
        url="https://example.org",
    )


class _ScopedFakeGraph(FakeGraph):
    """`FakeGraph` whose facet reads keep only the provenances the bound catalogue's source
    scope allows, as the origin graph client's admission does (`Catalogue.admit_series`)."""

    def observation_facets(
        self, *, stat_var: str, entity: str, needs_dates: bool = False
    ) -> list[Facet]:
        facets = super().observation_facets(
            stat_var=stat_var, entity=entity, needs_dates=needs_dates
        )
        scope = current_source_scope()
        return [f for f in facets if scope is None or f.provenance_id in scope]


@pytest.fixture
def fertility_sources(bind_catalogue: Callable[[Catalogue], None]) -> _ScopedFakeGraph:
    """Kenya's fertility rate under two base provenances, World Development Indicators,
    which the plain question serves, and the Kenya census, both admitted by name, beside
    the CRS, which publishes no fertility rate."""
    bind_catalogue(FakeCatalogue(provenances={p: _row(p) for p in (_WDI, _CENSUS, _CRS)}))
    return _ScopedFakeGraph(obs=_load_json("graph_obs.json"))


class _FirstShapeLLM:
    """A bind model that elects the first shape of its menu and binds nothing."""

    async def generate_structured(self, *, prompt, system, schema):
        return schema(chosen_shape_index=0, bindings=[], also_plausible=None, ask=None), None


def _resolve(parts: list[ParsedPart], graph: FakeGraph):
    request = ResolveRequest(input=ParsedInput(parts=parts))
    return asyncio.run(
        resolve_async(request, graph=graph, llm=_FirstShapeLLM(), sdmx=FakeSdmxClient())
    )


def _answer(graph: FakeGraph, sources: list[SourceMention]):
    part = _fertility_kenya_part().model_copy(update={"sources": sources})
    (answered,) = _resolve([part], graph).parts
    return answered


def _pin(answered) -> object:
    facets = answered.spec.definition.identity.facets or {}
    binding = facets.get("provenance")
    return None if binding is None else binding.ref


@pytest.mark.parametrize(
    ("sources", "served", "pinned"),
    [
        ([], _WDI, None),
        ([SourceMention(ref=_CENSUS)], _CENSUS, _CENSUS),
        ([SourceMention(ref=_WDI, polarity="exclude")], _CENSUS, _CENSUS),
        ([SourceMention(ref=_WDI)], _WDI, _WDI),
    ],
    ids=["no-mention", "include-census", "exclude-wdi", "include-wdi"],
)
def test_a_part_reads_its_scope_and_pins_the_provenance_that_served(
    fertility_sources, sources, served, pinned
):
    """A part naming or excluding a source reads only the provenances its scope allows, and
    its identity binds the one that served, even where the plain question would serve the
    same one. A part naming none keeps the plain identity. The restatement echoes the
    mentions with their references, and posting it back mints the same spec_id."""
    answered = _answer(fertility_sources, sources)

    provenance_facet = answered.spec.resolved.facets["provenance"]
    assert provenance_facet.ref == served
    assert provenance_facet.basis == ("fallback" if pinned is None else "bound")
    assert _pin(answered) == pinned
    assert answered.restatement.sources == sources
    assert answered.restatement.provenance is None
    assert answered.restatement.source_text is None
    (reposted,) = _resolve([answered.restatement], fertility_sources).parts
    assert reposted.spec.spec_id == answered.spec.spec_id


def test_a_variable_repeating_its_source_s_words_reads_the_measure_s_words_alone(
    fertility_sources,
):
    """ "Kenya Census fertility rate" naming the source "Kenya Census" reads "fertility rate"
    under that source: the words read as the source are not the measure's too. The part
    answers, restates and mints as the part posting the measure's words alone."""
    census = [SourceMention(text="Kenya Census")]
    worded = _fertility_kenya_part().model_copy(
        update={"variable_text": "Kenya Census fertility rate", "sources": census}
    )
    (answered,) = _resolve([worded], fertility_sources).parts
    plain = _answer(fertility_sources, census)

    assert answered.restatement.variable_text == "fertility rate"
    assert answered.spec.spec_id == plain.spec.spec_id
    assert _pin(answered) == _CENSUS


def test_the_deprecated_provenance_field_mints_the_spec_id_of_its_mention(fertility_sources):
    """`provenance` folds into one included mention with that ref, so a part posting it and
    one posting the mention answer under one spec_id, and both restate `sources` alone."""
    posted = _fertility_kenya_part().model_copy(update={"provenance": _CENSUS})
    by_field = _resolve([ParsedPart.model_validate(posted.model_dump())], fertility_sources)
    by_mention = _answer(fertility_sources, [SourceMention(ref=_CENSUS)])

    (answered,) = by_field.parts
    assert answered.spec.spec_id == by_mention.spec.spec_id
    assert answered.restatement.sources == [SourceMention(ref=_CENSUS)]
    assert answered.restatement.provenance is None


def test_a_mention_of_the_same_source_dedupes_with_the_deprecated_field():
    """The dedupe key reads the folded mentions, so a part posting `provenance` and one
    posting the same ref as a mention run one leg; an exclusion of it is another part."""
    plain = ParsedPart(variable_text="aid", provenance=_CENSUS)
    mention = ParsedPart(variable_text="aid", sources=[SourceMention(ref=_CENSUS)])
    excluded = ParsedPart(
        variable_text="aid", sources=[SourceMention(ref=_CENSUS, polarity="exclude")]
    )

    kept = _dedupe_parts_with_indices([plain, mention, excluded])

    assert [index for index, _part in kept] == [0, 2]


def test_restated_sources_echo_the_mentions_else_the_answered_table():
    """A part with mentions restates them, refused or not; one with none restates the table
    its answer binds, and nothing without one."""
    pinned = (SourceMention(text="crs", ref="provenance/OECDCreditorReportingSystem"),)
    scope = SourceScope(allowed=frozenset({"x"}), named=True, mentions=pinned)
    refusal = SourceRefusal(
        reason="source_not_resolved", terms=("xyz",), ambiguous=(), mentions=pinned
    )

    assert restated_sources(scope, "provenance/Other") == list(pinned)
    assert restated_sources(refusal, None) == list(pinned)
    assert restated_sources(None, "provenance/Other") == [SourceMention(ref="provenance/Other")]
    assert restated_sources(None, None) == []


# ---------------------------------------------------------------------------
# The pin rule
# ---------------------------------------------------------------------------


_SOURCE_SLOT = Slot(
    key=SlotKey(axis="source", property=None, label="data source"), binding=BindingUnbound()
)


@pytest.mark.parametrize(
    ("slots", "provenances", "facet"),
    [
        ([_SOURCE_SLOT], [_WDI], {"state": "value", "ref": _WDI}),
        ([_SOURCE_SLOT], [_WDI, _CENSUS], {"state": "set", "refs": [_WDI, _CENSUS]}),
        ([], [_CENSUS], {"state": "value", "ref": _CENSUS}),
    ],
    ids=["one-served", "several-served", "no-source-slot"],
)
def test_the_pin_binds_a_value_for_one_served_provenance_and_a_set_for_several(
    slots, provenances, facet
):
    """The identity binds `facets.provenance` to the provenances the read plan serves: one as
    a value, several as a set, so a mint of the spec_id reads every one of them and no
    other. A shape with no source slot gains one."""
    shape = {
        "kind": "statvar_schema",
        "population_type": "Person",
        "measured_property": "fertilityRate",
        "stat_type": "measuredValue",
    }

    pinned = with_served_provenance(slots, provenances)
    projection = build_identity_projection(shape, pinned, open_roles=())

    assert bound_facet_axes(pinned) == {"provenance"}
    assert projection["facets"]["provenance"] == facet


_IDS = "provenance/WorldBankInternationalDebtStatistics"
_WEO = "provenance/IMFWorldEconomicOutlook"
_GDP = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"


def _gdp_shape() -> ShapeDraft:
    """GDP at current prices, a five-tuple the code index holds no row of."""
    return _derive_one(
        {
            _GDP: {
                "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
                "name": {"nodes": [{"value": "GDP at current prices"}]},
                "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
                "measuredProperty": {"nodes": [{"dcid": "amount"}]},
                "statType": {"nodes": [{"dcid": "measuredValue"}]},
                "measurementQualifier": {"nodes": [{"dcid": "Nominal"}]},
                "constraintProperties": {"nodes": [{"dcid": "activitySource"}]},
                "activitySource": {"nodes": [{"dcid": "GrossDomesticProduction"}]},
            }
        }
    )


@pytest.mark.parametrize(
    ("shape_of", "provenance"),
    [(_ids_shape, _IDS), (_gdp_shape, _WEO)],
    ids=["ids-debt-service", "weo-gdp"],
)
def test_a_part_scoped_to_a_source_the_index_holds_no_rows_of_elects_its_pin_s_query(
    shape_of, provenance
):
    """A part naming a source whose rows the code index does not hold (IDS lists rows,
    WEO has none) binds no source in the bindings its members are elected from, as its
    pinned identity names no dataset (`code_index.members.named_dataset`), so the query
    built after election equals the identity's."""
    shape = shape_for_part(shape_of(), SourceScope(allowed=frozenset({provenance}), named=True))
    rep_sv = next(iter(shape.sv_arc_facts or {}))
    bindings = with_named_source(
        standard_bindings_from_arcs(shape=shape, rep_sv_dcid=rep_sv), shape
    )
    slots = [
        build_slot(
            key,
            draft,
            [GraphRef(ref=value, label=value) for value in draft.value_dcids]
            if draft is not None
            else [],
            property_ref=GraphRef(ref=key.property_dcid, label=key.property_dcid)
            if key.property_dcid is not None
            else None,
        )
        for key, draft in pair_slot_drafts(shape.slot_keys, bindings)
    ]
    _spec_id, identity = build_spec_identity(
        identity_shape_for(
            pop_type_dcid=shape.pop_type_dcid,
            meas_prop_dcid=shape.meas_prop_dcid,
            stat_type_dcid=shape.stat_type_dcid,
            meas_qual_dcid=shape.meas_qual_dcid,
            meas_denom_dcid=None,
        ),
        with_served_provenance(slots, [provenance]),
        open_roles=(),
    )

    assert (identity.facets or {})["provenance"] == IdentityBindingValue(ref=provenance)
    assert shape_query(shape, bindings, ()) == identity_query(identity)


# ---------------------------------------------------------------------------
# The UN gate and the source slot
# ---------------------------------------------------------------------------


def test_the_un_second_pass_runs_only_under_a_scope_allowing_a_un_provenance(bind_catalogue):
    bind_catalogue(
        FakeCatalogue(
            provenances={_WDI: _row(_WDI), _UN_ROW: _row(_UN_ROW, data_commons=UN_SYSTEM)}
        )
    )

    assert allows_un_system(None)
    assert allows_un_system(SourceScope(allowed=frozenset({_WDI, _UN_ROW}), named=False))
    assert not allows_un_system(SourceScope(allowed=frozenset({_WDI}), named=True))


# ---------------------------------------------------------------------------
# Recall's completion under a scope
# ---------------------------------------------------------------------------


def test_recall_adds_the_scoped_family_of_a_recalled_five_tuple():
    """With no scope, a hit on one family of a five-tuple stands alone where no family on
    fewer roles exists. Under a scope that allows another family's dataset, that family
    gains a row, so a part naming that dataset has a shape to answer from."""
    index = get_index()
    five_tuple, family, other = next(
        (five_tuple, families[0], candidate)
        for five_tuple, families in index.families_by_five_tuple.items()
        for candidate in families[1:]
        if set(candidate.datasets).isdisjoint(families[0].datasets)
        and len({tuple(sorted(f.properties)) for f in families}) > 1
    )
    hit = DetectedSV(
        dcid=family.rows[0].dcid, score=0.9, type_of="StatisticalVariable", sentence=None
    )
    plain = with_rival_families([hit], catalogue=FakeCatalogue(), index=index)
    scoped = with_rival_families(
        [hit], catalogue=FakeCatalogue().scoped(frozenset(other.datasets)), index=index
    )

    added = [c for c in scoped if c.rival]
    assert five_tuple == family.five_tuple
    assert any(index.family_by_dcid[c.dcid] is other for c in added)
    assert all(c.dcid != hit.dcid for c in added)
    assert len(added) > len([c for c in plain if c.rival and index.family_by_dcid[c.dcid] is other])


_CRS = "provenance/OECDCreditorReportingSystem"
_DAC2A = "provenance/OECDDAC2A"
_DAC_TABLES = frozenset({_CRS, "provenance/OECDDAC1", _DAC2A, "provenance/OECDDAC2B"})


def _flow_amount_hit(provenance: str) -> DetectedSV:
    """A hit on the `provenance` family of the nominal FinancialFlow amount five-tuple, which
    the CRS, DAC1, DAC2a and DAC2b each publish on the same roles."""
    family = next(
        family
        for families in get_index().families_by_five_tuple.values()
        for family in families
        if family.five_tuple.population_type == "FinancialFlow"
        and family.five_tuple.measured_property == "amount"
        and family.five_tuple.measurement_denominator is None
        and provenance in family.datasets
    )
    return DetectedSV(
        dcid=family.rows[0].dcid, score=0.9, type_of="StatisticalVariable", sentence=None
    )


@pytest.mark.parametrize(
    ("recalled", "allowed"),
    [
        (_CRS, lambda every: every - {"provenance/ONEImputedMultilateralSectors"}),
        (_DAC2A, lambda _every: _DAC_TABLES),
    ],
    ids=["idle-exclusion", "dac-scope-recalls-dac2a"],
)
def test_recall_under_a_scope_its_recalled_family_meets_is_the_unscoped_recall(
    recalled: str, allowed: Callable[[frozenset[str]], frozenset[str]]
):
    """A scope that allows a family recall found adds no family of that five-tuple: an
    exclusion of a table publishing nothing there, and a part naming the DAC tables whose
    recall holds DAC2a, recall what the unscoped question recalls."""
    index = get_index()
    hit = _flow_amount_hit(recalled)
    every = frozenset(
        dataset
        for families in index.families_by_five_tuple.values()
        for family in families
        for dataset in family.datasets
    )
    scope = FakeCatalogue().scoped(allowed(every))

    plain = with_rival_families([hit], catalogue=FakeCatalogue(), index=index)
    scoped = with_rival_families([hit], catalogue=scope, index=index)

    assert scoped == plain


_EMPTY = "dc/base/EmptySource"


@pytest.mark.parametrize(
    ("sources", "closing"),
    [
        ([SourceMention(ref=_EMPTY)], "The search read only EmptySource."),
        (
            [
                SourceMention(ref=_WDI, polarity="exclude"),
                SourceMention(ref=_CENSUS, polarity="exclude"),
            ],
            "No series for this query comes from a source other than "
            "WorldDevelopmentIndicators and Kenya_Census.",
        ),
    ],
    ids=["include-empty", "exclude-both"],
)
def test_a_scoped_part_the_read_leaves_empty_names_its_sources(bind_catalogue, sources, closing):
    """A part whose scope leaves the read no series answers no data naming the sources it
    names, or those it leaves out: the outcome says the source limited the answer."""
    bind_catalogue(FakeCatalogue(provenances={p: _row(p) for p in (_WDI, _CENSUS, _EMPTY)}))
    graph = _ScopedFakeGraph(obs=_load_json("graph_obs.json"))

    answered = _answer(graph, sources)

    assert answered.status == "no_data"
    assert answered.no_data.message.endswith(closing)
    assert answered.rendered_sentence.endswith(closing)


# ---------------------------------------------------------------------------
# A source refusal as a no-data part
# ---------------------------------------------------------------------------


def _part_for(region: RegionResult):
    return _build_part(
        region,
        representative=region,
        folded=False,
        max_candidates=2,
        include_observations=False,
        include_skills=False,
        skills_index=None,
        envelope_warnings=[],
    )


def test_an_unresolved_source_answers_no_data_quoting_its_words():
    refusal = SourceRefusal(
        reason="source_not_resolved",
        terms=("XYZ statistics",),
        ambiguous=(),
        mentions=(SourceMention(text="XYZ statistics"),),
    )
    region = source_refusal_result("aid", refusal, timing_by_step={})

    part = _part_for(region)

    assert part.no_data.reason == "source_not_resolved"
    assert part.no_data.source_terms == ["XYZ statistics"]
    assert part.no_data.ambiguous_sources == []
    assert part.no_data.data_query is None
    assert "XYZ statistics" in part.no_data.message


_IDS = "provenance/WorldBankInternationalDebtStatistics"
_EDUCATION = ResolvedQualifier(
    text="education",
    named=[GraphRef(ref="CRSPurposeCode110", label="education")],
    slot=None,
    disposition="unbound",
)


def _ids_scope(polarity: Literal["include", "exclude"]) -> SourceScope:
    mention = SourceMention(text="IDS", ref=_IDS, polarity=polarity)
    reference = SourceReference(ref=_IDS, label="International Debt Statistics", kind="provenance")
    return SourceScope(
        allowed=frozenset({_IDS}) if polarity == "include" else frozenset({_WDI}),
        named=polarity == "include",
        mentions=(mention,),
        references=(reference,),
        expansions=(frozenset({_IDS}),),
    )


@pytest.mark.parametrize(
    ("polarity", "reason", "terms", "message", "unbound"),
    [
        (
            "include",
            "qualifier_not_bound",
            (_EDUCATION,),
            '"education" names a category the search could not apply to this query. '
            "The search read only International Debt Statistics.",
            ["education", "International Debt Statistics"],
        ),
        (
            "include",
            "qualifier_not_bound",
            (),
            "No series for this query comes from International Debt Statistics.",
            ["International Debt Statistics"],
        ),
        (
            "exclude",
            "qualifier_not_bound",
            (),
            "No series for this query comes from a source other than "
            "International Debt Statistics.",
            ["International Debt Statistics"],
        ),
        (
            "exclude",
            "no_observations",
            (),
            'No data for "debt service". Try a different place, time period, or indicator. '
            "The search left out International Debt Statistics.",
            [],
        ),
    ],
    ids=["category-under-include", "candidacy-include", "candidacy-exclude", "read-exclude"],
)
def test_a_no_data_outcome_under_a_scope_names_its_sources(
    polarity, reason, terms, message, unbound
):
    """A no-data outcome read under a source scope names the sources that limited it, in its
    sentence and, for a qualifier it could not bind, among its unbound qualifiers, each
    source once with its references."""
    region = RegionResult(
        variable_text="debt service",
        status="no_data",
        specs=(),
        no_data_reason=reason,
        warnings=(),
        timing_by_step={},
        unbound_qualifiers=terms,
        source_scope=_ids_scope(polarity),
    )

    part = _part_for(region)

    assert part.no_data.message == message
    assert part.rendered_sentence == message
    assert [q.text for q in part.no_data.unbound_qualifiers] == unbound
    assert all(
        [ref.ref for ref in q.named] == [_IDS]
        for q in part.no_data.unbound_qualifiers
        if q.text != "education"
    )


def test_an_ambiguous_source_answers_no_data_with_its_candidates_and_the_part_to_post():
    """The part names each candidate by ref, label and kind, and `data_query` is the part
    with every resolved mention carrying its ref and the ambiguous one none."""
    candidates = (
        SourceReference(ref="dc/s/A", label="Source A", kind="source"),
        SourceReference(ref="dc/d/B", label="Dataset B", kind="dataset"),
        SourceReference(ref="provenance/C", label="Table C", kind="provenance"),
    )
    mentions = (
        SourceMention(text="ambiguous"),
        SourceMention(text="crs", ref="provenance/OECDCreditorReportingSystem", polarity="exclude"),
    )
    refusal = SourceRefusal(
        reason="source_ambiguous",
        terms=("ambiguous",),
        ambiguous=(AmbiguousSource(text="ambiguous", candidates=candidates),),
        mentions=mentions,
    )
    parsed_form = ParsedPart(variable_text="aid", sources=list(mentions))
    region = source_refusal_result("aid", refusal, timing_by_step={})
    region = dataclasses.replace(region, parsed_form=parsed_form, restatement=parsed_form)

    part = _part_for(region)

    assert part.no_data.reason == "source_ambiguous"
    assert part.no_data.source_terms == ["ambiguous"]
    (group,) = part.no_data.ambiguous_sources
    assert group.text == "ambiguous"
    assert [(c.ref, c.label, c.kind) for c in group.candidates] == [
        ("dc/s/A", "Source A", "source"),
        ("dc/d/B", "Dataset B", "dataset"),
        ("provenance/C", "Table C", "provenance"),
    ]
    assert part.no_data.data_query is not None
    assert part.no_data.data_query.parts[0].sources == list(mentions)


# ---------------------------------------------------------------------------
# The options door reads a provenance pin as a scope
# ---------------------------------------------------------------------------


def test_the_options_door_reads_under_the_target_s_provenance_pin(monkeypatch):
    """A pinned provenance scopes every read of the options door before DFO precedence, as
    a resubmit reads it, so a base pin answers at a place a DFO table also holds."""
    from search.models import IdentityBindingSet

    seen: list[frozenset[str] | None] = []

    def _capture(*_args, **_kwargs):
        seen.append(current_source_scope())
        raise _Stop

    class _Stop(Exception):
        pass

    monkeypatch.setattr(surface_module, "_observation_about_options_set", _capture)

    target = SimpleNamespace(facets={"provenance": IdentityBindingSet(refs=[_WDI, _CENSUS])})
    request = SimpleNamespace(include_coverage=False, for_=SimpleNamespace(role="observationAbout"))

    with pytest.raises(_Stop):
        surface_module.build_options_response(
            cast(OptionsRequest, request),
            cast(ResolvedOptionsTarget, target),
            graph=cast(EngineGraphClient, None),
            deadline=0.0,
            coverage=cast(CoverageSnapshot, None),
        )

    assert seen == [frozenset({_WDI, _CENSUS})]


# ---------------------------------------------------------------------------
# Served facts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provenance", "account"),
    [("provenance/OECDCreditorReportingSystem", "provider"), (_WDI, None)],
    ids=["declared-dataset", "base-provenance"],
)
def test_the_provenance_facet_states_the_account_of_a_declared_dataset(provenance, account):
    """A client reads whose account of a flow the answer gives from the served provenance:
    the code index's declaration for a dataset it declares, none for any other."""
    facets = _resolved_facets(
        [{"provenance": SeriesFacetRef(ref=provenance, label=provenance)}], {}
    )

    assert facets["provenance"].account == account


# ---------------------------------------------------------------------------
# An exclusion that removes nothing
# ---------------------------------------------------------------------------


_IMPUTED = "provenance/ONEImputedMultilateralSectors"
_GLOBAL_FUND = "dc/base/GlobalFund"


def _sourced_row(provenance: str, name: str, source: str, source_label: str) -> ProvenanceRow:
    return ProvenanceRow(
        id=provenance,
        data_commons=BASE,
        source=SourceRef(ref=source, label=source_label),
        name=name,
        url="https://example.org",
    )


_EXCLUSION_CATALOGUE = FakeCatalogue(
    provenances={
        row.id: row
        for row in (
            _sourced_row(_WDI, "World Development Indicators", "dc/s/WorldBank", "World Bank"),
            _sourced_row(_IDS, "International Debt Statistics", "dc/s/WorldBank", "World Bank"),
            _sourced_row(_GLOBAL_FUND, "Global Fund grants", "dc/s/GlobalFund", "Global Fund"),
            _sourced_row(_CRS, "Creditor Reporting System (CRS)", "dc/s/OECD", "OECD"),
            _sourced_row(_DAC2A, "DAC2A", "dc/s/OECD", "OECD"),
            _sourced_row(_IMPUTED, "Imputed multilateral sectors", "source/ONE", "ONE Data"),
        )
    }
)


def _index_shape(provenance: str) -> ShapeDraft:
    """A shape derived from three code-index rows of the dataset `provenance` publishes."""
    rows = [row for row in get_index().statvars if provenance in row.provenances][:3]
    return _derive_one({row.dcid: _crs_arcs(row) for row in rows})


# Each row: the excluded words, the provenances publishing the recalled shapes (an index
# dataset, or `None` for a shape whose publishers neither the index nor the catalogue
# knows), and whether the part warns. The first four are the probes the rule was settled
# on: "aid to Uganda, excluding the World Bank" and "health aid to Malawi not counting the
# Global Fund" recall OECD tables alone; "ODA to Kenya excluding the CRS" recalls a CRS
# shape; "education aid to Kenya excluding ONE Imputed Multilateral Sectors" recalls CRS
# shapes and a base shape of unknown publishers, and no imputed shape.
_EXCLUSIONS = [
    ("World Bank", [_CRS, _DAC2A], True),
    ("Global Fund", [_CRS], True),
    ("CRS", [_CRS, _DAC2A], False),
    ("Imputed multilateral sectors", [_CRS, None], True),
    ("Imputed multilateral sectors", [_CRS, _IMPUTED], False),
    # The World Bank may publish a base shape whose publishers are unknown.
    ("World Bank", [_CRS, None], False),
    # Words naming no admitted source stand for no provenance, so they always warn.
    ("Gavi", [_CRS, None], True),
]


@pytest.mark.parametrize(
    ("words", "publishers", "warns"),
    _EXCLUSIONS,
    ids=[f"{words}-{'warns' if warns else 'silent'}" for words, _p, warns in _EXCLUSIONS],
)
def test_an_exclusion_publishing_none_of_the_recalled_shapes_warns_and_is_never_named(
    bind_catalogue, words, publishers, warns
):
    """An excluding mention whose provenances publish none of the part's recalled shapes
    removes nothing, so the part warns `query_qualifier_dropped` quoting its words, and no
    refusal names it as a source that limited the answer. One that publishes a recalled
    shape, or may publish a shape of unknown publishers, is silent and named."""
    bind_catalogue(_EXCLUSION_CATALOGUE)
    scope = compile_sources(
        [SourceMention(text=words, polarity="exclude")],
        catalogue=_EXCLUSION_CATALOGUE,
        pointers=["/sources/0"],
    )
    assert isinstance(scope, SourceScope)
    shapes = [_gdp_shape() if p is None else _index_shape(p) for p in publishers]

    marked = with_idle_exclusions(scope, shapes)
    assert marked is not None
    warnings = idle_exclusion_warnings(marked)

    assert [w.code for w in warnings] == (["query_qualifier_dropped"] if warns else [])
    assert all(f'"{words}"' in w.message for w in warnings)
    named = _unbound_terms(
        RegionResult(
            variable_text="aid",
            status="no_data",
            specs=(),
            no_data_reason="qualifier_not_bound",
            warnings=(),
            timing_by_step={},
            source_scope=marked,
        )
    )[1]
    assert bool(named) is not warns


@pytest.mark.parametrize(
    "excluded",
    [
        SourceMention(ref=_CRS, polarity="exclude"),
        SourceMention(text="the Global Fund", polarity="exclude"),
    ],
    ids=["publishes-none", "names-no-source"],
)
def test_a_part_excluding_a_source_that_publishes_none_of_its_shapes_answers_and_warns(
    fertility_sources, excluded
):
    """Excluding the CRS, which publishes only code-index rows, or words that name no
    admitted source leaves Kenya's fertility rate as it was, so the part answers and warns
    that the exclusion removed nothing; the restatement keeps the mention as posted."""
    answered = _answer(fertility_sources, [excluded])

    assert answered.spec.resolved.facets["provenance"].ref == _WDI
    assert "query_qualifier_dropped" in [w.code for w in answered.warnings]
    assert answered.restatement.sources == [excluded]


# ---------------------------------------------------------------------------
# An exclusion chooses among the unscoped menu
# ---------------------------------------------------------------------------


_KEN = "country/KEN"
_UN_EDUCATION = "undata/sdg/SERIES-SD_XPD_ESED"
_FLOW_FACET = {"earliestDate": "2020", "latestDate": "2022", "obsCount": 3}


def _family_member(provenance: str, measured_property: str = "amount") -> str:
    """The first member of the nominal FinancialFlow family `provenance` alone publishes."""
    return next(
        family.rows[0].dcid
        for families in get_index().families_by_five_tuple.values()
        for family in families
        if family.datasets == (provenance,)
        and family.five_tuple.population_type == "FinancialFlow"
        and family.five_tuple.measured_property == measured_property
        and family.five_tuple.measurement_denominator is None
    )


def _excluding_crs(bind_catalogue: Callable[[Catalogue], None]) -> SourceScope:
    """The scope of a part excluding the CRS over the code index's datasets and one UN
    System provenance, so the UN pass may run unless the scope settles the answer."""
    un_row = _row(_UN_ROW, data_commons=UN_SYSTEM)
    bind_catalogue(FakeCatalogue(provenances={_UN_ROW: un_row}))
    mention = SourceMention(text="CRS", ref=_CRS, polarity="exclude")
    return SourceScope(
        allowed=(frozenset(get_index().datasets) | {_UN_ROW}) - {_CRS},
        named=False,
        mentions=(mention,),
        references=(SourceReference(ref=_CRS, label="CRS", kind="provenance"),),
        expansions=(frozenset({_CRS}),),
    )


def _resolve_flow(
    scope: SourceScope, recalled: dict[str, float], llm: _MenuLLM, query: str
) -> RegionResult:
    """Resolve `query` to Kenya under `scope`, recalling the code-index rows `recalled` at
    their scores and a UN education series just over the score floor, every series with
    data at Kenya."""
    rows = {row.dcid: row for row in get_index().statvars}
    nodes: dict = {_KEN: {"label": "Kenya", "type": "Country"}}
    nodes |= {
        sv: {"label": sv, "type": "StatisticalVariable", "arcs": _crs_arcs(rows[sv])}
        for sv in recalled
    }
    nodes[_UN_EDUCATION] = {
        "label": _UN_EDUCATION,
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": _UN_EDUCATION}]},
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": "expenditureShare"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
            "constraintProperties": {"nodes": []},
        },
    }
    graph = FakeGraph(
        nodes=nodes,
        obs={f"{sv}|{_KEN}": [_FLOW_FACET] for sv in (*recalled, _UN_EDUCATION)},
        detect={query: {"svs": list(recalled), "cosine_scores": list(recalled.values())}},
        un_detect={query: {"svs": [_UN_EDUCATION], "cosine_scores": [0.6502]}},
        resolve={"Kenya": _KEN},
    )
    return asyncio.run(
        resolve_variable(
            query,
            entities=["Kenya"],
            date_request=None,
            detect_query=query,
            role_query=f"{query} to Kenya",
            entity_roles={"Kenya": [DoorRole(axis="destination", span="Kenya")]},
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=llm,
            base_timing={},
            source_scope=scope,
            coverage=empty_coverage_snapshot(),
        )
    )


class _MenuLLM:
    """A bind model that elects the first shape of its menu and records each menu."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        self.prompts.append(prompt)
        return schema(chosen_shape_index=0, bindings=[], also_plausible=None, ask=None), None


def test_an_exclusion_of_every_publisher_the_question_reaches_refuses_naming_it(
    bind_catalogue,
):
    """ "education aid to Kenya excluding the CRS" reaches only CRS series in our catalogue,
    so the part refuses `qualifier_not_bound` naming the CRS before bind. The UN pass, which
    fills gaps in our catalogue, does not run, so a UN series just over the floor never
    answers."""
    llm = _MenuLLM()

    result = _resolve_flow(
        _excluding_crs(bind_catalogue),
        {_family_member(_CRS): 0.66},
        llm,
        "education aid",
    )

    assert result.status == "no_data"
    assert result.no_data_reason == "qualifier_not_bound"
    assert [ref.ref for record in result.unbound_sources for ref in record.named] == [_CRS]
    assert llm.prompts == []


_NOMINAL_GDP = "NominalGdpScoped"
_REAL_GDP = "RealGdpScoped"


@pytest.mark.parametrize(
    ("nominal_publishers", "excluded", "served"),
    [
        ((_WEO,), None, [(_NOMINAL_GDP, _WEO)]),
        ((_WEO,), _WEO, []),
        ((_WEO, _CENSUS), _WEO, [(_NOMINAL_GDP, _CENSUS)]),
    ],
    ids=["plain", "sole-publisher-excluded", "another-publisher-serves"],
)
def test_an_exclusion_ranks_on_the_evidence_the_unscoped_question_reaches(
    bind_catalogue, caplog, nominal_publishers, excluded, served
):
    """ "GDP of Kenya" answers nominal GDP, which covers real GDP at Kenya, so the default
    reading removes real GDP before bind. Excluding nominal GDP's only publisher refuses
    `qualifier_not_bound` naming it, without the UN pass, and never serves the real GDP the
    unscoped ranking discarded. Excluding one of two publishers serves nominal GDP from the
    other."""
    rows = {p: _row(p) for p in (_WEO, _WDI, _CENSUS)}
    catalogue = FakeCatalogue(provenances={**rows, _UN_ROW: _row(_UN_ROW, data_commons=UN_SYSTEM)})
    bind_catalogue(catalogue)
    scope = (
        None
        if excluded is None
        else compile_sources(
            [SourceMention(ref=excluded, polarity="exclude")],
            catalogue=catalogue,
            pointers=["/sources/0"],
        )
    )
    assert scope is None or isinstance(scope, SourceScope)
    graph = _gdp_graph(nominal_publishers, scores=(0.9, 0.85))

    with caplog.at_level(logging.INFO, logger="search.engine.region.pipeline"):
        result = _resolve_gdp(graph, scope)

    assert [
        (member.ref.ref, spec.resolved.facets["provenance"].ref)
        for spec in result.specs
        for member in spec.resolved.members
    ] == served
    if not served:
        assert result.status == "no_data"
        assert result.no_data_reason == "qualifier_not_bound"
        assert [ref.ref for record in result.unbound_sources for ref in record.named] == [_WEO]
        assert result.source_scope == scope
        (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("gap_fill ")]
        assert "pass2=skipped:source_scope" in line


def test_an_exclusion_whose_prunes_leave_no_reading_refuses_the_variable(bind_catalogue):
    """ "GDP of Kenya" excluding the CRS recalls nominal and real GDP under the score floor.
    The prunes run unscoped and leave no reading, so the exclusion removed none and the
    part refuses `variable_not_resolved`. The scope allows only code-index datasets, so
    both shapes lie outside it, and the refusal still does not blame it."""
    bind_catalogue(FakeCatalogue(provenances={_WEO: _row(_WEO), _WDI: _row(_WDI)}))
    mention = SourceMention(text="CRS", ref=_CRS, polarity="exclude")
    scope = SourceScope(
        allowed=frozenset(get_index().datasets) - {_CRS},
        named=False,
        mentions=(mention,),
        references=(SourceReference(ref=_CRS, label="CRS", kind="provenance"),),
        expansions=(frozenset({_CRS}),),
    )

    result = _resolve_gdp(_gdp_graph((_WEO,), scores=(0.6, 0.55)), scope)

    assert result.status == "no_data"
    assert result.no_data_reason == "variable_not_resolved"
    assert not result.unbound_sources


def _gdp_graph(
    nominal_publishers: tuple[str, ...], *, scores: tuple[float, float]
) -> _ScopedFakeGraph:
    """Nominal GDP from `nominal_publishers` and real GDP from the WDI at Kenya, recalled
    for "GDP of Kenya" at `scores`."""
    publishers = {_NOMINAL_GDP: nominal_publishers, _REAL_GDP: (_WDI,)}
    return _ScopedFakeGraph(
        nodes={
            _KEN: {"label": "Kenya", "type": "Country"},
            _NOMINAL_GDP: _node(
                "Nominal gross domestic product",
                "Nominal",
                {"activitySource": "GrossDomesticProduction"},
            ),
            _REAL_GDP: _node(
                "Real gross domestic product",
                "RealValue",
                {"activitySource": "GrossDomesticProduction"},
            ),
        },
        detect={"GDP of Kenya": {"svs": [_NOMINAL_GDP, _REAL_GDP], "cosine_scores": list(scores)}},
        resolve={"Kenya": _KEN},
        obs={
            f"{sv}|{_KEN}": [{**facet, "provenanceId": p} for p in provenances for facet in _FACETS]
            for sv, provenances in publishers.items()
        },
    )


def _resolve_gdp(graph: _ScopedFakeGraph, scope: SourceScope | None) -> RegionResult:
    return asyncio.run(
        resolve_variable(
            "GDP of Kenya",
            entities=["Kenya"],
            date_request=None,
            detect_query="GDP of Kenya",
            role_query="GDP of Kenya",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_SoleOptionLLM(None, _GDP_SOLE_OPTION),
            base_timing={},
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            source_scope=scope,
            coverage=empty_coverage_snapshot(),
        )
    )


class _AskLLM(_MenuLLM):
    """A bind model that declines its menu."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        self.prompts.append(prompt)
        return schema(
            chosen_shape_index=0, bindings=[], also_plausible=None, ask="Which flow?"
        ), None


@pytest.mark.parametrize(
    ("polarity", "source", "allowed"),
    [
        ("include", _DAC2A, frozenset({_DAC2A})),
        ("exclude", _CRS, frozenset({_DAC2A, _WDI})),
    ],
    ids=["naming-dac2a", "excluding-crs"],
)
def test_a_bind_declining_a_part_whose_scope_removed_shapes_refuses_naming_its_sources(
    polarity: Literal["include", "exclude"], source: str, allowed: frozenset[str]
):
    """Recall reaches CRS and DAC2a flows. Naming DAC2a removes the CRS shape at candidacy;
    excluding the CRS removes it from the ranked menu. Bind declines the DAC2a shape left.
    Readings exist outside the scope and none inside is accepted, so the part refuses
    `qualifier_not_bound` naming the source it mentions."""
    mention = SourceMention(text=source, ref=source, polarity=polarity)
    scope = SourceScope(
        allowed=allowed,
        named=polarity == "include",
        mentions=(mention,),
        references=(SourceReference(ref=source, label=source, kind="provenance"),),
        expansions=(frozenset({source}),),
    )
    llm = _AskLLM()

    result = _resolve_flow(
        scope, {_family_member(_CRS): 0.66, _family_member(_DAC2A): 0.61}, llm, "aid"
    )

    assert len(llm.prompts) == 1
    assert result.no_data_reason == "qualifier_not_bound"
    assert [ref.ref for record in result.unbound_sources for ref in record.named] == [source]
