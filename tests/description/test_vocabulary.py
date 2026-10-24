"""The vocabulary snapshot, its builder, its packaged publisher names and its `voc_` revision."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import MappingProxyType

import pytest

from code_index import CodeIndex, CodeRow, CodeSystemDeclaration, IndexStamp
from code_index.labels import HOW_CONNECTIVES, ROLE_CLASS_PROPERTIES
from code_index.types import MeasureAxis, MeasureIndex, MeasurePartitionLabel, MeasureValueRow
from description import vocabulary
from description.vocabulary import (
    MeasureWords,
    SourceDocumentFacts,
    Vocabulary,
    build_vocabulary,
    packaged_words_digest,
    publisher_names,
    vocabulary_revision,
)
from dfo_platform.roles import GENERIC_ROLE_WORDING, ROLE_WORDING
from origin_catalogue.registry import load_registry
from tests.code_index.support import INDEX_FIXTURE, load_index, load_measure_axes
from unit_words.currency_nouns import CURRENCY_NOUNS

_CRS = "provenance/OECDCreditorReportingSystem"
_OECD = "dc/s/OrganisationForEconomicCo-operationAndDevelopmentOecd"

_HEALTH = "Assistance to health."
_GROSS = "An amount counted in full, with nothing offset."
_GROSS_DISBURSEMENTS = "The total amount released before any repayments are subtracted."
_DEBT_SERVICE = "Principal and interest actually paid on external debt."

_INDEX = CodeIndex(
    stamp=IndexStamp(
        graph_endpoint="https://graph.test",
        exported_at="2026-10-11T00:00:00Z",
        code_counts={"crsPurpose": 1},
        statvar_shape_counts={},
        content_hash="index-hash",
    ),
    declarations=[
        CodeSystemDeclaration(
            code_system="crsPurpose",
            hierarchical=True,
            disjoint_from_children=True,
            summable=True,
            absent_reading="aggregate",
            plain_name="sector",
        )
    ],
    codes=[
        CodeRow(
            dcid="CRSPurposeCode120",
            code_system="crsPurpose",
            code="120",
            label="health",
            synonyms=(),
            parent=None,
            definition=_HEALTH,
        ),
        CodeRow(
            dcid="CRSPurposeCode130",
            code_system="crsPurpose",
            code="130",
            label="population policies",
            synonyms=(),
            parent=None,
        ),
    ],
    statvars=(),
)

# Keyed as the fixture measure index keys its partition labels and nouns: on the shape's
# members beside the constraint axes.
_MEASURES = MeasureIndex(
    axes=(
        MeasureAxis(axis="populationType", kind="shape", role="free", prefer=()),
        MeasureAxis(axis="measuredProperty", kind="shape", role="basis", prefer=()),
        MeasureAxis(axis="financialFlowType", kind="constraint", role="free", prefer=()),
        MeasureAxis(
            axis="nettingBasis", kind="constraint", role="free", prefer=(), label="netting basis"
        ),
    ),
    values=(
        MeasureValueRow(
            dcid="Gross",
            axis="nettingBasis",
            label="gross",
            synonyms=(),
            short_name=None,
            matchable=True,
            definition=_GROSS,
        ),
    ),
    partition_labels=(
        MeasurePartitionLabel(
            axis_values=(
                ("financialFlowType", "Disbursement"),
                ("measuredProperty", "amount"),
                ("nettingBasis", "Gross"),
                ("populationType", "FinancialFlow"),
            ),
            label="gross disbursements",
            named_values=(("dac2aAidType", "DAC2aAidType106"),),
            definition=_GROSS_DISBURSEMENTS,
        ),
    ),
    nouns=(
        MeasurePartitionLabel(
            axis_values=(
                ("populationType", "FinancialFlow"),
                ("measuredProperty", "amount"),
                ("statType", "measuredValue"),
                ("financialFlowType", "ServicingPayment"),
            ),
            label="external debt service",
            definition=_DEBT_SERVICE,
        ),
    ),
)
_FLOW_SHAPE = {
    "populationType": "FinancialFlow",
    "measuredProperty": "amount",
    "statType": "measuredValue",
}


def _vocabulary(**overrides: object) -> Vocabulary:
    inputs: dict[str, object] = {
        "index": _INDEX,
        "measure_index": _MEASURES,
        "labels": {"country/NGA": "Nigeria", "CRSPurposeCode120": "Health (CRS Purpose)"},
        "sources": {
            _CRS: SourceDocumentFacts(
                publisher=_OECD, last_downloaded="2026-10-08", next_release="2026-09"
            )
        },
        "currency_units": {"CurrencyUSD": "USD", "CurrencyNGN": "NGN"},
        "graph_generation": "rev-1",
        "catalogue_build": "build-1",
    }
    return build_vocabulary(**{**inputs, **overrides})


def test_labels_take_the_index_label_for_a_code_and_the_callers_for_the_rest():
    words = _vocabulary()

    assert words.label("CRSPurposeCode120") == "health"
    assert words.label("Gross") == "gross"
    assert words.label("country/NGA") == "Nigeria"
    assert words.label("country/KEN") is None
    assert dict(words.labels) == {
        "country/NGA": "Nigeria",
        "CRSPurposeCode120": "Health (CRS Purpose)",
    }, "the vocabulary copies the caller's labels only, never the index's"
    assert dict(words.definitions) == {}


def test_blank_words_are_absent():
    words = _vocabulary(
        labels={"country/NGA": "", "country/KEN": "  "},
        definitions={"country/NGA": ""},
        sources={_CRS: SourceDocumentFacts(publisher="", last_downloaded=" ", next_release="")},
    )

    assert words.label("country/NGA") is None
    assert words.label("country/KEN") is None
    assert dict(words.definitions) == {}
    assert words.sources[_CRS] == SourceDocumentFacts()
    assert words.publisher(_CRS) is None


@pytest.mark.parametrize(
    ("dcid", "definition"),
    [
        pytest.param("CRSPurposeCode120", _HEALTH, id="code-index-before-the-callers"),
        pytest.param("Gross", _GROSS, id="measure-value-before-the-callers"),
        pytest.param("CRSPurposeCode130", "Population policies.", id="code-without-one"),
        pytest.param("country/NGA", "A country in West Africa.", id="callers"),
        pytest.param("country/KEN", None, id="unknown"),
    ],
)
def test_definition_reads_the_index_then_the_callers(dcid: str, definition: str | None):
    words = _vocabulary(
        definitions={
            "CRSPurposeCode120": "The caller's health.",
            "CRSPurposeCode130": "Population policies.",
            "Gross": "The caller's gross.",
            "country/NGA": "A country in West Africa.",
        }
    )

    assert words.definition(dcid) == definition


def test_a_definition_is_clipped_to_4096_characters_and_a_label_to_512():
    long = "x" * 5000
    words = _vocabulary(labels={"country/NGA": long}, definitions={"country/NGA": long})

    label = words.label("country/NGA")
    definition = words.definition("country/NGA")
    assert label is not None
    assert definition is not None
    assert (len(label), len(definition)) == (512, 4096)
    assert definition.endswith("…")


@pytest.mark.parametrize(
    ("prop", "name"),
    [
        ("crsPurpose", "sector"),
        ("nettingBasis", "netting basis"),
        ("financialFlowType", None),
        ("crsTypeOfFinance", None),
    ],
)
def test_axis_names_come_from_code_systems_and_labelled_measure_axes(prop: str, name: str | None):
    assert _vocabulary().axis_name(prop) == name


@pytest.mark.parametrize(
    ("shape", "constraints", "measure"),
    [
        pytest.param(
            _FLOW_SHAPE,
            {"financialFlowType": "Disbursement", "nettingBasis": "Gross", "x": "y"},
            MeasureWords(
                "gross disbursements",
                frozenset({"financialFlowType", "nettingBasis"}),
                _GROSS_DISBURSEMENTS,
            ),
            id="partition-label-over-shape-and-constraints",
        ),
        pytest.param(
            _FLOW_SHAPE,
            {
                "financialFlowType": "Disbursement",
                "nettingBasis": "Gross",
                "dac2aAidType": "DAC2aAidType106",
            },
            MeasureWords(
                "gross disbursements",
                frozenset({"financialFlowType", "nettingBasis", "dac2aAidType"}),
                _GROSS_DISBURSEMENTS,
            ),
            id="a-code-the-label-names-is-stated",
        ),
        pytest.param(
            {},
            {"financialFlowType": "Disbursement", "nettingBasis": "Gross"},
            None,
            id="constraints-without-the-shape-name-no-partition",
        ),
        pytest.param(
            _FLOW_SHAPE,
            {"financialFlowType": "ServicingPayment", "debtor": "Government"},
            MeasureWords("external debt service", frozenset({"financialFlowType"}), _DEBT_SERVICE),
            id="measure-noun-without-a-partition-label",
        ),
        pytest.param(
            _FLOW_SHAPE, {"financialFlowType": "Commitment"}, None, id="no-label-covers-it"
        ),
    ],
)
def test_measure_is_the_partition_label_else_the_measure_noun(
    shape: dict[str, str], constraints: dict[str, str], measure: MeasureWords | None
):
    assert _vocabulary().measure(shape=shape, constraints=constraints) == measure


@pytest.mark.parametrize(
    ("unit", "noun"),
    [
        ("CurrencyUSD", "US dollars"),
        ("CurrencyNGN", "Nigerian naira"),
        ("ChainedUSDollarBaseYr2017", "US dollars"),
        ("Percent", None),
    ],
)
def test_unit_noun_names_a_currency_unit(unit: str, noun: str | None):
    assert _vocabulary().unit_noun(unit) == noun


def test_roles_fall_back_to_provider_and_recipient():
    words = _vocabulary()

    assert words.roles(None).origin.singular == "provider"
    assert words.roles("debt_service").origin.singular == "creditor"


def test_publisher_reads_the_short_name_of_the_sources_publisher():
    words = _vocabulary(
        sources={
            _CRS: SourceDocumentFacts(publisher=_OECD),
            "provenance/ONEImputedMultilateralSectors": SourceDocumentFacts(
                publisher="source/ONEData"
            ),
        }
    )

    assert words.publisher(_CRS) == "OECD"
    assert words.publisher("provenance/ONEImputedMultilateralSectors") is None
    assert words.publisher("provenance/Unread") is None


def test_the_vocabulary_is_a_copy_that_refuses_writes():
    labels = {"country/NGA": "Nigeria"}
    words = _vocabulary(labels=labels)
    labels["country/NGA"] = "changed"

    assert words.label("country/NGA") == "Nigeria"
    with pytest.raises(TypeError):
        words.labels["country/NGA"] = "changed"


def test_the_revision_is_the_same_for_the_same_inputs():
    revision = _vocabulary().revision

    assert revision == _vocabulary().revision
    assert revision.startswith("voc_")
    assert len(revision) == len("voc_") + 32


_REVISION_INPUTS = {
    "index_stamp": "index-hash",
    "graph_generation": "rev-1",
    "catalogue_build": "build-1",
}


@pytest.fixture
def fresh_words_digest() -> Iterator[None]:
    """Clears the cached table digest around a test that patches a word table."""
    packaged_words_digest.cache_clear()
    yield
    packaged_words_digest.cache_clear()


def _perturb(perturbed: str, inputs: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    match perturbed:
        case "composer_version":
            monkeypatch.setattr(
                vocabulary, "COMPOSER_VERSION", str(int(vocabulary.COMPOSER_VERSION) + 1)
            )
        case "publishers":
            changed = MappingProxyType({**publisher_names(), "dc/s/WorldBank": "WB"})
            monkeypatch.setattr(vocabulary, "publisher_names", lambda: changed)
        case "currency_nouns":
            changed_nouns = MappingProxyType({**CURRENCY_NOUNS, "USD": "American dollars"})
            monkeypatch.setattr(vocabulary, "CURRENCY_NOUNS", changed_nouns)
        case "role_wording":
            changed_roles = MappingProxyType({**ROLE_WORDING, "debt_service": GENERIC_ROLE_WORDING})
            monkeypatch.setattr(vocabulary, "ROLE_WORDING", changed_roles)
        case "how_connectives":
            changed_connectives = MappingProxyType({**HOW_CONNECTIVES, "crsPurpose": " towards "})
            monkeypatch.setattr(vocabulary, "HOW_CONNECTIVES", changed_connectives)
        case "role_class_properties":
            changed_classes = MappingProxyType({**ROLE_CLASS_PROPERTIES, "crsChannel": "origin"})
            monkeypatch.setattr(vocabulary, "ROLE_CLASS_PROPERTIES", changed_classes)
        case _:
            inputs[perturbed] = "other"
    packaged_words_digest.cache_clear()


@pytest.mark.parametrize(
    "perturbed",
    [
        "index_stamp",
        "graph_generation",
        "catalogue_build",
        "composer_version",
        "publishers",
        "currency_nouns",
        "role_wording",
        "how_connectives",
        "role_class_properties",
    ],
)
@pytest.mark.usefixtures("fresh_words_digest")
def test_each_input_moves_the_revision(perturbed: str, monkeypatch: pytest.MonkeyPatch):
    before = vocabulary_revision(**_REVISION_INPUTS)
    inputs = dict(_REVISION_INPUTS)
    _perturb(perturbed, inputs, monkeypatch)

    assert vocabulary_revision(**inputs) != before


def test_an_unknown_generation_differs_from_every_known_one():
    known = vocabulary_revision(**_REVISION_INPUTS)
    unknown = vocabulary_revision(**{**_REVISION_INPUTS, "graph_generation": None})

    assert unknown != known


def test_every_publisher_names_a_source_the_registry_admits():
    registry = load_registry()
    admitted = {entry.source for entry in registry.base} | {
        agency.provenance for agency in registry.un_system
    }
    names = publisher_names()

    assert names, "the packaged publisher names are empty"
    assert set(names) - admitted == set()
    assert all(name.strip() for name in names.values())


def test_the_packaged_publisher_file_has_no_blank_or_duplicate_key():
    path = Path(vocabulary.__file__).parent / "data" / "publishers.json"
    pairs = json.loads(path.read_text("utf-8"), object_pairs_hook=list)

    keys = [key for key, _ in pairs]
    assert len(keys) == len(set(keys))
    assert all(isinstance(name, str) and name.strip() for _, name in pairs)


def test_the_shipped_index_gives_the_worked_examples_words():
    source = INDEX_FIXTURE
    words = build_vocabulary(
        index=load_index(source),
        measure_index=load_measure_axes(source),
        labels={},
        sources={_CRS: SourceDocumentFacts(publisher=_OECD)},
        currency_units={"CurrencyUSD": "USD"},
        graph_generation=None,
        catalogue_build=None,
    )

    assert words.axis_name("financialFlowType") == "flow type"
    assert words.axis_name("nettingBasis") == "netting basis"
    assert words.axis_name("crsPurpose") == "sector"
    assert words.label("Disbursement") == "disbursement"
    assert words.label("Gross") == "gross"
    assert words.unit_noun("CurrencyUSD") == "US dollars"
    assert words.publisher(_CRS) == "OECD"
    measure = words.measure(
        shape={**_FLOW_SHAPE, "measurementQualifier": "Nominal"},
        constraints={"financialFlowType": "Disbursement", "nettingBasis": "Gross"},
    )
    assert measure is not None
    assert measure.label == "gross disbursements"
    assert measure.definition is not None
    assert words.definition("Disbursement") is not None


def test_importing_the_vocabulary_loads_no_graph_http_or_search_module():
    probe = (
        "import sys, description.vocabulary\n"
        "roots = {'materialise', 'dc_graph', 'httpx', 'search'}\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in roots)\n"
        "assert not bad, bad\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
