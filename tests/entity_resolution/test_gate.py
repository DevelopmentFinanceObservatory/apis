"""The listed-name gate (`gate.filter_listed_candidates`) folds hyphens in an organisation's
names, on the query and the stored name alike, and compares its codes as stored."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from resolvekit import EntityRecord
from resolvekit.core.model.entity import CodeRecord, NameRecord
from resolvekit.core.model.result import CandidateSummary

from entity_resolution.gate import filter_listed_candidates

_IDA = "dfo_graph/InternationalDevelopmentAssociation"

_RECORD = EntityRecord(
    entity_id=_IDA,
    entity_type="org.organisation",
    canonical_name="International Development Association",
    canonical_name_norm="international development association",
    names=[
        NameRecord(
            value="International Development Association",
            value_norm="international development association",
            kind="canonical",
        ),
        NameRecord(value="World Bank-IDA", value_norm="world bank-ida", kind="alias"),
    ],
    codes=[CodeRecord(system="dac", value="IDA-905", value_norm="ida-905")],
)

_CANDIDATE = CandidateSummary(
    entity_id=_IDA,
    canonical_name="International Development Association",
    entity_type="org.organisation",
    pack_id="org",
)


@pytest.mark.parametrize(
    ("text", "matched_as"),
    [
        pytest.param("World Bank-IDA", "alias", id="alias-as-stored"),
        pytest.param("World Bank IDA", "alias", id="alias-hyphen-as-space"),
        pytest.param("World Bank - IDA", "alias", id="alias-spaced-hyphen"),
        pytest.param("World Bank\u2013IDA", "alias", id="alias-en-dash"),
        pytest.param("World Bank\u2212IDA", "alias", id="alias-minus-sign"),
        pytest.param("World Bank\u00adIDA", "alias", id="alias-soft-hyphen"),
        pytest.param("World Bank\u2e3aIDA", "alias", id="alias-two-em-dash"),
        pytest.param("IDA-905", "code", id="hyphenated-code-as-stored"),
        pytest.param("IDA 905", None, id="code-not-folded"),
    ],
)
def test_an_organisation_name_matches_across_hyphens_and_a_code_as_stored(text, matched_as):
    resolver = SimpleNamespace(entity=lambda entity_id: _RECORD, pack_normalizers={})

    survivors = filter_listed_candidates(resolver, text, [_CANDIDATE])

    assert [kind for _candidate, kind in survivors] == ([matched_as] if matched_as else [])
