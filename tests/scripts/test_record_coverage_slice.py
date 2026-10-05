"""Pins which StatVars the recorded coverage slice keeps: the whole-token scan of fixtures and corpus."""

from __future__ import annotations

import scripts.record_coverage_slice as m

_IN_CASE = "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_Nominal"
_IN_FAMILY = "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow21_Nominal"
_IN_FIXTURE = "Amount_FinancialFlow_Gross_Commitment_Nominal"
_UNREFERENCED = "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow99_Nominal"


def test_referenced_variables_reads_fixtures_cases_and_families(monkeypatch, tmp_path):
    (tmp_path / "tests" / "fixtures").mkdir(parents=True)
    (tmp_path / "corpus" / "families").mkdir(parents=True)
    (tmp_path / "tests" / "fixtures" / "graph_nodes.json").write_text(f'{{"{_IN_FIXTURE}": 1}}')
    (tmp_path / "corpus" / "crs.yaml").write_text(f"- stat_vars: [{_IN_CASE}]\n")
    (tmp_path / "corpus" / "families" / "crs.yaml").write_text(f"stat_vars: [{_IN_FAMILY}]\n")
    monkeypatch.setattr(m, "_REPO_ROOT", tmp_path)

    found = m._referenced_variables({_IN_CASE, _IN_FAMILY, _IN_FIXTURE, _UNREFERENCED})

    assert found == {_IN_CASE, _IN_FAMILY, _IN_FIXTURE}


def test_referenced_variables_matches_whole_tokens_only(monkeypatch, tmp_path):
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "crs.yaml").write_text(f"- stat_vars: [{_IN_CASE}_Extra]\n")
    monkeypatch.setattr(m, "_REPO_ROOT", tmp_path)

    assert m._referenced_variables({_IN_CASE}) == set()
