"""`CodeIndex.filter` distinguishes bound and open properties from properties absent on a row."""

from __future__ import annotations

from code_index import Bound, BoundSet, CodeIndex, Open

_EDU_ODA_BASE = {
    "crsPurpose": Bound("CRSPurposeCode11110"),
    "crsTypeOfFinance": Bound("CRSTypeOfFinance110"),
    "crsTypeOfFlow": Bound("CRSTypeOfFlow10"),
    "financialFlowType": Bound("Disbursement"),
    "nettingBasis": Bound("Gross"),
}


def test_open_modality_keeps_both_the_undeclared_row_and_the_modality_coded_rows(
    code_index: CodeIndex,
):
    # An omitted key is equivalent to an explicit Open().
    rows = code_index.filter(_EDU_ODA_BASE)
    dcids = {row.dcid for row in rows}
    assert dcids == {
        "FixtureStatVar_Amount_FinancialFlow_Disbursement_Gross_Nominal_EduODA_NoModality",
        "FixtureStatVar_Amount_FinancialFlow_Disbursement_Gross_Nominal_EduODA_GBS",
        "FixtureStatVar_Amount_FinancialFlow_Disbursement_Gross_Nominal_EduODA_CoreNGO",
    }


def test_explicit_open_binding_behaves_the_same_as_an_absent_key(code_index: CodeIndex):
    query = dict(_EDU_ODA_BASE, crsCooperationModality=Open())
    implicit = code_index.filter(_EDU_ODA_BASE)
    explicit = code_index.filter(query)
    assert {row.dcid for row in implicit} == {row.dcid for row in explicit}


def test_bound_modality_excludes_the_row_that_declares_no_modality(code_index: CodeIndex):
    query = dict(_EDU_ODA_BASE, crsCooperationModality=Bound("CRSCooperationModalityA01"))
    rows = code_index.filter(query)
    assert {row.dcid for row in rows} == {
        "FixtureStatVar_Amount_FinancialFlow_Disbursement_Gross_Nominal_EduODA_GBS"
    }


def test_bound_set_modality_matches_either_named_value_but_not_the_undeclared_row(
    code_index: CodeIndex,
):
    query = dict(
        _EDU_ODA_BASE,
        crsCooperationModality=BoundSet(
            frozenset({"CRSCooperationModalityA01", "CRSCooperationModalityB01"})
        ),
    )
    rows = code_index.filter(query)
    assert {row.dcid for row in rows} == {
        "FixtureStatVar_Amount_FinancialFlow_Disbursement_Gross_Nominal_EduODA_GBS",
        "FixtureStatVar_Amount_FinancialFlow_Disbursement_Gross_Nominal_EduODA_CoreNGO",
    }


def test_bound_modality_to_an_absent_value_matches_nothing(code_index: CodeIndex):
    query = dict(_EDU_ODA_BASE, crsCooperationModality=Bound("CRSCooperationModalityZ99"))
    assert code_index.filter(query) == ()


# --- The absent-value reading: financialObligationType with no nettingBasis ---


def test_open_netting_basis_keeps_the_commitment_rows_that_declare_no_netting_basis(
    code_index: CodeIndex,
):
    query = {
        "crsPurpose": Bound("CRSPurposeCode120"),
        "crsTypeOfFinance": Bound("CRSTypeOfFinance110"),
        "financialObligationType": Bound("Commitment"),
    }
    rows = code_index.filter(query)
    assert len(rows) == 2
    for row in rows:
        assert "nettingBasis" not in row.constraint_properties


def test_bound_netting_basis_excludes_commitment_rows_that_do_not_declare_it(
    code_index: CodeIndex,
):
    # A bound nettingBasis requires a value, which these commitment rows lack.
    query = {
        "crsPurpose": Bound("CRSPurposeCode120"),
        "crsTypeOfFinance": Bound("CRSTypeOfFinance110"),
        "financialObligationType": Bound("Commitment"),
        "nettingBasis": Bound("Gross"),
    }
    assert code_index.filter(query) == ()


def test_disbursement_rows_are_unaffected_by_a_financial_obligation_type_filter(
    code_index: CodeIndex,
):
    query = {
        "crsPurpose": Bound("CRSPurposeCode120"),
        "crsTypeOfFinance": Bound("CRSTypeOfFinance110"),
        "financialFlowType": Bound("Disbursement"),
    }
    rows = code_index.filter(query)
    assert len(rows) == 4
    for row in rows:
        assert "financialObligationType" not in row.constraint_properties
