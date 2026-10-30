"""Every refusal the Datasets surface classifies renders the problem pinned for it in
`refusal_wire_golden.json`, each case's `ProblemSpec.to_canonical_data()`, apart from the
deliberate diffs `_DELIBERATE_DIFFS` lists. The cases are the
exemplars each door declares, plus raise-site shapes those miss: a pointer or alternatives on a
facet refusal, a pointer on a conversion refusal, a mixed-grain frame, permanent signing
failures, and the definition route's pointer-free form of a materialisation refusal.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from dfo_frame_grammar import FrameError
from dfo_spec_kit.identity import IdentityError

from artifact_store import (
    ArtifactInventoryError,
    ArtifactRecordConflictError,
    ArtifactSigningError,
    ArtifactWriteError,
    Tier,
)
from dataset_store.snapshot import ArtifactNotFound, SnapshotNotFound, SnapshotRevoked
from datasets_api.app import datasets_surface
from datasets_api.backpressure import Backpressure
from datasets_api.codes_materialise import (
    CapabilityUnavailable,
    CurrencyRoleUndeclared,
    DefinitionNotReproducible,
    ExcludedByRegistry,
    InvalidRoleRef,
    MalformedSidecarIdentity,
    MemberNotResolvable,
    RoleIncompatible,
    RoleSignatureIncompatible,
    SpecNotExportable,
    UnSystemUnavailable,
)
from datasets_api.codes_materialise import (
    SpecNotFound as MaterialiseSpecNotFound,
)
from datasets_api.codes_snapshot import (
    ArtifactExpired,
    ObservationsUnavailable,
    SnapshotSigningFailed,
)
from datasets_api.codes_snapshot import (
    SpecNotFound as SnapshotSpecNotFound,
)
from datasets_api.doors.definition import (
    DefinitionBackpressure,
    DefinitionMemberNotResolvable,
    StoredTraceMissingResolvedStatVars,
)
from datasets_api.doors.definition import (
    SpecNotFound as DefinitionSpecNotFound,
)
from datasets_api.doors.meta import OperationFailed, OperationNotFound
from datasets_api.doors.mint_errors import MintBackpressure
from datasets_api.doors.observations import (
    ObservationsBackpressure,
    ObservationsGraphUnavailable,
    StoredFrameCorrupt,
)
from datasets_api.doors.pin import PinBackpressure, PinBudgetExhausted
from datasets_api.doors.publish import (
    PublishBackpressure,
    PublishBodyEncodingInvalid,
    PublishBodyTooLarge,
    PublishBytesDiverge,
    PublishContentLengthRequired,
    PublishFrameInvalid,
    PublishLineageDiverge,
    PublishRecipeStoreMissing,
    PublishRecordMissing,
)
from datasets_api.doors.recipe import RecipeBackpressure, RecipeNotFound, ReplayNotFound
from datasets_api.doors.refresh import (
    FrameNotRefreshable,
    RefreshBackpressure,
    RefreshRebuildFailed,
    RenderingNotRecorded,
)
from datasets_api.doors.register import RegistryBackpressure
from datasets_api.doors.snapshot import SnapshotBackpressure
from datasets_api.principals import PinNotPermitted, PublishNotPermitted, RefreshNotPermitted
from datasets_api.query_options import QueryOptionsInvalid
from datasets_api.recipe_store import RecipeDiverges, RecipeFrameNotStored, ReplayKeyHeld
from datasets_api.store_errors import DatasetsStoreUnavailable
from dc_sdmx import (
    SdmxCapExceededError,
    SdmxContainmentFilterError,
    SdmxHttpError,
    SdmxIncompatibleVariablesError,
    SdmxMissingFilterError,
    SdmxQueryError,
    SdmxResponseError,
    SdmxRoleCeilingError,
    SdmxTransportError,
    SdmxUnsupportedFilterError,
)
from dfo_platform.problem import Alternative, ProblemIssue, Refusal
from dfo_platform.refusal import classify_default
from materialise.errors import (
    CollectionProbeCeilingExceededError,
    DatasetDisagreementError,
    DefaultFacetUndefinedError,
    FacetNotAvailableError,
    MembersNotHeldError,
    MixedGrainError,
    OverCeilingError,
    UpstreamResponseError,
    UpstreamVersionChangedError,
)
from materialise.formats import ExcelIllegalCharacterError, ExcelLimitError, ParquetSizeBoundError
from materialise.graph import GraphDeadlineExceededError, GraphReadError
from materialise.rate_tables import RateTableInvalidError
from materialise.recipes import NonCurrencyUnitError, RateCoverageGapError, RecipeNotDeclaredError
from materialise.reshape import (
    ColumnNotDroppableError,
    ColumnUnknownError,
    PivotNotHomogeneousError,
    PivotNotUniqueError,
    PivotOverCellCapError,
    PivotOverColumnCapError,
    RollupNotSummableError,
)
from spec_registry import (
    IdentityCollision,
    IdentityQuarantined,
    LiveSpecRegistry,
    SpecIdMismatch,
    SpecRegistryUnavailable,
)
from statvar_members import FiveTuple, MemberReadDeadline

from .conftest import FakeDatasetStore


def _definition_refusal(exc: Refusal) -> Refusal:
    """The definition GET route's form of a materialisation refusal."""
    exc.strip_pointer = True
    return exc


def _member_not_resolvable_at_spec_id() -> MemberNotResolvable:
    """The spec-id kind sets the issue pointer after construction."""
    exc = MemberNotResolvable(members=("dcid:a", "dcid:b"), truncated=True)
    exc.issue_pointer = "/input/spec_id"
    return exc


def _column_unknown_at_pivot() -> ColumnUnknownError:
    """The reshape step sets the pointer after construction."""
    exc = ColumnUnknownError("exemplar", column="exemplar", alternatives=("place.name",))
    exc.pointer = "/options/pivot"
    return exc


CASES: tuple[tuple[str, BaseException], ...] = (
    ("mint_errors", MemberNotResolvable(members=("dcid:exemplar",))),
    ("mint_errors", DefinitionNotReproducible(spec_ids=("spec_exemplar",), detail="exemplar")),
    ("mint_errors", SpecNotExportable(spec_id="spec_exemplar", detail="exemplar")),
    (
        "mint_errors",
        RoleSignatureIncompatible(
            spec_id="spec_exemplar", roles_by_stat_var={"dcid:exemplar": ("recipient",)}
        ),
    ),
    ("mint_errors", CapabilityUnavailable(detail="exemplar")),
    ("mint_errors", QueryOptionsInvalid(pointer="/options/groupby/levels", detail="exemplar")),
    ("mint_errors", MaterialiseSpecNotFound(subject_id="spec_exemplar", pointer="/input/spec_id")),
    ("mint_errors", MalformedSidecarIdentity(detail="exemplar", pointer="/input/spec_id")),
    ("mint_errors", InvalidRoleRef(stat_var="dcid:exemplar", role="exemplar")),
    ("mint_errors", RoleIncompatible(stat_var="dcid:exemplar", role="exemplar")),
    (
        "mint_errors",
        CurrencyRoleUndeclared(
            stat_var="dcid:exemplar", role="exemplar", declared_roles=("exemplar",)
        ),
    ),
    ("mint_errors", RecipeNotDeclaredError("exemplar", cause="recipe_undeclared")),
    (
        "mint_errors",
        RateCoverageGapError("exemplar", cause="currency_unmapped", entities=("CurrencyExemplar",)),
    ),
    ("mint_errors", NonCurrencyUnitError("exemplar", unit="Exemplar")),
    ("mint_errors", RateTableInvalidError("exemplar")),
    ("mint_errors", ExcludedByRegistry(variables=("Exemplar",), provenances=("dc/base/Exemplar",))),
    (
        "mint_errors",
        UnSystemUnavailable(variable="undata/exemplar/Exemplar", places=("country/Exemplar",)),
    ),
    ("mint_errors", DefaultFacetUndefinedError("exemplar")),
    ("mint_errors", FacetNotAvailableError("exemplar")),
    ("mint_errors", OverCeilingError("exemplar", ceiling_kind="rows", count=1, cap=1)),
    ("mint_errors", MembersNotHeldError(stat_vars=("exemplar",))),
    (
        "mint_errors",
        DatasetDisagreementError(
            "exemplar", dataset="provenance/Exemplar", stat_var="dcid:exemplar"
        ),
    ),
    ("mint_errors", FrameError("exemplar", "exemplar")),
    ("mint_errors", CollectionProbeCeilingExceededError("exemplar")),
    ("mint_errors", ExcelLimitError("exemplar")),
    ("mint_errors", ExcelIllegalCharacterError("exemplar", row=0, column=0)),
    ("mint_errors", ParquetSizeBoundError("exemplar")),
    (
        "mint_errors",
        RollupNotSummableError(
            "exemplar", axis="exemplar-axis", precondition="exemplar-precondition"
        ),
    ),
    (
        "mint_errors",
        PivotNotUniqueError(
            "exemplar",
            duplicate_keys=(("exemplar-index", "exemplar-columns"),),
            truncated=False,
        ),
    ),
    ("mint_errors", PivotOverColumnCapError("exemplar", count=1, cap=1)),
    ("mint_errors", PivotOverCellCapError("exemplar", count=1, cap=1)),
    (
        "mint_errors",
        PivotNotHomogeneousError(
            "exemplar",
            offenders=(("exemplar-column", "exemplar-index", ("exemplar-a", "exemplar-b")),),
            truncated=False,
        ),
    ),
    ("mint_errors", ColumnNotDroppableError("exemplar", columns=("exemplar",))),
    ("mint_errors", ColumnUnknownError("exemplar", column="exemplar")),
    ("mint_errors", UpstreamVersionChangedError("exemplar")),
    ("mint_errors", MintBackpressure()),
    ("mint_errors", GraphDeadlineExceededError("exemplar")),
    (
        "mint_errors",
        MemberReadDeadline(five_tuple=FiveTuple(), candidates_considered=0, candidate_cap=1),
    ),
    ("mint_errors", GraphReadError("exemplar")),
    ("mint_errors", SdmxQueryError("exemplar")),
    ("mint_errors", SdmxTransportError("exemplar")),
    ("mint_errors", SdmxHttpError("exemplar", upstream_status=502)),
    ("mint_errors", SdmxResponseError("exemplar")),
    ("mint_errors", SdmxRoleCeilingError("exemplar", upstream_status=400)),
    ("mint_errors", SdmxCapExceededError("exemplar", cap_kind="rows", cap=1, observed=2)),
    ("mint_errors", SdmxMissingFilterError("exemplar", upstream_status=400)),
    ("mint_errors", SdmxUnsupportedFilterError("exemplar", upstream_status=400)),
    ("mint_errors", SdmxContainmentFilterError("exemplar", upstream_status=400)),
    ("mint_errors", SdmxIncompatibleVariablesError("exemplar", upstream_status=400)),
    ("mint_errors", UpstreamResponseError("exemplar")),
    (
        "mint_errors",
        ArtifactRecordConflictError(
            digest="sha256:exemplar",
            tier=Tier.working,
            stored_bucket="exemplar-bucket",
            stored_object_name="sha256:exemplar",
            stored_size_bytes=1,
            incoming_bucket="exemplar-bucket-2",
            incoming_object_name="sha256:exemplar",
            incoming_size_bytes=2,
        ),
    ),
    ("mint_errors", ArtifactWriteError(object_name="sha256:exemplar", digest="sha256:exemplar")),
    ("mint_errors", ArtifactInventoryError(digest="sha256:exemplar")),
    ("mint_errors", DatasetsStoreUnavailable("exemplar")),
    ("definition", DefinitionSpecNotFound(id="spec_exemplar")),
    ("definition", DefinitionMemberNotResolvable(members=("spec_exemplar",))),
    (
        "definition",
        StoredTraceMissingResolvedStatVars(
            subject_id="spec_exemplar", content_version="cv_exemplar"
        ),
    ),
    (
        "definition",
        _definition_refusal(FacetNotAvailableError("exemplar", facet_pointer="/facets/exemplar")),
    ),
    (
        "definition",
        _definition_refusal(
            ExcludedByRegistry(variables=("Exemplar",), provenances=("dc/base/Exemplar",))
        ),
    ),
    ("definition", QueryOptionsInvalid(pointer="/options/exemplar", detail="exemplar")),
    ("definition", DefinitionBackpressure()),
    ("definition", MalformedSidecarIdentity(detail="exemplar", pointer="/definition")),
    ("definition", SpecIdMismatch(supplied="spec_exemplar", computed="spec_exemplar_computed")),
    ("definition", DefinitionNotReproducible(spec_ids=("spec_exemplar",), detail="exemplar")),
    ("definition", SpecNotExportable(spec_id="spec_exemplar", detail="exemplar")),
    (
        "definition",
        RoleSignatureIncompatible(
            spec_id="spec_exemplar", roles_by_stat_var={"dcid:exemplar": ("recipient",)}
        ),
    ),
    ("definition", OverCeilingError("exemplar", ceiling_kind="rows", count=1, cap=1)),
    ("definition", RecipeNotDeclaredError("exemplar", cause="recipe_undeclared")),
    (
        "definition",
        RateCoverageGapError("exemplar", cause="currency_unmapped", entities=("CurrencyExemplar",)),
    ),
    ("definition", NonCurrencyUnitError("exemplar", unit="Exemplar")),
    ("definition", RateTableInvalidError("exemplar")),
    ("definition", DatasetsStoreUnavailable("exemplar")),
    ("definition", SnapshotNotFound(id="spec_exemplar", content_version="cv_exemplar")),
    ("definition", SnapshotRevoked(id="spec_exemplar", content_version="cv_exemplar", reason=None)),
    ("definition", ArtifactExpired(id="spec_exemplar", content_version="cv_exemplar")),
    ("register", SpecIdMismatch(supplied="urn:spec:supplied", computed="urn:spec:computed")),
    ("register", IdentityCollision(subject_id="urn:spec:collision")),
    ("register", IdentityQuarantined(subject_id="urn:spec:quarantined", id_kind="spec")),
    ("register", SpecRegistryUnavailable("registry write pool exhausted")),
    ("register", RegistryBackpressure()),
    ("register", IdentityError("schema_violation", "identity document failed schema validation")),
    ("snapshot", SnapshotSpecNotFound(id="spec_exemplar")),
    ("snapshot", SnapshotNotFound(id="spec_exemplar", content_version="cv_exemplar")),
    ("snapshot", SnapshotRevoked(id="spec_exemplar", content_version="cv_exemplar", reason=None)),
    ("snapshot", ArtifactExpired(id="spec_exemplar", content_version="cv_exemplar")),
    ("snapshot", SnapshotBackpressure()),
    (
        "files",
        SnapshotSigningFailed(
            bucket="exemplar-bucket",
            object_name="sha256:exemplar",
            digest="sha256:exemplar",
            error_class_name=None,
            status_code=None,
        ),
    ),
    (
        "files",
        ArtifactSigningError(
            bucket="exemplar-bucket", object_name="sha256:exemplar", digest="sha256:exemplar"
        ),
    ),
    (
        "observations",
        ObservationsUnavailable(
            id="spec_exemplar", content_version="cv_exemplar", reason="holds no frame1"
        ),
    ),
    ("observations", ObservationsBackpressure()),
    ("observations", ObservationsGraphUnavailable()),
    (
        "observations",
        StoredFrameCorrupt(
            id="spec_exemplar", content_version="cv_exemplar", key=("Count_Person", ())
        ),
    ),
    ("pin", SnapshotSpecNotFound(id="spec_exemplar")),
    ("pin", SnapshotNotFound(id="spec_exemplar", content_version="cv_exemplar")),
    ("pin", SnapshotRevoked(id="spec_exemplar", content_version="cv_exemplar", reason=None)),
    ("pin", ArtifactExpired(id="spec_exemplar", content_version="cv_exemplar")),
    (
        "pin",
        ArtifactNotFound(
            id="spec_exemplar", content_version="cv_exemplar", artifact_digest="sha256:exemplar"
        ),
    ),
    ("pin", PinNotPermitted()),
    ("pin", PinBackpressure()),
    ("pin", PinBudgetExhausted(pinned=0, limit=0)),
    ("pin", DatasetsStoreUnavailable("exemplar")),
    ("publish", PublishFrameInvalid(category="schema_violation", message="exemplar")),
    ("publish", PublishBodyTooLarge(content_length=1, limit=1)),
    ("publish", PublishBodyEncodingInvalid("exemplar")),
    ("publish", PublishContentLengthRequired(content_length_header=None)),
    ("publish", PublishBytesDiverge(spec_id="ops_exemplar", content_version="ocv_exemplar")),
    ("publish", PublishLineageDiverge(spec_id="ops_exemplar", content_version="ocv_exemplar")),
    ("publish", SnapshotRevoked(id="ops_exemplar", content_version="ocv_exemplar", reason=None)),
    ("publish", PublishRecordMissing(spec_id="ops_exemplar", content_version="ocv_exemplar")),
    ("publish", PublishRecipeStoreMissing(spec_id="ops_exemplar", content_version="ocv_exemplar")),
    ("publish", PublishNotPermitted()),
    ("publish", OverCeilingError("exemplar", ceiling_kind="rows", count=1, cap=1)),
    ("publish", PublishBackpressure()),
    (
        "publish",
        ArtifactRecordConflictError(
            digest="sha256:exemplar",
            tier=Tier.working,
            stored_bucket="exemplar-bucket",
            stored_object_name="sha256:exemplar",
            stored_size_bytes=1,
            incoming_bucket="exemplar-bucket-2",
            incoming_object_name="sha256:exemplar",
            incoming_size_bytes=2,
        ),
    ),
    ("publish", ArtifactInventoryError(digest="sha256:exemplar")),
    ("publish", ArtifactWriteError(object_name="sha256:exemplar", digest="sha256:exemplar")),
    (
        "publish",
        RecipeDiverges(
            spec_id="ops_exemplar", content_version="ocv_exemplar", field="normal_form_hash"
        ),
    ),
    (
        "publish",
        ReplayKeyHeld(
            replay_key="rpk_exemplar",
            spec_id="ops_exemplar",
            content_version="ocv_exemplar",
            holder_spec_id="ops_holder",
            holder_content_version="ocv_holder",
        ),
    ),
    ("publish", DatasetsStoreUnavailable("exemplar")),
    ("recipe", RecipeNotFound(spec_id="ops_exemplar", content_version="ocv_exemplar")),
    ("recipe", ReplayNotFound(replay_key="rpk_exemplar")),
    ("recipe", RecipeBackpressure()),
    ("recipe", RecipeFrameNotStored(spec_id="ops_exemplar", content_version="ocv_exemplar")),
    (
        "recipe",
        RecipeDiverges(
            spec_id="ops_exemplar", content_version="ocv_exemplar", field="normal_form_hash"
        ),
    ),
    (
        "recipe",
        ReplayKeyHeld(
            replay_key="rpk_exemplar",
            spec_id="ops_exemplar",
            content_version="ocv_exemplar",
            holder_spec_id="ops_holder",
            holder_content_version="ocv_holder",
        ),
    ),
    ("recipe", PublishNotPermitted()),
    ("refresh", FrameNotRefreshable(spec_id="ops_exemplar")),
    ("refresh", RenderingNotRecorded(spec_id="spec_exemplar", content_version="cv_exemplar")),
    ("refresh", RefreshRebuildFailed(operation_id="op_exemplar", refusal_code="over_ceiling")),
    ("refresh", RefreshBackpressure()),
    ("refresh", RefreshNotPermitted(presented=False)),
    ("meta", OperationNotFound(operation_id="op_exemplar")),
    ("meta", OperationFailed(operation_id="op_exemplar", refusal_code="excel_limit")),
    # Shapes the exemplars above miss, as real raise sites build them.
    ("extra", FacetNotAvailableError("exemplar", facet_pointer="/facets/provenance")),
    (
        "extra",
        FacetNotAvailableError(
            "exemplar",
            facet_pointer="/facets/unit",
            alternatives=[Alternative(ref="Exemplar", label="Exemplar")],
        ),
    ),
    (
        "extra",
        FacetNotAvailableError(
            "exemplar",
            issues=(ProblemIssue(pointer="/facets/unit", axis="unit", code="facet_not_available"),),
        ),
    ),
    ("extra", DefaultFacetUndefinedError("exemplar", facet_pointer="/facets/unit")),
    (
        "extra",
        DefaultFacetUndefinedError(
            "exemplar",
            issues=(
                ProblemIssue(pointer="/facets/unit", axis="unit", code="default_facet_undefined"),
            ),
        ),
    ),
    (
        "extra",
        _definition_refusal(
            DefaultFacetUndefinedError(
                "exemplar",
                issues=(
                    ProblemIssue(
                        pointer="/facets/unit", axis="unit", code="default_facet_undefined"
                    ),
                ),
            )
        ),
    ),
    (
        "extra",
        _definition_refusal(DefaultFacetUndefinedError("exemplar", facet_pointer="/facets/unit")),
    ),
    (
        "extra",
        RecipeNotDeclaredError("exemplar", cause="recipe_undeclared", facet_pointer="/facets/unit"),
    ),
    ("extra", NonCurrencyUnitError("exemplar", unit="Exemplar", facet_pointer="/facets/unit")),
    ("extra", MixedGrainError({"year": "2019", "month": "2019-01"}, stat_vars=("dcid:exemplar",))),
    ("extra", _member_not_resolvable_at_spec_id()),
    ("extra", _column_unknown_at_pivot()),
    ("extra", PivotOverColumnCapError("exemplar", count=2, cap=1)),
    (
        "extra",
        PivotNotUniqueError(
            "exemplar",
            duplicate_keys=(("i", "c"),),
            varying=(("unit", 2),),
            truncated=True,
        ),
    ),
    ("extra", OverCeilingError("exemplar", detail="exemplar detail", ceiling_kind="bytes")),
    ("extra", DatasetsStoreUnavailable("exemplar", sqlstate="42P01")),
    (
        "extra",
        SnapshotSigningFailed(
            bucket="exemplar-bucket",
            object_name="sha256:exemplar",
            digest="sha256:exemplar",
            error_class_name=None,
            status_code=403,
        ),
    ),
    (
        "extra",
        ArtifactInventoryError(
            digest="sha256:exemplar", tier=Tier.working, bucket="b", object_name="sha256:exemplar"
        ),
    ),
    ("extra", RefreshNotPermitted(presented=False)),
)


_GOLDEN = Path(__file__).with_name("refusal_wire_golden.json")

# The deliberate body diffs from the golden, by case: the fields whose value changes, and why.
_DELIBERATE_DIFFS: dict[str, dict[str, object]] = {
    # A refusal renders its own alternatives. A facet refusal's alternatives reach the top level
    # only when it is raised outside aggregation with alternatives, which no raise site does.
    "extra-134-FacetNotAvailableError": {
        "alternatives": [{"ref": "Exemplar", "label": "Exemplar"}],
    },
    # A role-shaped currency refusal offers only the currency roles that bind a declared role,
    # and the exemplar's declared role `exemplar` is reachable through none.
    "mint_errors-10-CurrencyRoleUndeclared": {"alternatives": []},
    # `/facets/unit` locates the failure in the materialisation request, not the posted body,
    # so it stays the refusal's `facet_pointer` and leaves the top level.
    "extra-136-DefaultFacetUndefinedError": {"pointer": None},
    # Publish stores the canonical frame, so its refusals name a stored frame in place of frame1.
    "publish-105-PublishBytesDiverge": {
        "detail": "spec_id 'ops_exemplar' content_version 'ocv_exemplar' already holds a stored "
        "frame whose content differs from this publish call's frame; refusing to report success "
        "for a digest that was not actually recorded under this pair. This most likely means the "
        "analysis engine that minted this content_version did not reproduce the same frame on a "
        "second run -- publish the frame already held under this id, or mint a new "
        "content_version for the corrected recipe; retrying with this same frame will refuse "
        "identically",
    },
    "publish-106-PublishLineageDiverge": {
        "detail": "spec_id 'ops_exemplar' content_version 'ocv_exemplar' already holds a stored "
        "frame with this publish call's content but a different input_content_versions "
        "derivation trace than this publish call's own; refusing to report success for a "
        "derivation trace that was not actually recorded under this pair. This most likely "
        "means this content_version was minted twice for genuinely different inputs -- mint a "
        "new content_version for the corrected derivation, or publish the "
        "input_content_versions already held under this id; retrying with these same claimed "
        "inputs will refuse identically",
    },
    "publish-108-PublishRecordMissing": {
        "detail": "spec_id 'ops_exemplar' content_version 'ocv_exemplar' rendered and wrote its "
        "canonical frame to the artifact store, but no version row is readable back through "
        "this door's own snapshot_reader immediately after recording one; this publish door was "
        "composed with a version_writer that does not share snapshot_reader's own store, or "
        "that writer's insert silently failed to land -- retrying will refuse identically under "
        "the same wiring",
    },
}


# Every capacity refusal carries `Retry-After`. The golden predates the header.
_DELIBERATE_DIFFS.update(
    {
        f"{door}-{index}-{type(exc).__name__}": {"headers": {"Retry-After": "1"}}
        for index, (door, exc) in enumerate(CASES)
        if isinstance(exc, Backpressure)
    }
)


def _case_ids() -> list[str]:
    return [f"{door}-{index}-{type(exc).__name__}" for index, (door, exc) in enumerate(CASES)]


def _rendered() -> dict[str, object]:
    """Each case as the wire carries it: the surface's problem, or the platform catch-all's
    when the surface defers."""
    classify = datasets_surface(registry=LiveSpecRegistry(store=FakeDatasetStore())).classify
    return {
        case_id: (classify(exc) or classify_default(exc)).to_canonical_data()
        for case_id, (_door, exc) in zip(_case_ids(), CASES, strict=True)
    }


def _expected() -> dict[str, object]:
    """The golden with the deliberate diffs applied. A golden `None` is a case the surface
    deferred to the platform catch-all."""
    golden = json.loads(_GOLDEN.read_text())
    expected: dict[str, object] = {}
    for case_id, (_door, exc) in zip(_case_ids(), CASES, strict=True):
        body = golden[case_id] or classify_default(exc).to_canonical_data()
        expected[case_id] = {**body, **_DELIBERATE_DIFFS.get(case_id, {})}
    return expected


@pytest.fixture(scope="module")
def rendered() -> dict[str, object]:
    return _rendered()


@pytest.fixture(scope="module")
def expected() -> dict[str, object]:
    return _expected()


@pytest.mark.parametrize("case_id", _case_ids())
def test_refusal_renders_as_before(
    case_id: str, rendered: dict[str, object], expected: dict[str, object]
) -> None:
    assert rendered[case_id] == expected[case_id]


def test_golden_covers_exactly_the_cases() -> None:
    assert sorted(json.loads(_GOLDEN.read_text())) == sorted(_case_ids())
    assert set(_DELIBERATE_DIFFS) <= set(_case_ids())


def _backpressure_classes(base: type[Backpressure]) -> list[type[Backpressure]]:
    return [
        sub for child in base.__subclasses__() for sub in (child, *_backpressure_classes(child))
    ]


@pytest.mark.parametrize("cls", _backpressure_classes(Backpressure), ids=lambda cls: cls.__name__)
def test_capacity_refusal_carries_retry_after(cls: type[Backpressure]) -> None:
    classify = datasets_surface(registry=LiveSpecRegistry(store=FakeDatasetStore())).classify
    spec = classify(cls())
    assert spec is not None
    assert (spec.status, spec.code, spec.retryable) == (503, "service_unavailable", True)
    assert spec.headers == {"Retry-After": "1"}
