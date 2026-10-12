"""The mint door's exception inventory: the types its classifier maps, which
`datasets_api.doors.mint.build_mint_door` registers on its `DoorContribution`, and one real
instance of each.
"""

from __future__ import annotations

from dfo_frame_grammar import FrameError

from artifact_store import (
    ArtifactInventoryError,
    ArtifactRecordConflictError,
    ArtifactWriteError,
    Tier,
)
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
    SpecNotFound,
    UnSystemUnavailable,
)
from datasets_api.doors.mint_errors import MintBackpressure
from datasets_api.operation_store import OperationBacklogFull
from datasets_api.query_options import QueryOptionsInvalid
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
from dfo_platform.problem import Refusal
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
from statvar_members import FiveTuple, MemberReadDeadline

MINT_EXCEPTION_TYPES: tuple[type[Exception], ...] = (
    # Every door renders any refusal (`datasets_api.refusals`).
    Refusal,
    MixedGrainError,
    MemberNotResolvable,
    DefinitionNotReproducible,
    SpecNotExportable,
    RoleSignatureIncompatible,
    CapabilityUnavailable,
    QueryOptionsInvalid,
    SpecNotFound,
    MalformedSidecarIdentity,
    InvalidRoleRef,
    RoleIncompatible,
    CurrencyRoleUndeclared,
    RecipeNotDeclaredError,
    RateCoverageGapError,
    NonCurrencyUnitError,
    RateTableInvalidError,
    ExcludedByRegistry,
    UnSystemUnavailable,
    DefaultFacetUndefinedError,
    FacetNotAvailableError,
    OverCeilingError,
    MembersNotHeldError,
    DatasetDisagreementError,
    FrameError,
    CollectionProbeCeilingExceededError,
    ExcelLimitError,
    ExcelIllegalCharacterError,
    ParquetSizeBoundError,
    RollupNotSummableError,
    PivotNotUniqueError,
    PivotOverColumnCapError,
    PivotOverCellCapError,
    PivotNotHomogeneousError,
    ColumnNotDroppableError,
    ColumnUnknownError,
    UpstreamVersionChangedError,
    MintBackpressure,
    # A mint over the inline budget submits an operation, which a full backlog refuses.
    OperationBacklogFull,
    GraphDeadlineExceededError,
    MemberReadDeadline,
    GraphReadError,
    SdmxQueryError,
    SdmxTransportError,
    SdmxHttpError,
    SdmxResponseError,
    SdmxRoleCeilingError,
    SdmxCapExceededError,
    SdmxMissingFilterError,
    SdmxUnsupportedFilterError,
    SdmxContainmentFilterError,
    SdmxIncompatibleVariablesError,
    UpstreamResponseError,
    ArtifactRecordConflictError,
    ArtifactWriteError,
    ArtifactInventoryError,
    # Classified at the surface level, but listed here because version writes
    # can raise it.
    DatasetsStoreUnavailable,
)

MINT_EXCEPTION_EXEMPLARS: tuple[BaseException, ...] = (
    MixedGrainError({"year": "2019", "month": "2019-01"}, stat_vars=("dcid:exemplar",)),
    MemberNotResolvable(members=("dcid:exemplar",)),
    DefinitionNotReproducible(spec_ids=("spec_exemplar",), detail="exemplar"),
    SpecNotExportable(spec_id="spec_exemplar", detail="exemplar"),
    RoleSignatureIncompatible(
        spec_id="spec_exemplar", roles_by_stat_var={"dcid:exemplar": ("recipient",)}
    ),
    CapabilityUnavailable(detail="exemplar"),
    QueryOptionsInvalid(pointer="/options/groupby/levels", detail="exemplar"),
    SpecNotFound(subject_id="spec_exemplar", pointer="/input/spec_id"),
    MalformedSidecarIdentity(detail="exemplar", pointer="/input/spec_id"),
    InvalidRoleRef(stat_var="dcid:exemplar", role="exemplar"),
    RoleIncompatible(stat_var="dcid:exemplar", role="exemplar"),
    CurrencyRoleUndeclared(stat_var="dcid:exemplar", role="exemplar", declared_roles=("exemplar",)),
    RecipeNotDeclaredError("exemplar", cause="recipe_undeclared"),
    RateCoverageGapError("exemplar", cause="currency_unmapped", entities=("CurrencyExemplar",)),
    NonCurrencyUnitError("exemplar", unit="Exemplar"),
    RateTableInvalidError("exemplar"),
    ExcludedByRegistry(variables=("Exemplar",), provenances=("dc/base/Exemplar",)),
    UnSystemUnavailable(variable="undata/exemplar/Exemplar", places=("country/Exemplar",)),
    DefaultFacetUndefinedError("exemplar"),
    FacetNotAvailableError("exemplar"),
    OverCeilingError("exemplar", ceiling_kind="rows", count=1, cap=1),
    MembersNotHeldError(stat_vars=("exemplar",)),
    DatasetDisagreementError("exemplar", dataset="provenance/Exemplar", stat_var="dcid:exemplar"),
    FrameError("exemplar", "exemplar"),
    CollectionProbeCeilingExceededError("exemplar"),
    ExcelLimitError("exemplar"),
    ExcelIllegalCharacterError("exemplar", row=0, column=0),
    ParquetSizeBoundError("exemplar"),
    RollupNotSummableError("exemplar", axis="exemplar-axis", precondition="exemplar-precondition"),
    PivotNotUniqueError(
        "exemplar",
        duplicate_keys=(("exemplar-index", "exemplar-columns"),),
        truncated=False,
    ),
    PivotOverColumnCapError("exemplar", count=1, cap=1),
    PivotOverCellCapError("exemplar", count=1, cap=1),
    PivotNotHomogeneousError(
        "exemplar",
        offenders=(("exemplar-column", "exemplar-index", ("exemplar-a", "exemplar-b")),),
        truncated=False,
    ),
    ColumnNotDroppableError("exemplar", columns=("exemplar",)),
    ColumnUnknownError("exemplar", column="exemplar"),
    UpstreamVersionChangedError("exemplar"),
    MintBackpressure(),
    OperationBacklogFull(in_flight=1, limit=1),
    GraphDeadlineExceededError("exemplar"),
    MemberReadDeadline(five_tuple=FiveTuple(), candidates_considered=0, candidate_cap=1),
    GraphReadError("exemplar"),
    SdmxQueryError("exemplar"),
    SdmxTransportError("exemplar"),
    SdmxHttpError("exemplar", upstream_status=502),
    SdmxResponseError("exemplar"),
    SdmxRoleCeilingError("exemplar", upstream_status=400),
    SdmxCapExceededError("exemplar", cap_kind="rows", cap=1, observed=2),
    SdmxMissingFilterError("exemplar", upstream_status=400),
    SdmxUnsupportedFilterError("exemplar", upstream_status=400),
    SdmxContainmentFilterError("exemplar", upstream_status=400),
    SdmxIncompatibleVariablesError("exemplar", upstream_status=400),
    UpstreamResponseError("exemplar"),
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
    ArtifactWriteError(object_name="sha256:exemplar", digest="sha256:exemplar"),
    ArtifactInventoryError(digest="sha256:exemplar"),
    DatasetsStoreUnavailable("exemplar"),
)
