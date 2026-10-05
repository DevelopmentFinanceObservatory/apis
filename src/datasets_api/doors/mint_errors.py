"""The mint door's own exception classification: `MintBackpressure`,
`classify_mint_exception`, and the `DoorContribution` exception inventory
(`MINT_EXCEPTION_TYPES`/`MINT_EXCEPTION_EXEMPLARS`) that
`datasets_api.doors.mint.build_mint_door` wires into its own
`DoorContribution`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from dfo_frame_grammar import FrameError

from artifact_store import (
    ARTIFACT_INVENTORY_WRITE_FAILED_MARKER,
    ArtifactInventoryError,
    ArtifactRecordConflictError,
    ArtifactSigningError,
    ArtifactWriteError,
    Tier,
)
from datasets_api.codes_materialise import (
    CapabilityUnavailable,
    CodeAmbiguous,
    CodeNotFoundError,
    CurrencyRoleUndeclared,
    DefinitionNotReproducible,
    ExcludedByRegistry,
    InvalidRoleRef,
    MalformedSidecarIdentity,
    MemberNotResolvable,
    MemberSelectionMismatch,
    RenderedArtifactSigningFailed,
    RoleIncompatible,
    RoleSignatureIncompatible,
    SpecNotExportable,
    SpecNotFound,
    UnSystemUnavailable,
    signing_failure_is_permanent,
)
from datasets_api.doors.conversion_errors import classify_conversion_exception
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
from dfo_platform import ProblemSpec
from dfo_platform.problem import Alternative, ProblemIssue
from dfo_platform.refusal import TIMEOUT_DEFAULT
from materialise.errors import (
    CollectionProbeCeilingExceededError,
    DatasetDisagreementError,
    DefaultFacetUndefinedError,
    FacetNotAvailableError,
    MembersNotHeldError,
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

logger = logging.getLogger(__name__)


class MintBackpressure(Exception):
    """Mint or shared-materialisation capacity is full. Maps to retryable
    `service_unavailable` (503), including the definition door's shared gate.
    """


def _classify_member_not_resolvable(exc: MemberNotResolvable) -> ProblemSpec:
    pointer = exc.pointer or "/input/statvars"
    issues = [ProblemIssue(pointer=pointer, member=member) for member in exc.members]
    return ProblemSpec(
        status=422,
        code="member_not_resolvable",
        detail=str(exc),
        retryable=False,
        truncated=exc.truncated,
        issues=issues,
    )


def _classify_member_selection_mismatch(exc: MemberSelectionMismatch) -> ProblemSpec:
    issues = [
        ProblemIssue(pointer=f"{exc.pointer}/{index}", member=dcid) for index, dcid in exc.foreign
    ]
    return ProblemSpec(
        status=422,
        code="member_not_resolvable",
        detail=str(exc),
        retryable=False,
        issues=issues,
    )


def _classify_definition_not_reproducible(exc: DefinitionNotReproducible) -> ProblemSpec:
    issues = [ProblemIssue(member=spec_id) for spec_id in exc.spec_ids]
    issues.extend(
        ProblemIssue(member=spec_id, axis=role)
        for spec_id in exc.spec_ids
        for role in exc.undeclared_roles
    )
    return ProblemSpec(
        status=422,
        code="definition_not_reproducible",
        detail=str(exc),
        retryable=False,
        issues=issues,
    )


def _classify_spec_not_exportable(exc: SpecNotExportable) -> ProblemSpec:
    return ProblemSpec(status=409, code="spec_not_exportable", detail=str(exc), retryable=False)


def _classify_role_signature_incompatible(exc: RoleSignatureIncompatible) -> ProblemSpec:
    return ProblemSpec(
        status=422,
        code="role_signature_incompatible",
        detail=str(exc),
        retryable=False,
        issues=[ProblemIssue(member=stat_var) for stat_var in sorted(exc.roles_by_stat_var)],
    )


def _classify_code_ambiguous(exc: CodeAmbiguous) -> ProblemSpec:
    # Source-code resolution attaches each ambiguous entry's request index.
    # Direct raises use the broader `/input/codes` pointer.
    indices = getattr(exc, "indices", None)
    issues = [
        ProblemIssue(
            pointer=f"/input/codes/{indices[i]}" if indices is not None else "/input/codes",
            member=code,
            alternatives=[Alternative(ref=ref, label=label) for ref, label in candidates],
            truncated=False,
        )
        for i, (code, candidates) in enumerate(exc.ambiguous)
    ]
    return ProblemSpec(
        status=422, code="code_ambiguous", detail=str(exc), retryable=False, issues=issues
    )


def _classify_code_not_found(exc: CodeNotFoundError) -> ProblemSpec:
    # Resolution attaches the code's request index. Direct raises use the
    # broader `/input/codes` pointer.
    index = getattr(exc, "index", None)
    return ProblemSpec(
        status=404,
        code="code_not_found",
        detail=str(exc),
        retryable=False,
        pointer=f"/input/codes/{index}" if index is not None else "/input/codes",
    )


def _classify_capability_unavailable(exc: CapabilityUnavailable) -> ProblemSpec:
    return ProblemSpec(
        status=501,
        code="capability_unavailable",
        detail=exc.detail,
        retryable=False,
        pointer=exc.pointer,
    )


def _classify_spec_not_found(exc: SpecNotFound) -> ProblemSpec:
    return ProblemSpec(
        status=404, code="spec_not_found", detail=str(exc), retryable=False, pointer=exc.pointer
    )


def _classify_malformed_sidecar_identity(exc: MalformedSidecarIdentity) -> ProblemSpec:
    # Collection errors identify the member. Definition errors use the top-level pointer.
    issues = (
        [ProblemIssue(pointer=exc.pointer, member=exc.member_spec_id)]
        if exc.member_spec_id is not None
        else None
    )
    return ProblemSpec(
        status=400,
        code="request_invalid",
        detail=str(exc),
        retryable=False,
        pointer=exc.pointer if issues is None else None,
        issues=issues,
    )


def _classify_invalid_role_ref(exc: InvalidRoleRef) -> ProblemSpec:
    # Collection errors identify the member. Declared roles list valid
    # alternatives for the offending scope axis.
    return ProblemSpec(
        status=422,
        code="invalid_role_ref",
        detail=str(exc),
        retryable=False,
        issues=[
            ProblemIssue(
                pointer=f"/input/scope/entities/{exc.role}",
                member=exc.member_spec_id if exc.member_spec_id is not None else exc.role,
                axis=exc.role,
                alternatives=[Alternative(ref=role, label=role) for role in exc.declared_roles],
                truncated=False,
            )
        ],
    )


def _classify_role_incompatible(exc: RoleIncompatible) -> ProblemSpec:
    return ProblemSpec(
        status=422,
        code="role_incompatible",
        detail=str(exc),
        retryable=False,
        issues=[
            ProblemIssue(
                pointer=f"/input/scope/entities/{exc.role}",
                member=exc.member_spec_id if exc.member_spec_id is not None else exc.role,
                axis=exc.role,
                alternatives=[Alternative(ref=role, label=role) for role in exc.declared_roles],
                truncated=False,
            )
        ],
    )


def _classify_currency_role_undeclared(exc: CurrencyRoleUndeclared) -> ProblemSpec:
    # Collection errors carry a member issue. Stored-spec re-materialisation
    # has no request pointer. Declared roles appear as top-level alternatives.
    alternatives = [Alternative(ref=role, label=role) for role in exc.declared_roles]
    issues = (
        [ProblemIssue(pointer=exc.pointer, member=exc.member_spec_id, alternatives=alternatives)]
        if exc.member_spec_id is not None
        else None
    )
    return ProblemSpec(
        status=400,
        code="request_invalid",
        detail=str(exc),
        retryable=False,
        pointer=exc.pointer if issues is None else None,
        alternatives=alternatives if issues is None else None,
        issues=issues,
    )


def _classify_default_facet_undefined(exc: DefaultFacetUndefinedError) -> ProblemSpec:
    # Member aggregation carries every undefined axis. A direct
    # `select_facet` failure uses the top-level pointer.
    issues = list(exc.issues) if exc.issues else None
    return ProblemSpec(
        status=422,
        code="default_facet_undefined",
        detail=str(exc),
        retryable=False,
        pointer=exc.pointer if not issues else None,
        issues=issues,
    )


def _classify_excluded_by_registry(exc: ExcludedByRegistry) -> ProblemSpec:
    return ProblemSpec(status=422, code="excluded_by_registry", detail=str(exc), retryable=False)


def _classify_un_system_unavailable(exc: UnSystemUnavailable) -> ProblemSpec:
    return ProblemSpec(status=502, code="upstream_error", detail=str(exc), retryable=True)


def _classify_facet_not_available(exc: FacetNotAvailableError) -> ProblemSpec:
    return ProblemSpec(
        status=422,
        code="facet_not_available",
        detail=str(exc),
        retryable=False,
        issues=list(exc.issues) if exc.issues else None,
    )


def _classify_over_ceiling(exc: OverCeilingError) -> ProblemSpec:
    return ProblemSpec(
        status=422,
        code="over_ceiling",
        detail=exc.detail or str(exc),
        retryable=False,
        ceiling_kind=exc.ceiling_kind,
        count=exc.count,
        cap=exc.cap,
    )


def _classify_members_not_held(exc: MembersNotHeldError) -> ProblemSpec:
    # The fault is graph drift on the operator side, so it counts as a server error.
    logger.error("Datasets mint door named StatVars with no graph node: %s", exc, exc_info=exc)
    return ProblemSpec(
        status=500,
        code="members_not_held",
        detail=str(exc),
        retryable=False,
        issues=[ProblemIssue(member=stat_var) for stat_var in exc.stat_vars],
    )


def _classify_dataset_disagreement(exc: DatasetDisagreementError) -> ProblemSpec:
    # The code index and the series coverage disagree on a member's dataset, so the fault is
    # the operator's and a retry reproduces it.
    logger.error("Datasets mint door found a member outside its elected dataset: %s", exc)
    return ProblemSpec(
        status=500,
        code="internal_error",
        detail=str(exc),
        retryable=False,
        issues=[ProblemIssue(member=exc.stat_var)],
    )


def _classify_frame_error(exc: FrameError) -> ProblemSpec:
    # The mint built rows the frame grammar refuses. The same request reproduces it.
    logger.error("Datasets mint door built a frame the grammar refuses: %s", exc, exc_info=exc)
    return ProblemSpec(
        status=500,
        code="internal_error",
        detail=f"the frame this request builds does not conform to the frame grammar: {exc}",
        retryable=False,
    )


def _classify_collection_probe_ceiling_exceeded(
    exc: CollectionProbeCeilingExceededError,
) -> ProblemSpec:
    return ProblemSpec(
        status=422, code="collection_probe_ceiling", detail=str(exc), retryable=False
    )


def _classify_excel_limit(exc: ExcelLimitError) -> ProblemSpec:
    return ProblemSpec(
        status=422, code="excel_limit", detail=exc.detail or str(exc), retryable=False
    )


def _classify_excel_illegal_character(exc: ExcelIllegalCharacterError) -> ProblemSpec:
    return ProblemSpec(
        status=422, code="excel_illegal_character", detail=exc.detail or str(exc), retryable=False
    )


def _classify_parquet_size_bound(exc: ParquetSizeBoundError) -> ProblemSpec:
    return ProblemSpec(
        status=422, code="parquet_size_bound", detail=exc.detail or str(exc), retryable=False
    )


def _classify_rollup_not_summable(exc: RollupNotSummableError) -> ProblemSpec:
    # The structured issue lets clients branch on the axis without parsing detail.
    return ProblemSpec(
        status=422,
        code="rollup_not_summable",
        detail=str(exc),
        retryable=False,
        pointer="/options/groupby",
        issues=[ProblemIssue(pointer="/options/groupby", axis=exc.axis)],
    )


def _classify_pivot_not_unique(exc: PivotNotUniqueError) -> ProblemSpec:
    # Each duplicate key, including a stringify collision, becomes one issue
    # at `/options/pivot`.
    issues = [ProblemIssue(pointer="/options/pivot", member=str(key)) for key in exc.duplicate_keys]
    return ProblemSpec(
        status=422,
        code="pivot_not_unique",
        detail=str(exc),
        retryable=False,
        truncated=exc.truncated,
        issues=issues,
    )


def _classify_pivot_over_column_cap(exc: PivotOverColumnCapError) -> ProblemSpec:
    return ProblemSpec(
        status=422,
        code="pivot_over_column_cap",
        detail=str(exc),
        retryable=False,
        pointer="/options/pivot",
    )


def _classify_pivot_over_cell_cap(exc: PivotOverCellCapError) -> ProblemSpec:
    return ProblemSpec(
        status=422,
        code="pivot_over_cell_cap",
        detail=str(exc),
        retryable=False,
        pointer="/options/pivot",
    )


def _classify_pivot_not_homogeneous(exc: PivotNotHomogeneousError) -> ProblemSpec:
    # Report each offending column and index value as one `/options/pivot` issue.
    issues = [
        ProblemIssue(pointer="/options/pivot", member=f"{column}={index_value!r}")
        for column, index_value, _values in exc.offenders
    ]
    return ProblemSpec(
        status=422,
        code="pivot_not_homogeneous",
        detail=str(exc),
        retryable=False,
        truncated=exc.truncated,
        issues=issues,
    )


def _classify_column_not_droppable(exc: ColumnNotDroppableError) -> ProblemSpec:
    # Report every non-droppable column in table order at `/options/columns`.
    issues = [ProblemIssue(pointer="/options/columns", member=column) for column in exc.columns]
    return ProblemSpec(
        status=422, code="column_not_droppable", detail=str(exc), retryable=False, issues=issues
    )


def _classify_column_unknown(exc: ColumnUnknownError) -> ProblemSpec:
    # `_reshape` sets `/options/columns` for unresolved tokens and
    # `/options/pivot` for invalid pivot fields. Token alternatives list
    # qualified entity-field names with identical refs and labels. A missing
    # pointer falls back to `/options/columns`.
    return ProblemSpec(
        status=422,
        code="column_unknown",
        detail=str(exc),
        retryable=False,
        pointer=exc.pointer or "/options/columns",
        alternatives=exc.alternatives,
    )


def _classify_upstream_version_changed(exc: UpstreamVersionChangedError) -> ProblemSpec:
    return ProblemSpec(
        status=503, code="upstream_version_changed", detail=exc.detail, retryable=True
    )


def _classify_mint_backpressure(exc: MintBackpressure) -> ProblemSpec:
    return ProblemSpec(
        status=503,
        code="service_unavailable",
        detail="the mint door is at its concurrent-materialisation capacity right now; retry",
        retryable=True,
    )


def _classify_graph_deadline_exceeded(exc: GraphDeadlineExceededError) -> ProblemSpec:
    # Map this subtype before `GraphReadError`. A deadline breach is a request timeout.
    return ProblemSpec(
        status=TIMEOUT_DEFAULT.status,
        code=TIMEOUT_DEFAULT.code,
        detail="the request's deadline expired during the graph facet probe; retry",
        retryable=TIMEOUT_DEFAULT.retryable,
    )


def _classify_graph_read_error(exc: GraphReadError) -> ProblemSpec:
    logger.error("Datasets mint door's graph read failed: %s", exc, exc_info=exc)
    return ProblemSpec(
        status=502,
        code="upstream_error",
        detail="the graph read this mint needed failed upstream; retry",
        retryable=True,
    )


def _classify_sdmx_query_error(exc: SdmxQueryError) -> ProblemSpec:
    # Invalid role bindings and values are deterministic client errors. Callers
    # must change the query.
    return ProblemSpec(status=400, code="request_invalid", detail=str(exc), retryable=False)


def _classify_sdmx_role_ceiling_error(exc: SdmxRoleCeilingError) -> ProblemSpec:
    # The mixer accepts at most 3 entity-role slots. `scope.entities` allows
    # Larger bindings reach this client error before `SdmxHttpError`.
    return ProblemSpec(status=400, code="request_invalid", detail=str(exc), retryable=False)


def _classify_sdmx_composition_bug(
    exc: SdmxMissingFilterError
    | SdmxUnsupportedFilterError
    | SdmxContainmentFilterError
    | SdmxIncompatibleVariablesError,
) -> ProblemSpec:
    # Data queries always include `variableMeasured`. Custom components enter
    # through `availability()`. This door uses `data()`. `Scope.entities` accepts
    # plain dcids, and chunks group variables by `declared_roles`. These errors
    # signal a server composition bug. `materialise.generation._generate_chunk`
    # reports an incompatible-variables error caused by StatVars with no graph
    # node as `MembersNotHeldError`, so one reaching here names only held StatVars.
    logger.error(
        "Datasets mint door composed an SDMX query the mixer rejected as malformed "
        "in a way this surface should never produce: %s",
        exc,
        exc_info=exc,
    )
    return ProblemSpec(status=500, code="internal_error", detail=str(exc), retryable=False)


def _classify_sdmx_cap_exceeded(exc: SdmxCapExceededError) -> ProblemSpec:
    # A read past a caller's cap is a size refusal, the same as the raw-fetch guard a mint maps
    # to `OverCeilingError` itself.
    return ProblemSpec(
        status=422,
        code="over_ceiling",
        detail=(
            f"the SDMX read was aborted after {exc.observed} {exc.cap_kind}, past its cap of "
            f"{exc.cap} {exc.cap_kind}"
        ),
        retryable=False,
        ceiling_kind=exc.cap_kind,
        count=exc.observed,
        cap=exc.cap,
    )


def _classify_sdmx_upstream_error(
    exc: SdmxTransportError | SdmxHttpError | SdmxResponseError,
) -> ProblemSpec:
    logger.error("Datasets mint door's SDMX read failed: %s", exc, exc_info=exc)
    return ProblemSpec(
        status=502,
        code="upstream_error",
        detail="the SDMX read this mint needed failed upstream; retry",
        retryable=True,
    )


def _classify_upstream_response_error(exc: UpstreamResponseError) -> ProblemSpec:
    # Materialisation raises this when a fetched SDMX response conflicts with its request.
    logger.error("Datasets mint door's upstream response was inconsistent: %s", exc, exc_info=exc)
    return ProblemSpec(
        status=502,
        code="upstream_error",
        detail="the upstream response for this mint was inconsistent with the request; retry",
        retryable=True,
    )


def _classify_artifact_record_conflict(exc: ArtifactRecordConflictError) -> ProblemSpec:
    # The same bytes reproduce the conflict. Render attaches subject and
    # version for reconciliation. Other instances yield `None` for those fields.
    subject_id = getattr(exc, "subject_id", None)
    content_version = getattr(exc, "content_version", None)
    logger.warning(
        "Datasets mint door's artifact write hit a record conflict for subject %s, "
        "content_version %s, digest %s, tier %s: the freshly uploaded object has no "
        "inventory row pointing at it and needs manual reconciliation",
        subject_id,
        content_version,
        exc.digest,
        exc.tier.value,
    )
    return ProblemSpec(status=500, code="internal_error", detail=str(exc), retryable=False)


def _classify_rendered_artifact_signing_failed(exc: RenderedArtifactSigningFailed) -> ProblemSpec:
    # Storage and inventory are committed before signing. Digest-keyed writes
    # make retries reuse the same bytes unless bucket settings drift.
    # Permanent permission failures map to 500. Transient or unknown causes map
    # to retryable 503.
    if signing_failure_is_permanent(
        error_class_name=exc.error_class_name, status_code=exc.status_code
    ):
        logger.error(
            "Datasets mint door failed to sign a GET url for a rendered artifact "
            "for a permanent, operator-clearable reason: %s",
            exc,
            exc_info=exc,
        )
        return ProblemSpec(status=500, code="internal_error", detail=str(exc), retryable=False)
    logger.error(
        "Datasets mint door failed to sign a GET url for a rendered artifact: %s", exc, exc_info=exc
    )
    return ProblemSpec(
        status=503,
        code="service_unavailable",
        detail="signing the minted artifact's download url failed; retry",
        retryable=True,
    )


def _classify_artifact_signing_error(exc: ArtifactSigningError) -> ProblemSpec:
    # In-surface rendering wraps signing errors. This arm handles external failures.
    logger.error("Datasets mint door caught an unattributed signing failure: %s", exc, exc_info=exc)
    return ProblemSpec(
        status=503,
        code="service_unavailable",
        detail="signing an artifact's download url failed; retry",
        retryable=True,
    )


def _classify_artifact_inventory_error(exc: ArtifactInventoryError) -> ProblemSpec:
    # A failed inventory write leaves uploaded bytes orphaned. Bucket and
    # Object fields trigger reconciliation logging. Lookup failures lack them.
    if exc.bucket is not None and exc.object_name is not None:
        logger.error(
            "%s digest=%s tier=%s bucket=%s object_name=%s",
            ARTIFACT_INVENTORY_WRITE_FAILED_MARKER,
            exc.digest,
            exc.tier.value if exc.tier is not None else None,
            exc.bucket,
            exc.object_name,
        )
    else:
        logger.error("Datasets mint door's artifact inventory lookup failed: %s", exc, exc_info=exc)
    return ProblemSpec(
        status=503,
        code="service_unavailable",
        detail="writing the minted artifact failed; retry",
        retryable=True,
    )


def _classify_artifact_write_error(exc: ArtifactWriteError) -> ProblemSpec:
    logger.error("Datasets mint door's artifact store write failed: %s", exc, exc_info=exc)
    return ProblemSpec(
        status=503,
        code="service_unavailable",
        detail="writing the minted artifact failed; retry",
        retryable=True,
    )


# The first matching table entry wins. List subclasses before base classes.
# Handlers accept narrow exception types. `Any` fits the contravariant callable slot.
_MintClassifier = tuple[
    type[BaseException] | tuple[type[BaseException], ...], Callable[[Any], ProblemSpec]
]

_MINT_CLASSIFIERS: tuple[_MintClassifier, ...] = (
    (MemberNotResolvable, _classify_member_not_resolvable),
    (MemberSelectionMismatch, _classify_member_selection_mismatch),
    (DefinitionNotReproducible, _classify_definition_not_reproducible),
    (SpecNotExportable, _classify_spec_not_exportable),
    (RoleSignatureIncompatible, _classify_role_signature_incompatible),
    (CodeAmbiguous, _classify_code_ambiguous),
    (CodeNotFoundError, _classify_code_not_found),
    (CapabilityUnavailable, _classify_capability_unavailable),
    (SpecNotFound, _classify_spec_not_found),
    (MalformedSidecarIdentity, _classify_malformed_sidecar_identity),
    (InvalidRoleRef, _classify_invalid_role_ref),
    (RoleIncompatible, _classify_role_incompatible),
    (CurrencyRoleUndeclared, _classify_currency_role_undeclared),
    (ExcludedByRegistry, _classify_excluded_by_registry),
    (UnSystemUnavailable, _classify_un_system_unavailable),
    (DefaultFacetUndefinedError, _classify_default_facet_undefined),
    (FacetNotAvailableError, _classify_facet_not_available),
    (OverCeilingError, _classify_over_ceiling),
    (MembersNotHeldError, _classify_members_not_held),
    (DatasetDisagreementError, _classify_dataset_disagreement),
    (FrameError, _classify_frame_error),
    (CollectionProbeCeilingExceededError, _classify_collection_probe_ceiling_exceeded),
    (ExcelLimitError, _classify_excel_limit),
    (ExcelIllegalCharacterError, _classify_excel_illegal_character),
    (ParquetSizeBoundError, _classify_parquet_size_bound),
    (RollupNotSummableError, _classify_rollup_not_summable),
    (PivotNotUniqueError, _classify_pivot_not_unique),
    (PivotOverColumnCapError, _classify_pivot_over_column_cap),
    (PivotOverCellCapError, _classify_pivot_over_cell_cap),
    (PivotNotHomogeneousError, _classify_pivot_not_homogeneous),
    (ColumnNotDroppableError, _classify_column_not_droppable),
    (ColumnUnknownError, _classify_column_unknown),
    (UpstreamVersionChangedError, _classify_upstream_version_changed),
    (MintBackpressure, _classify_mint_backpressure),
    (GraphDeadlineExceededError, _classify_graph_deadline_exceeded),
    (GraphReadError, _classify_graph_read_error),
    (SdmxQueryError, _classify_sdmx_query_error),
    (SdmxRoleCeilingError, _classify_sdmx_role_ceiling_error),
    (SdmxCapExceededError, _classify_sdmx_cap_exceeded),
    (
        (
            SdmxMissingFilterError,
            SdmxUnsupportedFilterError,
            SdmxContainmentFilterError,
            SdmxIncompatibleVariablesError,
        ),
        _classify_sdmx_composition_bug,
    ),
    ((SdmxTransportError, SdmxHttpError, SdmxResponseError), _classify_sdmx_upstream_error),
    (UpstreamResponseError, _classify_upstream_response_error),
    (ArtifactRecordConflictError, _classify_artifact_record_conflict),
    (RenderedArtifactSigningFailed, _classify_rendered_artifact_signing_failed),
    (ArtifactSigningError, _classify_artifact_signing_error),
    (ArtifactInventoryError, _classify_artifact_inventory_error),
    (ArtifactWriteError, _classify_artifact_write_error),
)


def classify_mint_exception(exc: BaseException) -> ProblemSpec | None:
    """Return the first matching problem, then try conversion errors, else `None`."""
    for exc_types, handler in _MINT_CLASSIFIERS:
        if isinstance(exc, exc_types):
            return handler(exc)
    return classify_conversion_exception(exc)


MINT_EXCEPTION_TYPES: tuple[type[Exception], ...] = (
    MemberNotResolvable,
    MemberSelectionMismatch,
    DefinitionNotReproducible,
    SpecNotExportable,
    RoleSignatureIncompatible,
    CodeAmbiguous,
    CodeNotFoundError,
    CapabilityUnavailable,
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
    GraphDeadlineExceededError,
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
    RenderedArtifactSigningFailed,
    ArtifactSigningError,
    ArtifactWriteError,
    ArtifactInventoryError,
    # Classified at the surface level, but listed here because version writes
    # can raise it.
    DatasetsStoreUnavailable,
)

MINT_EXCEPTION_EXEMPLARS: tuple[BaseException, ...] = (
    MemberNotResolvable(members=("dcid:exemplar",)),
    MemberSelectionMismatch(foreign=((0, "exemplar"),)),
    DefinitionNotReproducible(spec_ids=("spec_exemplar",), detail="exemplar"),
    SpecNotExportable(spec_id="spec_exemplar", detail="exemplar"),
    RoleSignatureIncompatible(
        spec_id="spec_exemplar", roles_by_stat_var={"dcid:exemplar": ("recipient",)}
    ),
    CodeAmbiguous(ambiguous=(("exemplar", (("dcid:a", "A"), ("dcid:b", "B"))),)),
    CodeNotFoundError(code="exemplar", source=None),
    CapabilityUnavailable(detail="exemplar"),
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
    GraphDeadlineExceededError("exemplar"),
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
    RenderedArtifactSigningFailed(
        bucket="exemplar-bucket",
        object_name="sha256:exemplar",
        digest="sha256:exemplar",
        error_class_name=None,
        status_code=None,
    ),
    ArtifactSigningError(
        bucket="exemplar-bucket", object_name="sha256:exemplar", digest="sha256:exemplar"
    ),
    ArtifactWriteError(object_name="sha256:exemplar", digest="sha256:exemplar"),
    ArtifactInventoryError(digest="sha256:exemplar"),
    DatasetsStoreUnavailable("exemplar"),
)
