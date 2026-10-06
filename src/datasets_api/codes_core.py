"""Codes minted by the scaffold, register, and meta doors.

Platform codes stay in `PLATFORM_CODE_CATALOGUE` to avoid `CodeCollision`. This catalogue owns
`operation_not_found` and `operation_failed` for polling. Failed operations return RFC 9457
problem details. Successful polls return a 200 `OperationResource`. It also owns `spec_not_found`
for the mint door's `spec_id` kind and the subject routes.
"""

from __future__ import annotations

CORE_CODE_CATALOGUE: dict[str, tuple[str, str]] = {
    "spec_id_mismatch": (
        "Spec Id Mismatch",
        "The supplied definition does not hash to the id beside it.",
    ),
    "identity_collision": (
        "Identity Collision",
        "The registry write found byte-different content already held at this id.",
    ),
    "identity_quarantined": (
        "Identity Quarantined",
        "This id already has byte-different content collision-quarantined under it; "
        "registration is refused until the quarantine is resolved.",
    ),
    "spec_not_found": (
        "Spec Not Found",
        "No definition is registered at this id, and none was supplied beside it.",
    ),
    "operation_not_found": (
        "Operation Not Found",
        "No operation exists with this id, or it is done and this caller is not its authorized principal.",
    ),
    "operation_failed": (
        "Operation Failed",
        "The operation ran and did not complete successfully.",
    ),
}
