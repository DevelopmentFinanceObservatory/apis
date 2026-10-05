"""The origin catalogue: which upstream provenances, variables and topics the
API admits, and from which Data Commons each fact comes.

`registry` holds the reviewed list of upstream sources. The catalogue job
expands it into tables (`catalogue.CatalogueTables`) published through
`artifact`. `service.CatalogueService` loads them, and each request binds one
snapshot (`context`) that admission reads.

Exports backed by `artifact` and `service` resolve on first access, since
they need pyarrow. The registry, the snapshot types and the request binding
import without it.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from origin_catalogue.catalogue import (
    STALE_AFTER,
    BaseTables,
    BaseVariable,
    CardFacts,
    Catalogue,
    CatalogueSnapshot,
    CatalogueTableError,
    CatalogueTables,
    DatasetRef,
    ProvenanceRow,
    SeriesAdmission,
    SourceRef,
    entry_ref,
)
from origin_catalogue.context import (
    CatalogueNotBound,
    UnBlockedPlaces,
    UnReadFailure,
    UnUnserved,
    bind_catalogue,
    bound_catalogue,
    current_catalogue,
    note_un_blocked_places,
    note_un_unserved,
    record_un_read_failure,
    recording_un_blocked_places,
    unbind_catalogue,
)
from origin_catalogue.origins import BASE, DFO, ORIGINS, UN_SYSTEM, DataCommons, OriginId
from origin_catalogue.registry import Registry, RegistryError, load_registry
from origin_catalogue.schema import CatalogueManifest, CataloguePointer, ScanLevel

if TYPE_CHECKING:
    from origin_catalogue.artifact import (
        CatalogueArtifactError,
        CatalogueTooLargeError,
        load_catalogue,
    )
    from origin_catalogue.service import (
        CatalogueService,
        CatalogueUriMissingError,
        get_catalogue_service,
        reset_catalogue_service,
    )

_LAZY: dict[str, tuple[str, str]] = {
    "CatalogueArtifactError": ("origin_catalogue.artifact", "CatalogueArtifactError"),
    "CatalogueTooLargeError": ("origin_catalogue.artifact", "CatalogueTooLargeError"),
    "load_catalogue": ("origin_catalogue.artifact", "load_catalogue"),
    "CatalogueService": ("origin_catalogue.service", "CatalogueService"),
    "CatalogueUriMissingError": ("origin_catalogue.service", "CatalogueUriMissingError"),
    "get_catalogue_service": ("origin_catalogue.service", "get_catalogue_service"),
    "reset_catalogue_service": ("origin_catalogue.service", "reset_catalogue_service"),
}


def __getattr__(name: str) -> Any:
    """Imports and caches a lazy export. Unknown names raise `AttributeError`."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "BASE",
    "DFO",
    "ORIGINS",
    "STALE_AFTER",
    "UN_SYSTEM",
    "BaseTables",
    "BaseVariable",
    "CardFacts",
    "Catalogue",
    "CatalogueArtifactError",
    "CatalogueManifest",
    "CatalogueNotBound",
    "CataloguePointer",
    "CatalogueService",
    "CatalogueSnapshot",
    "CatalogueTableError",
    "CatalogueTables",
    "CatalogueTooLargeError",
    "CatalogueUriMissingError",
    "DataCommons",
    "DatasetRef",
    "OriginId",
    "ProvenanceRow",
    "Registry",
    "RegistryError",
    "ScanLevel",
    "SeriesAdmission",
    "SourceRef",
    "UnBlockedPlaces",
    "UnReadFailure",
    "UnUnserved",
    "bind_catalogue",
    "bound_catalogue",
    "current_catalogue",
    "entry_ref",
    "get_catalogue_service",
    "load_catalogue",
    "load_registry",
    "note_un_blocked_places",
    "note_un_unserved",
    "record_un_read_failure",
    "recording_un_blocked_places",
    "reset_catalogue_service",
    "unbind_catalogue",
]
