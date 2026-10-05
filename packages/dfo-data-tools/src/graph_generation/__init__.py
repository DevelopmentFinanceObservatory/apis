"""Process-wide graph generation stamps shared across API surfaces.

Caches capture one stamp before computing and clear when the token or the loaded origin
catalogue changes. Unreadable sources use the unknown generation's shorter cache lifetime while
requests continue."""

from __future__ import annotations

from graph_generation.fake import FakeGeneration
from graph_generation.protocol import CatalogueBuilds, GenerationProvider
from graph_generation.provider import UNKNOWN_GENERATION, GraphGeneration
from graph_generation.registry import RegistrationHandle
from graph_generation.shared import acquire_shared, release_shared
from graph_generation.source import (
    CloudRunGenerationSource,
    FixedGenerationSource,
    GenerationSource,
    GenerationSourceError,
)
from graph_generation.stamp import UNKNOWN_STAMP, GenerationStamp

__all__ = [
    "UNKNOWN_GENERATION",
    "UNKNOWN_STAMP",
    "CatalogueBuilds",
    "CloudRunGenerationSource",
    "FakeGeneration",
    "FixedGenerationSource",
    "GenerationProvider",
    "GenerationSource",
    "GenerationSourceError",
    "GenerationStamp",
    "GraphGeneration",
    "RegistrationHandle",
    "acquire_shared",
    "release_shared",
]
