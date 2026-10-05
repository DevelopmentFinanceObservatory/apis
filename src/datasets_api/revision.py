"""`SentinelRevisionReader` supplies a fixed revision while the dfo-dcp endpoint is absent. The
sentinel marks provenance unavailable in the trace and content-version preimage. Frame bytes set the
content version.

Every read returns the same revision, so `read_generation` treats upstream refreshes as stable. The
capability document reports `revision_provenance_live: false`.
"""

from __future__ import annotations

from materialise.revision import RevisionSnapshot

UNAVAILABLE_REVISION = "unavailable:dfo-dcp-revision-endpoint-absent"


class SentinelRevisionReader:
    """Returns the fixed unavailable-revision snapshot used by this deployment."""

    def read(self) -> RevisionSnapshot:
        return RevisionSnapshot(graph_revision=UNAVAILABLE_REVISION, source_revisions=())
