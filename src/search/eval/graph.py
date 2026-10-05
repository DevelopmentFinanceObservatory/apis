"""Graph access used by the Search eval checks.

``LiveGraphClient`` implements this protocol. Calls raise on network or API errors; the
calling checks record those failures as ``infra_error`` results.
"""

from typing import Protocol

from dc_graph import Facet


class GraphClient(Protocol):
    """Minimal graph-access interface used by the eval checks."""

    def exists_batch(self, dcids: list[str]) -> set[str]:
        """Return known dcids and raise on network or API errors."""
        ...

    def observation_facets_batch(
        self, stat_vars: list[str], entities: list[str], needs_dates: bool = False
    ) -> dict[str, dict[str, list[Facet]]]:
        """Return each StatVar's facets at each entity, with every observation date when
        ``needs_dates`` is set, and raise on errors. A pair without facets is absent or
        empty.

        The materialisation check counts the observations of the facets an answer states
        from this read.
        """
        ...
