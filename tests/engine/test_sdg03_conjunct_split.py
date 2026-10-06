"""A single-concept SDG 3.3.2 query must not split into a data-free definite answer.

The guard covers extraction that treats the bare code fragment as a second variable.
"""

from __future__ import annotations

from tests.engine._harness import make_request, offline_resolve

_SDG03_QUERY = "tuberculosis incidence rate SDG 3.3.2 Ethiopia"


def test_sdg03_never_grounds_empty_stat_vars_as_definite():
    """A bare SDG 3.3.2 fragment cannot ground a definite response without StatVars."""
    result = offline_resolve(make_request(_SDG03_QUERY))

    for part in result.parts:
        if part.status == "definite":
            assert part.spec.resolved.members, (
                "definite response must never carry empty stat_vars for a query with "
                "no real candidates of its own (the sdg-03 'SDG 3.3.2' empty-breadth regression)"
            )
        else:
            assert part.status == "no_data"
