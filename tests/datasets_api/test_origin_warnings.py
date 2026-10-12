"""Tests the origin catalogue's manifest warnings and the `excluded_by_registry` and
UN refusals."""

from __future__ import annotations

import itertools
import json
import time
from collections.abc import Mapping, Sequence

import httpx
import pytest

from datasets_api.codes_materialise import ExcludedByRegistry, UnSystemUnavailable
from datasets_api.doors.definition import classify_definition_exception
from datasets_api.doors.mint_errors import classify_mint_exception
from datasets_api.manifest_models import WarningModel
from datasets_api.origin import (
    WITHHELD_READ_MAX_BYTES,
    WithheldReader,
    refuse_unserved_empty_probe,
    refuse_unserved_selection,
    un_unavailable_warnings,
    withheld_warning,
    withheld_warnings,
)
from dc_graph import (
    Facet,
    GraphCapExceededError,
    GraphDeadlineExceededError,
    GraphReadError,
)
from dc_sdmx import ContainedIn, SdmxCapExceededError, SdmxClient, SdmxTransportError
from graph_generation import FakeGeneration, GenerationStamp
from materialise import FacetBinding, MaterialisationRequest
from materialise.errors import DefaultFacetUndefinedError
from materialise.place_scope import PlaceScopeReadError
from materialise.window import Coverage, RequestedWindow
from origin_catalogue import (
    BASE,
    DFO,
    UN_SYSTEM,
    UnReadFailure,
    note_un_blocked_places,
    recording_un_blocked_places,
)
from origin_catalogue.context import un_unavailability_record
from origin_catalogue.graph import build_graph_client
from tests.materialise._scope_availability import ScopeAvailability
from tests.materialise.conftest import FakeGraphReader
from tests.origin_catalogue.fake_catalogue import (
    provenance_row,
    restrictive_catalogue,
    un_mirror,
    un_variable,
)

from ._prewrite import prewrite

_KEN, _NGA, _UGA = "country/KEN", "country/NGA", "country/UGA"
_UN_VARIABLE = "undata/who/MORT_U5_RATE"


class _WithheldGraph:
    """Answers the unfiltered re-read with the given provenances at every requested entity,
    or at `at` alone when given, for a read naming any variable, or one of `of` when given.
    Records each call with its deadline and byte cap."""

    def __init__(
        self,
        withheld: Sequence[str] = (),
        *,
        at: Sequence[str] | None = None,
        of: Sequence[str] | None = None,
        fails: Exception | None = None,
    ) -> None:
        self._withheld = tuple(withheld)
        self._at = None if at is None else set(at)
        self._of = None if of is None else set(of)
        self._fails = fails
        self.calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
        self.limits: list[tuple[float | None, int | None]] = []

    def withheld_at(
        self,
        variables: Sequence[str],
        entities: Sequence[str],
        *,
        deadline: float | None = None,
        max_bytes: int | None = None,
    ) -> Mapping[str, tuple[str, ...]]:
        self.calls.append((tuple(variables), tuple(entities)))
        self.limits.append((deadline, max_bytes))
        if self._fails is not None:
            raise self._fails
        if not self._withheld or (self._of is not None and self._of.isdisjoint(variables)):
            return {}
        return {
            entity: self._withheld for entity in entities if self._at is None or entity in self._at
        }


_GENERATIONS = itertools.count()


class _ScopedWithheldGraph(_WithheldGraph):
    """A `_WithheldGraph` that also reaches a UN deployment whose availability lists
    `un_places` for every UN variable, or fails with `un_fails`, and records each such read.
    It holds every place it is asked about as a node. The countries fallback reads `countries`
    as Earth's countries and `held` as the `(StatVar, place)` pairs with admitted data. Each
    graph has a generation of its own, so no test reads another's memoised scope."""

    def __init__(
        self,
        *args,
        un_places: Sequence[str] = (),
        un_fails: Exception | None = None,
        countries: Sequence[str] = (),
        held: Sequence[tuple[str, str]] = (),
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._un_places = list(un_places)
        self._un_fails = un_fails
        self._countries = list(countries)
        self._held = frozenset(held)
        self._generation = FakeGeneration(token=f"scoped-withheld-{next(_GENERATIONS)}")
        self.scope_reads: list[tuple[str, ...]] = []

    def source_key(self) -> str:
        return "fake-graph"

    def generation_stamp(self) -> GenerationStamp:
        return self._generation.stamp()

    def child_places_batch(
        self, parent_dcids: Sequence[str], child_type: str, *, deadline: float | None = None
    ) -> dict[str, list[str]]:
        return {parent: list(self._countries) for parent in parent_dcids}

    def entities_with_data_by_variable(
        self,
        variable_dcids: Sequence[str],
        entity_dcids: Sequence[str],
        *,
        deadline: float | None = None,
    ) -> dict[str, list[str]]:
        return {
            variable: [entity for entity in entity_dcids if (variable, entity) in self._held]
            for variable in variable_dcids
        }

    def un_availability(self) -> _ScopedWithheldGraph:
        return self

    def availability_values(
        self,
        *,
        variables: Sequence[str],
        component: str,
        filters: object = None,
        max_values: int | None = None,
        max_bytes: int | None = None,
        deadline: float | None = None,
        read_seconds: float | None = None,
    ) -> list[str]:
        self.scope_reads.append(tuple(variables))
        if self._un_fails is not None:
            raise self._un_fails
        return list(self._un_places)

    def node_arcs_many(
        self, dcids: Sequence[str], *, properties: Sequence[str], deadline: float | None = None
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        return {dcid: dict.fromkeys(properties, ("Place",)) for dcid in dcids}

    def close(self) -> None:
        return None


def _scope(*held: tuple[str, str]) -> ScopeAvailability:
    """Availability for the default place scope: `held` lists the `(StatVar, place)` pairs with
    data."""
    return ScopeAvailability(data=held)


class _FailingAvailability:
    def __init__(self, failure: Exception) -> None:
        self._failure = failure

    def source_key(self) -> str:
        return "fake-availability"

    def availability_values(self, **_kwargs: object) -> list[str]:
        raise self._failure


def _unbound(stat_var: str = "dcid:Count_Person") -> MaterialisationRequest:
    """A subject-only request that binds no entity, so it covers the default place scope."""
    return MaterialisationRequest(
        stat_var=stat_var,
        role_bindings={},
        declared_roles=("observationAbout",),
        facet_bindings={},
        window=RequestedWindow(kind="all"),
    )


def _request(
    stat_var: str = "dcid:Count_Person",
    entities: object = (_KEN, _NGA),
    roles: tuple[str, ...] = ("observationAbout",),
) -> MaterialisationRequest:
    return MaterialisationRequest(
        stat_var=stat_var,
        role_bindings={roles[0]: entities},
        declared_roles=roles,
        facet_bindings={},
        window=RequestedWindow(kind="all"),
    )


def _served(*entities: str) -> Coverage:
    return Coverage(
        rows=len(entities),
        dimensions=(),
        window=None,
        entity_refs={"observationAbout": frozenset(entities)},
    )


def test_an_unserved_entity_with_withheld_data_names_the_provenances():
    graph = _WithheldGraph(withheld=["dc/base/UN_SDG"])

    found = withheld_warning(graph, [(_request(), _served(_KEN))], deadline=None)

    assert found is not None
    index, warning = found
    assert index == 0
    assert warning.code == "withheld_by_registry"
    assert warning.severity == "warn"
    assert "country/NGA" in warning.message
    assert "dc/base/UN_SDG" in warning.message
    assert graph.calls == [(("Count_Person",), (_NGA,))]


def test_every_entity_served_reads_nothing():
    graph = _WithheldGraph(withheld=["dc/base/UN_SDG"])

    assert withheld_warnings(graph, [(_request(), _served(_KEN, _NGA))], deadline=None) == ()
    assert graph.calls == []


def test_an_unserved_entity_with_nothing_withheld_warns_nothing():
    graph = _WithheldGraph()

    assert withheld_warnings(graph, [(_request(), _served())], deadline=None) == ()
    assert graph.calls == [(("Count_Person",), (_KEN, _NGA))]


def test_the_re_read_runs_once_for_the_first_request_with_an_unserved_entity():
    graph = _WithheldGraph()
    served = [
        (_request("dcid:A"), _served(_KEN, _NGA)),
        (_request("dcid:B"), _served(_KEN)),
        (_request("dcid:C", entities=(_UGA,)), _served()),
    ]

    assert withheld_warning(graph, served, deadline=None) is None
    assert graph.calls == [(("B",), (_NGA,))]


def test_the_warning_carries_the_index_of_its_request():
    graph = _WithheldGraph(withheld=["dc/base/X"])
    served = [(_request("dcid:A"), _served(_KEN, _NGA)), (_request("dcid:B"), _served(_KEN))]

    found = withheld_warning(graph, served, deadline=None)

    assert found is not None and found[0] == 1


def test_role_bearing_and_scoped_requests_are_skipped():
    graph = _WithheldGraph(withheld=["dc/base/X"])
    served = [
        (_request(roles=("provider", "recipient")), _served()),
        (_request(entities=ContainedIn(parent="Earth", child_type="Country")), _served()),
    ]

    assert withheld_warnings(graph, served, deadline=None) == ()
    assert graph.calls == []


def test_a_request_binding_no_entity_is_checked_at_the_countries_it_left_unserved():
    graph = _ScopedWithheldGraph(withheld=["dc/base/UN_SDG"])

    found = withheld_warning(
        graph, [(_unbound(), _served(_KEN))], default_scope=(_KEN, _NGA, _UGA), deadline=5.0
    )

    assert found is not None
    index, warning = found
    assert index == 0
    assert f"served no observations for {_NGA}, {_UGA};" in warning.message
    # Earth is the scope's fallback and no part of a country answer, so it is not re-read.
    assert graph.calls == [(("Count_Person",), (_NGA, _UGA))]
    # The countries are the ones the result carries, so the warning reads no containment.
    assert graph.scope_reads == []


def test_a_request_binding_no_entity_is_checked_at_each_default_place_it_did_not_serve():
    graph = _WithheldGraph(withheld=["dc/base/UN_SDG"])

    (warning,) = withheld_warnings(
        graph, [(_unbound(), _served(_UGA))], default_scope=(_KEN, _NGA, _UGA), deadline=None
    )

    assert f"served no observations for {_KEN}, {_NGA};" in warning.message


def test_a_request_binding_no_entity_served_at_every_country_reads_nothing():
    graph = _WithheldGraph(withheld=["dc/base/UN_SDG"])

    assert (
        withheld_warnings(
            graph, [(_unbound(), _served(_KEN, _NGA))], default_scope=(_KEN, _NGA), deadline=None
        )
        == ()
    )
    assert graph.calls == []


def test_a_request_binding_no_entity_is_skipped_without_a_default_scope():
    graph = _ScopedWithheldGraph(withheld=["dc/base/X"])

    assert withheld_warnings(graph, [(_unbound(), _served())], deadline=1.0) == ()
    assert graph.calls == []
    assert graph.scope_reads == []


def test_the_warning_names_only_the_entities_the_re_read_reports():
    # NGA has admitted data outside the served window, so the re-read leaves it out.
    graph = _WithheldGraph(withheld=["dc/base/UN_SDG"], at=[_UGA])
    request = _request(entities=(_KEN, _NGA, _UGA))

    (warning,) = withheld_warnings(graph, [(request, _served(_KEN))], deadline=None)

    assert f"served no observations for {_UGA};" in warning.message
    assert _NGA not in warning.message


def test_the_re_read_runs_under_the_deadline_and_the_byte_cap():
    graph = _WithheldGraph(withheld=["dc/base/X"])

    withheld_warnings(graph, [(_request(), _served())], deadline=123.0)

    assert graph.limits == [(123.0, WITHHELD_READ_MAX_BYTES)]


@pytest.mark.parametrize(
    "failure",
    [
        GraphDeadlineExceededError("deadline"),
        GraphCapExceededError("cap", cap_kind="bytes", cap=1, observed=2),
    ],
)
def test_a_failed_re_read_leaves_the_answer_without_the_warning(failure: Exception):
    graph = _WithheldGraph(withheld=["dc/base/X"], fails=failure)

    assert withheld_warnings(graph, [(_request(), _served())], deadline=1.0) == ()


def test_a_graph_without_the_re_read_reads_nothing():
    assert withheld_warnings(object(), [(_request(), _served())], deadline=None) == ()


def _check_empty_probe(
    graph: object,
    *requests: MaterialisationRequest,
    availability: object | None = None,
    deadline: float | None = None,
    refused: Sequence[str] = (),
    failed: UnReadFailure | None = None,
) -> tuple[WarningModel, ...]:
    """Run `refuse_unserved_empty_probe` over `requests` as `prewrite_definition` runs it for a
    result whose probe was empty, after the UN front door refused `refused` and the UN read
    `failed`. Returns the `un_system_unavailable` warnings of a check that refuses nothing."""
    availability = _scope() if availability is None else availability
    with recording_un_blocked_places() as blocked:
        note_un_blocked_places(refused)
        if failed is not None:
            record = un_unavailability_record()
            assert record is not None
            record.add_failure(failed)
        refuse_unserved_empty_probe(
            blocked,
            requests,
            graph,
            availability=availability,
            deadline=deadline,
        )
    return un_unavailable_warnings(blocked)


class TestEmptyProbe:
    def test_an_empty_probe_under_withheld_provenances_refuses_excluded_by_registry(self):
        graph = _WithheldGraph(withheld=["dc/base/UN_SDG"])

        with pytest.raises(ExcludedByRegistry) as refused:
            _check_empty_probe(graph, _request("dcid:B", entities=(_NGA, _UGA)))

        assert refused.value.code == "excluded_by_registry"
        assert refused.value.status == 422
        assert refused.value.variables == ("B",)
        assert refused.value.provenances == ("dc/base/UN_SDG",)
        assert "dc/base/UN_SDG" in str(refused.value)

    def test_the_re_read_covers_the_member_under_the_deadline_and_the_byte_cap(self):
        graph = _WithheldGraph(withheld=["dc/base/UN_SDG"])

        with pytest.raises(ExcludedByRegistry):
            _check_empty_probe(graph, _request("dcid:B", entities=(_NGA, _UGA)), deadline=7.0)

        assert graph.calls == [(("B",), (_NGA, _UGA))]
        assert graph.limits == [(7.0, WITHHELD_READ_MAX_BYTES)]

    def test_an_empty_group_re_reads_every_subject_only_member_once_per_list_of_entities(self):
        graph = _WithheldGraph()
        role_bearing = _request("dcid:A", roles=("provider", "recipient"))
        elsewhere = _request("dcid:D", entities=(_UGA,))

        _check_empty_probe(graph, role_bearing, _request("dcid:B"), elsewhere, _request("dcid:C"))

        assert graph.calls == [(("B", "C"), (_KEN, _NGA)), (("D",), (_UGA,))]

    def test_an_empty_group_refuses_excluded_by_registry_for_a_later_member(self):
        graph = _WithheldGraph(withheld=["dc/base/UN_SDG"], of=["C"])

        with pytest.raises(ExcludedByRegistry) as refused:
            _check_empty_probe(graph, _request("dcid:B"), _request("dcid:C"))

        assert (refused.value.code, refused.value.status) == ("excluded_by_registry", 422)
        assert refused.value.variables == ("B", "C")
        assert refused.value.provenances == ("dc/base/UN_SDG",)
        assert graph.calls == [(("B", "C"), (_KEN, _NGA))]

    def test_an_empty_group_names_the_members_of_the_read_that_found_withheld_data(self):
        graph = _WithheldGraph(withheld=["dc/base/UN_SDG"], at=[_UGA])

        with pytest.raises(ExcludedByRegistry) as refused:
            _check_empty_probe(graph, _request("dcid:B"), _request("dcid:C", entities=(_UGA,)))

        assert refused.value.variables == ("C",)
        assert graph.calls == [(("B",), (_KEN, _NGA)), (("C",), (_UGA,))]

    def test_empty_members_binding_no_place_read_their_scopes_and_share_a_re_read(self):
        graph = _ScopedWithheldGraph(withheld=["dc/base/UN_SDG"], of=["C"])
        availability = _scope(("B", _KEN), ("C", _KEN), ("D", _NGA))

        with pytest.raises(ExcludedByRegistry) as refused:
            _check_empty_probe(
                graph,
                _unbound("dcid:D"),
                _unbound("dcid:B"),
                _unbound("dcid:C"),
                availability=availability,
                deadline=time.monotonic() + 600,
            )

        assert refused.value.variables == ("B", "C")
        assert graph.calls == [(("D",), (_NGA,)), (("B", "C"), (_KEN,))]

    def test_an_empty_probe_with_nothing_withheld_is_not_refused(self):
        graph = _WithheldGraph()

        assert _check_empty_probe(graph, _request()) == ()

        assert len(graph.calls) == 1

    @pytest.mark.parametrize(
        ("failure", "problem"),
        [
            (GraphReadError("503", upstream_status=503), (502, "upstream_error", True)),
            (GraphDeadlineExceededError("late"), (504, "timeout", True)),
            (
                GraphCapExceededError("cap", cap_kind="bytes", cap=1, observed=2),
                (502, "upstream_error", True),
            ),
        ],
        ids=["read", "deadline", "byte_cap"],
    )
    def test_a_failed_re_read_refuses_as_a_retryable_upstream_failure(
        self, failure: Exception, problem: tuple[int, str, bool]
    ):
        graph = _WithheldGraph(withheld=["dc/base/X"], fails=failure)

        with pytest.raises(GraphReadError) as refused:
            _check_empty_probe(graph, _request(), deadline=1.0)

        assert refused.value is failure or refused.value.__cause__ is failure
        spec = classify_mint_exception(refused.value)
        assert spec is not None
        assert (spec.status, spec.code, spec.retryable) == problem

    def test_a_failed_re_read_of_a_later_list_of_entities_refuses(self):
        class _FailsAtUganda(_WithheldGraph):
            def withheld_at(self, variables, entities, **limits):
                found = super().withheld_at(variables, entities, **limits)
                if _UGA in entities:
                    raise GraphReadError("503", upstream_status=503)
                return found

        graph = _FailsAtUganda()

        with pytest.raises(GraphReadError):
            _check_empty_probe(graph, _request("dcid:B"), _request("dcid:C", entities=(_UGA,)))

        assert len(graph.calls) == 2

    def test_a_role_bearing_request_reads_nothing_and_is_not_refused(self):
        graph = _WithheldGraph(withheld=["dc/base/X"])

        assert _check_empty_probe(graph, _request(roles=("provider", "recipient"))) == ()

        assert graph.calls == []

    def test_a_un_member_at_refused_places_refuses_upstream_error_naming_them(self):
        graph = _WithheldGraph(withheld=["undata/p/X"])
        request = _request(f"dcid:{_UN_VARIABLE}", entities=(_KEN, _NGA))

        with pytest.raises(UnSystemUnavailable) as refused:
            _check_empty_probe(graph, request, refused=[_NGA, _UGA])

        assert refused.value.places == (_NGA,)
        assert (refused.value.code, refused.value.status, refused.value.retryable) == (
            "upstream_error",
            502,
            True,
        )
        assert _NGA in str(refused.value) and _UN_VARIABLE in str(refused.value)
        assert graph.calls == []

    def test_a_un_member_whose_read_failed_alongside_ours_refuses_upstream_error(self):
        request = _request(f"dcid:{_UN_VARIABLE}", entities=(_KEN, _NGA))
        failed = UnReadFailure(variables=(_UN_VARIABLE,), places=(_KEN, _NGA, _UGA), detail="503")

        with pytest.raises(UnSystemUnavailable) as refused:
            _check_empty_probe(_WithheldGraph(), request, failed=failed)

        assert refused.value.places == (_KEN, _NGA)

    def test_an_empty_group_refuses_upstream_error_for_its_un_member(self):
        # The UN member comes second and is checked before any member's registry re-read.
        graph = _WithheldGraph(withheld=["dc/base/X"])
        un_member = _request(f"dcid:{_UN_VARIABLE}", entities=(_KEN, _NGA))

        with pytest.raises(UnSystemUnavailable) as refused:
            _check_empty_probe(graph, _request(), un_member, refused=[_NGA])

        assert refused.value.variable == _UN_VARIABLE
        assert graph.calls == []

    def test_a_un_member_binding_no_place_refuses_upstream_error_when_its_reads_failed(self):
        graph = _ScopedWithheldGraph(un_places=[_KEN, _NGA])
        request = _unbound(f"dcid:{_UN_VARIABLE}")
        failed = UnReadFailure(
            variables=(_UN_VARIABLE,), places=(_KEN, _NGA, "Earth"), detail="503"
        )

        with pytest.raises(UnSystemUnavailable) as refused:
            _check_empty_probe(graph, request, failed=failed)

        assert refused.value.places == (_KEN, _NGA, "Earth")
        assert (refused.value.code, refused.value.status, refused.value.retryable) == (
            "upstream_error",
            502,
            True,
        )
        assert graph.calls == []

    def test_a_un_member_binding_no_place_is_not_refused_for_places_the_front_door_refuses(self):
        # Every country is in its scope, so the places the UN front door refuses on every read
        # are too. A retry would meet the same refusal, so the empty probe stands and the
        # warning names them, as it does on a request that served rows.
        graph = _ScopedWithheldGraph(un_places=[_KEN, _NGA])

        (warning,) = _check_empty_probe(graph, _unbound(f"dcid:{_UN_VARIABLE}"), refused=[_NGA])

        assert warning.code == "un_system_unavailable"
        assert warning.message.endswith(f"No UN data was read for {_NGA}.")

    def test_a_member_binding_no_place_under_withheld_provenances_refuses_excluded(self):
        graph = _ScopedWithheldGraph(withheld=["dc/base/UN_SDG"])
        # The mixer holds data for B at Kenya and Nigeria, all of it withheld.
        availability = _scope(("B", _KEN), ("B", _NGA))
        deadline = time.monotonic() + 600

        with pytest.raises(ExcludedByRegistry) as refused:
            _check_empty_probe(
                graph, _unbound("dcid:B"), availability=availability, deadline=deadline
            )

        assert refused.value.variables == ("B",)
        assert refused.value.provenances == ("dc/base/UN_SDG",)
        # The re-read covers every place availability names for the member.
        assert graph.calls == [(("B",), (_KEN, _NGA))]
        assert graph.limits == [(deadline, WITHHELD_READ_MAX_BYTES)]
        assert graph.scope_reads == []

    def test_a_member_whose_scope_falls_back_to_countries_re_reads_every_country_and_earth(self):
        # Availability lists too many places, so the scope is B's countries with admitted data,
        # which is none: all its data is withheld. The re-read covers Earth's countries and
        # Earth, every place that fallback could hold.
        graph = _ScopedWithheldGraph(withheld=["dc/base/UN_SDG"], countries=[_KEN, _NGA])
        capped = SdmxCapExceededError("too many places", cap_kind="values", cap=1, observed=2)

        with pytest.raises(ExcludedByRegistry) as refused:
            _check_empty_probe(graph, _unbound("dcid:B"), availability=_FailingAvailability(capped))

        assert refused.value.variables == ("B",)
        assert graph.calls == [(("B",), (_KEN, _NGA, "Earth"))]

    def test_a_memoised_countries_scope_still_re_reads_every_country_and_earth(self):
        # The first check settles B's countries scope. The second takes it from the memo,
        # which carries no countries read, and re-reads Earth's countries all the same.
        graph = _ScopedWithheldGraph(
            withheld=["dc/base/UN_SDG"], countries=[_KEN, _NGA], held=[("B", _KEN)]
        )
        capped = SdmxCapExceededError("too many places", cap_kind="values", cap=1, observed=2)

        for _attempt in range(2):
            with pytest.raises(ExcludedByRegistry):
                _check_empty_probe(
                    graph, _unbound("dcid:B"), availability=_FailingAvailability(capped)
                )

        assert graph.calls == [(("B",), (_KEN, _NGA, "Earth"))] * 2

    def test_a_member_binding_no_place_with_nothing_withheld_is_not_refused(self):
        graph = _ScopedWithheldGraph(un_places=[_KEN, _NGA])
        availability = _scope(("Count_Person", _KEN))

        assert _check_empty_probe(graph, _unbound(), availability=availability) == ()

        assert graph.calls == [(("Count_Person",), (_KEN,))]

    def test_a_member_binding_no_place_refuses_upstream_error_when_its_scope_read_fails(self):
        failure = SdmxTransportError("availability read failed")
        graph = _ScopedWithheldGraph(withheld=["dc/base/X"])

        with pytest.raises(PlaceScopeReadError) as refused:
            _check_empty_probe(graph, _unbound(), availability=_FailingAvailability(failure))

        assert refused.value.__cause__ is failure
        problem = classify_mint_exception(refused.value)
        assert problem is not None
        assert (problem.status, problem.code, problem.retryable) == (502, "upstream_error", True)
        assert graph.calls == []

    def test_a_un_member_binding_no_place_whose_reads_failed_needs_no_scope_read(self):
        graph = _ScopedWithheldGraph(un_fails=SdmxTransportError("UN availability read failed"))
        request = _unbound(f"dcid:{_UN_VARIABLE}")
        failed = UnReadFailure(variables=(_UN_VARIABLE,), places=(_KEN, "Earth"), detail="503")

        with pytest.raises(UnSystemUnavailable) as refused:
            _check_empty_probe(graph, request, failed=failed)

        assert refused.value.places == (_KEN, "Earth")
        assert graph.scope_reads == []

    def test_a_non_un_member_is_not_blamed_on_refused_places(self):
        (warning,) = _check_empty_probe(_WithheldGraph(), _request(), refused=[_KEN])

        assert warning.code == "un_system_unavailable"

    def test_places_the_un_front_door_refused_become_one_warning(self):
        with recording_un_blocked_places() as blocked:
            note_un_blocked_places([_KEN, _NGA])
            note_un_blocked_places([_KEN])

        (warning,) = un_unavailable_warnings(blocked)
        assert warning.code == "un_system_unavailable"
        assert warning.message.endswith(f"No UN data was read for {_KEN}, {_NGA}.")

    def test_no_refused_place_warns_nothing(self):
        with recording_un_blocked_places() as blocked:
            pass

        assert un_unavailable_warnings(blocked) == ()

    def test_a_member_warning_names_only_its_own_refused_places(self):
        with recording_un_blocked_places() as blocked:
            note_un_blocked_places([_KEN, _UGA])

        un_member = _request(f"dcid:{_UN_VARIABLE}", entities=(_KEN, _NGA))
        (warning,) = un_unavailable_warnings(blocked, request=un_member)
        assert warning.message.endswith(f"No UN data was read for {_KEN}.")
        assert un_unavailable_warnings(blocked, request=_request()) == ()
        # A member binding no place covers every country, so every refused place is its own.
        (unbound_warning,) = un_unavailable_warnings(
            blocked, request=_unbound(f"dcid:{_UN_VARIABLE}")
        )
        assert unbound_warning.message.endswith(f"No UN data was read for {_KEN}, {_UGA}.")


class _WithholdingGraph(FakeGraphReader):
    """A graph holding no admitted facet, whose unfiltered re-read finds `dc/base/UN_SDG` at
    every entity."""

    def withheld_at(
        self,
        variables: Sequence[str],
        entities: Sequence[str],
        *,
        deadline: float | None = None,
        max_bytes: int | None = None,
    ) -> Mapping[str, tuple[str, ...]]:
        return dict.fromkeys(entities, ("dc/base/UN_SDG",))


def _sdmx_never_read() -> SdmxClient:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"an empty probe fetched observations: {request.url}")

    return SdmxClient(base_url="https://sdmx.example.org", transport=httpx.MockTransport(refuse))


def test_a_prewrite_refuses_an_empty_probe_under_withheld_provenances():
    with pytest.raises(ExcludedByRegistry) as refused:
        prewrite([_request()], graph=_WithholdingGraph(), sdmx=_sdmx_never_read())

    assert (refused.value.code, refused.value.status) == ("excluded_by_registry", 422)
    assert refused.value.variables == ("Count_Person",)
    assert refused.value.provenances == ("dc/base/UN_SDG",)


def test_a_prewrite_serves_an_empty_probe_with_nothing_withheld_as_the_empty_result():
    minted = prewrite([_request()], graph=FakeGraphReader(), sdmx=_sdmx_never_read())

    assert minted.result.probe_empty
    assert minted.manifest.coverage.rows == 0
    assert [w.code for w in minted.manifest.warnings if w.severity == "warn"] == ["empty_result"]
    assert minted.manifest.citation.sources == []


def test_a_prewrite_whose_withheld_re_read_fails_refuses_and_returns_no_manifest():
    class _FailingReRead(FakeGraphReader):
        def withheld_at(self, variables: Sequence[str], entities: Sequence[str], **_limits):
            raise GraphReadError("503", upstream_status=503)

    with pytest.raises(GraphReadError) as refused:
        prewrite([_request()], graph=_FailingReRead(), sdmx=_sdmx_never_read())

    spec = classify_mint_exception(refused.value)
    assert spec is not None
    assert (spec.status, spec.code, spec.retryable) == (502, "upstream_error", True)


def _selection_check(
    graph: object, request: MaterialisationRequest, *, refused: Sequence[str] = ()
) -> None:
    """Run `refuse_unserved_selection` for `request` as `prewrite_definition` runs it for a
    member's `default_facet_undefined` refusal, after the UN front door refused `refused`."""
    with recording_un_blocked_places() as blocked:
        note_un_blocked_places(refused)
        refuse_unserved_selection(blocked, request, graph, availability=_scope(), deadline=None)


class TestUndefinedFacetSelection:
    def test_a_un_member_at_a_refused_place_refuses_upstream_error(self):
        graph = _WithheldGraph(withheld=["undata/p/X"])
        request = _request(f"dcid:{_UN_VARIABLE}")

        with pytest.raises(UnSystemUnavailable) as refused:
            _selection_check(graph, request, refused=[_NGA])

        assert refused.value.places == (_NGA,)
        assert graph.calls == []

    def test_withheld_data_at_its_places_refuses_excluded_by_registry(self):
        graph = _WithheldGraph(withheld=["dc/base/UN_SDG"], at=[_NGA])

        with pytest.raises(ExcludedByRegistry) as refused:
            _selection_check(graph, _request())

        assert refused.value.variables == ("Count_Person",)
        assert refused.value.provenances == ("dc/base/UN_SDG",)
        assert graph.limits == [(None, WITHHELD_READ_MAX_BYTES)]

    def test_nothing_withheld_leaves_the_refusal_standing(self):
        graph = _WithheldGraph()

        _selection_check(graph, _request())

        assert graph.calls == [(("Count_Person",), (_KEN, _NGA))]

    @pytest.mark.parametrize(
        "failure",
        [
            GraphDeadlineExceededError("deadline"),
            GraphCapExceededError("cap", cap_kind="bytes", cap=1, observed=2),
        ],
    )
    def test_a_failed_re_read_leaves_the_refusal_standing(self, failure: Exception):
        graph = _WithheldGraph(withheld=["dc/base/X"], fails=failure)

        _selection_check(graph, _request())

        assert len(graph.calls) == 1

    def test_a_role_bearing_member_reads_nothing(self):
        graph = _WithheldGraph(withheld=["dc/base/X"])

        _selection_check(graph, _request(roles=("provider", "recipient")), refused=[_KEN])

        assert graph.calls == []


class _CandidateGraph(FakeGraphReader):
    """Holds one facet for `variable` at Kenya and none at Nigeria. Its facet read records
    `refused` as places the UN front door refused, and its unfiltered re-read finds
    `dc/base/UN_SDG` at the entities in `withheld`."""

    def __init__(
        self,
        variable: str,
        facet: Facet,
        *,
        refused: Sequence[str] = (),
        withheld: Sequence[str] = (),
    ) -> None:
        super().__init__(facets={(variable, _KEN): [facet]})
        self._refused = tuple(refused)
        self._withheld = frozenset(withheld)

    def variable_facets_batch(self, **kwargs):
        note_un_blocked_places(self._refused)
        return super().variable_facets_batch(**kwargs)

    def withheld_at(
        self, variables: Sequence[str], entities: Sequence[str], **_limits: object
    ) -> Mapping[str, tuple[str, ...]]:
        return {entity: ("dc/base/UN_SDG",) for entity in entities if entity in self._withheld}


def _candidate(provenance: str | None) -> Facet:
    return Facet(
        facet_id="facet1",
        provenance_id=provenance,
        measurement_method="Amount",
        unit="USD",
        obs_count=3,
        earliest_date="2000",
        latest_date="2002",
    )


# A member holding a candidate whose selection refuses `default_facet_undefined`: by a bound
# provenance no candidate carries, or by a winning candidate with no provenance.
_UNDEFINED_SELECTIONS = {
    "bound_axis_nobody_serves": (
        _candidate("OECD_DAC2A"),
        {"provenance": FacetBinding(state="bound", ref="NO_SUCH_PROVENANCE")},
    ),
    "no_provenance": (_candidate(None), {}),
}


def _undefined_selection_mint(selection: str, variable: str, **world: Sequence[str]):
    facet, bindings = _UNDEFINED_SELECTIONS[selection]
    request = MaterialisationRequest(
        stat_var=f"dcid:{variable}",
        role_bindings={"observationAbout": [_KEN, _NGA]},
        declared_roles=("observationAbout",),
        facet_bindings=bindings,
        window=RequestedWindow(kind="all"),
    )
    graph = _CandidateGraph(variable, facet, **world)
    return prewrite([request], graph=graph, sdmx=_sdmx_never_read())


@pytest.mark.parametrize("selection", list(_UNDEFINED_SELECTIONS))
class TestAPrewriteWhoseSelectionIsUndefined:
    def test_it_refuses_default_facet_undefined(self, selection: str):
        with pytest.raises(DefaultFacetUndefinedError) as refused:
            _undefined_selection_mint(selection, "Count_Person")

        assert (refused.value.code, refused.value.status) == ("default_facet_undefined", 422)

    def test_a_un_member_at_a_refused_place_refuses_upstream_error(self, selection: str):
        with pytest.raises(UnSystemUnavailable) as refused:
            _undefined_selection_mint(selection, _UN_VARIABLE, refused=[_NGA])

        assert refused.value.places == (_NGA,)
        assert (refused.value.code, refused.value.status, refused.value.retryable) == (
            "upstream_error",
            502,
            True,
        )

    def test_withheld_data_at_a_bound_place_refuses_excluded_by_registry(self, selection: str):
        with pytest.raises(ExcludedByRegistry) as refused:
            _undefined_selection_mint(selection, "Count_Person", withheld=[_NGA])

        assert (refused.value.code, refused.value.status) == ("excluded_by_registry", 422)
        assert refused.value.variables == ("Count_Person",)
        assert refused.value.provenances == ("dc/base/UN_SDG",)


def test_the_mint_door_serves_the_un_refusal_as_a_retryable_upstream_error():
    spec = classify_mint_exception(UnSystemUnavailable(variable=_UN_VARIABLE, places=(_KEN,)))

    assert spec is not None
    assert (spec.status, spec.code, spec.retryable) == (502, "upstream_error", True)
    assert spec.detail is not None and _KEN in spec.detail


def test_the_mint_and_definition_doors_serve_the_refusal_as_422():
    def refusal(*, strip_pointer: bool) -> ExcludedByRegistry:
        exc = ExcludedByRegistry(variables=("Count_Person",), provenances=("dc/base/UN_SDG",))
        exc.strip_pointer = strip_pointer
        return exc

    for spec in (
        classify_mint_exception(refusal(strip_pointer=False)),
        classify_definition_exception(refusal(strip_pointer=True)),
    ):
        assert spec is not None
        assert (spec.status, spec.code, spec.retryable) == (422, "excluded_by_registry", False)
        assert spec.detail is not None and "dc/base/UN_SDG" in spec.detail


def test_the_live_origin_client_implements_the_re_read():
    # `refuse_unserved_empty_probe` and `withheld_warning` read nothing from a graph without
    # the re-read, so a live client missing it would drop every `excluded_by_registry` and
    # `withheld_by_registry`.
    client = build_graph_client(
        base_url="https://ours.example.org/core/api",
        token=None,
    )
    try:
        assert isinstance(client, WithheldReader)
    finally:
        client.close()


_CRS = "provenance/OECDCreditorReportingSystem"
_WDI = "dc/base/WorldDevelopmentIndicators"
_BASE_SDG_PROVENANCE = "dc/base/UN_SDG"
_UN_WHO = "undata/p/WHO"
_BASE_SDG_VARIABLE = "sdg/SH_DYN_MORT"
_UN_EXCLUDED_VARIABLE = "undata/who/ODA_1_3"


def _facets_by_provenance(request: httpx.Request) -> httpx.Response:
    """Each deployment's observation read: every named variable holds one facet at every
    named entity, under WHO on the UN deployment and under the base SDG import on ours."""
    if request.method == "POST":
        payload = json.loads(request.content)
        variables, entities = payload["variable"]["dcids"], payload["entity"]["dcids"]
        provenance = _BASE_SDG_PROVENANCE
    else:
        variables = request.url.params.get_list("variable.dcids")
        entities = request.url.params.get_list("entity.dcids")
        provenance = _UN_WHO
    block = {"orderedFacets": [{"facetId": "f", "obsCount": 1}]}
    return httpx.Response(
        200,
        json={
            "byVariable": {v: {"byEntity": dict.fromkeys(entities, block)} for v in variables},
            "facets": {"f": {"provenanceId": provenance}},
        },
    )


def test_a_mint_naming_a_superseded_variable_refuses_excluded_by_registry(bind_catalogue):
    # A variable only the base provenance superseded by the UN System holds. A variable of an
    # excluded UN series never reaches this re-read: the build holds no node for it, so the
    # door refuses `member_not_resolvable` before materialising
    # (`test_the_mint_door_refuses_a_superseded_or_excluded_variable`).
    variable, holder = _BASE_SDG_VARIABLE, _BASE_SDG_PROVENANCE
    bind_catalogue(
        restrictive_catalogue(
            provenances=[
                provenance_row(_CRS, DFO),
                provenance_row(_WDI, BASE),
                provenance_row(_UN_WHO, UN_SYSTEM, segment="who"),
            ],
            withheld_variables=[_BASE_SDG_VARIABLE],
            un_mirror=un_mirror(
                {_UN_VARIABLE: un_variable("undata/who/SERIES-MORT_U5_RATE", name="Mortality")},
                excluded={_UN_EXCLUDED_VARIABLE: "undata/who/SERIES-ODA_1_3"},
                superseded_by=_CRS,
            ),
        )
    )
    client = build_graph_client(
        base_url="https://ours.example.org/core/api",
        token=None,
        un_base_url="https://un.example.org/core/api",
        transport=httpx.MockTransport(_facets_by_provenance),
    )
    with client:
        # The control: the same reads serve the admitted UN variable.
        served = client.observation_facets_batch([_UN_VARIABLE], [_KEN])
        assert [facet.provenance_id for facet in served[_UN_VARIABLE][_KEN]] == [_UN_WHO]

        # The mint's facet probe finds no candidate, so the empty probe is checked.
        assert client.observation_facets_batch([variable], [_KEN]) == {variable: {_KEN: []}}
        with pytest.raises(ExcludedByRegistry) as refused:
            _check_empty_probe(client, _request(f"dcid:{variable}", entities=(_KEN,)))

    assert refused.value.variables == (variable,)
    assert refused.value.provenances == (holder,)


_STAT_VAR_ARCS = {"typeOf": ["StatisticalVariable"], "measuredProperty": ["value"]}


def _both_deployments(request: httpx.Request) -> httpx.Response:
    """Each deployment answers a node read with a StatisticalVariable's arcs for the two
    variables these mints name, and an observation read as `_facets_by_provenance` does."""
    if request.url.path.endswith("/v2/node"):
        nodes = json.loads(request.content)["nodes"]
        named = {_BASE_SDG_VARIABLE, _UN_EXCLUDED_VARIABLE}
        arcs = {
            arc: {"nodes": [{"dcid": v} for v in values]} for arc, values in _STAT_VAR_ARCS.items()
        }
        return httpx.Response(
            200, json={"data": {node: {"arcs": arcs} if node in named else {} for node in nodes}}
        )
    return _facets_by_provenance(request)


@pytest.mark.parametrize(
    ("variable", "code"),
    [
        # Our mixer holds it only under the base provenance the UN System supersedes.
        (_BASE_SDG_VARIABLE, "excluded_by_registry"),
        # A definition of a variable of an excluded series, minted before the exclusion: the
        # loaded build holds no node for it, so the door cannot resolve the member.
        (_UN_EXCLUDED_VARIABLE, "member_not_resolvable"),
    ],
)
def test_the_mint_door_refuses_a_superseded_or_excluded_variable(variable: str, code: str):
    from spec_registry import LiveSpecRegistry
    from spec_registry.reader import LiveSpecReader
    from tests.datasets_api import test_un_doors as doors
    from tests.datasets_api._app_factories import make_client
    from tests.datasets_api.conftest import FakeDatasetStore
    from tests.origin_catalogue.fake_catalogue import fake_catalogue_service

    snapshot = restrictive_catalogue(
        provenances=[
            provenance_row(_CRS, DFO),
            provenance_row(_WDI, BASE),
            provenance_row(_UN_WHO, UN_SYSTEM, segment="who"),
        ],
        withheld_variables=[_BASE_SDG_VARIABLE],
        un_mirror=un_mirror(
            {_UN_VARIABLE: un_variable("undata/who/SERIES-MORT_U5_RATE", name="Mortality")},
            excluded={_UN_EXCLUDED_VARIABLE: "undata/who/SERIES-ODA_1_3"},
            superseded_by=_CRS,
        ),
    )
    graph = build_graph_client(
        base_url="https://ours.example.org/core/api",
        token=None,
        un_base_url="https://un.example.org/core/api",
        transport=httpx.MockTransport(_both_deployments),
        max_attempts=1,
    )
    store = FakeDatasetStore()
    with graph:
        # The definition is built under the default catalogue, which admits everything.
        sidecar = doors._sidecar(graph, variable, (_KEN,))
        client = make_client(
            registry=LiveSpecRegistry(store=store),
            spec_reader=LiveSpecReader(store=store),
            graph=graph,
            sdmx=doors._sdmx_never_called(),
            artifact_store=doors.FakeArtifactStore(),
            snapshot_reader=doors.FakeSnapshotReader(),
            reference_writer=doors.FakeArtifactReferenceWriter(),
            coverage=doors._coverage(),
            catalogue=fake_catalogue_service(snapshot),
        )
        resp = client.post(
            "/api/datasets",
            json={
                "schema_version": 1,
                "input": {"kind": "spec_id", "spec_id": sidecar["spec_id"], "definition": sidecar},
                "options": {"bundle": False},
            },
        )

    assert resp.status_code == 422, resp.text
    assert (resp.json()["code"], resp.json()["retryable"]) == (code, False)
