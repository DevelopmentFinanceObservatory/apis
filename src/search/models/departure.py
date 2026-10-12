from __future__ import annotations

from pydantic import Field

from .base import StrictModel
from .vocab import DEPARTURE_ORDER, DepartureCode

WHERE_REF = "where:"
"""The `binding_ref` of the `where` token, which carries the subject and the
destination role."""
ORIGIN_REF = "origin:"
"""The `binding_ref` of the token naming the origin role's entities."""


def role_ref(role: str) -> str:
    """The `binding_ref` of the token that names `role`'s entities, open or bound. The
    subject and the destination share the `where` slot, so they read `where:`; the origin
    reads `origin:`."""
    return ORIGIN_REF if role == "origin" else WHERE_REF


UNIT_REF = "unit:"
"""The `binding_ref` of the byline's unit token (axis `unit`). Its options are the Spec's
`resolved.alternative_measures` entries with axis `unit`, and a departure on the price basis
carries this ref."""

SOURCE_REF = "source:"
"""The ref of the byline's source token. The token states it whenever it has text, and
carries it as `binding_ref` only where the answer binds a source slot."""

METHOD_REF = "method:"
"""The `binding_ref` of the byline's method token (axis `method`), present only when the
part's identity binds `measurementMethod`. Its options are the Spec's `facet_options`
entry with axis `measurementMethod`."""


class Departure(StrictModel):
    """One way this part's answer differs from what its request asked, stated
    for people in `text` and for code in `code`, `asked`, and `shown`.

    Departures live on the part (`_PartBase.departures`).
    `spec.resolved.rendered_sentence` states what the answer is, and a
    departure states how that differs from the request that produced it.
    """

    code: DepartureCode = Field(
        description=(
            "what kind of departure this is. 'closest_match': the answer shows the "
            "nearest measure the graph holds, not the one asked; a client draws a "
            "muted 'Closest match:' lead before the heading, and on a candidates part, "
            "whose readings stand in for a measure with no data, once before the first "
            "reading. 'measure_not_applied': "
            "a named measure qualifier was not applied. 'operation_not_applied': an "
            "operation the request named (a share, a ranking) was not run, so the "
            "amounts behind it are shown. 'breakdown_not_applied': a breakdown the "
            "request named was not applied, so the total is shown. "
            "'entity_unavailable': an entity the request named holds no values in the "
            "series the answer reads, one departure per role; for a role the request "
            "named as a group, asked is the request's words for the group and the text "
            "counts the members without values; the text names any source holding values "
            "there that Search does not read. "
            "'year_unavailable': a named year or range falls outside the years "
            "served. 'modelled_values': the served values are modelled or estimated. "
            "'group_read_as_members': a named group was read as its member "
            "entities; on a flow role the text names the amounts the group's published "
            "total counts that no member holds. 'direction_discarded': a flow direction the request's own words "
            "named was not applied. 'scope_limited': a no_data answer's check covered "
            "only values the engine chose, which the request did not name; its text "
            "names only what the check covered ('This check covered only ODA.')"
        )
    )
    binding_ref: str | None = Field(
        default=None,
        description=(
            "the identity binding whose edit clears this departure ('when:' for a "
            "year, 'how:<property dcid>' for a measure value, 'where:' or 'origin:' "
            "for a role's entities, 'unit:' for the price basis); null for a "
            "departure no binding edit clears, such as an "
            "operation the engine does not run. A client links the departure to the "
            "sentence token carrying the same binding_ref when the plan has one; a "
            "candidates or no_data plan, and a how slot the heading leaves silent, may "
            "carry no such token"
        ),
    )
    asked: str | None = Field(
        default=None,
        description="what the request named, in its own wording; null when it named nothing",
    )
    shown: str | None = Field(
        default=None,
        description="what the answer shows instead, as the heading states it; null when nothing",
    )
    text: str = Field(
        min_length=1,
        description=(
            "the departure as one or two short sentences for people: 'You asked for "
            "2030. The data ends in 2024, so this shows 1973 to 2024.' On a no_data "
            "part no text claims the part shows anything. The envelope's composed "
            "sentence carries the same text "
            "as a segment='departure' token after its part's heading and byline, and "
            "so does the rendered_sentence of a part that is not definite"
        ),
    )


def departures_in_precedence_order(departures: list[Departure]) -> list[Departure]:
    """Returns `departures` unchanged when they follow `DepartureCode` precedence order, and
    raises `ValueError` otherwise. A model holding a departure list validates it with this."""
    ranks = [DEPARTURE_ORDER.index(departure.code) for departure in departures]
    if ranks != sorted(ranks):
        raise ValueError("departures must follow DepartureCode precedence order")
    return departures
