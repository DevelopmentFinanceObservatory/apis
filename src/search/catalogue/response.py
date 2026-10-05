"""The body of `GET /api/search/catalogue`, projected from a published build.

The unscoped body carries the global counts and every example at its
default place. The place-scoped body carries the counts at one country and
the examples whose StatVar has data there, filled with its name.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from search.catalogue.artifact import CatalogueArtifact, Counts, PlaceCounts
from search.catalogue.examples import PLACE_SLOT


class CatalogueExampleSentence(BaseModel):
    """An example sentence a client offers and runs as a query."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(description="the template filled with the place's name")
    template: str = Field(description=f"the sentence with a {PLACE_SLOT} slot")
    stat_var: str = Field(description="the StatVar the sentence resolves to")
    place: str = Field(description="the country dcid the sentence is filled with")


class CatalogueResponse(BaseModel):
    """What DFO holds and example sentences to ask of it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    catalogue_version: str = Field(description="the published build this body comes from")
    counts: Counts | PlaceCounts = Field(
        description="global counts, or the counts at the requested place"
    )
    examples: list[CatalogueExampleSentence]


def catalogue_response(artifact: CatalogueArtifact) -> CatalogueResponse:
    """The unscoped body: global counts and every example at its default place."""
    return CatalogueResponse(
        catalogue_version=artifact.catalogue_version,
        counts=artifact.counts,
        examples=[
            CatalogueExampleSentence(
                text=example.text,
                template=example.template,
                stat_var=example.stat_var,
                place=example.place,
            )
            for example in artifact.examples
        ],
    )


def place_catalogue_response(artifact: CatalogueArtifact, place: str) -> CatalogueResponse | None:
    """The body scoped to `place`, or `None` when the build counts no such
    country. A country the graph gave no name for has no examples."""
    entry = artifact.places.get(place)
    if entry is None:
        return None
    name = entry.name
    offered = set(entry.examples)
    return CatalogueResponse(
        catalogue_version=artifact.catalogue_version,
        counts=entry.counts,
        examples=[]
        if name is None
        else [
            CatalogueExampleSentence(
                text=example.template.replace(PLACE_SLOT, name),
                template=example.template,
                stat_var=example.stat_var,
                place=place,
            )
            for example in artifact.examples
            if example.id in offered
        ],
    )


__all__ = [
    "CatalogueExampleSentence",
    "CatalogueResponse",
    "catalogue_response",
    "place_catalogue_response",
]
