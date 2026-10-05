"""A resolved Spec's confirmation sentence plan: its heading and byline as tokens.

`render.heading` decides what the heading and byline state; `render.tokens` lays them
out. The flat sentence joins the plan's token text.
"""

from __future__ import annotations

from collections.abc import Mapping

from search.models import SentenceToken, Spec

from .heading import heading_for
from .tokens import plan_tokens, sentence_from_plan


def render_sentence_plan(
    spec: Spec,
    *,
    keep_how_properties: frozenset[str] = frozenset(),
    axis_declarations: Mapping[str, str] | None = None,
) -> list[SentenceToken]:
    """The tokens of a resolved Spec's heading and byline (`heading_for`, `plan_tokens`).

    ``axis_declarations`` supplies how-property names and ordering, the code index's
    (`declared_axis_names`) when None.
    ``keep_how_properties`` keeps sibling-distinguishing constraints visible.
    """
    return plan_tokens(
        heading_for(
            spec,
            axis_declarations=axis_declarations,
            keep_how_properties=keep_how_properties,
        )
    )


def render_sentence(spec: Spec) -> str:
    """Render a resolved Spec's confirmation sentence without returning its token plan."""
    return sentence_from_plan(render_sentence_plan(spec))


def plain_sentence_plan(text: str) -> list[SentenceToken]:
    """Wrap text without slot structure as one token."""
    return [SentenceToken(kind="text", text=text)]
