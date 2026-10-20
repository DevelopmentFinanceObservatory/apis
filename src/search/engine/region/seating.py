"""Seat the bind menu by retrieval evidence.

Recall returns hits: a scored shape, or a topic root that stands for its leaves. A root's
leaves are expansions, shapes with no recall score of their own. They share one rank and
carry no order among themselves beyond the order the topic walk reached them in, so a menu
that holds some of a root's leaves holds an arbitrary sample of it, and in rank order alone
the leaves of one large root fill the menu ahead of every scored shape below them ("aid to
Bangladesh": 26 leaves of one root).

The menu never holds part of a root. The hits are walked in rank order under the seat
count:

- a scored shape takes one seat while a seat is left;
- a root takes one seat for each of its leaves, consecutive at the root's place in the
  order they came in, when all of them fit in the seats left; otherwise none of its leaves
  is seated and the walk goes on to the next hit;
- the walk ends when the seats are full or the hits run out.

In the order of the hits the scored shapes keep their relative order, and a root ranks at
its own recall score, capped at the best scored shape of the list: it stands behind every
scored shape at or above that rank, wherever those shapes stand, so a root never leads the
reading recall scored best.

The seat count is also the root-size limit: a root of more than that many leaves never
reaches the menu, so a lower count removes whole roots, not a sample of them.

Each resolver scores on its own scale (`shape.Pool`), so the walk seats our pool's hits
first and the UN System Data Commons' take the seats left, each pool ranked and capped by
its own scores. Only a pass that merged the UN pool (`phase_intake._merge_un_pool`) holds
both.

The bind menu and the not-electable list beside it are both seated this way
(`phase_rank_bind._phase_prune_and_rank`, `phase_rank_bind._refused_on_pinned`).
"""

from __future__ import annotations

from collections.abc import Sequence

from search.engine.shape import Pool, ShapeDraft


def is_expansion(shape: ShapeDraft) -> bool:
    """Whether recall reached `shape` only through a topic root."""
    return shape.representative_score is None and shape.expansion_root is not None


# The order the pools take seats in.
_POOL_ORDER: tuple[Pool, ...] = ("ours", "un_system")


def seat_menu(shapes: Sequence[ShapeDraft], seats: int) -> list[ShapeDraft]:
    """The readings of `shapes`, the admissible ones in rank order, that take the `seats` of
    a menu: at most `seats` of them, in seated order. Our pool's shapes are seated first
    (`_seat_pool`), then the UN pool's in the seats left."""
    seated: list[ShapeDraft] = []
    for pool in _POOL_ORDER:
        held = [shape for shape in shapes if shape.pool == pool]
        seated.extend(_seat_pool(held, seats - len(seated)))
    return seated


def _seat_pool(shapes: Sequence[ShapeDraft], seats: int) -> list[ShapeDraft]:
    """The shapes of one pool that take `seats` seats, in seated order. A list with no
    expansion keeps its first `seats` shapes. A list of nothing but the leaves of roots
    larger than the menu seats none.

    The ceiling on a root's rank is the best recall score among `shapes`, the readings
    still admissible when the list is seated, so a shape refused before then caps nothing.

    The scored shapes need not come in score order: a collapse can stand a family's best
    score where its earliest member stood. Whatever their order, no root stands ahead of a
    scored shape that scores at or above its rank, and the scored shapes keep their
    relative order: a root stands directly behind the last such shape, so on a list in
    score order it stands between the shapes it ranks between. Roots at one place stand in
    rank order, and at one rank in the order their first leaves came in. A shape with
    neither a score nor a root stays among the scored shapes and holds no root back.
    """
    scored = [shape for shape in shapes if not is_expansion(shape)]
    scores = [shape.representative_score for shape in scored]
    ceiling = max((score for score in scores if score is not None), default=None)
    leaves_of: dict[str | None, list[ShapeDraft]] = {}
    for shape in shapes:
        if is_expansion(shape):
            leaves_of.setdefault(shape.expansion_root, []).append(shape)

    def rank(leaves: list[ShapeDraft]) -> float:
        root = leaves[0].expansion_root_score
        root = 0.0 if root is None else root
        return root if ceiling is None else min(root, ceiling)

    # The place of each root: how many scored shapes stand ahead of it. The sort is
    # stable, so roots at one rank keep the order they came in.
    behind: dict[int, list[list[ShapeDraft]]] = {}
    for root_rank, leaves in sorted(
        ((rank(leaves), leaves) for leaves in leaves_of.values()), key=lambda hit: -hit[0]
    ):
        ahead = max(
            (i + 1 for i, score in enumerate(scores) if score is not None and score >= root_rank),
            default=0,
        )
        behind.setdefault(ahead, []).append(leaves)

    hits: list[list[ShapeDraft]] = [*behind.get(0, ())]
    for i, shape in enumerate(scored, start=1):
        hits.append([shape])
        hits.extend(behind.get(i, ()))
    seated: list[ShapeDraft] = []
    for hit in hits:
        if len(seated) == seats:
            break
        if len(hit) <= seats - len(seated):
            seated.extend(hit)
    return seated
