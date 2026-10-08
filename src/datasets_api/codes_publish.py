"""Refusal codes owned by the publish and recipe doors."""

from __future__ import annotations

PUBLISH_DOOR_CODE_CATALOGUE: dict[str, tuple[str, str]] = {
    "publish_conflict": (
        "Publish Conflict",
        "A publish call names an identity pair already held, and what is held differs "
        "from what the call carries, either in the frame's bytes or in its recorded "
        "derivation.",
    ),
    "publish_not_permitted": (
        "Publish Not Permitted",
        "The authenticated caller lacks the publish permission, or programmatic "
        "publishing is not live on this deployment.",
    ),
    "recipe_not_found": (
        "Recipe Not Found",
        "The named frame holds no recipe, or no frame is stored under the named replay key.",
    ),
}
