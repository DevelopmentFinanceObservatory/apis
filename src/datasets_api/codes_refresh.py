"""Refusal codes owned by the frame refresh door. It reuses `codes_core`'s `spec_not_found`,
`codes_snapshot`'s `snapshot_not_found`, and the platform's `refresh_not_permitted`, which the
Presentation render routes serve too."""

from __future__ import annotations

REFRESH_DOOR_CODE_CATALOGUE: dict[str, tuple[str, str]] = {
    "frame_not_refreshable": (
        "Frame Not Refreshable",
        "The named frame is a derived frame. Only a dataset frame has a current rendering "
        "to refresh to.",
    ),
    "rendering_not_recorded": (
        "Rendering Not Recorded",
        "The named frame is not one rendering of the current data. It is a derived version "
        "and no file linked under it records the groupby or party-totals cut it was minted "
        "with, or its version is recorded under several renderings, which were one frame when "
        "recorded, and the current data renders them as different frames, or a rebuilt "
        "dataset cannot render one of them or renders them as one frame other than the one "
        "it served.",
    ),
}
