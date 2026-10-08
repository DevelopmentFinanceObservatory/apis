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
        "The stored records do not say which rendering the named frame is. It is a groupby "
        "collapse and no file linked under its version records the groupby, or its version "
        "links both a groupby sum and a plain file and the two renderings of the current "
        "data differ.",
    ),
}
