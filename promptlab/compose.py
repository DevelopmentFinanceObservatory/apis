"""Compose stage blocks byte-for-byte with shipped prompts.

Strip each block's trailing newlines, join with `\n\n` or `\n`, then add one
trailing newline.
"""

from __future__ import annotations

from pathlib import Path

from promptlab.manifest import PROMPTLAB_ROOT, ManifestError, StageSpec, block_path


def compose(stage: StageSpec, *, root: Path = PROMPTLAB_ROOT) -> str:
    """Read and concatenate the stage's block files. Raises `ManifestError` for a
    missing block file."""
    pieces: list[str] = []
    for i, block in enumerate(stage.blocks):
        path = block_path(stage.stage, block.id, root)
        if not path.is_file():
            raise ManifestError(f"unknown block file: {block.id!r} ({path})")
        text = path.read_text().rstrip("\n")
        if i > 0:
            pieces.append("\n\n" if block.sep == "blank" else "\n")
        pieces.append(text)
    return "".join(pieces) + "\n"
