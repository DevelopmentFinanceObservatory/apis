"""Sizes, digests, the manifest and the content version of a skill bundle.

Every figure is computed from the decoded UTF-8 bytes of a file. A digest is
`sha256:` plus the lower-case hex SHA-256 of those bytes. The manifest lists
`SKILL.md` first, then each reference in path order. The content version is
`sha256:` plus the hex SHA-256 of one UTF-8 string holding, for each manifest
entry in order, its path, a tab, its digest and a newline.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from skills_index.config import SKILL_FILE_PATH
from skills_index.types import SkillFile

_DIGEST_PREFIX = "sha256:"


def _sha256(data: bytes) -> str:
    return f"{_DIGEST_PREFIX}{hashlib.sha256(data).hexdigest()}"


def skill_file(path: str, content: str) -> SkillFile:
    """Returns the bundle file serving `content` at `path`."""
    data = content.encode("utf-8")
    return SkillFile(path=path, size=len(data), digest=_sha256(data), content=content)


def build_manifest(content_text: str, references: Sequence[SkillFile]) -> tuple[SkillFile, ...]:
    """Returns the manifest: `SKILL.md` from `content_text`, then `references`, which
    arrive in path order."""
    return (skill_file(SKILL_FILE_PATH, content_text), *references)


def content_version(manifest: Sequence[SkillFile]) -> str:
    """Returns the content version of `manifest`, which moves with any file's digest or path."""
    lines = "".join(f"{entry.path}\t{entry.digest}\n" for entry in manifest)
    return _sha256(lines.encode("utf-8"))
