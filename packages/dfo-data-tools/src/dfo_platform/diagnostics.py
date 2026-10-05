"""Informational warnings attached to successful (2xx) responses.

Warnings have no refusal code, status, or retryability and sit outside the problem-details body.
This vocabulary stays apart from `dfo_platform.problem`, so a change here does not move the
`PLATFORM_CAPABILITY_VERSION` fingerprint of the refusal body.
"""

from __future__ import annotations

from typing import Literal

WarningSeverity = Literal["info", "warn", "degraded"]
"""Every API surface's own Warning-shaped model constrains its `severity`
field to this literal, so every surface mints only values every other
surface already recognizes. Shared by `search.models.Warning.severity`
and `materialise.codes.MaterialisationWarning`'s `severity` key."""
