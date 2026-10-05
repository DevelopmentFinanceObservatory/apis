"""Presentation API routes adapt `assembler.pipeline` and its refusal types to HTTP.

`presentation_api.app.presentation_surface` composes the doors at `/api/presentation` and
wires them to `presentation_store` implementations and the baked release registry.
"""

from __future__ import annotations
