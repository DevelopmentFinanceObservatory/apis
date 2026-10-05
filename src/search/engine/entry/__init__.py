"""ResolveInput entry paths live in `raw_text`, `parsed`, and `dataset_resubmit`.
`params` holds their shared prologue values. Core dispatches each module, and
entry modules import shared params and engine services.
"""

from __future__ import annotations
