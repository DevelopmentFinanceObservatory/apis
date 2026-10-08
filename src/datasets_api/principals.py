"""Compare a presented principal with a configured one.

The publish, recipe, and live-refresh routes each admit callers by the value of one principal
header. `same_principal` is the comparison they share. It runs in time independent of where two
values of one length first differ, so a response time does not reveal a prefix of a configured
principal.
"""

from __future__ import annotations

import hmac
from collections.abc import Iterable


def same_principal(presented: str, configured: str) -> bool:
    """Whether `presented` is `configured`, compared in constant time. An empty `configured`
    matches nothing."""
    return bool(configured) and hmac.compare_digest(presented.encode(), configured.encode())


def principal_allowed(presented: str, allowed: Iterable[str]) -> bool:
    """Whether `presented` is one of `allowed`. Every entry is compared, so the time does not
    depend on which entry matched."""
    matched = False
    for configured in allowed:
        # No short circuit, so a match does not end the loop early.
        matched |= same_principal(presented, configured)
    return matched


__all__ = ["principal_allowed", "same_principal"]
