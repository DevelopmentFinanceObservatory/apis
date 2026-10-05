"""Async operation submission, dispatch, worker execution, and task authorization.

The worker calls shared serve functions after admission, so the sync budget runs once.
"""

from __future__ import annotations
