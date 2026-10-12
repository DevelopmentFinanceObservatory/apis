"""The time a worker claim owns an operation.

`LiveOperationStore` reads `LEASE_SECONDS` to judge a `running` row that carries no lease of its
own, so the constants sit apart from the worker, which imports the store.
"""

from __future__ import annotations

# The run's work deadline, counted from the claim time. No write starts after it.
RUN_DEADLINE_SECONDS = 1_500

# A claim's static lease is the work deadline plus 120 s for the writes that started before it,
# under the 1,800-second `DISPATCH_DEADLINE_SECONDS` after which Cloud Tasks retries the delivery.
LEASE_SECONDS = RUN_DEADLINE_SECONDS + 120

__all__ = ["LEASE_SECONDS", "RUN_DEADLINE_SECONDS"]
