"""Each Datasets API door contributes a router, classifier, and optional lifespan or admission
checks to the `Surface` built by `datasets_api.app.datasets_surface`.

A router carries full paths. Classifiers return `None` for unrecognised exceptions so the combined
classifier can try the next door. Admission checks are synchronous `Request` functions that return
a `ProblemSpec` or `None`. Each check filters requests itself. The platform runs them before
FastAPI parses request bodies.
"""

from __future__ import annotations
