"""The verb engines: Arrow kernels over `Frame.table`, with per-group and
per-distinct-value arithmetic in Python. No numpy, no pandas, no SQL.

`dfo_analysis_engine.engine.common` holds what every verb needs before it computes
(narrowing a frame to the caller's scope, resolving unit/facet
heterogeneity, recording exclusions). `dfo_analysis_engine.engine.columnar` holds
the Arrow kernels the verbs share (grouping, masks, output columns and
frames). `aggregate`, `compare`, `filter`, `growth`, `join`, `quotient_of`,
`rank`, `share_of`, and `sum_of` are the nine verb engines built on top of
them; `dfo_analysis_engine.engine.two_sided` holds what `share_of` and `quotient_of`
share, and `dfo_analysis_engine.engine.paired` the form of `compare` and `sum_of`
that pairs members by `on`.
"""

from __future__ import annotations
