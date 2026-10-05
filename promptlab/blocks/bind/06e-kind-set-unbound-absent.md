- "set": the variable names a concept spanning 2+ taxonomy members AND no single
  aggregate/rollup member covers them all. Emit kind=set, value_dcids=[<all matching>].
- "unbound": the taxonomy offers 2+ real alternatives and every member of this shape carries
  the property, but the variable says nothing specific about it and no rollup covers "all
  values". Emit kind=unbound, value_dcids=[].
- "absent": the property does not decompose the resolved series at all — simply not relevant.
  Emit kind=absent, value_dcids=[].
