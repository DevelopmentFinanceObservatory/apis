"""Scope choices: how each part was counted, the options it could take, and the identity a
selection compiles to.

A part's scope is settled on four axes (`models.vocab.ScopeAxis`): sector, region, measure
basis and provider scope. An explicit entry (the request's own words, or a switch) wins over
a carried one (a choice the conversation settled earlier), and every other axis takes its
declared default, derived after both. `compile.compile` is the one pure function from a base
identity, the entries and the axis tables to the active selection, the availability of every
option and the identity the selection compiles to. `build.build_part_scope` evaluates the
registered axis rules over explicit engine facts and wraps the result as the wire record,
`models.scope.PartScope`, on `ResolvedSpec.scope_choices`.

An axis goes live by adding values to these extension points:

- `vocabulary`: the axis's option ids (`<axis>:<value>`, the same string on every part).
- `models.scope.Entry` and `state.ScopeState`: binders read the entries on their axis from
  `ResolveState.scope_selection` (`ScopeState.entries_on`, explicit first) and write back
  one `state.AxisRecord` per axis they settle (`ScopeState.write`). A definite answer whose
  pass wrote a record states its scope choices; one with none states None.
- `rules.AXIS_RULES`: the axis's rule, which reads the base identity, the binder's record (or
  none, on a dataset_resubmit) and `tables.ScopeFacts` and returns the axis's
  `tables.AxisTable` (options with their bindings and intrinsic availability, default
  preference, the option the base holds), or None when the axis is absent from the part.
- `rules.CONFLICTS`: pairs of options of different axes that cannot hold together.
- `edits.EDITS`: the identity edit of each binding kind, `edits.AXIS_BINDING`, the binding
  kind each axis's options carry, and `edits.PUBLISHED_BINDING`, the one they may also carry
  on a part counting a published total.
- `tables.ScopeFacts`: the in-memory indexes a rule reads, passed explicitly, so compile
  reads no global and does no I/O.
- `models.vocab.ScopeReasonCode`: why an option is partial or unavailable.
- `models.scope.Component` roles: the series a compiled identity is served from.

The declared tables the rules read live in `code_index.scope_conventions`, which the
declarations hash and `rules.rule_version` cover.
"""
