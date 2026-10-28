# dfo-analysis-engine

`dfo-analysis-engine` computes the nine analysis verbs (`aggregate`, `compare`, `filter`, `growth`, `join`, `quotient_of`, `rank`, `share_of`, `sum_of`) over `dfo_frame_grammar.Frame` objects. It has no store, no network and no web framework.

## Modules

| Module                                        | Responsibility                                                                                                |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `engine`                                      | One `run_<verb>` function per verb, over Arrow tables, and the helpers they share.                            |
| `op_spec`                                     | The pydantic op-spec of each verb.                                                                            |
| `identity`, `execution_identity`              | The `ops_` op-spec hash, the `ocv_` content version, and the engine version they bind to.                     |
| `artifact`, `result`, `steps`, `units`        | The result, step and unit shapes a verb reports.                                                              |
| `rolling`, `recipe`                           | Rolling populations and the anchor key; a recipe's normal form, `normal_form_hash` and `replay_key`.          |
| `place_type`, `time_grain`, `unit_refs`, `price_basis`, `hashed_fields`, `frame_ref` | Small pure helpers the modules above share, and `FrameRef`. |

The tests under `tests/` pin every `ops_`, `ocv_` and op-spec hash against recorded vectors, and replay a recorded set of engine cases.
