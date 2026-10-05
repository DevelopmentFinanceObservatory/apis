# promptlab

A harness for composing and measuring variants of the search engine's
extract and bind system prompts. Not part of the installed wheel: a
development and measurement tool, not a runtime dependency of the shipped
search API.

## Layout

```
promptlab/
  blocks/
    extract/<id>.md
    bind/<id>.md
  rungs/
    <name>-extract.json
    <name>-bind.json
  manifest.py   # BlockRef, StageSpec, RunConfig, Rung, load_rung, units
  compose.py    # compose
  rung.schema.json
```

Generated rungs (ablation cells, build-upward candidates) are written to
`runs/ladder/rungs/` and are gitignored; only durable rungs -- the
production prompts and named milestones -- live under `promptlab/rungs/`.

## Block files

A block file at `promptlab/blocks/<stage>/<id>.md` holds one fragment of a
system prompt verbatim: no front matter, no headings the prompt itself does
not have, and exactly one trailing newline. `<id>` is `<NN>-<slug>`, where
`NN` is the block's zero-based position in the stage's production prompt.

## Rung manifests

A rung is named by two sibling JSON files validated against
`rung.schema.json`: `<name>-extract.json` and `<name>-bind.json`, one per
stage. Each file lists that stage's blocks in order:

```json
{
  "name": "example",
  "description": "One sentence describing this stage's composition.",
  "stage": "bind",
  "blocks": [
    { "id": "00-role", "core": true },
    { "id": "01-shape-definition" },
    { "id": "02-part1-choose", "sep": "line", "unit": "choose-and-bind" }
  ],
  "run": {
    "capture": "k8-2026-08-31",
    "top_k": 8,
    "seeds": [1, 2, 3],
    "model": "gemini-3.5-flash-lite",
    "thinking": { "extract": null, "bind": "minimal" }
  }
}
```

- `sep` -- the separator emitted before this block: `"blank"` (default) for
  a blank line, `"line"` for a single newline. Ignored on a stage's first
  block.
- `unit` -- the ablation unit this block belongs to; a block with no `unit`
  is its own unit. `units()` groups a stage's blocks this way.
- `core` -- excludes the block from leave-one-out ablation.
- `run` -- appears on the bind stage's file only, and applies to the whole
  rung: the capture to replay, the bind menu's top-k, the seeds to run, the
  model, and each stage's thinking level.

`load_rung(extract_path, bind_path)` reads both files, validates each
against the schema, checks that every block id they name has a matching
file on disk, and returns one `Rung` combining both stages plus the run
configuration. `Rung.name` and `Rung.description` are the bind file's own
fields.

## Composing a prompt

`compose(stage_spec)` reads a stage's block files in order and concatenates
them: each block file's trailing newline is stripped, a separator is
emitted before every block but the first, and the composed prompt carries
exactly one trailing newline. A block id with no matching file raises
`ManifestError` rather than composing a silently short prompt.
