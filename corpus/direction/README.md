# Directional eval set

Labelled cases measuring flow-direction recovery (which named place is the donor and which is the recipient) independently of the main `corpus/*.yaml` cases. Those cases score full engine responses from a running app against the live graph and an LLM. This set scores one pure function against hand-labelled text, offline, with no model or graph call.

## Why a separate set

Flow direction is recovered structurally. The raw-text door's extraction stage fills each `ExtractedEntity.flow_role` from the query's own wording, a schema field the model fills directly. The parsed door reads the caller's own `role_text` through `search.engine.role_text.role_axis_for_text`. The only direction function in `role_text.py` is `contradicts`, a cross-check that can veto a door-supplied axis the query's own wording denies. The few two-place role-bearing cases in `corpus/*.yaml` carry explicit prepositions on both places ("aid from Germany to Kenya"), so they exercise nothing harder. This set's categories (coordinated lists, possessives, passive voice, substring collisions, non-English wording, world-knowledge bait) are the adversarial slice the main cases miss. Scoring them offline against fixed hand-labelled entities needs no graph or model call, so the result is reproducible for any pure function shaped like a direction recoverer. The frozen scanner scored below is such a function, and its numbers are the baseline for the current extraction path.

## Record format

`role_direction_cases.yaml` is a flat list of records:

- `id`: unique, `dir-<category>-NN`.
- `category`: one of `coordinated_donor_list`, `coordinated_recipient_list`, `three_plus_list`, `possessive`, `verb_no_preposition`, `passive`, `leading_unevidenced`, `bare_comparison`, `between_x_and_y`, `substring_collision`, `non_english`, `world_knowledge_bait`, `prior_contradicting`.
- `query`: the full query text.
- `entities`: one entry per named place, each `{surface, dcid, expected_axis}`. `surface` and `dcid` are recorded explicitly. A scanner matching `detect_directions`'s shape, `(query, [(surface, dcid), ...]) -> verdicts`, is a pure function of exactly these inputs, so the set scores any revision of such a scanner, including the frozen one below, which is not part of `search.engine.role_text`. `expected_axis` is `origin`, `destination`, or `unspecified`.
- `note`: why the label is what it is, and what defect (if any) the case targets.

## The labelling rule

`expected_axis` is `origin`/`destination` only when the query's own wording states the direction for that entity: a preposition, a verb whose semantics fix a direction ("funds", "received support from"), a possessive ("Taiwan's ODA to X"), an idiomatic juxtaposition ("Kenya aid from Norway" states Kenya as recipient), or passive voice. It is `unspecified` whenever the query states no direction for that entity, even where world knowledge would supply an answer: "ODA Kenya Uganda" and "aid between Norway and Kenya" are `unspecified` for every entity. This mirrors `Extraction`'s own field-description ban. A direction supplied from world knowledge instead of the query's own wording is the silent-wrong-answer class this evaluation exists to catch, so the ground truth does not reward it.

`world_knowledge_bait` and `prior_contradicting` exercise this rule directly. `bare_comparison` and `between_x_and_y` cover only a symmetric pair (Kenya/Uganda, both plausible recipients) or the word "between", which signals bidirectionality on its own. `world_knowledge_bait` pairs bare, preposition-free queries with countries carrying an asymmetric real-world donor/recipient prior (e.g. Norway/Kenya), and every entity is `unspecified` because the query states nothing. `prior_contradicting` is the sharper test. The wording states a direction that runs against the prior ("aid from Kenya to Norway"), and the correct label is the stated direction. A system that "corrects" explicit wording toward its prior overrides the query.

## Scoring

`scripts/eval_role_direction.py` loads this file and calls a scanner matching `detect_directions`'s signature, `(query, entities) -> verdicts`, over every case, producing:

- **directional accuracy**: over every entity-level label, treating a scanner verdict of `None` as `"unspecified"` for comparison.
- **false-confident rate**: over the subset of entity-level labels whose expected axis is `"unspecified"`, the fraction the scanner answered with a confident (non-`None`) axis. The ban on world-knowledge directions is meant to drive this toward zero.
- **confidently-wrong rate**: over the subset of entity-level labels whose expected axis is directional (`origin`/`destination`), the fraction the scanner answered with the opposite directional axis. A miss to `unspecified` counts against accuracy above but not here. The substring-collision cases below name this number directly.

Any scanner matching that shape, including one pulled from another commit via `git show <rev>:<path>`, can be scored the same way by putting its module on `sys.path` (or `PYTHONPATH`) before running the script.

```
uv run scripts/eval_role_direction.py \
    --scanner corpus.direction.role_text_pinned:detect_directions
```

### The frozen historical scanner

`corpus/direction/role_text_pinned.py` is `src/search/engine/role_text.py` frozen at commit `9fd49497`, the last revision with the lexical scanner: a three-token adjacency scan for "from"/"to" near each place's mention, with a coordination-inheritance pass for a list of three or more names. `9fd49497` sits on a rewritten branch and is unreachable from any branch, so the object may be pruned from this repository's object store. The vendored module keeps the baseline reproducible.

Over the full 40-case, 88-entity set:

```
entities scored:            88
directional accuracy:       67.0%
unspecified-truth entities: 23
false-confident rate:       0.0%
confidently-wrong count:    5
confidently-wrong rate:     7.7%
```

Over the 29 cases outside `world_knowledge_bait` and `prior_contradicting`, 65 entity-level labels:

```
entities scored:            65
directional accuracy:       61.5%
unspecified-truth entities: 8
false-confident rate:       0.0%
confidently-wrong count:    5
confidently-wrong rate:     8.8%
```

Both runs have the same 5 confidently-wrong case ids: the four `substring_collision` pairs `dir-substring-sudan-01`, `dir-substring-korea-01`, `dir-substring-guinea-01`, `dir-substring-congo-01`, plus `dir-possessive-03`. Each collision case names a place whose short form is a token subsequence of a longer place named earlier in the same query ("aid to South Sudan from Sudan"), and the scanner reports `destination` for the short form, anchored on the "to" before the longer name, when the query's own wording states `origin` for it. `score()` computes the count and rate directly; `tests/scripts/test_role_text_pinned_baseline.py` pins the 5 case ids, since a count alone would miss a change in which labels are wrong, and asserts the full-set numbers above so the baseline cannot drift.

`detect_directions` is a pure function, so its verdict over these fixed inputs depends only on the revision.

### The production extraction path, scored on recorded output

`corpus/direction/recorded_extraction.py` scores the production path against this set: the raw-text door's extraction stage, which fills `ExtractedEntity.flow_role` from the query's own wording, and `search.engine.extract._validate_role_spans`, the offline guard that downgrades a fabricated or mis-anchored `role_span` to `"unspecified"` before a caller sees it.

The module replays recorded output and makes no model call. Each of the 39 distinct queries in `role_direction_cases.yaml` has an `Extraction` fixture in `tests/fixtures/llm_responses.json`, recorded by `scripts/harvest_llm_fixtures.py`, whose direction-corpus collector replays this file's queries through the extraction stage. The module exposes two scanners matching `detect_directions`'s shape, so both score through the same `eval_role_direction.py:score()` as the frozen scanner:

```
uv run scripts/eval_role_direction.py \
    --scanner corpus.direction.recorded_extraction:raw_extraction
uv run scripts/eval_role_direction.py \
    --scanner corpus.direction.recorded_extraction:guarded_extraction
```

`raw_extraction` reads `Extraction.entities` as the model returned them. `guarded_extraction` reads the same recorded entities after `_validate_role_spans`, which is what the pipeline answers. Over the same 88-entity set as the frozen scanner:

```
                       accuracy         false-confident   confidently-wrong
raw model output       94.3%  (83/88)   8.7%  (2/23)      1  (1.5% of 65)
after the guard        94.3%  (83/88)   0.0%  (0/23)      1  (1.5% of 65)
frozen scanner         67.0%            0.0%              5  (7.7%)
```

The guard costs two labels and saves two. The model gives a place's own name as its `role_span` in `dir-leading-unevidenced-01` (Kenya, "Kenya aid from Norway") and `dir-leading-unevidenced-02` (Ethiopia, "Ethiopia support from the United States"); both are correct as `destination` in the raw output, and the guard downgrades both to `"unspecified"`, since a place's own name before the measure states no direction. It does the same in `dir-worldknowledge-06` ("2020 ODA Japan Bangladesh"), where the raw output reads Japan as `origin` and Bangladesh as `destination` against `"unspecified"` labels, so the guard turns both false-confident labels into correct ones. Three labels are wrong in both outputs. `dir-three-plus-list-01`'s "the United States" ("aid from the United States to Kenya, Uganda, or Ethiopia") and `dir-leading-unevidenced-02`'s "the United States" are read as `"unspecified"`. `dir-possessive-03`'s second "Kenya" ("aid from Kenya to Kenya") is a scorer limitation: `_axis_for` matches an entity by surface and returns the first match, so the model's `destination` occurrence scores against the `origin` label. `tests/scripts/test_recorded_extraction_baseline.py` pins both scanners' exact counts and the exact wrong labels, as `tests/scripts/test_role_text_pinned_baseline.py` does for the frozen scanner, so a change to the case file or the recorded fixtures fails the test.
