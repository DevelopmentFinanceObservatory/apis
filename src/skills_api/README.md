The Skills API surface serves a catalogue of agent skills read from the Data Commons graph: every skill's front matter, attachment and breadth figures, and manifest; one skill's `SKILL.md`; and any one file of a skill's bundle, `SKILL.md` included.

The surface mounts at `/api/skills` and operates as an independent composition root within `dfo-apis`, gated behind `DFO_SKILLS_SURFACE=true` (or `1`).

## Architecture

```
                  +--------------------------------------------+
                  |      skills_index.SharedSkillsIndex         |
                  |  process-wide, acquired by Search first,    |
                  | rebuilt on release swap, no graph read here |
                  +--------------------------------------------+
                        |                |                  |
                        v                v                  v
              GET /api/skills   GET /api/skills/    GET /api/skills/{skill_id}/
                                {skill_id}           resources/{path}
```

This surface holds four doors: health, list, skill, and resources. None of the three content doors reads the graph or builds a graph client. All three read the `index` attribute of one process-wide `skills_index.SharedSkillsIndex`, acquired at this surface's startup (`skills_api.app.skills_surface`) and shared with the Search surface, which is mounted first and supplies the live graph client. The index rebuilds in the background each time the process's graph release service swaps to a newer release, so an edit to a skill's node reaches these doors with the release that carries it.

### Components

- Health door (`doors/health.py`): `GET /api/skills/healthz`. Reads no index and always answers.
- List door (`doors/list.py`): `GET /api/skills`. Every catalogue skill's entry, ordered by id, plus the corpus warnings.
- Skill door (`doors/fetch.py`): `GET /api/skills/{skill_id}`. One skill's list entry plus its `SKILL.md` text.
- Resources door (`doors/resource.py`): `GET /api/skills/{skill_id}/resources/{path}`. One file of the skill's manifest, matched exactly by path.
- Wire models (`models.py`) and codes (`codes.py`). Refusals are RFC 9457 problem details, classified by each door's classifier and combined in registration order (`dfo_platform.doors.combine_classifiers`).

---

## Trust boundary

This surface serves a skill's text to clients verbatim and presents it as platform guidance, with no sanitisation, review, or rewriting. Its soundness rests on one deployment assumption, enforced outside this repository: graph-write access to `AgentSkill` and `AgentSkillReference` nodes is restricted to the approved publishing pipeline. Nothing in this surface, or in the graph read path beneath it, verifies that restriction at read time.

If that restriction stops holding, anything able to mint a node of either type can place its own instructions into a surface every client treats as platform guidance, and nothing in the response distinguishes an approved publisher's content from anyone else's.

A provenance allowlist cannot be added at the read layer as it stands. The reads the index build runs through, `node_arcs_many` and `node_subjects_many` (`skills_index.graph.SkillsGraph`, satisfied structurally by `search.engine.clients.graph_client.LiveGraphClient`), collapse a value asserted by more than one provenance to a single entry and drop `provenanceId`, even though the wire response carries it and a sibling reader on the same client, `_dedup_wire_arcs`, keeps it. Narrowing trust by provenance means changing that read layer.

---

## Endpoints

Every response, success or refusal, carries `x-dfo-trace-id`, `x-dfo-commit`, and a `Server-Timing: total` header, per the platform convention.

### GET /api/skills

Every skill in the catalogue, ordered by id, so an unchanged catalogue keeps the same body and ETag across requests and rebuilds.

```json
{
  "skills": [
    {
      "id": "oda-concepts",
      "name": "oda-concepts",
      "description": "Core ODA concepts ...",
      "frontmatter": { "name": "oda-concepts", "description": "Core ODA concepts ..." },
      "content_version": "sha256:9c1e...",
      "url": "https://github.com/DevelopmentFinanceObservatory/skills/tree/main/skills/oda-concepts",
      "attached": true,
      "excluded": false,
      "statvar_count": 4695,
      "statvar_count_truncated": false,
      "warnings": [],
      "resources": [
        { "path": "SKILL.md", "size": 2955, "digest": "sha256:1f0a...", "media_type": "text/markdown" },
        { "path": "references/core-oda.md", "size": 5631, "digest": "sha256:77b2...", "media_type": "text/markdown" }
      ]
    }
  ],
  "warnings": []
}
```

- `id` is the skill's dcid without its `skill/` prefix. `name` and `description` come from the front matter, and `name` equals `id`.
- `frontmatter` is the whole parsed front matter, every key the author wrote, each scalar a string as YAML's base loader reads it, so `version: 1.10` stays `"1.10"`.
- `content_version` is `sha256:` plus the hex SHA-256 of each manifest entry's path, a tab, its digest and a newline, in manifest order. Search carries the same value.
- `attached` is true when the skill's reach names at least one node on this build, which is when search can name it. It is false for a skill with no attachment, one whose every target was rejected, and one that is `excluded`.
- `excluded` is true when search cannot match the skill on this build: its reach broke a bound, or a read the build depended on did not complete. Its warnings say why.
- `statvar_count` counts the served StatVars the skill's reach covers, and `statvar_count_truncated` is true when the count stopped at its cap. A provenance attachment adds nothing to the count, so a skill attached only to provenances reports 0 although it covers every series of those datasets.
- `url` points a human reader at the skill's directory in `skills`, or is null. The platform never fetches it.
- `resources` is the skill's manifest: `SKILL.md` first, then its references in path order. `size` is the byte length of the file's UTF-8 content and `digest` is `sha256:` plus the hex SHA-256 of those bytes. `media_type` comes from the extension (`skills_api.models.media_type`), `text/markdown` for `.md`, the only extension the index admits.

The index reads a skill's attachments from two arcs on its node. A `skillAppliesTo` target is an ontology node: a tuple member, a constraint property, or a constraint value, classified by its incoming arcs, and its reach runs down the hierarchy beneath it. A `skillAppliesToProvenance` target is a `Provenance` node, known as one by the arc alone. Its reach is the provenance itself, and search matches it when a spec's selected provenance facet names the same dcid. Search lists skills matched through an ontology node first, by hop, then exact before capped breadth, then breadth ascending, and skills matched only through a provenance after them, by id, since a whole dataset is wider than any node inside it. A skill declaring more than 500 targets across both arcs is excluded.

A skill-level warning rides on that skill's entry. The response's own `warnings` carry the corpus warnings: a truncated roster read (`skill_id` null), and one warning per skill left out of the catalogue, whose `skill_id` is that skill's roster dcid (`skill/<id>`) and whose code names the rule it broke.

### GET /api/skills/{skill_id}

The list entry plus `content_text`, which is `SKILL.md` verbatim, front matter included. `content_text` is always present, because a skill without usable text is left out of the catalogue.

### GET /api/skills/{skill_id}/resources/{path}

```json
{
  "skill_id": "oda-concepts",
  "path": "references/core-oda.md",
  "size": 5631,
  "digest": "sha256:77b2...",
  "media_type": "text/markdown",
  "content_text": "# Core ODA\n\n..."
}
```

`path` is matched exactly against the skill's manifest. `SKILL.md` is a manifest entry, so `resources/SKILL.md` returns the same text as the skill door's `content_text`, and every file of a bundle is read through this one door. The door never joins `skill_id` and `path` into a dcid, so a path carrying a percent-encoded `..`, a leading slash, or a backslash matches no entry and refuses `404 skill_resource_not_found`. An unencoded `../` is resolved away by the client before sending, so the URL matches no route and refuses `404 not_found`.

An unknown `skill_id` refuses `404 skill_not_found` whatever the path. A known skill whose manifest lacks the path refuses `404 skill_resource_not_found`. The remedies differ: re-read the catalogue for the first, re-read the skill for the second.

### GET /api/skills/healthz

Reports status and the capability version token.

```json
{
  "status": "ok",
  "build": "2618ed596fb472588307ca2a7fc59d183a3f311c",
  "capability_version": "skills_a1b2c3d4e5f67890"
}
```

---

## Caching

Every 200 from the list, skill and resources doors carries a strong `ETag` and `Cache-Control: public, no-cache`. A release swap rebuilds the index at once, so a cache revalidates each response by its ETag. A matching `If-None-Match` answers 304. Each door caches encoded bodies per index generation (`skills_api.doors._response_cache.GenerationBodyCache`), up to 8 MiB per door, and drops them all when the generation changes.

Before the index's first build the three content doors refuse `503 service_unavailable`. A 200 with an empty list would read as an empty catalogue, and a 503 is never cached.

## Configuration

| Variable             | Type    | Default | Description                                                                |
| :------------------- | :------ | :------ | :------------------------------------------------------------------------- |
| `DFO_SKILLS_SURFACE` | Boolean | `false` | Mounts the Skills surface on the root FastAPI application (`1` or `true`). |

The surface reads no other configuration. It opens no connection of its own: the graph reads that keep the shared index current run through the graph client the Search surface's lifespan builds.

## Path parameters

`skill_id` uses the `dfo_skill_id` convertor (`skills_api.routing.SkillIdConvertor`): any one path segment except a reserved route name (`skills_index.config.RESERVED_SKILL_IDS`, `healthz`), so `/api/skills/healthz` reaches the health door whichever door is mounted first. `path` uses Starlette's `path` convertor, so everything after `resources/`, slashes included, is the path.

| Parameter | Ceiling | Constant |
| :-- | --: | :-- |
| `skill_id` | 512 | `skills_api.routing.SKILL_ID_MAX_LENGTH` |
| `path` | 512 | `skills_index.config.RESOURCE_PATH_MAX_LENGTH` |

A value past its ceiling refuses `400 request_invalid` before any lookup. Any other `skill_id` outside the catalogue refuses `404 skill_not_found`. The path ceiling is the one the index applies to reference paths, so every listed path is one the door accepts.

## Content bounds

The index build (`skills_index.build.build_index`) enforces the bounds the publishing pipeline enforces first:

| Bound | Ceiling | Constant | On breach |
| :--- | ---: | :--- | :--- |
| A skill's decoded `SKILL.md` | 256 KiB | `skills_index.config.SKILL_BODY_MAX_BYTES` | The skill is left out, named `skill_content_too_large`. A skill with no text is left out as `skill_content_absent`, one whose text does not decode as `skill_content_undecodable`, and one whose front matter is missing or invalid as `skill_frontmatter_invalid`. |
| Collection nesting in front matter | 32 levels | `skills_index.frontmatter._MAX_NESTING_DEPTH` | The skill is left out, named `skill_frontmatter_invalid`. So is one whose front matter uses a YAML anchor or alias, or does not serialise as the `frontmatter` field of this API's JSON. |
| One reference's decoded content | 256 KiB | `skills_index.config.SKILL_REFERENCE_MAX_BYTES` | The reference is dropped from the manifest, named `skill_resource_rejected`. So is a reference whose content is absent or does not decode, or whose path is invalid. |
| References one skill's `agentSkillReferences` declares | 50 | `skills_index.config.SKILL_REFERENCES_MAX_ENTRIES` | The excess is dropped before it is read, each named `skill_resource_rejected`. |
| Retained content across the catalogue | 128 MiB | `skills_index.config.SKILLS_CORPUS_CONTENT_BUDGET_BYTES` | Each skill reached after the budget is spent is left out, named `skills_corpus_budget_exhausted`, and search marks its skill matches `skills_index_incomplete`. A later build may include it. Two references sharing a path are both dropped, and their bytes are refunded as the second is read, so they never count against a later skill. |

---

## Error handling

| HTTP status | `code` | When | Pointer | Retryable |
| :-- | :-- | :-- | :-- | :-- |
| `400` | `request_invalid` | `skill_id` longer than 512 characters | `/skill_id` | No |
| `400` | `request_invalid` | `path` longer than 512 characters | `/path` | No |
| `404` | `skill_not_found` | No skill in the catalogue has this id | `/skill_id` | No |
| `404` | `skill_resource_not_found` | The skill exists and its manifest has no such path | `/path` | No |
| `404` | `not_found` | No route matches | None | No |
| `503` | `service_unavailable` | The index has not finished its first build | None | Yes |

Each replica swaps releases on its own recheck (every 60 seconds, `graph_release.service.DEFAULT_RECHECK_SECONDS`) and then rebuilds, so an id one replica names can be unknown to another until that replica has swapped and rebuilt. The remedy for that `skill_not_found` is to retry once or re-read the list.
