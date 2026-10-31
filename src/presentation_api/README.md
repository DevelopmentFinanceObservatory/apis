# Presentation API

The Presentation API manages interactive presentation frames, component specifications, and layout sessions for the Development Finance Observatory (DFO). It binds observation data from dataset frames into structured visual components (such as line charts, stat cards, and bar charts), tracks revisions across user sessions, and persists component layouts in PostgreSQL.

The surface mounts on the shared FastAPI platform at `/api/presentation` when enabled by environment configuration (`DFO_PRESENTATION_SURFACE=1` or `true`).

---

## Architecture and Doors

The surface exposes five doors through a single router, error classifier, and lifespan context.

```
/api/presentation
├── /healthz                              (Meta door)
├── /capabilities                         (Meta door)
├── /release                              (Release door)
├── /frames/{spec_id}/{content_version}   (Frames door)
├── /components[/{component_id}]          (Components door)
└── /shares[/{share_id}[/snapshot|/history|/slots/{slot}/data]]  (Shares door)
```

### 1. Meta Door

Handles liveness, build identification, and the deployment's capability version. Mounted first for static path precedence.

| Method | Endpoint                         | Description                                                                                                                  |
| :----- | :------------------------------- | :--------------------------------------------------------------------------------------------------------------------------- |
| `GET`  | `/api/presentation/healthz`      | Returns liveness status, build commit, and capability version (`{"status": "ok", "build": ..., "capability_version": ...}`). |
| `GET`  | `/api/presentation/capabilities` | Returns `{capability_version, catalog_version, build}`, publicly cacheable for five minutes.                                 |

`capability_version` is a content hash over the release document as `/release` serves it, including schema wording, plus the intent grammar and the refusal surface, so it names what the deployment can do. `catalog_version` names the release that produced a given payload; inside that hash, its value is pinned to a placeholder. A catalogue republish under an otherwise unchanged grammar moves `catalog_version` without moving `capability_version`.

Every response, healthz and capabilities included, also carries `x-dfo-trace-id`, `x-dfo-commit`, and a `Server-Timing: total` header; `build` in both bodies equals the commit `x-dfo-commit` carries.

### 2. Release Door

Exposes metadata for the pinned component catalogue release (`BakedReleaseRegistry`). Clients and tool bridges (such as `presentation-mcp`) query this endpoint to discover supported component schemas, constraint definitions, retry modes, and minimal intent structures.

| Method | Endpoint                    | Description                                                                                                                                                                                                                              |
| :----- | :-------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GET`  | `/api/presentation/release` | Returns catalogue versions (`catalog_version`, `schema_version`), member types, constraint IDs, retry modes, the envelope JSON Schema and its per-member-type component schemas, tolerant intent schemas, and minimal intent structures. |

### 3. Frames Door

Inspects underlying observation frames (`get_frame_view` / `describe_frame`) across dataset specifications and content versions.

| Method | Endpoint                                               | Description                                                                                  |
| :----- | :----------------------------------------------------- | :------------------------------------------------------------------------------------------- |
| `GET`  | `/api/presentation/frames/{spec_id}/{content_version}` | Returns structural dimension summaries, full facet lists, or filtered values.                |
| `GET`  | `/api/presentation/frames/{spec_id}`                   | The same views of a dataset's newest version, with the `spec_id` and `content_version` read. |

Query parameters:

- `dimension`: Isolates a single dimension and returns its complete list of permissible values.
- `query`: Filters dimension values across all facets by substring match.

`GET /api/presentation/frames/{spec_id}` names a dataset by its `spec_` id alone (`spec_` and 32 lowercase hex digits; any other id refuses 400 `request_invalid`). It reads the version the analysis runner reads for a Spec root `{spec_id}` (`datasets_api.analysis_roots.LiveRootSource.resolve`): the served definition's newest version, rebuilt when its head is stale. It takes the same `dimension` and `query` and answers the pair route's body, read from that version's canonical frame, so it holds no default enrichment column and a `dimension` such as `financingProvider_type` refuses 400 `binding_invalid`. It adds two fields, `spec_id`, the served id it read (another id when the one sent serves under it), and `content_version`. A version still building refuses 503 `service_unavailable`, retryable, with `Retry-After`. A Datasets surface at capacity refuses `service_unavailable` with `cause.code` `busy`, and any other retryable Datasets refusal with `cause.code` `io`, each keeping its `Retry-After` when it names one. Any other Datasets refusal refuses 404 `frame_unavailable`, not retryable, with the Datasets code in its `detail`, and `cause.code` `revoked` for a revoked version, `quarantined` for a quarantined identity and `unknown` for the rest. An app that does not mount the Datasets surface resolves no bare id and refuses 404 `frame_unavailable`.

Each dimension lists its non-null values in `values`, counted by `total_count`. `plural` says whether the frame's series hold more than one value of the dimension, a null counting as one, so a `unit` dimension holding `USDollar` on some series and null on others is plural with one listed value. A series its source's estimates continue holds the method of the rows it continues (`assembler.frame_view.series_keys`), so on a frame whose only other method is that continuation `measurementMethod` is not plural, and its continuing method stays listed as a value a select binds to draw the estimates alone.

### 4. Components Door

Handles creation, revision, and readback of visual components and layout compositions via `assembler.pipeline`.

| Method  | Endpoint                                                                                                               | Description                                                                                                                                                                                                                                                                                                 |
| :------ | :--------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `POST`  | `/api/presentation/components`                                                                                         | Creates and binds a new component within an optional composition layout.                                                                                                                                                                                                                                    |
| `PATCH` | `/api/presentation/components/{component_id}`                                                                          | Revises an existing component draft, validating incremental intent updates.                                                                                                                                                                                                                                 |
| `GET`   | `/api/presentation/components/{component_id}[?revision]`                                                               | Reads back a stored component definition by ID and optional revision number.                                                                                                                                                                                                                                |
| `GET`   | `/api/presentation/components/{component_id}/revisions/{revision}/data[?format&bundle]`                                | Downloads the frame rows behind every value one revision shows, as CSV, XLSX, or Parquet, in a zip bundle by default (`README.md`, the data file, `data/codes.csv`, `SOURCES.md`, `CITATION.cff`, `datapackage.json`). The download is named `dfo_<card-title-slug>_<YYYY-MM-DD>_<format>_r<revision>.zip`. |
| `POST`  | `/api/presentation/components/{component_id}/revisions/{revision}/render`                                              | Draws one stored revision on another frame for a live share. Live-refresh principal only.                                                                                                                                                                                                                   |
| `GET`   | `/api/presentation/components/{component_id}/revisions/{revision}/render/data?spec_id&content_version[&format&bundle]` | Downloads the rows behind that drawing. Live-refresh principal only.                                                                                                                                                                                                                                        |

A create's `intent.frame_ref` may name a dataset id alone, `{"spec_id": "spec_..."}`. The create resolves it as the bare frames route does, draws that version, and stores the pair, so a read's `stored_intent.frame_ref` and the reply's `resolved_payload.provenance.inputs` name it. A create retried with its `write_token` replays the first reply and resolves nothing again. The bare frames route's refusals apply. A revise keeps its component's frame.

### 5. Shares Door

Publishes cards under one public `/s/<id>` id (`presentation_api.doors.shares`, records in `presentation_store.shares`). Every route serves the code registry endpoint `presentation.shares`.

| Method   | Endpoint                                                | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| :------- | :------------------------------------------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `POST`   | `/api/presentation/shares`                              | Body `{cards: [{component_id, revision}], latest, title?, kind?, payload?}`, at most 24 cards. `latest` has no default. `true` makes a live share, which follows new data, and `false` a snapshot, which keeps the versions its cards draw now. `kind` is `card` (the default, 1 to 24 cards), `canvas` (0 to 24), `search` or `conversation` (no cards, and a conversation is a snapshot). `payload` is website-owned JSON of at most 10 MiB as compact UTF-8, stored as given; a larger one refuses 413 `payload_too_large`. Answers 201 `{id, url, embed_url, data_urls, write_token, revision, latest}`. |
| `GET`    | `/api/presentation/shares/{share_id}`                   | Public, with `Access-Control-Allow-Origin: *`: `{id, kind, mode, title, revision, slots, payload, created_at}`, each slot `{slot, card, data_url}`. The write token is not returned.                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `PUT`    | `/api/presentation/shares/{share_id}`                   | Body `{cards}`, which takes the card count the share's kind takes on create. Writes a live share's next revision; a snapshot refuses first. Needs `x-dfo-write-token`. Answers 200 `{id, url, embed_url, data_urls, revision, latest}`.                                                                                                                                                                                                                                                                                                                                                                      |
| `POST`   | `/api/presentation/shares/{share_id}/snapshot`          | Turns a live share into a snapshot under the same URLs. Needs `x-dfo-write-token`. Answers 200 `{id, url, embed_url, data_urls, revision, latest: false}`; a snapshot answers as it is.                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `PUT`    | `/api/presentation/shares/{share_id}/import`            | The website's share migration job stores one of its shares under its own id. Admits `PRESENTATION_SHARE_IMPORT_PRINCIPAL` alone. Body: the share's export line less `id`, plus `write_token_sha256`. Answers 201 `{id, imported: true}`, or 200 `{id, imported: false}` when an earlier import stored it.                                                                                                                                                                                                                                                                                                    |
| `GET`    | `/api/presentation/shares/{share_id}/slots/{slot}/data` | Public, with `Access-Control-Allow-Origin: *`. `?format=json` (default), `csv` or `parquet`: the rows behind the slot's card.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `GET`    | `/api/presentation/shares/{share_id}/history`           | Public, with `Access-Control-Allow-Origin: *`: `{id, served: [{slot, component_id, revision, frame, rendering, first_served_at, retention, fetchable}], next}`, newest first, `limit` entries a page (1 to 500, default 100); `next` is the cursor `before` takes for the following page, null on the last; an emptied slot's entry has `component_id`, `revision`, `frame`, `rendering` and `retention` null and `fetchable` false.                                                                                                                                                                         |
| `DELETE` | `/api/presentation/shares/{share_id}`                   | Takes the share down, 204, and answers 204 again to the token's holder once it is down. Needs `x-dfo-write-token`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |

A share id is 8 characters from the case-sensitive `[A-Za-z0-9]` alphabet, the shape of the website's `/s/<id>` links, so shares imported from the website keep their ids. `url` is `<DFO_PUBLIC_SITE_URL>/s/<id>`, `embed_url` is `/embed/<id>` under the same origin, and slot n's data URL is `/api/v1/presentation/shares/<id>/slots/<n>/data` under it.

Slots are numbered from 1. A create puts the nth card in slot n. A replace puts its nth card in slot n and leaves every further slot the share has had empty (`card: null`), so each slot keeps its embed and data URLs and a later replacement with more cards fills them again. A snapshot cannot be replaced. Each slot records the frame its card revision draws.

A create names its caller with `dfo_platform.caller.resolve_caller` and spends one unit of the caller's daily `share` allowance (`DATASETS_SHARE_DAILY_ALLOWANCE`, default 10, `datasets_api.allowance`) after it has read every card and before it pins or writes anything, so a malformed request or an unknown card spends nothing, and a pin or write that fails gives the unit back, so a retry spends one unit in all. A live share holds at most 20 data cards, those that draw a frame, on create and on replace (the website's `LIVE_SHARE_MAX_DATA_CARDS`); more refuses 400 `request_invalid` at `/cards`. Every share pins the frames its cards draw when it first includes them (`presentation_store.pin_card_frames`). A create pins every card, live or snapshot, and a replace pins the card revisions the share did not show before. A create, a replace and a snapshot conversion spend the caller's `pin` allowance and the shared pin budgets for each new pinned copy of a frame (`datasets_api.retention_admission`), and only a create spends the `share` allowance. A delete spends nothing.

A slot's data route serves the card's data export, unbundled, on the frame the slot serves (`presentation_api.share_slots`). A snapshot slot serves the frame its revision records. A live slot draws its card's intent on the newest data: it asks for the newest frame of the card's frame and keeps the answer in `presentation.share_slot_check`, which the reads before its `next_check_at` serve. A `current` or `stale` answer is due again after 15 minutes. A `pending` one is due after the retry time the refresh or the run gave, clamped to 1 to 900 seconds. A dataset (`spec_`) frame moves as the Datasets frame refresh moves it, and a derived (`ops_`) frame is the analysis runner's answer for the bare derived root with `latest=true` (`presentation_api.newest_frames`), both in process when the app mounts Datasets beside Presentation. The JSON body `{share_id, slot, mode, freshness, component_id, revision, frame, rows}` and the `x-dfo-freshness` header state the rows `current` (the newest data at the last check, or a snapshot's rows), `pending` (newer data is building, and these are the last check's rows) or `stale` (the check failed, the card cannot be drawn on the newer frame, or no Datasets surface answers, and these rows are older). A check is stored only once the card draws on its frame. When the newer frame fails transiently, a `FrameUnavailable` whose reason is `io` or `busy`, or a presentation store, Datasets store or spec registry failing to answer, the read serves the card's own frame `stale` and stores a check `stale` on that frame, due after the failure's `Retry-After` or 30 seconds when it names none. The reads before then serve that frame without drawing the newer one, and the first read after it checks again. Such a read records no frame served. Any other failure stores the slot `stale` on the card's own frame until the next check. A live slot's `x-dfo-checked-at` names the check's time. Each reply carries a strong `ETag` over its bytes, weak on a gzip body, and `If-None-Match` answers 304. A live slot answers `Cache-Control: no-cache`, a snapshot `public, max-age=300`. Every read records the frame it served in `presentation.served_frame`, and a create, replace, snapshot conversion or import records each slot's frame at the write, so the history route lists what a share shows before any read. A slot records an entry only when what it shows changes, and a slot a replace leaves empty, or whose card is a note, records an entry with no card or frame. A read of a slot whose newest entry already shows what it serves takes no lock. The history route lists them with each frame's retention counted from when DFO stored it (`pinned`, `held_until`) and `fetchable`, whether that retention still holds.

A snapshot conversion keeps, for each slot, the frame it serves now (its last check's, else its card's own), pins those frames under the card's own reference, and writes the share's next revision with them and the `snapshot` mode. A share replaced while it converts refuses `edit_conflict`. No route turns a snapshot live.

`PUT /api/presentation/shares/{share_id}/import` stores one of the website's shares under its own id, owner, payload and creation time (`presentation_store.share_import`), for the website's share migration job, which calls it once per share. It admits one caller, the value of `PRESENTATION_SHARE_IMPORT_PRINCIPAL` in the `DATASETS_PRINCIPAL_HEADER` header, compared in constant time before the body is read; any other caller, and every caller while the principal is unset, is refused `share_import_not_permitted` (403). The body is the share's line as website `docs/share-export.md` defines it, less `id`, which the path names: `{kind, mode, owner, created_at, title, slots, payload, write_token_sha256}`. `write_token_sha256` is the lowercase hex SHA-256 of the `shw_` write token the website keeps for a signed-in user's share, or null for a guest's share, which no token then changes; a digest needs an `owner`, and the token itself never reaches this service. The export's 0-based `slot` index `i` is slot `i + 1`, and a slot exported with `frame: null` draws the frame its card revision is stored on. Before storing the share it pins every frame the share shows, spending no allowance or pin budget. Each slot's history entry records its frame and its card revision's rendering from the share's creation time, as a create's does. It answers 201 `{id, imported: true}` when it stores the share and 200 `{id, imported: false}` when the id already holds this share with the same kind, `created_at` and digest, taken down since or not, so a re-run changes nothing. The first import of an id wins: a re-run whose owner, mode, title, slots or payload differ also answers 200 `imported: false` and leaves the stored share as it is. An id holding another share, this share under another digest, or a digest another share holds, is refused `share_id_taken` (409). Neither a 200 nor a 409 pins anything. A body that is not such a line, or whose slot names no frame and whose card revision draws none, is refused `request_invalid` (400), a card revision the store lacks `not_found` (404), and a frame that cannot be pinned `frame_unavailable`, as on a create. Run `datasets-maintenance normalise-recipes` after the migration job, which moves imported cards drawing a derived frame to the recipe hash this service computes (Datasets README).

`datasets-maintenance take-down-share <share_id>` takes a share down for its owner without the write token, as the website does for each share id its `share_cleanup` table holds, the shares whose delete it could not complete. It calls the store's `delete_share`, as the owner's delete does, so every route then answers the share `not_found` and the token's holder still gets 204 from a delete. It logs `presentation_share_taken_down`, or `presentation_share_take_down_not_found` when no share is up at the id, and exits 0 for both, so a re-run is safe. An id that does not match the 8-character share id shape is refused with a `presentation_share_take_down_refused` `ERROR` line and exit 1 before the store opens.

The create returns a `shw_` write token once, and the store keeps its SHA-256 digest. A replace or delete checks the token before the share's mode, so a caller without the token learns nothing more about the share. A share stored without a token, as a guest's share imports from the website, refuses every token.

| Error Code                   | HTTP Status | When                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| :--------------------------- | :---------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `write_token_required`       | 403         | A replace or delete sent no `x-dfo-write-token`.                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `write_token_invalid`        | 403         | The token sent is not the share's, or the share has none.                                                                                                                                                                                                                                                                                                                                                                                                          |
| `share_import_not_permitted` | 403         | An import from a caller other than the share import principal, or from any caller while `PRESENTATION_SHARE_IMPORT_PRINCIPAL` is unset.                                                                                                                                                                                                                                                                                                                            |
| `share_not_replaceable`      | 409         | A replace names a snapshot.                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `share_id_taken`             | 409         | An import names an id holding another share, or this share under another `write_token_sha256`.                                                                                                                                                                                                                                                                                                                                                                     |
| `edit_conflict`              | 409         | A snapshot conversion found the share replaced meanwhile. Retry.                                                                                                                                                                                                                                                                                                                                                                                                   |
| `allowance_exhausted`        | 429         | The caller has spent today's share allowance. Not retryable; `Retry-After` names the reset, the next UTC midnight.                                                                                                                                                                                                                                                                                                                                                 |
| `not_found`                  | 404         | No share at the id, or it was taken down; on a create or replace, a card or revision the store lacks; on a data read, a slot the share has not had, one a replacement left empty, or a note's.                                                                                                                                                                                                                                                                     |
| `request_invalid`            | 400         | A malformed body (`latest` missing or not a boolean, a card count or `latest` the `kind` does not take, more than 24 cards, an unknown field), id, slot or format, or a website request whose `x-dfo-caller` is missing or malformed.                                                                                                                                                                                                                              |
| `payload_too_large`          | 413         | A create's `payload` is over 10 MiB as compact UTF-8 (`presentation_store.MAX_SHARE_PAYLOAD_BYTES`).                                                                                                                                                                                                                                                                                                                                                               |
| `frame_unavailable`          | 404         | A card's frame is gone: unknown, expired, revoked or quarantined, as on the component routes. A pin of a derived frame with no recipe yet answers it too, retryable, with `cause.code` `recipe_pending`, and succeeds once its recipe is written. One with no recipe that a reference already names pins alone (`presentation_store` README, pin step 3).                                                                                                                                                                                                                                                                                                                                                                      |
| `service_unavailable`        | 503         | A frame read or store call failed, or the door is at capacity. Retryable. |
| `internal_error`             | 500         | The caller cannot be named: `DFO_CALLER_HMAC_KEY` is unset or the trusted `X-Forwarded-For` entry is missing. A pin whose recipe cannot be read answers it too, with `cause.code` `recipe_unreadable`, for a store with no recipe reader or a recipe root naming no dataset version. Recipes are write-once, so no retry clears it.                                                                                                                                                                                                                                                                                                                                                      |

The shares door mounts when the surface builds its stores live or is given a `share_store`. A surface composed over an injected `revision_store` without one serves no share route.

---

## Quickstart

A create request carries the intent body:

```http
POST /api/presentation/components HTTP/1.1
Content-Type: application/json

{
  "intent": {
    "type": "line_chart",
    "frame_ref": {
      "spec_id": "spec_9f2e41a7c03b58d2a4c1f6b2e8d35790",
      "content_version": "cv_f4a81c22d9075e3ba61c2ee98f83b405"
    },
    "select": {
      "stat_var": "dcid:MalariaGrantDisbursements",
      "recipient": "country/KEN"
    },
    "window": { "start": "2015", "end": "2024" }
  }
}
```

`frame_ref` is a placeholder above. A live `spec_id`/`content_version` pair comes from the Datasets API, pinned to the frame the component is about to bind against.

The reply carries the new component's id, its fact summary, its resolved payload, an affordance naming the routes that revise or read it back, and the component's `write_token`. A create with `composition_id: "new"` also carries `composition_write_token`. The example below omits `resolved_payload`, which [Read Reply](#read-reply-get) shows in full:

```json
{
  "component_id": "cmp_01ARYZ6S41YYYYYYYYYYYYYYYY",
  "revision": 1,
  "catalog_version": "2026.8.0",
  "composition_id": null,
  "composition_revision": null,
  "changed": [],
  "fact_summary": {
    "changed": [],
    "series": [
      {
        "id": "series_1",
        "label": "Malaria grant disbursements, Kenya",
        "coordinates": { "recipient": "country/KEN" },
        "point_count": 9,
        "first_value": "12.4",
        "last_value": "18.7"
      }
    ],
    "series_cut": 0,
    "window": { "start": "2015", "end": "2024" },
    "unit": null,
    "change": null,
    "source_refs": null,
    "defaults": []
  },
  "affordance": "Revise this component with PATCH /api/presentation/components/cmp_01ARYZ6S41YYYYYYYYYYYYYYYY, sending only the fields to change and this reply's write_token as x-dfo-write-token, or read it back with GET /api/presentation/components/cmp_01ARYZ6S41YYYYYYYYYYYYYYYY.",
  "write_token": "cmw_Zm9vYmFyYmF6cXV4cXV1eA"
}
```

---

## Interaction Model

### Read grant and write grant

The `cmp_` component id and the `cps_` composition id grant read. Anyone holding a component id can read that component at any revision. Each id is a ULID: 48 bits of creation time and 80 bits from `os.urandom`, so an id cannot be guessed or enumerated. An id that names no draft returns HTTP `404` with problem code `not_found`.

A write token grants write. A create returns a `cmw_` token for the new component as the top-level `write_token` field, and a create with `composition_id: "new"` also returns a `cpw_` token for the new composition as `composition_write_token`. Each token is its prefix plus `secrets.token_urlsafe(16)`, 128 random bits. A create reply is the only place a token appears: the store keeps its unsalted SHA-256 digest, and no read, revise reply, stored intent, or resolved payload carries it.

- A revise sends the component's token as the `x-dfo-write-token` header. A composition member needs only its own component token.
- A create into an existing composition sends that composition's token in the same header.

A missing or wrong token, or a draft created before write tokens and so holding no digest, returns HTTP `403` with problem code `write_refused`, retry mode `do_not_retry`. The caller creates a new component instead, and may read the old one first to reuse its intent. The API checks no session or owner.

The `/components` and `/frames` routes read no session header. A request that sends `x-dfo-session` is served as if it had not. The two render routes admit one caller, described under [Rendering a revision on another frame](#rendering-a-revision-on-another-frame).

A create or revise spends one unit of its caller's daily `card` allowance (`DATASETS_CARD_DAILY_ALLOWANCE`, default 500, `datasets_api.allowance`), the caller named by `dfo_platform.caller.resolve_caller` as a share create names it. A spent allowance refuses 429 `allowance_exhausted` with `Retry-After` set to the next UTC midnight. A write that fails gives its unit back, so a refused intent spends nothing. A caller whose `x-dfo-principal` holds the publish grant (`DATASETS_FIRST_PARTY_PRINCIPAL`, which `ai` sends) spends none.

### Idempotent create

A create may send its own component token in the body as `write_token`: `cmw_` plus exactly 22 url-safe base64 characters, minted as `"cmw_" + secrets.token_urlsafe(16)` or from an equivalent CSPRNG. The token becomes the component's write token and keys the create, so a client that retries a timed-out or failed create sends the same token on every attempt:

```json
{
  "intent": { "type": "line_chart", "...": "..." },
  "write_token": "cmw_Zm9vYmFyYmF6cXV4cXV1eA"
}
```

- A create whose token names a stored component, sent with the same `intent` and `composition_id`, answers `201` with that component's create reply rebuilt from revision 1, carrying the same token. It resolves and pins no frame. `composition_revision` reports the composition's head at replay time.
- The same token with a different request returns `400` `request_invalid` at pointer `/write_token`, as does a token held by a component whose create stored no request hash. Mint a new token for each create.
- Two creates racing with one token store one component: the second waits on the first's uncommitted row and then answers with the first's reply.
- A malformed token, or a token sent with `composition_id: "new"`, returns `400` `request_invalid` at `/write_token`. A revise body cannot carry `write_token`.
- An intent the request hash cannot canonicalise, such as an integer beyond 2^53, returns `400` `request_invalid` at `/intent` when sent with a token.

The request hash is the SHA-256 of the RFC 8785 canonical JSON of `{"intent", "composition_id"}` as sent, so reordered keys still match. Without a body `write_token` the server mints the token and the create is not idempotent: a retry after a timeout can store a second draft. The `x-dfo-write-token` header keeps its meaning on a create, the token of the composition the component joins.

### Request and Reply Structures

Component creation and revision accept an intent document specifying component type, data bindings, frame references, and visual encodings. Supported member types are `line_chart`, `stat_card`, `bar_chart`, `table`, `note_narration`, and `source_card`.

#### Request Payload (`POST` / `PATCH`)

```json
{
  "intent": {
    "type": "line_chart",
    "frame_ref": {
      "spec_id": "spec_9f2e41a7c03b58d2a4c1f6b2e8d35790",
      "content_version": "cv_f4a81c22d9075e3ba61c2ee98f83b405"
    },
    "select": {
      "stat_var": "dcid:MalariaGrantDisbursements",
      "recipient": "country/KEN"
    }
  },
  "composition_id": "cps_01ARYZ6S41ZZZZZZZZZZZZZZZZ"
}
```

`frame_ref` and the `select` values are placeholders above. A live `spec_id`/`content_version` pair comes from the Datasets API, pinned to the frame the component is about to bind against, and the `select` values come from bindings valid against that frame.

The `composition_id` field is set during initial creation (`POST`), either referencing an existing composition or `"new"` to mint one. Supplying `composition_id` on `PATCH` returns a `binding_invalid` rejection (`pointer="/composition_id"`).

#### Write Reply (`WriteReply`)

Responses to `POST` (201 Created) and `PATCH` (200 OK) return a structured write acknowledgement. A revise response, `resolved_payload` omitted:

```json
{
  "component_id": "cmp_01ARYZ6S41YYYYYYYYYYYYYYYY",
  "revision": 2,
  "catalog_version": "2026.8.0",
  "composition_id": "cps_01ARYZ6S41ZZZZZZZZZZZZZZZZ",
  "composition_revision": 1,
  "changed": [
    {
      "field": "title",
      "prior": "Malaria grant disbursements",
      "new": "Malaria grant disbursements, Kenya"
    }
  ],
  "fact_summary": {
    "changed": ["title"],
    "series": [
      {
        "id": "series_1",
        "label": "Malaria grant disbursements, Kenya",
        "coordinates": { "recipient": "country/KEN" },
        "point_count": 9,
        "first_value": "12.4",
        "last_value": "18.7"
      }
    ],
    "series_cut": 0,
    "window": { "start": "2015", "end": "2024" },
    "unit": null,
    "change": null,
    "source_refs": null,
    "defaults": []
  }
}
```

A revise response omits the `affordance` key. A create response carries the same shape with `changed` empty (`[]`), `revision` `1`, and an `affordance` string naming the routes that revise or read the component back. The Quickstart above shows one. In `changed`, scalar value transitions record explicit `prior` and `new` keys, while replaced or cleared objects and arrays record only `field`.

`fact_summary` always carries eight keys: `changed`, `series`, `series_cut`, `window`, `unit`, `change`, `source_refs`, and `defaults`. `series_cut` counts series dropped for exceeding the per-summary series cap of 12, plus any further series dropped once the summary crosses its token budget. `window` and `unit` are null for a member the grammar defines neither for. A component's `unit` takes one of four forms, and `fact_summary.unit` carries the same value. A unit the frame gives as a dcid is a string holding the unit's graph name, or the dcid when the graph has no name, cut to 200 characters. A converted unit is the frame's `{currency, prices, base_year}` object as stored. A composed unit is `{numerator, denominator}`, where each side is a string unit, a converted unit, or `{stat_var, label}` for a side that counts a StatVar and declares no unit; `label` holds the StatVar's graph name, or its ref without `dcid:` when the graph has no name, cut to 200 characters. A series whose rows declare no unit has a null `unit`. `change` carries a stat_card's derived comparison and is null for every other member. `source_refs` carries a source_card's provenance list and is null for every other member. `defaults` lists each choice the component made where its intent left one open, as `{pointer, detail}`, and is empty when it made none. A create's `fact_summary` equals the `fact_summary` a later `GET` of the same revision returns.

A label on a card is the graph name of the ref it stands for, or the ref when the graph has no name. A StatVar that an analysis sum, quotient or share wrote its rows under has no graph node, and its version's description record names it. A step that writes a new StatVar names it in `stat_var`, and the StatVar's label is the name of that step's output, worded from the step's inputs and operation: "Gross ODA disbursements per gross national income" for a quotient, `<left> plus <right>` for a sum and `<numerator> as a share of <denominator>` for a share. An input is named by its version's label, and an earlier step by the name of what it produced. A version whose steps name no StatVar, or that holds no steps, words the label from the composed entry its stored body kept: `<first> per <second>` for a quotient, `<first> plus <second>` for a sum, `<first> minus <second>` for a difference, and `<first> as a share of <second>` for a share, where a side reads as its StatVars' labels joined by " and " and a StatVar with no label reads as its dcid without `dcid:`. With equal sides such an entry reads `Ratio of <side>` for a quotient, `Difference in <side>` for a difference, `Share of <side>` for a share, and `<side>` for a sum. Each label starts with a capital and holds at most 200 characters, and a label of two sides is shortened side by side, so both sides and the joining word stay in it. Series labels, the value axis label, a stat_card's `label`, a table's `stat_var` cell, the fact summary, and an export's `series` and `stat_var_label` columns carry that label. A StatVar no step and no composed entry names, and no graph name, keeps its ref.

`composition_id` and `composition_revision` name the real composition and its current head revision on a create, a substantive revise, a no-op revise, and a `GET`, for a component that belongs to a composition. Both are null on every operation for a component that belongs to no composition.

`resolved_payload` is the stamped envelope stored for the reply's revision, verbatim: `{schema_version, catalog_version, component_id, revision, provenance, component}`, plus `description` on a described card, validated when minted against the envelope schema of the catalogue release its `catalog_version` names. [Release Door](#2-release-door) serves only the pinned release's `envelope_schema` and `component_schemas`, the per-member-type schemas `envelope_schema`'s `component` field `$ref`s by `$id`. A deployment reads every stored revision. The envelope keeps the `catalog_version` it was minted under, and the `fact_summary` is formatted with the formatter profile stored on the revision. A create or revise mints under the pinned release, and a revise under a new release mints a new revision carrying it, even when the component content is unchanged. A write whose head carries a newer `catalog_version` than the pinned release refuses `release_mismatch`, so a write cannot move a head back to an older release. Only an instance left behind by a rollback refuses this way, and its `retry_after_delay` means retrying after the newer release is deployed again. `component` carries every resolved point, so a renderer draws from it without a frame read. A no-op revise returns the head revision's stored envelope unchanged. The create, revise, and read replies for one revision carry the same stored envelope, as equal parsed JSON values. The create and revise doors serve it in mint key order. `GET` serves it in the key order Postgres returns from JSONB storage, and a number can come back reformatted (`1e2` as `100`). A consumer that compares or hashes the bytes should canonicalise first.

A card whose frame's version holds a description record carries `description`, composed from that record, the series the card draws and the words of the day it was minted: `describes`, `title`, `about`, `sources`, `freshness`, `steps`, `caveats`, `cite`, `gaps` and `built_from`. A source card's description names the sources it lists. A described card's `provenance` keeps lineage alone (`inputs`, `execution_ref`, `generated_at`, `refreshed_from`), since the description carries the sources, freshness, caveats and citation. Each `source_ref` and `source_refs` entry on a described card names a source's provenance dcid, a `description.sources[].ref`. A described card's `about` names its version's measure with its definition, the values the version's definition fixes except those a measure label or noun states (a partition label states every measure axis it covers, and a noun the constraints its key covers, so a measure axis a noun leaves uncovered stays among them), each with its definition, and per dimension the codes the card draws with theirs, listed only when it draws 50 or fewer. A card drawn from a computed version names no measure or fixed values. A card over a version whose definition resolved to one World Development Indicators variable (`worldBank/`) takes that variable's graph description, without its series code, as its measure's definition when the measure has no authored one. The frame store reads that one description with the frame's labels. A version's caveats reach a described card narrowed to what it draws: a caveat names only the drawn variables and entities, without the version-wide `count`, and one naming variables or entities the card draws none of is left out. A coverage gap names places that hold no data, which no card draws, so a card that draws every entity the version holds in each of its roles keeps the version's coverage gaps whole, places and count. A card over a version with no record, a card naming a source with no provenance dcid, and a source card listing no source carry no `description`, and their `provenance` holds `sources`, `freshness`, `warnings` and `citation` as well, with `source_ref` naming a `provenance.sources[].id`. The example below is such a card. A release whose envelope schema has no `description` mints every card without one. Under a release whose schema has one, a `note_narration` carries no `description` and its `provenance` keeps lineage alone.

#### Read Reply (`GET`)

`GET /api/presentation/components/{component_id}` returns the head revision's stored intent, its fact summary with `changed` empty, and its resolved payload. After the revise above:

```json
{
  "component_id": "cmp_01ARYZ6S41YYYYYYYYYYYYYYYY",
  "revision": 2,
  "catalog_version": "2026.8.0",
  "composition_id": "cps_01ARYZ6S41ZZZZZZZZZZZZZZZZ",
  "composition_revision": 1,
  "stored_intent": {
    "type": "line_chart",
    "frame_ref": {
      "spec_id": "spec_9f2e41a7c03b58d2a4c1f6b2e8d35790",
      "content_version": "cv_f4a81c22d9075e3ba61c2ee98f83b405"
    },
    "select": {
      "stat_var": "dcid:MalariaGrantDisbursements",
      "recipient": "country/KEN"
    },
    "window": { "start": "2015", "end": "2024" },
    "title": "Malaria grant disbursements, Kenya"
  },
  "fact_summary": {
    "changed": [],
    "series": [
      {
        "id": "series_1",
        "label": "Malaria grant disbursements, Kenya",
        "coordinates": { "recipient": "country/KEN" },
        "point_count": 9,
        "first_value": "12.4",
        "last_value": "18.7"
      }
    ],
    "series_cut": 0,
    "window": { "start": "2015", "end": "2024" },
    "unit": null,
    "change": null,
    "source_refs": null,
    "defaults": []
  },
  "resolved_payload": {
    "schema_version": 1,
    "catalog_version": "2026.8.0",
    "component_id": "cmp_01ARYZ6S41YYYYYYYYYYYYYYYY",
    "revision": 2,
    "provenance": {
      "inputs": [
        {
          "spec_id": "spec_9f2e41a7c03b58d2a4c1f6b2e8d35790",
          "content_version": "cv_f4a81c22d9075e3ba61c2ee98f83b405"
        }
      ],
      "execution_ref": null,
      "sources": [
        {
          "id": "src_oecd_dac",
          "name": "OECD DAC",
          "url": "https://stats.oecd.org/",
          "licence": "CC BY 4.0",
          "retrieved": "2026-06-12"
        }
      ],
      "freshness": {
        "data_through": "2024",
        "last_refreshed": "2026-06-12",
        "next_expected": "2026-10",
        "partial_period": false
      },
      "warnings": [],
      "generated_at": "2026-07-14T10:02:00Z",
      "citation": "OECD DAC (2026), via the Development Finance Observatory",
      "refreshed_from": null
    },
    "component": {
      "type": "line_chart",
      "title": {
        "text": "Malaria grant disbursements, Kenya",
        "authorship": "model"
      },
      "series": [
        {
          "id": "series_1",
          "label": "Malaria grant disbursements, Kenya",
          "unit": null,
          "source_ref": "src_oecd_dac",
          "points": [
            { "period": "2015", "value": 12.4 },
            { "period": "2016", "value": 13.1 },
            { "period": "2017", "value": 14.0 },
            { "period": "2018", "status": "missing" },
            { "period": "2019", "value": 15.2 },
            { "period": "2020", "value": 15.9 },
            { "period": "2021", "value": 16.8 },
            { "period": "2022", "value": 17.3 },
            { "period": "2023", "value": 18.1 },
            { "period": "2024", "value": 18.7 }
          ]
        }
      ],
      "x_axis": { "window": { "start": "2015", "end": "2024" } },
      "y_axis": {
        "label": "dcid:MalariaGrantDisbursements",
        "scale": "linear",
        "range": { "min": null, "max": null }
      }
    }
  }
}
```

#### Intent grammar: scale, range, and change

`line_chart` and `bar_chart` accept an authored `scale`, `linear` or `log`, and an authored `range`, `{min, max}`, each a number or null. Both `range` keys are required together. Absent means `linear` and an unbounded range that renders from the data. `line_chart` resolves these onto `y_axis`. `bar_chart` resolves them onto `value_axis`, alongside `category_axis`, which names either an entity comparison (one bar per selected entity at their shared observed period) or a period comparison (one series across a window).

A `bar_chart` whose `select` lists values on one dimension draws one bar per value, in list order. A `select` that resolves to one series draws it across the window. A `select` that lists no values and resolves to several series draws one bar per value of the one dimension they vary on, other than a facet, largest first, when the window holds data in one period and every series holds a value in it; `fact_summary.defaults` states this at `/select/<dimension>`, and listing values on that dimension narrows the chart. Any other selection refuses `binding_invalid` at `/select`: series that vary on two dimensions name both with their value counts and list the values of the ones to bind to one value, series that hold data in several periods name the periods and the two calls that fit, one value to draw its periods or a list to compare at the latest shared period, and series of which some lack a value in the one period name the values that lack it and the call that lists the values to draw. `alternatives` carries the values of the first varying dimension, or for a period some series lack, the values observed in it.

`stat_card` accepts an authored `change`. `basis` names two periods, `[from, to]`, both of which must be observed points of the primary series. `derivation` selects how the comparison is computed, `difference` or `percent_change`, absent meaning `difference`. `percent_change` needs a positive `from` value; a zero or negative basis has no defined ratio and refuses `binding_invalid` at `/change`. The resolved `change` carries `derivation`, `basis`, and the computed `value`.

### Revision-scoped reads

`GET /api/presentation/components/{component_id}?revision=N` scopes the read to one stored revision: its stored intent, its fact summary, and its own resolved payload. Every read carries `retention`, the stored frame's: `{"pinned": true}`, or `{"pinned": false, "held_until": ...}` 30 days after DFO stored the frame, and null for a note. A card stays in the working tier until a share first includes it. An unknown component id, or a revision number outside the component's revision range, returns 404 `not_found`, retryable `false`. `0` and a negative number return `request_invalid`, since `revision` accepts a positive integer only. Any revise or read whose target has vanished or been rewritten between this call's own read and its write refuses `edit_conflict`. A revise that starts after another revise of the same component commits rebases onto the new head. Two overlapping revises of a standalone component leave one refusing `edit_conflict` (409, retryable `true`). Revises of composition members serialise on the composition and rebase.

### Frame Coordinates and JSONB Serialisation

Component creation and revision resolve series coordinates (`fact_summary.series[].coordinates`) from bindings against the frame.

A selector may leave the facet dimensions `unit`, `measurementMethod`, and `provenance` unbound unless one of them splits a series. Within one selector's matches, `assembler.resolve` keys each series on every coordinate except the unbound facets, and refuses `binding_invalid` at `/select/<facet>` when one key holds more than one value of a facet, a null counting as a value. The first such facet in column order is named, and `alternatives` lists the non-null values that key holds. Series that differ in other coordinates may carry different facet values, so GDP in `USDollar` beside population with a null unit resolves as two series. A line chart over them refuses `comparable_units_required`. A series its source's estimates continue past its last period (a dataset version's `trace.continuations`, or the continuations a computed version stored when it was published from the versions it was computed from) is one series: each row of the continuing method files under the series it extends (`assembler.frame_view.series_keys`), so the series resolves with `measurementMethod` unbound, each row keeps its own method cell, the frame view counts it once, and a table row names its own method. A selector that binds `measurementMethod` resolves rows by their own cells, so binding the continuing method draws the estimates alone.

PostgreSQL persists these coordinate mappings in a dedicated `series_coordinates` JSONB column alongside the revision payload in `presentation.component_revision`. On `GET /api/presentation/components/{component_id}`, the response reconstructs coordinate definitions matching the original create/revise response.

List-valued parameters (such as `series_coordinates` and `blocks`) bound in SQL statements are pre-serialised to JSON text strings to match PostgreSQL JSONB types under `pg8000`.

---

## Rendering a revision on another frame

A live share redraws a card on current data without changing the card. `POST /api/presentation/components/{component_id}/revisions/{revision}/render` takes the frame to draw on:

```json
{
  "frame": {
    "spec_id": "spec_9f2e41a7c03b58d2a4c1f6b2e8d35790",
    "content_version": "cv_0b1c..."
  }
}
```

It answers 200 with the body a read of that revision answers (`component_id`, `revision`, `catalog_version`, `composition_id`, `composition_revision`, `stored_intent`, `fact_summary`, `resolved_payload`) and `Cache-Control: no-store`. `resolved_payload` and `fact_summary` are built from the revision's stored intent on the named frame, under the deployment's catalogue release. `resolved_payload.provenance.inputs` names that frame and `resolved_payload.provenance.refreshed_from` is the revision number. Where the named frame's version holds a description record, `resolved_payload.description` is composed for the request from that day's words and carries `rendering: "live"` and `frame`, the `{id, content_version}` it drew on. `stored_intent` is the revision's own, so its `frame_ref` names the frame the revision was stored on. The body also carries `retention`, the named frame's. The route appends no revision, leaves the head where it is, and pins nothing, so the named frame stays in the working tier. A dataset target is checked against the rendering the revision recorded when the card was made.

`GET .../render/data?spec_id=&content_version=` serves the rows behind that drawing. `format` and `bundle` behave as on the stored revision's `/data` route. The file is dated by the frame's retrieval date, a bundle's `dfo.component.frame_ref` names the frame, and the bundle carries no `dfo.reproducing_call`, since a reader cannot call this route. The route pins nothing.

Both routes admit one caller, the live-refresh principal: the value of `LIVE_REFRESH_PRINCIPAL` in the `DATASETS_PRINCIPAL_HEADER` header (default `x-dfo-principal`). The website and `ai` hold that value as a shared secret and put it in the header themselves. Nothing in this service or in front of it sets, replaces, or removes the header, so knowing the value is what admits a caller. Any other caller is refused `refresh_not_permitted` (403) before the path, query, or body is read. An unset or empty `LIVE_REFRESH_PRINCIPAL` refuses every caller.

The frame must be able to stand in for the frame the revision is stored on:

- the stored frame itself is always accepted;
- a dataset frame (`spec_`) is accepted when it has the stored frame's `spec_id` and is a frame the frame refresh answers with for the stored frame (`datasets_api.doors.refresh.refresh_can_answer`). A plain card accepts any raw version of its dataset, whatever files are linked under it. A card on a groupby sum accepts a collapse recording that groupby, and a raw version linking a file of it, which is the sum when it leaves the frame unchanged. A stored frame whose raw version links a plain file and one or more groupby files accepts only a raw version linking a file of each of those groupbys. A stored frame whose raw version links files of two or more different groupbys and no plain file, or whose collapsed version links files of two or more different groupbys, accepts no other frame, since its records do not say which sum it is, and the frame refresh refuses it `rendering_not_recorded`;
- a derived frame (`ops_`) is accepted when the recipe store holds a recipe for it and for the stored frame with the same `normal_form_hash`.

Any other frame is refused `render_target_mismatch` (409, not retryable). That covers another dataset or rendering, a dataset frame for a card on a derived frame and the reverse, a version or recipe that is not on record, and a `note_narration`, which binds no frame.

The render routes serve each refusal at the status the code registry gives its code on `presentation.render`, which for `binding_invalid`, `constraint_violation`, `frame_unavailable`, `release_mismatch`, and `resource_exhausted` differs from the write routes' status:

| Error Code               | HTTP Status | When                                                                                                                                                            |
| :----------------------- | :---------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `refresh_not_permitted`  | 403         | The caller is not the live-refresh principal.                                                                                                                   |
| `render_target_mismatch` | 409         | The frame may not stand in for the stored one.                                                                                                                  |
| `not_found`              | 404         | Unknown component or revision. On `/render/data`, also a `source_card`, which shows no data values.                                                             |
| `edit_conflict`          | 409         | The component has no current revision.                                                                                                                         |
| `request_invalid`        | 400         | A malformed id, body, `format`, or `bundle`.                                                                                                                    |
| `schema_invalid`         | 400         | The stored intent fails this release's schema.                                                                                                                  |
| `binding_invalid`        | 422         | The stored intent does not bind the frame: a stored binding with no value left on the frame, or a single or facet value the frame lacks while it holds several. |
| `constraint_violation`   | 422         | The drawing fails a catalogue constraint on the frame's data.                                                                                                   |
| `frame_unavailable`      | 409         | The frame does not load. `retryable` follows the cause, as on the write routes.                                                                                 |
| `release_mismatch`       | 409         | The revision was minted under a newer catalogue release than this deployment's.                                                                                 |
| `resource_exhausted`     | 409         | The drawing passes a resource maximum.                                                                                                                          |
| `service_unavailable`    | 503         | A store, the frame store included, fails to answer, or the door is at capacity.                                                                                |

On these routes `retryable` is true only for a refusal the same request may pass later, one whose code's retry mode is `retry_after_delay` or `refresh_then_retry` (`codes.replay_retryable`). The caller replays a stored intent and cannot change it, so `binding_invalid`, `constraint_violation`, `schema_invalid`, and `request_invalid` answer `retryable: false`.

A render binds the stored `select` to the frame one dimension at a time. A dimension whose selected values the frame all holds binds as stored. A list that held every value the stored frame holds for a dimension other than a facet (`unit`, `measurementMethod`, `provenance`) binds every value the new frame holds, so a card on every recipient draws the recipients the frame holds now. Any other list keeps the values the frame holds and drops the rest. A single value, or a facet value, binds as stored; when the frame lacks it and holds exactly one value for that dimension, it binds that value. On a frame with a rolling anchor the rendering names the anchor's left-out series that the select admits, whatever the intent states, and then each value the stored frame or the select held that the frame lacks, once. A described card names them in one `coverage_gap` caveat first in `description.caveats`, on the card's `select` step at that period. A card without a description names them in one `coverage_gap` warning first in `provenance.warnings`, for example `2 series have no 2025 value: Kenya, Chad`. On a frame without an anchor the period named is the window's end. `stored_intent` keeps the select as written. A create or revise with no relative period refuses a selected value the frame lacks.

### Rolling periods on a derived frame

A derived frame's recipe can record a rolling anchor: the period `latest` names on that frame and the series its rule left out. The frame store attaches it to the loaded frame, so a card with a relative period on a derived frame is created, revised, and rendered at the recipe's anchor, and the left-out series the card's select admits are named in one `coverage_gap` caveat on the card's selection, or in `provenance.warnings` as one `coverage_gap` entry on a card without a description. A left-out series whose coordinates hold a value the select does not name is left unnamed. Each left-out series is named by the labels of its entity coordinates, read from the graph as a frame's own refs are, and by the raw ref where the graph holds no name. A dataset frame has no recipe, and the assembler derives its anchor from the card's selected series.

## Refusals and Error Classification

Errors conform to RFC 9457 Problem Details.

The per-response `retryable` field is authoritative for every code. `retry_modes` on the release document names each code's headline mode, the mode a caller reads without seeing any individual response. `schema_invalid` and `frame_unavailable` carry more than one real mode, so `retryable` differs by the cause behind a given occurrence of either. `request_invalid` reports retryable `false` on every occurrence, even though its headline mode is `correct_and_retry`.

| Error Code                   | HTTP Status | Retry Mode           | Description / Disclosure Fields                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| :--------------------------- | :---------- | :------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `schema_invalid`             | 400         | Contextual           | Intent fails schema. A missing required field is `correct_and_retry` and returns the type's minimal legal intent in `detail`. An illegal field for the type is `do_not_retry` and returns the type's legal fields as `alternatives`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `binding_invalid`            | 400         | `correct_and_retry`  | Unbound selectors or invalid values. Returns `alternatives`, `truncated`, and `issues`. Each alternative pairs a legal value with the label the frame resolved for it, falling back to the value itself when the frame names none.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `constraint_violation`       | 400         | `correct_and_retry`  | Violates catalogue constraints. Returns `constraint_id`, `pointer`, and `detail`. `pointer` is expressed against the intent, for example `/scale`, `/range`, or `/change/basis`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `edit_conflict`              | 409         | `refresh_then_retry` | The component, a requested revision, or on create the referenced composition vanished or was rewritten mid flight, raised by a scoped or unscoped `GET` as well as by writes.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `frame_unavailable`          | 404         | Contextual           | `cause.code` carries the reason named here, so a client tells a frame it can rebuild from one taken down. `unknown` (404, not retryable) names a `spec_id`/`content_version` pair that was never stored. `expired` (404, retryable) names a stored version that aged out, and its `detail` ends by naming a fetch of the ref with latest=True to draw from current data. `ops_unsupported` (404, retryable) names an `ops_`/`ocv_` pair analysis-mcp minted but never published to the platform; publishing it is the retry that succeeds. `revoked` and the other permanent reasons (`missing_rendering`, `digest_mismatch`) return 404, not retryable. `quarantined` (404, not retryable) names a `spec_id` the spec registry holds in quarantine. A frame store that fails to answer (`io`), and a Datasets surface at capacity on a bare `spec_id` (`busy`), refuse `service_unavailable` instead. |
| `resource_exhausted`         | 400         | `do_not_retry`       | Selection exceeds allowed limits, including a minted component whose `component`, `provenance` and `description` subtrees serialise past `MAX_ENVELOPE_BYTES` (2 MiB, compact UTF-8 JSON), refused before any store write. Returns `limit_name` in `detail`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `release_mismatch`           | 503         | `retry_after_delay`  | The component's head carries a newer catalogue release than this deployment writes, which happens after a rollback and clears once the newer release is deployed again, or a stored payload has a shape the release's schema does not recognise, such as a bar_chart row migration 0006 has not rewritten.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `request_invalid`            | 400         | `correct_and_retry`  | Retryable is always `false`. A malformed body field, an unparseable path identifier, or a non-positive `revision` query parameter. Also a create `write_token` reused with a different request.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `unsupported_schema_version` | 422         | `do_not_retry`       | A top-level `schema_version` field in the request body. A platform code outside the release document's `retry_modes`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `refresh_not_permitted`      | 403         | `do_not_retry`       | A render route called by anyone other than the live-refresh principal.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `render_target_mismatch`     | 409         | `do_not_retry`       | A render route given a frame that may not stand in for the stored one.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `not_found`                  | 404         | `do_not_retry`       | Unknown component or composition id (`DraftNotFound`), or a revision number never minted for a revision-scoped `GET`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `write_refused`              | 403         | `do_not_retry`       | A revise without the component's write token, or a create into an existing composition without that composition's token, in `x-dfo-write-token`. Also a draft created before write tokens. `detail` ends with a fix: create a new component.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `allowance_exhausted`        | 429         | `do_not_retry`       | A create or revise past the caller's daily card allowance (`DATASETS_CARD_DAILY_ALLOWANCE`). `Retry-After` names the seconds to the next UTC midnight. A platform code outside the release document's `retry_modes`. A caller holding the `publish` grant, and a create retried with its first `write_token`, spend none.                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `service_unavailable`        | 503         | `retry_after_delay`  | Database failure, a frame store that fails to answer, or door thread pool backpressure exhaustion. |
| `timeout`                    | 504         | `retry_after_delay`  | Execution exceeded door timeout threshold.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `internal_error`             | 500         | `do_not_retry`       | Permanent non-retryable store or system failure.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |

---

## Stored revisions

Every stored `bar_chart` revision uses `category_axis` and `value_axis` keys. Revisions are append-only. A `component_revision` row changes only through a numbered migration under `src/presentation_store/migrations/`.

---

## Configuration and Deployment

The surface mounts through the platform application factory when `DFO_PRESENTATION_SURFACE` is set to `1` or `true`.

### Environment Variables

| Variable                                   | Default           | Description                                                                     |
| :----------------------------------------- | :---------------- | :------------------------------------------------------------------------------ |
| `DFO_PRESENTATION_SURFACE`                 | `false`           | Enables mounting of the Presentation API surface on `/api/presentation`.        |
| `PRESENTATION_COMPONENTS_MAX_CONCURRENCY`  | `4`               | Worker thread pool size and backpressure limit for component read/write routes. |
| `PRESENTATION_COMPONENTS_TIMEOUT_SECONDS`  | `30`              | Maximum execution timeout for component store operations.                       |
| `PRESENTATION_FRAMES_MAX_CONCURRENCY`      | `4`               | Worker thread pool size and backpressure limit for frame inspection routes.     |
| `PRESENTATION_FRAMES_BARE_MAX_CONCURRENCY` | `2`               | Worker thread pool size and backpressure limit for the bare frames route.       |
| `PRESENTATION_FRAMES_TIMEOUT_SECONDS`      | `30`              | Maximum execution timeout for frame store operations.                           |
| `PRESENTATION_STORE_RUNTIME_DB_USER`       | _(Required)_      | Database username for presentation store connection pooling.                    |
| `DATASET_STORE_RUNTIME_DB_USER`            | _(Required)_      | Database username for shared dataset and frame store connections.               |
| `DATASETS_GRAPH_BASE_URL`                  | _(Required)_      | Base URL for upstream graph reader resolution.                                  |
| `UN_GRAPH_BASE_URL`                        | UN System DC      | API root for `undata/` dcids, shared with every surface and the catalogue job.  |
| `ORIGIN_CATALOGUE_URI`                     | _(Required)_      | Origin catalogue root; a missing or unloadable catalogue fails startup.         |
| `DATASETS_SDMX_BASE_URL`                   | _(Required)_      | Base URL for SDMX client endpoints.                                             |
| `DATASETS_ARTIFACT_BUCKET_WORKING`         | _(Required)_      | GCS bucket name for unpinned/working frame artifacts.                           |
| `DATASETS_ARTIFACT_BUCKET_PINNED`          | _(Required)_      | GCS bucket name for pinned presentation frame artifacts.                        |
| `DFO_PUBLIC_SITE_URL`                      | _(Required)_      | Website origin the components door builds export bundle links on.               |
| `LIVE_REFRESH_PRINCIPAL`                   | _(empty)_         | The one caller the render routes admit. Empty refuses every caller.             |
| `PRESENTATION_SHARE_IMPORT_PRINCIPAL`      | _(empty)_         | The one caller the share import route admits. Empty refuses every caller.       |
| `DATASETS_PRINCIPAL_HEADER`                | `x-dfo-principal` | Header a calling service sends its principal value in.                          |

### Concurrency and Thread Pools

Component and frame operations execute in dedicated `ThreadPoolExecutor` instances with bounded semaphores (`PRESENTATION_COMPONENTS_MAX_CONCURRENCY` and `PRESENTATION_FRAMES_MAX_CONCURRENCY`). When execution permits are saturated, component and frame requests return HTTP `503` with problem code `service_unavailable` immediately. The shares door runs on its own executor of `PRESENTATION_COMPONENTS_MAX_CONCURRENCY` threads, and a share request that finds every thread busy waits up to `SHARES_ADMISSION_WAIT_SECONDS` (5 s) for one before it returns the same `503`, so a share page reading more slots than the executor holds is served.
