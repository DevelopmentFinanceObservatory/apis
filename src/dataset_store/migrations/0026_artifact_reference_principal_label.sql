-- A pin reference names its caller by `datasets_api.principals.Principal.label`:
-- `secret-sha256:` and the first 16 hex digits of the SHA-256 of the
-- presented header value. Pin rows written before it hold the raw shared
-- secret in `principal`. This migration rewrites each such row to the label
-- of its secret, so the stored value is what the pin door writes now and one
-- caller keeps one key.
--
-- Only `pin` rows carry a secret. Presentation rows name a component
-- (`presentation:<component_id>`) and stay as they are.
--
-- A raw row whose label already has a row with the same digest, subject,
-- version and kind (a pin the labelled code wrote) folds into that row: it
-- keeps the earlier `created_at`, and it stays live if either row is live.

UPDATE datasets.artifact_reference AS labelled
SET created_at = LEAST(labelled.created_at, raw.created_at),
    removed_at = CASE
        WHEN labelled.removed_at IS NULL OR raw.removed_at IS NULL THEN NULL
        ELSE GREATEST(labelled.removed_at, raw.removed_at)
    END
FROM datasets.artifact_reference AS raw
WHERE raw.reference_kind = 'pin'
  AND NOT starts_with(raw.principal, 'secret-sha256:')
  AND labelled.principal
      = 'secret-sha256:' || left(encode(sha256(convert_to(raw.principal, 'UTF8')), 'hex'), 16)
  AND labelled.artifact_digest = raw.artifact_digest
  AND labelled.subject_id = raw.subject_id
  AND labelled.content_version = raw.content_version
  AND labelled.reference_kind = raw.reference_kind;

DELETE FROM datasets.artifact_reference AS raw
USING datasets.artifact_reference AS labelled
WHERE raw.reference_kind = 'pin'
  AND NOT starts_with(raw.principal, 'secret-sha256:')
  AND labelled.principal
      = 'secret-sha256:' || left(encode(sha256(convert_to(raw.principal, 'UTF8')), 'hex'), 16)
  AND labelled.artifact_digest = raw.artifact_digest
  AND labelled.subject_id = raw.subject_id
  AND labelled.content_version = raw.content_version
  AND labelled.reference_kind = raw.reference_kind;

UPDATE datasets.artifact_reference
SET principal = 'secret-sha256:' || left(encode(sha256(convert_to(principal, 'UTF8')), 'hex'), 16)
WHERE reference_kind = 'pin'
  AND NOT starts_with(principal, 'secret-sha256:');
