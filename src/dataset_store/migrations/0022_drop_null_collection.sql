-- Removes the `collection` key from version bodies that store it as JSON null.
--
-- An earlier `encode_version_body` dumped every `Manifest` field, so each definition body
-- stores `"collection": null`. `Manifest` has no `collection` field and forbids unknown keys, so
-- each such body fails validation on every snapshot read, head hit, re-mint and stored rendering
-- until the key is dropped. No other key changes.
--
-- A collection body's non-null `collection` stays, and a body without the key, such as a
-- published `ops_` body, is left alone. A repeat run changes nothing.

UPDATE datasets.version_record
SET immutable_body = immutable_body - 'collection'
WHERE jsonb_typeof(immutable_body -> 'collection') = 'null';
