-- Renames the party-kind enrichment field from `kind` to `party_kind` in every
-- stored output profile, in the `output_profile` fact column and in the
-- `output_profile` key of `immutable_body`.
--
-- An output profile (`materialise.profile.OutputProfile`) lists one entry per
-- enrichment in `enrichments`, each naming its field in `field`.
-- `datasets_api.output_options` drops an entry whose field the deployment does
-- not enable, so a version profiled under `kind` would lose its party-kind
-- coverage from its options document. Each rewrite changes only the entries
-- whose field is `kind` and keeps the order of the list; a column profile's
-- own `kind` (`grain`, `role`, ...) sits under `columns` and is left alone.
--
-- The trigger `version_record_party_kind_field` applies the rewrite to every
-- row inserted from here on. The previous image writes `kind` until the
-- services move and from requests it still holds afterwards, and its insert
-- does nothing on a conflicting row, so a later mint cannot repair a row it
-- stored. Postgres fires BEFORE triggers of the same event in name order, so
-- this trigger fires after `version_record_fill_facts` (migration 0030) and
-- also rewrites the `output_profile` column that trigger copies from the body
-- of a row arriving with only `immutable_body`. The two UPDATEs rewrite the
-- rows stored before this migration. `datasets-maintenance settle` counts the
-- rows still holding `kind` and fails the deploy when one exists. The trigger
-- and its function exist only for the previous image's writes, and a later
-- migration drops them once no running image writes `kind`.
--
-- CREATE TRIGGER takes SHARE ROW EXCLUSIVE on `version_record` and holds it
-- until this migration's transaction commits, so an insert waits until both
-- UPDATEs have run and then passes through the trigger. While the lock waits
-- behind a long write, every later write of the table queues behind it, so
-- `lock_timeout` fails it after five seconds instead. Re-running this
-- migration changes nothing: the function and trigger are replaced by
-- themselves, and both UPDATEs touch only an entry still named `kind`.

SET LOCAL lock_timeout = '5s';

CREATE OR REPLACE FUNCTION datasets.version_record_party_kind_field() RETURNS trigger
LANGUAGE plpgsql AS $party_kind_field$
BEGIN
    IF jsonb_typeof(NEW.output_profile -> 'enrichments') = 'array'
        AND NEW.output_profile -> 'enrichments' @> '[{"field": "kind"}]'
    THEN
        NEW.output_profile := jsonb_set(
            NEW.output_profile,
            '{enrichments}',
            (
                SELECT jsonb_agg(
                    CASE
                        WHEN entry.value ->> 'field' = 'kind'
                        THEN jsonb_set(entry.value, '{field}', '"party_kind"')
                        ELSE entry.value
                    END
                    ORDER BY entry.position
                )
                FROM jsonb_array_elements(NEW.output_profile -> 'enrichments')
                    WITH ORDINALITY AS entry (value, position)
            )
        );
    END IF;
    IF jsonb_typeof(NEW.immutable_body -> 'output_profile' -> 'enrichments') = 'array'
        AND NEW.immutable_body -> 'output_profile' -> 'enrichments' @> '[{"field": "kind"}]'
    THEN
        NEW.immutable_body := jsonb_set(
            NEW.immutable_body,
            '{output_profile,enrichments}',
            (
                SELECT jsonb_agg(
                    CASE
                        WHEN entry.value ->> 'field' = 'kind'
                        THEN jsonb_set(entry.value, '{field}', '"party_kind"')
                        ELSE entry.value
                    END
                    ORDER BY entry.position
                )
                FROM jsonb_array_elements(NEW.immutable_body -> 'output_profile' -> 'enrichments')
                    WITH ORDINALITY AS entry (value, position)
            )
        );
    END IF;
    RETURN NEW;
END;
$party_kind_field$;

CREATE OR REPLACE TRIGGER version_record_party_kind_field
    BEFORE INSERT ON datasets.version_record
    FOR EACH ROW EXECUTE FUNCTION datasets.version_record_party_kind_field();

UPDATE datasets.version_record
SET output_profile = jsonb_set(
    output_profile,
    '{enrichments}',
    (
        SELECT jsonb_agg(
            CASE
                WHEN entry.value ->> 'field' = 'kind'
                THEN jsonb_set(entry.value, '{field}', '"party_kind"')
                ELSE entry.value
            END
            ORDER BY entry.position
        )
        FROM jsonb_array_elements(output_profile -> 'enrichments')
            WITH ORDINALITY AS entry (value, position)
    )
)
WHERE jsonb_typeof(output_profile -> 'enrichments') = 'array'
    AND output_profile -> 'enrichments' @> '[{"field": "kind"}]';

UPDATE datasets.version_record
SET immutable_body = jsonb_set(
    immutable_body,
    '{output_profile,enrichments}',
    (
        SELECT jsonb_agg(
            CASE
                WHEN entry.value ->> 'field' = 'kind'
                THEN jsonb_set(entry.value, '{field}', '"party_kind"')
                ELSE entry.value
            END
            ORDER BY entry.position
        )
        FROM jsonb_array_elements(immutable_body -> 'output_profile' -> 'enrichments')
            WITH ORDINALITY AS entry (value, position)
    )
)
WHERE jsonb_typeof(immutable_body -> 'output_profile' -> 'enrichments') = 'array'
    AND immutable_body -> 'output_profile' -> 'enrichments' @> '[{"field": "kind"}]';
