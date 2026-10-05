-- Rewrites `component_revision.resolved_payload` for every stored bar_chart
-- revision. The component object's `x_axis` key becomes `category_axis` and
-- `y_axis` becomes `value_axis`. Values stay unchanged and every other key
-- stays untouched. The scope is
-- `resolved_payload->'component'->>'type' = 'bar_chart'`: `component` sits
-- beside `component_id`, `revision` and `provenance` under the envelope, and
-- its own `type` field names the member
-- (`assembler.mint.dispatch.stamp_envelope`'s own shape). The WHERE clause
-- requires both `x_axis` and `y_axis` to be present: a row missing either
-- key, including one already carrying `category_axis`/`value_axis`, fails
-- the check, so a repeat run of this file touches zero rows.

UPDATE presentation.component_revision
SET resolved_payload = jsonb_set(
    resolved_payload,
    '{component}',
    ((resolved_payload -> 'component') - 'x_axis' - 'y_axis') || jsonb_build_object(
        'category_axis', resolved_payload -> 'component' -> 'x_axis',
        'value_axis', resolved_payload -> 'component' -> 'y_axis'
    )
)
WHERE resolved_payload -> 'component' ->> 'type' = 'bar_chart'
  AND resolved_payload -> 'component' ? 'x_axis'
  AND resolved_payload -> 'component' ? 'y_axis';
