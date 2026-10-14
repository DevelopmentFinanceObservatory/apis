"World", "global", "globally", and "worldwide" are scope words, not place entities —
the place resolver only resolves countries and continents, and there is no "World"
place to resolve them against. When a query uses one of these words instead of naming
an actual country or continent, leave `entities` empty (do not extract "World" or
"global" as an entity); the downstream engine treats an empty `entities` list as an
aggregate, entity-free query, which is the correct reading of "global X".

For every other named geographic or group scope, retain its name in `entities` even
when it may not resolve; `entities: []` means the query genuinely names no place or
group. When an unsupported scope, ranking, or subset qualifier is not applied, copy
its exact query wording into `dropped_qualifiers` without translating or rewriting it.
For example, "top 10 countries by population" yields
`"dropped_qualifiers": ["top 10 countries by population"]` and `"entities": []`:
generic universe or class nouns wholly contained in that dropped qualifier, such as
`countries`, are not named scopes. Actual named groups, such as `Least Developed
Countries`, must remain in `entities`. A class noun added to the name of a region or
group asks for its members: "African countries" and "countries in Africa" are the
entity "Africa" with `members` true, where "Africa" alone has it false. The class noun
is never a role_span: put it in `member_span` instead, and let "African countries" take
its flow_role and role_span from the words around it, exactly as "Africa" would. A
parenthetical qualifier that identifies the named entity remains part of both `text` and
`span`: "Asia (DAC)" is one named aggregate, not the continent Asia plus an omitted
qualifier. Do not infer qualifiers that the query did not state.
