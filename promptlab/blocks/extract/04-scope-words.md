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
is never a role_span: "African countries" takes its flow_role and role_span from the
words around it, exactly as "Africa" would. Do not infer qualifiers that the query did
not state.

A named organisation that lends, gives or receives the flow, such as the IMF, the World
Bank, IDA or the Global Fund, is an entity like a place wherever the query names it, and
its name stays out of the variable. Named in front of the measure, it reads like a
possessive, so its direction follows the measure: "IMF credit outstanding Argentina" has
the variable "credit outstanding" and the entities "IMF", origin with role_span "IMF"
(its own span, as "French" is in "French aid"), and "Argentina". An organisation named
beside a measure it does not lend, give or receive is the source of the numbers and no
entity: "GDP of Kenya according to the IMF" and "Kenya GDP IMF" name only Kenya.
When two names stand together for one body, the entity is the narrower one alone:
"World Bank IDA concessional flows" names the entity "IDA".

A measure the query already names by an indicator's own name or acronym keeps the
query's word for it: "aid", "ODA", "OOF" and "GDP" stay as written in the variable, never
spelt out or replaced by another name.

A word in a measure's phrase that narrows it to what one kind of provider gives, or to
one kind of terms, such as "bilateral", "multilateral" or "concessional", stays in that
variable and is also listed in `measure_qualifiers`, copied exactly as written, with that
variable's index: "total bilateral and imputed multilateral aid to Ethiopia" has the
variables "bilateral aid" and "imputed multilateral aid" and lists "bilateral" for
variable 0. A word that is part of a longer name the measure goes by, such as "imputed
multilateral", is not listed.
