The query may be written in any language. ALWAYS emit the `variables` and `entities`
fields in English — the downstream indicator search and place resolver are English-only.
Translate foreign-language measures and place names into their standard English form;
preserve proper nouns that have no distinct English equivalent. Dates are
language-independent (emit ISO year strings regardless of the query language).
