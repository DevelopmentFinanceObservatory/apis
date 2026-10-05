"""The search catalogue: what DFO holds and example questions to ask of it.

`python -m search.catalogue.job` builds it from `examples.toml`, one origin
catalogue build and one coverage index build, checks each example against
the search engine, and publishes the result (`search.catalogue.artifact`),
which `GET /api/search/catalogue` serves.
"""
