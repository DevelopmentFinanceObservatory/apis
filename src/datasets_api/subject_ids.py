"""The subject ids a version addressed under one spec id can be recorded under.

A posted, non-canonical spec id resolves to a served id, the canonical form of its identity
that `kinds.spec_id.resolve_served_definition` keys versions on, and the registry stores each
such binding (`datasets.spec_alias`). A version-addressed read uses the stored bindings, so an
address issued under a posted id keeps resolving after a code index export changes the
canonical form. A request naming the served id honours revocations under it, and a request
naming a posted id honours revocations under the posted id and every served id bound to it.
"""

from __future__ import annotations

from datasets_api.codes_snapshot import SpecNotFound
from spec_registry.reader import SpecReader


def served_subject_ids(id: str, *, spec_reader: SpecReader) -> tuple[str, ...]:
    """The subject ids a version addressed under the registered id `id` can be recorded under,
    in the order to read them: the served ids bound to `id`, newest binding first, then `id`
    itself, for a version recorded under the posted id. An id with no binding, as a served id
    is, returns itself alone. One registry transaction reads them
    (`SpecReader.read_served_spec_ids`), and nothing is registered or bound here. Raises
    `SpecNotFound` for an unregistered id and `IdentityQuarantined` when `id` or a served id
    bound to it is quarantined.
    """
    served = spec_reader.read_served_spec_ids(id)
    if served is None:
        raise SpecNotFound(id=id)
    return (*(served_id for served_id in served if served_id != id), id)
