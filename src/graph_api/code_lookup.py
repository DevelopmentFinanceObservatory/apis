"""The `code` kind's lookup over the code index, which the resolver reads."""

from __future__ import annotations

from dataclasses import dataclass

from code_index.query import CodeIndex, fold_text
from entity_resolution import CodeHit, Entity


@dataclass(frozen=True, slots=True)
class CodeIndexCodes:
    """The `code` kind's lookup (`entity_resolution.CodeLookup`) over the code index.

    A code matches when its label or a synonym equals the text under `code_index.fold_text`;
    the hit carries the first of its texts that does, the label before any synonym. A code's
    system is named by its graph id, the declaration's `enum_dcid`, which is also the hit's
    `graph_type`. A code whose system declares no graph id cannot be named by one, so it is
    left out."""

    index: CodeIndex

    def codes_by_text(self, text: str, *, code_system: str | None) -> list[CodeHit]:
        folded = fold_text(text)
        hits: list[CodeHit] = []
        for row in self.index.codes_by_text(text):
            declaration = self.index.declarations.get(row.code_system)
            enum_dcid = declaration.enum_dcid if declaration is not None else None
            if enum_dcid is None or (code_system is not None and enum_dcid != code_system):
                continue
            entity = Entity(dcid=row.dcid, name=row.label, kind="code", graph_type=enum_dcid)
            listed = next(t for t in (row.label, *row.synonyms) if fold_text(t) == folded)
            matched_as = "name" if listed == row.label else "alias"
            hits.append(CodeHit(entity=entity, text=listed, matched_as=matched_as))
        return hits
