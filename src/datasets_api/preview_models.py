"""Runtime response models built by `datasets_api.preview`. `/schema` keeps its separate,
contract-stable declaration. A preview carries `schema_version` and `members`, separate from mint
identity and artifact fields. Each member's `content_version` matches a mint of the same request, so
the version lives at member level.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from datasets_api.manifest_models import CitationModel, CoverageModel, WarningModel


class PreviewFrameModel(BaseModel):
    """A bounded row sample: `columns` names each cell in `rows`' own
    fixed order, the same `{"columns": [...], "rows": [[cell, ...], ...]}`
    shape `materialise.formats.Frame1Table`/frame1's own stored document
    use. `rows` already carries at most the request's own `limit` --
    `materialise.materialise_preview`'s own truncation, not a second
    truncation applied here.
    """

    model_config = ConfigDict(extra="forbid")
    columns: list[str]
    rows: list[list[object]]


class PreviewMemberModel(BaseModel):
    """One preview entry. `datasets_api.preview.preview_definition` reports
    one entry for a `spec_id` preview: the definition's members preview as
    the one group its mint serves, and a family of StatVars is named by the
    family reference `dcid:<spec_id>` in `stat_var`. A `statvar_set`,
    `source_codes` or `collection_id` preview that resolves to more than one
    `(stat_var, entity/role-tuple)` pair reports one entry per pair, in
    resolution order (`datasets_api.preview.preview_requests`), as its mint
    serves each pair alone. A single entry is still a one-entry `members`
    list, never a bare top-level member -- one shape for every input kind, so
    a caller never has to branch on member count the way `Manifest`'s own
    `definition`-xor-`collection` split requires.

    `content_version` and `total_rows` are those a mint of the same request
    reports: the preview runs the mint's generation over the full survivor
    set rather than the `limit`-bounded sample, so the two tokens are equal
    for the same definition at any `limit`. `content_version` is `None` only
    for a preview minted with no revision reader, which this door does not
    do: `datasets_api.preview` always passes `deps.revision_reader`.
    """

    model_config = ConfigDict(extra="forbid")
    stat_var: str
    content_version: str | None
    coverage: CoverageModel
    citation: CitationModel
    warnings: list[WarningModel] = Field(default_factory=list)
    sample: PreviewFrameModel
    total_rows: int
    truncated: bool


class PreviewEnvelope(BaseModel):
    """`POST /api/datasets`'s response to a preview request
    (`MintRequest.preview` set). Never a mint result: no `artifacts`, no
    `definition`/`collection` identity block, and never written to the
    registry, the artifact store, or a version/head record -- see
    `datasets_api.preview`'s own module docstring for what stays genuinely
    write-free and what a preview still costs. Each entry carries the
    `content_version` a mint of that entry's own definition would report;
    there is no top-level aggregate.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    members: list[PreviewMemberModel] = Field(min_length=1)
