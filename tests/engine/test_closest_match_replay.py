"""The closest-match rule replayed over every reading the recorded sentence panel shows.

`tests/fixtures/sentence_panel/*.jsonl.gz` holds each panel query's parts with their Specs
(`scripts/sentence_golden.py`), each Spec with its members' sentence facts. Each Spec of a
definite or candidates part is one reading, judged as assembly judges it
(`spec_closest_match_fact`): its heading's measure noun (`heading_for`) and the inputs
`closest_match.reading_inputs` builds from the Spec. The rule runs on each at every
retrieval score, and the readings it fires on must be exactly `PANEL_FIRES`. The offline corpus asserts its own firing set beside its other invariants
(`tests/eval/test_corpus_offline.CLOSEST_MATCH_FIRES`).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from scripts.sentence_golden import golden_parts
from search.engine.closest_match import spec_closest_match_fact
from search.models import CandidatesPart, DefinitePart, Spec
from search.render.heading import heading_for

PANEL_DIR = Path(__file__).parents[1] / "fixtures" / "sentence_panel"

PANEL_FIRES = frozenset(
    {
        # "loans from Japan to Vietnam for transport" answered with gross ODA disbursements
        # of every finance type.
        ("crs-033", "loans"),
        # One reading of two: DAC1's contribution to regional development banks, for the
        # declared measure "imputed multilateral ODA".
        ("mls-030", "imputed multilateral ODA"),
        # The declared measure GDP answered with GNI at purchasing power parity per capita.
        ("weo-020", "GDP per capita PPP"),
        ("weo-035", "GDP per capita PPP"),
        ("weo-043", "GDP per capita PPP"),
        # An IMF indicator code answered with GDP.
        ("weo-039", "GGXWDG_NGDP"),
    }
)
"""The (query id, variable text) readings the rule fires on. "aid" answered with ODA gross
disbursements, from CRS, DAC1 or DAC2a, stays silent."""


def _fires(spec: Spec, variable_text: str) -> bool:
    fact = spec_closest_match_fact(
        spec,
        variable_text=variable_text,
        measure_noun=heading_for(spec).heading.noun.inline_text,
    )
    return fact is not None


def test_the_rule_fires_on_the_panel_readings_it_names() -> None:
    fired: set[tuple[str, str]] = set()
    readings = 0
    for path in sorted(PANEL_DIR.glob("*.jsonl.gz")):
        for text in gzip.decompress(path.read_bytes()).splitlines():
            line = json.loads(text)
            for part in golden_parts(line["parts"]):
                if part.variable_text is None:
                    continue
                if isinstance(part, DefinitePart):
                    specs = [part.spec]
                elif isinstance(part, CandidatesPart):
                    specs = part.candidates.specs
                else:
                    continue
                for spec in specs:
                    readings += 1
                    if _fires(spec, part.variable_text):
                        fired.add((line["id"], part.variable_text))
    assert readings, f"no reading under {PANEL_DIR}"
    assert fired == PANEL_FIRES
