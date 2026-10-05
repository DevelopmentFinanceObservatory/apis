"""Starlette URL convertor for slash-delimited Graph dcids.

It rejects empty segments and trailing slashes. Importing the module
registers the convertor before the node routes are built.
"""

from __future__ import annotations

from starlette.convertors import Convertor, register_url_convertor


class DcidConvertor(Convertor[str]):
    """Match non-empty slash-joined segments from `[A-Za-z0-9_.\\-]+`.

    Starlette backtracks for trailing literal segments, but a bare `{dcid}`
    route can absorb them. Routes therefore use disjoint literal prefixes.
    """

    # The handler enforces 512 characters so oversized dcids return 400, not a routing 404.
    # The longest graph fixture dcid is 174 characters.
    regex = r"[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)*"

    def convert(self, value: str) -> str:
        return value

    def to_string(self, value: str) -> str:
        return value


register_url_convertor("dcid", DcidConvertor())

# The node door's own ceiling on a matched `{dcid}`'s length. See the
# comment above `DcidConvertor.regex` for why the regex itself stays
# unbounded while this constant does the length policing.
DCID_MAX_LENGTH = 512
