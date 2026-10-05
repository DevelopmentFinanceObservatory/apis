"""The leading characters a spreadsheet reads as a formula, and the one escape every data file,
heading and codes cell is written with, kept free of format dependencies."""

FORMULA_TRIGGERS = ("=", "+", "-", "@", "\t", "\r")
"""The leading characters a spreadsheet reads as a formula, and openpyxl infers one from. A CSV
cell that starts with one, after any leading apostrophes, is written with one apostrophe more."""


def spreadsheet_safe(text: str) -> str:
    """`text` as a tabular writer writes it: with one apostrophe more when, after any leading
    apostrophes, it starts with a formula trigger, else unchanged. Removing that apostrophe
    recovers the original."""
    if text.lstrip("'").startswith(FORMULA_TRIGGERS):
        return "'" + text
    return text
