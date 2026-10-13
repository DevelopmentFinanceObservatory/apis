"""The name the platform publishes and is cited under.

`description.citation` writes it into citation lines and `materialise.bundle` into a bundle's
contributors, CFF author and contact. Both import `dfo_platform` already, and this
standard-library module adds no dependency between them.
"""

from __future__ import annotations

from typing import Final

DFO_NAME: Final = "Development Finance Observatory"
"""The DFO's name as publisher, contributor, CFF author and contact."""
