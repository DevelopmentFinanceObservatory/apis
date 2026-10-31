"""Puts this directory on `sys.path`, so a test imports the support modules beside it
(`frame_fixtures`) and the `differential` package without a `tests` package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
