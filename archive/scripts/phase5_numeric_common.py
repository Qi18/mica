"""Strict integer parser shared by Phase5 numeric training and evaluation."""
from __future__ import annotations

import re

STRICT_INTEGER_RE = re.compile(r"\s*(?:<answer>\s*)?(-?\d+)(?:\s*</answer>)?[。.!]?\s*", re.I)


def parse_integer(text: str) -> int | None:
    """Return an integer only when the entire completion is one unambiguous answer."""
    match = STRICT_INTEGER_RE.fullmatch(text)
    return int(match.group(1)) if match else None
