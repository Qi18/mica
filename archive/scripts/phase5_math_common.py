"""Shared exact option parser for Phase5 training and evaluation."""
from __future__ import annotations

import re

STRICT_CHOICE_RE = re.compile(r"\s*(?:<answer>\s*)?([ABCD])(?:\s*</answer>)?[。.!]?\s*", re.I)
ANSWER_SENTENCE_RE = re.compile(
    r"(?i)\b(?:the\s+)?correct\s+answer\s+is\s*[:：]?\s*([ABCD])(?=[\s.、:：]|$)"
)
LEADING_LABEL_RE = re.compile(r"(?im)(?:^|\n)\s*([ABCD])(?=[\s.、:：]|$)")


def strict_choice(text: str) -> str | None:
    match = STRICT_CHOICE_RE.fullmatch(text)
    return match.group(1).upper() if match else None


def parse_choice(text: str) -> str | None:
    """Return one unambiguous leading/correct-answer option label, otherwise None."""
    strict = strict_choice(text.strip())
    if strict:
        return strict
    matches = [m.group(1).upper() for m in ANSWER_SENTENCE_RE.finditer(text)]
    matches.extend(m.group(1).upper() for m in LEADING_LABEL_RE.finditer(text))
    return matches[0] if matches and len(set(matches)) == 1 else None
