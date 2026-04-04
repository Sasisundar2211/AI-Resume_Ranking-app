"""Text cleaning utilities."""

import re


def clean_text(text: str) -> str:
    """Normalize text by removing excess whitespace and control chars."""
    if not text:
        return ""
    normalized = re.sub(r"\s+", " ", text)
    return normalized.strip()
