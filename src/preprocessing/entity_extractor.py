"""Simple entity extraction for resumes."""

from __future__ import annotations

import re
from typing import List, Optional

from .text_cleaner import clean_text

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?:\+?\d{1,2}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})")

SKILL_KEYWORDS = {
    "python": "Python",
    "machine learning": "Machine Learning",
    "ml": "ML",
    "nlp": "NLP",
    "tensorflow": "TensorFlow",
    "pytorch": "PyTorch",
    "docker": "Docker",
    "aws": "AWS",
    "gcp": "GCP",
    "java": "Java",
    "spring": "Spring",
    "sql": "SQL",
}


def extract_email(text: str) -> Optional[str]:
    match = _EMAIL_RE.search(text or "")
    return match.group(0) if match else None


def extract_phone(text: str) -> Optional[str]:
    match = _PHONE_RE.search(text or "")
    return match.group(0) if match else None


def extract_skills(text: str) -> List[str]:
    lowered = clean_text(text).lower()
    found: List[str] = []
    for key, canonical in SKILL_KEYWORDS.items():
        if re.search(rf"\b{re.escape(key)}\b", lowered):
            found.append(canonical)
    return sorted(set(found))
