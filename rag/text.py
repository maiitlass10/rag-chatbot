"""Arabic-aware text normalization and tokenization (used by keyword search)."""

from __future__ import annotations

import re
import unicodedata

# Harakat (diacritics), tatweel and Quranic marks.
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_TOKEN = re.compile(r"[\w]+", re.UNICODE)

_CHAR_MAP = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ة": "ه",
        "ؤ": "و",
        "ئ": "ي",
    }
)

# A small stop-word list (Arabic + English) so BM25 focuses on meaningful terms.
STOPWORDS = frozenset(
    """
    في من على الى عن مع هذا هذه ذلك تلك التي الذي الذين هو هي هم انا انت نحن كان
    كانت يكون ما ماذا لماذا كيف متى اين هل او ثم قد لا لم لن ان كل بعد قبل عند حتى
    شو ليش وين كيفك يعني هاد هيك
    the a an and or of to in on for is are was were be been with by at as it this
    that from what which who how why when where do does did not
    """.split()
)


def normalize(text: str) -> str:
    """Lower-case and unify Arabic letter variants so search ignores spelling noise."""
    text = unicodedata.normalize("NFKC", text)
    text = _DIACRITICS.sub("", text)
    text = text.translate(_CHAR_MAP)
    return text.lower()


def _strip_prefix(token: str) -> str:
    # Light stemming: drop the definite article "ال" (and "وال"/"بال"...) on longer words.
    for prefix in ("وال", "بال", "كال", "فال", "لل", "ال"):
        if token.startswith(prefix) and len(token) - len(prefix) >= 2:
            return token[len(prefix) :]
    return token


def tokenize(text: str) -> list[str]:
    tokens = (_strip_prefix(t) for t in _TOKEN.findall(normalize(text)))
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]
