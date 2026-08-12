"""Deterministic review-text normalization and token processing."""

from __future__ import annotations

import re
import string
from collections.abc import Iterable

ENGLISH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "with",
}

MALAY_STOPWORDS = {
    "adalah",
    "akan",
    "anda",
    "boleh",
    "dan",
    "dari",
    "dengan",
    "di",
    "ini",
    "itu",
    "juga",
    "kalau",
    "kami",
    "ke",
    "kerana",
    "lagi",
    "mereka",
    "pada",
    "saya",
    "tidak",
    "untuk",
    "ya",
    "yang",
}

TOKEN_PATTERN = re.compile(r"\b\w+\b")
PUNCTUATION_PATTERN = re.compile(f"[{re.escape(string.punctuation)}]")


def normalise_whitespace(text: str) -> str:
    """Collapse repeated whitespace to single spaces."""
    return " ".join(str(text).split())


def remove_punctuation(text: str) -> str:
    """Remove punctuation and normalize the remaining whitespace."""
    return normalise_whitespace(PUNCTUATION_PATTERN.sub("", str(text)))


def tokenize(text: str) -> list[str]:
    """Return lowercase alphanumeric word tokens."""
    return TOKEN_PATTERN.findall(str(text).lower())


def remove_stopwords(
    text: str,
    language: str = "english",
    *,
    extra_stopwords: Iterable[str] | None = None,
) -> str:
    """Remove configured or NLTK stopwords from a text string."""
    language_key = language.lower()
    if language_key == "english":
        stopwords = set(ENGLISH_STOPWORDS)
    elif language_key == "malay":
        stopwords = set(MALAY_STOPWORDS)
    else:
        stopwords = set()

    if extra_stopwords:
        stopwords.update(word.lower() for word in extra_stopwords)

    return " ".join(token for token in tokenize(text) if token not in stopwords)


def lemmatise_text(text: str, language: str = "english") -> str:
    """Lemmatize English text when NLTK resources are available."""
    if language.lower() != "english":
        return normalise_whitespace(text)

    try:
        from nltk.stem import WordNetLemmatizer
    except ImportError:
        return normalise_whitespace(text)

    lemmatizer = WordNetLemmatizer()
    try:
        return " ".join(lemmatizer.lemmatize(token) for token in tokenize(text))
    except LookupError:
        return normalise_whitespace(text)


def preprocess_review(text: str, language: str = "english") -> str:
    """Apply lowercase, punctuation, stopword, and lemmatization steps."""
    lowered = str(text).lower()
    without_punctuation = remove_punctuation(lowered)
    without_stopwords = remove_stopwords(without_punctuation, language)
    return lemmatise_text(without_stopwords, language)
