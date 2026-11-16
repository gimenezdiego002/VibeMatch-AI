"""Song language vocabulary, a rule-based estimator, and manual overrides.

Language is NOT provided by any of the catalog's data sources. The real catalog's language
column is therefore an *estimate*, made once at catalog-build time (by Gemini when a key is
available, otherwise by the rules below) and always overridable by a person through
data/language_overrides.csv. Nothing here runs a model at playlist time.
"""

from __future__ import annotations

import csv
from pathlib import Path
import re
from typing import Iterable, Mapping, MutableMapping


LANGUAGES = (
    "english", "spanish", "portuguese", "french", "german", "italian", "dutch",
    "korean", "japanese", "hindi", "arabic", "indonesian", "other", "instrumental",
)
LANGUAGE_ALIASES = {
    "español": "spanish", "espanol": "spanish", "latin": "spanish", "latino": "spanish", "latina": "spanish",
    "portugues": "portuguese", "português": "portuguese", "brazilian": "portuguese",
    "français": "french", "francais": "french", "deutsch": "german", "italiano": "italian",
    "k-pop": "korean", "kpop": "korean", "j-pop": "japanese", "jpop": "japanese",
    "bollywood": "hindi", "indian": "hindi",
}
SOURCE_GEMINI = "gemini estimate"
SOURCE_RULES = "rule estimate"
SOURCE_MANUAL = "manual override"
SOURCE_UNSPECIFIED = "unspecified"

_SCRIPTS = (
    (re.compile(r"[가-힯ᄀ-ᇿ]"), "korean"),
    (re.compile(r"[぀-ヿ]"), "japanese"),
    (re.compile(r"[ऀ-ॿ]"), "hindi"),
    (re.compile(r"[؀-ۿ]"), "arabic"),
)
_WORDS = {
    "spanish": {"el", "la", "los", "las", "de", "que", "mi", "tu", "te", "amor", "corazón", "corazon", "quiero",
                "baila", "bailar", "noche", "vida", "mujer", "contigo", "sin", "con", "por", "para", "soy",
                "hoy", "ojos", "solo", "nada", "todo", "fuego", "otra", "vez", "dime", "aquí", "aqui", "adiós"},
    "portuguese": {"não", "nao", "você", "voce", "meu", "minha", "coração", "coracao", "saudade", "favela",
                   "obrigado", "pra", "uma", "ela", "eu", "nós", "nos", "beijo", "comigo", "então"},
    "french": {"je", "tu", "le", "les", "mon", "ma", "amour", "pas", "est", "et", "une", "des", "mots", "toi", "moi"},
    "german": {"ich", "du", "und", "der", "die", "das", "nicht", "mein", "liebe", "zusammen", "ist", "wir"},
    "italian": {"il", "che", "non", "sono", "amore", "cuore", "della", "per", "una", "mio", "figli", "delle", "stelle"},
    "dutch": {"het", "een", "ik", "jij", "niet", "wij", "verliefd", "weg", "gaan", "man", "is"},
    "indonesian": {"di", "yang", "aku", "kamu", "cinta", "dan", "saat", "bahagia", "sesuatu", "tahun"},
}
_SPANISH_CHARS = re.compile(r"[ñ¿¡]")
_PORTUGUESE_CHARS = re.compile(r"(ão|ões|ç)")


def normalize_language(value: object) -> str | None:
    """Map free text to the fixed vocabulary, or None if it is not a known language."""
    text = str(value or "").strip().casefold()
    text = LANGUAGE_ALIASES.get(text, text)
    return text if text in LANGUAGES else None


def rule_guess(title: str, artist: str = "", album: str = "", genre: str = "") -> tuple[str, float]:
    """Cheap deterministic guess from script and common words. Returns (language, confidence)."""
    text = " ".join(part for part in (title, album) if part)
    for pattern, language in _SCRIPTS:
        if pattern.search(text):
            return language, 0.9
    words = re.findall(r"[^\W\d_]+", text.casefold())
    scores = {language: sum(word in vocabulary for word in words) for language, vocabulary in _WORDS.items()}
    if _SPANISH_CHARS.search(text.casefold()):
        scores["spanish"] += 2
    if _PORTUGUESE_CHARS.search(text.casefold()):
        scores["portuguese"] += 2
    best = max(scores, key=lambda language: (scores[language], language == "spanish"))
    if scores[best] >= 2:
        return best, 0.6
    if scores[best] == 1:
        return best, 0.35
    genre_hint = {"latin": "spanish", "reggaeton": "spanish", "traditional mexicano": "spanish",
                  "brazilian music": "portuguese", "indian music": "hindi"}.get(genre.casefold())
    if genre_hint:
        return genre_hint, 0.4
    return "english", 0.25


def load_overrides(path: str | Path) -> dict[str, str]:
    """Read data/language_overrides.csv: columns key (spotify_id or isrc) and language."""
    overrides: dict[str, str] = {}
    path = Path(path)
    if not path.is_file():
        return overrides
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            language = normalize_language(row.get("language"))
            key = str(row.get("key", "")).strip()
            if key and language:
                overrides[key] = language
    return overrides


def apply_overrides(rows: Iterable[MutableMapping[str, object]], overrides: Mapping[str, str]) -> int:
    """Overrides always win. Returns how many rows changed."""
    changed = 0
    for row in rows:
        for key in (str(row.get("spotify_id", "")), str(row.get("isrc", ""))):
            if key and key in overrides:
                row["language"] = overrides[key]
                row["language_source"] = SOURCE_MANUAL
                row["language_confidence"] = 1.0
                changed += 1
                break
    return changed


def catalog_languages(songs: Iterable[Mapping[str, object]]) -> set[str]:
    """Languages actually present in a catalog. Rows without the column count as unspecified."""
    found = {normalize_language(song.get("language")) for song in songs}
    return {language for language in found if language}
