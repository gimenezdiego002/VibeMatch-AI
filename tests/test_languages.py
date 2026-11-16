"""Language vocabulary, rule estimates, overrides, and the build-time Gemini labeler (mocked)."""

import json
from pathlib import Path

import pytest

from src.gemini_service import GeminiService
from src.languages import (
    LANGUAGES, SOURCE_MANUAL, apply_overrides, catalog_languages, load_overrides, normalize_language, rule_guess,
)
from src.recommender import load_songs


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "value,expected",
    [("Spanish", "spanish"), ("español", "spanish"), ("latin", "spanish"), ("K-Pop", "korean"), ("klingon", None), (None, None)],
)
def test_normalize_language(value, expected):
    assert normalize_language(value) == expected


@pytest.mark.parametrize(
    "title,expected",
    [("Corazón de Fuego", "spanish"), ("Não faz isso Comigo não", "portuguese"), ("夜に駆ける", "japanese"),
     ("사랑해", "korean"), ("Blinding Lights", "english")],
)
def test_rule_guess_language(title, expected):
    language, confidence = rule_guess(title)
    assert language == expected
    assert 0 < confidence <= 1


def test_rule_guess_uses_genre_only_as_weak_hint():
    language, confidence = rule_guess("Untitled", genre="reggaeton")
    assert language == "spanish" and confidence < 0.5


def test_overrides_always_win(tmp_path):
    path = tmp_path / "overrides.csv"
    path.write_text("key,language,note\nABC,spanish,checked\nXYZ,not-a-language,ignored\n", encoding="utf-8")
    overrides = load_overrides(path)
    assert overrides == {"ABC": "spanish"}
    rows = [{"spotify_id": "ABC", "language": "english", "language_source": "gemini estimate"}, {"isrc": "Q", "language": "english"}]
    assert apply_overrides(rows, overrides) == 1
    assert rows[0]["language"] == "spanish" and rows[0]["language_source"] == SOURCE_MANUAL


def test_shipped_catalog_languages_are_valid_and_labeled():
    path = ROOT / "data" / "real_songs.csv"
    if not path.is_file():
        pytest.skip("real catalog not built")
    songs = load_songs(str(path))
    assert all(song["language"] in LANGUAGES for song in songs)
    assert all(song["language_source"] in {"gemini estimate", "rule estimate", "manual override"} for song in songs)
    assert "spanish" in catalog_languages(songs)
    assert catalog_languages(load_songs(str(ROOT / "data" / "songs.csv"))) == set()  # fictional: unspecified


class _Resp:
    def __init__(self, parsed):
        self.parsed, self.text = parsed, json.dumps(parsed)


class _Models:
    def __init__(self, parsed):
        self.parsed = parsed

    def generate_content(self, **kwargs):
        return _Resp(self.parsed)


class _Client:
    def __init__(self, parsed):
        self.models = _Models(parsed)


def test_gemini_language_labels_are_validated():
    tracks = [{"id": 1, "title": "A"}, {"id": 2, "title": "B"}, {"id": 3, "title": "C"}]
    parsed = {"labels": [
        {"id": 1, "language": "spanish", "confidence": 0.9},
        {"id": 2, "language": "klingon", "confidence": 0.9},   # not in vocabulary: dropped
        {"id": 99, "language": "english", "confidence": 0.9},  # not a requested id: dropped
        {"id": 3, "language": "english", "confidence": 7},     # clamped to 1.0
    ]}
    labels = GeminiService(client=_Client(parsed)).classify_languages(tracks)
    assert labels == {1: ("spanish", 0.9), 3: ("english", 1.0)}


def test_builder_falls_back_to_rules_for_ids_gemini_skipped():
    import importlib.util

    spec = importlib.util.spec_from_file_location("builder", ROOT / "scripts" / "build_real_catalog.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    rows = [{"id": 1, "title": "Corazón", "artist": "X", "genre": "latin", "energy": 0.5},
            {"id": 2, "title": "Hello", "artist": "Y", "genre": "pop", "energy": 0.5}]
    service = GeminiService(client=_Client({"labels": [{"id": 2, "language": "english", "confidence": 0.8}]}))
    sources = builder.label_languages(rows, service=service)
    assert rows[0]["language_source"] == "rule estimate" and rows[0]["language"] == "spanish"
    assert rows[1]["language_source"] == "gemini estimate"
    assert sum(sources.values()) == 2
