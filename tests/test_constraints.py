"""Language / release-year constraints: validation, offline parsing, and the Gemini schema."""

import json

import pytest

from src.gemini_service import GeminiService, profile_schema
from src.profiles import infer_profile_offline, validate_profile


GENRES = {"latin", "reggaeton", "pop", "rock", "electronic", "dance"}
MOODS = {"energetic", "intense", "happy", "chill", "euphoric", "focused"}
LANGS = {"spanish", "english", "portuguese"}
CATALOG = dict(supported_genres=GENRES, supported_moods=MOODS, supported_languages=LANGS,
               catalog_has_years=True, reference_year=2026)


def validate(raw, **overrides):
    return validate_profile(raw, **{**CATALOG, **overrides})


# ----- validation -----

def test_valid_language_and_years_are_kept():
    profile, warnings = validate({"language": "Spanish", "language_strict": True, "min_release_year": 2021, "recency_strict": "false"})
    assert (profile.language, profile.language_strict, profile.min_release_year, profile.recency_strict) == ("spanish", True, 2021, False)
    assert not warnings


def test_unknown_or_missing_language_is_dropped_with_a_warning():
    profile, warnings = validate({"language": "klingon"})
    assert profile.language is None and any("not recognized" in w for w in warnings)
    profile, warnings = validate({"language": "korean"})
    assert profile.language is None and any("no korean songs" in w for w in warnings)


def test_catalog_without_language_data_explains_why():
    profile, warnings = validate({"language": "spanish", "language_strict": True}, supported_languages=set())
    assert profile.language is None
    assert any("no language information" in w and "real-music catalog" in w for w in warnings)


def test_years_are_clamped_swapped_and_need_a_catalog_with_dates():
    profile, _ = validate({"min_release_year": 2035, "max_release_year": 1850})
    assert (profile.min_release_year, profile.max_release_year) == (1900, 2026)
    profile, warnings = validate({"min_release_year": 2021}, catalog_has_years=False)
    assert profile.min_release_year is None and any("no release dates" in w for w in warnings)
    profile, _ = validate({"min_release_year": 0})
    assert profile.min_release_year is None  # 0 is the schema's "no constraint" sentinel


def test_priorities_are_clamped_and_unknown_features_ignored():
    profile, _ = validate({"priorities": {"energy": 3, "language": -2, "genre": 0.4, "danceability": 1}})
    assert dict(profile.priorities) == {"energy": 1.0, "language": 0.0, "genre": 0.4}  # clamped into 0-1


def test_workout_cannot_silently_become_steady():
    profile, warnings = validate({"activity": "workout", "playlist_structure": "consistent"})
    assert profile.playlist_structure == "warmup_peak_cooldown"
    assert profile.notes and not warnings  # a note in Details, not a warning
    profile, _ = validate({"activity": "workout", "playlist_structure": "consistent", "steady_requested": True})
    assert profile.playlist_structure == "consistent"
    profile, _ = validate({"activity": "studying", "playlist_structure": "consistent"})
    assert profile.playlist_structure == "consistent" and not profile.notes


# ----- offline parsing -----

@pytest.mark.parametrize(
    "request_text,length,language,strict",
    [
        ("I am working out I want 15 songs all in spanish kinda recent", 15, "spanish", True),
        ("I need workout music in spanish 15", 15, "spanish", True),
        ("15 spanish songs for running", 15, "spanish", True),
        ("preferably spanish party mix 20", 20, "spanish", False),
        ("spanish if possible, 12 songs for driving", 12, "spanish", False),
        ("Make 30 songs for a workout at 140 bpm", 30, None, False),
    ],
)
def test_offline_length_and_language(request_text, length, language, strict):
    profile, _ = infer_profile_offline(request_text, **CATALOG)
    assert profile.playlist_length == length
    assert profile.language == language
    assert profile.language_strict is strict


@pytest.mark.parametrize(
    "request_text,min_year,max_year,strict",
    [
        ("I want 15 songs all in spanish kinda recent", 2021, None, False),
        ("only new songs for a party", 2021, None, True),
        ("workout from the 90s only", 1990, 1999, True),
        ("some 2010s pop", 2010, 2019, False),
        ("happy songs for a road trip", None, None, False),
    ],
)
def test_offline_release_years(request_text, min_year, max_year, strict):
    profile, _ = infer_profile_offline(request_text, **CATALOG)
    assert (profile.min_release_year, profile.max_release_year, profile.recency_strict) == (min_year, max_year, strict)


def test_spanish_without_explicit_genre_prefers_latin_genres():
    profile, _ = infer_profile_offline("15 spanish songs for a workout", **CATALOG)
    assert set(profile.preferred_genres) <= {"latin", "reggaeton"} and profile.preferred_genres


# ----- Gemini schema and wiring (mocked) -----

def test_schema_language_enum_comes_from_the_catalog():
    schema = profile_schema(["portuguese", "spanish"], 2026)
    assert schema["properties"]["language"]["enum"] == ["none", "portuguese", "spanish"]
    assert schema["properties"]["min_release_year"]["maximum"] == 2026
    assert "language_strict" in schema["required"] and "priorities" not in schema["required"]


class _Resp:
    def __init__(self, parsed):
        self.parsed, self.text = parsed, json.dumps(parsed)


class _RecordingModels:
    def __init__(self, parsed):
        self.parsed, self.calls = parsed, []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return _Resp(self.parsed)


def test_gemini_interpretation_is_validated_against_the_catalog():
    models = _RecordingModels({
        "activity": "workout", "playlist_length": 15, "preferred_genres": ["latin"], "target_mood": "energetic",
        "target_energy": 0.85, "target_tempo_bpm": 130, "playlist_structure": "consistent",
        "steady_requested": False, "language": "spanish", "language_strict": True,
        "min_release_year": 2021, "max_release_year": 0, "recency_strict": False,
        "priorities": {"language": 0.9, "energy": 0.7},
    })
    client = type("C", (), {"models": models})()
    profile, warnings = GeminiService(client=client).interpret(
        "workout 15 all in spanish kinda recent", supported_genres=GENRES, supported_moods=MOODS,
        supported_languages=LANGS, catalog_has_years=True,
    )
    sent_schema = models.calls[0]["config"]["response_json_schema"]
    assert sent_schema["properties"]["language"]["enum"] == ["none", "english", "portuguese", "spanish"]
    assert profile.language == "spanish" and profile.language_strict
    assert profile.min_release_year == 2021 and profile.max_release_year is None
    assert profile.playlist_structure == "warmup_peak_cooldown"  # the structure guard applied to Gemini too
    assert dict(profile.priorities) == {"energy": 0.7, "language": 0.9}
