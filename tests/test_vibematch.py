import json

import pytest

from src.app import create_playlist
from src.gemini_service import GeminiService
from src.playlist import energy_curve, generate_playlist, score_song_for_profile
from src.profiles import MAX_PLAYLIST_LENGTH, PreferenceProfile, validate_profile
from src.recommender import load_songs


CATALOG = "data/songs.csv"


def study_profile(length=10):
    return PreferenceProfile("studying", length, ("lofi", "ambient"), "focused", 0.30, 78.0, "consistent")


def test_improved_score_is_normalized_and_rewards_closeness():
    profile = study_profile()
    close = {"genre": "lofi", "mood": "focused", "energy": 0.31, "tempo_bpm": 79}
    far = {"genre": "metal", "mood": "aggressive", "energy": 0.98, "tempo_bpm": 180}
    close_score, contributions, _ = score_song_for_profile(close, profile)
    far_score, _, _ = score_song_for_profile(far, profile)
    assert 0 <= far_score < close_score <= 100
    assert set(contributions) == {"energy", "mood", "genre", "tempo"}


def test_profile_validation_clamps_numbers_and_replaces_categories():
    songs = load_songs(CATALOG)
    profile, warnings = validate_profile(
        {
            "activity": "teleporting",
            "playlist_length": 999,
            "preferred_genres": ["not-a-genre"],
            "target_mood": "unknown",
            "target_energy": 4,
            "target_tempo_bpm": -8,
            "playlist_structure": "random",
        },
        supported_genres={song["genre"] for song in songs},
        supported_moods={song["mood"] for song in songs},
    )
    assert profile.activity == "general"
    assert profile.playlist_length == MAX_PLAYLIST_LENGTH  # was 20 before the cap was raised to 50
    assert profile.target_energy == 1.0
    assert profile.target_tempo_bpm == 40.0
    assert warnings


def test_workout_curve_has_warmup_peak_and_cooldown():
    profile = PreferenceProfile("workout", 10, ("rock",), "intense", 0.88, 140, "warmup_peak_cooldown")
    curve = energy_curve(profile)
    assert len(curve) == 10
    assert curve[0] < max(curve)
    assert curve[-1] < max(curve)


def test_workout_playlist_follows_energy_arc():
    songs = load_songs(CATALOG)
    profile = PreferenceProfile("workout", 10, ("rock", "pop", "electronic"), "intense", 0.88, 138, "warmup_peak_cooldown")
    result = generate_playlist(songs, profile)
    energies = [float(item.song["energy"]) for item in result.songs]
    assert energies[0] < max(energies)
    assert energies[-1] < max(energies)


def test_playlist_length_ids_and_determinism():
    songs = load_songs(CATALOG)
    first = generate_playlist(songs, study_profile())
    second = generate_playlist(songs, study_profile())
    catalog_ids = {song["id"] for song in songs}
    first_ids = [item.song["id"] for item in first.songs]
    assert len(first_ids) == 10
    assert len(set(first_ids)) == 10
    assert set(first_ids) <= catalog_ids
    assert first_ids == [item.song["id"] for item in second.songs]


def test_consistent_playlist_is_descending_by_score():
    result = generate_playlist(load_songs(CATALOG), study_profile())
    scores = [item.score for item in result.songs]
    assert scores == sorted(scores, reverse=True)


def test_requested_length_larger_than_catalog_returns_honest_warning():
    songs = load_songs(CATALOG)[:2]
    result = generate_playlist(songs, study_profile(length=10))
    assert len(result.songs) == 2
    assert "catalog contains only 2" in result.warnings[0]


@pytest.mark.parametrize(
    "bad_value,column",
    [("1.5", "energy"), ("999", "tempo_bpm"), ("not-a-number", "energy")],
)
def test_loader_rejects_invalid_energy_and_tempo(tmp_path, bad_value, column):
    values = {
        "id": "1", "title": "Test", "artist": "Test", "genre": "pop", "mood": "happy",
        "energy": "0.5", "tempo_bpm": "100", "valence": "0.5", "danceability": "0.5", "acousticness": "0.5",
    }
    values[column] = bad_value
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text(
        ",".join(values) + "\n" + ",".join(values.values()) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_songs(str(csv_path))


class FakeResponse:
    def __init__(self, parsed=None, text=""):
        self.parsed = parsed
        self.text = text


class FakeModels:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    def generate_content(self, **kwargs):
        if self.error:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response=None, error=None):
        self.models = FakeModels(response=response, error=error)


def test_malformed_gemini_profile_raises_cleanly():
    service = GeminiService(client=FakeClient(FakeResponse(parsed="bad", text="not JSON")))
    with pytest.raises((ValueError, json.JSONDecodeError)):
        service.interpret("study", supported_genres={"lofi"}, supported_moods={"focused"})


def test_api_failure_uses_deterministic_fallback():
    service = GeminiService(client=FakeClient(error=RuntimeError("offline")))
    result = create_playlist("Make 6 calm songs for studying", offline=False, gemini=service)
    assert result.profile.activity == "studying"
    assert len(result.songs) == 6
    assert not result.used_gemini_interpretation
    assert any("failed" in warning for warning in result.warnings)


def test_conflicting_preferences_remain_valid():
    songs = load_songs(CATALOG)
    profile = PreferenceProfile("workout", 5, ("classical",), "aggressive", 0.98, 170, "balanced")
    result = generate_playlist(songs, profile)
    assert len(result.songs) == 5
    assert all(0 <= item.score <= 100 for item in result.songs)
