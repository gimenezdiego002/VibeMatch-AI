"""Graded similarity, bounded priorities, soft/hard constraints, and their dashboard display."""

from dataclasses import replace
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import media
from src.app import create_playlist
from src.playlist import (
    GENRE_SIMILARITY, MOOD_SIMILARITY, PRIORITY_BOUNDS, WEIGHTS, effective_weights, generate_playlist,
    label_similarity, score_song_for_profile,
)
from src.profiles import PreferenceProfile
from src.recommender import load_songs


ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data" / "real_songs.csv"
needs_real = pytest.mark.skipif(not REAL.is_file(), reason="real catalog not built")
BASE = PreferenceProfile("workout", 10, ("latin",), "intense", 0.85, 130.0, "warmup_peak_cooldown")


def song(**overrides):
    data = {"id": 1, "title": "T", "artist": "A", "genre": "latin", "mood": "intense", "energy": 0.85,
            "tempo_bpm": 130, "language": "spanish", "release_date": "2024-01-01"}
    data.update(overrides)
    return data


# ----- similarity tables -----

def test_similarity_exact_partial_unrelated_and_symmetric():
    assert label_similarity("pop", "pop", GENRE_SIMILARITY) == 1.0
    assert label_similarity("latin", "reggaeton", GENRE_SIMILARITY) == 0.8
    assert label_similarity("reggaeton", "latin", GENRE_SIMILARITY) == 0.8
    assert label_similarity("metal", "lofi", GENRE_SIMILARITY) == 0.0
    assert label_similarity("chill", "peaceful", MOOD_SIMILARITY) == 0.8


def test_similarity_tables_stay_strictly_between_zero_and_one():
    for table in (GENRE_SIMILARITY, MOOD_SIMILARITY):
        assert all(0.0 < value < 1.0 for value in table.values())
        assert all(len(pair) == 2 for pair in table)


def test_close_genre_now_scores_between_exact_and_unrelated():
    exact, _, _ = score_song_for_profile(song(genre="latin"), BASE)
    close, _, _ = score_song_for_profile(song(genre="reggaeton"), BASE)
    unrelated, _, _ = score_song_for_profile(song(genre="classical"), BASE)
    assert exact > close > unrelated


# ----- weights -----

def test_default_weights_are_unchanged_without_constraints():
    assert effective_weights(BASE) == pytest.approx(WEIGHTS)


def test_soft_constraints_add_features_and_hard_ones_do_not():
    soft = replace(BASE, language="spanish", language_strict=False, min_release_year=2021, recency_strict=False)
    hard = replace(BASE, language="spanish", language_strict=True, min_release_year=2021, recency_strict=True)
    assert set(effective_weights(soft)) == {"energy", "mood", "genre", "tempo", "language", "recency"}
    assert set(effective_weights(hard)) == set(WEIGHTS)


@pytest.mark.parametrize("priorities", [(), (("energy", 1.0),), (("energy", 0.0), ("mood", 1.0)), (("tempo", 0.6), ("genre", 0.6))])
def test_priorities_are_bounded_and_weights_sum_to_one(priorities):
    weights = effective_weights(replace(BASE, priorities=priorities))
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)
    low, high = PRIORITY_BOUNDS
    assert min(weights.values()) >= low / (sum(WEIGHTS.values()) + 2 * high) - 1e-9  # never zeroed out


def test_priority_cannot_zero_out_a_feature():
    weights = effective_weights(replace(BASE, priorities=(("energy", 0.0),)))
    assert weights["energy"] > 0


def test_contributions_always_sum_to_the_score():
    profile = replace(BASE, language="spanish", min_release_year=2021, priorities=(("language", 0.9),))
    for candidate in (song(), song(language="english", release_date="1999-01-01", genre="rock", energy=0.2)):
        score, contributions, _ = score_song_for_profile(candidate, profile)
        assert 0 <= score <= 100
        assert score == pytest.approx(sum(contributions.values()) * 100, abs=0.01)


def test_soft_recency_rewards_newer_songs_and_unknown_dates_are_neutral():
    profile = replace(BASE, min_release_year=2021)
    new, _, _ = score_song_for_profile(song(release_date="2023-05-01"), profile)
    old, _, _ = score_song_for_profile(song(release_date="2005-05-01"), profile)
    unknown, _, _ = score_song_for_profile(song(release_date=""), profile)
    assert new > unknown > old


# ----- hard constraints and playlist generation -----

def test_hard_language_filter_and_honest_shortage_warning():
    catalog = [song(id=i, artist=f"A{i}", language="spanish" if i < 4 else "english") for i in range(10)]
    result = generate_playlist(catalog, replace(BASE, language="spanish", language_strict=True))
    assert len(result.songs) == 4
    assert all(item.song["language"] == "spanish" for item in result.songs)
    assert "Only 4 Spanish songs fit this catalog; returning 4 instead of 10." in result.warnings


def test_hard_year_filter_with_no_matches_returns_nothing_and_says_so():
    catalog = [song(id=i, artist=f"A{i}", release_date="2010-01-01") for i in range(5)]
    result = generate_playlist(catalog, replace(BASE, min_release_year=2021, recency_strict=True))
    assert result.songs == ()
    assert result.warnings == ("No songs released 2021 or later exist in this catalog, so no songs were returned.",)


def test_shortage_still_gets_a_full_energy_arc():
    catalog = [song(id=i, artist=f"A{i}", energy=e) for i, e in enumerate([0.4, 0.6, 0.9, 0.95, 0.5, 0.3])]
    result = generate_playlist(catalog, replace(BASE, playlist_length=20))
    targets = [item.position_target_energy for item in result.songs]
    assert len(targets) == 6
    assert targets[0] < max(targets) and targets[-1] < max(targets)  # warm-up and cooldown both present


def test_constrained_generation_is_deterministic():
    songs = load_songs(str(ROOT / "data" / "songs.csv"))
    profile = replace(BASE, preferred_genres=("pop",), target_mood="happy", priorities=(("mood", 0.6),))
    first = [item.song["id"] for item in generate_playlist(songs, profile).songs]
    assert first == [item.song["id"] for item in generate_playlist(songs, profile).songs]


@needs_real
def test_original_failure_case_now_honors_spanish():
    result = create_playlist("I am working out I want 15 songs all in spanish kinda recent", catalog_path=REAL, offline=True)
    assert result.songs and all(item.song["language"] == "spanish" for item in result.songs)
    assert result.profile.playlist_structure == "warmup_peak_cooldown"
    assert any("Spanish songs fit this catalog" in w for w in result.warnings) or len(result.songs) == 15


# ----- dashboard -----

@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr("src.env.load_local_env", lambda *a, **k: False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(media, "fetch_preview_url", lambda song: None)
    st.cache_data.clear()
    return AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60).run()


def _generate(at, text, real=True):
    if real:
        at.segmented_control(key="catalog_choice").set_value("real")
    at.text_area(key="request_text").set_value(text)
    at.form_submit_button[0].click().run()
    return at


@needs_real
def test_dashboard_shows_language_constraint_and_shortage(app):
    _generate(app, "I am working out I want 15 songs all in spanish kinda recent")
    assert not app.exception
    labels = {m.label: m.value for m in app.metric}
    assert labels["Language"] == "Spanish (only)"
    assert labels["Release years"].endswith("(preferred)")
    assert any("Spanish songs fit this catalog" in str(w.value) for w in app.warning)


@needs_real
def test_dashboard_handles_a_request_with_no_matching_songs(app):
    _generate(app, "only spanish songs from the 1950s only, 10 songs")
    assert not app.exception
    assert any("No songs matched this request" in str(i.value) for i in app.info)


def test_fictional_catalog_explains_missing_language_data(app):
    _generate(app, "15 spanish songs for a workout", real=False)
    assert not app.exception
    assert any("no language information" in str(w.value) for w in app.warning) or any(
        "no language information" in str(m.value) for m in app.markdown
    )
