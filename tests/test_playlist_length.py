"""Playlist length: the cap is a single constant, and trimming is never silent."""

import csv
from pathlib import Path

import pytest

from src.app import create_playlist
from src.dashboard_data import system_checks
from src.gemini_service import PROFILE_SCHEMA
from src.profiles import MAX_PLAYLIST_LENGTH, validate_profile
from src.recommender import load_songs


ROOT = Path(__file__).resolve().parents[1]
FICTIONAL = ROOT / "data" / "songs.csv"
REAL = ROOT / "data" / "real_songs.csv"


def test_cap_is_one_constant_shared_with_gemini_schema():
    assert MAX_PLAYLIST_LENGTH == 50
    assert PROFILE_SCHEMA["properties"]["playlist_length"]["maximum"] == MAX_PLAYLIST_LENGTH


def test_values_within_the_cap_are_not_trimmed_or_warned():
    profile, warnings = validate_profile({"playlist_length": 30}, supported_genres={"pop"}, supported_moods={"happy"})
    assert profile.playlist_length == 30
    assert not any("maximum playlist length" in w for w in warnings)


@pytest.mark.parametrize("catalog", [FICTIONAL, REAL])
def test_thirty_song_request_returns_thirty_unique_songs(catalog):
    if not catalog.is_file():
        pytest.skip(f"{catalog.name} has not been built")
    result = create_playlist("Make 30 songs for a workout", catalog_path=catalog, offline=True)
    ids = [item.song["id"] for item in result.songs]
    assert result.profile.playlist_length == 30
    assert len(ids) == 30 and len(set(ids)) == 30


def test_sixty_songs_returns_fifty_with_a_visible_warning():
    result = create_playlist("Make 60 songs for a party", catalog_path=FICTIONAL, offline=True)
    assert len(result.songs) == MAX_PLAYLIST_LENGTH
    assert any("Requested 60 songs" in w and str(MAX_PLAYLIST_LENGTH) in w for w in result.warnings)
    checks = {c.name: c for c in system_checks(result, load_songs(str(FICTIONAL)), gemini_requested=False)}
    assert "Requested 60 songs" in checks["Playlist length limits"].detail


def test_three_digit_requests_are_read_and_capped_not_ignored():
    result = create_playlist("play 100 songs for studying", catalog_path=FICTIONAL, offline=True)
    assert len(result.songs) == MAX_PLAYLIST_LENGTH
    assert any("Requested 100 songs" in w for w in result.warnings)


def test_empty_catalog_raises_a_clear_error_instead_of_crashing(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text(FICTIONAL.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contains no songs"):
        create_playlist("Make 10 songs for a workout", catalog_path=path, offline=True)


def test_request_larger_than_catalog_warns_about_catalog_size(tmp_path):
    rows = load_songs(str(FICTIONAL))[:12]
    path = tmp_path / "small.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    result = create_playlist("Make 30 songs for a workout", catalog_path=path, offline=True)
    assert len(result.songs) == 12
    assert any("Requested 30 songs, but the catalog contains only 12" in w for w in result.warnings)
    checks = {c.name: c for c in system_checks(result, rows, gemini_requested=False)}
    assert checks["Returned length matches requested length"].passed is False
    assert "catalog contains only 12" in checks["Playlist length limits"].detail
