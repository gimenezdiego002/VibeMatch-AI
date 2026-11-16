"""Tests for scripts/build_real_catalog.py. All HTTP is mocked; no live calls."""

import csv
import importlib.util
from pathlib import Path

import pytest

from src.recommender import load_songs


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_real_catalog.py"
spec = importlib.util.spec_from_file_location("build_real_catalog", SCRIPT)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

MOODS_IN_SCORER = {"aggressive", "chill", "confident", "energetic", "euphoric", "focused", "happy",
                   "intense", "moody", "peaceful", "reflective", "relaxed", "romantic"}


def sid(n: int) -> str:
    return f"{n:022d}"


@pytest.mark.parametrize(
    "energy,valence,acoustic,expected",
    [
        (0.95, 0.9, 0.1, "euphoric"),
        (0.9, 0.6, 0.1, "energetic"),
        (0.85, 0.2, 0.1, "intense"),
        (0.7, 0.8, 0.2, "happy"),
        (0.7, 0.5, 0.2, "confident"),
        (0.7, 0.2, 0.2, "moody"),
        (0.5, 0.7, 0.3, "happy"),
        (0.5, 0.4, 0.3, "focused"),
        (0.5, 0.1, 0.3, "reflective"),
        (0.2, 0.2, 0.9, "reflective"),
        (0.2, 0.6, 0.9, "peaceful"),
        (0.2, 0.4, 0.5, "focused"),
        (0.2, 0.6, 0.5, "chill"),
    ],
)
def test_derived_mood_rules(energy, valence, acoustic, expected):
    assert builder.derive_mood(energy, valence, acoustic) == expected
    assert builder.derive_mood(energy, valence, acoustic) in MOODS_IN_SCORER


def test_derived_mood_always_lands_in_scorer_vocabulary():
    for energy in (0.0, 0.2, 0.39, 0.4, 0.59, 0.6, 0.79, 0.8, 1.0):
        for valence in (0.0, 0.3, 0.5, 0.7, 1.0):
            for acoustic in (0.0, 0.7):
                assert builder.derive_mood(energy, valence, acoustic) in MOODS_IN_SCORER


def test_genre_mapping_and_spotify_id_parsing():
    assert builder.map_genre(["Rap/Hip Hop"]) == "hip-hop"
    assert builder.map_genre(["Unknown Thing", "Dance"]) == "dance"
    assert builder.map_genre(["Unknown Thing"]) is None
    assert builder.spotify_id("https://open.spotify.com/track/0VjIjW4GlUZAMYd2vXMi3b?si=abc") == "0VjIjW4GlUZAMYd2vXMi3b"
    assert builder.spotify_id("not a link") is None


@pytest.mark.parametrize(
    "title",
    ["Jingle Bells (Christmas)", "Feliz Navidad", "Song - Live at Wembley", "Track (Remix)", "X - Karaoke Version", "Abc Song"],
)
def test_title_filter_rejects_unwanted_tracks(title):
    assert builder.EXCLUDED_TITLE.search(title)


def test_title_filter_keeps_normal_titles():
    for title in ("Blinding Lights", "Sunrise City", "Olive Garden"):
        assert not builder.EXCLUDED_TITLE.search(title)


def test_pick_next_seed_steers_toward_the_emptiest_band():
    pool = {
        sid(1): {"features": {"energy": 0.9}},
        sid(2): {"features": {"energy": 0.85}},
        sid(3): {"features": {"energy": 0.15}},
    }
    # Bands 0-7 are empty or near-empty; the nearest unused track to the emptiest band wins.
    assert builder.pick_next_seed(pool, {sid(1)}, want=2) == sid(3)
    # Nothing left to try once every candidate has been used.
    assert builder.pick_next_seed(pool, set(pool), want=2) is None


def _rec(n: int, energy: float) -> dict:
    return {
        "id": f"uuid-{n}",
        "trackTitle": f"Song {n}",
        "artists": [{"name": f"Artist {n}"}],
        "isrc": f"ISRC{n:08d}",
        "href": f"https://open.spotify.com/track/{sid(n)}",
        "_energy": energy,
    }


def test_expand_grows_pool_and_attaches_features(monkeypatch):
    catalog = {n: _rec(n, round(n / 10, 2)) for n in range(1, 9)}

    def fake_get_json(url, retries=3):
        if "/track/recommendation" in url:
            return {"content": [catalog[n] for n in (1, 2, 3, 4)]}
        if "/audio-features" in url:
            ids = url.split("ids=")[1].split(",")
            rows = []
            for track_id in ids:
                n = int(track_id)
                rows.append(
                    {"href": f"https://open.spotify.com/track/{track_id}", "energy": catalog[n]["_energy"],
                     "valence": 0.5, "danceability": 0.5, "acousticness": 0.3, "tempo": 100.0}
                )
            return {"content": rows}
        if "/track?ids=" in url:
            return {"content": [catalog[1]]}
        raise AssertionError(url)

    monkeypatch.setattr(builder, "get_json", fake_get_json)
    monkeypatch.setattr(builder.time, "sleep", lambda *_: None)
    pool = builder.expand([sid(1)], per_call=10, want_per_band=2, max_rounds=3)
    assert set(pool) == {sid(n) for n in (1, 2, 3, 4)}
    assert all(item["features"]["energy"] is not None for item in pool.values())


def test_deezer_enrichment_rejects_explicit_and_excluded_genres(monkeypatch):
    def make(explicit, genre):
        def fake_get_json(url, retries=3):
            if "/track/isrc:" in url:
                return {"explicit_lyrics": explicit, "link": "https://deezer.test/t", "album": {"id": 1, "title": "Album", "cover_big": "https://img.test/c.jpg"}}
            return {"genres": {"data": [{"name": genre}]}, "release_date": "2020-01-01"}
        return fake_get_json

    monkeypatch.setattr(builder, "get_json", make(False, "Pop"))
    ok = builder.enrich_deezer("ISRC1")
    assert ok["genre"] == "pop" and ok["cover_url"].startswith("https://") and ok["release_date"] == "2020-01-01"

    monkeypatch.setattr(builder, "get_json", make(True, "Pop"))
    assert builder.enrich_deezer("ISRC1") is None
    assert builder.enrich_deezer("ISRC1", allow_explicit=True) is not None

    monkeypatch.setattr(builder, "get_json", make(False, "Kids"))
    assert builder.enrich_deezer("ISRC1") is None


def test_enrichment_failure_returns_none_instead_of_raising(monkeypatch):
    def boom(url, retries=3):
        raise RuntimeError("offline")

    monkeypatch.setattr(builder, "get_json", boom)
    assert builder.enrich_deezer("ISRC1") is None


def test_shipped_real_catalog_passes_the_same_validation_as_the_loader():
    path = SCRIPT.parents[1] / "data" / "real_songs.csv"
    if not path.is_file():
        pytest.skip("data/real_songs.csv has not been built")
    songs = load_songs(str(path))  # type, range, and duplicate-ID validation
    assert len(songs) >= 12
    assert len({s["isrc"] for s in songs}) == len(songs)
    assert {s["mood"] for s in songs} <= MOODS_IN_SCORER
    for song in songs:
        assert len(song["spotify_id"]) == 22
        assert song["cover_url"].startswith("https://")
    with open(path, newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert "preview" not in " ".join(header).lower()  # preview URLs expire and must not be stored
