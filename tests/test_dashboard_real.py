"""Dashboard tests for the real-music catalog, cards, now-playing panel, and styling.

All network access is stubbed: no test reaches Deezer, Apple, ReccoBeats, Spotify, or Gemini.
"""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import media
from src.app import create_playlist
from src.dashboard_data import playlist_csv
from src.recommender import load_songs


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = str(ROOT / "streamlit_app.py")
CATALOG = "data/songs.csv"
REAL_CATALOG = ROOT / "data" / "real_songs.csv"
FAKE_KEY = "test-key-should-never-render"
REAL_REQUEST = "Make a 12-song high-energy gym workout with a warm-up and cooldown"
PREVIEW_CALLS: list = []

needs_real_catalog = pytest.mark.skipif(not REAL_CATALOG.is_file(), reason="data/real_songs.csv has not been built")


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    """No real .env key, and previews come from a recording stub instead of the network."""
    monkeypatch.setattr("src.env.load_local_env", lambda *args, **kwargs: False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    PREVIEW_CALLS.clear()
    st.cache_data.clear()

    def stub(song):
        PREVIEW_CALLS.append(song.get("isrc"))
        return "https://example.test/preview.mp3", "Deezer"

    monkeypatch.setattr(media, "fetch_preview_url", stub)
    yield
    st.cache_data.clear()


def new_app() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=60)


def generate(at: AppTest, request: str, *, real: bool = False) -> AppTest:
    if real:
        at.segmented_control(key="catalog_choice").set_value("real")
    at.text_area(key="request_text").set_value(request)
    at.form_submit_button[0].click().run()
    return at


def all_text(at: AppTest) -> str:
    parts = [str(el.value) for el in [*at.markdown, *at.caption, *at.info, *at.warning, *at.success, *at.error]]
    parts += [f"{m.label} {m.value}" for m in at.metric]
    return "\n".join(parts)


def sidebar_text(at: AppTest) -> str:
    return "\n".join(str(el.value) for el in [*at.sidebar.markdown, *at.sidebar.caption])


def metric_value(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def play_buttons(at: AppTest) -> list:
    return [b for b in at.button if str(b.key).startswith("play-")]


@needs_real_catalog
def test_real_catalog_switch_generates_only_real_catalog_songs():
    real = {song["id"]: song for song in load_songs(str(REAL_CATALOG))}
    at = new_app().run()
    generate(at, REAL_REQUEST, real=True)
    assert not at.exception

    run = at.session_state["last_run"]
    assert run["catalog_key"] == "real"
    songs = run["result"].songs
    assert len(songs) == 12
    ids = [item.song["id"] for item in songs]
    assert len(set(ids)) == 12 and set(ids) <= set(real)
    for item in songs:
        assert item.song["title"] == real[item.song["id"]]["title"]
    assert "Real music catalog" in all_text(at)
    assert metric_value(at, "Request interpretation") == "Local fallback"
    assert "Selected IDs exist in the local catalog" in all_text(at)


@needs_real_catalog
def test_real_results_render_cards_covers_links_and_player():
    at = new_app().run()
    generate(at, REAL_REQUEST, real=True)
    assert not at.exception
    songs = at.session_state["last_run"]["result"].songs

    assert len(play_buttons(at)) == len(songs)
    assert len(at.link_button) >= len(songs)  # one Spotify link per card, plus the sidebar link
    assert all(str(link.url).startswith("https://open.spotify.com/track/") for link in at.link_button)
    assert len(at.get("image")) >= len(songs)  # cover art
    assert any(e.type == "iframe" for e in at.sidebar)  # Spotify embed
    assert any(e.type == "audio" for e in at.sidebar)  # fresh 30-second preview
    assert PREVIEW_CALLS == [songs[0].song["isrc"]]  # only the playing track is looked up


@needs_real_catalog
def test_play_button_changes_now_playing_track():
    at = new_app().run()
    generate(at, REAL_REQUEST, real=True)
    songs = at.session_state["last_run"]["result"].songs
    at.button(key="play-3").click().run()
    assert not at.exception
    assert at.session_state["now_playing"] == 2
    assert f"Track 3 of {len(songs)}" in sidebar_text(at)
    assert songs[2].song["isrc"] in PREVIEW_CALLS


@needs_real_catalog
def test_origin_notes_only_appear_with_a_source():
    at = new_app().run()
    generate(at, REAL_REQUEST, real=True)
    notes = [str(c.value) for c in at.caption if "From the album" in str(c.value) or "Released in" in str(c.value)]
    assert notes, "real catalog rows have album data, so notes should render"
    assert all("(Source: " in note for note in notes)
    assert "Source: Deezer catalog data" in sidebar_text(at)
    assert "not AI-written" in sidebar_text(at)


@needs_real_catalog
def test_missing_preview_degrades_without_crashing(monkeypatch):
    monkeypatch.setattr(media, "fetch_preview_url", lambda song: None)
    at = new_app().run()
    generate(at, REAL_REQUEST, real=True)
    assert not at.exception
    assert "No separate preview was found" in sidebar_text(at)
    assert not any(e.type == "audio" for e in at.sidebar)


def test_fictional_playlist_has_tiles_but_no_links_covers_or_notes():
    at = new_app().run()
    generate(at, "Create 8 peaceful songs for relaxing after work")
    assert not at.exception
    assert not at.link_button
    assert any("vm-tile" in str(h.value) for h in at.html)  # generated cover tiles
    assert not any("From the album" in str(c.value) for c in at.caption)
    assert "fictional song" in sidebar_text(at)
    assert PREVIEW_CALLS == []


@needs_real_catalog
def test_thirty_song_request_renders_thirty_cards():
    at = new_app().run()
    generate(at, "Make 30 songs for a workout", real=True)
    assert not at.exception
    songs = at.session_state["last_run"]["result"].songs
    assert len(songs) == 30 and len({item.song["id"] for item in songs}) == 30
    assert len(play_buttons(at)) == 30
    assert metric_value(at, "Songs returned") == "30 of 30"
    assert not any("Playlist length adjusted" in str(w.proto) for w in at.warning)


def test_trimmed_request_shows_a_visible_length_warning():
    at = new_app().run()
    generate(at, "Make 60 songs for a party")
    assert not at.exception
    assert metric_value(at, "Songs returned") == "50 of 50"
    adjusted = [w for w in at.warning if "Playlist length adjusted" in str(w.proto)]
    assert adjusted and "Requested 60 songs" in adjusted[0].value
    assert "Playlist length limits" in all_text(at)  # also explained in the System checks tab


def test_catalog_switch_defaults_to_fictional():
    at = new_app().run()
    assert not at.exception
    assert at.segmented_control(key="catalog_choice").value == "fictional"


@needs_real_catalog
def test_twenty_song_request_renders_every_card():
    at = new_app().run()
    generate(at, "I want 20 songs for a long road trip", real=True)
    assert not at.exception
    assert len(at.session_state["last_run"]["result"].songs) == 20
    assert len(play_buttons(at)) == 20


@needs_real_catalog
def test_real_csv_export_has_media_fields_and_no_secrets(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    result = create_playlist(REAL_REQUEST, catalog_path=REAL_CATALOG, offline=True)
    csv_text = playlist_csv(result)
    header = csv_text.splitlines()[0].split(",")
    for column in ("cover_url", "spotify_url", "album", "release_date", "origin_note", "origin_note_source", "isrc"):
        assert column in header
    assert FAKE_KEY not in csv_text
    assert "preview" not in csv_text.splitlines()[0].lower()  # expiring preview links are never exported


def test_custom_css_is_one_small_removable_file():
    app_source = Path(APP_PATH).read_text(encoding="utf-8")
    css = (ROOT / "assets" / "vibematch.css").read_text(encoding="utf-8")
    assert "unsafe_allow_html" not in app_source
    assert "use_container_width" not in app_source
    assert "prefers-reduced-motion" in css and "hover: hover" in css
    assert "transition: all" not in css
    assert "<script" not in css.lower()
    assert css.count("{") == css.count("}")
