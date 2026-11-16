"""Pure tests for link, cover, note, and preview helpers. No real network calls."""

import io
import json

from src import media
from src.dashboard_data import placeholder_tile_html


REAL_SONG = {
    "id": 7,
    "title": "Blinding Lights",
    "artist": "The Weeknd",
    "genre": "r&b",
    "spotify_id": "0VjIjW4GlUZAMYd2vXMi3b",
    "isrc": "USUG11904206",
    "album": "After Hours",
    "release_date": "2020-03-20",
    "cover_url": "https://cdn-images.dzcdn.net/images/cover/x/500x500.jpg",
    "deezer_url": "https://www.deezer.com/track/908604612",
}


def test_links_and_cover_come_only_from_valid_fields():
    assert media.spotify_url(REAL_SONG) == "https://open.spotify.com/track/0VjIjW4GlUZAMYd2vXMi3b"
    assert media.spotify_embed_url(REAL_SONG).endswith("/embed/track/0VjIjW4GlUZAMYd2vXMi3b")
    assert media.cover_url(REAL_SONG).startswith("https://")
    assert media.spotify_url({"spotify_id": "not-a-valid-id"}) is None
    assert media.spotify_url({"title": "Fictional"}) is None
    assert media.cover_url({"cover_url": "http://insecure.example/x.jpg"}) is None
    assert media.cover_url({"title": "Fictional"}) is None


def test_origin_note_is_built_only_from_catalog_fields_and_carries_a_source():
    text, source = media.origin_note(REAL_SONG)
    assert "After Hours" in text and "2020" in text and "r&b" in text
    assert source == "Deezer catalog data"


def test_origin_note_is_absent_without_data():
    assert media.origin_note({"title": "Fictional Song", "genre": "pop"}) is None
    year_only = media.origin_note({"release_date": "1999-01-01", "genre": "other"})
    assert year_only == ("Released in 1999.", "Catalog data")


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_urlopen(routes):
    def opener(request, timeout=0):
        url = request.full_url
        for fragment, payload in routes.items():
            if fragment in url:
                if isinstance(payload, Exception):
                    raise payload
                return _FakeResponse(json.dumps(payload).encode())
        raise AssertionError(f"unexpected URL {url}")

    return opener


def test_preview_prefers_deezer_then_falls_back_to_apple(monkeypatch):
    monkeypatch.setattr(
        media.urllib.request,
        "urlopen",
        _fake_urlopen({"api.deezer.com": {"preview": "https://cdn.test/deezer.mp3"}}),
    )
    assert media.fetch_preview_url(REAL_SONG) == ("https://cdn.test/deezer.mp3", "Deezer")

    monkeypatch.setattr(
        media.urllib.request,
        "urlopen",
        _fake_urlopen(
            {
                "api.deezer.com": {"error": {"type": "DataException"}},
                "itunes.apple.com": {"results": [{"previewUrl": "https://cdn.test/apple.m4a"}]},
            }
        ),
    )
    assert media.fetch_preview_url(REAL_SONG) == ("https://cdn.test/apple.m4a", "Apple")


def test_preview_returns_none_when_every_service_fails(monkeypatch):
    error = media.urllib.error.URLError("offline")
    monkeypatch.setattr(media.urllib.request, "urlopen", _fake_urlopen({"api.deezer.com": error, "itunes.apple.com": error}))
    assert media.fetch_preview_url(REAL_SONG) is None


def test_placeholder_tile_is_stable_and_escapes_text():
    song = {"id": 3, "title": '<script>alert("x")</script>'}
    html = placeholder_tile_html(song)
    assert "<script" not in html
    assert html == placeholder_tile_html(song)
    assert placeholder_tile_html({"id": 4, "title": "A"}) != placeholder_tile_html({"id": 5, "title": "A"})
