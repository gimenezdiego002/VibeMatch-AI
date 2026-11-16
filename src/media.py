"""Links, previews, and sourced notes for real-music catalog rows.

Everything here is presentation support. It never scores, selects, or reorders songs, and it
never produces a claim that is not present in the catalog row or in a live API response.
Network helpers use short timeouts and return None on any failure so the UI can degrade.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Mapping


USER_AGENT = "VibeMatchClassProject/0.1 (student project)"
SPOTIFY_ID = re.compile(r"^[A-Za-z0-9]{22}$")


def _text(song: Mapping[str, Any], key: str) -> str:
    value = song.get(key)
    return str(value).strip() if value not in (None, "") else ""


def spotify_id(song: Mapping[str, Any]) -> str | None:
    value = _text(song, "spotify_id")
    return value if SPOTIFY_ID.fullmatch(value) else None


def spotify_url(song: Mapping[str, Any]) -> str | None:
    track_id = spotify_id(song)
    return f"https://open.spotify.com/track/{track_id}" if track_id else None


def spotify_embed_url(song: Mapping[str, Any]) -> str | None:
    track_id = spotify_id(song)
    return f"https://open.spotify.com/embed/track/{track_id}" if track_id else None


def cover_url(song: Mapping[str, Any]) -> str | None:
    value = _text(song, "cover_url")
    return value if value.startswith("https://") else None


def release_year(song: Mapping[str, Any]) -> str | None:
    match = re.match(r"^(\d{4})", _text(song, "release_date"))
    return match.group(1) if match else None


def origin_note(song: Mapping[str, Any]) -> tuple[str, str] | None:
    """Build a short note ONLY from catalog fields. Returns (text, source) or None.

    Nothing is invented: if the row has no album or year, there is no note.
    """
    album, year, genre = _text(song, "album"), release_year(song), _text(song, "genre")
    if not album and not year:
        return None
    if album and year:
        text = f"From the album “{album}”, released in {year}."
    elif album:
        text = f"From the album “{album}”."
    else:
        text = f"Released in {year}."
    if genre and genre != "other":
        text += f" Filed under {genre}."
    return text, "Deezer catalog data" if _text(song, "deezer_url") else "Catalog data"


def _fetch_json(url: str) -> Any | None:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=6) as response:
            return json.load(response)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


def fetch_preview_url(song: Mapping[str, Any]) -> tuple[str, str] | None:
    """Look up a fresh 30-second preview. Preview URLs expire, so they are never stored.

    Returns (url, provider) or None when no preview is available.
    """
    isrc = _text(song, "isrc")
    if isrc:
        data = _fetch_json(f"https://api.deezer.com/track/isrc:{urllib.parse.quote(isrc)}")
        if isinstance(data, dict) and not data.get("error") and str(data.get("preview", "")).startswith("https://"):
            return data["preview"], "Deezer"
    title, artist = _text(song, "title"), _text(song, "artist").split(",")[0]
    if title and artist:
        query = urllib.parse.urlencode({"term": f"{title} {artist}", "entity": "song", "limit": 1})
        data = _fetch_json(f"https://itunes.apple.com/search?{query}")
        results = data.get("results", []) if isinstance(data, dict) else []
        if results and str(results[0].get("previewUrl", "")).startswith("https://"):
            return results[0]["previewUrl"], "Apple"
    return None
