"""Application orchestration for VibeMatch AI."""

from __future__ import annotations

from dataclasses import replace
import logging
import time
from pathlib import Path

from src.gemini_service import GeminiService
from src.languages import catalog_languages
from src.logging_setup import short
from src.playlist import PlaylistResult, generate_playlist
from src.profiles import infer_profile_offline
from src.recommender import load_songs


LOGGER = logging.getLogger(__name__)
DEFAULT_CATALOG = Path(__file__).resolve().parents[1] / "data" / "songs.csv"


def create_playlist(
    request: str,
    *,
    catalog_path: str | Path = DEFAULT_CATALOG,
    offline: bool = False,
    gemini: GeminiService | None = None,
) -> PlaylistResult:
    """Run interpretation, deterministic generation, and optional explanation."""
    if not request.strip():
        LOGGER.info("Rejected an empty playlist request.")
        raise ValueError("Playlist request cannot be empty.")
    started = time.perf_counter()
    LOGGER.info("Playlist request on %s (offline=%s): %s", Path(catalog_path).name, offline, short(request))
    songs = load_songs(str(catalog_path))
    if not songs:
        raise ValueError(f"The catalog at {Path(catalog_path).name} contains no songs.")
    genres = {str(song["genre"]).casefold() for song in songs}
    moods = {str(song["mood"]).casefold() for song in songs}
    catalog_info = {
        "supported_languages": catalog_languages(songs),
        "catalog_has_years": any(str(song.get("release_date", "")).strip()[:4].isdigit() for song in songs),
    }
    service = gemini or GeminiService()
    warnings: list[str] = []
    used_interpretation = False

    if not offline and service.available:
        try:
            profile, validation_warnings = service.interpret(
                request, supported_genres=genres, supported_moods=moods, **catalog_info
            )
            warnings.extend(validation_warnings)
            used_interpretation = True
        except Exception as exc:  # API/network/model responses are an external boundary.
            LOGGER.warning("Gemini interpretation failed (%s: %s); using local fallback.", type(exc).__name__, short(exc, 90))
            profile, fallback_warnings = infer_profile_offline(
                request, supported_genres=genres, supported_moods=moods, **catalog_info
            )
            warnings.append(f"Gemini interpretation failed ({type(exc).__name__}); used the local fallback.")
            warnings.extend(fallback_warnings)
    else:
        profile, fallback_warnings = infer_profile_offline(
            request, supported_genres=genres, supported_moods=moods, **catalog_info
        )
        warnings.extend(fallback_warnings)

    LOGGER.info(
        "Profile (%s): activity=%s length=%s language=%s%s years=%s-%s%s structure=%s",
        "Gemini" if used_interpretation else "local", profile.activity, profile.playlist_length,
        profile.language, " hard" if profile.language_strict else "", profile.min_release_year,
        profile.max_release_year, " hard" if profile.recency_strict else "", profile.playlist_structure,
    )
    result = generate_playlist(songs, profile)
    result = replace(
        result,
        warnings=tuple([*warnings, *result.warnings]),
        used_gemini_interpretation=used_interpretation,
    )
    # Skip explanations when interpretation already failed: the same outage or quota limit
    # would fail again, doubling latency for nothing. Deterministic reasons are shown instead.
    if not offline and service.available and used_interpretation:
        try:
            result = service.explain(result)
        except Exception as exc:
            LOGGER.warning("Gemini explanation failed (%s: %s); keeping deterministic reasons.", type(exc).__name__, short(exc, 90))
            result = replace(
                result,
                warnings=tuple(
                    [*result.warnings, f"Gemini explanations failed ({type(exc).__name__}); showing deterministic reasons."]
                ),
            )
    LOGGER.info(
        "Playlist ready: %d of %d songs, explanations=%s, %d warning(s), %.0f ms",
        len(result.songs), profile.playlist_length, "Gemini" if result.used_gemini_explanations else "deterministic",
        len(result.warnings), (time.perf_counter() - started) * 1000,
    )
    for warning in result.warnings:
        LOGGER.info("Warning shown to user: %s", short(warning, 160))
    return result
