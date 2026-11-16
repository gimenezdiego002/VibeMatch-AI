"""Validated preference profiles and deterministic natural-language fallback."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Mapping


SUPPORTED_ACTIVITIES = {
    "workout",
    "studying",
    "driving",
    "relaxing",
    "running",
    "party",
    "general",
}
SUPPORTED_STRUCTURES = {"warmup_peak_cooldown", "build", "consistent", "balanced"}
MAX_PLAYLIST_LENGTH = 50  # Single source of truth; also used by the Gemini response schema.
DEFAULT_PLAYLIST_LENGTH = 10

ACTIVITY_DEFAULTS = {
    "workout": (0.86, 138.0, "intense", "warmup_peak_cooldown", ["pop", "electronic", "rock"]),
    "running": (0.88, 150.0, "energetic", "warmup_peak_cooldown", ["electronic", "pop", "rock"]),
    "studying": (0.30, 78.0, "focused", "consistent", ["lofi", "ambient", "classical"]),
    "driving": (0.68, 112.0, "confident", "balanced", ["synthwave", "pop", "rock"]),
    "relaxing": (0.24, 72.0, "chill", "consistent", ["ambient", "lofi", "folk"]),
    "party": (0.90, 126.0, "euphoric", "build", ["dance", "pop", "reggaeton"]),
    "general": (0.55, 105.0, "happy", "balanced", []),
}


@dataclass(frozen=True)
class PreferenceProfile:
    """A validated interpretation of a natural-language playlist request."""

    activity: str
    playlist_length: int
    preferred_genres: tuple[str, ...]
    target_mood: str
    target_energy: float
    target_tempo_bpm: float
    playlist_structure: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["preferred_genres"] = list(self.preferred_genres)
        return result


def _number(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return min(max(number, minimum), maximum)


def validate_profile(
    raw: Mapping[str, Any],
    *,
    supported_genres: set[str],
    supported_moods: set[str],
) -> tuple[PreferenceProfile, list[str]]:
    """Validate and normalize untrusted model output without raising on bad values."""
    warnings: list[str] = []
    activity = str(raw.get("activity", "general")).strip().casefold()
    if activity not in SUPPORTED_ACTIVITIES:
        warnings.append(f"Unsupported activity '{activity}' replaced with 'general'.")
        activity = "general"

    default_energy, default_tempo, default_mood, default_structure, default_genres = ACTIVITY_DEFAULTS[activity]
    raw_length = _number(raw.get("playlist_length"), DEFAULT_PLAYLIST_LENGTH, 1, float("inf"))
    requested_length = int(min(raw_length, MAX_PLAYLIST_LENGTH))
    if raw_length > MAX_PLAYLIST_LENGTH:
        warnings.append(
            f"Requested {int(raw_length)} songs; the maximum playlist length is {MAX_PLAYLIST_LENGTH}, "
            f"so {MAX_PLAYLIST_LENGTH} will be returned."
        )

    supplied_genres = raw.get("preferred_genres", default_genres)
    if isinstance(supplied_genres, str):
        supplied_genres = [supplied_genres]
    if not isinstance(supplied_genres, (list, tuple)):
        supplied_genres = default_genres
    genres = tuple(
        dict.fromkeys(
            str(genre).strip().casefold()
            for genre in supplied_genres
            if str(genre).strip().casefold() in supported_genres
        )
    )
    if supplied_genres and not genres:
        warnings.append("Unsupported genres were ignored; scoring will use mood, energy, and tempo.")

    mood = str(raw.get("target_mood", default_mood)).strip().casefold()
    if mood not in supported_moods:
        warnings.append(f"Unsupported mood '{mood}' replaced with '{default_mood}'.")
        mood = default_mood if default_mood in supported_moods else next(iter(sorted(supported_moods)))

    structure = str(raw.get("playlist_structure", default_structure)).strip().casefold()
    if structure not in SUPPORTED_STRUCTURES:
        warnings.append(f"Unsupported structure '{structure}' replaced with '{default_structure}'.")
        structure = default_structure

    profile = PreferenceProfile(
        activity=activity,
        playlist_length=requested_length,
        preferred_genres=genres,
        target_mood=mood,
        target_energy=_number(raw.get("target_energy"), default_energy, 0.0, 1.0),
        target_tempo_bpm=_number(raw.get("target_tempo_bpm"), default_tempo, 40.0, 220.0),
        playlist_structure=structure,
    )
    return profile, warnings


def infer_profile_offline(
    request: str,
    *,
    supported_genres: set[str],
    supported_moods: set[str],
) -> tuple[PreferenceProfile, list[str]]:
    """Infer a safe profile locally when Gemini is unavailable or disabled."""
    text = request.casefold()
    activity = "general"
    activity_terms = {
        "workout": ("gym", "workout", "lifting", "exercise"),
        "running": ("run", "running", "jog", "marathon"),
        "studying": ("study", "studying", "homework", "focus", "coding", "reading"),
        "driving": ("drive", "driving", "road trip", "commute"),
        "relaxing": ("relax", "relaxing", "unwind", "sleep", "calm"),
        "party": ("party", "dance floor", "celebration"),
    }
    for candidate, terms in activity_terms.items():
        if any(term in text for term in terms):
            activity = candidate
            break

    energy, tempo, mood, structure, genres = ACTIVITY_DEFAULTS[activity]
    length_match = re.search(r"\b(\d{1,3})\b(?=.{0,25}\b(?:songs?|tracks?)\b)", text)
    length = int(length_match.group(1)) if length_match else DEFAULT_PLAYLIST_LENGTH

    explicit_genres = [genre for genre in sorted(supported_genres, key=len, reverse=True) if genre in text]
    explicit_moods = [candidate for candidate in sorted(supported_moods, key=len, reverse=True) if candidate in text]
    if explicit_genres:
        genres = explicit_genres[:3]
    if explicit_moods:
        mood = explicit_moods[0]
    if any(term in text for term in ("low energy", "gentle", "quiet", "calm")):
        energy = min(energy, 0.28)
    if any(term in text for term in ("high energy", "intense", "motivated", "fast")):
        energy = max(energy, 0.88)
    if "start slow" in text or "warm-up" in text or "warmup" in text or "cooldown" in text:
        structure = "warmup_peak_cooldown"
    if "consistent" in text or "steady" in text:
        structure = "consistent"

    profile, warnings = validate_profile(
        {
            "activity": activity,
            "playlist_length": length,
            "preferred_genres": genres,
            "target_mood": mood,
            "target_energy": energy,
            "target_tempo_bpm": tempo,
            "playlist_structure": structure,
        },
        supported_genres=supported_genres,
        supported_moods=supported_moods,
    )
    return profile, ["Gemini was not used; local keyword-based interpretation is active.", *warnings]
