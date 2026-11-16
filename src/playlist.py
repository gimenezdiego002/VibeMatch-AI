"""Deterministic scoring and situation-aware playlist sequencing.

Python computes every number here. Gemini may only *suggest* feature priorities through the
validated profile; this module clamps and normalizes them before any song is scored.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

from src.profiles import PRIORITY_FEATURES, PreferenceProfile


# Default weights for the four core features (unchanged from the original scorer).
WEIGHTS = {"energy": 0.35, "mood": 0.25, "genre": 0.25, "tempo": 0.15}
# Default weights for optional features. They join only when the profile has a SOFT
# constraint for them; hard constraints are enforced as filters instead.
OPTIONAL_WEIGHTS = {"language": 0.20, "recency": 0.15}
PRIORITY_BOUNDS = (0.05, 0.60)  # a suggested priority can never zero out or dominate a feature
RECENCY_DECAY_YEARS = 10  # a song this many years outside the requested range scores 0 for recency


def _pairs(table: dict[tuple[str, str], float]) -> dict[frozenset[str], float]:
    return {frozenset(pair): value for pair, value in table.items()}


# Graded similarity. Exact match is always 1.0; anything not listed is 0.0. Symmetric.
GENRE_SIMILARITY = _pairs({
    ("latin", "reggaeton"): 0.8, ("latin", "traditional mexicano"): 0.7, ("reggaeton", "hip-hop"): 0.5,
    ("pop", "indie pop"): 0.7, ("pop", "dance"): 0.6, ("pop", "r&b"): 0.5, ("dance", "electronic"): 0.8,
    ("electronic", "synthwave"): 0.7, ("rock", "metal"): 0.6, ("rock", "indie pop"): 0.4,
    ("hip-hop", "r&b"): 0.6, ("lofi", "ambient"): 0.7, ("lofi", "jazz"): 0.5, ("ambient", "classical"): 0.6,
    ("folk", "country"): 0.6, ("folk", "indie pop"): 0.5, ("brazilian music", "latin"): 0.6,
})
MOOD_SIMILARITY = _pairs({
    ("intense", "energetic"): 0.8, ("intense", "aggressive"): 0.8, ("energetic", "euphoric"): 0.7,
    ("energetic", "confident"): 0.6, ("euphoric", "happy"): 0.7, ("happy", "confident"): 0.6,
    ("chill", "peaceful"): 0.8, ("chill", "relaxed"): 0.9, ("peaceful", "relaxed"): 0.8,
    ("focused", "chill"): 0.6, ("focused", "reflective"): 0.5, ("reflective", "moody"): 0.7,
    ("romantic", "happy"): 0.4, ("moody", "intense"): 0.4,
})


@dataclass(frozen=True)
class ScoredSong:
    song: Mapping[str, Any]
    score: float
    contributions: Mapping[str, float]
    deterministic_reason: str
    position_target_energy: float
    explanation: str = ""


@dataclass(frozen=True)
class PlaylistResult:
    name: str
    profile: PreferenceProfile
    songs: tuple[ScoredSong, ...]
    warnings: tuple[str, ...] = ()
    used_gemini_interpretation: bool = False
    used_gemini_explanations: bool = False
    weights: tuple[tuple[str, float], ...] = ()  # the normalized weights actually used


def label_similarity(a: str, b: str, table: Mapping[frozenset[str], float]) -> float:
    a, b = str(a).casefold(), str(b).casefold()
    if a == b:
        return 1.0
    return table.get(frozenset((a, b)), 0.0)


def song_year(song: Mapping[str, Any]) -> int | None:
    text = str(song.get("release_date", "") or "").strip()
    return int(text[:4]) if text[:4].isdigit() else None


def effective_weights(profile: PreferenceProfile) -> dict[str, float]:
    """Active features and their weights (summing to 1) for this profile.

    Core features always apply. Language/recency join only for SOFT constraints. Gemini's
    suggested priorities replace a feature's default weight after clamping to PRIORITY_BOUNDS,
    then everything is normalized. Deterministic for a given profile.
    """
    base = dict(WEIGHTS)
    if profile.language and not profile.language_strict:
        base["language"] = OPTIONAL_WEIGHTS["language"]
    if profile.has_year_range and not profile.recency_strict:
        base["recency"] = OPTIONAL_WEIGHTS["recency"]
    low, high = PRIORITY_BOUNDS
    for feature, priority in profile.priorities:
        if feature in base:
            base[feature] = min(max(float(priority), low), high)
    total = sum(base.values())
    ordered = [feature for feature in PRIORITY_FEATURES if feature in base]
    # Not rounded: rounded weights can sum to slightly more than 1. Round only for display.
    return {feature: base[feature] / total for feature in ordered}


def _similarity(value: float, target: float, range_size: float) -> float:
    return max(0.0, 1.0 - abs(value - target) / range_size)


def _recency_fit(year: int | None, profile: PreferenceProfile) -> float:
    if year is None:
        return 0.5  # unknown release date: neutral, not rewarded or punished
    if profile.min_release_year and year < profile.min_release_year:
        return max(0.0, 1.0 - (profile.min_release_year - year) / RECENCY_DECAY_YEARS)
    if profile.max_release_year and year > profile.max_release_year:
        return max(0.0, 1.0 - (year - profile.max_release_year) / RECENCY_DECAY_YEARS)
    return 1.0


def score_song_for_profile(
    song: Mapping[str, Any],
    profile: PreferenceProfile,
    *,
    target_energy: float | None = None,
    weights: Mapping[str, float] | None = None,
) -> tuple[float, dict[str, float], str]:
    """Return a 0-100 compatibility score, per-feature contributions, and a short reason.

    Contributions are component (0-1) times weight, so they always sum to the score / 100.
    """
    weights = weights or effective_weights(profile)
    energy_target = profile.target_energy if target_energy is None else target_energy
    genre = str(song["genre"])
    components = {
        "energy": _similarity(float(song["energy"]), energy_target, 1.0),
        "mood": label_similarity(song["mood"], profile.target_mood, MOOD_SIMILARITY),
        "genre": max((label_similarity(genre, g, GENRE_SIMILARITY) for g in profile.preferred_genres), default=0.0)
        if profile.preferred_genres
        else 0.5,
        "tempo": _similarity(float(song["tempo_bpm"]), profile.target_tempo_bpm, 180.0),
    }
    if "language" in weights:
        song_language = str(song.get("language", "") or "").casefold()
        components["language"] = 0.5 if not song_language else float(song_language == profile.language)
    if "recency" in weights:
        components["recency"] = _recency_fit(song_year(song), profile)
    # 6 decimals keeps the sum of shown contributions within 0.01 points of the score.
    contributions = {name: round(components[name] * weight, 6) for name, weight in weights.items()}
    raw_total = sum(components[name] * weight for name, weight in weights.items())
    score = min(100.0, max(0.0, round(raw_total * 100.0, 2)))
    strongest = sorted(contributions, key=lambda name: (-contributions[name], name))[:2]
    reason = "; ".join(f"{name} fit {components[name]:.0%}" for name in strongest)
    return score, contributions, reason


def matches_hard_constraints(song: Mapping[str, Any], profile: PreferenceProfile) -> bool:
    if profile.language and profile.language_strict:
        if str(song.get("language", "") or "").casefold() != profile.language:
            return False
    if profile.has_year_range and profile.recency_strict:
        year = song_year(song)
        if year is None:
            return False
        if profile.min_release_year and year < profile.min_release_year:
            return False
        if profile.max_release_year and year > profile.max_release_year:
            return False
    return True


def describe_hard_constraints(profile: PreferenceProfile) -> str:
    """Plain-language phrase for the active hard filters, e.g. 'Spanish songs released 2021 or later'."""
    language = profile.language.title() + " " if profile.language and profile.language_strict else ""
    years = ""
    if profile.has_year_range and profile.recency_strict:
        low, high = profile.min_release_year, profile.max_release_year
        years = (f" released {low}-{high}" if low and high
                 else f" released {low} or later" if low
                 else f" released {high} or earlier")
    return f"{language}songs{years}" if language or years else ""


def energy_curve(profile: PreferenceProfile) -> list[float]:
    """Create per-position energy targets appropriate to the requested structure."""
    length = profile.playlist_length
    base = profile.target_energy
    if length == 1:
        return [base]
    if profile.playlist_structure == "consistent":
        return [base] * length
    if profile.playlist_structure == "build":
        start = max(0.0, base - 0.25)
        return [round(start + (base - start) * index / (length - 1), 3) for index in range(length)]
    if profile.playlist_structure == "warmup_peak_cooldown":
        warm = max(0.35, base - 0.28)
        cool = max(0.20, base - 0.38)
        peak = min(1.0, base + 0.08)
        targets: list[float] = []
        for index in range(length):
            progress = index / (length - 1)
            if progress <= 0.25:
                value = warm + (peak - warm) * progress / 0.25
            elif progress <= 0.75:
                value = peak
            else:
                value = peak + (cool - peak) * (progress - 0.75) / 0.25
            targets.append(round(value, 3))
        return targets
    return [round(min(1.0, max(0.0, base + offset)), 3) for offset in ([0.0, 0.06, -0.04] * length)[:length]]


def generate_playlist(
    songs: Iterable[Mapping[str, Any]],
    profile: PreferenceProfile,
    *,
    name: str | None = None,
) -> PlaylistResult:
    """Filter by hard constraints, then greedily sequence verified catalog songs.

    Ties break on song ID, so the same validated profile always yields the same playlist.
    """
    catalog = list(songs)
    warnings: list[str] = []
    wanted = profile.playlist_length
    weights = effective_weights(profile)

    pool = [song for song in catalog if matches_hard_constraints(song, profile)]
    constraint_text = describe_hard_constraints(profile)
    if constraint_text and len(pool) < wanted:
        if pool:
            warnings.append(
                f"Only {len(pool)} {constraint_text} fit this catalog; returning {len(pool)} instead of {wanted}."
            )
        else:
            warnings.append(f"No {constraint_text} exist in this catalog, so no songs were returned.")
    elif wanted > len(catalog):
        warnings.append(f"Requested {wanted} songs, but the catalog contains only {len(catalog)}; returning all available songs.")
    count = min(wanted, len(pool))

    unique_artists = {str(song["artist"]).casefold() for song in pool}
    artist_limit = 1 if len(unique_artists) >= count else 2
    artist_counts: dict[str, int] = {}
    selected_ids: set[int] = set()
    selected: list[ScoredSong] = []
    # Build the energy arc for the number of songs actually available, so a shortage still
    # gets a complete warm-up / peak / cooldown instead of a truncated one.
    curve = energy_curve(replace(profile, playlist_length=count)) if count else []

    def candidates_for(target: float, limit: int) -> list[tuple[float, float, int, Mapping[str, Any], dict[str, float], str]]:
        found = []
        for song in pool:
            song_id = int(song["id"])
            artist = str(song["artist"]).casefold()
            if song_id in selected_ids or artist_counts.get(artist, 0) >= limit:
                continue
            score, contributions, reason = score_song_for_profile(song, profile, target_energy=target, weights=weights)
            progression_penalty = (
                0.0 if profile.playlist_structure == "consistent" else 100.0 * abs(float(song["energy"]) - target)
            )
            found.append((score - progression_penalty, score, song_id, song, contributions, reason))
        return found

    for target in curve:
        candidates = candidates_for(target, artist_limit)
        while not candidates:
            artist_limit += 1
            candidates = candidates_for(target, artist_limit)
        _, score, song_id, song, contributions, reason = sorted(candidates, key=lambda item: (-item[0], item[2]))[0]
        selected_ids.add(song_id)
        artist = str(song["artist"]).casefold()
        artist_counts[artist] = artist_counts.get(artist, 0) + 1
        selected.append(
            ScoredSong(
                song=song,
                score=score,
                contributions=contributions,
                deterministic_reason=reason,
                position_target_energy=target,
                explanation=reason,
            )
        )

    playlist_name = name or f"{profile.activity.title()} VibeMatch"
    return PlaylistResult(
        name=playlist_name,
        profile=profile,
        songs=tuple(selected),
        warnings=tuple(warnings),
        weights=tuple(weights.items()),
    )
