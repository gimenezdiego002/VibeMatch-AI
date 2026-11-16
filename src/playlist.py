"""Deterministic scoring and situation-aware playlist sequencing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from src.profiles import PreferenceProfile


WEIGHTS = {"energy": 0.35, "mood": 0.25, "genre": 0.25, "tempo": 0.15}


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


def _similarity(value: float, target: float, range_size: float) -> float:
    return max(0.0, 1.0 - abs(value - target) / range_size)


def score_song_for_profile(
    song: Mapping[str, Any],
    profile: PreferenceProfile,
    *,
    target_energy: float | None = None,
) -> tuple[float, dict[str, float], str]:
    """Return a normalized compatibility score and auditable contributions."""
    energy_target = profile.target_energy if target_energy is None else target_energy
    components = {
        "energy": _similarity(float(song["energy"]), energy_target, 1.0),
        "mood": float(str(song["mood"]).casefold() == profile.target_mood),
        "genre": float(str(song["genre"]).casefold() in profile.preferred_genres)
        if profile.preferred_genres
        else 0.5,
        "tempo": _similarity(float(song["tempo_bpm"]), profile.target_tempo_bpm, 180.0),
    }
    contributions = {name: round(components[name] * weight, 4) for name, weight in WEIGHTS.items()}
    score = round(sum(contributions.values()) * 100.0, 2)
    strongest = sorted(contributions, key=lambda name: (-contributions[name], name))[:2]
    reason = "; ".join(f"{name} fit {components[name]:.0%}" for name in strongest)
    return score, contributions, reason


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
    """Greedily sequence verified catalog songs with deterministic tie-breaking."""
    catalog = list(songs)
    warnings: list[str] = []
    wanted = profile.playlist_length
    if wanted > len(catalog):
        warnings.append(f"Requested {wanted} songs, but the catalog contains only {len(catalog)}; returning all available songs.")
    count = min(wanted, len(catalog))
    unique_artists = {str(song["artist"]).casefold() for song in catalog}
    artist_limit = 1 if len(unique_artists) >= count else 2
    artist_counts: dict[str, int] = {}
    selected_ids: set[int] = set()
    selected: list[ScoredSong] = []

    for target in energy_curve(profile)[:count]:
        candidates: list[tuple[float, float, int, Mapping[str, Any], dict[str, float], str]] = []
        for song in catalog:
            song_id = int(song["id"])
            artist = str(song["artist"]).casefold()
            if song_id in selected_ids or artist_counts.get(artist, 0) >= artist_limit:
                continue
            score, contributions, reason = score_song_for_profile(song, profile, target_energy=target)
            progression_penalty = (
                0.0
                if profile.playlist_structure == "consistent"
                else 100.0 * abs(float(song["energy"]) - target)
            )
            candidates.append((score - progression_penalty, score, song_id, song, contributions, reason))
        if not candidates:
            artist_limit += 1
            for song in catalog:
                song_id = int(song["id"])
                if song_id not in selected_ids:
                    score, contributions, reason = score_song_for_profile(song, profile, target_energy=target)
                    progression_penalty = (
                        0.0
                        if profile.playlist_structure == "consistent"
                        else 100.0 * abs(float(song["energy"]) - target)
                    )
                    candidates.append((score - progression_penalty, score, song_id, song, contributions, reason))
        _, score, song_id, song, contributions, reason = sorted(
            candidates, key=lambda item: (-item[0], item[2])
        )[0]
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
    return PlaylistResult(name=playlist_name, profile=profile, songs=tuple(selected), warnings=tuple(warnings))
