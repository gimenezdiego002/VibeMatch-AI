"""Pure presentation helpers for the Streamlit dashboard.

These functions only reshape results that the recommendation pipeline already
produced. They never score, select, or reorder songs; that remains the job of
``src.playlist`` and ``src.app``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from html import escape
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from src.media import origin_note, spotify_url
from src.playlist import WEIGHTS, PlaylistResult, generate_playlist


SONG_FIELDS = (
    "id",
    "title",
    "artist",
    "genre",
    "mood",
    "energy",
    "tempo_bpm",
    "valence",
    "danceability",
    "acousticness",
)
# Optional columns present only in the real-music catalog.
EXTRA_FIELDS = ("spotify_id", "isrc", "album", "release_date", "cover_url", "deezer_url", "fact_sources")


def is_real_catalog(catalog: Sequence[Mapping[str, Any]]) -> bool:
    return bool(catalog) and "spotify_id" in catalog[0]


def placeholder_tile_html(song: Mapping[str, Any]) -> str:
    """Generated cover tile for songs without artwork (the fictional catalog).

    Colors come from a hash of the song ID so a song always gets the same tile. The only
    dynamic text is one escaped character.
    """
    digest = hashlib.sha256(str(song.get("id", "")).encode()).digest()
    hue = digest[0] / 255 * 360
    letter = escape((str(song.get("title", "?")).strip() or "?")[0].upper())
    return (
        f'<div class="vm-tile" role="img" aria-label="Generated cover tile" '
        f'style="background:linear-gradient(135deg,hsl({hue:.0f} 55% 38%),hsl({(hue + 50) % 360:.0f} 60% 24%))">{letter}</div>'
    )


def title_html(song: Mapping[str, Any]) -> str:
    return (
        f'<div class="vm-title">{escape(str(song.get("title", "")))}</div>'
        f'<div class="vm-sub">{escape(str(song.get("artist", "")))}</div>'
    )


def card_meta_html(rank: int, score: float, genre: str, mood: str) -> str:
    """Rank chip, score pill, and tags. All dynamic text is escaped."""
    return (
        '<div class="vm-meta">'
        f'<span class="vm-rank">#{int(rank)}</span>'
        f'<span class="vm-score" title="Compatibility score out of 100">{float(score):.0f}</span>'
        f'<span class="vm-tags">{escape(str(genre))} · {escape(str(mood))}</span>'
        "</div>"
    )


def energy_meter_html(energy: float, target: float, tempo: Any) -> str:
    """Accessible meter: the fill is the song's energy, the tick is the position's target."""
    energy_pct = min(100.0, max(0.0, float(energy) * 100))
    target_pct = min(100.0, max(0.0, float(target) * 100))
    label = f"Energy {float(energy):.2f}, target {float(target):.2f}"
    return (
        f'<div class="vm-meter" role="meter" aria-valuemin="0" aria-valuemax="1" '
        f'aria-valuenow="{float(energy):.2f}" aria-label="{escape(label)}">'
        f'<span style="width:{energy_pct:.1f}%"></span><i style="left:{target_pct:.1f}%"></i></div>'
        f'<div class="vm-meter-label"><span>Energy {float(energy):.2f} · target {float(target):.2f}</span>'
        f"<span>{escape(str(tempo))} BPM</span></div>"
    )


def phase_bands(result: PlaylistResult) -> pd.DataFrame:
    """Warm-up / peak / cooldown spans (by position) for warm-up structures; empty otherwise.

    Mirrors the 25% / 50% / 25% split used by src.playlist.energy_curve.
    """
    count = len(result.songs)
    if result.profile.playlist_structure != "warmup_peak_cooldown" or count < 3:
        return pd.DataFrame(columns=["phase", "start", "end"])
    labels = []
    for index in range(count):
        progress = index / (count - 1)
        labels.append("Warm-up" if progress <= 0.25 else "Peak" if progress <= 0.75 else "Cooldown")
    rows: list[dict[str, Any]] = []
    for position, label in enumerate(labels, start=1):
        if rows and rows[-1]["phase"] == label:
            rows[-1]["end"] = position + 0.5
        else:
            rows.append({"phase": label, "start": position - 0.5, "end": position + 0.5})
    return pd.DataFrame(rows, columns=["phase", "start", "end"])


def interpretation_source(result: PlaylistResult) -> str:
    return "Gemini" if result.used_gemini_interpretation else "Local fallback"


def explanation_source(result: PlaylistResult) -> str:
    return "Gemini" if result.used_gemini_explanations else "Deterministic reasons"


def playlist_dataframe(result: PlaylistResult) -> pd.DataFrame:
    """One row per selected song, in playlist order."""
    rows = [
        {
            "rank": rank,
            "id": int(item.song["id"]),
            "title": str(item.song["title"]),
            "artist": str(item.song["artist"]),
            "genre": str(item.song["genre"]),
            "mood": str(item.song["mood"]),
            "score": float(item.score),
            "energy": float(item.song["energy"]),
            "target_energy": float(item.position_target_energy),
            "tempo_bpm": int(item.song["tempo_bpm"]),
            "explanation": item.explanation,
        }
        for rank, item in enumerate(result.songs, start=1)
    ]
    columns = ["rank", "id", "title", "artist", "genre", "mood", "score", "energy", "target_energy", "tempo_bpm", "explanation"]
    return pd.DataFrame(rows, columns=columns)


def contributions_dataframe(result: PlaylistResult) -> pd.DataFrame:
    """Feature contributions in score points (0-100 scale) for every selected song."""
    rows = []
    for rank, item in enumerate(result.songs, start=1):
        row: dict[str, Any] = {"rank": rank, "title": str(item.song["title"])}
        for feature in WEIGHTS:
            row[feature] = round(float(item.contributions.get(feature, 0.0)) * 100.0, 2)
        row["score"] = float(item.score)
        rows.append(row)
    return pd.DataFrame(rows, columns=["rank", "title", *WEIGHTS, "score"])


def song_contributions(result: PlaylistResult, rank: int) -> pd.DataFrame:
    """Earned versus maximum points per feature for one ranked song."""
    item = result.songs[rank - 1]
    return pd.DataFrame(
        [
            {
                "feature": feature,
                "earned_points": round(float(item.contributions.get(feature, 0.0)) * 100.0, 2),
                "max_points": round(weight * 100.0, 2),
            }
            for feature, weight in WEIGHTS.items()
        ]
    )


def energy_chart_data(result: PlaylistResult) -> pd.DataFrame:
    """Long-form data comparing actual and target energy by playlist position."""
    rows = []
    for position, item in enumerate(result.songs, start=1):
        title = str(item.song["title"])
        rows.append({"position": position, "series": "Actual song energy", "energy": float(item.song["energy"]), "title": title})
        rows.append({"position": position, "series": "Target energy", "energy": float(item.position_target_energy), "title": title})
    return pd.DataFrame(rows, columns=["position", "series", "energy", "title"])


def describe_energy_shape(result: PlaylistResult) -> str:
    """Plain-language summary of the energy curve, used for chart alt text."""
    energies = [float(item.song["energy"]) for item in result.songs]
    if not energies:
        return "No songs were selected."
    peak = max(energies)
    peak_position = energies.index(peak) + 1
    return (
        f"Energy across {len(energies)} positions for a {result.profile.playlist_structure.replace('_', ' ')} playlist: "
        f"starts at {energies[0]:.2f}, peaks at {peak:.2f} at position {peak_position}, ends at {energies[-1]:.2f}."
    )


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool | None  # None means informational rather than pass/fail.
    detail: str


def system_checks(
    result: PlaylistResult,
    catalog: Sequence[Mapping[str, Any]],
    *,
    gemini_requested: bool,
) -> list[Check]:
    """Verify catalog integrity of a finished result and summarize AI/fallback usage."""
    catalog_ids = {int(song["id"]) for song in catalog}
    ids = [int(item.song["id"]) for item in result.songs]
    requested = result.profile.playlist_length
    unknown = sorted(set(ids) - catalog_ids)

    catalog_by_id = {int(song["id"]): song for song in catalog}
    metadata_mismatches = [
        song_id
        for song_id, item in zip(ids, result.songs)
        if song_id in catalog_by_id
        and any(str(item.song[field]) != str(catalog_by_id[song_id][field]) for field in SONG_FIELDS if field in item.song)
    ]

    # Re-run only the deterministic Python selection on the already-validated profile.
    # This never calls Gemini, so it is safe and cheap to repeat.
    rerun_ids = [int(item.song["id"]) for item in generate_playlist(catalog, result.profile).songs]

    checks = [
        Check(
            "Selected IDs exist in the local catalog",
            not unknown,
            f"{len(ids)} of {len(ids)} IDs found." if not unknown else f"Unknown IDs: {unknown}",
        ),
        Check(
            "Metadata matches the catalog",
            not metadata_mismatches,
            "Title, artist, genre, mood, energy, and tempo match the CSV."
            if not metadata_mismatches
            else f"Mismatched IDs: {metadata_mismatches}",
        ),
        Check("Selected IDs are unique", len(ids) == len(set(ids)), f"{len(set(ids))} unique of {len(ids)} returned."),
        Check(
            "Returned length matches requested length",
            len(ids) == requested,
            f"Requested {requested}, returned {len(ids)}."
            + ("" if len(ids) == requested else " See warnings for the reason."),
        ),
        Check(
            "Python selection is deterministic",
            rerun_ids == ids,
            "Re-running selection on the same validated profile produced the same ordered IDs."
            if rerun_ids == ids
            else "A repeat run produced a different order.",
        ),
    ]

    length_notes = [w for w in result.warnings if "maximum playlist length" in w or "catalog contains only" in w]
    if length_notes:
        checks.append(Check("Playlist length limits", None, " ".join(length_notes)))

    if result.used_gemini_interpretation:
        interpretation ="Gemini interpreted the request; Python validated the profile."
    elif gemini_requested:
        interpretation = "Gemini was requested but not used successfully; the local keyword fallback interpreted the request."
    else:
        interpretation = "Gemini was off; the local keyword fallback interpreted the request."
    checks.append(Check("Request interpretation", None, interpretation))
    checks.append(
        Check(
            "Explanations",
            None,
            "Gemini wrote explanations for the verified selections (IDs validated)."
            if result.used_gemini_explanations
            else "Deterministic score reasons are shown; no AI-generated text.",
        )
    )
    checks.append(
        Check(
            "Warnings",
            True if not result.warnings else None,
            "No warnings raised." if not result.warnings else f"{len(result.warnings)} warning(s) raised; review them above.",
        )
    )
    return checks


def catalog_dataframe(catalog: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    songs = list(catalog)
    extras = [field for field in ("cover_url", "album", "release_date") if songs and field in songs[0]]
    columns = [*SONG_FIELDS, *extras]
    return pd.DataFrame([{field: song[field] for field in columns} for song in songs], columns=columns)


def catalog_summary(catalog: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    energies = [float(song["energy"]) for song in catalog]
    tempos = [int(song["tempo_bpm"]) for song in catalog]
    return {
        "songs": len(catalog),
        "genres": len({str(song["genre"]).casefold() for song in catalog}),
        "moods": len({str(song["mood"]).casefold() for song in catalog}),
        "energy_range": (min(energies), max(energies)) if energies else (0.0, 0.0),
        "tempo_range": (min(tempos), max(tempos)) if tempos else (0, 0),
    }


def playlist_csv(result: PlaylistResult) -> str:
    """CSV export of verified metadata, scores, targets, and explanations. Contains no secrets."""
    rows = []
    for rank, item in enumerate(result.songs, start=1):
        row: dict[str, Any] = {"rank": rank}
        row.update({field: item.song[field] for field in SONG_FIELDS})
        row["compatibility_score"] = float(item.score)
        row["position_target_energy"] = float(item.position_target_energy)
        for feature in WEIGHTS:
            row[f"{feature}_points"] = round(float(item.contributions.get(feature, 0.0)) * 100.0, 2)
        row["deterministic_reason"] = item.deterministic_reason
        row["explanation"] = item.explanation
        for field in EXTRA_FIELDS:
            if field in item.song:
                row[field] = item.song[field]
        link = spotify_url(item.song)
        if link:
            row["spotify_url"] = link
        note = origin_note(item.song)
        if note:
            row["origin_note"], row["origin_note_source"] = note
        row["explanation_source"] = explanation_source(result)
        row["interpretation_source"] = interpretation_source(result)
        row["activity"] = result.profile.activity
        rows.append(row)
    return pd.DataFrame(rows).to_csv(index=False)
