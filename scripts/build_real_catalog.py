"""One-time builder for data/real_songs.csv (real tracks with real audio features).

Run it yourself; the dashboard never calls these services to build a catalog:

    python scripts/build_real_catalog.py            # uses data/seed_tracks.txt if present
    python scripts/build_real_catalog.py --target 120

Sources (all keyless, unofficial third-party services that can change):
  * ReccoBeats  - track discovery, Spotify IDs, ISRCs, audio features (energy, tempo, ...)
  * Deezer      - cover art, album, release date, genre, looked up by ISRC

Moods are DERIVED from valence/energy by the explicit rules in derive_mood(); they are not
provided by any source. Preview URLs are never stored because they expire.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED_FILE = ROOT / "data" / "seed_tracks.txt"
OUT_FILE = ROOT / "data" / "real_songs.csv"
USER_AGENT = "VibeMatchClassProject/0.1 (student project)"
DEFAULT_SEEDS = ["0VjIjW4GlUZAMYd2vXMi3b"]  # one ID verified to work

RECCO = "https://api.reccobeats.com/v1"
DEEZER = "https://api.deezer.com"

# ReccoBeats ignores energy/tempo filters on /track/recommendation (verified), so coverage is
# achieved by re-seeding from discovered tracks nearest the emptiest energy band instead.
BANDS = 10  # energy bands of width 0.1
EXCLUDED_GENRES = {"kids", "soundtrack"}
EXCLUDED_TITLE = re.compile(r"christmas|navidad|let it snow|cold outside|winter serenade|abc song|ao vivo|\blive\b|karaoke|\binstrumental\b|remix|sped up|slowed", re.I)
GENRE_MAP = {
    "pop": "pop", "dance": "dance", "electro": "electronic", "rap/hip hop": "hip-hop",
    "r&b": "r&b", "rock": "rock", "alternative": "indie pop", "indie": "indie pop",
    "jazz": "jazz", "classical": "classical", "latin music": "latin", "reggaeton": "reggaeton",
    "metal": "metal", "folk": "folk", "singer & songwriter": "folk", "chill out/trip-hop/lounge": "chill",
    "films/games": "soundtrack", "reggae": "reggae", "country": "country", "soul & funk": "r&b",
}
FIELDS = [
    "id", "title", "artist", "genre", "mood", "energy", "tempo_bpm", "valence", "danceability",
    "acousticness", "spotify_id", "isrc", "album", "release_date", "cover_url", "deezer_url", "fact_sources",
]


def get_json(url: str, retries: int = 3):
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=25) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Request failed after {retries} tries: {url.split('?')[0]} ({type(last).__name__})")


def spotify_id(href: str) -> str | None:
    match = re.search(r"/track/([A-Za-z0-9]{22})", href or "")
    return match.group(1) if match else None


def derive_mood(energy: float, valence: float, acousticness: float) -> str:
    """Documented rules mapping audio features to the catalog's existing mood vocabulary.

    Calm tracks are split so the study ("focused") and relax ("chill") presets have candidates:
      energy < 0.4:  valence < 0.3 -> reflective; acousticness >= 0.8 -> peaceful;
                     valence < 0.5 -> focused; otherwise chill
    """
    if energy >= 0.8:
        return "euphoric" if valence >= 0.7 else "energetic" if valence >= 0.5 else "intense"
    if energy >= 0.6:
        return "happy" if valence >= 0.65 else "confident" if valence >= 0.4 else "moody"
    if energy >= 0.4:
        return "happy" if valence >= 0.6 else "focused" if valence >= 0.35 else "reflective"
    if valence < 0.3:
        return "reflective"
    if acousticness >= 0.8:
        return "peaceful"
    return "focused" if valence < 0.5 else "chill"


def map_genre(names: list[str]) -> str | None:
    for name in names:
        key = name.strip().casefold()
        if key in GENRE_MAP:
            return GENRE_MAP[key]
    return None


def read_seeds() -> list[str]:
    if SEED_FILE.is_file():
        seeds = [spotify_id(line) or line.strip() for line in SEED_FILE.read_text(encoding="utf-8").splitlines()]
        seeds = [s for s in seeds if s and re.fullmatch(r"[A-Za-z0-9]{22}", s)]
        if seeds:
            return seeds
    return list(DEFAULT_SEEDS)


def add_features(pool: dict[str, dict], ids: list[str]) -> None:
    for start in range(0, len(ids), 40):
        batch = ids[start : start + 40]
        data = get_json(f"{RECCO}/audio-features?ids={','.join(batch)}")
        for row in data.get("content", []):
            sid = spotify_id(row.get("href", ""))
            if sid in pool:
                pool[sid]["features"] = row
        time.sleep(0.3)


def band_of(item: dict) -> int:
    return min(BANDS - 1, max(0, int(float(item["features"]["energy"]) * BANDS)))


def pick_next_seed(pool: dict[str, dict], used: set[str], want: int) -> str | None:
    """Pick an unused discovered track closest to the emptiest energy band."""
    counts = Counter(band_of(item) for item in pool.values() if item.get("features"))
    for band in sorted(range(BANDS), key=lambda b: (counts.get(b, 0), b)):
        if counts.get(band, 0) >= want:
            return None
        centre = (band + 0.5) / BANDS
        options = [
            (abs(float(item["features"]["energy"]) - centre), sid)
            for sid, item in pool.items()
            if item.get("features") and sid not in used
        ]
        if options:
            return min(options)[1]
    return None


def track_record(track: dict) -> dict | None:
    sid = spotify_id(track.get("href", ""))
    if not sid or not track.get("isrc"):
        return None
    artists = track.get("artists") or []
    return {
        "sid": sid,
        "title": track.get("trackTitle", ""),
        "artist": ", ".join(a["name"] for a in artists[:2]) or "Unknown",
        "isrc": track["isrc"],
        "primary": (artists[0] if artists else {}).get("name", ""),
    }


def expand(seeds: list[str], per_call: int, want_per_band: int, max_rounds: int) -> dict[str, dict]:
    """Grow a pool of tracks with features, steering toward under-filled energy bands."""
    pool: dict[str, dict] = {}

    def absorb(tracks: list[dict]) -> None:
        fresh = []
        for track in tracks:
            record = track_record(track)
            if record and record["sid"] not in pool:
                pool[record.pop("sid")] = record
                fresh.append(spotify_id(track["href"]))
        add_features(pool, fresh)

    seed_info = get_json(f"{RECCO}/track?ids={','.join(seeds[:40])}")
    absorb(seed_info.get("content", []))
    used: set[str] = set()
    queue = [s for s in seeds if s in pool]
    next_seed: str | None = queue.pop(0) if queue else None
    for round_number in range(1, max_rounds + 1):
        if next_seed is None:
            break
        used.add(next_seed)
        try:
            data = get_json(f"{RECCO}/track/recommendation?{urllib.parse.urlencode({'size': per_call, 'seeds': next_seed})}")
        except RuntimeError as exc:
            print(f"  skip seed {next_seed}: {exc}", file=sys.stderr)
            data = {}
        absorb(data.get("content", []))
        counts = Counter(band_of(item) for item in pool.values() if item.get("features"))
        print(f"  round {round_number}: pool {len(pool)}; per band {[counts.get(b, 0) for b in range(BANDS)]}")
        next_seed = queue.pop(0) if queue else pick_next_seed(pool, used, want_per_band)
        time.sleep(0.3)
    return {sid: item for sid, item in pool.items() if item.get("features")}


def enrich_deezer(isrc: str, allow_explicit: bool = False) -> dict | None:
    try:
        track = get_json(f"{DEEZER}/track/isrc:{urllib.parse.quote(isrc)}")
    except RuntimeError:
        return None
    if not isinstance(track, dict) or track.get("error") or not track.get("album"):
        return None
    if track.get("explicit_lyrics") and not allow_explicit:
        return None
    album = track["album"]
    cover = album.get("cover_big") or album.get("cover_medium")
    if not cover:
        return None
    genres: list[str] = []
    try:
        detail = get_json(f"{DEEZER}/album/{album['id']}")
        genres = [g["name"] for g in detail.get("genres", {}).get("data", [])]
        release = detail.get("release_date") or track.get("release_date") or ""
    except RuntimeError:
        release = track.get("release_date") or ""
    genre = map_genre(genres) or (re.sub(r"[^a-z&/ -]", "", genres[0].casefold()) if genres else "other")
    if genre in EXCLUDED_GENRES:
        return None
    return {
        "album": album.get("title", ""),
        "cover_url": cover,
        "deezer_url": track.get("link", ""),
        "release_date": release,
        "genre": genre,
    }


def zip_longest_flat(groups: list[list]) -> list[list]:
    """Yield one-item rounds taking the next element from each non-empty group."""
    rounds: list[list] = []
    depth = max((len(group) for group in groups), default=0)
    for index in range(depth):
        rounds.append([group[index] for group in groups if index < len(group)])
    return rounds


def build(target: int, per_call: int, max_per_artist: int, allow_explicit: bool = False, max_rounds: int = 30) -> list[dict]:
    seeds = read_seeds()
    print(f"Seeds: {len(seeds)} ({'from data/seed_tracks.txt' if SEED_FILE.is_file() else 'built-in default'})")
    print("Discovering tracks via ReccoBeats (steering toward every energy band)...")
    pool = expand(seeds, per_call, want_per_band=max(6, target // 4), max_rounds=max_rounds)
    print(f"Usable pool: {len(pool)} tracks with audio features")

    rows: list[dict] = []
    per_artist: Counter[str] = Counter()
    seen_isrc: set[str] = set()
    # Visit candidates round-robin across 10 energy bands so every scenario keeps candidates.
    bands: list[list[tuple[str, dict]]] = [[] for _ in range(10)]
    for sid, item in sorted(pool.items(), key=lambda kv: kv[0]):
        if item.get("features"):
            band = min(9, max(0, int(float(item["features"].get("energy", 0)) * 10)))
            bands[band].append((sid, item))
    ordered = [pair for group in zip_longest_flat(bands) for pair in group]
    for sid, item in ordered:
        if len(rows) >= target:
            break
        f = item["features"]
        if EXCLUDED_TITLE.search(item["title"]) or item["isrc"] in seen_isrc:
            continue
        if per_artist[item["primary"].casefold()] >= max_per_artist:
            continue
        try:
            energy, valence = float(f["energy"]), float(f["valence"])
            dance, acoustic, tempo = float(f["danceability"]), float(f["acousticness"]), round(float(f["tempo"]))
        except (KeyError, TypeError, ValueError):
            continue
        if not (40 <= tempo <= 220) or not all(0 <= v <= 1 for v in (energy, valence, dance, acoustic)):
            continue
        extra = enrich_deezer(item["isrc"], allow_explicit)
        time.sleep(0.15)
        if not extra:
            continue
        seen_isrc.add(item["isrc"])
        per_artist[item["primary"].casefold()] += 1
        rows.append(
            {
                "title": item["title"],
                "artist": item["artist"],
                "genre": extra["genre"],
                "mood": derive_mood(energy, valence, acoustic),
                "energy": round(energy, 3),
                "tempo_bpm": tempo,
                "valence": round(valence, 3),
                "danceability": round(dance, 3),
                "acousticness": round(acoustic, 3),
                "spotify_id": sid,
                "isrc": item["isrc"],
                "album": extra["album"],
                "release_date": extra["release_date"],
                "cover_url": extra["cover_url"],
                "deezer_url": extra["deezer_url"],
                "fact_sources": "Deezer (album, release date, cover); ReccoBeats (audio features); mood derived locally",
            }
        )
        print(f"  + {len(rows):>3} {item['title'][:34]:<34} {item['artist'][:22]:<22} {extra['genre']:<10} e={energy:.2f}")
    rows.sort(key=lambda r: (r["energy"], r["title"]))
    for index, row in enumerate(rows, start=1):
        row["id"] = index
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", type=int, default=100, help="approximate number of tracks to keep")
    parser.add_argument("--per-call", type=int, default=40, help="recommendations requested per call")
    parser.add_argument("--max-per-artist", type=int, default=2)
    parser.add_argument("--allow-explicit", action="store_true", help="include tracks Deezer marks explicit (default: skip)")
    parser.add_argument("--max-rounds", type=int, default=30, help="maximum recommendation calls")
    parser.add_argument("--out", default=str(OUT_FILE))
    args = parser.parse_args()

    rows = build(args.target, args.per_call, args.max_per_artist, args.allow_explicit, args.max_rounds)
    if len(rows) < 12:
        raise SystemExit(f"Only {len(rows)} usable tracks found; not writing a catalog. Add more seeds to data/seed_tracks.txt.")
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    energies = [r["energy"] for r in rows]
    print(f"\nWrote {len(rows)} tracks to {args.out}")
    print(f"Energy range {min(energies):.2f}-{max(energies):.2f}; moods {dict(Counter(r['mood'] for r in rows))}")
    print(f"Genres {dict(Counter(r['genre'] for r in rows))}")
    print("Review the list above and delete any rows you do not want before using the catalog.")


if __name__ == "__main__":
    main()
