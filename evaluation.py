"""Reproducible offline evaluation for representative VibeMatch scenarios."""

from src.app import create_playlist
from src.recommender import load_songs


SCENARIOS = {
    "workout": "Make a 10-song high-energy gym workout with a warm-up and cooldown",
    "study": "Give me 10 relaxing low-energy songs for studying with a steady feel",
    "relax": "Create 8 peaceful songs for relaxing after work",
}


def main() -> None:
    catalog_ids = {song["id"] for song in load_songs("data/songs.csv")}
    print(f"Catalog records: {len(catalog_ids)}")
    for label, request in SCENARIOS.items():
        first = create_playlist(request, offline=True)
        second = create_playlist(request, offline=True)
        ids = [item.song["id"] for item in first.songs]
        energies = [float(item.song["energy"]) for item in first.songs]
        print(f"\n{label.upper()}: {request}")
        print(f"activity={first.profile.activity}; length={len(ids)}; structure={first.profile.playlist_structure}")
        print(f"ids={ids}")
        print(f"energies={[round(value, 2) for value in energies]}")
        print(f"catalog_only={set(ids) <= catalog_ids}")
        print(f"unique_ids={len(ids) == len(set(ids))}")
        print(f"deterministic={ids == [item.song['id'] for item in second.songs]}")


if __name__ == "__main__":
    main()
