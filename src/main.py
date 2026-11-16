"""Command-line interface for VibeMatch AI."""

from __future__ import annotations

import argparse
import json
import logging

from src.app import create_playlist
from src.env import load_local_env
from src.logging_setup import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an explainable, context-aware playlist.")
    parser.add_argument("--request", "-r", help="Natural-language playlist request")
    parser.add_argument("--offline", action="store_true", help="Skip Gemini and use the deterministic local interpreter")
    parser.add_argument("--catalog", default="data/songs.csv", help="Path to a compatible song catalog CSV")
    parser.add_argument("--verbose", "-v", action="store_true", help="Log each pipeline step to stderr")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(logging.INFO if args.verbose else logging.WARNING)
    load_local_env()
    request = args.request or input("Describe the playlist you want: ").strip()
    try:
        result = create_playlist(request, catalog_path=args.catalog, offline=args.offline)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    print(f"\n{result.name}")
    print("=" * len(result.name))
    print(f"Request: {request}")
    print(f"Detected activity: {result.profile.activity}")
    print("Inferred profile:")
    print(json.dumps(result.profile.to_dict(), indent=2))
    print(
        "AI status: "
        f"interpretation={'Gemini' if result.used_gemini_interpretation else 'local fallback'}, "
        f"explanations={'Gemini' if result.used_gemini_explanations else 'deterministic'}"
    )
    for warning in result.warnings:
        print(f"Warning: {warning}")

    print(f"\nOrdered playlist ({len(result.songs)} songs):\n")
    for index, item in enumerate(result.songs, start=1):
        song = item.song
        print(f"{index:>2}. {song['title']} — {song['artist']} [{song['genre']}, {song['mood']}]")
        print(
            f"    Compatibility: {item.score:.2f}/100 | energy={float(song['energy']):.2f} "
            f"(position target={item.position_target_energy:.2f}) | tempo={song['tempo_bpm']} BPM"
        )
        print(f"    Why: {item.explanation}")


if __name__ == "__main__":
    main()
