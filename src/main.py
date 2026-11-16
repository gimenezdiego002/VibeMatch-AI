"""Command-line interface for VibeMatch AI."""

from __future__ import annotations

import argparse
import json
import logging

from src.app import create_playlist
from src.chatbot import answer_question
from src.env import load_local_env
from src.gemini_service import GeminiService
from src.recommender import load_songs
from src.logging_setup import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an explainable, context-aware playlist.")
    parser.add_argument("--request", "-r", help="Natural-language playlist request")
    parser.add_argument("--offline", action="store_true", help="Skip Gemini and use the deterministic local interpreter")
    parser.add_argument("--catalog", default="data/songs.csv", help="Path to a compatible song catalog CSV")
    parser.add_argument("--verbose", "-v", action="store_true", help="Log each pipeline step to stderr")
    parser.add_argument("--ask", action="append", metavar="QUESTION",
                        help="After the playlist, ask the catalog chatbot a question (repeatable)")
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

    if args.ask:
        catalog = load_songs(str(args.catalog))
        use_gemini = not args.offline
        service = GeminiService() if use_gemini else None
        print("\nQuestions about this playlist and catalog:")
        for question in args.ask:
            answer = answer_question(
                question,
                catalog=catalog,
                playlist_songs=[item.song for item in result.songs],
                gemini=service if service and service.available else None,
                use_gemini=bool(service and service.available),
            )
            print(f"\nQ: {question}\n[{answer.source}] {answer.text}")
            cited = [r for r in answer.records if r["id"] in set(answer.cited_ids)]
            if cited:
                print("Sources: " + "; ".join(f"ID {r['id']} {r['title']}" for r in cited[:10]))
            for warning in answer.warnings:
                print(f"Note: {warning}")


if __name__ == "__main__":
    main()
