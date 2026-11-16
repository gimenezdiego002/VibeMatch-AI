"""Reproducible evaluation for VibeMatch AI.

    python evaluation.py            # offline fallback + mocked Gemini; no network
    python evaluation.py --live     # also one clearly labeled live Gemini pass (needs a key)
    python evaluation.py --write    # also save the report to evaluation_results.txt

Part 1 keeps the original fictional-catalog scenarios. Part 2 checks the real catalog's language
and release-year constraints. Part 3 checks the catalog chatbot: answer correctness, citation
validity, and whether off-catalog questions are refused.
"""

from __future__ import annotations

import argparse
import time
from datetime import date
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from src.app import create_playlist
from src.chatbot import REFUSAL, answer_question
from src.gemini_service import GeminiService
from src.playlist import song_year
from src.recommender import load_songs


ROOT = Path(__file__).resolve().parent
FICTIONAL = ROOT / "data" / "songs.csv"
REAL = ROOT / "data" / "real_songs.csv"

SCENARIOS = {
    "workout": "Make a 10-song high-energy gym workout with a warm-up and cooldown",
    "study": "Give me 10 relaxing low-energy songs for studying with a steady feel",
    "relax": "Create 8 peaceful songs for relaxing after work",
}
CONSTRAINT_CASES = [
    # (label, request, expected language, expected min year, hard language?, hard years?)
    ("spanish workout", "I am working out I want 15 songs all in spanish kinda recent", "spanish", True),
    ("recent party", "only new songs for a party, 10 songs", None, False),
    ("soft spanish", "preferably spanish party mix 12", "spanish", False),
    ("shortage", "20 songs all in portuguese for running", "portuguese", True),
]
CHAT_QUESTIONS = [
    # (question, expected substring or None, should_refuse)
    ("What album is #1 from?", "album", False),
    ("Which songs are over 140 BPM?", "BPM", False),
    ("Is there anything quieter than #3?", "quieter than #3", False),
    ("How many Spanish songs do you have in the catalog?", "14", False),
    ("When was Blinding Lights released?", "released in 2020", False),
    ("What are the lyrics of #2?", None, True),
    ("Did Avril Lavigne win a Grammy?", None, True),
    ("Tell me about song #40", "There is no song #40", False),
]
LIVE_SPACING_SECONDS = 7
CHAT_PLAYLIST_REQUEST ="Make a 10-song high-energy gym workout with a warm-up and cooldown"


class _Resp:
    def __init__(self, parsed: Any):
        self.parsed, self.text = parsed, json.dumps(parsed)


class MockGeminiClient:
    """A fixed, deterministic stand-in for Gemini used to exercise the AI code paths offline."""

    def __init__(self) -> None:
        self.models = self

    def generate_content(self, *, model: str, contents: str, config: dict) -> _Resp:
        properties = config["response_json_schema"].get("properties", {})
        if "playlist_length" in properties:  # interpretation of the Spanish workout request
            return _Resp({
                "activity": "workout", "playlist_length": 15, "preferred_genres": ["latin", "reggaeton", "pop"],
                "target_mood": "energetic", "target_energy": 0.85, "target_tempo_bpm": 130,
                "playlist_structure": "consistent", "steady_requested": False, "language": "spanish",
                "language_strict": True, "min_release_year": date.today().year - 5, "max_release_year": 0,
                "recency_strict": False, "priorities": {"energy": 0.8, "language": 0.9},
            })
        if "explanations" in properties:  # explanations: valid, covering every selected id
            records = json.loads(contents.split("Selected records: ", 1)[1])
            return _Resp({"explanations": [{"id": r["id"], "explanation": f"{r['title']} keeps the energy up."} for r in records]})
        raise ValueError("unexpected request")


class MockChatGemini:
    """Chat stand-in. Usually answers well; two questions get deliberately bad answers
    (one vague, one citing a song that was not retrieved) to show the checks catching them."""

    def parse_question(self, question: str, **kwargs: Any) -> dict:
        from src.chatbot import parse_question_offline

        return parse_question_offline(question, genres=set(kwargs["genres"]), moods=set(kwargs["moods"]),
                                      languages=set(kwargs["languages"]))

    def answer_from_records(self, question: str, *, records: list, total_matches: int, reference: Any = None) -> dict:
        ids = [r["id"] for r in records]
        if "Blinding Lights" in question:  # adversarial: cites a song that was not retrieved
            return {"answer": f"{records[0]['title']} came out in {records[0]['release_year']}.", "cited_ids": [*ids, 999]}
        if "quieter" in question:  # adversarial: grounded but vague (no energy values given)
            return {"answer": "Several calmer songs: " + "; ".join(r["title"] for r in records) + ".", "cited_ids": ids}
        if "how many" in question.casefold():
            return {"answer": f"There are {total_matches} matching songs in the catalog.", "cited_ids": ids}
        details = "; ".join(
            f"{r['title']} (album {r['album']}, {r['release_year']}, {r['tempo_bpm']} BPM, energy {r['energy']:.2f}, "
            f"{r['genre']}, {r['mood']}, {r['language']})" for r in records
        )
        return {"answer": f"Here is what the catalog has: {details}.", "cited_ids": ids}


def constraint_report(result: Any, language: str | None, min_year: int | None, catalog_ids: set[int]) -> list[str]:
    ids = [item.song["id"] for item in result.songs]
    lines = [f"returned={len(ids)} of requested={result.profile.playlist_length}"]
    if language:
        share = sum(1 for item in result.songs if item.song.get("language") == language) / max(1, len(ids))
        lines.append(f"{language}_share={share:.0%}")
    if min_year:
        share = sum(1 for item in result.songs if (song_year(item.song) or 0) >= min_year) / max(1, len(ids))
        lines.append(f"released_{min_year}_or_later={share:.0%}")
    lines.append(f"catalog_only={set(ids) <= catalog_ids}")
    lines.append(f"unique_ids={len(ids) == len(set(ids))}")
    return lines


def part_one() -> None:
    catalog_ids = {song["id"] for song in load_songs(str(FICTIONAL))}
    print("PART 1 - fictional catalog scenarios (offline fallback)")
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


def part_two(live: GeminiService | None) -> None:
    songs = load_songs(str(REAL))
    catalog_ids = {song["id"] for song in songs}
    print("\n\nPART 2 - real catalog: language and release-year constraints")
    print(f"Catalog records: {len(songs)}; spanish={sum(s['language'] == 'spanish' for s in songs)}; "
          f"portuguese={sum(s['language'] == 'portuguese' for s in songs)} (language labels are estimates)")
    modes = [("offline fallback", dict(offline=True)),
             ("mocked Gemini", dict(offline=False, gemini=GeminiService(client=MockGeminiClient())))]
    for label, request, language, hard in CONSTRAINT_CASES:
        print(f"\n{label.upper()}: {request}")
        for mode, kwargs in modes:
            if mode == "mocked Gemini" and label != "spanish workout":
                continue  # the mock only knows this one request
            first = create_playlist(request, catalog_path=REAL, **kwargs)
            second = create_playlist(request, catalog_path=REAL, **kwargs)
            p = first.profile
            print(f"  [{mode}] interpretation={'Gemini' if first.used_gemini_interpretation else 'local'}; "
                  f"language={p.language}{' (hard)' if p.language_strict else ' (soft)' if p.language else ''}; "
                  f"years={p.min_release_year}-{p.max_release_year}{' (hard)' if p.recency_strict else ''}; "
                  f"structure={p.playlist_structure}")
            print("    " + "; ".join(constraint_report(first, language, p.min_release_year, catalog_ids)))
            print(f"    deterministic={[i.song['id'] for i in first.songs] == [i.song['id'] for i in second.songs]}")
            for warning in first.warnings:
                if "fit this catalog" in warning or "were returned" in warning:
                    print(f"    warning: {warning}")
            expected_hard = bool(language) and hard
            print(f"    constraint_read_correctly={p.language == language and p.language_strict == expected_hard}")
    if live is not None:
        print("\n  [LIVE Gemini] spanish workout")
        try:
            result = create_playlist(CONSTRAINT_CASES[0][1], catalog_path=REAL, gemini=live)
            p = result.profile
            print(f"    interpretation={'Gemini' if result.used_gemini_interpretation else 'local (Gemini failed)'}; "
                  f"language={p.language} hard={p.language_strict}; years>={p.min_release_year}")
            print("    " + "; ".join(constraint_report(result, "spanish", p.min_release_year, catalog_ids)))
            for warning in result.warnings:
                print(f"    note: {warning}")
        except Exception as exc:  # report, never crash the evaluation
            print(f"    live run failed: {type(exc).__name__}")


def part_three(live: GeminiService | None) -> None:
    songs = load_songs(str(REAL))
    playlist = [item.song for item in create_playlist(CHAT_PLAYLIST_REQUEST, catalog_path=REAL, offline=True).songs]
    print("\n\nPART 3 - catalog chatbot (retrieval first; AI answers verified)")
    print(f"Playlist for '#N' questions: {CHAT_PLAYLIST_REQUEST}")
    modes = [("offline", dict()), ("mocked Gemini", dict(gemini=MockChatGemini(), use_gemini=True))]
    if live is not None:
        modes.append(("LIVE Gemini", dict(gemini=live, use_gemini=True)))
    for mode, kwargs in modes:
        correct = refusals_ok = ai_shown = ai_rejected = 0
        cite_ok = True
        print(f"\n  [{mode}]")
        for question, expected, should_refuse in CHAT_QUESTIONS:
            if mode.startswith("LIVE"):
                time.sleep(LIVE_SPACING_SECONDS)  # stay under the free tier's per-minute request limit
            answer = answer_question(question, catalog=songs, playlist_songs=playlist, **kwargs)
            refused = answer.text == REFUSAL
            refusals_ok += refused == should_refuse
            ok = refused if should_refuse else (expected is None or expected.lower() in answer.text.lower())
            correct += ok
            retrieved = {r["id"] for r in answer.records}
            cite_ok &= set(answer.cited_ids) <= retrieved | {0}
            ai_shown += answer.source == "AI answer"
            ai_rejected += any("discarded" in w for w in answer.warnings)
            print(f"    {'PASS' if ok else 'FAIL'} [{answer.source}] {question} -> {answer.text.splitlines()[0][:90]}")
            for warning in answer.warnings:
                print(f"         note: {warning[:110]}")
        print(f"    answer_correct={correct}/{len(CHAT_QUESTIONS)}; refusal_correct={refusals_ok}/{len(CHAT_QUESTIONS)}; "
              f"shown_citations_valid={cite_ok}; ai_answers_shown={ai_shown}; ai_answers_rejected={ai_rejected}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="also run one labeled live Gemini pass (uses quota)")
    parser.add_argument("--write", action="store_true", help="save the report to evaluation_results.txt")
    args = parser.parse_args()
    live = None
    if args.live:
        from src.env import load_local_env

        load_local_env()
        service = GeminiService()
        live = service if service.available else None
        if live is None:
            print("--live requested but no GEMINI_API_KEY is configured; skipping live checks.")

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print(f"VibeMatch AI evaluation - run date {date.today().isoformat()}")
        print("Command: python evaluation.py" + (" --live" if args.live else ""))
        print("Modes: offline fallback and mocked Gemini need no network" + ("; LIVE results are labeled" if live else "") + ".\n")
        part_one()
        if REAL.is_file():
            part_two(live)
            part_three(live)
        else:
            print("\n(data/real_songs.csv not built; parts 2 and 3 skipped)")
    report = buffer.getvalue()
    print(report, end="")
    if args.write:
        (ROOT / "evaluation_results.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
