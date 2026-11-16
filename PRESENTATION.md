# 5–7 Minute VibeMatch AI Demo

## 0:00–0:45 — Problem and goal

- Project 3 ranked songs from a manually created preference profile.
- Project 4 accepts a natural-language situation and combines Gemini interpretation with deterministic Python selection.
- Key guardrail: Gemini never chooses or invents songs.

## 0:45–1:45 — Architecture

Open `diagrams/architecture.mmd`. Trace the request through Gemini, validation, local scoring, sequencing, explanation, and fallback paths. Point out the two validation gates and the local CSV as the only source of songs.

## 1:45–3:30 — Live CLI demo

Use a configured Gemini key if available:

```powershell
python -m src.main -r "Make a 10-song high-energy gym workout with a warm-up and cooldown"
```

Show the detected activity, validated profile, compatibility score, position energy target, and explanation. Note the moderate first song, high-energy middle, and lower-energy ending.

Then demonstrate guaranteed offline behavior:

```powershell
python -m src.main --offline -r "Give me 10 relaxing low-energy songs for studying with a steady feel"
```

## 3:30–4:30 — Explainability and guardrails

- Show `WEIGHTS` and `score_song_for_profile()` in `src/playlist.py`.
- Show that score components are normalized and deterministic.
- Show `GeminiService.explain()`: it receives only selected local records, and returned IDs must exactly match them.
- Mention honest truncation when the requested length exceeds the catalog.

## 4:30–5:30 — Reliability evidence

```powershell
python -m pytest -q
python evaluation.py
```

Discuss CSV validation, malformed Gemini output, API failure fallback, no fabricated IDs, deterministic reruns, conflicting preferences, and activity-specific energy curves.

## 5:30–6:30 — Limits and future work

- All 50 songs and their attributes are synthetic classroom data.
- Exact genre/mood labels are simplistic.
- No listening history or feedback loop is used.
- Future work: larger balanced catalog, user feedback, soft label similarity, and live Gemini evaluation with recorded latency/cost.

## Backup plan

If network access or the Gemini key fails, use `--offline`. The fallback is an intentional reliability feature, not a separate demo build.
