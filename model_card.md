# Model Card: VibeMatch AI 2.0

## System purpose

VibeMatch AI is a classroom demonstration of a hybrid applied AI system. It turns a natural-language situation into an explainable playlist drawn from a local catalog, and answers questions about those songs. It is meant for learning about structured model output, retrieval-augmented answers, deterministic ranking, validation, fallback design, and responsible AI. It is not a production music service and should not be used for decisions about artists.

## System components

VibeMatch is a system around one external model (Google Gemini, `gemini-2.5-flash` by default), not a trained model of its own:

1. Gemini optionally turns the request into a structured profile with language and release-year constraints and suggested priorities. A local keyword interpreter does the same offline.
2. Python validates the profile against what the active catalog contains, then applies hard filters.
3. A deterministic formula scores the remaining songs; a sequencer applies the energy arc and artist diversity.
4. Gemini optionally explains the selected songs; Python requires exact ID coverage or shows deterministic reasons.
5. A catalog chatbot retrieves matching songs with Python first. Gemini may phrase an answer only from those records, and Python verifies citations, numbers, coverage, and the values asked for before showing it.
6. At catalog-build time only, Gemini estimates each real track's language. Python validates the labels, people can override them, and Gemini is never asked for this while a playlist is being made.

Gemini cannot select a song, create an ID, compute a score, alter metadata, change the order, or have its answer shown without passing Python's checks.

## Data used

- **Fictional catalog (default):** `data/songs.csv`, 50 synthetic songs including the 17 from Module 3. All titles, artists, and attributes were authored for testing. It has no language or release-date data, so language and year requests on it are dropped with a visible warning.
- **Real-music catalog (optional):** `data/real_songs.csv`, 95 real tracks built once on November 16, 2026 by `scripts/build_real_catalog.py` from a single seed track.
  - **Audio features** (energy, tempo, valence, danceability, acousticness): ReccoBeats.
  - **Album, cover, release date, genre:** Deezer.
  - **Links and embedded player:** Spotify.
  - None of these values was checked against the recordings, and the services' terms were not independently verified.
  - **Mood is derived** by documented rules from valence, energy, and acousticness. These are the author's assumptions, not listener labels.
  - **Language is an AI estimate.** Gemini labeled each track from its title, artist, album, and genre. Each row records `language_source` (`gemini estimate`, `rule estimate`, or `manual override`) and a confidence score, and corrections in `data/language_overrides.csv` always win. One label has already been corrected: a track with a featured vocalist was marked instrumental. Current estimates: 49 English, 14 Spanish, 9 Portuguese, and 23 in other languages or instrumental.

No personal data, listening history, protected traits, lyrics, likes, skips, location, or account data is collected or used.

## Algorithm

The default score out of 100 is:

```text
0.35 × energy fit + 0.25 × mood fit + 0.25 × genre fit + 0.15 × tempo fit
```

- Each fit is between 0 and 1. Energy and tempo fits measure closeness to a target, not "higher is better".
- Mood and genre give documented partial credit to close labels (for example latin and reggaeton 0.8, chill and peaceful 0.8); unrelated labels get 0.
- Soft language and release-year preferences join as their own weighted features. Hard ones ("all in Spanish", "only new songs") filter the catalog before scoring.
- Gemini may suggest priorities per feature; Python clamps each to 0.05–0.60 and normalizes, and the contributions always sum to the score.
- Dynamic playlists add an ordering penalty based on each position's energy target. It changes the order but is never shown as part of the compatibility score.
- Ties break on song ID, so the same validated profile always gives the same playlist.

## Gemini's role

| Task | When | Output allowed | Checked by Python |
| --- | --- | --- | --- |
| Interpret request | each playlist | profile fields, constraints, priorities (schema-constrained) | clamping, catalog vocabulary, workout structure guard |
| Explain picks | each playlist | one sentence per selected ID | exactly the selected IDs |
| Parse a chat question | each question | a structured query | known fields, filters, limits only |
| Answer a chat question | each question | short answer and cited IDs, from retrieved records only | cited IDs retrieved, numbers match, every match covered, asked values present |
| Estimate track language | catalog build only | language from a fixed list, plus confidence | enum validation, rule fallback, manual overrides |

When the key is missing, the SDK is absent, the network fails, the daily free-tier quota (20 requests) runs out, or the output is malformed, each task falls back to its deterministic equivalent and the UI labels it.

## Intended use

- CodePath AI110 coursework, demonstrations, and portfolio review;
- exploring how to combine generative models with deterministic, verifiable code;
- discussing recommendation weights, explainability, retrieval grounding, and guardrails.

## Limitations and biases

- **Small, skewed catalogs:**
  - The real catalog reflects what one recommendation service returned around a single seed. It mixes languages and genres unevenly, and has only 4 Spanish tracks at workout energy, so Spanish workouts come out short and low-energy.
  - The fictional catalog has no language or year data at all.
- **Estimated and derived labels:**
  - Language labels are AI estimates. Two models disagreed on at least one track in testing, and titles can mislead (an English title on a Spanish song).
  - Moods come from the author's rules.
  - Either can be wrong in ways that systematically favor some songs.
- **Author assumptions:** activity defaults (a workout means high energy, a party builds up), similarity tables, and weights are hand-designed, not learned from listeners.
- **English-centric parsing:** the offline interpreter recognizes a short, mostly English list of words and misses negation ("not Spanish") and indirect phrasing.
- **Model inference:** Gemini may infer preferences the user did not state. For example, it chose "recent" to mean 2019 or later, while the offline rule uses 2021. The full profile is shown so a person can see it.
- **Fluent answers that say too little:** chatbot answers can be accurate but unhelpful. The asked-value check catches the obvious cases, not every one.
- **Limited live evaluation:** live Gemini quality, latency, and cost were measured only on a handful of calls because of the free-tier quota.
- **Simple diversity rule:** artist diversity is a repeat limit, not a fairness guarantee.

## Misuse and prevention

| Possible misuse | Prevention in the system |
| --- | --- |
| Treating AI text as facts about real artists or songs (trivia, biographies, lyrics, awards) | The chatbot answers only from catalog fields and refuses everything else. Notes on cards are assembled from catalog data, never written by Gemini. Every answer is labeled with its source. |
| Getting the model to invent songs, numbers, or citations | Selection is Python-only. Chat answers that cite unretrieved songs, quote unmatched numbers, or skip matches are discarded and replaced with a deterministic lookup. |
| Prompt injection through the request or a question | User text only becomes a schema-constrained query or profile that Python validates; extra fields are dropped. Catalog records are sent as JSON data, and the output is verified regardless of what the model says. |
| Leaking the API key | The key lives in git-ignored `.env` or secrets and is never shown, exported, or logged. Logs redact anything key-shaped, and a test enforces it. |
| Spamming the model or burning quota | Questions are length-limited (300 characters) and rate-limited (6 per minute). Gemini is called only on form submit, and a failed interpretation skips the explanation call. |
| Using rankings to judge artists, or for health or safety advice | Documented as non-intended use. Scores describe fit to a request, not quality. |

**Non-intended uses:** a real streaming, medical, wellness, or safety-critical service; decisions about artists; claims about popularity or listener behavior; inferring sensitive traits.

## Safeguards and human oversight

- Schema plus semantic validation on every model output; catalog-only selection; exact ID coverage for explanations; citation, number, coverage, and asked-value checks for chat answers.
- Hard constraints never relax silently. Shortages, caps (50 songs), and dropped constraints show plain-language warnings with counts.
- Deterministic tie-breaking and repeatable evaluation; structured logs of each step, with key redaction.
- People stay in charge:
  - the dashboard shows the inferred profile, weights, warnings, and AI or fallback labels;
  - System checks re-verify IDs, metadata, uniqueness, length, and determinism after every playlist;
  - a person reviews the built catalog and can override any language label;
  - the listener decides what to play.

## Reliability evaluation

On November 16, 2026, `python -m pytest -q` produced **180 passed**, with all network access mocked. One original assertion was deliberately updated when the playlist cap rose from 20 to 50, and one test double was updated to recognize the extended schema. The tests cover:

- **Validation:** CSV validation; profile and constraint validation and clamping; offline parsing of language, hedges, numbers, and decades; the Gemini schema built from the catalog.
- **Scoring:** similarity tables, bounded and normalized weights, contributions that sum to the score, hard filters with shortage warnings, the zero-song case, and determinism.
- **Chatbot:** query validation, retrieval, comparisons, refusals, and rejection of invented IDs, wrong numbers, incomplete answers, and off-topic answers.
- **Language labeling:** labeling, fallback, and overrides.
- **Logging and the dashboard:** logging and key redaction, and AppTest checks of the dashboard and the chat launcher.

`python evaluation.py` (results in `evaluation_results.txt`):

- **Constraints:** read correctly in 5 of 5 offline runs and 1 of 1 mocked-Gemini runs. Hard-language playlists were 100% in the requested language, and shortages were reported with counts (14 of 15 Spanish, 9 of 20 Portuguese).
- **Repeatability:** every playlist was deterministic, catalog-only, and had unique IDs.
- **Chatbot:** 8 of 8 answers correct and both off-catalog questions refused, offline and with mocked Gemini. In the mocked run, both deliberately bad AI answers (one vague, one citing a nonexistent song) were rejected.
- **Live Gemini, one labeled pass:**
  - interpreted the Spanish request as a hard constraint and produced a 100%-Spanish playlist;
  - gave one fully verified AI answer;
  - hit the daily free-tier quota on the remaining calls, all of which fell back to labeled deterministic output, still 8 of 8 correct.

Problems found during testing and fixed: a silent 20-song cap; dropped language requests; a hedge-word bug where "ish" inside "spanish" softened every language request; a score of 100.01 caused by rounding; a crash on an empty catalog; and chatbot answers that were grounded but incomplete or off-topic.

## What surprised me while testing

> **Student placeholder:** In your own words, describe what surprised you while testing the system's reliability. You can draw on the factual findings above (for example the grounded-but-vague AI answers, the hidden 20-song cap, the hedge-word bug, the free-tier quota, or how few Spanish workout tracks the catalog had), but the reflection must be yours.

## Collaboration with AI

> **Student placeholder:** Describe, in your own words, how you collaborated with AI tools during this project: what you asked for, how you checked the results, and where you made the final decisions.

### AI suggestion accepted

> **Student placeholder:** Describe one AI suggestion you personally reviewed and accepted, why it fit the assignment, and how you verified it. A possible topic to evaluate in your own words is separating Gemini interpretation from deterministic selection.

### AI suggestion rejected or modified

> **Student placeholder:** Describe one suggestion you rejected or changed and why. A possible topic to evaluate in your own words is preserving the displayed compatibility score while adding a separate energy-ordering utility after the first workout run revealed a sequencing problem.

## Responsible AI reflection

> **Student placeholder:** Write your own reflection on what responsibility remains with the developer when model output is schema-valid but semantically poor, and why showing fallbacks and inferred preferences matters. Do not submit this placeholder unchanged.

## Future improvements

- Rebuild the real catalog with Spanish and other non-English seed tracks, and verify language labels with a second source.
- Collect consented listener feedback to evaluate ranking quality and tune weights, instead of relying on intuition.
- Evaluate live Gemini interpretation and chat accuracy, latency, and cost on a fixed prompt set once a higher quota is available.
- Handle negation ("no English", "not too fast") in both interpreters.
- Let users edit the inferred profile before generating.
- Add accessibility preferences and multilingual request evaluation.
