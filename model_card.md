# Model Card: VibeMatch AI 2.0

## System purpose

VibeMatch AI is a classroom demonstration of a hybrid applied AI system. It converts a natural-language situation into an explainable playlist drawn from a local catalog. It is intended for learning about structured model output, deterministic ranking, validation, fallback design, and responsible AI—not for production music discovery or decisions about artists.

## System components

VibeMatch is a system rather than one trained model:

1. Gemini optionally infers a structured `PreferenceProfile` from the request.
2. Python validates categories and numeric ranges.
3. A deterministic weighted formula scores verified local songs.
4. A deterministic generator applies activity-specific energy order and artist diversity.
5. Gemini optionally explains selected records.
6. Python validates explanation IDs or displays deterministic reasons.

Gemini cannot select a song, create an ID, calculate a compatibility score, alter metadata, or modify playlist order.

## Data used

The catalog contains 50 fictional songs in `data/songs.csv`, including the 17 records from Project 3. Each record has an ID, fictional title and artist, genre, mood, energy, tempo, valence, danceability, and acousticness. All metadata is synthetic classroom data. It was authored to provide varied test cases and was not derived from audio analysis or verified real-world sources.

### Optional real-music catalog

`data/real_songs.csv` holds 95 real tracks built once by `scripts/build_real_catalog.py` on November 16, 2026 from a single seed track. When the user selects *Real music*, this catalog replaces the fictional one; the scoring and selection code is identical.

- **Source of attributes:** energy, tempo, valence, danceability, and acousticness come from ReccoBeats; cover art, album, release date, and genre come from Deezer; Spotify supplies track IDs, links, and the embedded player. All are unofficial third-party services, their terms of use were not independently verified, and the values were not checked against the recordings.
- **Derived mood:** no source provides mood. It is computed locally from valence, energy, and acousticness using the rules in `derive_mood()`. These are the author's assumptions, not listener-validated labels.
- **Selection into the catalog:** tracks are whatever ReccoBeats recommended around the seed, then filtered (explicit, live/remix/holiday titles, duplicates, per-artist cap) and spread across the energy range. This is not a representative sample of music; it skews toward the seed's neighborhood, mixes languages, and under-represents calm, positive-sounding tracks.
- **Real artist and track names are used.** Notes shown for songs are assembled from catalog fields (album, release year, genre) with a visible source. Gemini does not write them, and a song without that data shows no note.

No personal data, listening history, protected traits, lyrics, likes, skips, location, or account data is used. The fictional catalog contains no real artist information.

## Algorithm

The Project 4 compatibility score is:

```text
0.35 × energy similarity
+ 0.25 × exact mood match
+ 0.25 × preferred-genre match
+ 0.15 × tempo similarity
```

Each component is in `[0, 1]`; the final value is shown on a 0–100 scale. Numerical similarity uses distance from the requested target. Dynamic playlists also use an ordering penalty based on distance from each position's energy target. That penalty changes selection order but is not presented as part of the compatibility score.

## Gemini's role

Gemini performs two bounded language tasks:

- infer activity, length (1–50, enforced by the schema and again by Python), up to three catalog-supported genres, mood, energy, tempo, and playlist structure;
- write one concise explanation for each already-selected verified record.

Both calls request schema-constrained JSON. Python still validates values after generation. When Gemini or the SDK is absent, authentication/network fails, JSON is malformed, categories are unsupported, or explanation IDs are incomplete, the application uses local interpretation and/or deterministic reasons.

## Intended use

- CodePath AI110 coursework and demonstrations;
- exploration of hybrid generative/deterministic systems;
- discussion of recommendation weights, explainability, and guardrails;
- testing activity-based energy sequencing on synthetic data.

## Non-intended use and potential misuse

- a real streaming or medical/wellness recommendation service;
- evaluation, ranking, compensation, or career decisions about artists;
- claims about real songs, audio properties, popularity, or listener behavior;
- inference of sensitive traits or psychological state;
- safety-critical activity guidance.

Fluent explanations could be mistaken for evidence about real music. The CLI and documentation label the catalog synthetic, restrict explanations to supplied metadata, and keep human choice as the final step.

## Limitations and biases

- The catalog is small, synthetic, and uneven across genre/mood combinations.
- Exact labels miss relationships such as `pop` and `indie pop`.
- Activity defaults encode author assumptions (for example, workout means high energy).
- A user may want conflicting properties that no record satisfies.
- The offline keyword interpreter can miss nuance, negation, and indirect requests.
- Gemini may infer preferences the user did not state. The full profile is displayed for review.
- Compatibility weights are hand-designed rather than learned or validated with listeners.
- Artist diversity is a simple repeat limit, not a fairness guarantee.
- No live Gemini quality, latency, availability, or cost benchmark is included in the checked-in evaluation.

## Safeguards and human oversight

- Environment-only API credentials and a committed `.env.example` without secrets. The real `.env` and `.streamlit/secrets.toml` are git-ignored.
- Preview links expire, so they are fetched fresh and never stored or exported. Cover, link, and note fields come only from validated catalog rows.
- JSON schema plus semantic validation for external outputs.
- Clamped/defaulted request values and validated CSV ranges.
- Catalog-only ID selection with duplicate prevention.
- Exact selected-ID coverage required from explanations.
- Explicit warnings when fallbacks or truncation occur.
- Deterministic tie-breaking and repeatable offline evaluation.
- The user sees the inferred profile, scores, reasons, and AI/fallback status and can reject the playlist.
- The Streamlit dashboard labels interpretation and explanation sources separately, never presents local fallback output as AI-generated, and re-verifies catalog IDs, uniqueness, length, and determinism after every generation.

## Reliability evaluation

On November 16, 2026, `python -m pytest -q` produced `91 passed` (18 core tests, 25 Streamlit dashboard tests, 6 media-helper tests, 27 catalog-builder tests, 8 playlist-length tests, and 7 design-system tests; all network access mocked). One original assertion was deliberately updated when the playlist cap rose from 20 to 50. Tests cover:

- CSV type conversion, ranges, and invalid energy/tempo;
- legacy score calculations and catalog non-mutation;
- normalized compatibility and closeness behavior;
- descending consistent-playlist ranking;
- requested lengths up to 50, with visible warnings when a request is trimmed by the cap or by catalog size, and a clear error for an empty catalog;
- catalog-only unique IDs;
- deterministic identical-input behavior;
- warm-up/peak/cooldown energy shape;
- malformed Gemini output and API exceptions;
- conflicting genre, mood, energy, and tempo preferences;
- dashboard behavior with `streamlit.testing.v1.AppTest`: loading without a key, rejecting empty requests, offline generation, accurate Gemini/fallback labels (offline, mocked failure, mocked success), catalog-only results, and a CSV export with no secrets.

The offline evaluation ran workout, studying, and relaxation scenarios twice. All results contained only valid unique catalog IDs and repeated deterministically. The workout energies were `[0.67, 0.91, 0.93, 0.96, 0.98, 0.93, 0.92, 0.90, 0.72, 0.48]`, demonstrating a moderate opening, peak section, and cooldown. Full evidence is in `evaluation_results.txt`.

No live Gemini success result is asserted without a user-provided key. Fake-client tests exercise the error and malformed-response boundary, but they do not measure live model quality.

## AI suggestion accepted

> **Student placeholder:** Describe one AI suggestion you personally reviewed and accepted, why it fit the assignment, and how you verified it. A possible topic to evaluate in your own words is separating Gemini interpretation from deterministic selection.

## AI suggestion rejected or modified

> **Student placeholder:** Describe one suggestion you rejected or changed and why. A possible topic to evaluate in your own words is preserving the displayed compatibility score while adding a separate energy-ordering utility after the first workout run revealed a sequencing problem.

## Responsible AI reflection

> **Student placeholder:** Write your own reflection on what responsibility remains with the developer when model output is schema-valid but semantically poor, and why showing fallbacks and inferred preferences matters. Do not submit this placeholder unchanged.

## Future improvements

- Collect consented listener feedback and evaluate ranking quality rather than relying only on intuition.
- Expand and rebalance the synthetic catalog or use licensed, documented audio features.
- Support soft similarity between related moods and genres.
- Add user-editable inferred profiles before generation.
- Evaluate live Gemini accuracy, latency, cost, and explanation faithfulness across a fixed prompt set.
- Add accessibility preferences and multilingual request evaluation.
