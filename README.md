# VibeMatch AI: Context-Aware Playlist Generator

VibeMatch AI is a CodePath AI110 Applied AI System. A user describes a situation in natural language, such as studying, running, or working out. Gemini converts that request into a validated preference profile; deterministic Python code then scores and orders songs from a local catalog. Gemini may explain the final selections, but it cannot add songs, alter metadata, calculate scores, or change rankings.

The application remains fully usable without an API key. If Gemini is disabled, unavailable, or returns invalid output, a local keyword interpreter and deterministic explanations take over and the CLI or dashboard shows a warning.

VibeMatch can be used from the command line (`python -m src.main`) or from a Streamlit dashboard (`streamlit run streamlit_app.py`). Both call the same `create_playlist()` pipeline.

## From Project 3 to Project 4

The original Music Recommender Simulation loaded fictional records from CSV, accepted a manually constructed profile, and ranked songs with genre, mood, energy, and acousticness bonuses. That baseline remains in `src/recommender.py` for comparison and backward-compatible tests.

Project 4 adds:

- natural-language request interpretation;
- strict validation of external model output;
- normalized energy, mood, genre, and tempo scoring;
- workout, running, study, driving, relaxing, party, and general defaults;
- activity-specific energy sequencing and artist diversity;
- optional Gemini explanations constrained to selected records;
- safe offline and API-failure fallbacks;
- a 50-record synthetic catalog and expanded reliability tests.

## Architecture

```text
Request -> Gemini interpreter -----> profile validator -> Python scorer
              | API/JSON failure           ^                 |
              v                            |                 v
         local fallback -------------------+          playlist sequencer
                                                               |
                               Gemini explainer <---------------+
                                  | invalid IDs/API failure     |
                                  v                             v
                         deterministic reasons ----> CLI / dashboard
```

The detailed Mermaid source is in [`diagrams/architecture.mmd`](diagrams/architecture.mmd).

### Responsibility boundary

| Component | Allowed responsibility |
| --- | --- |
| Gemini interpreter | Infer a structured preference profile from user language |
| Profile validator | Clamp numbers and reject/replace unsupported categories |
| Python scoring engine | Calculate every compatibility score and contribution |
| Playlist generator | Select only catalog IDs, order energy, limit artist repeats |
| Gemini explainer | Describe already-selected verified metadata concisely |
| CLI | Display profile, ordered results, warnings, and AI/fallback status |
| Streamlit dashboard | Collect the request, call `create_playlist()`, display results, re-verify guardrails, export CSV. It does not score or select songs |

## Scoring and ordering

The Project 4 score is normalized from 0 to 100:

```text
0.35 × energy similarity
+ 0.25 × mood match
+ 0.25 × genre match
+ 0.15 × tempo similarity
```

Numerical features reward closeness to a target, not high values by default. Every component is between 0 and 1. A separate ordering utility applies only to dynamic structures so a workout can begin moderately, peak, and cool down without misrepresenting the compatibility score. Consistent study/relaxation playlists retain straightforward descending-score order.

## Dataset

`data/songs.csv` contains 50 fictional songs. The original 17 records were preserved and 33 fictional records were added for broader genre, mood, energy, and tempo coverage. All titles, artists, and attributes are synthetic classroom data—not verified claims about real music. Energy, valence, danceability, and acousticness must be within 0–1; tempo must be within 40–220 BPM; IDs must be unique. This is the default catalog and the offline fallback.

`data/real_songs.csv` is an optional second catalog of **real tracks**, described in [Real-music catalog](#real-music-catalog). It uses the same columns and the same validation, plus cover, album, release date, and link fields.

## Setup

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a Gemini API key in [Google AI Studio](https://aistudio.google.com/apikey), then expose it as an environment variable. Do not paste it into source code or commit it.

```powershell
$env:GEMINI_API_KEY="your-key-here"
# Optional; defaults to gemini-2.5-flash
$env:GEMINI_MODEL="gemini-2.5-flash"
```

You can also copy `.env.example` to `.env` in the project root. `.env` is git-ignored, and both the CLI and the dashboard load it at startup through `src/env.py` (needs `python-dotenv`, which is in `requirements.txt`). Variables you set in the shell or in Streamlit secrets take priority over `.env`.

For the Streamlit dashboard you can use the same environment variable, or Streamlit secrets:

```powershell
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
# then edit .streamlit\secrets.toml and set GEMINI_API_KEY (and optionally GEMINI_MODEL)
```

The dashboard checks `st.secrets` first and then `GEMINI_API_KEY` in the environment. `.streamlit/secrets.toml` is listed in `.gitignore`, so never commit it. The key is passed straight to `GeminiService(api_key=...)` and is never displayed, logged, or written to the CSV export.

The Gemini integration follows Google's structured-output guidance: [Gemini structured outputs](https://ai.google.dev/gemini-api/docs/structured-output) and the [Google Gen AI Python SDK](https://googleapis.github.io/python-genai/).

## Run

Interactive mode:

```powershell
python -m src.main
```

Scripted request with Gemini when a key is available:

```powershell
python -m src.main -r "Make a 10-song high-energy gym workout with a warm-up and cooldown"
```

Guaranteed local mode:

```powershell
python -m src.main --offline -r "Give me 10 relaxing low-energy songs for studying with a steady feel"
```

The terminal displays the detected activity, full inferred profile, AI/fallback status, warnings, ordered records, compatibility scores, position energy targets, and explanations.

## Streamlit dashboard

```powershell
streamlit run streamlit_app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

### What the dashboard shows

| Area | Contents |
| --- | --- |
| Header | App name, a one-line description of the Gemini/Python split, and a Gemini status badge: *available (API key configured)*, *SDK not installed*, or *no API key*. "Available" means a key is configured. Whether a call worked is only shown after you generate. |
| Request form | Natural-language text area, example requests (workout, studying, relaxing, driving, running, party), a **Use Gemini** toggle (disabled with an explanation when no key/SDK is present), and **Generate playlist**. It's an `st.form`, so Gemini is called only on submit, never on reruns. Empty requests are rejected. A spinner shows while generation runs. |
| AI and fallback status | Separate labels for **request interpretation** (Gemini / Local fallback) and **explanations** (Gemini / Deterministic reasons), a success, partial-fallback, or offline notice, and an expandable list of every validation, API, fallback, and catalog warning. Local fallback output is never described as AI-generated. |
| Inferred preference profile | Activity, playlist length, preferred genres, target mood, target energy, target tempo, and playlist structure, all after Python validation. |
| Catalog switch | A **Catalog** control in the form chooses *Fictional classroom songs* (default) or *Real music*. It is disabled, with an explanation, when `data/real_songs.csv` has not been built. |
| Playlist length | Up to **50** songs (`MAX_PLAYLIST_LENGTH` in `src/profiles.py`, shared with the Gemini schema). A longer request is trimmed to 50, and a smaller catalog returns what it has. Either way a **Playlist length adjusted** warning names the requested and returned counts, the *Songs returned* metric explains it, and System checks lists it. Nothing is trimmed silently. |
| Playlist tab | A responsive grid of ordered cards: three columns on a laptop, two at mid width, one on a narrow window. Each card has cover art (a generated tile for fictional songs), a title clamped to two lines, artist, a rank chip, the score as a gradient pill, genre and mood, an energy meter whose fill is the song's energy and whose white tick is the position's target, tempo, the explanation, a sourced "where it's from" line, a **Play** button, and an **Open in Spotify** link when a Spotify ID exists. Below the grid are the **Download playlist CSV** button and a *Table view* with rank, ID, title, artist, genre, mood, score, energy, target energy, tempo, and explanation. |
| Now playing (sidebar) | Stays visible while you scroll. Shows the selected track's cover, a sourced album/year note, an **Open in Spotify** button, a Spotify player, and a 30-second preview fetched fresh from Deezer (Apple as fallback). Missing previews or covers degrade to a short message or a placeholder. |
| Energy flow tab | Altair chart of actual song energy (violet line with a gradient fill) versus the position's target energy (dashed coral line). Warm-up playlists also get labeled *Warm-up*, *Peak*, and *Cooldown* bands. A generated `alt` description summarizes the shape. |
| Score details tab | Per-feature score points (energy, mood, genre, tempo) for every song. A song inspector shows earned versus maximum points, the deterministic reason, and the Gemini explanation when one exists. No model reasoning is requested or shown. |
| System checks tab | Re-verifies the finished result against the CSV: IDs exist, metadata matches, IDs are unique, returned versus requested length, and deterministic re-selection on the validated profile (Python only, no Gemini call). It also lists which steps used Gemini or the fallback, plus all warnings. |
| Catalog tab | Song, genre, and mood counts, energy and tempo ranges, and a table filterable by genre, mood, and energy, for whichever catalog is selected. The fictional catalog is labeled fictional/synthetic classroom data. The real catalog is labeled as third-party data with derived moods. |
| Data sources and attribution | An expander at the bottom names every source and states that their terms of use were not independently verified. |

The CSV export contains rank, verified catalog metadata, compatibility score, position target energy, per-feature points, deterministic reason, displayed explanation, explanation and interpretation sources, and activity. For the real catalog it also has album, release date, cover URL, ISRC, Spotify URL, and the sourced note with its source. It contains no credentials and no preview URLs (those expire).

### Offline behavior

With no key (or with **Use Gemini** off), the dashboard runs the same local keyword interpreter and deterministic reasons as `python -m src.main --offline`. It shows *Local fallback* / *Deterministic reasons* and the notice "No AI model was used for this playlist." If Gemini is on but a call fails (network, authentication, missing SDK, or invalid JSON), the result still renders, labeled as a partial or full fallback, with the failure type in the warnings.

### Implementation notes

- `streamlit_app.py` contains only UI code. It calls `create_playlist()` from `src/app.py`.
- `src/dashboard_data.py` turns `PlaylistResult` objects into tables, chart data, checks, and CSV. It never scores or reorders songs.
- `GeminiService` gained an optional keyword-only `api_key` argument and an `sdk_installed()` helper. The existing constructor, fake-client injection, and `GEMINI_API_KEY` behavior are unchanged.
- The catalog is loaded with `st.cache_data`. The latest result is kept in `st.session_state` so it survives reruns such as switching tabs, inspecting songs, or filtering the catalog.
- **Design system ("Midnight Vinyl"):** a deep plum background (`#0d0a1a`) with two soft glows, an electric-violet primary (`#9d8cff`), and a warm coral secondary (`#ff9a6b`). The two meet only as one violet-pink-coral gradient, used for the title, primary buttons, score pills, and energy meters. Space Grotesk is used for headings and Inter for body text, both loaded from Google Fonts with a system-font fallback. Tracking tightens as text grows and loosens on small uppercase labels. The hero and the "Now playing" panel are translucent blurred surfaces. Colors are defined in both `.streamlit/config.toml` (so native widgets match) and CSS variables.
- **Contrast:** every text and background pairing was checked against WCAG AA. The lowest is primary-colored text on a card surface at 6.4:1. Dark text sits on the gradient (6.9:1 or better), and body text on the background is 17.4:1.
- **Motion:** only `transform` and `opacity` are transitioned, at 120–180 ms with a strong ease-out. Cards lift 3px on hover (real pointers only), buttons scale to 0.97 when pressed, and nothing animates on page load or rerun. `prefers-reduced-motion` removes motion, `prefers-reduced-transparency` swaps glass for solid surfaces, and `prefers-contrast: more` strengthens borders and drops the gradient title.
- The only custom CSS is [`assets/vibematch.css`](assets/vibematch.css), loaded with `st.html(Path(...))`. It mostly targets documented `st.container(key=...)` classes; the few selectors that depend on Streamlit's internal markup (`.stApp`, images, primary-button test IDs, the grid's children) are labeled FRAGILE in the file. Delete the file and the one `st.html` line to return to pure native Streamlit. `unsafe_allow_html` is not used.
- `src/media.py` builds Spotify links and the sourced notes from catalog fields and fetches previews. It never scores, selects, or reorders songs. Notes are assembled directly from album, release year, and genre; Gemini does not write them, and a row with no album or year gets no note.
- The "Now playing" panel lives in the sidebar so Play on a card at the bottom of a long playlist doesn't require scrolling back up.

## Real-music catalog

The real catalog is built once by a script you run yourself, then committed as a normal CSV. The app never builds it at runtime, so playlists stay deterministic and work offline once the file exists.

```powershell
python scripts/build_real_catalog.py                 # about 100 tracks
python scripts/build_real_catalog.py --target 120 --allow-explicit
```

Sources, all keyless and unofficial third-party services that can change:

| Need | Source |
| --- | --- |
| Track discovery, Spotify IDs, ISRCs, energy, tempo, valence, danceability, acousticness | [ReccoBeats](https://reccobeats.com) |
| Cover art, album, release date, genre (by ISRC) | [Deezer public API](https://developers.deezer.com) |
| Playable player and links | [Spotify](https://open.spotify.com) embed and track URLs |
| 30-second previews (fetched fresh, never stored) | Deezer, with Apple as fallback |

How it works:

1. Seeds come from `data/seed_tracks.txt` (Spotify links, one per line) if you create it. Otherwise it starts from one verified Spotify ID.
2. ReccoBeats ignores energy and tempo filters on its recommendation endpoint (tested), so recommendations only resemble the seed. To cover calm music too, the builder repeatedly re-seeds from the discovered track nearest the emptiest 0.1-wide energy band.
3. Python filters drop explicit tracks by default, live/remix/karaoke/holiday titles, kids and soundtrack genres, missing or out-of-range features, duplicate ISRCs, and more than two tracks per artist. It then spreads the kept tracks across the energy range.
4. Genre is mapped from Deezer. **Mood is derived**, not provided by any source, using the documented rules in `derive_mood()`, into the same mood vocabulary the scorer already uses.

The checked-in file was built on November 16, 2026 from the single default seed: 95 tracks, energy 0.08–0.95, selected for variety rather than taste. It is eclectic, mixing languages and genres, and the builder printed a review list for you to prune. I removed five holiday and nursery-rhyme rows after the first run. Review it yourself and delete rows you don't want; IDs only need to stay unique. Adding your own seeds in `data/seed_tracks.txt` and re-running gives a catalog closer to your taste.

MusicBrainz enrichment was considered and **not implemented**, so the "where it's from" notes contain only album, release year, and genre. Creating a playlist inside a Spotify account requires user login and is out of scope; each song has a link and a player instead.

## Example results

These abbreviated outputs were produced with the deterministic offline mode on November 16, 2026.

### Workout

Input: `Make a 10-song high-energy gym workout with a warm-up and cooldown`

```text
activity=workout; structure=warmup_peak_cooldown; length=10
energies=[0.67, 0.91, 0.93, 0.96, 0.98, 0.93, 0.92, 0.90, 0.72, 0.48]
first/middle/last: Circuit Focus / Final Rep / Steady State
```

### Study

Input: `Give me 10 relaxing low-energy songs for studying with a steady feel`

```text
activity=studying; structure=consistent; length=10
top 3: Notebook Sunrise (98.78), Quiet Geometry (97.23), Focus Flow (96.33)
```

### Relaxation

Input: `Create 8 peaceful songs for relaxing after work`

```text
activity=relaxing; structure=consistent; length=8
top 3: Cloud Notes (98.12), Window Seat Dreams (98.08), Ocean Breathing (96.03)
```

## Reliability and guardrails

- Structured JSON schema constrains Gemini's interpretation and explanation output.
- Python performs semantic validation after schema-constrained generation.
- Unsupported activities, genres, moods, and structures are replaced or ignored with warnings.
- Energy and tempo inputs are clamped; non-finite values receive safe defaults.
- CSV types, ranges, and duplicate IDs are validated before scoring.
- Only records loaded from the catalog can enter a playlist.
- Explanation IDs must exactly cover selected IDs; otherwise deterministic reasons are shown.
- Network, authentication, SDK, and malformed-response failures use the local fallback.
- Playlists are capped at 50 songs; requests above the cap or above the catalog size return fewer songs with a visible warning, never silently.
- An empty catalog raises a clear error instead of crashing.
- Tie-breaking uses song ID, making identical inputs deterministic.

## Tests and evaluation

```powershell
python -m pytest -q
python evaluation.py
```

Latest verified local result (November 16, 2026):

```text
91 passed

tests/test_recommender.py       5 passed   (original Project 3 tests)
tests/test_vibematch.py        13 passed   (original Project 4 tests)
tests/test_dashboard.py        13 passed   (dashboard, Gemini labeling, CSV)
tests/test_dashboard_real.py   12 passed   (real-catalog cards, player, notes, 30-card and trim warnings)
tests/test_media.py             6 passed   (links, notes, preview fallbacks)
tests/test_catalog_builder.py  27 passed   (filters, mood rules, enrichment)
tests/test_playlist_length.py   8 passed   (50-song cap, visible trim warnings, empty catalog)
tests/test_design.py            7 passed   (HTML escaping, meter accessibility, chart bands, CSS rules)
```

One original assertion changed deliberately: `test_profile_validation_clamps_numbers_and_replaces_categories` used to expect a 999-song request to clamp to 20. It now expects `MAX_PLAYLIST_LENGTH` (50). Every other original test is unchanged. Every test that touches the network is mocked.

> Note: in the verification environment the default pytest temp folder was not writable, so the suite was run as `python -m pytest -q -p no:cacheprovider --basetemp=<writable folder>`. The test code is the same either way.

Dashboard tests (`tests/test_dashboard.py`, using `streamlit.testing.v1.AppTest` for the UI):

```text
tests/test_dashboard.py::test_dashboard_loads_without_api_key PASSED
tests/test_dashboard.py::test_empty_request_cannot_generate PASSED
tests/test_dashboard.py::test_preset_fills_request_box PASSED
tests/test_dashboard.py::test_offline_request_generates_labeled_fallback_results PASSED
tests/test_dashboard.py::test_dashboard_results_contain_only_catalog_songs PASSED
tests/test_dashboard.py::test_result_persists_across_reruns PASSED
tests/test_dashboard.py::test_gemini_failure_is_labeled_as_fallback PASSED
tests/test_dashboard.py::test_mocked_gemini_success_is_labeled_gemini PASSED
tests/test_dashboard.py::test_gemini_toggle_off_uses_fallback_even_with_key PASSED
tests/test_dashboard.py::test_system_checks_pass_for_offline_result PASSED
tests/test_dashboard.py::test_playlist_csv_has_verified_fields_and_no_secrets PASSED
tests/test_dashboard.py::test_gemini_service_accepts_explicit_api_key_without_exposing_it PASSED
tests/test_dashboard.py::test_gemini_service_env_key_still_works PASSED
13 passed in 4.56s
```

Dashboard startup check: `streamlit run streamlit_app.py --server.headless true --server.port 8765` started without errors, `/_stcore/health` returned `ok`, and `/` returned HTTP 200. The server was then stopped. Running the script itself (header, form, generation, tabs) is covered by the AppTest suite above.

Visual check: with a throwaway script that seeds a finished playlist into the session, I loaded the dashboard in headless Microsoft Edge and inspected screenshots of both catalogs. Observed: real album covers in an aligned three-column card grid, the active card's accent edge, the sidebar's cover, Spotify player, and 30-second preview, and generated gradient tiles for fictional songs. Observed limitation: in a fresh headless browser the Material icon font loaded late, so icon names briefly showed as text and then rendered correctly after about 25 seconds. Not tested: mobile widths, Safari/Firefox, and clicking Play in a real browser (that path is covered by AppTest only).

Visual check after the redesign (same method; headless Edge driven over the DevTools protocol with a real-time wait). Inspected at 1440, 1100, and 800px wide:

- A 30-song real-catalog playlist showed *Playlist length 30* and *Track 1 of 30*.
- The responsive grid gave three, two, and one column respectively.
- The chart showed Warm-up, Peak, and Cooldown bands.
- A seeded worst case (a 240-character title, a long artist list, a song with no cover, and a Japanese title with emoji in a 50-song playlist) rendered without overlap: the title clamped to two lines, the artist to one, and the missing cover became a gradient tile.

The break-ui pass also found a real crash: an empty catalog raised a bare `StopIteration`. It now raises a clear `ValueError`, which the dashboard shows as an error message. Not inspected: light mode (the theme is dark-only by design), and the reduced-motion, reduced-transparency, and high-contrast modes, whose CSS rules exist and are checked by tests but were not viewed.

Real-catalog check from the CLI (offline, `--catalog data/real_songs.csv`): a "12-song high-energy gym workout" returned energies `[0.57, 0.8, 0.9, 0.9, 0.94, 0.92, 0.92, 0.89, 0.88, 0.81, 0.67, 0.48]`, a moderate start, a peak, and a cooldown.

The "Gemini" dashboard tests use an in-process fake client or a forced failure. They check labeling and validation, not live model quality.

The automated suite covers CSV conversion and validation, legacy score math, normalized score closeness, descending consistent ranking, playlist length, catalog-only IDs, deterministic reruns, energy progression, malformed Gemini JSON, API failures, conflicting preferences, and catalog shortages. `evaluation.py` runs workout, study, and relaxation scenarios twice and reports ID validity, uniqueness, energy sequence, and determinism. See [`evaluation_results.txt`](evaluation_results.txt) for captured evidence.

Live checks that were actually run on November 16, 2026:

- **Gemini, CLI:** one live request, `python -m src.main -r "Make a 6-song high-energy gym workout with a warm-up and cooldown"`, using the key from `.env`. The output reported `interpretation=Gemini, explanations=Gemini`, all six songs came from the catalog, and the energies followed a warm-up, peak, and cooldown shape. That is a single request, not a quality, latency, or cost benchmark.
- **ReccoBeats, Deezer, Apple, Spotify embed:** exercised live while building the real catalog and while viewing the dashboard (covers, previews, and the Spotify player loaded).

Not tested live: the dashboard's Gemini path through a real browser session (the screenshots above used the offline fallback), and live Gemini against the real-music catalog. Those paths are covered by mocked tests only. The services above are unofficial and can change.

> **Student placeholder:** If you run a live Gemini check yourself, record the date, model, request, and whether the dashboard showed *Gemini* for interpretation and explanations. Paste only text you observed.

## Design decisions and trade-offs

- The legacy scorer was preserved rather than overwritten, making the Project 3-to-4 change inspectable.
- Gemini is optional and isolated behind `GeminiService`; this improves reliability and testability at the cost of a simpler offline interpretation.
- Exact genre/mood matches are transparent but do not recognize related labels.
- A progression penalty influences sequence placement but not the displayed compatibility score.
- One-track-per-artist is preferred when enough unique artists exist; relaxing this constraint may sometimes improve pure score.
- No RAG, vector database, learning loop, or multi-agent framework was added because the local structured catalog and deterministic task do not require them.
- The real-music catalog is built once and saved to a CSV rather than fetched at runtime. That keeps selection deterministic, testable, and offline-capable, at the cost of staleness and a one-time review step.
- Real music was added as a second catalog beside the fictional one, which stays the default, instead of replacing it.
- Custom CSS was allowed for the first time, in one removable file, because Streamlit has no native hover states, fixed-aspect covers, or title clamping. Everything else stays native.
- "Where it's from" notes are assembled from catalog fields instead of generated, so they can't contain invented facts. The trade-off is that they are short and generic.

## Limitations

- The synthetic catalog is small and cannot represent real musical diversity.
- Synthetic metadata reflects author-chosen values, not audio analysis.
- The real catalog is only 95 tracks from one seed, so it is eclectic and skews toward whatever ReccoBeats recommends. It mixes languages and genres, and it includes some songs you may not want in a classroom demo; review and prune it.
- ReccoBeats' recommendation filters for energy and tempo are ignored, so coverage of the calm and high-energy ends depends on the builder's re-seeding walk. Calm, positive-sounding tracks are scarce, so relaxing playlists lean on "peaceful" and "focused" moods.
- Real-catalog moods are derived from valence and energy and are approximations. Audio features come from a third-party service and were not independently checked against the recordings.
- ReccoBeats, Deezer, Apple, and Spotify embeds are unofficial or third-party services that can change or disappear, and their terms of use were not independently verified. Embeds depend on Spotify being reachable in your browser.
- Keyword fallback interpretation is intentionally limited.
- The system has no listening history, skips, likes, cultural context, or accessibility preferences.
- Gemini explanations can sound fluent without adding evidence; validation restricts their inputs and IDs, but human review remains important.
- The system plays only short previews and Spotify's own embedded player. It does not stream full tracks, and it should not be treated as a production music service.

## Project reflection

> **Student placeholder:** Add 1–2 paragraphs in your own words about what you learned by combining probabilistic language interpretation with deterministic scoring, what surprised you during testing, and what you would change next. Do not submit this placeholder as your final personal reflection.

## Presentation

Use [`PRESENTATION.md`](PRESENTATION.md) for a timed 5–7 minute walkthrough with a network-free backup path.
