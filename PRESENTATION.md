# VibeMatch AI: 5–7 minute demo

**Before you present:**
- Run `streamlit run streamlit_app.py` and open http://localhost:8501.
- Gemini's free tier allows only **20 requests per day**, so don't spend them rehearsing. Every step below also works with **Use Gemini** off.
- Have a terminal open in the project folder for the backup commands.

## 0:00–0:45 — Problem and goal

- **Module 3:** my Music Recommender Simulation ranked 17 fictional songs from a hand-built profile, with fixed bonuses for genre, mood, energy and acousticness.
- **Now:** you describe a situation in plain language, for example "I'm working out, 15 songs, all in Spanish, kinda recent", and get an ordered, explained playlist of real songs. You can then ask questions about those songs.
- **The rule that shapes everything:** Gemini interprets and phrases; Python selects, scores and verifies. Gemini never chooses or invents a song, a score or a fact.

## 0:45–1:30 — Architecture

Show the Mermaid diagram in the README (GitHub renders it) or `diagrams/architecture.mmd`, and trace three paths:

1. **Request → profile.** Gemini, or the local fallback, produces a profile. Python validates it against what the catalog actually has.
2. **Selection.** Hard filters, then Python scoring, then the energy-arc sequencer. Gemini may explain the picks, but only if it covers exactly those IDs.
3. **Chatbot (RAG).** Python retrieves first; Gemini phrases; Python checks the citations, numbers, coverage and asked values, or shows a plain lookup instead.

## 1:30–3:30 — Live dashboard demo

1. Choose **Real music**, type `I am working out I want 15 songs all in spanish kinda recent`, and press **Generate**.
   - **Point out:** "Playlist length adjusted: Only 14 Spanish songs fit this catalog". That's an honest shortage, not English filler.
   - **Point out:** the profile shows **Language: Spanish (only)** and **Release years: 2021 or later (preferred)**, the difference between a hard filter and a soft preference.
   - **Point out:** the interpretation and explanation labels say whether Gemini or the fallback did each step.
2. Press **Play** on a card. The sidebar shows the cover, the sourced album note, the Spotify player and a 30-second preview.
3. Open **Energy flow** to show the warm-up, peak and cooldown bands, and the target line.
4. Open **Ask** and ask:
   - `What album is #1 from?`: answered with a source.
   - `Is there anything quieter than #2?`: names the reference song and lists every match.
   - `What are the lyrics of #3?`: an honest refusal, because the catalog doesn't contain lyrics.

## 3:30–4:30 — Guardrails

- **System checks tab:** IDs exist, metadata matches, IDs are unique, the length is explained, and re-running gives the same result.
- **Score details tab:** contributions sum to the score; the weights are shown; close genres get partial credit.
- In `src/chatbot.py`, `verify_ai_answer()` shows the four reasons an AI answer is thrown away.
- The language column is an AI estimate made once at build time, marked as an estimate and overridable in `data/language_overrides.csv`. One correction is already there.

## 4:30–5:30 — Reliability evidence

```powershell
python -m pytest -q          # 179 passed, every network call mocked
python evaluation.py         # constraint satisfaction, determinism, chatbot accuracy and refusals
```

- **Evaluation:** hard Spanish playlists are 100% Spanish; the chatbot gets 8 of 8 answers right and refuses both off-catalog questions.
- **Rejected answers:** two deliberately bad mocked AI answers were both rejected.
- **Bugs found and fixed through testing:** a silent 20-song cap, ignored language requests, the "ish" in "spanish" softening requests, a score of 100.01, an empty-catalog crash, and vague AI answers passing the checks.

## 5:30–6:30 — Limits and what's next

- The real catalog is small and comes from one seed: only 4 Spanish tracks are workout-energy.
- Language labels are estimates and moods are derived.
- Live Gemini evaluation is thin because of the quota.
- **Next:** Spanish seed tracks, listener feedback to tune the weights, handling negation, and a fuller live evaluation.

## Backup plan (no network or no quota)

Turn **Use Gemini** off, or run the CLI offline. The fallback is a designed feature, and every result is labeled.

```powershell
python -m src.main --offline --verbose --catalog data/real_songs.csv `
  -r "I am working out I want 15 songs all in spanish kinda recent" `
  --ask "What album is #1 from?" --ask "What are the lyrics of #3?"
```

The same outputs are saved in [`evidence/`](evidence/) and in the README in case the laptop itself fails.
