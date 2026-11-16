# Execution evidence

Plain-text captures of real command runs, so the system can be checked without a video. Each file starts with the exact command. Re-run any command from the project root to reproduce it; the offline runs are deterministic.

| File | What it shows |
| --- | --- |
| `01_spanish_workout_with_chatbot.txt` | Hard Spanish constraint honored (14/14), honest shortage warning, chatbot answers with sources, and a refusal for lyrics |
| `02_fictional_workout_energy_arc.txt` | Warm-up, peak, and cooldown on the fictional catalog |
| `03_guardrail_60_song_cap.txt` | A 60-song request trimmed to the 50-song cap, with a visible warning |
| `04_guardrail_portuguese_shortage.txt` | Only 9 Portuguese songs exist; returned 9 with a warning instead of padding |
| `05_guardrail_no_language_data.txt` | Language request on a catalog with no language data is dropped with an explanation |
| `06_guardrail_empty_request.txt` | Empty request rejected with a clear error |
| `07_verbose_pipeline_log.txt` | Step-by-step logs (`--verbose`): request, profile, counts, timing, warnings |

Live Gemini results (interpretation and chatbot), the mocked-Gemini rejection tests, and the full evaluation are in [`../evaluation_results.txt`](../evaluation_results.txt).
