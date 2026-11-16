"""Dashboard tests: Streamlit AppTest for UI behavior, plain pytest for pure helpers.

No test contacts the real Gemini API. "Gemini" paths are exercised only with an
in-process fake client or a forced failure.
"""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.app import create_playlist
from src.dashboard_data import playlist_csv, playlist_dataframe, system_checks
from src.gemini_service import PROFILE_SCHEMA, GeminiService
from src.recommender import load_songs


APP_PATH = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")
CATALOG = "data/songs.csv"
FAKE_KEY = "test-key-should-never-render"


@pytest.fixture(autouse=True)
def no_ambient_gemini_key(monkeypatch):
    # Never let tests read a developer's real .env key or make live API calls.
    monkeypatch.setattr("src.env.load_local_env", lambda *args, **kwargs: False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)


def new_app() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=30)


def all_text(at: AppTest) -> str:
    parts = [str(el.value) for el in [*at.markdown, *at.caption, *at.info, *at.warning, *at.success, *at.error]]
    parts += [f"{m.label} {m.value}" for m in at.metric]
    return "\n".join(parts)


def metric_value(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def generate(at: AppTest, request: str) -> AppTest:
    at.text_area(key="request_text").set_value(request)
    at.form_submit_button[0].click().run()
    return at


# ---------------------------------------------------------------------------
# AppTest: UI behavior
# ---------------------------------------------------------------------------


def test_dashboard_loads_without_api_key():
    at = new_app().run()
    assert not at.exception
    assert not at.error
    text = all_text(at)
    assert "Gemini unavailable: no API key" in text
    assert "fictional" in text.casefold()
    toggle = at.toggle(key="use_gemini")
    assert toggle.disabled
    assert toggle.value is False
    # The catalog table is available before any playlist is generated.
    assert metric_value(at, "Songs") == "50"


def test_empty_request_cannot_generate():
    at = new_app().run()
    generate(at, "   ")
    assert not at.exception
    assert at.session_state["last_run"] is None
    assert any("Enter a playlist request" in w.value for w in at.warning)


def test_preset_fills_request_box():
    at = new_app().run()
    at.pills(key="preset").set_value("workout").run()
    assert not at.exception
    assert "gym workout" in at.text_area(key="request_text").value


def test_offline_request_generates_labeled_fallback_results():
    at = new_app().run()
    generate(at, "Make a 10-song high-energy gym workout with a warm-up and cooldown")
    assert not at.exception

    run = at.session_state["last_run"]
    assert run is not None and run["gemini_requested"] is False
    result = run["result"]
    assert len(result.songs) == 10

    assert metric_value(at, "Request interpretation") == "Local fallback"
    assert metric_value(at, "Explanations") == "Deterministic reasons"
    assert any("No AI model was used" in i.value for i in at.info)
    assert not at.success  # never claims a Gemini success offline

    playlist_table = at.dataframe[0].value
    assert list(playlist_table["rank"]) == list(range(1, 11))


def test_dashboard_results_contain_only_catalog_songs():
    catalog = {song["id"]: song for song in load_songs(CATALOG)}
    at = new_app().run()
    generate(at, "Give me 10 relaxing low-energy songs for studying with a steady feel")
    assert not at.exception
    table = at.dataframe[0].value
    assert set(table["id"]) <= set(catalog)
    assert table["id"].is_unique
    for row in table.itertuples():
        assert row.title == catalog[row.id]["title"]
        assert row.artist == catalog[row.id]["artist"]
    assert "Selected IDs exist in the local catalog" in all_text(at)


def test_result_persists_across_reruns():
    at = new_app().run()
    generate(at, "Create 8 peaceful songs for relaxing after work")
    first_ids = [item.song["id"] for item in at.session_state["last_run"]["result"].songs]
    at.selectbox(key="inspect_rank").set_value(3).run()
    assert not at.exception
    assert [item.song["id"] for item in at.session_state["last_run"]["result"].songs] == first_ids


def test_gemini_failure_is_labeled_as_fallback(monkeypatch):
    def failing_client(self):
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr(GeminiService, "sdk_installed", staticmethod(lambda: True))
    monkeypatch.setattr(GeminiService, "_client_or_create", failing_client)

    at = new_app()
    at.secrets["GEMINI_API_KEY"] = FAKE_KEY
    at.run()
    assert "Gemini available: API key configured" in all_text(at)
    assert at.toggle(key="use_gemini").value is True

    generate(at, "Make 6 calm songs for studying")
    assert not at.exception
    assert metric_value(at, "Request interpretation") == "Local fallback"
    assert metric_value(at, "Explanations") == "Deterministic reasons"
    assert any("Partial or full fallback" in str(w.proto) for w in at.warning)
    assert not at.success
    assert any("failed" in warning for warning in at.session_state["last_run"]["result"].warnings)
    assert FAKE_KEY not in all_text(at)


class _FakeResponse:
    def __init__(self, parsed):
        self.parsed = parsed
        self.text = json.dumps(parsed)


class _FakeModels:
    """Returns a valid profile, then explanations for exactly the IDs it was sent."""

    def generate_content(self, *, model, contents, config):
        if config.get("response_json_schema") == PROFILE_SCHEMA:
            return _FakeResponse(
                {
                    "activity": "studying",
                    "playlist_length": 5,
                    "preferred_genres": ["lofi"],
                    "target_mood": "focused",
                    "target_energy": 0.3,
                    "target_tempo_bpm": 78,
                    "playlist_structure": "consistent",
                }
            )
        records = json.loads(contents.split("Selected records: ", 1)[1])
        return _FakeResponse({"explanations": [{"id": r["id"], "explanation": f"Mock note for {r['title']}."} for r in records]})


class _FakeClient:
    models = _FakeModels()


def test_mocked_gemini_success_is_labeled_gemini(monkeypatch):
    monkeypatch.setattr(GeminiService, "sdk_installed", staticmethod(lambda: True))
    monkeypatch.setattr(GeminiService, "_client_or_create", lambda self: _FakeClient())

    at = new_app()
    at.secrets["GEMINI_API_KEY"] = FAKE_KEY
    at.run()
    generate(at, "something calm to study to")
    assert not at.exception
    assert metric_value(at, "Request interpretation") == "Gemini"
    assert metric_value(at, "Explanations") == "Gemini"
    assert at.success
    assert "Mock note for" in all_text(at)
    assert FAKE_KEY not in all_text(at)


def test_gemini_toggle_off_uses_fallback_even_with_key(monkeypatch):
    def must_not_call(self):
        raise AssertionError("Gemini must not be called when the toggle is off")

    monkeypatch.setattr(GeminiService, "sdk_installed", staticmethod(lambda: True))
    monkeypatch.setattr(GeminiService, "_client_or_create", must_not_call)

    at = new_app()
    at.secrets["GEMINI_API_KEY"] = FAKE_KEY
    at.run()
    at.toggle(key="use_gemini").set_value(False)
    generate(at, "Create 8 peaceful songs for relaxing after work")
    assert not at.exception
    assert at.session_state["last_run"]["gemini_requested"] is False
    assert metric_value(at, "Request interpretation") == "Local fallback"


# ---------------------------------------------------------------------------
# Plain pytest: pure helpers and the GeminiService refactor
# ---------------------------------------------------------------------------


def test_system_checks_pass_for_offline_result():
    catalog = load_songs(CATALOG)
    result = create_playlist("Make a 10-song high-energy gym workout", offline=True)
    checks = {check.name: check for check in system_checks(result, catalog, gemini_requested=False)}
    for name in (
        "Selected IDs exist in the local catalog",
        "Metadata matches the catalog",
        "Selected IDs are unique",
        "Returned length matches requested length",
        "Python selection is deterministic",
    ):
        assert checks[name].passed is True, name
    assert "Gemini was off" in checks["Request interpretation"].detail


def test_playlist_csv_has_verified_fields_and_no_secrets(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    result = create_playlist("Create 8 peaceful songs for relaxing after work", offline=True)
    csv_text = playlist_csv(result)
    header = csv_text.splitlines()[0].split(",")
    for column in ("rank", "id", "title", "artist", "compatibility_score", "position_target_energy", "explanation"):
        assert column in header
    assert len(csv_text.strip().splitlines()) == 1 + len(result.songs)
    assert FAKE_KEY not in csv_text
    assert len(playlist_dataframe(result)) == len(result.songs)


def test_gemini_service_accepts_explicit_api_key_without_exposing_it():
    service = GeminiService(api_key=FAKE_KEY)
    assert service.available
    assert FAKE_KEY not in repr(service)
    assert not GeminiService(api_key="   ").available
    assert not GeminiService().available


def test_gemini_service_env_key_still_works(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    assert GeminiService().available
