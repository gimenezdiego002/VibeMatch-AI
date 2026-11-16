"""Catalog chatbot (RAG): validation, retrieval, grounding checks, fallbacks, and the Ask tab.

Gemini is always a local fake here; no test makes a network call.
"""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import media
from src.chatbot import (
    REFUSAL, answer_question, build_index, parse_question_offline, retrieve, validate_query, verify_ai_answer,
)


ROOT = Path(__file__).resolve().parents[1]


def make_song(i, title, artist, energy, bpm, year, genre="pop", mood="happy", language="english", album=None):
    return {"id": i, "title": title, "artist": artist, "genre": genre, "mood": mood, "energy": energy,
            "tempo_bpm": bpm, "release_date": f"{year}-01-01", "language": language, "album": album or f"{title} LP"}


CATALOG = [
    make_song(1, "Sunrise", "Ana", 0.40, 100, 2015),
    make_song(2, "Fuego Lento", "Beto", 0.80, 128, 2022, genre="latin", language="spanish"),
    make_song(3, "Night Drive", "Cara", 0.90, 145, 2019),
    make_song(4, "Quiet Hours", "Dani", 0.20, 70, 2010),
    make_song(5, "Corazón", "Eva", 0.70, 150, 2024, genre="reggaeton", language="spanish"),
]
PLAYLIST = [CATALOG[2], CATALOG[1], CATALOG[0]]  # ranks: #1 Night Drive, #2 Fuego Lento, #3 Sunrise


def ask(question, **kwargs):
    return answer_question(question, catalog=CATALOG, playlist_songs=PLAYLIST, **kwargs)


# ----- validation -----

def test_validate_query_drops_unknown_fields_filters_and_bad_values():
    query, warnings = validate_query({
        "intent": "hack", "fields": ["album", "lyrics"], "filters": {"bpm_min": "fast", "explicit": True, "genre": "Pop"},
        "sort_by": "popularity", "limit": 999, "relative": {"field": "energy", "direction": "sideways", "reference_rank": 2},
    })
    assert query.intent == "lookup" and query.fields == ("album",)
    assert dict(query.filters) == {"genre": "pop"}
    assert query.sort_by == "" and query.limit == 10 and query.relative is None
    assert len(warnings) == 6


# ----- retrieval -----

def test_rank_lookup_and_missing_rank():
    index = build_index(CATALOG, PLAYLIST)
    records, total, notes = retrieve(validate_query({"target_ranks": [2]})[0], index)
    assert [r["title"] for r in records] == ["Fuego Lento"] and total == 1 and not notes
    _, _, notes = retrieve(validate_query({"target_ranks": [9]})[0], index)
    assert notes == ["There is no song #9 in the current playlist."]


def test_filters_scope_and_relative_comparison():
    index = build_index(CATALOG, PLAYLIST)
    records, total, _ = retrieve(validate_query({"filters": {"bpm_min": 130}})[0], index)
    assert [r["title"] for r in records] == ["Night Drive"]  # playlist scope by default
    records, total, _ = retrieve(validate_query({"scope": "catalog", "filters": {"language": "spanish"}})[0], index)
    assert {r["title"] for r in records} == {"Fuego Lento", "Corazón"} and total == 2
    records, _, _ = retrieve(validate_query({"relative": {"field": "energy", "direction": "lower", "reference_rank": 2}})[0], index)
    assert [r["title"] for r in records] == ["Sunrise"]  # quieter than #2 Fuego Lento (0.80), within the playlist


def test_fuzzy_title_match():
    index = build_index(CATALOG, PLAYLIST)
    records, _, _ = retrieve(validate_query({"target_text": "night drive"})[0], index)
    assert [r["title"] for r in records] == ["Night Drive"]


# ----- offline parser and answers -----

@pytest.mark.parametrize(
    "question,expected_in_answer",
    [
        ("What album is #2 from?", "Fuego Lento LP"),
        ("Which songs are over 140 BPM?", "Night Drive"),
        ("Is there anything quieter than #2?", "quieter than #2 Fuego Lento"),
        ("How many Spanish songs do you have in the catalog?", "2 songs in this catalog match."),
        ("When was Night Drive released?", "released in 2019"),
    ],
)
def test_offline_answers_come_from_the_catalog(question, expected_in_answer):
    answer = ask(question)
    assert answer.source == "Catalog lookup"
    assert expected_in_answer in answer.text


@pytest.mark.parametrize("question", ["What are the lyrics of #1?", "Who is Cara married to?", "Did Night Drive win a Grammy?"])
def test_questions_outside_the_catalog_are_refused(question):
    answer = ask(question)
    assert answer.text == REFUSAL and not answer.records


def test_unknown_song_says_so_instead_of_guessing():
    assert "couldn't find a song" in ask('Tell me about "Bohemian Rhapsody"').text


def test_offline_parser_reads_counts_and_bpm():
    raw = parse_question_offline("how many songs are under 110 bpm", genres=set(), moods=set(), languages=set())
    assert raw["intent"] == "count" and raw["filters"] == {"bpm_max": 110.0}


# ----- grounding checks -----

RECORDS = build_index(CATALOG, PLAYLIST)[:2]


def test_verify_accepts_a_grounded_complete_answer():
    text = "Sunrise (100 BPM, 2015) and Fuego Lento (128 BPM) match."
    assert verify_ai_answer(text, [1, 2], RECORDS, 2) is None


@pytest.mark.parametrize(
    "text,cited,reason",
    [
        ("Sunrise and Fuego Lento.", [1, 2, 99], "cited song ID 99"),
        ("Sunrise is 175 BPM.", [1, 2], "quoted the number 175"),
        ("Only Sunrise matches.", [1], "left out 1 of the 2"),
        ("", [1, 2], "empty"),
    ],
)
def test_verify_rejects_invented_ids_numbers_and_incomplete_answers(text, cited, reason):
    assert reason in verify_ai_answer(text, cited, RECORDS, 2)


def test_verify_requires_the_values_that_were_asked_for():
    vague = "Sunrise and Fuego Lento are in the playlist."
    assert "did not give the tempo bpm" in verify_ai_answer(vague, [1, 2], RECORDS, 2, required_fields=("tempo_bpm",))
    specific = "Sunrise is 100 BPM and Fuego Lento is 128 BPM."
    assert verify_ai_answer(specific, [1, 2], RECORDS, 2, required_fields=("tempo_bpm",)) is None


def test_verify_count_answers_need_only_the_count():
    assert verify_ai_answer("There are 2 matching songs.", [1, 2], RECORDS, 2, required_fields=("language",), intent="count") is None
    assert "did not state the count" in verify_ai_answer("A couple of songs.", [1, 2], RECORDS, 2, intent="count")


class FakeGemini:
    def __init__(self, query=None, reply=None, parse_error=None):
        self.query, self.reply, self.parse_error = query, reply, parse_error
        self.answer_calls = []

    def parse_question(self, question, **kwargs):
        if self.parse_error:
            raise self.parse_error
        return self.query

    def answer_from_records(self, question, *, records, total_matches, reference=None):
        self.answer_calls.append({"records": records, "reference": reference})
        return self.reply


def test_grounded_ai_answer_is_shown_and_labeled():
    fake = FakeGemini({"intent": "lookup", "target_ranks": [2], "fields": ["album"]},
                      {"answer": "#2 Fuego Lento is from the album Fuego Lento LP.", "cited_ids": [2]})
    answer = ask("what album is the second one from", gemini=fake, use_gemini=True)
    assert answer.source == "AI answer" and answer.cited_ids == (2,)


def test_ai_answer_citing_an_unretrieved_song_is_discarded():
    fake = FakeGemini({"intent": "lookup", "target_ranks": [2]}, {"answer": "Fuego Lento and Quiet Hours.", "cited_ids": [2, 4]})
    answer = ask("tell me about #2", gemini=fake, use_gemini=True)
    assert answer.source == "Catalog lookup" and "Fuego Lento" in answer.text
    assert any("cited song ID 4" in w for w in answer.warnings)


def test_incomplete_comparison_answer_is_discarded_and_reference_is_passed():
    fake = FakeGemini({"intent": "filter", "relative": {"field": "tempo_bpm", "direction": "higher", "reference_rank": 3}},
                      {"answer": "Night Drive is faster.", "cited_ids": [3]})
    answer = ask("which are faster than #3", gemini=fake, use_gemini=True)
    assert fake.answer_calls[0]["reference"]["title"] == "Sunrise"
    assert answer.source == "Catalog lookup" and "faster than #3 Sunrise" in answer.text
    assert any("left out" in w for w in answer.warnings)


def test_gemini_parse_failure_falls_back_to_local_parser():
    answer = ask("What album is #2 from?", gemini=FakeGemini(parse_error=RuntimeError("quota")), use_gemini=True)
    assert answer.source == "Catalog lookup" and "Fuego Lento LP" in answer.text
    assert any("used the local parser" in w for w in answer.warnings)


def test_gemini_unsupported_intent_is_refused_without_an_answer_call():
    fake = FakeGemini({"intent": "unsupported"}, {"answer": "made up", "cited_ids": []})
    assert ask("lyrics please", gemini=fake, use_gemini=True).text == REFUSAL
    assert fake.answer_calls == []


# ----- Ask tab -----

@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr("src.env.load_local_env", lambda *a, **k: False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(media, "fetch_preview_url", lambda song: None)
    st.cache_data.clear()
    return AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60).run()


def test_ask_tab_answers_about_the_playlist_offline(app):
    app.text_area(key="request_text").set_value("Make a 10-song high-energy gym workout")
    app.form_submit_button[0].click().run()
    first = app.session_state["last_run"]["result"].songs[0].song
    app.chat_input(key="chat_input").set_value("What album is #1 from?").run()
    assert not app.exception
    chat = app.session_state["chat"]
    assert chat[0] == {"role": "user", "content": "What album is #1 from?"}
    assert chat[1]["source"] == "Catalog lookup" and first["title"] in chat[1]["content"]
    assert any("Catalog lookup" in str(b.value) for b in app.markdown)


def test_ask_tab_refuses_trivia_and_suggestions_work(app):
    app.pills(key="chat_suggestion").set_value(":material/speed: Which songs are over 130 BPM?").run()
    assert not app.exception
    assert app.session_state["chat"][0]["content"] == "Which songs are over 130 BPM?"
    app.chat_input(key="chat_input").set_value("What are the lyrics of #1?").run()
    assert app.session_state["chat"][-1]["content"] == REFUSAL
