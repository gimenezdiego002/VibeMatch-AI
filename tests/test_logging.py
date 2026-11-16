"""The pipeline logs what it does, and never logs secrets."""

import logging

import pytest

from src.app import create_playlist
from src.chatbot import answer_question
from src.gemini_service import GeminiService
from src.logging_setup import short
from src.recommender import load_songs

SECRET = "AIza-FAKE-KEY-FOR-TESTS-ONLY"


@pytest.fixture
def logs(caplog, monkeypatch):
    monkeypatch.setattr(logging.getLogger("src"), "propagate", True)  # the app may have set False
    caplog.set_level(logging.INFO, logger="src")
    return caplog


class FailingModels:
    def generate_content(self, **kwargs):
        raise RuntimeError(f"429 quota exceeded for key {SECRET}")


def test_playlist_pipeline_logs_each_step(logs):
    create_playlist("15 songs for a workout", offline=True)
    text = logs.text
    assert "Playlist request on songs.csv" in text
    assert "Profile (local): activity=workout length=15" in text
    assert "Playlist ready: 15 of 15 songs" in text


def test_failures_are_logged_briefly_and_never_include_the_key(logs, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    service = GeminiService(client=type("C", (), {"models": FailingModels()})())
    create_playlist("15 songs for a workout", gemini=service)
    assert "Gemini interpretation failed (RuntimeError" in logs.text
    assert SECRET not in logs.text  # truncation keeps the tail of the error (and the key) out
    assert SECRET not in repr(service)


def test_chatbot_logs_retrieval_and_refusals(logs):
    catalog = load_songs("data/songs.csv")
    answer_question("What are the lyrics of #1?", catalog=catalog)
    answer_question("How many pop songs are in the catalog?", catalog=catalog)
    assert "Chat refused" in logs.text
    assert "Chat retrieval:" in logs.text


def test_short_truncates_user_text():
    assert short("a  b\n c") == "a b c"
    assert len(short("x" * 500)) == 120
