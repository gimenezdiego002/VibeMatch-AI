"""Isolated, optional Gemini integration with strict output validation."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
import importlib.util
import json
import os
from typing import Any, Mapping, Sequence

from src.playlist import PlaylistResult
from src.profiles import MAX_PLAYLIST_LENGTH, PRIORITY_FEATURES, PreferenceProfile, validate_profile


PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "activity": {"type": "string", "enum": ["workout", "studying", "driving", "relaxing", "running", "party", "general"]},
        "playlist_length": {"type": "integer", "minimum": 1, "maximum": MAX_PLAYLIST_LENGTH},
        "preferred_genres": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "target_mood": {"type": "string"},
        "target_energy": {"type": "number", "minimum": 0, "maximum": 1},
        "target_tempo_bpm": {"type": "number", "minimum": 40, "maximum": 220},
        "playlist_structure": {"type": "string", "enum": ["warmup_peak_cooldown", "build", "consistent", "balanced"]},
    },
    "required": ["activity", "playlist_length", "preferred_genres", "target_mood", "target_energy", "target_tempo_bpm", "playlist_structure"],
    "additionalProperties": False,
}


def profile_schema(languages: Sequence[str], current_year: int) -> dict[str, Any]:
    """PROFILE_SCHEMA plus optional constraints whose language enum comes from the catalog."""
    schema = json.loads(json.dumps(PROFILE_SCHEMA))  # deep copy; PROFILE_SCHEMA stays the base contract
    year = {"type": "integer", "minimum": 0, "maximum": current_year}
    schema["properties"].update(
        {
            "steady_requested": {"type": "boolean"},
            "language": {"type": "string", "enum": ["none", *languages]},
            "language_strict": {"type": "boolean"},
            "min_release_year": year,
            "max_release_year": year,
            "recency_strict": {"type": "boolean"},
            "priorities": {
                "type": "object",
                "properties": {f: {"type": "number", "minimum": 0, "maximum": 1} for f in PRIORITY_FEATURES},
                "additionalProperties": False,
            },
        }
    )
    schema["required"] = [
        *PROFILE_SCHEMA["required"],
        "steady_requested", "language", "language_strict", "min_release_year", "max_release_year", "recency_strict",
    ]
    return schema


class GeminiService:
    """Gemini adapter; callers can inject a fake client for tests."""

    def __init__(
        self,
        client: Any | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
    ) -> None:
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self._client = client
        # An explicit key (for example from Streamlit secrets) takes precedence over
        # the GEMINI_API_KEY environment variable. The key is never logged or exposed.
        self._api_key = api_key.strip() if isinstance(api_key, str) and api_key.strip() else None

    def __repr__(self) -> str:
        return f"GeminiService(model={self.model!r}, available={self.available})"

    def _resolved_api_key(self) -> str | None:
        return self._api_key or os.getenv("GEMINI_API_KEY") or None

    @property
    def available(self) -> bool:
        """True when a client or credential is configured; not proof that a call will succeed."""
        return self._client is not None or bool(self._resolved_api_key())

    @staticmethod
    def sdk_installed() -> bool:
        """Report whether the optional google-genai SDK can be imported."""
        try:
            return importlib.util.find_spec("google.genai") is not None
        except (ImportError, ValueError):
            return False

    def _client_or_create(self) -> Any:
        if self._client is not None:
            return self._client
        api_key = self._resolved_api_key()
        if not api_key:
            raise RuntimeError("No Gemini API key is configured.")
        from google import genai

        self._client = genai.Client(api_key=api_key)
        return self._client

    def interpret(
        self,
        request: str,
        *,
        supported_genres: set[str],
        supported_moods: set[str],
        supported_languages: set[str] | None = None,
        catalog_has_years: bool = False,
    ) -> tuple[PreferenceProfile, list[str]]:
        """Convert free text to a validated profile using structured output."""
        year = date.today().year
        languages = sorted(supported_languages or ())
        prompt = (
            "Interpret this playlist request. Infer reasonable preferences. "
            f"Use only these genres: {sorted(supported_genres)}. "
            f"Use only these moods: {sorted(supported_moods)}.\n"
            "Constraints:\n"
            f"- language: the language the lyrics should be in, one of {languages or ['none']}, or 'none' if the "
            "user did not ask for one. language_strict is true when the user wants ALL songs in it "
            "('all in Spanish', 'only Spanish', 'Spanish songs'), false for a soft wish ('preferably', 'some').\n"
            f"- min_release_year / max_release_year: the current year is {year}. 'recent' or 'new' means "
            f"min_release_year={year - 5}; a decade like 'the 90s' means 1990 to 1999; use 0 when not asked. "
            "recency_strict is true only for wording like 'only new songs'.\n"
            "- steady_requested: true only if the user explicitly asks for steady or consistent energy.\n"
            "- priorities (optional): how much this user cares about each factor, 0 to 1. "
            "Do not score songs.\n"
            f"Request: {request}"
        )
        response = self._client_or_create().models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": profile_schema(languages, year),
                "temperature": 0.1,
            },
        )
        raw = getattr(response, "parsed", None)
        if not isinstance(raw, Mapping):
            raw = json.loads(response.text)
        if not isinstance(raw, Mapping):
            raise ValueError("Gemini profile response was not a JSON object.")
        return validate_profile(
            raw,
            supported_genres=supported_genres,
            supported_moods=supported_moods,
            supported_languages=supported_languages,
            catalog_has_years=catalog_has_years,
            reference_year=year,
        )

    def classify_languages(self, tracks: Sequence[Mapping[str, Any]]) -> dict[int, tuple[str, float]]:
        """Catalog-build-time only: estimate the sung language of each track.

        Returns {id: (language, confidence)} for the IDs Gemini answered validly. The caller
        must fall back to rules for any ID that is missing. Never called at playlist time.
        """
        from src.languages import LANGUAGES, normalize_language

        payload = [
            {"id": int(t["id"]), "title": t.get("title", ""), "artist": t.get("artist", ""),
             "album": t.get("album", ""), "genre": t.get("genre", "")}
            for t in tracks
        ]
        schema = {
            "type": "object",
            "properties": {
                "labels": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "language": {"type": "string", "enum": list(LANGUAGES)},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": ["id", "language", "confidence"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["labels"],
            "additionalProperties": False,
        }
        response = self._client_or_create().models.generate_content(
            model=self.model,
            contents=(
                "For each track, give the main language its lyrics are sung in, chosen from the allowed list. "
                "Use 'instrumental' only if it has no vocals and 'other' if the language is not listed. "
                "Give a lower confidence when you are unsure. Answer for every id.\n"
                f"Tracks: {json.dumps(payload, ensure_ascii=False)}"
            ),
            config={"response_mime_type": "application/json", "response_json_schema": schema, "temperature": 0.0},
        )
        raw = getattr(response, "parsed", None)
        if not isinstance(raw, Mapping):
            raw = json.loads(response.text)
        valid_ids = {int(t["id"]) for t in tracks}
        labels: dict[int, tuple[str, float]] = {}
        for item in raw.get("labels", []) if isinstance(raw, Mapping) else []:
            if not isinstance(item, Mapping):
                continue
            item_id, language = item.get("id"), normalize_language(item.get("language"))
            try:
                confidence = min(1.0, max(0.0, float(item.get("confidence", 0))))
            except (TypeError, ValueError):
                continue
            if isinstance(item_id, int) and item_id in valid_ids and language:
                labels[item_id] = (language, round(confidence, 2))
        return labels

    def explain(self, result: PlaylistResult) -> PlaylistResult:
        """Explain only selected verified records; never permit ranking changes."""
        payload = [
            {
                "id": int(item.song["id"]),
                "title": item.song["title"],
                "artist": item.song["artist"],
                "genre": item.song["genre"],
                "mood": item.song["mood"],
                "energy": item.song["energy"],
                "tempo_bpm": item.song["tempo_bpm"],
                "score": item.score,
                "deterministic_reason": item.deterministic_reason,
            }
            for item in result.songs
        ]
        schema = {
            "type": "object",
            "properties": {
                "explanations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}, "explanation": {"type": "string"}},
                        "required": ["id", "explanation"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["explanations"],
            "additionalProperties": False,
        }
        response = self._client_or_create().models.generate_content(
            model=self.model,
            contents=(
                "Write one concise, situation-specific sentence for each selected song. "
                "Do not change order, scores, metadata, or make claims beyond the supplied fields.\n"
                f"Activity: {result.profile.activity}\nSelected records: {json.dumps(payload)}"
            ),
            config={"response_mime_type": "application/json", "response_json_schema": schema, "temperature": 0.2},
        )
        raw = getattr(response, "parsed", None)
        if not isinstance(raw, Mapping):
            raw = json.loads(response.text)
        selected_ids = {int(item.song["id"]) for item in result.songs}
        explanations: dict[int, str] = {}
        for item in raw.get("explanations", []) if isinstance(raw, Mapping) else []:
            item_id = item.get("id") if isinstance(item, Mapping) else None
            text = item.get("explanation") if isinstance(item, Mapping) else None
            if isinstance(item_id, int) and item_id in selected_ids and isinstance(text, str) and text.strip():
                explanations[item_id] = text.strip()[:300]
        if set(explanations) != selected_ids:
            raise ValueError("Gemini explanations did not cover exactly the selected catalog IDs.")
        songs = tuple(replace(item, explanation=explanations[int(item.song["id"])]) for item in result.songs)
        return replace(result, songs=songs, used_gemini_explanations=True)
