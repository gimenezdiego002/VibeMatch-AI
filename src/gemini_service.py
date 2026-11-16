"""Isolated, optional Gemini integration with strict output validation."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import os
from typing import Any, Mapping, Sequence

from src.playlist import PlaylistResult
from src.profiles import MAX_PLAYLIST_LENGTH, PreferenceProfile, validate_profile


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
    ) -> tuple[PreferenceProfile, list[str]]:
        """Convert free text to a validated profile using structured output."""
        prompt = (
            "Interpret this playlist request. Infer reasonable preferences. "
            f"Use only these genres: {sorted(supported_genres)}. "
            f"Use only these moods: {sorted(supported_moods)}.\nRequest: {request}"
        )
        response = self._client_or_create().models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": PROFILE_SCHEMA,
                "temperature": 0.1,
            },
        )
        raw = getattr(response, "parsed", None)
        if not isinstance(raw, Mapping):
            raw = json.loads(response.text)
        if not isinstance(raw, Mapping):
            raise ValueError("Gemini profile response was not a JSON object.")
        return validate_profile(raw, supported_genres=supported_genres, supported_moods=supported_moods)

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
