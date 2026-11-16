"""Catalog question answering: retrieve with Python first, then (optionally) let Gemini phrase it.

Pipeline (retrieval-augmented generation):
  1. Understand: Gemini turns the question into a small structured query, or the offline
     parser does. Python validates it; unknown fields or filters are dropped with a warning.
  2. Retrieve: Python runs the query against the catalog and the current playlist.
  3. Answer: Gemini writes a short answer from ONLY the retrieved records and must cite song
     IDs. Python rejects the answer if it cites a song that was not retrieved or quotes a number
     that is not in those records, and falls back to a deterministic answer built from them.
Questions the catalog cannot answer (lyrics, biographies, trivia) get an honest refusal.

No Streamlit imports here, so everything is unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from difflib import SequenceMatcher
import re
from typing import Any, Mapping, Sequence

from src.languages import normalize_language


FIELDS = ("title", "artist", "album", "release_year", "genre", "mood", "language", "tempo_bpm", "energy")
NUMERIC_FILTERS = ("year_min", "year_max", "bpm_min", "bpm_max", "energy_min", "energy_max")
TEXT_FILTERS = ("genre", "mood", "language")
INTENTS = ("lookup", "filter", "count", "unsupported")
SORTABLE = ("energy", "tempo_bpm", "release_year")
MAX_RESULTS = 10
MAX_QUESTION_CHARS = 300
REFUSAL = (
    "The catalog doesn't contain that. I can only answer from each song's title, artist, album, "
    "release year, genre, mood, language, tempo, and energy."
)
UNSUPPORTED_TERMS = (
    "lyric", "lyrics", "biography", "born", "married", "wife", "husband", "net worth", "grammy", "award",
    "chart", "billboard", "streams", "followers", "meaning of", "what does it mean", "who wrote", "producer",
    "concert", "tour", "tickets", "record label",
)
FIELD_TERMS = {
    "album": ("album", "record", "from which"),
    "release_year": ("year", "when", "released", "release", "how old", "come out", "came out"),
    "tempo_bpm": ("bpm", "tempo", "how fast", "speed"),
    "energy": ("energy", "how intense", "how energetic"),
    "genre": ("genre", "style", "kind of music", "type of music"),
    "mood": ("mood", "vibe", "feel"),
    "language": ("language", "what language", "sung in"),
    "artist": ("artist", "who sings", "who is it by", "singer", "band"),
}


@dataclass(frozen=True)
class Query:
    intent: str = "lookup"
    scope: str = "playlist"  # "playlist" or "catalog"
    target_ranks: tuple[int, ...] = ()
    target_text: str = ""
    fields: tuple[str, ...] = ()
    filters: tuple[tuple[str, Any], ...] = ()
    relative: tuple[str, str, int] | None = None  # (field, "lower"/"higher", reference rank)
    sort_by: str = ""
    descending: bool = True
    limit: int = MAX_RESULTS


@dataclass(frozen=True)
class ChatAnswer:
    text: str
    source: str  # "AI answer" or "Catalog lookup"
    records: tuple[Mapping[str, Any], ...] = ()
    cited_ids: tuple[int, ...] = ()
    total_matches: int = 0
    warnings: tuple[str, ...] = ()


# ----- records -----

def song_record(song: Mapping[str, Any], rank: int | None = None) -> dict[str, Any]:
    """The only fields the chatbot may use. Everything comes from the validated catalog row."""
    release = str(song.get("release_date", "") or "")
    return {
        "id": int(song["id"]),
        "rank": rank,
        "title": str(song["title"]),
        "artist": str(song["artist"]),
        "album": str(song.get("album", "") or "") or None,
        "release_year": int(release[:4]) if release[:4].isdigit() else None,
        "genre": str(song["genre"]),
        "mood": str(song["mood"]),
        "language": str(song.get("language", "") or "") or None,
        "tempo_bpm": int(song["tempo_bpm"]),
        "energy": round(float(song["energy"]), 2),
    }


def build_index(catalog: Sequence[Mapping[str, Any]], playlist_songs: Sequence[Mapping[str, Any]] = ()) -> list[dict[str, Any]]:
    ranks = {int(song["id"]): position for position, song in enumerate(playlist_songs, start=1)}
    return [song_record(song, ranks.get(int(song["id"]))) for song in catalog]


# ----- validation -----

def validate_query(raw: Mapping[str, Any]) -> tuple[Query, list[str]]:
    """Accept only known intents, fields, filters, and limits. Anything else is dropped."""
    warnings: list[str] = []
    intent = str(raw.get("intent", "lookup")).casefold()
    if intent not in INTENTS:
        warnings.append(f"Unknown intent '{intent}' treated as a lookup.")
        intent = "lookup"
    scope = "catalog" if str(raw.get("scope", "playlist")).casefold() == "catalog" else "playlist"

    ranks = []
    for value in raw.get("target_ranks", []) or []:
        try:
            ranks.append(int(value))
        except (TypeError, ValueError):
            warnings.append(f"Ignored invalid song number '{value}'.")
    fields = []
    for name in raw.get("fields", []) or []:
        if name in FIELDS:
            fields.append(name)
        else:
            warnings.append(f"Ignored unknown field '{name}'.")

    filters: list[tuple[str, Any]] = []
    raw_filters = raw.get("filters", {}) or {}
    if isinstance(raw_filters, Mapping):
        for name, value in raw_filters.items():
            if value in (None, "", 0) and name not in ("energy_min",):
                continue
            if name in TEXT_FILTERS:
                text = normalize_language(value) if name == "language" else str(value).strip().casefold()
                if text:
                    filters.append((name, text))
            elif name in NUMERIC_FILTERS:
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    warnings.append(f"Ignored non-numeric filter '{name}'.")
                    continue
                if name.startswith("energy"):
                    number = min(max(number, 0.0), 1.0)
                filters.append((name, number))
            else:
                warnings.append(f"Ignored unknown filter '{name}'.")

    relative = None
    raw_relative = raw.get("relative")
    if isinstance(raw_relative, Mapping) and raw_relative:
        rel_field, direction = raw_relative.get("field"), raw_relative.get("direction")
        try:
            reference = int(raw_relative.get("reference_rank"))
        except (TypeError, ValueError):
            reference = 0
        if rel_field in SORTABLE and direction in ("lower", "higher") and reference > 0:
            relative = (rel_field, direction, reference)
        else:
            warnings.append("Ignored an invalid comparison.")

    sort_by = raw.get("sort_by") or ""
    if sort_by and sort_by not in SORTABLE:
        warnings.append(f"Ignored unknown sort field '{sort_by}'.")
        sort_by = ""
    try:
        limit = int(raw.get("limit", MAX_RESULTS) or MAX_RESULTS)
    except (TypeError, ValueError):
        limit = MAX_RESULTS
    limit = min(max(limit, 1), MAX_RESULTS)
    query = Query(
        intent=intent, scope=scope, target_ranks=tuple(ranks), target_text=str(raw.get("target_text", "") or "")[:80],
        fields=tuple(dict.fromkeys(fields)), filters=tuple(filters), relative=relative, sort_by=sort_by,
        descending=str(raw.get("order", "desc")).casefold() != "asc", limit=limit,
    )
    return query, warnings


# ----- retrieval -----

def _fuzzy_match(text: str, records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    needle = text.casefold().strip()
    if not needle:
        return []
    scored = []
    for record in records:
        for hay in (record["title"].casefold(), record["artist"].casefold()):
            ratio = 1.0 if needle in hay or (len(hay) > 3 and hay in needle) else SequenceMatcher(None, needle, hay).ratio()
            scored.append((ratio, record["id"], record))
    best = max((s[0] for s in scored), default=0)
    if best < 0.6:
        return []
    return [record for ratio, _, record in sorted(scored, key=lambda s: (-s[0], s[1])) if ratio == best]


def retrieve(query: Query, index: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, list[str]]:
    """Run a validated query. Returns (records capped to the limit, total matches, notes)."""
    notes: list[str] = []
    playlist = sorted((r for r in index if r["rank"]), key=lambda r: r["rank"])
    by_rank = {r["rank"]: r for r in playlist}
    pool = list(index) if query.scope == "catalog" or not playlist else playlist

    if query.target_ranks:
        missing = [rank for rank in query.target_ranks if rank not in by_rank]
        if missing:
            notes.append(f"There is no song #{missing[0]} in the current playlist." if playlist
                         else "Generate a playlist first to ask about songs by number.")
        matches = [by_rank[rank] for rank in query.target_ranks if rank in by_rank]
    elif query.target_text:
        matches = _fuzzy_match(query.target_text, pool) or _fuzzy_match(query.target_text, index)
    else:
        matches = list(pool)

    filters = dict(query.filters)
    def keep(record: dict[str, Any]) -> bool:
        for name in TEXT_FILTERS:
            if name in filters and (record[name] or "").casefold() != filters[name]:
                return False
        checks = (("year_min", "release_year", 1), ("year_max", "release_year", -1), ("bpm_min", "tempo_bpm", 1),
                  ("bpm_max", "tempo_bpm", -1), ("energy_min", "energy", 1), ("energy_max", "energy", -1))
        for name, field_name, sign in checks:
            if name in filters:
                value = record[field_name]
                if value is None or (value - filters[name]) * sign < 0:
                    return False
        return True
    matches = [record for record in matches if keep(record)]

    if query.relative:
        field_name, direction, reference_rank = query.relative
        reference = by_rank.get(reference_rank)
        if reference is None or reference[field_name] is None:
            notes.append(f"There is no song #{reference_rank} in the current playlist to compare with.")
            matches = []
        else:
            matches = [r for r in matches if r["id"] != reference["id"] and r[field_name] is not None
                       and ((r[field_name] < reference[field_name]) if direction == "lower" else (r[field_name] > reference[field_name]))]
            if not query.sort_by:
                query = replace(query, sort_by=field_name, descending=direction == "higher")

    if query.sort_by:
        matches.sort(key=lambda r: (r[query.sort_by] is None, -(r[query.sort_by] or 0) if query.descending else (r[query.sort_by] or 0), r["id"]))
    elif matches and matches[0]["rank"]:
        matches.sort(key=lambda r: (r["rank"] or 10**6, r["id"]))
    return matches[: query.limit], len(matches), notes


# ----- deterministic answers -----

def _label(record: dict[str, Any]) -> str:
    prefix = f"#{record['rank']} " if record["rank"] else ""
    return f"{prefix}{record['title']} by {record['artist']}"


def _describe(record: dict[str, Any], fields: Sequence[str]) -> str:
    parts = []
    for name in fields:
        value = record.get(name)
        if name in ("title", "artist"):
            continue
        if value is None:
            parts.append(f"{name.replace('_', ' ')} unknown")
        elif name == "release_year":
            parts.append(f"released in {value}")
        elif name == "tempo_bpm":
            parts.append(f"{value} BPM")
        elif name == "energy":
            parts.append(f"energy {value:.2f}")
        elif name == "album":
            parts.append(f"from the album “{value}”")
        elif name == "language":
            parts.append(f"sung in {value} (estimated)")
        else:
            parts.append(f"{name} {value}")
    return ", ".join(parts)


def deterministic_answer(
    query: Query, records: Sequence[dict[str, Any]], total: int, reference: dict[str, Any] | None = None
) -> str:
    fields = query.fields or ("album", "release_year", "genre", "tempo_bpm", "energy")
    if query.relative and reference is not None:
        field_name, direction, _ = query.relative
        word = {"energy": ("quieter", "more energetic"), "tempo_bpm": ("slower", "faster"),
                "release_year": ("older", "newer")}[field_name][direction == "higher"]
        header = f"{total} song{'s are' if total != 1 else ' is'} {word} than {_label(reference)} ({_describe(reference, [field_name])})"
        if not records:
            return header + "."
        lines = [f"- {_label(r)}: {_describe(r, [field_name])}" for r in records]
        more = f"\n…and {total - len(records)} more." if total > len(records) else ""
        return header + ":\n" + "\n".join(lines) + more
    if query.intent == "count":
        scope = "in this catalog" if query.scope == "catalog" else "in this playlist"
        return f"{total} song{'s' if total != 1 else ''} {scope} match."
    if not records:
        return "No songs match that."
    if len(records) == 1:
        return f"{_label(records[0])}: {_describe(records[0], fields)}."
    lines = [f"- {_label(r)}: {_describe(r, fields)}" for r in records]
    more = f"\n…and {total - len(records)} more." if total > len(records) else ""
    return f"{total} song{'s' if total != 1 else ''} match:\n" + "\n".join(lines) + more


# ----- verification of AI answers -----

_NUMBER = re.compile(r"(?<![\w.#])(\d+(?:\.\d+)?)(?![\w.])")


def verify_ai_answer(
    text: str,
    cited_ids: Sequence[int],
    records: Sequence[dict[str, Any]],
    total: int,
    reference: dict[str, Any] | None = None,
    required_fields: Sequence[str] = (),
    intent: str = "lookup",
) -> str | None:
    """Return None when the answer is grounded, complete, and on-topic, or a reason it was rejected.

    Grounded: every cited ID was retrieved (or is the comparison's reference song) and every
    number quoted appears in those records. Complete: every retrieved song is cited, so an
    answer cannot pass by quietly answering a different, narrower question. On-topic: the
    values the question asked for (album, BPM, year, the count...) actually appear.
    """
    context = [*records, *([reference] if reference else [])]
    allowed_ids = {r["id"] for r in context}
    if not text.strip():
        return "the answer was empty"
    unknown = [i for i in cited_ids if i not in allowed_ids]
    if unknown:
        return f"it cited song ID {unknown[0]}, which was not retrieved"
    missing = [r["id"] for r in records if r["id"] not in set(cited_ids)]
    if missing:
        return f"it left out {len(missing)} of the {len(records)} matching songs"
    lowered = text.casefold()
    if intent == "count":
        # For "how many" the count is the answer; per-song details are not required.
        return None if str(total) in lowered else f"it did not state the count ({total})"
    for record in records:
        for name in required_fields:
            value = record.get(name)
            if value is None or name in ("title", "artist"):
                continue
            accepted = {f"{value:.2f}", f"{value:.1f}"} if name == "energy" else {str(value).casefold()}
            if not any(option.casefold() in lowered for option in accepted):
                return f"it did not give the {name.replace('_', ' ')} that was asked for"
    allowed_numbers: set[str] = {str(total), str(len(records))}
    for r in context:
        allowed_numbers.update(str(v) for v in (r["id"], r["rank"], r["release_year"], r["tempo_bpm"]) if v is not None)
        allowed_numbers.update({f"{r['energy']:.2f}", f"{r['energy']:.1f}", str(r["energy"])})
        for text_field in ("title", "artist", "album"):
            allowed_numbers.update(_NUMBER.findall(str(r[text_field] or "")))
    for number in _NUMBER.findall(text):
        if number not in allowed_numbers:
            return f"it quoted the number {number}, which is not in the retrieved records"
    return None


# ----- offline question parser -----

_RANK = re.compile(r"(?:#\s*|\b(?:song|track|number|no\.?)\s*#?\s*)(\d{1,2})\b")
_COMPARE_NUMBER = re.compile(r"\b(over|above|more than|faster than|at least|under|below|less than|slower than|at most)\s+(\d{2,3}(?:\.\d+)?)\s*(bpm)?")
_RELATIVE = re.compile(r"\b(quieter|calmer|softer|louder|more energetic|faster|slower|newer|older)\s+than\s+(?:#\s*|song\s*|track\s*)?(\d{1,2})\b")


def parse_question_offline(question: str, *, genres: set[str], moods: set[str], languages: set[str]) -> dict[str, Any]:
    text = question.casefold()
    raw: dict[str, Any] = {"intent": "lookup", "scope": "playlist", "filters": {}, "fields": []}
    if any(term in text for term in UNSUPPORTED_TERMS):
        raw["intent"] = "unsupported"
        return raw
    if any(term in text for term in ("catalog", "do you have", "in total", "available", "library")):
        raw["scope"] = "catalog"
    if text.startswith("how many") or "how many" in text or "count" in text:
        raw["intent"] = "count"

    relative = _RELATIVE.search(text)
    if relative:
        word, rank = relative.group(1), int(relative.group(2))
        field_name, direction = {
            "quieter": ("energy", "lower"), "calmer": ("energy", "lower"), "softer": ("energy", "lower"),
            "louder": ("energy", "higher"), "more energetic": ("energy", "higher"),
            "faster": ("tempo_bpm", "higher"), "slower": ("tempo_bpm", "lower"),
            "newer": ("release_year", "higher"), "older": ("release_year", "lower"),
        }[word]
        raw["relative"] = {"field": field_name, "direction": direction, "reference_rank": rank}
        raw["intent"] = "filter"
        raw["fields"] = [field_name]
    else:
        ranks = [int(n) for n in _RANK.findall(text)]
        if ranks:
            raw["target_ranks"] = ranks[:3]

    for match in _COMPARE_NUMBER.finditer(text):
        word, number = match.group(1), float(match.group(2))
        is_min = word in ("over", "above", "more than", "faster than", "at least")
        if match.group(3) or number >= 40:
            raw["filters"]["bpm_min" if is_min else "bpm_max"] = number
            raw["fields"].append("tempo_bpm")
        raw["intent"] = "count" if raw["intent"] == "count" else "filter"

    for language in sorted(languages):
        if language in text or (language == "spanish" and ("español" in text or "latin" in text)):
            raw["filters"]["language"] = language
            raw["fields"].append("language")
            raw["intent"] = "count" if raw["intent"] == "count" else "filter"
            break
    for genre in sorted(genres, key=len, reverse=True):
        if re.search(rf"\b{re.escape(genre)}\b", text):
            raw["filters"]["genre"] = genre
            raw["intent"] = "count" if raw["intent"] == "count" else "filter"
            break
    for mood in sorted(moods, key=len, reverse=True):
        if re.search(rf"\b{re.escape(mood)}\b", text):
            raw["filters"]["mood"] = mood
            raw["intent"] = "count" if raw["intent"] == "count" else "filter"
            break
    if "recent" in text or "newest" in text or "latest" in text:
        raw["sort_by"], raw["order"] = "release_year", "desc"

    for field_name, terms in FIELD_TERMS.items():
        if any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms):
            raw["fields"].append(field_name)

    if raw["intent"] == "lookup" and not raw.get("target_ranks"):
        quoted = re.search(r"[\"“']([^\"”']{2,60})[\"”']", question)
        about = re.search(r"\b(?:about|is|by)\s+(.{3,60}?)[?.!]*$", question, re.I)
        target = quoted.group(1) if quoted else (about.group(1) if about else "")
        if target and target.casefold() not in ("it", "this", "that", "the playlist"):
            raw["target_text"] = target
    return raw


def mentioned_title(question: str, index: Sequence[dict[str, Any]]) -> str:
    """The longest catalog title (4+ characters) that appears verbatim in the question, if any."""
    text = question.casefold()
    titles = sorted({r["title"] for r in index if len(r["title"]) >= 4}, key=len, reverse=True)
    for title in titles:
        if re.search(rf"(?<!\w){re.escape(title.casefold())}(?!\w)", text):
            return title
    return ""


# ----- orchestration -----

def answer_question(
    question: str,
    *,
    catalog: Sequence[Mapping[str, Any]],
    playlist_songs: Sequence[Mapping[str, Any]] = (),
    gemini: Any | None = None,
    use_gemini: bool = False,
) -> ChatAnswer:
    """Answer one question. ``gemini`` must expose parse_question() and answer_from_records()."""
    question = question.strip()[:MAX_QUESTION_CHARS]
    if not question:
        return ChatAnswer(text="Ask a question about the songs.", source="Catalog lookup")
    index = build_index(catalog, playlist_songs)
    genres = {r["genre"].casefold() for r in index}
    moods = {r["mood"].casefold() for r in index}
    languages = {r["language"] for r in index if r["language"]}
    warnings: list[str] = []
    gemini_on = bool(use_gemini and gemini is not None)

    raw = None
    if gemini_on:
        try:
            raw = gemini.parse_question(
                question, genres=sorted(genres), moods=sorted(moods), languages=sorted(languages),
                has_playlist=bool(playlist_songs),
            )
        except Exception as exc:  # external boundary
            warnings.append(f"Gemini could not read the question ({type(exc).__name__}); used the local parser.")
            gemini_on = False
    if raw is None:
        raw = parse_question_offline(question, genres=genres, moods=moods, languages=languages)
    query, query_warnings = validate_query(raw)
    warnings.extend(query_warnings)
    if query.intent == "lookup" and not (query.target_ranks or query.target_text or query.filters or query.relative):
        named = mentioned_title(question, index)
        if named:
            query = replace(query, target_text=named)

    if query.intent == "unsupported":
        return ChatAnswer(text=REFUSAL, source="Catalog lookup", warnings=tuple(warnings))
    records, total, notes = retrieve(query, index)
    if notes:
        return ChatAnswer(text=" ".join(notes), source="Catalog lookup", warnings=tuple(warnings))
    if not records and query.intent != "count" and not query.relative:
        if query.target_text:
            return ChatAnswer(text=f"I couldn't find a song matching “{query.target_text}”.", source="Catalog lookup",
                              warnings=tuple(warnings))
        return ChatAnswer(text="No songs match that.", source="Catalog lookup", warnings=tuple(warnings))

    reference = None
    if query.relative:
        reference = next((r for r in index if r["rank"] == query.relative[2]), None)
    fallback = deterministic_answer(query, records, total, reference)
    cited = tuple(r["id"] for r in records)
    if gemini_on and records:
        try:
            reply = gemini.answer_from_records(question, records=records, total_matches=total, reference=reference)
            text = str(reply.get("answer", "")).strip()
            cited_ids = tuple(int(i) for i in reply.get("cited_ids", []) if isinstance(i, int))
            problem = verify_ai_answer(text, cited_ids, records, total, reference, query.fields, query.intent)
            if problem is None:
                return ChatAnswer(text=text, source="AI answer", records=tuple(records), cited_ids=cited_ids,
                                  total_matches=total, warnings=tuple(warnings))
            warnings.append(f"The AI answer was discarded because {problem}; showing a catalog lookup instead.")
        except Exception as exc:  # external boundary
            warnings.append(f"Gemini could not write an answer ({type(exc).__name__}); showing a catalog lookup.")
    return ChatAnswer(text=fallback, source="Catalog lookup", records=tuple(records), cited_ids=cited,
                      total_matches=total, warnings=tuple(warnings))
