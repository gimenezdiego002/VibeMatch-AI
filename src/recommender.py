import csv
from dataclasses import asdict, dataclass
import math
from typing import Dict, List, Tuple

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top k songs ranked for a user profile."""
        user_prefs = asdict(user)
        return sorted(
            self.songs,
            key=lambda song: score_song(user_prefs, asdict(song))[0],
            reverse=True,
        )[:max(0, k)]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Explain which song attributes contributed to its score."""
        _, reasons = score_song(asdict(user), asdict(song))
        return "; ".join(reasons)

def load_songs(csv_path: str) -> List[Dict]:
    """Load and validate songs from CSV, converting numeric fields."""
    songs = []
    seen_ids = set()
    float_fields = {
        "energy",
        "valence",
        "danceability",
        "acousticness",
    }

    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        for line_number, row in enumerate(csv.DictReader(csv_file), start=2):
            try:
                row["id"] = int(row["id"])
                row["tempo_bpm"] = int(row["tempo_bpm"])
                for field in float_fields:
                    row[field] = float(row[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric song data on CSV line {line_number}.") from exc
            if row["id"] in seen_ids:
                raise ValueError(f"Duplicate song id {row['id']} on CSV line {line_number}.")
            if not 40 <= row["tempo_bpm"] <= 220:
                raise ValueError(f"tempo_bpm must be between 40 and 220 on CSV line {line_number}.")
            for field in float_fields:
                if not math.isfinite(row[field]) or not 0.0 <= row[field] <= 1.0:
                    raise ValueError(f"{field} must be between 0 and 1 on CSV line {line_number}.")
            seen_ids.add(row["id"])
            songs.append(row)

    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score one song and return both its numeric score and explanations."""
    score = 0.0
    reasons = []

    favorite_genre = user_prefs.get("favorite_genre", user_prefs.get("genre", ""))
    favorite_mood = user_prefs.get("favorite_mood", user_prefs.get("mood", ""))
    target_energy = float(
        user_prefs.get("target_energy", user_prefs.get("energy", 0.5))
    )

    if str(song["genre"]).casefold() == str(favorite_genre).casefold():
        score += 3.0
        reasons.append("genre match (+3.00)")

    if str(song["mood"]).casefold() == str(favorite_mood).casefold():
        score += 2.0
        reasons.append("mood match (+2.00)")

    energy_similarity = max(0.0, 1.0 - abs(float(song["energy"]) - target_energy))
    energy_points = 2.0 * energy_similarity
    score += energy_points
    reasons.append(f"energy similarity (+{energy_points:.2f})")

    if "likes_acoustic" in user_prefs:
        acousticness = float(song["acousticness"])
        acoustic_fit = acousticness if user_prefs["likes_acoustic"] else 1.0 - acousticness
        score += acoustic_fit
        preference = "acoustic" if user_prefs["likes_acoustic"] else "non-acoustic"
        reasons.append(f"{preference} fit (+{acoustic_fit:.2f})")

    return score, reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Rank all songs by score and return the top k with explanations."""
    scored_songs = []

    for song in songs:
        score, reasons = score_song(user_prefs, song)
        explanation = "; ".join(reasons)
        scored_songs.append((song, score, explanation))

    return sorted(scored_songs, key=lambda result: result[1], reverse=True)[:max(0, k)]
