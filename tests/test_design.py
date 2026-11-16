"""Design-system guards: HTML helpers escape input and stay accessible; the CSS file follows its rules."""

import re
from pathlib import Path

from src.app import create_playlist
from src.dashboard_data import card_meta_html, energy_meter_html, phase_bands, title_html


ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "assets" / "vibematch.css").read_text(encoding="utf-8")
CONFIG = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")


def test_html_helpers_escape_untrusted_text():
    evil = '<img src=x onerror="alert(1)">'
    for html in (
        title_html({"title": evil, "artist": evil}),
        card_meta_html(1, 88.4, evil, evil),
        energy_meter_html(0.5, 0.6, evil),
    ):
        assert "<img" not in html and "onerror=\"" not in html


def test_card_meta_shows_rank_and_rounded_score():
    html = card_meta_html(7, 88.6, "pop", "happy")
    assert ">#7<" in html and ">89<" in html and "pop · happy" in html


def test_energy_meter_is_accessible_and_clamped():
    html = energy_meter_html(1.7, -0.2, 120)
    assert 'role="meter"' in html and 'aria-label="Energy 1.70, target -0.20"' in html
    assert "width:100.0%" in html and "left:0.0%" in html  # visual position never leaves the track


def test_phase_bands_cover_every_position_for_warmup_playlists():
    result = create_playlist("Make a 12-song high-energy gym workout with a warm-up and cooldown", offline=True)
    bands = phase_bands(result)
    assert list(bands["phase"]) == ["Warm-up", "Peak", "Cooldown"]
    assert bands["start"].iloc[0] == 0.5 and bands["end"].iloc[-1] == len(result.songs) + 0.5
    assert all(bands["start"].iloc[1:].values == bands["end"].iloc[:-1].values)  # contiguous


def test_phase_bands_are_empty_for_steady_playlists():
    result = create_playlist("Give me 10 relaxing low-energy songs for studying with a steady feel", offline=True)
    assert phase_bands(result).empty


def test_css_follows_motion_and_accessibility_rules():
    assert "transition: all" not in CSS
    assert "<script" not in CSS.lower()
    assert CSS.count("{") == CSS.count("}")
    for required in ("prefers-reduced-motion", "prefers-reduced-transparency", "prefers-contrast", "(hover: hover)"):
        assert required in CSS
    # Only transform/opacity are ever transitioned.
    for rule in re.findall(r"transition:\s*([^;]+);", CSS):
        properties = {part.strip().split()[0] for part in rule.split(",")}
        assert properties <= {"transform", "opacity", "none"}, rule
    # No animations that would replay on every Streamlit rerun.
    assert "@keyframes" not in CSS and "animation:" not in CSS


def test_theme_and_css_share_the_same_palette():
    for color in ("#9d8cff", "#0d0a1a", "#f4f0ff"):
        assert color in CONFIG.lower() and color in CSS.lower()
