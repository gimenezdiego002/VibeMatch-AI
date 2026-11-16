"""VibeMatch AI Streamlit dashboard.

Run with:  streamlit run streamlit_app.py

This file is presentation only. Interpretation, validation, scoring, selection, ordering, and
explanation all run through ``src.app.create_playlist`` and the modules it calls. Gemini is
reached only through ``GeminiService``. Covers, links, and previews come from catalog fields
and the helpers in ``src.media``; they never influence which songs are chosen.
"""

from __future__ import annotations

from datetime import datetime
from html import escape
import os
from pathlib import Path

import altair as alt
import streamlit as st

from src import env as local_env
from src.app import DEFAULT_CATALOG, create_playlist
from src.dashboard_data import (
    card_meta_html,
    catalog_dataframe,
    energy_meter_html,
    phase_bands,
    title_html,
    catalog_summary,
    contributions_dataframe,
    describe_energy_shape,
    energy_chart_data,
    explanation_source,
    interpretation_source,
    is_real_catalog,
    placeholder_tile_html,
    playlist_csv,
    length_notes,
    playlist_dataframe,
    result_weights,
    song_contributions,
    system_checks,
)
from src.gemini_service import GeminiService
from src.media import cover_url, fetch_preview_url, origin_note, spotify_embed_url, spotify_url
from src.recommender import load_songs


ROOT = Path(__file__).resolve().parent
CSS_FILE = ROOT / "assets" / "vibematch.css"  # The only custom CSS. Safe to delete; see the file header.
REAL_CATALOG = ROOT / "data" / "real_songs.csv"
CATALOGS = {
    "fictional": {"label": "Fictional classroom songs", "path": DEFAULT_CATALOG},
    "real": {"label": "Real music", "path": REAL_CATALOG},
}
AVAILABLE_CATALOGS = [key for key, meta in CATALOGS.items() if Path(meta["path"]).is_file()]

PRESETS = {
    "workout": "Make a 10-song high-energy gym workout with a warm-up and cooldown",
    "studying": "Give me 10 relaxing low-energy songs for studying with a steady feel",
    "relaxing": "Create 8 peaceful songs for relaxing after work",
    "driving": "Build a 10-song confident playlist for a long evening drive",
    "running": "I need 12 fast songs for running that start slow and finish with a cooldown",
    "party": "Make a 15-song euphoric party playlist that builds up the dance floor",
}
PRESET_LABELS = {
    "workout": ":material/fitness_center: Workout",
    "studying": ":material/menu_book: Studying",
    "relaxing": ":material/spa: Relaxing",
    "driving": ":material/directions_car: Driving",
    "running": ":material/directions_run: Running",
    "party": ":material/celebration: Party",
}
STRUCTURE_NOTES = {
    "warmup_peak_cooldown": "Warm-up across the first quarter, peak through the middle, cooldown in the final quarter.",
    "build": "Energy rises steadily from a lower start up to the target.",
    "consistent": "Energy stays steady around the target, so order follows compatibility score.",
    "balanced": "Energy alternates gently around the target to keep variety.",
}
CARD_WIDTH = 300  # native fallback width; the CSS grid makes cards fluid


# ---------------------------------------------------------------------------
# Configuration, styling, and cached data
# ---------------------------------------------------------------------------

st.set_page_config(page_title="VibeMatch AI", page_icon=":material/queue_music:", layout="wide")
if CSS_FILE.is_file():
    st.html(CSS_FILE)  # Streamlit wraps a .css path in <style> tags; no unsafe HTML is involved.


@st.cache_data(show_spinner=False, max_entries=4)
def load_catalog(path: str) -> list[dict]:
    return load_songs(path)


@st.cache_data(show_spinner=False, ttl=600, max_entries=128)
def cached_preview(isrc: str, title: str, artist: str) -> tuple[str, str] | None:
    """Fresh preview lookup. Cached briefly because Deezer preview links expire."""
    return fetch_preview_url({"isrc": isrc, "title": title, "artist": artist})


def read_secret(name: str) -> str | None:
    """Read an optional value from st.secrets without failing when no secrets file exists."""
    try:
        value = st.secrets.get(name)
    except Exception:  # No secrets.toml, or an unreadable one: treat as not configured.
        return None
    return str(value).strip() if value else None


local_env.load_local_env()  # Optional git-ignored .env; real environment variables take precedence.
api_key = read_secret("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY", "").strip() or None
gemini_model = read_secret("GEMINI_MODEL") or os.getenv("GEMINI_MODEL", "").strip() or None
key_configured = api_key is not None
sdk_installed = GeminiService.sdk_installed()
gemini_ready = key_configured and sdk_installed

# Session state is initialized in one place.
st.session_state.setdefault("request_text", "")
st.session_state.setdefault("use_gemini", gemini_ready)
st.session_state.setdefault("catalog_choice", "fictional")
st.session_state.setdefault("last_run", None)
st.session_state.setdefault("now_playing", 0)
if not gemini_ready:
    st.session_state.use_gemini = False
if st.session_state.catalog_choice not in AVAILABLE_CATALOGS:
    st.session_state.catalog_choice = "fictional"


def apply_preset() -> None:
    choice = st.session_state.get("preset")
    if choice:
        st.session_state.request_text = PRESETS[choice]
    st.session_state.preset = None  # Allow the same preset to be chosen again.


def select_track(index: int) -> None:
    st.session_state.now_playing = index


run = st.session_state.last_run
result = run["result"] if run else None
run_catalog = load_catalog(str(CATALOGS[run["catalog_key"]]["path"])) if run else None
now_index = min(st.session_state.now_playing, len(result.songs) - 1) if result and result.songs else 0


def song_cover(song, *, alt_text: str) -> None:
    """Cover art when the row has one, otherwise a generated tile."""
    url = cover_url(song)
    if url:
        st.image(url, width="stretch", alt=alt_text)
    else:
        st.html(placeholder_tile_html(song))


def title_block(song) -> None:
    st.html(title_html(song))


# ---------------------------------------------------------------------------
# Sidebar: now playing (stays visible while scrolling a long playlist)
# ---------------------------------------------------------------------------

with st.sidebar, st.container(key="vm-nowplaying"):
    st.html('<div class="vm-eyebrow">Now playing</div>')
    if result is None or not result.songs:
        st.markdown("**Nothing queued yet**")
        st.caption("Generate a playlist, then press Play on any song to listen here.")
    else:
        playing = result.songs[now_index]
        song = playing.song
        song_cover(song, alt_text=f"Album cover for {song['title']} by {song['artist']}")
        title_block(song)
        st.caption(f"Track {now_index + 1} of {len(result.songs)} · score {playing.score:.0f}/100")
        note = origin_note(song)
        if note:
            st.markdown(note[0])
            st.caption(f"Source: {note[1]}. Built from catalog fields, not AI-written.")
        embed, link = spotify_embed_url(song), spotify_url(song)
        if link:
            st.link_button("Open in Spotify", link, icon=":material/open_in_new:", width="stretch")
        if embed:
            st.iframe(embed, height=152, alt=f"Spotify player for {song['title']}")
            preview = cached_preview(str(song.get("isrc", "")), str(song["title"]), str(song["artist"]))
            if preview:
                st.audio(preview[0], format="audio/mpeg" if preview[1] == "Deezer" else "audio/mp4", alt=f"30-second preview of {song['title']}")
                st.caption(f"30-second preview from {preview[1]}.")
            else:
                st.caption("No separate preview was found for this track.")
        else:
            st.caption("This is a fictional song, so it has no audio, cover art, or links.")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

with st.container(key="vm-hero"):
    st.html('<div class="vm-eyebrow">Context-aware playlists · <b>AI-assisted, Python-verified</b></div>')
    st.title("VibeMatch AI", anchor=False)
    st.caption(
        "Describe a situation in plain language. Gemini can interpret the request and explain the picks, "
        "but Python validates the profile and selects, scores, and orders every song from a verified local catalog."
    )
    with st.container(horizontal=True, vertical_alignment="center"):
        if gemini_ready:
            st.badge("Gemini available: API key configured", icon=":material/check_circle:", color="green")
            gemini_note = "A live Gemini call is confirmed only after a playlist is generated."
        elif key_configured:
            st.badge("Gemini unavailable: SDK not installed", icon=":material/error:", color="orange")
            gemini_note = "Install `google-genai` from requirements.txt. Until then the local fallback is used."
        else:
            st.badge("Gemini unavailable: no API key", icon=":material/cloud_off:", color="gray")
            gemini_note = "The dashboard runs fully offline with the local keyword interpreter and deterministic reasons."
        if st.session_state.catalog_choice == "real":
            st.badge("Real music catalog", icon=":material/library_music:", color="blue")
        else:
            st.badge("Fictional classroom catalog", icon=":material/science:", color="violet")
        st.caption(gemini_note)


# ---------------------------------------------------------------------------
# Request form
# ---------------------------------------------------------------------------

with st.container(border=True, key="vm-composer"):
    st.subheader("What are you listening for?", anchor=False)
    st.pills(
        "Example requests",
        options=list(PRESETS),
        format_func=PRESET_LABELS.get,
        key="preset",
        on_change=apply_preset,
        help="Fills the request box. You can edit it before generating.",
    )
    with st.form("playlist_form", border=False):
        st.text_area(
            "Playlist request",
            key="request_text",
            height=100,
            max_chars=500,
            placeholder="For example: 10 calm lofi songs for a late-night study session",
        )
        with st.container(horizontal=True, vertical_alignment="center"):
            submitted = st.form_submit_button(
                "Generate playlist", type="primary", icon=":material/auto_awesome:"
            )
            st.toggle(
                "Use Gemini",
                key="use_gemini",
                disabled=not gemini_ready,
                help=(
                    "When on, Gemini interprets the request and explains the final picks. Python still does all selection."
                    if gemini_ready
                    else "Unavailable: configure GEMINI_API_KEY in .streamlit/secrets.toml or the environment "
                    "and install google-genai. The local fallback will be used."
                ),
            )
            st.segmented_control(
                "Catalog",
                options=AVAILABLE_CATALOGS,
                format_func=lambda key: CATALOGS[key]["label"],
                key="catalog_choice",
                required=True,
                disabled=len(AVAILABLE_CATALOGS) < 2,
                help=(
                    "Real music uses tracks, covers, and audio features from public music APIs. "
                    "Run scripts/build_real_catalog.py to create it."
                    if len(AVAILABLE_CATALOGS) < 2
                    else "Fictional songs are synthetic classroom data. Real music has covers, links, and previews."
                ),
            )

    if submitted:
        request = st.session_state.request_text.strip()
        if not request:
            st.warning("Enter a playlist request before generating.", icon=":material/edit_note:")
        else:
            use_gemini = bool(st.session_state.use_gemini) and gemini_ready
            catalog_key = st.session_state.catalog_choice
            service = GeminiService(api_key=api_key, model=gemini_model)
            spinner_text = "Interpreting with Gemini and selecting songs…" if use_gemini else "Selecting songs locally…"
            with st.spinner(spinner_text, show_time=True):
                try:
                    result = create_playlist(
                        request,
                        catalog_path=CATALOGS[catalog_key]["path"],
                        offline=not use_gemini,
                        gemini=service,
                    )
                except (ValueError, FileNotFoundError) as exc:
                    st.error(f"Could not generate a playlist: {exc}", icon=":material/error:")
                else:
                    st.session_state.last_run = {
                        "request": request,
                        "gemini_requested": use_gemini,
                        "catalog_key": catalog_key,
                        "result": result,
                        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    st.session_state.now_playing = 0
                    st.rerun()  # Redraw the sidebar and results with the new playlist.


# ---------------------------------------------------------------------------
# AI / fallback status and inferred profile
# ---------------------------------------------------------------------------

if result is not None:
    st.header(result.name, anchor=False)
    st.caption(
        f"Request: “{run['request']}” · {CATALOGS[run['catalog_key']]['label'].lower()} · generated {run['generated_at']}"
    )

    interp = interpretation_source(result)
    expl = explanation_source(result)
    with st.container(horizontal=True):
        st.metric(
            "Request interpretation",
            interp,
            border=True,
            help="Gemini, or the local keyword interpreter. Either way, Python validates the profile.",
        )
        st.metric(
            "Explanations",
            expl,
            border=True,
            help="Gemini-written sentences (IDs validated) or deterministic score reasons.",
        )
        shortage_notes = length_notes(result)
        st.metric(
            "Songs returned",
            f"{len(result.songs)} of {result.profile.playlist_length}",
            border=True,
            help=" ".join(shortage_notes) if shortage_notes else "Every requested song was returned.",
        )
    if shortage_notes:
        st.warning(" ".join(shortage_notes), icon=":material/playlist_remove:", title="Playlist length adjusted")

    if result.used_gemini_interpretation and result.used_gemini_explanations:
        st.success(
            "Gemini interpreted the request and explained the picks. Python selected, scored, and ordered every song.",
            icon=":material/verified:",
        )
    elif run["gemini_requested"]:
        st.warning(
            "Gemini was requested but did not complete every step. Steps marked *Local fallback* or "
            "*Deterministic reasons* were produced without AI. See the warnings below.",
            icon=":material/sync_problem:",
            title="Partial or full fallback",
        )
    else:
        st.info(
            "Generated entirely with the local fallback: keyword interpretation and deterministic score reasons. "
            "No AI model was used for this playlist.",
            icon=":material/offline_bolt:",
        )

    if result.warnings:
        with st.expander(f"Warnings ({len(result.warnings)})", icon=":material/warning:", expanded=run["gemini_requested"]):
            for warning in result.warnings:
                st.markdown(f"- {warning}")

    profile = result.profile
    with st.container(border=True):
        st.subheader("Inferred preference profile", anchor=False)
        st.caption("Validated and clamped by Python before any song was scored.")
        with st.container(horizontal=True):
            st.metric("Activity", profile.activity.title(), border=True)
            st.metric("Playlist length", profile.playlist_length, border=True)
            st.metric("Target mood", profile.target_mood, border=True)
            st.metric("Target energy", f"{profile.target_energy:.2f}", border=True)
            st.metric("Target tempo", f"{profile.target_tempo_bpm:.0f} BPM", border=True)
            st.metric("Structure", profile.playlist_structure.replace("_", " "), border=True)
            if profile.language:
                st.metric(
                    "Language",
                    f"{profile.language.title()} ({'only' if profile.language_strict else 'preferred'})",
                    border=True,
                    help="'only' filters the catalog; 'preferred' adds score. Song languages are estimates.",
                )
            if profile.has_year_range:
                low, high = profile.min_release_year, profile.max_release_year
                years = f"{low}–{high}" if low and high else f"{low} or later" if low else f"{high} or earlier"
                st.metric("Release years", f"{years} ({'only' if profile.recency_strict else 'preferred'})", border=True)
        genres = " ".join(f":blue-badge[{genre}]" for genre in profile.preferred_genres)
        st.markdown(f"**Preferred genres:** {genres or 'none matched; genre is scored neutrally'}")
        for note in profile.notes:
            st.caption(f":material/tune: {note}")


# ---------------------------------------------------------------------------
# Detail tabs
# ---------------------------------------------------------------------------

tab_playlist, tab_energy, tab_scores, tab_checks, tab_catalog = st.tabs(
    [
        ":material/queue_music: Playlist",
        ":material/show_chart: Energy flow",
        ":material/analytics: Score details",
        ":material/shield: System checks",
        ":material/library_music: Catalog",
    ]
)

EMPTY_HINT = "Generate a playlist to see results here."
NO_SONGS_HINT = "No songs matched this request. See the note above for why, then try a broader request."

with tab_playlist:
    if result is None or not result.songs:
        st.info(EMPTY_HINT if result is None else NO_SONGS_HINT, icon=":material/info:")
    else:
        playlist_df = playlist_dataframe(result)
        st.html(f'<div class="vm-eyebrow">{len(result.songs)} tracks · ordered for {escape(result.profile.activity)}</div>')
        st.subheader("Your playlist", anchor=False)
        st.caption(f"Explanation source: {explanation_source(result)}. Press Play to listen in the sidebar.")
        songs = result.songs
        # One wrapping container: the CSS turns it into a responsive grid (3 columns on a laptop,
        # 1 on a narrow window). Without the CSS it still wraps natively at a fixed card width.
        with st.container(key="vm-grid", horizontal=True, wrap=True, gap="medium"):
            for index, item in enumerate(songs):
                song = item.song
                active = index == now_index
                with st.container(border=True, width=CARD_WIDTH, key=f"vmcard-{'active-' if active else ''}{index + 1}"):
                    song_cover(song, alt_text=f"Album cover for {song['title']} by {song['artist']}")
                    title_block(song)
                    st.html(card_meta_html(index + 1, item.score, song["genre"], song["mood"]))
                    st.html(energy_meter_html(song["energy"], item.position_target_energy, song["tempo_bpm"]))
                    st.markdown(item.explanation)
                    note = origin_note(song)
                    if note:
                        st.caption(f"{note[0]} (Source: {note[1]})")
                    with st.container(horizontal=True):
                        st.button(
                            "Playing" if active else "Play",
                            key=f"play-{index + 1}",
                            icon=":material/graphic_eq:" if active else ":material/play_arrow:",
                            type="primary" if active else "secondary",
                            on_click=select_track,
                            args=(index,),
                        )
                        link = spotify_url(song)
                        if link:
                            st.link_button("Spotify", link, icon=":material/open_in_new:", key=f"open-{index + 1}")

        st.download_button(
            "Download playlist CSV",
            data=playlist_csv(result),
            file_name=f"vibematch_{result.profile.activity}_playlist.csv",
            mime="text/csv",
            icon=":material/download:",
            on_click="ignore",
        )
        with st.expander("Table view", icon=":material/table:"):
            st.dataframe(
                playlist_df,
                hide_index=True,
                alt=f"Ordered {len(playlist_df)}-song playlist with compatibility scores, energy, tempo, and explanations",
                column_config={
                    "rank": st.column_config.NumberColumn("Rank", width="small"),
                    "id": st.column_config.NumberColumn("ID", width="small"),
                    "title": st.column_config.TextColumn("Title"),
                    "artist": st.column_config.TextColumn("Artist"),
                    "genre": st.column_config.TextColumn("Genre", width="small"),
                    "mood": st.column_config.TextColumn("Mood", width="small"),
                    "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
                    "energy": st.column_config.ProgressColumn("Energy", min_value=0.0, max_value=1.0, format="%.2f"),
                    "target_energy": st.column_config.NumberColumn("Target energy", format="%.2f"),
                    "tempo_bpm": st.column_config.NumberColumn("Tempo", format="%d BPM"),
                    "explanation": st.column_config.TextColumn("Explanation", width="large"),
                },
            )

with tab_energy:
    if result is None or not result.songs:
        st.info(EMPTY_HINT if result is None else NO_SONGS_HINT, icon=":material/info:")
    else:
        st.caption(STRUCTURE_NOTES.get(result.profile.playlist_structure, ""))
        chart_df = energy_chart_data(result)
        count = len(result.songs)
        series = ["Actual song energy", "Target energy"]
        x_axis = alt.X(
            "position:Q",
            title="Playlist position",
            scale=alt.Scale(domain=[0.5, count + 0.5], nice=False),
            axis=alt.Axis(
                values=list(range(1, count + 1, max(1, -(-count // 25)))),  # every position, thinned past 25
                format="d",
                labelAngle=0,
                grid=False,
            ),
        )
        y_axis = alt.Y("energy:Q", title="Energy (0–1)", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(tickCount=5))
        series_color = alt.Color(
            "series:N",
            title=None,
            scale=alt.Scale(domain=series, range=["#9d8cff", "#ff9a6b"]),
            legend=alt.Legend(orient="top", direction="horizontal", symbolStrokeWidth=3),
        )
        tooltip = [
            alt.Tooltip("position:Q", title="Position", format="d"),
            alt.Tooltip("title:N", title="Song"),
            alt.Tooltip("series:N", title="Series"),
            alt.Tooltip("energy:Q", title="Energy", format=".2f"),
        ]
        actual = alt.Chart(chart_df).transform_filter(alt.datum.series == series[0])
        target = alt.Chart(chart_df).transform_filter(alt.datum.series == series[1])
        layers = []
        bands = phase_bands(result)
        if not bands.empty:
            band_base = alt.Chart(bands)
            layers.append(
                band_base.mark_rect(opacity=0.07, color="#f4f0ff").encode(
                    x="start:Q", x2="end:Q", opacity=alt.condition(alt.datum.phase == "Peak", alt.value(0.09), alt.value(0.035))
                )
            )
            layers.append(
                band_base.transform_calculate(mid="(datum.start + datum.end) / 2")
                .mark_text(dy=-6, baseline="bottom", fontSize=11, fontWeight=600, color="#b4abd3", y=0)
                .encode(x="mid:Q", text="phase:N")
            )
        layers += [
            actual.mark_area(
                interpolate="monotone",
                line=False,
                color=alt.Gradient(
                    gradient="linear",
                    stops=[
                        alt.GradientStop(color="rgba(157,140,255,0)", offset=0),
                        alt.GradientStop(color="rgba(157,140,255,0.38)", offset=1),
                    ],
                    x1=1, x2=1, y1=1, y2=0,
                ),
            ).encode(x=x_axis, y=y_axis),
            target.mark_line(interpolate="monotone", strokeDash=[6, 5], strokeWidth=2).encode(
                x=x_axis, y=y_axis, color=series_color, tooltip=tooltip
            ),
            actual.mark_line(interpolate="monotone", strokeWidth=3, strokeCap="round").encode(
                x=x_axis, y=y_axis, color=series_color, tooltip=tooltip
            ),
            actual.mark_circle(size=70, opacity=1, stroke="#0d0a1a", strokeWidth=2).encode(
                x=x_axis, y=y_axis, color=series_color, tooltip=tooltip
            ),
        ]
        chart = (
            alt.layer(*layers)
            .properties(height=360)
            .configure(background="transparent", font="Inter, system-ui, sans-serif")
            .configure_view(strokeWidth=0)
            .configure_axis(
                labelColor="#b4abd3",
                titleColor="#b4abd3",
                labelFontSize=12,
                titleFontSize=12,
                titleFontWeight=500,
                gridColor="rgba(255,255,255,0.07)",
                domainColor="rgba(255,255,255,0.18)",
                tickColor="rgba(255,255,255,0.18)",
            )
            .configure_legend(labelColor="#f4f0ff", labelFontSize=12)
        )
        st.altair_chart(chart, theme=None, alt=describe_energy_shape(result))

with tab_scores:
    if result is None or not result.songs:
        st.info(EMPTY_HINT if result is None else NO_SONGS_HINT, icon=":material/info:")
    else:
        weights = result_weights(result)
        weights_text = " + ".join(f"{weight:.2f} × {name}" for name, weight in weights.items())
        st.caption(
            f"Compatibility (0–100) = 100 × ({weights_text}). Each feature fit is between 0 and 1. "
            "Genre and mood give partial credit to close matches (for example latin and reggaeton). "
            "Weights come from defaults, adjusted only within safe limits when Gemini suggests priorities. "
            "Energy is compared with each position's target, so ordering never inflates the score."
        )
        contrib_df = contributions_dataframe(result)
        feature_columns = {
            feature: st.column_config.ProgressColumn(
                f"{feature.title()} pts", min_value=0.0, max_value=weight * 100, format="%.1f"
            )
            for feature, weight in weights.items()
        }
        st.dataframe(
            contrib_df,
            hide_index=True,
            alt="Score points contributed by energy, mood, genre, and tempo for each selected song",
            column_config={
                "rank": st.column_config.NumberColumn("Rank", width="small"),
                "title": st.column_config.TextColumn("Title"),
                **feature_columns,
                "score": st.column_config.NumberColumn("Total", format="%.1f"),
            },
        )

        rank = st.selectbox(
            "Inspect a song",
            options=list(range(1, len(result.songs) + 1)),
            format_func=lambda r: f"{r}. {result.songs[r - 1].song['title']} · {result.songs[r - 1].song['artist']}",
            key="inspect_rank",
        )
        item = result.songs[rank - 1]
        with st.container(border=True):
            st.markdown(f"**{item.song['title']}** · {item.song['artist']}")
            st.bar_chart(
                song_contributions(result, rank),
                x="feature",
                y=["earned_points", "max_points"],
                x_label="Points",
                y_label="Feature",
                horizontal=True,
                stack=False,
                height=220,
                alt=f"Earned versus maximum score points per feature for {item.song['title']}",
            )
            st.markdown(f"**Deterministic reason:** {item.deterministic_reason}")
            if result.used_gemini_explanations:
                st.markdown(f"**Gemini explanation:** {item.explanation}")
            else:
                st.caption("No Gemini explanation for this playlist; the deterministic reason is shown instead.")
        st.caption("Only final outputs are shown. Model reasoning is not requested or displayed.")

with tab_checks:
    if result is None:
        st.info(EMPTY_HINT, icon=":material/info:")
    else:
        st.caption("Guardrails re-verified against the local CSV after generation.")
        status_icons = {
            True: ":green[:material/check_circle:]",
            False: ":red[:material/cancel:]",
            None: ":blue[:material/info:]",
        }
        with st.container(border=True):
            for check in system_checks(result, run_catalog, gemini_requested=run["gemini_requested"]):
                st.markdown(f"{status_icons[check.passed]} **{check.name}**  \n{check.detail}")
            for warning in result.warnings:
                st.markdown(f":orange[:material/warning:] {warning}")

with tab_catalog:
    catalog_key = st.session_state.catalog_choice
    catalog = load_catalog(str(CATALOGS[catalog_key]["path"]))
    summary = catalog_summary(catalog)
    if is_real_catalog(catalog):
        st.caption(
            "Real tracks gathered once from public music APIs. Energy, tempo, valence, danceability, and acousticness "
            "come from ReccoBeats; covers, albums, and release dates from Deezer. Genre is mapped from Deezer. "
            "Mood is derived locally from valence and energy, so treat it as an approximation."
        )
    else:
        st.caption(
            "Every title, artist, and attribute in this catalog is fictional, synthetic classroom data. "
            "It is not a claim about real music."
        )
    with st.container(horizontal=True):
        st.metric("Songs", summary["songs"], border=True)
        st.metric("Genres", summary["genres"], border=True)
        st.metric("Moods", summary["moods"], border=True)
        st.metric("Energy range", f"{summary['energy_range'][0]:.2f}–{summary['energy_range'][1]:.2f}", border=True)
        st.metric("Tempo range", f"{summary['tempo_range'][0]}–{summary['tempo_range'][1]} BPM", border=True)

    catalog_df = catalog_dataframe(catalog)
    with st.container(horizontal=True):
        genre_filter = st.multiselect("Genre", sorted(catalog_df["genre"].unique()), key="catalog_genres", placeholder="All genres")
        mood_filter = st.multiselect("Mood", sorted(catalog_df["mood"].unique()), key="catalog_moods", placeholder="All moods")
        energy_filter = st.slider("Energy", 0.0, 1.0, (0.0, 1.0), step=0.05, key="catalog_energy")
    filtered = catalog_df[catalog_df["energy"].between(*energy_filter)]
    if genre_filter:
        filtered = filtered[filtered["genre"].isin(genre_filter)]
    if mood_filter:
        filtered = filtered[filtered["mood"].isin(mood_filter)]
    kind = "real" if is_real_catalog(catalog) else "fictional"
    st.caption(f"Showing {len(filtered)} of {len(catalog_df)} {kind} songs.")
    st.dataframe(
        filtered,
        hide_index=True,
        alt=f"Filterable table of the {kind} song catalog",
        column_config={
            "id": st.column_config.NumberColumn("ID", width="small"),
            "cover_url": st.column_config.ImageColumn("Cover", width="small"),
            "title": "Title",
            "artist": "Artist",
            "genre": "Genre",
            "mood": "Mood",
            "album": "Album",
            "release_date": "Released",
            "energy": st.column_config.ProgressColumn("Energy", min_value=0.0, max_value=1.0, format="%.2f"),
            "tempo_bpm": st.column_config.NumberColumn("Tempo", format="%d BPM"),
            "valence": st.column_config.NumberColumn("Valence", format="%.2f"),
            "danceability": st.column_config.NumberColumn("Danceability", format="%.2f"),
            "acousticness": st.column_config.NumberColumn("Acousticness", format="%.2f"),
        },
    )

with st.expander("Data sources and attribution", icon=":material/info:"):
    st.markdown(
        "- **Fictional catalog:** `data/songs.csv`, synthetic classroom data.\n"
        "- **Real music catalog:** built once by `scripts/build_real_catalog.py` from [ReccoBeats](https://reccobeats.com) "
        "(track discovery and audio features) and [Deezer](https://developers.deezer.com) (covers, albums, release dates, genres).\n"
        "- **Players and links:** [Spotify](https://open.spotify.com) embeds and links; 30-second previews fetched fresh from Deezer or Apple.\n"
        "- **Sourced notes:** the “where it's from” lines are assembled directly from catalog fields. Gemini does not write them.\n"
        "- All services are unofficial third-party data sources used for a classroom project. Their terms of use were not "
        "independently verified, and this project is not affiliated with Spotify, Deezer, Apple, or ReccoBeats."
    )
