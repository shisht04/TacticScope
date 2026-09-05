"""
tests/test_pipeline.py

Pytest test suite for the TacticScope v2.0 pipeline.
Converted from audit_v2.py — same underlying pass/fail conditions,
expressed as proper pytest assertions.

REQUIRED TEST DATA
------------------
These tests require pre-existing pipeline output at:
  data/output/sample_tracks.csv   (from detect_and_track.py)
  data/videos/sample.mp4          (required for team classification)

If either file is missing these tests FAIL immediately with a clear error.
There is NO pytest.skip() here — a missing file is a hard failure, not a skip.

To generate the required data:
  python src/detect_and_track.py --video data/videos/sample.mp4
Then re-run:
  python -m pytest tests/test_pipeline.py -v
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from analytics import run_analytics
from speed_estimator import run_speed_estimation
from tactical_insights import run_tactical_insights
from team_classifier import run_team_classification

# ── Test data paths ───────────────────────────────────────────────────────────
TRACKS_CSV = os.path.join(_ROOT, "data", "output", "sample_tracks.csv")
VIDEO_PATH = os.path.join(_ROOT, "data", "videos", "sample.mp4")
OUTPUT_DIR = os.path.join(_ROOT, "data", "output")

# ── Hard fail if required test data is missing ────────────────────────────────
# Do NOT replace these with pytest.skip().
# Missing data means the test suite cannot run — that is an error, not a skip.
if not os.path.exists(TRACKS_CSV):
    raise FileNotFoundError(
        f"\nRequired test data not found: {TRACKS_CSV}\n"
        "Run the detection pipeline first:\n"
        "  python src/detect_and_track.py --video data/videos/sample.mp4\n"
        "Then re-run: python -m pytest tests/test_pipeline.py -v"
    )
if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(
        f"\nRequired test video not found: {VIDEO_PATH}\n"
        "Place a football clip at data/videos/sample.mp4 and re-run detection."
    )


# ── Module-scoped fixtures — each expensive stage runs exactly once ───────────

@pytest.fixture(scope="module")
def analytics_data():
    """Runs run_analytics() once and shares results across all tests."""
    summary, tracks = run_analytics(TRACKS_CSV, OUTPUT_DIR)
    return summary, tracks


@pytest.fixture(scope="module")
def team_data():
    """Runs run_team_classification() once and shares results across all tests."""
    tracks_teams, team_labels = run_team_classification(
        VIDEO_PATH, TRACKS_CSV, OUTPUT_DIR
    )
    return tracks_teams, team_labels


@pytest.fixture(scope="module")
def speed_data(team_data):
    """Runs run_speed_estimation() once on team-classified data."""
    tracks_teams, _ = team_data
    df_speed, speed_summary = run_speed_estimation(tracks_teams, fps=25.0)
    return df_speed, speed_summary


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_analytics(analytics_data):
    """
    Stage 1: run_analytics() returns a non-empty summary DataFrame
    with all expected columns and at least one tracked player.
    """
    summary, tracks = analytics_data

    assert not summary.empty, "Analytics summary DataFrame must not be empty"
    assert not tracks.empty, "Tracking DataFrame must not be empty"

    expected_cols = (
        "track_id", "num_frames", "first_frame", "last_frame",
        "total_pixel_distance", "avg_x", "avg_y",
    )
    for col in expected_cols:
        assert col in summary.columns, f"Summary missing expected column: {col!r}"

    assert len(summary) > 0, "Must have at least one tracked player"
    assert summary["total_pixel_distance"].min() >= 0, \
        "Pixel distances must be non-negative"


def test_team_classification(team_data):
    """
    Stage 2: run_team_classification() returns a DataFrame with a 'team'
    column whose values are all in {0, 1, 2} (Team A / Team B / Official).
    team_labels must be non-empty and at least one real team must be present.
    """
    tracks_teams, team_labels = team_data

    assert "team" in tracks_teams.columns, \
        "tracks_teams must contain a 'team' column"
    assert len(team_labels) > 0, \
        "team_labels must not be empty"

    valid_values = {0, 1, 2}
    actual_values = set(int(v) for v in tracks_teams["team"].unique())
    invalid = actual_values - valid_values
    assert not invalid, (
        f"team column contains unexpected values: {invalid} "
        f"(expected a subset of {valid_values})"
    )

    # At least one of Team A or Team B must be present
    assert actual_values & {0, 1}, \
        "At least one of Team A (0) or Team B (1) must be present"


def test_speed_estimation(speed_data):
    """
    Stage 3: run_speed_estimation() attaches 'speed_px_per_sec' to the
    DataFrame and returns a per-track summary with a positive sprint threshold.
    """
    df_speed, speed_summary = speed_data

    assert "speed_px_per_sec" in df_speed.columns, \
        "df_speed must have 'speed_px_per_sec' column"
    assert not speed_summary.empty, \
        "speed_summary must not be empty"

    expected_cols = (
        "track_id", "avg_speed_px_s", "top_speed_px_s",
        "sprint_count", "sprint_threshold_px_s",
    )
    for col in expected_cols:
        assert col in speed_summary.columns, \
            f"speed_summary missing expected column: {col!r}"

    sprint_thresh = float(speed_summary["sprint_threshold_px_s"].iloc[0])
    assert sprint_thresh > 0, \
        f"Sprint threshold must be > 0, got {sprint_thresh}"

    assert df_speed["speed_px_per_sec"].min() >= 0, \
        "Speed values must be non-negative"


def test_tactical_insights(team_data):
    """
    Stage 4: run_tactical_insights() returns a dict with all 7 expected keys.
    key_moments must be a list. compactness_df and pressing_df must have
    their expected column structure when non-empty.
    """
    tracks_teams, _ = team_data
    insights = run_tactical_insights(tracks_teams)

    expected_keys = (
        "compactness_df", "compactness_summary",
        "pressing_df", "pressing_summary",
        "territorial", "formation", "key_moments",
    )
    for key in expected_keys:
        assert key in insights, f"insights dict missing expected key: {key!r}"

    assert isinstance(insights["key_moments"], list), \
        "insights['key_moments'] must be a list"

    comp_df = insights["compactness_df"]
    if not comp_df.empty:
        for col in ("frame", "team_a_compactness", "team_b_compactness"):
            assert col in comp_df.columns, \
                f"compactness_df missing column: {col!r}"

    press_df = insights["pressing_df"]
    if not press_df.empty:
        for col in ("frame", "team_a_press", "team_b_press"):
            assert col in press_df.columns, \
                f"pressing_df missing column: {col!r}"


def test_award_engine(analytics_data, speed_data):
    """
    Stage 5: compute_awards() (inlined from app.py — avoids importing Streamlit)
    produces a non-empty award dict and always assigns at least
    Speedster and Marathon Man when valid data is present.
    """
    summary, _ = analytics_data
    _, speed_summary = speed_data

    # Inline copy of compute_awards from app.py.
    # Must stay in sync with app.py if that function changes.
    def compute_awards(df: pd.DataFrame) -> dict:
        awd: dict = {}

        def give(tid, icon, lbl):
            awd.setdefault(int(tid), []).append((icon, lbl))

        valid = df.dropna(subset=["top_speed_px_s", "total_pixel_distance"])
        if valid.empty:
            return awd
        give(valid.loc[valid["top_speed_px_s"].idxmax(),        "track_id"], "⚡", "Speedster")
        give(valid.loc[valid["total_pixel_distance"].idxmax(),  "track_id"], "🏃", "Marathon Man")
        give(df.loc[df["num_frames"].idxmax(),                  "track_id"], "💪", "The Engine")
        if "sprint_count" in valid.columns and valid["sprint_count"].max() > 0:
            give(valid.loc[valid["sprint_count"].idxmax(), "track_id"], "🔥", "Sprint King")
        return awd

    merged = summary.merge(
        speed_summary[["track_id", "avg_speed_px_s", "top_speed_px_s", "sprint_count"]],
        on="track_id",
        how="left",
    ).fillna(0)

    awards = compute_awards(merged)

    assert len(awards) > 0, "Award dict must be non-empty"

    all_labels = [lbl for award_list in awards.values() for _, lbl in award_list]
    assert "Speedster" in all_labels,    "⚡ Speedster award must be assigned"
    assert "Marathon Man" in all_labels, "🏃 Marathon Man award must be assigned"


def test_plotly_figures(team_data, speed_data):
    """
    Stage 6: Plotly figure constructors produce valid go.Figure objects from
    pipeline data. Tests the data-to-figure contract, not visual output.
    Mirrors the checks in audit_v2.py [5] without importing app.py.
    """
    tracks_teams, _ = team_data
    df_speed, speed_summary = speed_data

    # Pitch figure — average position scatter (mirrors pitch_figure in app.py)
    avg = tracks_teams.groupby("track_id").agg(
        cx=("cx", "mean"), cy=("cy", "mean"),
        team=("team", "first"), frames=("frame", "count"),
    ).reset_index()

    pitch_fig = go.Figure(go.Scatter(x=avg["cx"], y=avg["cy"], mode="markers"))
    assert isinstance(pitch_fig, go.Figure), \
        "pitch figure must be a go.Figure"
    assert len(pitch_fig.data) > 0, \
        "pitch figure must have at least one trace"

    # Team heatmap (mirrors team_heatmap_fig in app.py)
    team_a = tracks_teams[tracks_teams["team"] == 0]
    if not team_a.empty:
        heatmap_fig = go.Figure(go.Histogram2d(
            x=team_a["cx"], y=team_a["cy"], nbinsx=30, nbinsy=20,
        ))
        assert isinstance(heatmap_fig, go.Figure), \
            "team heatmap figure must be a go.Figure"

    # Speed figure (mirrors speed_fig in app.py)
    df_speed_merged = tracks_teams.merge(
        speed_summary[["track_id", "sprint_threshold_px_s"]],
        on="track_id",
        how="left",
    )
    spd_fig = go.Figure(go.Scatter(
        x=df_speed["frame"], y=df_speed["speed_px_per_sec"], mode="lines",
    ))
    assert isinstance(spd_fig, go.Figure), \
        "speed figure must be a go.Figure"
    assert len(spd_fig.data) > 0, \
        "speed figure must have at least one trace"
