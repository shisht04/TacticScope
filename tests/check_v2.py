"""
Manual smoke-test script for the TacticScope v2.0 pipeline.
Run from the project root:
    python tests/check_v2.py

Requires pre-existing output data at data/output/sample_tracks.csv
and a video at data/videos/sample.mp4.
"""
import os
import sys

# Resolve project root regardless of where this script is invoked from
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

_CSV = os.path.join(_ROOT, "data", "output", "sample_tracks.csv")
_VID = os.path.join(_ROOT, "data", "videos", "sample.mp4")
_OUT = os.path.join(_ROOT, "data", "output")

from analytics import run_analytics
from team_classifier import run_team_classification
from speed_estimator import run_speed_estimation
from tactical_insights import run_tactical_insights
import plotly.graph_objects as go

print("--- v2.0 pipeline check ---")
summary, tracks = run_analytics(_CSV, _OUT)
print(f"analytics OK: {len(summary)} tracks")

tracks_t, labels = run_team_classification(_VID, _CSV, _OUT)
teams = tracks_t.groupby("team")["track_id"].nunique().to_dict()
print(f"teams OK: {teams}")

spd, spd_sum = run_speed_estimation(tracks_t, 25.0)
thresh = spd_sum["sprint_threshold_px_s"].iloc[0]
print(f"speed OK: sprint thresh = {thresh:.1f}")

ins = run_tactical_insights(tracks_t)
print(f"insights OK: {len(ins['key_moments'])} key moments")

fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
print("plotly OK")

print()
print("ALL v2.0 CHECKS PASSED")

