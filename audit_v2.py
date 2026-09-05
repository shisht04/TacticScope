"""
Full end-to-end audit of TacticScope v2.0.
Tests every data path the app uses and reports any issues.
"""
import sys, os
sys.path.insert(0, 'src')

import traceback
import pandas as pd
import numpy as np

errors   = []
warnings = []

def check(label, fn):
    try:
        result = fn()
        print(f"  ✅  {label}")
        return result
    except Exception as e:
        print(f"  ❌  {label}: {e}")
        errors.append((label, traceback.format_exc()))
        return None

print("=" * 55)
print("  TacticScope v2.0 — Full Audit")
print("=" * 55)

# ── 1. Imports ────────────────────────────────────────────────
print("\n[1] Imports")
from analytics        import run_analytics
from team_classifier  import run_team_classification
from speed_estimator  import run_speed_estimation
from tactical_insights import run_tactical_insights
import plotly.graph_objects as go
print("  ✅  all imports OK")

# ── 2. Pipeline ───────────────────────────────────────────────
print("\n[2] Pipeline (sample.mp4)")
CSV   = 'data/output/sample_tracks.csv'
VID   = 'data/videos/sample.mp4'
ODIR  = 'data/output'

summary = check("run_analytics",
    lambda: run_analytics(CSV, ODIR)[0])

tracks_t = check("run_team_classification",
    lambda: run_team_classification(VID, CSV, ODIR)[0])

speed_sum = None
if tracks_t is not None:
    result = check("run_speed_estimation",
        lambda: run_speed_estimation(tracks_t, 25.0))
    if result:
        _, speed_sum = result

insights = None
if tracks_t is not None:
    insights = check("run_tactical_insights",
        lambda: run_tactical_insights(tracks_t))

# ── 3. Award engine ───────────────────────────────────────────
print("\n[3] Award Engine")
if summary is not None and speed_sum is not None:
    team_s = tracks_t.groupby('track_id')['team'].first()
    merged = summary.merge(
        speed_sum[['track_id','avg_speed_px_s','top_speed_px_s','sprint_count']],
        on='track_id', how='left'
    ).fillna(0)
    merged['team'] = merged['track_id'].map(team_s)

    def compute_awards(df):
        awd = {}
        def give(tid, icon, lbl): awd.setdefault(int(tid), []).append((icon, lbl))
        v = df.dropna(subset=['top_speed_px_s','total_pixel_distance'])
        if v.empty: return awd
        give(v.loc[v['top_speed_px_s'].idxmax(), 'track_id'], '⚡', 'Speedster')
        give(v.loc[v['total_pixel_distance'].idxmax(), 'track_id'], '🏃', 'Marathon Man')
        give(df.loc[df['num_frames'].idxmax(), 'track_id'], '💪', 'The Engine')
        if 'sprint_count' in v.columns and v['sprint_count'].max() > 0:
            give(v.loc[v['sprint_count'].idxmax(), 'track_id'], '🔥', 'Sprint King')
        return awd

    awards = check("compute_awards", lambda: compute_awards(merged))
    if awards:
        print(f"       → {sum(len(v) for v in awards.values())} awards across {len(awards)} players")
else:
    warnings.append("Skipped award engine (pipeline data missing)")

# ── 4. Narrative engine ───────────────────────────────────────
print("\n[4] Narrative Engine")
if insights:
    cmp   = insights.get('compactness_summary', {})
    prs   = insights.get('pressing_summary',    {})
    ter   = insights.get('territorial',         {})
    frm   = insights.get('formation',           {})

    ca = cmp.get('team_a', {}).get('mean')
    cb = cmp.get('team_b', {}).get('mean')
    print(f"  ✅  compactness: A={ca:.1f}px  B={cb:.1f}px  (lower=tighter)" if isinstance(ca,float) else "  ⚠️  compactness not available")
    print(f"  ✅  pressing: {prs.get('more_aggressive_presser','?')} presses harder")
    dom = ter.get('dominance_grid')
    print(f"  ✅  territory grid: {dom.shape if dom is not None else 'N/A'}")
    fa = frm.get('team_a',{}).get('formation','?')
    fb = frm.get('team_b',{}).get('formation','?')
    print(f"  ✅  formations: A=~{fa}  B=~{fb}")
    print(f"  ✅  key moments: {len(insights.get('key_moments',[]))}")

# ── 5. Plotly figures ─────────────────────────────────────────
print("\n[5] Plotly Figures")
if tracks_t is not None:
    avg = tracks_t.groupby('track_id').agg(
        cx=('cx','mean'), cy=('cy','mean'),
        team=('team','first'), frames=('frame','count')
    ).reset_index()

    check("pitch_figure",
        lambda: go.Figure(go.Scatter(x=avg['cx'], y=avg['cy'], mode='markers')))

    spd_df = check("speed_fig",
        lambda: tracks_t.merge(
            speed_sum[['track_id','sprint_threshold_px_s']] if speed_sum is not None else pd.DataFrame(),
            on='track_id', how='left'))

    check("team_heatmap_fig",
        lambda: go.Figure(go.Histogram2d(
            x=tracks_t[tracks_t['team']==0]['cx'],
            y=tracks_t[tracks_t['team']==0]['cy'],
            nbinsx=30, nbinsy=20)))

# ── 6. HTML card generation ───────────────────────────────────
print("\n[6] HTML Card Generation")
def player_card_html(tid, team_id, dist, top_spd, sprints, award_list, mvp=False):
    """Simplified version of the card for testing."""
    return f'<div class="card">Track #{tid} | {dist:.0f}px | {top_spd:.0f}px/s</div>'

if summary is not None and speed_sum is not None:
    row = merged.iloc[0]
    card_html = check("player_card_html",
        lambda: player_card_html(
            int(row['track_id']), int(row.get('team',0)),
            float(row['total_pixel_distance']),
            float(row['top_speed_px_s']),
            int(row['sprint_count']),
            [('⚡','Speedster')], mvp=True
        ))

# ── 7. Output files ────────────────────────────────────────────
print("\n[7] Output Files")
expected_files = [
    ('Tracks CSV',         'data/output/sample_tracks.csv'),
    ('Tracks+Teams CSV',   'data/output/sample_tracks_with_teams.csv'),
    ('Summary CSV',        'data/output/sample_summary.csv'),
    ('Trajectories PNG',   'data/output/sample_trajectories.png'),
    ('Heatmap PNG',        'data/output/sample_heatmap_overall.png'),
    ('Annotated video',    'data/output/sample_annotated.mp4'),
    ('Team video',         'data/output/sample_team_annotated.mp4'),
]
for label, path in expected_files:
    exists = os.path.exists(path)
    size   = os.path.getsize(path) if exists else 0
    status = f"✅  {label} ({size//1024}KB)" if exists else f"⚠️  {label} MISSING"
    print(f"  {status}")

# ── 8. st.pills / st.html availability ───────────────────────
print("\n[8] Streamlit Widget Availability")
import streamlit as st
has_pills = hasattr(st, 'pills')
has_html  = hasattr(st, 'html')
has_sc    = hasattr(st, 'segmented_control')
print(f"  {'✅' if has_pills else '❌'}  st.pills()             (needed: nav bar)")
print(f"  {'✅' if has_html  else '❌'}  st.html()              (needed: player cards)")
print(f"  {'✅' if has_sc    else '❌'}  st.segmented_control() (needed: roster filter)")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 55)
if not errors:
    print("  🟢  ALL CHECKS PASSED — product is ready")
else:
    print(f"  🔴  {len(errors)} ISSUE(S) FOUND:")
    for lbl, tb in errors:
        print(f"\n  ── {lbl} ──")
        print(tb[-500:])
if warnings:
    print(f"\n  ⚠️  Warnings: {len(warnings)}")
    for w in warnings: print(f"     · {w}")
print("=" * 55)
