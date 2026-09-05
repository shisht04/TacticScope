"""
TacticScope v2.0 — Match Intelligence Platform
Not a dashboard. An experience.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from detect_and_track import run_detection_and_tracking
from analytics import run_analytics
from team_classifier import run_team_classification
from speed_estimator import run_speed_estimation
from tactical_insights import run_tactical_insights


# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TacticScope",
    layout="wide",
    page_icon="⚽",
    initial_sidebar_state="collapsed",
)

# ─── DESIGN SYSTEM ─────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
}
.stApp { background: #050810 !important; }
.block-container {
    max-width: 1380px !important;
    padding: 0 36px 60px 36px !important;
    margin: 0 auto !important;
}
#MainMenu { visibility: hidden !important; }
footer    { visibility: hidden !important; }
header    { visibility: hidden !important; }
[data-testid="stDeployButton"]  { display: none !important; }
[data-testid="stToolbar"]       { display: none !important; }
[data-testid="stSidebar"] {
    background: #07101e !important;
    border-right: 1px solid #1a2844 !important;
}
/* Pills navigation */
[data-testid="stPills"] { gap: 4px !important; }
[data-testid="stPills"] button {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    border-radius: 100px !important;
    padding: 6px 18px !important;
    background: transparent !important;
    border: 1px solid #1a2844 !important;
    color: #475569 !important;
    transition: all 0.2s !important;
}
[data-testid="stPills"] button[aria-checked="true"] {
    background: #3b82f6 !important;
    border-color: #3b82f6 !important;
    color: #fff !important;
}
[data-testid="stPills"] button:hover {
    background: #131e33 !important;
    color: #cbd5e1 !important;
}
/* Segmented control */
[data-testid="stSegmentedControl"] button {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
}
/* Videos */
video { border-radius: 12px !important; }
/* Plotly */
.js-plotly-plot .plotly { border-radius: 14px !important; overflow: hidden !important; }
/* Dividers */
hr { border-color: #1a2844 !important; margin: 20px 0 !important; }
/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #050810; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
/* Buttons */
[data-testid="stButton"] button {
    background: #0d1421 !important;
    border: 1px solid #1a2844 !important;
    color: #64748b !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
[data-testid="stButton"] button:hover {
    border-color: #3b82f6 !important;
    color: #3b82f6 !important;
}
</style>
""", unsafe_allow_html=True)


# ─── CONSTANTS ─────────────────────────────────────────────────────────────────
TA   = "#3b82f6"
TB   = "#ef4444"
OFF  = "#f59e0b"
GOLD = "#f59e0b"
GRN  = "#10b981"
BG   = "#050810"
SRF  = "#0d1421"
SRF2 = "#131e33"
BDR  = "#1a2844"
LABEL = {0: "Team A", 1: "Team B", 2: "Official"}
TCOL  = {0: TA, 1: TB, 2: OFF}


# ─── ROLE DETECTION ───────────────────────────────────────────────────────────
def detect_player_roles(tracks_df: pd.DataFrame, merged_df: pd.DataFrame) -> dict:
    """
    Post-processing layer on top of k-means team labels.
    Returns dict: track_id -> role string
      e.g. {116: 'Team A GK', 124: 'Team A', 163: 'Team B GK', 178: 'Referee', ...}

    GK detection:
      For each team, the player with the lowest combined score of
      (distance_to_nearest_goal / pitch_width) + (positional_std_x / pitch_width)
      is the most goal-anchored = goalkeeper.

    Official cleanup:
      Among k-means 'officials', the one with highest total distance is kept
      as 'Referee'. The rest are reassigned to the team whose average X centroid
      they are closest to (they were likely misclassified by jersey colour).
    """
    roles = {}
    avg_pos = tracks_df.groupby("track_id").agg(
        avg_x=("cx", "mean"),
        std_x=("cx", "std"),
    ).reset_index().fillna(0)

    fw = float(tracks_df["cx"].max()) or 640.0

    # ── Goalkeeper per team ──────────────────────────────────────────────────
    for team_id, lbl in [(0, "A"), (1, "B")]:
        tids = merged_df[merged_df["team"] == team_id]["track_id"].values
        if len(tids) == 0:
            continue
        tp = avg_pos[avg_pos["track_id"].isin(tids)].copy()
        if tp.empty:
            for tid in tids:
                roles[int(tid)] = f"Team {lbl}"
            continue
        g_left, g_right = fw * 0.10, fw * 0.90
        tp["goal_dist"]  = tp["avg_x"].apply(lambda x: min(abs(x - g_left), abs(x - g_right))) / fw
        tp["spread_x"]   = tp["std_x"] / fw
        tp["gk_score"]   = tp["goal_dist"] + tp["spread_x"]
        gk_id = int(tp.loc[tp["gk_score"].idxmin(), "track_id"])
        for tid in tids:
            roles[int(tid)] = f"Team {lbl} GK" if int(tid) == gk_id else f"Team {lbl}"

    # ── Official cleanup ─────────────────────────────────────────────────────
    off_tids = merged_df[merged_df["team"] == 2]["track_id"].values
    if len(off_tids) == 0:
        return roles

    off_rows = merged_df[merged_df["team"] == 2].sort_values(
        "total_pixel_distance", ascending=False
    ).reset_index(drop=True)

    # Centroids for each team (to reassign misclassified officials)
    def team_centroid_x(tid_vals):
        p = avg_pos[avg_pos["track_id"].isin(tid_vals)]["avg_x"]
        return float(p.mean()) if len(p) else fw / 2

    cx_a = team_centroid_x(merged_df[merged_df["team"] == 0]["track_id"].values)
    cx_b = team_centroid_x(merged_df[merged_df["team"] == 1]["track_id"].values)

    for i, row in off_rows.iterrows():
        tid = int(row["track_id"])
        if i == 0:                          # highest distance = real referee
            roles[tid] = "Referee"
        else:                               # reassign to nearest team
            ax = float(avg_pos.loc[avg_pos["track_id"] == tid, "avg_x"].iloc[0]
                       if not avg_pos[avg_pos["track_id"] == tid].empty else fw / 2)
            roles[tid] = "Team A" if abs(ax - cx_a) < abs(ax - cx_b) else "Team B"

    return roles


# Map role string -> display colour
def role_color(role: str) -> str:
    if "Team A" in role: return TA
    if "Team B" in role: return TB
    if "Referee" in role: return OFF
    return "#64748b"


# ─── STATE ─────────────────────────────────────────────────────────────────────
for k, v in [("results", None), ("view", "Match Overview"), ("player", None)]:
    if k not in st.session_state:
        st.session_state[k] = v


# ─── AWARD ENGINE ──────────────────────────────────────────────────────────────
def compute_awards(df: pd.DataFrame) -> dict:
    awd = {}
    def give(tid, icon, lbl):
        awd.setdefault(int(tid), []).append((icon, lbl))

    valid = df.dropna(subset=["top_speed_px_s", "total_pixel_distance"])
    if valid.empty:
        return awd
    give(valid.loc[valid["top_speed_px_s"].idxmax(),       "track_id"], "⚡", "Speedster")
    give(valid.loc[valid["total_pixel_distance"].idxmax(), "track_id"], "🏃", "Marathon Man")
    give(df.loc[df["num_frames"].idxmax(),                 "track_id"], "💪", "The Engine")
    if "sprint_count" in valid.columns and valid["sprint_count"].max() > 0:
        give(valid.loc[valid["sprint_count"].idxmax(), "track_id"], "🔥", "Sprint King")
    return awd


# ─── NARRATIVE ENGINE ─────────────────────────────────────────────────────────
def match_story(insights: dict) -> list:
    """Generate insights purely from analytics. Zero hallucinations."""
    out = []
    cmp = insights.get("compactness_summary", {})
    prs = insights.get("pressing_summary", {})
    ter = insights.get("territorial", {})

    ca = cmp.get("team_a", {}).get("mean")
    cb = cmp.get("team_b", {}).get("mean")
    if isinstance(ca, float) and isinstance(cb, float):
        if ca < cb:
            out.append((TA, "Team A", "🛡️",
                "stayed compact and well-organised — tight lines, limited the space between units, and made it harder to play through them."))
        else:
            out.append((TB, "Team B", "🛡️",
                "maintained the tighter defensive shape — disciplined horizontal compactness throughout the clip."))

    presser = prs.get("more_aggressive_presser", "")
    if presser:
        c = TA if presser == "Team A" else TB
        out.append((c, presser, "💪",
            "was the more aggressive pressing side — consistently closing down opponents, reducing time on the ball and forcing quicker decisions."))

    dom = ter.get("dominance_grid")
    if dom is not None:
        ga = float(ter.get("team_a_grid", np.zeros((1,1))).sum())
        gb = float(ter.get("team_b_grid", np.zeros((1,1))).sum())
        if ga > gb * 1.15:
            out.append((TA, "Team A", "🗺️",
                "controlled more of the pitch — higher presence across multiple zones, suggesting sustained positional dominance during this clip."))
        elif gb > ga * 1.15:
            out.append((TB, "Team B", "🗺️",
                "controlled more of the pitch — their movement patterns showed greater zone coverage across the horizontal thirds."))

    frm = insights.get("formation", {})
    for key, lbl, c in [("team_a","Team A",TA), ("team_b","Team B",TB)]:
        f = frm.get(key, {}).get("formation", "")
        if f:
            out.append((c, f"{lbl}'s Shape", "⚙️",
                f"aligned in an estimated {f} structure — based on average player positions during the clip. Indicative only, not ground truth."))
    return out


# ─── HTML COMPONENTS (via st.html) ────────────────────────────────────────────

def _rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


def html_player_card(tid, team_id, dist, top_spd, sprints, award_list,
                     mvp=False, compact=False, role: str = "") -> str:
    """
    Clean data card — inspired by Whoscored/FotMob.
    Large ghost number watermark, stats in a horizontal row with dividers,
    team color as a bottom border. No gradients, no left stripe.
    """
    color    = role_color(role) if role else TCOL.get(int(team_id), "#64748b")
    role_lbl = (role if role else LABEL.get(int(team_id), "Unknown")).upper()
    gk       = "GK" in role_lbl
    ref      = "REFEREE" in role_lbl
    dot_icon = "&#9670;" if gk else ("&#9711;" if ref else "&#9632;")  # ◆ ○ ■

    # Abbreviate distance for compact display
    dist_disp = f"{dist/1000:.1f}k" if dist >= 1000 else f"{dist:.0f}"
    spd_disp  = f"{top_spd:.0f}"
    spr_disp  = str(sprints)

    mvp_tag = (
        f'<div style="display:inline-block;background:{GOLD};color:#0a0f1a;'
        f'font-size:0.52rem;font-weight:900;letter-spacing:0.12em;'
        f'padding:1px 6px;margin-left:6px;vertical-align:middle;">MVP</div>'
        if mvp else ""
    )

    award_html = ""
    if award_list:
        pills = "".join(
            f'<span style="font-size:0.58rem;color:#64748b;margin-right:8px;">'
            f'{ic}&thinsp;{lb}</span>'
            for ic, lb in award_list
        )
        award_html = f'<div style="margin-top:10px;padding-top:8px;border-top:1px solid #141f30;">{pills}</div>'

    # Ghost number size
    ghost_sz  = "4.5rem" if compact else "6rem"
    pad       = "14px 16px 12px" if compact else "18px 20px 14px"
    stat_sz   = "1.35rem" if compact else "1.6rem"
    lbl_sz    = "0.48rem"

    return (
        # Outer card — dark, flat, bottom accent
        f'<div style="background:#0b1623;border:1px solid #162030;'
        f'border-bottom:2px solid {color};padding:{pad};'
        f'position:relative;overflow:hidden;font-family:Inter,sans-serif;">'

        # Ghost track number (watermark)
        f'<div style="position:absolute;right:-4px;top:-14px;'
        f'font-size:{ghost_sz};font-weight:900;line-height:1;'
        f'color:rgba(255,255,255,0.028);font-family:Rajdhani,sans-serif;'
        f'pointer-events:none;user-select:none;">#{tid}</div>'

        # Row 1: dot + role + MVP badge
        f'<div style="display:flex;align-items:center;gap:5px;margin-bottom:8px;">'
        f'<span style="color:{color};font-size:0.45rem;line-height:1;">{dot_icon}</span>'
        f'<span style="font-size:0.52rem;color:{color};font-weight:700;'
        f'text-transform:uppercase;letter-spacing:0.12em;">{role_lbl}</span>'
        f'{mvp_tag}</div>'

        # Row 2: Track ID
        f'<div style="font-size:0.95rem;font-weight:700;color:#dde4f0;'
        f'font-family:Rajdhani,sans-serif;letter-spacing:0.04em;margin-bottom:14px;">'
        f'#{tid}</div>'

        # Stats row with dividers
        f'<div style="display:flex;border-top:1px solid #142030;padding-top:10px;">'

        # Distance
        f'<div style="flex:1;text-align:center;">'
        f'<div style="font-size:{stat_sz};font-weight:800;color:#e8edf5;'
        f'font-family:Rajdhani,sans-serif;line-height:1;letter-spacing:-0.01em;">{dist_disp}</div>'
        f'<div style="font-size:{lbl_sz};color:#3a5070;text-transform:uppercase;'
        f'letter-spacing:0.1em;margin-top:4px;">Distance</div></div>'

        # Divider
        f'<div style="width:1px;background:#142030;margin:2px 0;"></div>'

        # Top Speed
        f'<div style="flex:1;text-align:center;">'
        f'<div style="font-size:{stat_sz};font-weight:800;color:#e8edf5;'
        f'font-family:Rajdhani,sans-serif;line-height:1;letter-spacing:-0.01em;">{spd_disp}</div>'
        f'<div style="font-size:{lbl_sz};color:#3a5070;text-transform:uppercase;'
        f'letter-spacing:0.1em;margin-top:4px;">Speed</div></div>'

        # Divider
        f'<div style="width:1px;background:#142030;margin:2px 0;"></div>'

        # Sprints
        f'<div style="flex:1;text-align:center;">'
        f'<div style="font-size:{stat_sz};font-weight:800;color:#e8edf5;'
        f'font-family:Rajdhani,sans-serif;line-height:1;letter-spacing:-0.01em;">{spr_disp}</div>'
        f'<div style="font-size:{lbl_sz};color:#3a5070;text-transform:uppercase;'
        f'letter-spacing:0.1em;margin-top:4px;">Sprints</div></div>'

        f'</div>'
        + award_html
        + '</div>'
    )


def html_story_card(color, team, icon, text) -> str:
    rgb = _rgb(color)
    return f"""
    <div style="background:rgba({rgb},0.05);border:1px solid rgba({rgb},0.18);
                border-left:4px solid {color};border-radius:12px;
                padding:15px 20px;margin:8px 0;font-family:'Inter',sans-serif;">
      <div style="font-size:1rem;margin-bottom:4px;">{icon}</div>
      <div style="font-size:0.62rem;color:{color};text-transform:uppercase;letter-spacing:0.1em;
                  font-weight:700;margin-bottom:3px;">{team}</div>
      <div style="color:#cbd5e1;font-size:0.88rem;line-height:1.6;">{text}</div>
    </div>"""


def html_match_hero(n_a, n_b, dur, frames, fps, vname) -> str:
    return f"""
    <div style="background:linear-gradient(135deg,#0d1928 0%,{BG} 100%);
                border:1px solid {BDR};border-radius:20px;
                padding:40px 48px;margin:24px 0 32px 0;
                position:relative;overflow:hidden;font-family:'Inter',sans-serif;text-align:center;">
      <div style="position:absolute;top:-80px;left:50%;transform:translateX(-50%);
                  width:350px;height:160px;pointer-events:none;
                  background:radial-gradient(ellipse,rgba(59,130,246,0.18) 0%,transparent 70%);"></div>
      <div style="font-size:0.6rem;color:#334155;text-transform:uppercase;letter-spacing:0.2em;
                  font-weight:700;margin-bottom:6px;">Match Report</div>
      <div style="font-size:0.78rem;color:#475569;margin-bottom:24px;font-style:italic;">
        {vname}</div>
      <div style="display:flex;align-items:center;justify-content:center;gap:40px;margin-bottom:28px;">
        <div style="text-align:right;">
          <div style="font-size:0.6rem;color:{TA};text-transform:uppercase;letter-spacing:0.12em;
                      font-weight:700;margin-bottom:3px;">Team A</div>
          <div style="font-size:4rem;font-weight:900;color:{TA};font-family:'Rajdhani',sans-serif;line-height:1;">{n_a}</div>
          <div style="font-size:0.68rem;color:#334155;">players tracked</div>
        </div>
        <div style="width:64px;height:64px;border-radius:50%;
                    background:linear-gradient(135deg,{TA},{TB});
                    display:flex;align-items:center;justify-content:center;
                    font-size:1.8rem;flex-shrink:0;
                    box-shadow:0 0 40px rgba(59,130,246,0.25);">⚽</div>
        <div style="text-align:left;">
          <div style="font-size:0.6rem;color:{TB};text-transform:uppercase;letter-spacing:0.12em;
                      font-weight:700;margin-bottom:3px;">Team B</div>
          <div style="font-size:4rem;font-weight:900;color:{TB};font-family:'Rajdhani',sans-serif;line-height:1;">{n_b}</div>
          <div style="font-size:0.68rem;color:#334155;">players tracked</div>
        </div>
      </div>
      <div style="display:flex;gap:16px;justify-content:center;flex-wrap:wrap;">
        <div style="background:{SRF2};border:1px solid {BDR};border-radius:100px;
                    padding:5px 16px;font-size:0.74rem;color:#64748b;">⏱ {dur}s</div>
        <div style="background:{SRF2};border:1px solid {BDR};border-radius:100px;
                    padding:5px 16px;font-size:0.74rem;color:#64748b;">🎞 {frames} frames</div>
        <div style="background:{SRF2};border:1px solid {BDR};border-radius:100px;
                    padding:5px 16px;font-size:0.74rem;color:#64748b;">📹 {fps:.0f} fps</div>
      </div>
    </div>"""


def html_section_head(title: str, sub: str = "") -> str:
    """Editorial section header — heavy title, muted sub, no decorative bar."""
    return (
        f'<div style="margin:40px 0 16px;font-family:Inter,sans-serif;">'
        f'<div style="font-size:0.55rem;text-transform:uppercase;letter-spacing:0.18em;'
        f'color:#334155;font-weight:700;margin-bottom:6px;">TacticScope</div>'
        f'<div style="font-size:1.5rem;font-weight:800;color:#f1f5f9;'
        f'letter-spacing:-0.03em;line-height:1.1;font-family:Rajdhani,sans-serif;'
        f'text-transform:uppercase;">{title}</div>'
        + (f'<div style="font-size:0.73rem;color:#475569;margin-top:4px;">{sub}</div>' if sub else '')
        + '</div>'
    )


def html_key_moments(events: list) -> str:
    if not events:
        return f'<div style="color:#334155;font-size:0.82rem;padding:8px 0;font-family:Inter,sans-serif;">No significant transitions detected. Try a longer clip.</div>'
    items = ""
    for e in events[:16]:
        dot = TA if "Team A" in e.get("team", "") else TB
        items += f"""
        <div style="display:flex;align-items:flex-start;gap:14px;padding:11px 16px;
                    border-radius:10px;background:{SRF};border:1px solid {BDR};
                    margin:6px 0;font-family:'Inter',sans-serif;">
          <div style="width:9px;height:9px;border-radius:50%;background:{dot};
                      margin-top:5px;flex-shrink:0;"></div>
          <div>
            <div style="font-size:0.62rem;color:#475569;margin-bottom:1px;">Frame {e['frame']}</div>
            <div style="font-size:0.84rem;color:#cbd5e1;">{e['description']}</div>
          </div>
        </div>"""
    return items


def html_battle_bar(label, va, vb, higher_better=True) -> str:
    fa = float(str(va).replace(",","")) if isinstance(va, str) else float(va)
    fb = float(str(vb).replace(",","")) if isinstance(vb, str) else float(vb)
    total = fa + fb or 1
    pa = fa / total
    win_a = pa >= 0.5 if higher_better else pa <= 0.5
    ca = TA if win_a else "#334155"
    cb = TB if not win_a else "#334155"
    return f"""
    <div style="background:{SRF};border:1px solid {BDR};border-radius:12px;
                padding:14px 18px;margin:8px 0;font-family:'Inter',sans-serif;">
      <div style="font-size:0.6rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;
                  font-weight:600;margin-bottom:10px;">{label}</div>
      <div style="display:flex;align-items:center;gap:12px;">
        <div style="color:{ca};font-size:1.2rem;font-weight:800;font-family:'Rajdhani',sans-serif;
                    min-width:72px;text-align:right;">{va}</div>
        <div style="flex:1;height:6px;border-radius:3px;background:{BDR};position:relative;overflow:hidden;">
          <div style="position:absolute;left:0;top:0;bottom:0;width:{pa*100:.1f}%;
                      background:{TA};border-radius:3px;"></div>
        </div>
        <div style="color:{cb};font-size:1.2rem;font-weight:800;font-family:'Rajdhani',sans-serif;
                    min-width:72px;text-align:left;">{vb}</div>
      </div>
      <div style="display:flex;justify-content:space-between;padding:0 84px;
                  font-size:0.58rem;color:#334155;margin-top:4px;">
        <span>Team A</span><span>Team B</span>
      </div>
    </div>"""


# ─── PLOTLY BUILDERS ─────────────────────────────────────────────────────────

_dark = dict(
    paper_bgcolor=SRF, plot_bgcolor=SRF2,
    margin=dict(l=8, r=8, t=36, b=8), height=230,
    font=dict(family="Inter", color="#475569", size=9),
    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, linecolor=BDR),
    yaxis=dict(showgrid=True, gridcolor="#1e293b", gridwidth=0.5, zeroline=False,
               tickfont=dict(size=8)),
)


def pitch_figure(tracks_df: pd.DataFrame) -> go.Figure:
    avg = tracks_df.groupby("track_id").agg(
        cx=("cx","mean"), cy=("cy","mean"),
        team=("team","first"), frames=("frame","count")
    ).reset_index()

    fw = float(tracks_df["cx"].max())
    fh = float(tracks_df["cy"].max())
    r  = min(fw, fh) * 0.11

    shapes = [
        dict(type="rect", x0=0, y0=0, x1=fw, y1=fh,
             line=dict(color="rgba(255,255,255,0.55)", width=2.5), fillcolor="rgba(0,0,0,0)"),
        dict(type="line", x0=fw/2, y0=0, x1=fw/2, y1=fh,
             line=dict(color="rgba(255,255,255,0.45)", width=1.8)),
        dict(type="circle", x0=fw/2-r, y0=fh/2-r, x1=fw/2+r, y1=fh/2+r,
             line=dict(color="rgba(255,255,255,0.4)", width=1.5), fillcolor="rgba(0,0,0,0)"),
        dict(type="circle", x0=fw/2-3, y0=fh/2-3, x1=fw/2+3, y1=fh/2+3,
             line=dict(color="rgba(255,255,255,0.5)", width=1),
             fillcolor="rgba(255,255,255,0.5)"),
        # Left penalty box
        dict(type="rect", x0=0, y0=fh*0.25, x1=fw*0.16, y1=fh*0.75,
             line=dict(color="rgba(255,255,255,0.35)", width=1.2), fillcolor="rgba(0,0,0,0)"),
        dict(type="rect", x0=0, y0=fh*0.37, x1=fw*0.06, y1=fh*0.63,
             line=dict(color="rgba(255,255,255,0.25)", width=1), fillcolor="rgba(0,0,0,0)"),
        # Right penalty box
        dict(type="rect", x0=fw*0.84, y0=fh*0.25, x1=fw, y1=fh*0.75,
             line=dict(color="rgba(255,255,255,0.35)", width=1.2), fillcolor="rgba(0,0,0,0)"),
        dict(type="rect", x0=fw*0.94, y0=fh*0.37, x1=fw, y1=fh*0.63,
             line=dict(color="rgba(255,255,255,0.25)", width=1), fillcolor="rgba(0,0,0,0)"),
    ]

    fig = go.Figure()
    max_f = float(avg["frames"].max()) or 1

    for tid, color, name, sym in [(0, TA, "Team A", "circle"),
                                   (1, TB, "Team B", "circle"),
                                   (2, OFF, "Official", "diamond")]:
        pts = avg[avg["team"] == tid]
        if pts.empty:
            continue
        sizes = 13 + (pts["frames"] / max_f * 24)
        hover = [
            f"<b>Track #{int(r['track_id'])}</b><br>"
            f"<span style='color:{color}'>{name}</span><br>"
            f"Activity: {int(r['frames'])} frames"
            for _, r in pts.iterrows()
        ]
        fig.add_trace(go.Scatter(
            x=pts["cx"], y=pts["cy"],
            mode="markers+text", name=name,
            marker=dict(color=color, size=sizes, symbol=sym,
                        line=dict(color="rgba(255,255,255,0.85)", width=1.8), opacity=0.93),
            text=[f"#{int(r['track_id'])}" for _, r in pts.iterrows()],
            textfont=dict(color="white", size=6.5, family="Inter"),
            textposition="middle center",
            hovertext=hover, hoverinfo="text",
            customdata=pts[["track_id"]].values,
        ))

    fig.update_layout(
        shapes=shapes,
        plot_bgcolor="#1a7a40", paper_bgcolor=SRF,
        height=430, margin=dict(l=8, r=8, t=14, b=8),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(color="#94a3b8", size=11, family="Inter"),
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(range=[-fw*0.01, fw*1.01], showgrid=False,
                   showticklabels=False, zeroline=False, fixedrange=True),
        yaxis=dict(range=[fh*1.04, -fh*0.04], showgrid=False,
                   showticklabels=False, zeroline=False, fixedrange=True),
        hoverlabel=dict(bgcolor="#1e293b", bordercolor=BDR,
                        font=dict(color="white", size=12, family="Inter")),
        dragmode=False,
        clickmode="event+select",
    )
    return fig


def trajectory_fig(tracks_df, track_id, team_id) -> go.Figure:
    td = tracks_df[tracks_df["track_id"] == track_id].sort_values("frame")
    c = TCOL.get(int(team_id), "#94a3b8")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=td["cx"], y=td["cy"], mode="lines",
        line=dict(color=c, width=2.5), showlegend=False,
    ))
    if len(td) > 1:
        fig.add_trace(go.Scatter(x=[td["cx"].iloc[0]],  y=[td["cy"].iloc[0]],
                                  mode="markers", name="Start",
                                  marker=dict(color=GRN, size=11, symbol="circle"),
                                  showlegend=True))
        fig.add_trace(go.Scatter(x=[td["cx"].iloc[-1]], y=[td["cy"].iloc[-1]],
                                  mode="markers", name="End",
                                  marker=dict(color=GOLD, size=11, symbol="circle"),
                                  showlegend=True))
    layout = dict(**_dark)
    layout["yaxis"] = {**_dark["yaxis"], "autorange": "reversed"}
    layout["title"] = dict(text="Trajectory", font=dict(size=10, color="#64748b"))
    layout["legend"] = dict(orientation="h", y=1.12, font=dict(size=9),
                             bgcolor="rgba(0,0,0,0)")
    fig.update_layout(**layout)
    return fig


def heatmap_fig(tracks_df, track_id, color) -> go.Figure:
    td = tracks_df[tracks_df["track_id"] == track_id]
    rgb = _rgb(color)
    fig = go.Figure(go.Histogram2d(
        x=td["cx"], y=td["cy"],
        colorscale=[[0,"rgba(0,0,0,0)"],[0.4,f"rgba({rgb},0.5)"],[1,color]],
        showscale=False, nbinsx=25, nbinsy=18,
    ))
    layout = dict(**_dark)
    layout["yaxis"] = {**_dark["yaxis"], "autorange": "reversed"}
    layout["title"] = dict(text="Activity Zones", font=dict(size=10, color="#64748b"))
    fig.update_layout(**layout)
    return fig


def speed_fig(speed_df, track_id, sprint_thresh) -> go.Figure:
    td = speed_df[speed_df["track_id"] == track_id].sort_values("frame")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=td["frame"], y=td["speed_px_per_sec"], mode="lines",
        fill="tozeroy", line=dict(color="#a78bfa", width=1.8),
        fillcolor="rgba(167,139,250,0.1)", showlegend=False,
    ))
    fig.add_hline(y=sprint_thresh, line=dict(color=GOLD, width=1.5, dash="dot"),
                  annotation_text="Sprint threshold", annotation_font_color=GOLD,
                  annotation_font_size=8)
    layout = dict(**_dark)
    layout["title"] = dict(text="Speed Over Time", font=dict(size=10, color="#64748b"))
    layout["xaxis"] = {**_dark["xaxis"], "showticklabels": True,
                        "title": dict(text="Frame", font=dict(size=8, color="#475569"))}
    layout["yaxis"] = {**_dark["yaxis"],
                        "title": dict(text="px/s", font=dict(size=8, color="#475569"))}
    fig.update_layout(**layout)
    return fig


def formation_pitch_fig(formation, territorial) -> go.Figure:
    fw = float(territorial.get("frame_w", 640))
    fh = float(territorial.get("frame_h", 360))
    r  = min(fw, fh) * 0.11
    fig = go.Figure()

    # Territory overlay
    dom = territorial.get("dominance_grid")
    if dom is not None:
        rows_d, cols_d = dom.shape
        cw, ch = fw / cols_d, fh / rows_d
        for ri in range(rows_d):
            for ci in range(cols_d):
                v = float(dom[ri, ci])
                if abs(v) > 0.08:
                    c = TA if v > 0 else TB
                    rgb = _rgb(c)
                    fig.add_shape(type="rect", layer="below",
                                  x0=ci*cw, y0=ri*ch, x1=(ci+1)*cw, y1=(ri+1)*ch,
                                  fillcolor=f"rgba({rgb},{abs(v)*0.38:.2f})",
                                  line=dict(width=0))

    # Pitch markings
    for shp in [
        dict(type="rect", x0=0, y0=0, x1=fw, y1=fh,
             line=dict(color="rgba(255,255,255,0.5)", width=2), fillcolor="rgba(0,0,0,0)"),
        dict(type="line", x0=fw/2, y0=0, x1=fw/2, y1=fh,
             line=dict(color="rgba(255,255,255,0.4)", width=1.5)),
        dict(type="circle", x0=fw/2-r, y0=fh/2-r, x1=fw/2+r, y1=fh/2+r,
             line=dict(color="rgba(255,255,255,0.4)", width=1.5), fillcolor="rgba(0,0,0,0)"),
    ]:
        fig.add_shape(**shp)

    # Convex hulls + dots per team
    for key, tid, color, name in [("team_a",0,TA,"Team A"),("team_b",1,TB,"Team B")]:
        data = formation.get(key, {})
        pts  = data.get("positions", np.array([]))
        hull = formation.get(f"{key}_hull")
        if len(pts) < 2:
            continue
        if hull is not None:
            hp = pts[hull.vertices]
            hp = np.vstack([hp, hp[0]])
            fig.add_trace(go.Scatter(
                x=hp[:,0], y=hp[:,1], fill="toself", showlegend=False,
                fillcolor=f"rgba({_rgb(color)},0.12)",
                line=dict(color=color, width=2), hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=pts[:,0], y=pts[:,1], mode="markers", name=name,
            marker=dict(color=color, size=16, symbol="circle",
                        line=dict(color="white", width=2)),
            hoverinfo="skip"))

    fig.update_layout(
        plot_bgcolor="#1a7a40", paper_bgcolor=SRF,
        height=400, margin=dict(l=8,r=8,t=40,b=8),
        title=dict(text="Formation Snapshot + Territorial Control",
                   font=dict(size=12, color="#94a3b8"), x=0.5),
        xaxis=dict(range=[-fw*0.01,fw*1.01], showgrid=False,
                   showticklabels=False, zeroline=False, fixedrange=True),
        yaxis=dict(range=[fh*1.04,-fh*0.04], showgrid=False,
                   showticklabels=False, zeroline=False, fixedrange=True),
        legend=dict(orientation="h", y=1.06, font=dict(color="#94a3b8",size=11),
                    bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def timeline_fig(df, col_a, col_b, title, y_label) -> go.Figure:
    fig = go.Figure()
    for col, name, color in [(col_a,"Team A",TA),(col_b,"Team B",TB)]:
        s = df[col].rolling(15, min_periods=1).mean()
        rgb = _rgb(color)
        fig.add_trace(go.Scatter(
            x=df["frame"], y=s, name=name,
            line=dict(color=color, width=2.5),
            fill="tozeroy",
            fillcolor=f"rgba({rgb},0.07)",
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=11, color="#64748b")),
        paper_bgcolor=SRF, plot_bgcolor=SRF2,
        height=220, margin=dict(l=8,r=8,t=36,b=8),
        legend=dict(orientation="h", y=1.15, font=dict(color="#94a3b8",size=10),
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, color="#334155",
                   title=dict(text="Frame", font=dict(size=8))),
        yaxis=dict(showgrid=True, gridcolor="#1e293b", color="#475569",
                   title=dict(text=y_label, font=dict(size=8))),
        font=dict(family="Inter"),
    )
    return fig


def team_heatmap_fig(tracks_df, team_id, color, title) -> go.Figure:
    td = tracks_df[tracks_df["team"] == team_id]
    if td.empty:
        return go.Figure()
    rgb = _rgb(color)
    fig = go.Figure(go.Histogram2d(
        x=td["cx"], y=td["cy"],
        colorscale=[[0,"rgba(0,0,0,0)"],[0.35,f"rgba({rgb},0.5)"],[1,color]],
        showscale=False, nbinsx=30, nbinsy=20,
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=10, color="#64748b")),
        paper_bgcolor=SRF, plot_bgcolor=SRF2,
        height=260, margin=dict(l=8,r=8,t=36,b=8),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, autorange="reversed"),
    )
    return fig


# ─── PIPELINE ─────────────────────────────────────────────────────────────────
def run_pipeline(video_path, output_dir, fps_override, progress_cb) -> dict:
    def rpt(p, m): progress_cb(p, m)

    rpt(5,  "🔍 Detecting players — YOLOv8 scanning every frame…")
    ann_vid, tracks_csv, thumb_paths = run_detection_and_tracking(video_path, output_dir)

    rpt(35, "📊 Building trajectories and distance profiles…")
    summary_df, tracks_df = run_analytics(tracks_csv, output_dir)

    rpt(55, "👕 Reading jersey colours — classifying teams…")
    tracks_teams, team_labels = run_team_classification(video_path, tracks_csv, output_dir)

    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = fps_override or (cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.release()

    rpt(68, "⚡ Measuring speed and sprint intensity…")
    tracks_speed, speed_summary = run_speed_estimation(tracks_teams, fps)

    rpt(82, "🧠 Reading tactical patterns and key moments…")
    insights = run_tactical_insights(tracks_teams)

    rpt(97, "✅ Assembling your match report…")
    vname = os.path.splitext(os.path.basename(tracks_csv))[0].replace("_tracks","")

    return dict(
        video_name=vname, fps=fps,
        annotated_video=ann_vid,
        thumb_paths=thumb_paths,
        team_annotated_video=os.path.join(output_dir, f"{vname}_team_annotated.mp4"),
        summary_df=summary_df, tracks_df=tracks_teams,
        tracks_speed=tracks_speed, speed_summary=speed_summary,
        team_labels=team_labels,
        trajectories_png=os.path.join(output_dir, f"{vname}_trajectories.png"),
        heatmap_overall_png=os.path.join(output_dir, f"{vname}_heatmap_overall.png"),
        insights=insights,
    )


# ─── URL DOWNLOADER ───────────────────────────────────────────────────────────────
def download_video_url(url: str, max_mb: int = 500) -> str:
    """
    Stream-download a direct video URL to a named temp file.
    Returns the local file path on success.
    Raises ValueError for bad URLs/types, RuntimeError for download failures.
    """
    import urllib.request
    import urllib.parse

    # Basic sanity checks before hitting the network
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http:// and https:// URLs are supported.")

    path_lower = parsed.path.lower()
    allowed    = (".mp4", ".mov", ".avi", ".mkv", ".webm")
    if not any(path_lower.endswith(ext) for ext in allowed):
        raise ValueError(
            f"URL must point to a direct video file ({', '.join(allowed)}). "
            "YouTube and other streaming sites are not supported."
        )

    max_bytes = max_mb * 1024 * 1024
    suffix    = os.path.splitext(parsed.path)[-1] or ".mp4"

    # Write to a named temp file so the pipeline can open it by path
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="tacticscope_url_")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TacticScope/2.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            # Respect Content-Length if provided
            cl = resp.headers.get("Content-Length")
            if cl and int(cl) > max_bytes:
                raise ValueError(f"File too large ({int(cl)//1024//1024} MB). Limit: {max_mb} MB.")
            downloaded = 0
            chunk_size = 1024 * 256  # 256 KB chunks
            with os.fdopen(tmp_fd, "wb") as f:
                tmp_fd = None  # fd now owned by f
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        raise ValueError(f"Download exceeded {max_mb} MB limit.")
                    f.write(chunk)
    except Exception:
        # Clean up on any failure
        if tmp_fd is not None:
            os.close(tmp_fd)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    return tmp_path


# ─── SIDEBAR (post-analysis quick actions only) ───────────────────────────────
# The sidebar is a secondary utility panel — not the primary entry point.
# Upload controls live on the landing page so the sidebar can safely be collapsed.
video_path = st.session_state.get("_staged_video", None)
fps_val    = st.session_state.get("_staged_fps",   None)
run_btn    = st.session_state.get("_run",           False)

if st.session_state.results:
    with st.sidebar:
        st.markdown("## ⚽ TacticScope")
        st.caption("v2.0 · Match Intelligence")
        st.divider()
        st.markdown(f"**{R['video_name'] if 'R' in dir() else 'Loaded'}**" if st.session_state.results else "")
        if st.button("↩  Analyse New Match", width='stretch'):
            st.session_state.results = None
            st.session_state.player  = None
            st.session_state["_staged_video"] = None
            st.session_state["_run"]   = False
            st.rerun()
        st.caption("⏱ CPU inference — keep clips ≤ 20 sec")






# ─── LANDING ─────────────────────────────────────────────────────────────────
if st.session_state.results is None:
    # Hero
    st.html(f"""
    <div style="text-align:center;padding:56px 32px 20px;font-family:'Inter',sans-serif;">
      <div style="font-size:4.5rem;margin-bottom:16px;
                  filter:drop-shadow(0 0 32px rgba(59,130,246,0.55));">⚽</div>
      <div style="font-size:3.6rem;font-weight:900;color:#f8fafc;letter-spacing:-0.03em;
                  font-family:'Rajdhani',sans-serif;margin-bottom:6px;line-height:1.1;">
        TacticScope
      </div>
      <div style="font-size:0.8rem;color:#475569;text-transform:uppercase;letter-spacing:0.18em;
                  font-weight:600;margin-bottom:48px;">Match Intelligence Platform</div>
    </div>""")

    # Upload card — centred, full-width controls
    _, card_col, _ = st.columns([1, 2, 1])
    with card_col:
        st.html(f"""
        <div style="background:{SRF};border:1px solid {BDR};border-radius:20px;
                    padding:36px 40px 28px;font-family:'Inter',sans-serif;text-align:center;
                    margin-bottom:4px;">
          <div style="font-size:0.62rem;color:{TA};text-transform:uppercase;letter-spacing:0.14em;
                      font-weight:700;margin-bottom:10px;">Step 1 — Select Your Clip</div>
          <div style="font-size:1.05rem;font-weight:700;color:#e2e8f0;margin-bottom:6px;">
            Upload a Match Clip</div>
          <div style="font-size:0.8rem;color:#475569;line-height:1.6;margin-bottom:18px;">
            TacticScope detects players, classifies teams,<br>measures movement and builds your match report.
          </div>
          <div style="font-size:0.68rem;color:#334155;margin-bottom:20px;">
            Recommended: 10–20 sec &nbsp;·&nbsp; 720p+
          </div>
        </div>""")

        # ── Source mode tabs: Upload vs URL ───────────────────────────────────
        src_tab = st.radio(
            "Input method",
            ["📤 Upload Video", "🔗 Video URL"],
            horizontal=True,
            label_visibility="collapsed",
            key="landing_src",
        )

        lp_video      = None   # path handed to the pipeline
        _url_tmp_path = None   # temp file created by URL download (needs cleanup)

        if src_tab == "📤 Upload Video":
            upl = st.file_uploader(
                "Drop your clip here", type=["mp4", "mov", "avi"],
                label_visibility="visible", key="landing_upl"
            )
            if upl:
                tmp = os.path.join(tempfile.gettempdir(), upl.name)
                with open(tmp, "wb") as f:
                    f.write(upl.read())
                lp_video = tmp

        else:  # URL mode
            url_input = st.text_input(
                "Paste a direct video URL (.mp4 / .mov / .avi …)",
                placeholder="https://example.com/match_clip.mp4",
                key="landing_url",
                label_visibility="visible",
            )
            st.caption(
                "⚠️ Direct file links only · No YouTube / streaming sites · Max 500 MB"
            )
            if url_input and url_input.strip():
                st.session_state["_pending_url"] = url_input.strip()
                lp_video = "__url__"  # sentinel; actual download happens on Analyse

        fps_lp     = st.number_input(
            "FPS override (leave 0 for auto-detect)", 0.0, 120.0, 0.0, 1.0,
            key="landing_fps", label_visibility="visible"
        )
        fps_val_lp = fps_lp if fps_lp > 0 else None

        go_btn = st.button(
            "▶  Analyse Match", type="primary",
            disabled=not lp_video, width="stretch", key="landing_go"
        )

        if go_btn and lp_video:
            bar = st.progress(0, text="Initialising…")
            try:
                # ── Resolve URL sentinel ────────────────────────────────────
                if lp_video == "__url__":
                    pending_url = st.session_state.get("_pending_url", "")
                    if not pending_url:
                        raise ValueError("No URL found — please paste a URL and try again.")
                    bar.progress(2, text="📥 Downloading video…")
                    lp_video      = download_video_url(pending_url)
                    _url_tmp_path = lp_video   # remember for cleanup

                st.session_state.results = run_pipeline(
                    lp_video, os.path.join("data", "output"), fps_val_lp,
                    lambda p, m: bar.progress(p, text=m)
                )
                st.session_state.view   = "Match Overview"
                st.session_state.player = None
                bar.empty()
            except Exception as e:
                import traceback
                bar.empty()
                st.session_state.results = None
                st.error(f"❌ {e}")
                with st.expander("Full traceback"):
                    st.code(traceback.format_exc())
            finally:
                # Clean up any URL-downloaded temp file whether pipeline passed or failed
                if _url_tmp_path and os.path.exists(_url_tmp_path):
                    try:
                        os.remove(_url_tmp_path)
                    except OSError:
                        pass

            if st.session_state.results:
                st.rerun()

    # Feature pills at bottom
    st.html(f"""
    <div style="display:flex;gap:40px;justify-content:center;padding:36px 0 24px;
                font-family:'Inter',sans-serif;flex-wrap:wrap;">
      {"  ".join(f'<div style="text-align:center;"><div style="font-size:1.5rem;margin-bottom:6px;">{ic}</div><div style="font-size:0.68rem;color:#334155;">{lbl}</div></div>'
               for ic, lbl in [("🔍","YOLOv8 Detection"),("🏃","ByteTrack Tracking"),
                               ("👕","Team Classification"),("🧠","Tactical Insights")])}
    </div>""")
    st.stop()



# ─── UNPACK RESULTS ───────────────────────────────────────────────────────────
R         = st.session_state.results
summary   = R["summary_df"]
tracks    = R["tracks_df"]
speed_df  = R["tracks_speed"]
speed_sum = R["speed_summary"]
insights  = R["insights"]
fps_r     = R["fps"]

team_s = tracks.groupby("track_id")["team"].first()
n_a = int((team_s == 0).sum())
n_b = int((team_s == 1).sum())
n_o = int((team_s == 2).sum())
total_f = int(tracks["frame"].max()) + 1
dur     = round(total_f / fps_r, 1)

merged = summary.merge(
    speed_sum[["track_id","avg_speed_px_s","top_speed_px_s","sprint_count"]],
    on="track_id", how="left"
).fillna(0)
merged["team"] = merged["track_id"].map(team_s)

awards = compute_awards(merged)

# ── Role detection: GK labels + official cleanup ───────────────────────
roles = detect_player_roles(tracks, merged)
# Recompute hero counts using roles (misclassified officials now reassigned)
n_a = sum(1 for r in roles.values() if "Team A" in r)
n_b = sum(1 for r in roles.values() if "Team B" in r)
n_ref = sum(1 for r in roles.values() if "Referee" in r)

s_thresh = float(speed_sum["sprint_threshold_px_s"].iloc[0]) if len(speed_sum) else 0

def _score(r):
    dm = merged["total_pixel_distance"].max() or 1
    sm = merged["top_speed_px_s"].max() or 1
    km = merged["sprint_count"].max() or 1
    return (r["total_pixel_distance"]/dm)*0.4 + (r["top_speed_px_s"]/sm)*0.4 + (r["sprint_count"]/km)*0.2

merged["_sc"] = merged.apply(_score, axis=1)
mvp_row = merged.loc[merged["_sc"].idxmax()]
mvp_id  = int(mvp_row["track_id"])

if st.session_state.player is None:
    st.session_state.player = mvp_id


# ─── TOP NAV ─────────────────────────────────────────────────────────────────
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

NAV = ["Match Overview", "Player Explorer", "Team Battle", "Tactical Story", "Full Roster"]
try:
    view = st.pills("Navigation", NAV, default=st.session_state.view, key="nav_pills",
                    label_visibility="collapsed")
except (AttributeError, TypeError):
    view = st.selectbox("Navigation", NAV, index=NAV.index(st.session_state.view), label_visibility="collapsed")

if view and view != st.session_state.view:
    st.session_state.view = view

V = st.session_state.view


# ══════════════════════════════════════════════════════════════════════════════
# ── VIEW 1: MATCH OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if V == "Match Overview":
    st.html(html_match_hero(n_a, n_b, dur, total_f, fps_r, R["video_name"]))

    # ── Match Preview — frame thumbnails ──────────────────────────────────────
    # Use stored paths if available; fall back to scanning thumbs dir by name
    thumbs = [p for p in R.get("thumb_paths", []) if os.path.exists(p)]
    if not thumbs:
        thumb_dir = os.path.join("data", "output", "thumbs")
        vname_r   = R.get("video_name", "")
        if os.path.isdir(thumb_dir) and vname_r:
            import glob
            thumbs = sorted(glob.glob(os.path.join(thumb_dir, f"{vname_r}_thumb_*.jpg")))
    if thumbs:
        st.html(html_section_head("Match Preview",
                "Key frames extracted from the clip"))
        thumb_cols = st.columns(len(thumbs), gap="small")
        fstep = total_f // max(len(thumbs), 1)
        for i, (col, path) in enumerate(zip(thumb_cols, thumbs)):
            with col:
                approx_sec = round((fstep * i) / fps_r, 1)
                st.image(path, width='stretch')
                st.html(f"""<div style="text-align:center;font-size:0.62rem;
                            color:#475569;margin-top:2px;font-family:Inter,sans-serif;">
                            ⏱ ~{approx_sec}s</div>""")

    col_l, col_r = st.columns([6, 4], gap="large")

    with col_l:
        st.html(html_section_head("Match Story",
                "Auto-generated from tracking data — no AI guesswork"))
        for color, team, icon, text in match_story(insights):
            st.html(html_story_card(color, team, icon, text))

        st.html(html_section_head("Key Moments",
                "Tactical transitions detected during the clip"))
        km = insights.get("key_moments", [])
        st.html(html_key_moments(km[:6]))

    with col_r:
        st.html(html_section_head("Match MVP",
                "Highest combined distance + speed + sprint score"))
        mvp_t = int(team_s.get(mvp_id, -1))
        st.html(html_player_card(
            mvp_id, mvp_t,
            float(mvp_row["total_pixel_distance"]),
            float(mvp_row["top_speed_px_s"]),
            int(mvp_row["sprint_count"]),
            awards.get(mvp_id, []), mvp=True,
            role=roles.get(mvp_id, ""),
        ))

        st.html(html_section_head("Annotated Clip",
                "YOLOv8 detections with ByteTrack IDs"))
        vid_path = R["annotated_video"]
        if os.path.exists(vid_path):
            # Serve as bytes — works regardless of codec (avc1 or mp4v fallback)
            with open(vid_path, "rb") as vf:
                st.video(vf.read(), format="video/mp4")
        else:
            st.caption("Video not found — re-run analysis to generate it.")



# ══════════════════════════════════════════════════════════════════════════════
# ── VIEW 2: PLAYER EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif V == "Player Explorer":
    st.html(html_section_head("Player Explorer",
            "Click a player on the pitch to open their profile"))

    col_pitch, col_profile = st.columns([58, 42], gap="large")

    with col_pitch:
        # Interactive pitch
        p_fig  = pitch_figure(tracks)
        p_event = st.plotly_chart(p_fig, key="pitch", on_select="rerun",
                                   width='stretch')
        # Handle click
        if p_event and hasattr(p_event, "selection"):
            pts_sel = getattr(p_event.selection, "points", []) or []
            if pts_sel:
                cd = pts_sel[0].get("customdata")
                if cd is not None:
                    try:
                        st.session_state.player = int(cd[0] if hasattr(cd,"__len__") else cd)
                    except (TypeError, ValueError, IndexError):
                        pass
        st.caption("🔵 Team A  ·  🔴 Team B  ·  🟡 Referee  ·  🧤 = Goalkeeper  ·  Dot size = activity")

    with col_profile:
        sel = st.session_state.player
        if sel is not None and int(sel) in merged["track_id"].values:
            pr   = merged[merged["track_id"] == int(sel)].iloc[0]
            p_t  = int(team_s.get(int(sel), -1))
            p_c  = role_color(roles.get(int(sel), "")) if roles.get(int(sel)) else TCOL.get(p_t, "#94a3b8")
            st.html(html_player_card(
                int(sel), p_t,
                float(pr["total_pixel_distance"]),
                float(pr["top_speed_px_s"]),
                int(pr["sprint_count"]),
                awards.get(int(sel), []),
                mvp=(int(sel) == mvp_id),
                role=roles.get(int(sel), ""),
            ))
            mc1, mc2 = st.columns(2)
            mc1.metric("Frames Active", int(pr["num_frames"]))
            mc2.metric("Avg Speed", f"{float(pr['avg_speed_px_s']):.0f} px/s")
            st.plotly_chart(trajectory_fig(tracks, int(sel), p_t),
                            width='stretch')
            st.plotly_chart(heatmap_fig(tracks, int(sel), p_c),
                            width='stretch')
            st.plotly_chart(speed_fig(speed_df, int(sel), s_thresh),
                            width='stretch')
        else:
            st.html(f"""
            <div style="height:320px;display:flex;flex-direction:column;
                        align-items:center;justify-content:center;
                        border:1px dashed {BDR};border-radius:16px;
                        font-family:Inter,sans-serif;text-align:center;">
              <div style="font-size:2rem;margin-bottom:10px;color:#1a2844;">👆</div>
              <div style="color:#334155;font-size:0.85rem;">
                Click a player on the pitch<br>to view their profile
              </div>
            </div>""")

    # Player card grid
    st.html(html_section_head("All Players",
            f"{len(merged)} tracks — sorted by distance covered"))
    top_sorted = merged.sort_values("total_pixel_distance", ascending=False)
    for chunk in [top_sorted.iloc[i:i+4] for i in range(0, len(top_sorted), 4)]:
        cols = st.columns(4, gap="small")
        for i, (_, pr) in enumerate(chunk.iterrows()):
            with cols[i]:
                tid = int(pr["track_id"])
                tt  = int(team_s.get(tid, -1))
                st.html(html_player_card(tid, tt,
                    float(pr["total_pixel_distance"]),
                    float(pr["top_speed_px_s"]),
                    int(pr["sprint_count"]),
                    awards.get(tid, []),
                    mvp=(tid == mvp_id), compact=True,
                    role=roles.get(tid, ""),
                ))
                if st.button(f"Explore #{tid}", key=f"sel_{tid}", width='stretch'):
                    st.session_state.player = tid
                    st.session_state.view = "Player Explorer"
                    # Delete widget key so st.pills falls back to default= on rerun
                    st.session_state.pop("nav_pills", None)
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ── VIEW 3: TEAM BATTLE
# ══════════════════════════════════════════════════════════════════════════════
elif V == "Team Battle":
    st.html(html_section_head("Team Battle", "Head-to-head performance comparison"))

    ta = merged[merged["team"] == 0]
    tb = merged[merged["team"] == 1]

    def ts(df):
        return dict(
            players=len(df),
            dist=int(df["total_pixel_distance"].sum()),
            avg_dist=int(df["total_pixel_distance"].mean()) if len(df) else 0,
            top_spd=float(df["top_speed_px_s"].max()) if len(df) else 0,
            avg_spd=float(df["avg_speed_px_s"].mean()) if len(df) else 0,
            sprints=int(df["sprint_count"].sum()),
        )
    sa, sb = ts(ta), ts(tb)

    # Team headers
    hc1, hc2, hc3 = st.columns([5, 2, 5], gap="large")
    for col, label, col_h, n, color in [
        (hc1, "Team A", TA, sa["players"], TA),
        (hc3, "Team B", TB, sb["players"], TB),
    ]:
        rgb = _rgb(color)
        with col:
            st.html(f"""
            <div style="text-align:center;padding:28px;
                        background:linear-gradient(135deg,rgba({rgb},0.12),{SRF});
                        border:1px solid rgba({rgb},0.3);border-radius:16px;
                        font-family:'Inter',sans-serif;">
              <div style="font-size:0.6rem;color:{color};text-transform:uppercase;
                          letter-spacing:0.12em;font-weight:700;margin-bottom:4px;">{label}</div>
              <div style="font-size:4rem;font-weight:900;color:{color};
                          font-family:'Rajdhani',sans-serif;line-height:1;">{n}</div>
              <div style="font-size:0.7rem;color:#475569;margin-top:4px;">players</div>
            </div>""")
    with hc2:
        st.html(f"""
        <div style="height:100%;display:flex;align-items:center;justify-content:center;
                    font-size:0.85rem;font-weight:800;color:#334155;
                    font-family:'Rajdhani',sans-serif;letter-spacing:0.15em;">VS</div>""")

    st.divider()
    st.html(html_section_head("Head-to-Head Stats"))

    st.html("".join([
        html_battle_bar("Total Distance (px)", f"{sa['dist']:,}", f"{sb['dist']:,}"),
        html_battle_bar("Avg Distance Per Player (px)", sa["avg_dist"], sb["avg_dist"]),
        html_battle_bar("Highest Top Speed (px/s)", f"{sa['top_spd']:.0f}", f"{sb['top_spd']:.0f}"),
        html_battle_bar("Avg Speed (px/s)", f"{sa['avg_spd']:.0f}", f"{sb['avg_spd']:.0f}"),
        html_battle_bar("Total Sprint Bursts", sa["sprints"], sb["sprints"]),
    ]))

    cmp_sum  = insights.get("compactness_summary", {})
    prs_sum  = insights.get("pressing_summary", {})
    ca_cm = cmp_sum.get("team_a", {}).get("mean")
    cb_cm = cmp_sum.get("team_b", {}).get("mean")
    pa_pr = prs_sum.get("team_a", {}).get("avg_press_dist")
    pb_pr = prs_sum.get("team_b", {}).get("avg_press_dist")

    extra = ""
    if isinstance(ca_cm, float) and isinstance(cb_cm, float):
        extra += html_battle_bar("Avg Compactness (px — lower = tighter)",
                                  f"{ca_cm:.0f}", f"{cb_cm:.0f}", higher_better=False)
    if isinstance(pa_pr, float) and isinstance(pb_pr, float):
        extra += html_battle_bar("Pressing Distance (px — lower = harder press)",
                                  f"{pa_pr:.0f}", f"{pb_pr:.0f}", higher_better=False)
    if extra:
        st.html(extra)

    # Heatmaps
    st.html(html_section_head("Activity Zones", "Where each team spent their time"))
    hc_a, hc_b = st.columns(2, gap="large")
    with hc_a:
        st.plotly_chart(team_heatmap_fig(tracks, 0, TA, "Team A · Activity Zones"),
                        width='stretch')
    with hc_b:
        st.plotly_chart(team_heatmap_fig(tracks, 1, TB, "Team B · Activity Zones"),
                        width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# ── VIEW 4: TACTICAL STORY
# ══════════════════════════════════════════════════════════════════════════════
elif V == "Tactical Story":
    st.html(html_section_head("Tactical Story",
            "Formation, territory, and tactical patterns — derived from player positions"))

    formation   = insights.get("formation", {})
    territorial = insights.get("territorial", {})

    if formation and territorial:
        st.plotly_chart(formation_pitch_fig(formation, territorial),
                        width='stretch')
        fc1, fc2 = st.columns(2, gap="large")
        for col, key, label, color in [(fc1,"team_a","Team A",TA),(fc2,"team_b","Team B",TB)]:
            data  = formation.get(key, {})
            raw_f = data.get("formation", "")
            bands = data.get("bands", [])
            rgb   = _rgb(color)
            # Sanitise formation label
            if not raw_f or raw_f in ("", "unknown", "?") or (str(raw_f).isdigit() and int(str(raw_f)) < 3):
                f_display = "Analysing…"
                f_sub = "Clip too short for reliable formation fingerprinting."
            else:
                f_display = f"~{raw_f}"
                f_sub = f"Band split: {' – '.join(str(b) for b in bands)}"
            with col:
                st.html(f"""
                <div style="background:rgba({rgb},0.07);border:1px solid rgba({rgb},0.22);
                            border-left:3px solid {color};border-radius:8px;
                            padding:20px 24px;font-family:'Inter',sans-serif;">
                  <div style="font-size:0.55rem;color:{color};text-transform:uppercase;
                              letter-spacing:0.14em;font-weight:700;margin-bottom:6px;">{label}</div>
                  <div style="font-size:2.4rem;font-weight:900;color:#f8fafc;
                              font-family:'Rajdhani',sans-serif;line-height:1;">{f_display}</div>
                  <div style="font-size:0.72rem;color:#475569;margin-top:6px;">{f_sub}</div>
                  <div style="font-size:0.68rem;color:#334155;margin-top:8px;line-height:1.55;">
                    Estimate from average positions. Not ground truth.
                  </div>
                </div>""")
    else:
        st.info("Insufficient team data to render formation diagram.")

    st.html(html_section_head("Match Narrative",
            "What the data tells us — no AI, pure analytics"))
    for color, team, icon, text in match_story(insights):
        st.html(html_story_card(color, team, icon, text))

    st.html(html_section_head("Tactical Transitions",
            "Frames where shape or pressing changed significantly"))
    km = insights.get("key_moments", [])
    st.html(html_key_moments(km))

    # Timeline charts
    comp_df  = insights.get("compactness_df", pd.DataFrame())
    press_df = insights.get("pressing_df",    pd.DataFrame())

    if not comp_df.empty and "team_a_compactness" in comp_df.columns:
        st.html(html_section_head("Compactness Over Time",
                "Lower = tighter, more compact shape"))
        st.plotly_chart(
            timeline_fig(comp_df, "team_a_compactness", "team_b_compactness",
                         "Compactness Timeline", "Avg inter-player distance (px)"),
            width='stretch',
        )

    if not press_df.empty and "team_a_press" in press_df.columns:
        st.html(html_section_head("Pressing Intensity Over Time",
                "Lower = closer to opponents = pressing harder"))
        st.plotly_chart(
            timeline_fig(press_df, "team_a_press", "team_b_press",
                         "Pressing Distance Timeline", "Nearest opponent dist (px)"),
            width='stretch',
        )


# ══════════════════════════════════════════════════════════════════════════════
# ── VIEW 5: FULL ROSTER
# ══════════════════════════════════════════════════════════════════════════════
elif V == "Full Roster":
    st.html(html_section_head("Full Roster",
            f"{len(merged)} tracks  ·  🧤 = GK  ·  🟡 = Referee"))

    try:
        filt = st.segmented_control(
            "Filter", ["All", "Team A", "Team B", "Referee"],
            default="All", key="roster_filter"
        )
    except AttributeError:
        filt = st.radio("Filter", ["All", "Team A", "Team B", "Referee"],
                        horizontal=True, key="roster_filter_rb")

    if filt == "All":
        filtered = merged
    else:
        matched_tids = [tid for tid, r in roles.items() if filt in r]
        filtered = merged[merged["track_id"].isin(matched_tids)]
    filtered = filtered.sort_values("total_pixel_distance", ascending=False)

    for chunk in [filtered.iloc[i:i+3] for i in range(0, len(filtered), 3)]:
        cols = st.columns(3, gap="medium")
        for i, (_, pr) in enumerate(chunk.iterrows()):
            with cols[i]:
                tid = int(pr["track_id"])
                tt  = int(team_s.get(tid, -1))
                st.html(html_player_card(
                    tid, tt,
                    float(pr["total_pixel_distance"]),
                    float(pr.get("top_speed_px_s", 0)),
                    int(pr.get("sprint_count", 0)),
                    awards.get(tid, []),
                    mvp=(tid == mvp_id),
                    role=roles.get(tid, ""),
                ))
                if st.button(f"Explore #{tid}", key=f"r_{tid}", width='stretch'):
                    st.session_state.player = tid
                    st.session_state.view = "Player Explorer"
                    # Delete widget key so st.pills falls back to default= on rerun
                    st.session_state.pop("nav_pills", None)
                    st.rerun()


# ─── FOOTER (every page) ──────────────────────────────────────────────────────
st.html("""
<div style="margin-top:64px;padding:24px 0 16px;
            border-top:1px solid #0f1c2e;text-align:center;
            font-family:Inter,sans-serif;">
  <span style="font-size:0.72rem;color:#1e3352;letter-spacing:0.04em;">
    Made with&nbsp;
    <span style="color:#e2364a;font-size:0.8rem;">&#10084;</span>
    &nbsp;by&nbsp;
    <span style="color:#2a4a6b;font-weight:600;">shisht</span>
  </span>
</div>
""")



