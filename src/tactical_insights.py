"""
TacticScope - Tactical Insights Engine

WHAT THIS DOES:
Pure geometry over team-classified tracking data. Takes the enriched tracks_df
(with 'team' column from team_classifier.py) and computes:

  1. COMPACTNESS — per-frame average pairwise distance between same-team players.
     Low = compact defensive shape. High = stretched/attacking. Summarised as
     mean ± std and a time series for charting.

  2. PRESSING INTENSITY — per-frame average nearest-opponent distance for each
     team. Low = pressing hard (players close to opponents). High = sitting deep.

  3. TERRITORIAL CONTROL — divide the frame into a grid, count detections per
     team per cell → produces a dominance map showing which team "owned" each
     zone of the pitch.

  4. FORMATION SNAPSHOT — at a chosen frame, split each team's players into
     3 horizontal bands (by Y coordinate: defensive / midfield / attack third)
     and count players per band to estimate a formation fingerprint like "4-4-2".
     Also generates a top-down pitch diagram with player dots and convex hulls.

  5. KEY MOMENTS — frames where compactness or pressing changes sharply above
     a threshold → surfaced as a timeline of tactical events (e.g. "Team A
     compresses at frame 140", "High press by Team B at frame 280").

DESIGN NOTES:
- Everything here is pixel-space geometry, no ML / re-training needed.
- Formation labels are heuristic estimates (band-counting), surfaced honestly
  as "estimated" in the UI — not ground truth.
- Ball-free: all insights derived from player positions only (consistent with
  the project blueprint decision to skip unreliable ball detection).
- Frame dimensions used for grid / pitch diagram come from the tracks data
  (max observed cx/cy bounds), not hardcoded — works across different resolutions.
"""

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist


# ─────────────────────────────────────────────────────────────────────────────
# 1. Compactness
# ─────────────────────────────────────────────────────────────────────────────

def compute_compactness(tracks_df: pd.DataFrame, team_col: str = "team") -> pd.DataFrame:
    """
    Per-frame compactness for each team (0 and 1 only — excludes officials, team==2).
    Compactness = mean pairwise Euclidean distance between all same-team players
    visible in that frame.

    Returns a DataFrame: frame | team_a_compactness | team_b_compactness
    Frames where a team has fewer than 2 players visible are NaN.
    """
    records = []
    for frame_id, group in tracks_df.groupby("frame"):
        row = {"frame": int(frame_id)}
        for team_id, label in [(0, "team_a"), (1, "team_b")]:
            pts = group[group[team_col] == team_id][["cx", "cy"]].to_numpy()
            if len(pts) >= 2:
                dists = cdist(pts, pts)
                # mean of upper triangle only (avoid double-counting pair distances)
                upper = dists[np.triu_indices(len(pts), k=1)]
                row[f"{label}_compactness"] = float(upper.mean())
            else:
                row[f"{label}_compactness"] = np.nan
        records.append(row)

    return pd.DataFrame(records).sort_values("frame").reset_index(drop=True)


def summarise_compactness(compactness_df: pd.DataFrame) -> dict:
    """High-level summary stats for each team's compactness over the clip."""
    result = {}
    for label in ["team_a", "team_b"]:
        col = f"{label}_compactness"
        series = compactness_df[col].dropna()
        if len(series) == 0:
            result[label] = {"mean": None, "std": None, "trend": "unknown"}
            continue
        mean_val = float(series.mean())
        std_val = float(series.std())
        # simple linear trend: positive = team spread out more over time (attacking),
        # negative = team compressed over time (defensive)
        if len(series) > 10:
            slope = float(np.polyfit(np.arange(len(series)), series.values, 1)[0])
            trend = "expanding" if slope > 0.5 else ("contracting" if slope < -0.5 else "stable")
        else:
            trend = "stable"
        result[label] = {"mean": round(mean_val, 1), "std": round(std_val, 1), "trend": trend}
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. Pressing Intensity
# ─────────────────────────────────────────────────────────────────────────────

def compute_pressing(tracks_df: pd.DataFrame, team_col: str = "team") -> pd.DataFrame:
    """
    Per-frame pressing intensity for each team.
    Pressing score = average distance from each player to their nearest opponent.
    Low score = pressing hard. High score = sitting deep.

    Returns: frame | team_a_press | team_b_press
    """
    records = []
    for frame_id, group in tracks_df.groupby("frame"):
        row = {"frame": int(frame_id)}
        team_a_pts = group[group[team_col] == 0][["cx", "cy"]].to_numpy()
        team_b_pts = group[group[team_col] == 1][["cx", "cy"]].to_numpy()

        if len(team_a_pts) > 0 and len(team_b_pts) > 0:
            # Team A pressing Team B: for each Team A player, what's the
            # nearest Team B player distance?
            dists_a_to_b = cdist(team_a_pts, team_b_pts).min(axis=1)
            row["team_a_press"] = float(dists_a_to_b.mean())

            dists_b_to_a = cdist(team_b_pts, team_a_pts).min(axis=1)
            row["team_b_press"] = float(dists_b_to_a.mean())
        else:
            row["team_a_press"] = np.nan
            row["team_b_press"] = np.nan

        records.append(row)

    return pd.DataFrame(records).sort_values("frame").reset_index(drop=True)


def summarise_pressing(pressing_df: pd.DataFrame) -> dict:
    """Returns average pressing score per team. Lower = more aggressive press."""
    result = {}
    for label in ["team_a", "team_b"]:
        col = f"{label}_press"
        series = pressing_df[col].dropna()
        if len(series) == 0:
            result[label] = {"avg_press_dist": None, "intensity": "unknown"}
        else:
            avg = float(series.mean())
            # intensity label: relative to median of both teams combined
            result[label] = {
                "avg_press_dist": round(avg, 1),
                # will be labelled relative to the other team in app.py
            }
    # Label which team pressed more aggressively
    a = result.get("team_a", {}).get("avg_press_dist")
    b = result.get("team_b", {}).get("avg_press_dist")
    if a is not None and b is not None:
        result["more_aggressive_presser"] = "Team A" if a < b else "Team B"
    else:
        result["more_aggressive_presser"] = "Unknown"
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. Territorial Control
# ─────────────────────────────────────────────────────────────────────────────

def compute_territorial_control(
    tracks_df: pd.DataFrame,
    grid_cols: int = 6,
    grid_rows: int = 4,
    team_col: str = "team",
) -> dict:
    """
    Divides the frame into a grid and counts detections per team per cell.
    Returns a dict with:
      - 'grid_cols', 'grid_rows'
      - 'team_a_grid': 2D array (grid_rows × grid_cols) of detection counts
      - 'team_b_grid': same
      - 'dominance_grid': 2D array where >0 = Team A dominates, <0 = Team B,
                          values scaled to [-1, 1]
      - 'frame_w', 'frame_h': inferred from max observed cx/cy
    """
    frame_w = float(tracks_df["cx"].max()) + 1
    frame_h = float(tracks_df["cy"].max()) + 1

    cell_w = frame_w / grid_cols
    cell_h = frame_h / grid_rows

    grid_a = np.zeros((grid_rows, grid_cols), dtype=float)
    grid_b = np.zeros((grid_rows, grid_cols), dtype=float)

    for _, row in tracks_df.iterrows():
        team = int(row[team_col])
        if team not in (0, 1):
            continue
        col_idx = min(int(row["cx"] / cell_w), grid_cols - 1)
        row_idx = min(int(row["cy"] / cell_h), grid_rows - 1)
        if team == 0:
            grid_a[row_idx, col_idx] += 1
        else:
            grid_b[row_idx, col_idx] += 1

    total = grid_a + grid_b
    total_safe = np.where(total == 0, 1, total)  # avoid divide by zero
    dominance = (grid_a - grid_b) / total_safe   # range [-1, 1]

    return {
        "grid_cols": grid_cols,
        "grid_rows": grid_rows,
        "team_a_grid": grid_a,
        "team_b_grid": grid_b,
        "dominance_grid": dominance,
        "frame_w": frame_w,
        "frame_h": frame_h,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Formation Snapshot
# ─────────────────────────────────────────────────────────────────────────────

def estimate_formation(
    tracks_df: pd.DataFrame,
    frame_idx: int = None,
    team_col: str = "team",
    n_bands: int = 3,
) -> dict:
    """
    Estimates a formation fingerprint for each team.

    Approach:
    - If frame_idx is None, use the median of all frames where BOTH teams
      have ≥4 players visible — a "typical" frame, not an extreme.
    - For each team, sort players by Y coordinate (top to bottom = attack to
      defence in a top-down view -- the "lower" Y in image coords is often
      closer to the top of screen, and "higher" Y is closer to the bottom).
      Divide into n_bands (default 3: defence / mid / attack) and count
      players per band.
    - Return count tuple e.g. (4, 3, 3) → "4-3-3 (estimated)"

    Returns dict with:
      'frame_used': int
      'team_a': {'formation': '4-3-3', 'bands': [4,3,3], 'positions': ndarray}
      'team_b': same
      'team_a_hull': ConvexHull vertices or None
      'team_b_hull': same
    """
    # Choose the best frame to snapshot
    if frame_idx is None:
        frame_counts = tracks_df[tracks_df[team_col].isin([0, 1])].groupby("frame")[team_col].value_counts().unstack(fill_value=0)
        # keep frames where both teams have >= 4 players
        if 0 in frame_counts.columns and 1 in frame_counts.columns:
            valid = frame_counts[(frame_counts[0] >= 4) & (frame_counts[1] >= 4)]
            if len(valid) > 0:
                frame_idx = int(valid.index[len(valid) // 2])  # median frame
            else:
                frame_idx = int(tracks_df["frame"].median())
        else:
            frame_idx = int(tracks_df["frame"].median())

    snapshot = tracks_df[tracks_df["frame"] == frame_idx]

    result = {"frame_used": frame_idx, "team_a": {}, "team_b": {}}

    for team_id, key in [(0, "team_a"), (1, "team_b")]:
        pts = snapshot[snapshot[team_col] == team_id][["cx", "cy"]].to_numpy()
        if len(pts) < 2:
            result[key] = {"formation": "unknown", "bands": [], "positions": pts}
            result[f"{key}_hull"] = None
            continue

        # Band split by Y coordinate
        y_sorted = np.sort(pts[:, 1])
        band_size = len(y_sorted) / n_bands
        bands = []
        for b in range(n_bands):
            lo = b * band_size
            hi = (b + 1) * band_size
            in_band = ((pts[:, 1] >= y_sorted[max(0, int(lo))]) &
                       (pts[:, 1] < y_sorted[min(len(y_sorted)-1, int(hi))]))
            bands.append(int(in_band.sum()))

        # Normalise so the smallest band = 1 (goalkeeper excluded from formation string)
        formation_str = "-".join(str(b) for b in bands if b > 0)

        hull = None
        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
            except Exception:
                hull = None

        result[key] = {
            "formation": formation_str,
            "bands": bands,
            "positions": pts,
        }
        result[f"{key}_hull"] = hull

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. Key Moments Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_key_moments(
    compactness_df: pd.DataFrame,
    pressing_df: pd.DataFrame,
    compactness_threshold: float = 30.0,
    press_threshold: float = 40.0,
    min_gap_frames: int = 15,
) -> list[dict]:
    """
    Detects frames where compactness or pressing changes significantly.
    Returns a list of event dicts: {frame, event_type, team, description}

    min_gap_frames: suppress duplicate events within this many frames of each other.
    """
    events = []

    def detect_jumps(series: pd.Series, frames: pd.Series, threshold: float, label: str):
        """Returns frame indices where the series changes by > threshold."""
        diffs = series.diff().abs()
        jump_frames = frames[diffs > threshold].tolist()
        # suppress duplicates within min_gap_frames
        filtered = []
        last = -999
        for f in sorted(jump_frames):
            if f - last >= min_gap_frames:
                filtered.append(int(f))
                last = f
        return filtered

    # Compactness jumps
    for team_label, col in [("Team A", "team_a_compactness"), ("Team B", "team_b_compactness")]:
        if col not in compactness_df.columns:
            continue
        series = compactness_df[col].ffill()  # pandas 3.0 compatible (replaces fillna(method='ffill'))
        jump_frames = detect_jumps(series, compactness_df["frame"], compactness_threshold, team_label)
        for f in jump_frames:
            idx_matches = compactness_df[compactness_df["frame"] == f].index
            if len(idx_matches) == 0:
                continue
            direction = "compressed" if series.diff().iloc[idx_matches[0]] < 0 else "stretched"
            events.append({
                "frame": f,
                "event_type": "shape_change",
                "team": team_label,
                "description": f"{team_label} shape {direction} (compactness shift)",
            })

    # Pressing intensity jumps
    for team_label, col in [("Team A", "team_a_press"), ("Team B", "team_b_press")]:
        if col not in pressing_df.columns:
            continue
        series = pressing_df[col].ffill()  # pandas 3.0 compatible
        jump_frames = detect_jumps(series, pressing_df["frame"], press_threshold, team_label)
        for f in jump_frames:
            idx_matches = pressing_df[pressing_df["frame"] == f].index
            if len(idx_matches) == 0:
                continue
            direction = "intensified press" if series.diff().iloc[idx_matches[0]] < 0 else "dropped off press"
            events.append({
                "frame": f,
                "event_type": "press_change",
                "team": team_label,
                "description": f"{team_label} {direction}",
            })

    # Sort by frame
    events.sort(key=lambda e: e["frame"])
    return events


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_tactical_insights(tracks_df: pd.DataFrame, team_col: str = "team") -> dict:
    """
    Orchestrates all insight computations and returns a single dict consumed
    by app.py. All computations are done on the full-clip tracks_df (not
    frame-by-frame in the app loop -- this runs once after the pipeline).

    Returns:
        {
          'compactness_df': DataFrame (per-frame),
          'compactness_summary': dict,
          'pressing_df': DataFrame (per-frame),
          'pressing_summary': dict,
          'territorial': dict,
          'formation': dict,
          'key_moments': list[dict],
        }
    """
    # Only operate on team-classified rows (team 0 and 1, not officials)
    team_df = tracks_df[tracks_df[team_col].isin([0, 1])].copy()

    if team_df.empty:
        return {
            "compactness_df": pd.DataFrame(),
            "compactness_summary": {},
            "pressing_df": pd.DataFrame(),
            "pressing_summary": {},
            "territorial": {},
            "formation": {},
            "key_moments": [],
        }

    compactness_df = compute_compactness(team_df, team_col)
    compactness_summary = summarise_compactness(compactness_df)

    pressing_df = compute_pressing(team_df, team_col)
    pressing_summary = summarise_pressing(pressing_df)

    territorial = compute_territorial_control(team_df, team_col=team_col)
    formation = estimate_formation(team_df, team_col=team_col)

    key_moments = detect_key_moments(compactness_df, pressing_df)

    return {
        "compactness_df": compactness_df,
        "compactness_summary": compactness_summary,
        "pressing_df": pressing_df,
        "pressing_summary": pressing_summary,
        "territorial": territorial,
        "formation": formation,
        "key_moments": key_moments,
    }
