"""
TacticScope - Speed Estimator

WHAT THIS DOES:
Given the tracks DataFrame (frame, track_id, cx, cy) and the video FPS,
computes per-player speed metrics:

  - Per-frame instantaneous speed (pixels/sec), smoothed with a rolling window
    to suppress detection jitter (single-frame bbox jumps that aren't real
    player movement)
  - Top speed (max of the smoothed series)
  - Average speed (mean across all frames where the track is visible)
  - Sprint count: number of distinct sprint bouts where speed exceeds a
    clip-level threshold (default: 75th percentile of all smoothed speeds)
  - Sprint frames: list of frame indices during sprints (useful for the
    frame-scrubber / key moment detection in tactical_insights.py)

WHY PIXELS/SEC NOT KM/H:
Converting to real-world units requires a homography (mapping pixel coords
to pitch coordinates in metres). That requires manual calibration per video
(picking 4+ known pitch points) and is finicky across camera angles --
noted as a stretch goal in the blueprint. Pixel-based speed is still a
meaningful RELATIVE metric: rankings and sprint counts are valid, and the
unit is surfaced honestly in the UI.

WHY ROLLING SMOOTHING:
YOLOv8's detected bounding box centre can jump a few pixels frame-to-frame
even for a stationary player (detector noise). Without smoothing, this
shows up as noise in the speed signal. A 5-frame rolling mean suppresses
this while still capturing genuine player bursts.

RUN (standalone test):
    python src/speed_estimator.py --csv data/output/sample_tracks.csv --fps 25
"""

import argparse

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Core computation
# ----------------------------------------------------------------------------

def compute_speed_per_frame(
    df: pd.DataFrame,
    fps: float,
    smooth_window: int = 5,
) -> pd.DataFrame:
    """
    Adds a 'speed_px_per_sec' column to the dataframe (one value per row).

    Steps:
      1. Sort each track by frame
      2. Compute displacement between consecutive frames: sqrt(Δcx² + Δcy²)
      3. Multiply by FPS to get pixels/second (displacement is per-frame, not
         per-second)
      4. Apply a rolling mean over `smooth_window` frames within each track
      5. Fill the very first frame of each track (no predecessor) with 0
    """
    df = df.copy().sort_values(["track_id", "frame"]).reset_index(drop=True)

    speed_col = []
    for _, group in df.groupby("track_id", sort=False):
        coords = group[["cx", "cy"]].to_numpy()
        # displacement per frame transition
        diffs = np.diff(coords, axis=0)
        step_dists = np.sqrt((diffs ** 2).sum(axis=1))
        # convert to per-second
        speeds_raw = np.concatenate([[0.0], step_dists * fps])
        # rolling smooth within track
        series = pd.Series(speeds_raw)
        speeds_smooth = (
            series.rolling(window=smooth_window, min_periods=1, center=True)
            .mean()
            .to_numpy()
        )
        speed_col.extend(speeds_smooth.tolist())

    df["speed_px_per_sec"] = speed_col
    return df


def compute_speed_summary(
    df_with_speed: pd.DataFrame,
    sprint_percentile: float = 75.0,
    min_sprint_frames: int = 3,
) -> pd.DataFrame:
    """
    Produces a per-track speed summary DataFrame:

      track_id | avg_speed | top_speed | sprint_count | sprint_frames

    sprint_percentile: speeds above this percentile of the clip are
        classified as "sprinting". 75th percentile is a reasonable
        threshold that labels the fastest ~25% of movement bursts as sprints
        without overcounting normal jogging.

    min_sprint_frames: a sprint bout must last at least this many consecutive
        frames to count (avoids counting single-frame noise spikes).
    """
    if "speed_px_per_sec" not in df_with_speed.columns:
        raise ValueError("df must have 'speed_px_per_sec' column -- run compute_speed_per_frame first")

    # Global sprint threshold (clips vary widely in resolution/FPS so this
    # is relative to the clip rather than an absolute pixel value)
    sprint_threshold = float(np.percentile(df_with_speed["speed_px_per_sec"], sprint_percentile))

    records = []
    for track_id, group in df_with_speed.groupby("track_id"):
        group = group.sort_values("frame")
        speeds = group["speed_px_per_sec"].to_numpy()
        frames = group["frame"].to_numpy()

        avg_speed = float(np.mean(speeds))
        top_speed = float(np.max(speeds))

        # Sprint detection: find runs of consecutive frames above threshold
        is_sprint = speeds >= sprint_threshold
        sprint_count = 0
        sprint_frames_list = []
        i = 0
        while i < len(is_sprint):
            if is_sprint[i]:
                j = i
                while j < len(is_sprint) and is_sprint[j]:
                    j += 1
                bout_length = j - i
                if bout_length >= min_sprint_frames:
                    sprint_count += 1
                    sprint_frames_list.extend(frames[i:j].tolist())
                i = j
            else:
                i += 1

        records.append({
            "track_id": int(track_id),
            "avg_speed_px_s": round(avg_speed, 2),
            "top_speed_px_s": round(top_speed, 2),
            "sprint_count": sprint_count,
            "sprint_threshold_px_s": round(sprint_threshold, 2),
            "sprint_frames": sprint_frames_list,
        })

    result = pd.DataFrame(records).sort_values("top_speed_px_s", ascending=False).reset_index(drop=True)
    return result


# ----------------------------------------------------------------------------
# Public entry point (called by app.py pipeline orchestrator and by analytics)
# ----------------------------------------------------------------------------

def run_speed_estimation(
    tracks_df: pd.DataFrame,
    fps: float,
    smooth_window: int = 5,
    sprint_percentile: float = 75.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        tracks_df_with_speed: original df + 'speed_px_per_sec' column
        speed_summary: per-track summary (avg/top speed, sprints)
    """
    df_speed = compute_speed_per_frame(tracks_df, fps, smooth_window)
    summary = compute_speed_summary(df_speed, sprint_percentile)
    return df_speed, summary


# ----------------------------------------------------------------------------
# Standalone test
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to tracks.csv")
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df_speed, summary = run_speed_estimation(df, args.fps)

    print(f"Speed estimation complete. Clip FPS: {args.fps}")
    print(f"Sprint threshold: {summary['sprint_threshold_px_s'].iloc[0]} px/s")
    print("\nTop 10 tracks by top speed:")
    print(summary.head(10)[["track_id", "avg_speed_px_s", "top_speed_px_s", "sprint_count"]].to_string(index=False))
