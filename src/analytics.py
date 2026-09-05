"""
TacticScope - Step 2: Analytics Engine

WHAT THIS DOES:
Reads the tracks.csv produced by detect_and_track.py (one row per detected
player per frame: frame, track_id, bbox, center point) and computes, per
track_id:

1. TRAJECTORY  - the sequence of (x, y) positions over time, saved as a plot
2. DISTANCE COVERED - total pixel distance traveled (sum of frame-to-frame
   displacement)
3. HEATMAP - a 2D histogram of where that track spent time on screen

IMPORTANT FRAMING (matches what we discussed):
Each track_id is treated as its own independent "segment" - if a player's ID
switched partway through the clip (camera cut, occlusion), that shows up as
TWO separate track_ids in these results, not one continuous player. That's a
known, honest limitation of tracking-by-detection - not something this script
tries to silently paper over. A production system would add appearance-based
re-identification (ReID) to stitch segments back together; that's out of
scope here and worth naming explicitly as future work.

OUTPUTS (per run):
- data/output/<video>_summary.csv       one row per track_id: distance, frame
                                          count, avg position, etc.
- data/output/<video>_trajectories.png   all trajectories overlaid on one plot
- data/output/<video>_heatmap_overall.png combined heatmap of all detections
- data/output/heatmaps/<video>_track<ID>.png   per-track heatmap (top N tracks
                                          by frame count, to avoid generating
                                          dozens of near-empty heatmaps for
                                          very short/noisy tracks)

RUN:
    python src/analytics.py --csv data/output/sample_tracks.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt


def compute_distance_per_track(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each track_id, sum up frame-to-frame Euclidean displacement of the
    center point (cx, cy). This is PIXEL distance, not real-world meters -
    a known simplification (see project blueprint: real-world distance would
    need homography/pitch calibration, which is a stretch goal, not core).
    """
    records = []
    for track_id, group in df.groupby("track_id"):
        group = group.sort_values("frame")
        coords = group[["cx", "cy"]].to_numpy()

        if len(coords) < 2:
            distance = 0.0
        else:
            diffs = np.diff(coords, axis=0)
            step_dists = np.sqrt((diffs ** 2).sum(axis=1))
            distance = float(step_dists.sum())

        records.append({
            "track_id": track_id,
            "num_frames": len(group),
            "first_frame": int(group["frame"].min()),
            "last_frame": int(group["frame"].max()),
            "total_pixel_distance": round(distance, 2),
            "avg_x": round(group["cx"].mean(), 2),
            "avg_y": round(group["cy"].mean(), 2),
        })

    summary = pd.DataFrame(records).sort_values(
        "total_pixel_distance", ascending=False
    ).reset_index(drop=True)
    return summary


def plot_trajectories(df: pd.DataFrame, output_path: str, top_n: int = 15):
    """
    Overlay the movement path of each track_id on one plot. Limited to the
    top N tracks by frame count so the plot stays readable (very short/noisy
    tracks would just clutter it).
    """
    frame_counts = df.groupby("track_id").size().sort_values(ascending=False)
    top_ids = frame_counts.head(top_n).index

    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap("tab20")

    for i, track_id in enumerate(top_ids):
        group = df[df["track_id"] == track_id].sort_values("frame")
        plt.plot(
            group["cx"], group["cy"],
            marker="o", markersize=2, linewidth=1,
            color=cmap(i % 20), label=f"ID {track_id}"
        )

    plt.gca().invert_yaxis()  # image coords: y increases downward
    plt.title("Player Trajectories (top tracks by frame count)")
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.legend(fontsize=7, loc="upper right", ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_heatmap(x, y, title: str, output_path: str, bins: int = 40):
    """2D histogram heatmap of position data."""
    plt.figure(figsize=(8, 5))
    plt.hist2d(x, y, bins=bins, cmap="hot")
    plt.gca().invert_yaxis()
    plt.colorbar(label="Time spent (frame count)")
    plt.title(title)
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_analytics(csv_path: str, output_dir: str = "data/output", top_n_heatmaps: int = 5) -> tuple:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"No tracking data found in {csv_path}")

    video_name = os.path.splitext(os.path.basename(csv_path))[0].replace("_tracks", "")
    os.makedirs(output_dir, exist_ok=True)
    heatmap_dir = os.path.join(output_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    # 1. Per-track summary (distance, frame count, etc.)
    summary = compute_distance_per_track(df)
    summary_path = os.path.join(output_dir, f"{video_name}_summary.csv")
    summary.to_csv(summary_path, index=False)

    # 2. Trajectories overlay
    traj_path = os.path.join(output_dir, f"{video_name}_trajectories.png")
    plot_trajectories(df, traj_path)

    # 3. Overall heatmap (all detections combined)
    overall_heatmap_path = os.path.join(output_dir, f"{video_name}_heatmap_overall.png")
    plot_heatmap(df["cx"], df["cy"], "Overall Player Position Heatmap", overall_heatmap_path)

    # 4. Per-track heatmaps for the top N tracks (by frame count - i.e. the
    #    tracks with enough data to make a heatmap meaningful)
    top_tracks = summary.sort_values("num_frames", ascending=False).head(top_n_heatmaps)
    for _, row in top_tracks.iterrows():
        track_id = int(row["track_id"])
        track_df = df[df["track_id"] == track_id]
        track_heatmap_path = os.path.join(heatmap_dir, f"{video_name}_track{track_id}.png")
        plot_heatmap(
            track_df["cx"], track_df["cy"],
            f"Heatmap - Track ID {track_id} ({int(row['num_frames'])} frames)",
            track_heatmap_path,
        )

    print(f"Analytics complete for {video_name}")
    print(f"  Tracks found: {len(summary)}")
    print(f"  Summary CSV:  {summary_path}")
    print(f"  Trajectories: {traj_path}")
    print(f"  Overall heatmap: {overall_heatmap_path}")
    print(f"  Per-track heatmaps ({len(top_tracks)}): {heatmap_dir}/")
    print("\nTop 5 tracks by distance covered (pixels):")
    print(summary.head(5).to_string(index=False))

    # Return both the summary and the raw df so callers (e.g. app.py) don't
    # need to re-read the CSV from disk for downstream modules.
    return summary, df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to tracks.csv from detect_and_track.py")
    parser.add_argument("--output_dir", default="data/output")
    parser.add_argument("--top_n_heatmaps", type=int, default=5)
    args = parser.parse_args()

    run_analytics(args.csv, args.output_dir, args.top_n_heatmaps)
