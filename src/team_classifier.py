"""
TacticScope - Step 3: Team Classification (Jersey Color Clustering)

WHAT THIS DOES:
Reads the original video + tracks.csv, and for each detected player crops
their JERSEY region (not the whole bounding box - see why below), extracts
a representative color, averages that color per track_id across all frames
that track appears in, then runs k-means (k=2) on those per-track average
colors to split all tracks into "Team A" / "Team B".

WHY NOT USE THE WHOLE BOUNDING BOX:
A player's full bounding box includes grass (green background), skin tone,
hair, shorts, socks, and shoes - all of that would dilute/bias the color
signal away from the jersey, which is the actual team signal. Instead we
crop just the upper-middle portion of the box (roughly torso height, and
narrowed a bit horizontally) to bias toward jersey pixels and away from
background/skin.

WHY K-MEANS AND NOT A TRAINED CLASSIFIER:
No labeled data is needed - this works because the two teams' kits are
(almost always) visually distinct colors. It's simple enough to fully
understand and explain, and doesn't require training data collection.
Two clusters (k=2) matches the two-team assumption; this would need
adjustment (or manual exclusion) for a clip that also includes the referee
or goalkeepers in distinctly different kit colors - worth naming as a real
limitation.

WHY WE AGGREGATE PER TRACK (not per-frame classification):
A single frame's jersey crop can be noisy (motion blur, partial occlusion,
lighting/shadow). Averaging color across every frame a track appears in
gives a much more stable signal before clustering - this is a simple form
of "temporal smoothing" and worth mentioning as a design choice.

OUTPUTS:
- data/output/<video>_tracks_with_teams.csv   original tracking data + team
                                                 column (0 or 1) per row
- data/output/<video>_team_annotated.mp4       annotated video with boxes
                                                 colored by team (blue vs red)

RUN:
    python src/team_classifier.py --video data/videos/sample.mp4 --csv data/output/sample_tracks.csv
"""

import argparse
import os

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def crop_jersey_region(frame, x1, y1, x2, y2):
    """
    Crop the jersey-likely region of a player's bounding box: upper ~50% of
    the box height (torso area, avoiding legs/socks/shoes and, at the top,
    avoiding too much head/hair), and narrowed slightly on both sides to
    avoid overlapping detections/background at the box edges.
    """
    h = y2 - y1
    w = x2 - x1

    top = y1 + 0.15 * h      # skip a bit of head/neck
    bottom = y1 + 0.55 * h   # end before shorts
    left = x1 + 0.2 * w
    right = x2 - 0.2 * w

    top, bottom = int(max(0, top)), int(bottom)
    left, right = int(max(0, left)), int(right)

    crop = frame[top:bottom, left:right]
    return crop


def dominant_color(crop, k=1):
    """
    Get the dominant (mean) color of a small image crop as (B, G, R) - using
    simple mean is sufficient here since we're already cropping to a
    jersey-focused region; k-means per-crop (k>1) would separate e.g. jersey
    vs. shorts colors, but plain mean is a reasonable simplification for a
    project at this scope.
    """
    if crop.size == 0:
        return None
    pixels = crop.reshape(-1, 3).astype(np.float32)
    return pixels.mean(axis=0)  # BGR


def extract_track_colors(video_path: str, df: pd.DataFrame, sample_every: int = 3):
    """
    Walk through the video once, and for frames that have detections, crop
    the jersey region for each detection and record its color.

    sample_every: only process every Nth frame's detections (default 3) to
    keep this fast - color doesn't change frame-to-frame, so we don't need
    every single frame's data to get a stable per-track average.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    # group detections by frame for fast lookup
    frame_to_rows = {}
    for _, row in df.iterrows():
        if row["frame"] % sample_every != 0:
            continue
        frame_to_rows.setdefault(int(row["frame"]), []).append(row)

    track_colors = {}  # track_id -> list of colors (BGR arrays)

    frame_idx = 0
    max_frame = df["frame"].max()
    while frame_idx <= max_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in frame_to_rows:
            for row in frame_to_rows[frame_idx]:
                crop = crop_jersey_region(frame, row["x1"], row["y1"], row["x2"], row["y2"])
                color = dominant_color(crop)
                if color is not None:
                    track_colors.setdefault(int(row["track_id"]), []).append(color)

        frame_idx += 1

    cap.release()
    return track_colors


def cluster_teams(track_colors: dict, n_clusters: int = 3, separate_officials: bool = True):
    """
    Average each track's collected colors into one representative color,
    then k-means cluster those representative colors into n_clusters groups.

    WHY n_clusters=3 BY DEFAULT (not 2):
    Real match footage usually has a THIRD visually distinct kit color on
    screen: the referee/linesmen. Forcing k=2 makes every official get
    folded into whichever team's color happens to be closer - which is
    exactly the failure mode observed when testing this on real footage.
    Clustering with k=3 and then treating the two LARGEST clusters (by
    track count) as the real teams - with the smallest cluster labeled
    "Official" - is a simple, effective fix: teams have many players each,
    officials are typically just 1-3 people, so cluster size is a reliable
    way to tell them apart after clustering.

    Returns team_labels where values are:
      0, 1  -> the two team clusters (relabeled by size, largest = 0)
      2     -> "Official" cluster (only when separate_officials=True and
                n_clusters=3), i.e. referee/linesmen
    """
    track_ids = list(track_colors.keys())
    avg_colors = np.array([
        np.mean(track_colors[tid], axis=0) for tid in track_ids
    ])

    if len(track_ids) < n_clusters:
        # not enough tracks to form the requested number of clusters
        team_labels = {tid: 0 for tid in track_ids}
        return team_labels, avg_colors

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    raw_labels = kmeans.fit_predict(avg_colors)

    if n_clusters == 3 and separate_officials:
        # count tracks per cluster, sort clusters largest -> smallest
        counts = pd.Series(raw_labels).value_counts()
        cluster_order = counts.index.tolist()  # [largest, 2nd largest, smallest]

        # remap: largest -> 0, 2nd largest -> 1, smallest -> 2 ("Official")
        remap = {old: new for new, old in enumerate(cluster_order)}
        final_labels = [remap[l] for l in raw_labels]
    else:
        final_labels = raw_labels

    team_labels = {tid: int(label) for tid, label in zip(track_ids, final_labels)}
    return team_labels, avg_colors


def save_team_annotated_video(video_path, df, team_labels, output_path):
    """
    Redraw the video with bounding boxes colored by team assignment
    (team 0 = blue, team 1 = red). Tracks with no team assignment
    (e.g. too few samples) are drawn in gray.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    frame_to_rows = {}
    for _, row in df.iterrows():
        frame_to_rows.setdefault(int(row["frame"]), []).append(row)

    TEAM_COLORS = {0: (255, 100, 0), 1: (0, 0, 255), 2: (0, 255, 255)}  # BGR: team0=blue, team1=red, 2(Official)=yellow

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in frame_to_rows:
            for row in frame_to_rows[frame_idx]:
                track_id = int(row["track_id"])
                team = team_labels.get(track_id, -1)
                color = TEAM_COLORS.get(team, (128, 128, 128))

                x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                if team == 2:
                    label = f"ID{track_id} Official"
                elif team == -1:
                    label = f"ID{track_id}"
                else:
                    label = f"ID{track_id} T{team}"
                cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()


def run_team_classification(video_path: str, csv_path: str, output_dir: str = "data/output",
                             n_clusters: int = 3, separate_officials: bool = True):
    df = pd.read_csv(csv_path)
    video_name = os.path.splitext(os.path.basename(csv_path))[0].replace("_tracks", "")
    os.makedirs(output_dir, exist_ok=True)

    print("Extracting jersey colors per track (this reads through the video once)...")
    track_colors = extract_track_colors(video_path, df)
    print(f"Collected color samples for {len(track_colors)} tracks.")

    print(f"Clustering tracks into groups (k={n_clusters})...")
    team_labels, avg_colors = cluster_teams(track_colors, n_clusters=n_clusters, separate_officials=separate_officials)

    # attach team label to every row of the dataframe
    df["team"] = df["track_id"].map(team_labels).fillna(-1).astype(int)
    out_csv = os.path.join(output_dir, f"{video_name}_tracks_with_teams.csv")
    df.to_csv(out_csv, index=False)

    out_video = os.path.join(output_dir, f"{video_name}_team_annotated.mp4")
    print("Rendering team-colored annotated video...")
    save_team_annotated_video(video_path, df, team_labels, out_video)

    team_counts = pd.Series(team_labels.values()).value_counts().to_dict()
    if n_clusters == 3 and separate_officials:
        label_map = {0: "Team A", 1: "Team B", 2: "Official"}
        readable_counts = {label_map.get(k, k): v for k, v in team_counts.items()}
        print(f"\nDone.")
        print(f"  Tracks per group: {readable_counts}")
    else:
        print(f"\nDone.")
        print(f"  Tracks per team: {team_counts}")
    print(f"  Tracks-with-teams CSV: {out_csv}")
    print(f"  Team-annotated video:  {out_video}")

    return df, team_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to original video")
    parser.add_argument("--csv", required=True, help="Path to tracks.csv from detect_and_track.py")
    parser.add_argument("--output_dir", default="data/output")
    parser.add_argument("--n_clusters", type=int, default=3,
                         help="3 = two teams + officials cluster (recommended for real match footage). "
                              "2 = two teams only, no official separation.")
    args = parser.parse_args()

    run_team_classification(args.video, args.csv, args.output_dir, n_clusters=args.n_clusters)
