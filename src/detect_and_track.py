"""
TacticScope - Step 1: Player Detection + Tracking

WHAT THIS DOES:
1. Loads a pretrained YOLOv8 model (trained on COCO, which includes "person" class)
2. Runs it frame-by-frame on a video with tracking enabled
3. YOLOv8's built-in .track() uses ByteTrack under the hood - it assigns each
   detected person a persistent ID that stays the same across frames (as long
   as tracking doesn't lose them)
4. Draws boxes + IDs on each frame, saves an annotated output video
5. Logs every detection (frame number, track ID, bounding box center) to a CSV
   -- this CSV is the raw data our analytics layer (Step 2) will consume

WHY model.track() INSTEAD OF model.predict():
- predict() just detects objects independently in each frame - no memory
  between frames, so the same player would get treated as a "new" object
  every frame.
- track() adds an ID association step (this is where ByteTrack comes in):
  it predicts where each existing tracked object should be in the next frame
  (motion prediction) and matches new detections to existing tracks based on
  position/overlap (IoU). This is what gives us STABLE per-player IDs over
  time, which we need for trajectories/distance/heatmaps later.

RUN:
    python src/detect_and_track.py --video data/videos/sample.mp4
"""

import argparse
import csv
import os

import cv2
from ultralytics import YOLO


def run_detection_and_tracking(video_path: str, output_dir: str = "data/output"):
    os.makedirs(output_dir, exist_ok=True)

    # yolov8n = "nano" - smallest/fastest variant. Good choice for CPU.
    # Ultralytics auto-downloads the pretrained weights on first run.
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    # ── Extract thumbnail frames before detection ───────────────────────────────
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    thumb_dir  = os.path.join(output_dir, "thumbs")
    os.makedirs(thumb_dir, exist_ok=True)
    thumb_paths = []
    margin = max(1, int(total * 0.05))
    picks  = [int(total * p) for p in [0.08, 0.28, 0.50, 0.72, 0.92]]
    picks  = [max(margin, min(p, total - margin)) for p in picks]
    for i, fnum in enumerate(picks):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
        ok, frame = cap.read()
        if ok:
            path = os.path.join(thumb_dir, f"{video_name}_thumb_{i}.jpg")
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
            thumb_paths.append(path)
    cap.release()

    output_video_path = os.path.join(output_dir, f"{video_name}_annotated.mp4")
    output_csv_path   = os.path.join(output_dir, f"{video_name}_tracks.csv")

    # ── Use avc1 (H.264) — required for browser playback via st.video() ────────
    # mp4v works on disk but browsers cannot decode it natively.
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        # avc1 unavailable on this system — fall back, Streamlit will byte-serve it
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    rows = []  # will hold: frame_idx, track_id, x1, y1, x2, y2, cx, cy

    # stream=True: memory-efficient frame-by-frame generator
    # classes=[0]: person only (COCO class 0)
    results = model.track(
        source=video_path,
        classes=[0],
        persist=True,
        stream=True,
        verbose=False,
    )

    frame_idx = 0
    for r in results:
        frame = r.orig_img.copy()

        if r.boxes is not None and r.boxes.id is not None:
            boxes     = r.boxes.xyxy.cpu().numpy()
            track_ids = r.boxes.id.cpu().numpy().astype(int)
            confs     = r.boxes.conf.cpu().numpy()

            for box, track_id, conf in zip(boxes, track_ids, confs):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                rows.append({
                    "frame": frame_idx,
                    "track_id": int(track_id),
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2),
                    "cx": float(cx), "cy": float(cy),
                    "conf": float(conf),
                })

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"ID {track_id}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

        writer.write(frame)
        frame_idx += 1

        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx} frames...")

    writer.release()

    with open(output_csv_path, "w", newline="") as f:
        fieldnames = ["frame", "track_id", "x1", "y1", "x2", "y2", "cx", "cy", "conf"]
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    print(f"\nDone. {frame_idx} frames processed.")
    print(f"Annotated video: {output_video_path}")
    print(f"Tracking data:   {output_csv_path}")
    return output_video_path, output_csv_path, thumb_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output_dir", default="data/output")
    args = parser.parse_args()

    run_detection_and_tracking(args.video, args.output_dir)
