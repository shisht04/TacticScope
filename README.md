# TacticScope v2.0 — Match Intelligence Platform

> Upload a football clip. Get player tracking, team classification, speed analysis, and tactical insights — all in a sleek dark UI.

Built with **YOLOv8 + ByteTrack + Streamlit**.

---

## Features

| Module | What it does |
|---|---|
| 🔍 **Detection & Tracking** | YOLOv8n detects players, ByteTrack assigns persistent IDs |
| 👕 **Team Classification** | K-means on jersey colours splits players into Team A / Team B / Officials |
| ⚡ **Speed Estimation** | Per-player speed (px/s), top speed, sprint count |
| 🧠 **Tactical Insights** | Compactness, pressing intensity, territorial control, formation snapshots, key moments |
| 📊 **Dashboard** | 5-view Streamlit app: Match Overview, Player Explorer, Team Battle, Tactical Story, Full Roster |

---

## Quick Start

`ash
# 1. Clone
git clone https://github.com/<your-username>/TacticScope.git
cd TacticScope

# 2. Create venv + install deps
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

# 3. Run
streamlit run app.py
`

Then open http://localhost:8501 and upload a short football clip (10–20 sec, 720p+).

> **Note:** YOLOv8n weights (yolov8n.pt) auto-download on first run (~6 MB). Needs internet access once.

---

## Project Structure

`
tacticscope/
├── app.py                  # Streamlit app (all 5 views)
├── requirements.txt
├── data/
│   ├── videos/             # Put your input clips here (.gitkeep included)
│   └── output/             # Generated files land here (.gitkeep included)
└── src/
    ├── detect_and_track.py  # Step 1: YOLOv8 + ByteTrack
    ├── analytics.py         # Step 2: Trajectories, distance, heatmaps
    ├── team_classifier.py   # Step 3: Jersey colour k-means
    ├── speed_estimator.py   # Step 4: Speed + sprint detection
    └── tactical_insights.py # Step 5: Compactness, pressing, formation
`

---

## Tech Stack

- **Detection**: [Ultralytics YOLOv8n](https://github.com/ultralytics/ultralytics) (COCO person class)
- **Tracking**: ByteTrack (via model.track())
- **Clustering**: scikit-learn KMeans (k=3: Team A, Team B, Officials)
- **Dashboard**: Streamlit ≥ 1.31 with Plotly
- **Speed**: Pixel-space px/s with rolling smoothing (real-world km/h needs homography calibration — noted as stretch goal)

---

## Known Limitations

- **ID switches**: Occlusion or crossing paths can reset a player's track ID — this is a known limitation of detection-based tracking (no ReID).
- **Pixel units**: Speed/distance are in pixels, not metres. Rankings are still valid relatively.
- **Team classifier**: Works best when teams have clearly different jersey colours. k=3 separates officials automatically.
- **CPU speed**: ~2–5 FPS on CPU for yolov8n — keep demo clips ≤ 20 sec.

---

Made with ❤️ by shisht
