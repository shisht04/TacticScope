import sys
sys.path.insert(0, 'src')
from analytics import run_analytics
from team_classifier import run_team_classification
from speed_estimator import run_speed_estimation
from tactical_insights import run_tactical_insights
import plotly.graph_objects as go

print('--- v2.0 pipeline check ---')
summary, tracks = run_analytics('data/output/sample_tracks.csv', 'data/output')
print(f'analytics OK: {len(summary)} tracks')

tracks_t, labels = run_team_classification(
    'data/videos/sample.mp4', 'data/output/sample_tracks.csv', 'data/output')
teams = tracks_t.groupby('team')['track_id'].nunique().to_dict()
print(f'teams OK: {teams}')

spd, spd_sum = run_speed_estimation(tracks_t, 25.0)
thresh = spd_sum['sprint_threshold_px_s'].iloc[0]
print(f'speed OK: sprint thresh = {thresh:.1f}')

ins = run_tactical_insights(tracks_t)
print(f'insights OK: {len(ins["key_moments"])} key moments')

fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
print('plotly OK')

print()
print('ALL v2.0 CHECKS PASSED')
