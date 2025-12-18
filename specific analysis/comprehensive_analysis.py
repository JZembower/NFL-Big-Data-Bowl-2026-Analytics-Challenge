import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Disable pandas truncation
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("=" * 80)
print("NFL BIG DATA BOWL - COMPREHENSIVE ANALYSIS")
print("=" * 80)

# Directory of this script: .../NFL-Big-Data-Bowl-2026-Analytics-Challenge
SCRIPT_DIR = Path(__file__).resolve().parent

# Project root = go up one level from analysis/
PROJECT_ROOT = SCRIPT_DIR.parent

# Data directories
upload_dir = PROJECT_ROOT / "data" / "raw"
output_dir = PROJECT_ROOT / "specific analysis" / "data"

print("SCRIPT_DIR:", SCRIPT_DIR)
print("PROJECT_ROOT:", PROJECT_ROOT)
print("upload_dir:", upload_dir)

print("\n1. Loading supplementary data...")
supp_data = pd.read_csv(upload_dir / "supplementary_data.csv")
print(f"   Supplementary data shape: {supp_data.shape}")
print(f"   Columns: {list(supp_data.columns)}")

# Load all input files (tracking data)
print("\n2. Loading input tracking data for weeks 1-18...")
input_dfs = []
weeks_available = []
for week in range(1, 19):
    if week == 16:  # Skip week 16 as mentioned
        continue
    file_path = upload_dir / f"input_2023_w{week:02d}.csv"
    if file_path.exists():
        df = pd.read_csv(file_path)
        df['week'] = week
        input_dfs.append(df)
        weeks_available.append(week)
        print(f"   Week {week:02d}: {df.shape[0]:,} rows")

input_data = pd.concat(input_dfs, ignore_index=True)
print(f"\n   Total input data shape: {input_data.shape}")
print(f"   Weeks loaded: {weeks_available}")

# Get sample of input data structure
print("\n3. Input data structure (first 3 rows):")
print(input_data.head(3))

# Identify unique players and positions
print("\n4. Analyzing player data...")
players_summary = input_data.groupby(['nfl_id', 'player_name', 'player_position']).agg({
    'play_id': 'count',
    's': 'mean',  # average speed
    'a': 'mean',  # average acceleration
}).reset_index()
players_summary.columns = ['nfl_id', 'player_name', 'position', 'total_frames', 'avg_speed', 'avg_acceleration']

qbs = players_summary[players_summary['position'] == 'QB'].sort_values('total_frames', ascending=False)
wrs = players_summary[players_summary['position'] == 'WR'].sort_values('total_frames', ascending=False)
print(f"   Unique QBs: {len(qbs)}")
print(f"   Unique WRs: {len(wrs)}")
print(f"   Top 5 QBs by frames: {qbs.head()['player_name'].tolist()}")
print(f"   Top 5 WRs by frames: {wrs.head()['player_name'].tolist()}")

# Analyze supplementary data for situational context
print("\n5. Analyzing situational data...")
print(f"   Total plays in supplementary: {len(supp_data)}")
print(f"   Pass result distribution:\n{supp_data['pass_result'].value_counts()}")
print(f"   Down distribution:\n{supp_data['down'].value_counts()}")

# Identify high-pressure situations
print("\n6. Identifying high-pressure situations...")

# 4th down plays
fourth_down_plays = supp_data[supp_data['down'] == 4]
print(f"   4th down plays: {len(fourth_down_plays)}")
print(f"   4th down completions: {(fourth_down_plays['pass_result'] == 'C').sum()}")

# Red zone plays (inside 20 yards)
red_zone_plays = supp_data[supp_data['yardline_number'] <= 20]
print(f"   Red zone plays: {len(red_zone_plays)}")
print(f"   Red zone completions: {(red_zone_plays['pass_result'] == 'C').sum()}")

# 2-minute drill (last 2 minutes of 2nd and 4th quarters)
def is_two_minute_drill(row):
    if pd.isna(row['game_clock']) or pd.isna(row['quarter']):
        return False
    try:
        time_parts = str(row['game_clock']).split(':')
        if len(time_parts) == 2:
            minutes, seconds = int(time_parts[0]), int(time_parts[1])
            return (row['quarter'] in [2, 4]) and (minutes < 2 or (minutes == 2 and seconds == 0))
    except:
        return False
    return False

supp_data['is_two_minute_drill'] = supp_data.apply(is_two_minute_drill, axis=1)
two_min_plays = supp_data[supp_data['is_two_minute_drill']]
print(f"   2-minute drill plays: {len(two_min_plays)}")
print(f"   2-minute drill completions: {(two_min_plays['pass_result'] == 'C').sum()}")

# Save initial exploration results
print("\n7. Saving initial exploration results...")
exploration_summary = {
    'total_input_rows': int(input_data.shape[0]),
    'weeks_analyzed': weeks_available,
    'unique_qbs': int(len(qbs)),
    'unique_wrs': int(len(wrs)),
    'total_plays': int(len(supp_data)),
    'fourth_down_plays': int(len(fourth_down_plays)),
    'red_zone_plays': int(len(red_zone_plays)),
    'two_minute_drill_plays': int(len(two_min_plays)),
}

with open(output_dir / 'exploration_summary.json', 'w') as f:
    json.dump(exploration_summary, f, indent=2)

# Save player summaries
qbs.to_csv(output_dir / 'qb_summary.csv', index=False)
wrs.to_csv(output_dir / 'wr_summary.csv', index=False)

print("\n✓ Phase 1 complete: Data exploration finished")
print(f"  Files saved to: {output_dir}")
