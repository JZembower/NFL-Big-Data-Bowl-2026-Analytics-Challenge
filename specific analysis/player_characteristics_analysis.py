import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)

print("=" * 80)
print("PLAYER CHARACTERISTICS & SUCCESS PATTERN ANALYSIS")
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

# Extract QB and receiver info
def extract_qb_and_receiver(description):
    if pd.isna(description):
        return None, None
    import re
    pattern = r'([A-Z]\.[A-Za-z\-\']+)\s+(?:pass|sacked).*?to\s+([A-Z]\.[A-Za-z\-\']+)'
    match = re.search(pattern, description)
    if match:
        return match.group(1), match.group(2)
    pattern_qb = r'([A-Z]\.[A-Za-z\-\']+)\s+(?:pass|sacked)'
    match_qb = re.search(pattern_qb, description)
    if match_qb:
        return match_qb.group(1), None
    return None, None

supp_data[['qb_name', 'receiver_name']] = supp_data['play_description'].apply(
    lambda x: pd.Series(extract_qb_and_receiver(x))
)

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

supp_data['is_fourth_down'] = supp_data['down'] == 4
supp_data['is_red_zone'] = supp_data['yardline_number'] <= 20
supp_data['is_two_minute_drill'] = supp_data.apply(is_two_minute_drill, axis=1)
supp_data['is_completion'] = supp_data['pass_result'] == 'C'
supp_data['is_high_pressure'] = (supp_data['is_fourth_down'] | 
                                   supp_data['is_red_zone'] | 
                                   supp_data['is_two_minute_drill'])

# Sample weeks for detailed tracking analysis (to avoid loading all data)
print("\n2. Loading sample tracking data (weeks 1, 5, 10, 15)...")
sample_weeks = [1, 5, 10, 15]
tracking_data = []

for week in sample_weeks:
    file_path = upload_dir / f"input_2023_w{week:02d}.csv"
    if file_path.exists():
        df = pd.read_csv(file_path)
        df['week'] = week
        tracking_data.append(df)
        print(f"   Week {week:02d}: {len(df):,} rows")

tracking_df = pd.concat(tracking_data, ignore_index=True)
print(f"\n   Total tracking data loaded: {len(tracking_df):,} rows")

# Analyze player characteristics by position
print("\n3. Analyzing player physical characteristics...")

# Get unique players with their attributes
players = tracking_df.groupby(['nfl_id', 'player_name', 'player_position', 
                                 'player_height', 'player_weight']).size().reset_index()
players = players[['nfl_id', 'player_name', 'player_position', 'player_height', 'player_weight']]

# Parse height to inches
def height_to_inches(h):
    if pd.isna(h):
        return None
    try:
        parts = str(h).split('-')
        if len(parts) == 2:
            return int(parts[0]) * 12 + int(parts[1])
    except:
        pass
    return None

players['height_inches'] = players['player_height'].apply(height_to_inches)

# Get WRs and QBs
wrs = players[players['player_position'] == 'WR'].copy()
qbs = players[players['player_position'] == 'QB'].copy()

print(f"   Unique WRs: {len(wrs)}")
print(f"   WR height range: {wrs['height_inches'].min()}-{wrs['height_inches'].max()} inches")
print(f"   WR weight range: {wrs['player_weight'].min()}-{wrs['player_weight'].max()} lbs")
print(f"   Unique QBs: {len(qbs)}")

# Analyze tracking metrics (speed, acceleration) by player
print("\n4. Analyzing player movement characteristics...")

# Get WR tracking stats at snap (frame_id == 1) and during routes
wr_tracking = tracking_df[tracking_df['player_position'] == 'WR'].copy()

# Calculate per-player averages
wr_stats = wr_tracking.groupby(['nfl_id', 'player_name']).agg({
    's': ['mean', 'max'],
    'a': ['mean', 'max'],
    'play_id': 'count'
}).reset_index()

wr_stats.columns = ['nfl_id', 'player_name', 'avg_speed', 'max_speed', 
                     'avg_accel', 'max_accel', 'total_frames']

# Merge with physical attributes
wr_stats = wr_stats.merge(wrs[['nfl_id', 'player_height', 'player_weight', 'height_inches']], 
                           on='nfl_id', how='left')

print(f"\n   WR movement statistics:")
print(f"   Average speed range: {wr_stats['avg_speed'].min():.2f} - {wr_stats['avg_speed'].max():.2f} yards/sec")
print(f"   Average max speed range: {wr_stats['max_speed'].min():.2f} - {wr_stats['max_speed'].max():.2f} yards/sec")

# Calculate separation metrics (distance from nearest defender at catch point)
print("\n5. Analyzing separation and route characteristics...")

# For each play, calculate separation at key frames
def calculate_separation_metrics(play_data):
    """Calculate separation between WR and nearest defender"""
    if len(play_data) == 0:
        return None
    
    # Get the target receiver (player_to_predict == True or targeted WR)
    target_wr = play_data[play_data['player_position'] == 'WR']
    defenders = play_data[play_data['player_side'] == 'Defense']
    
    if len(target_wr) == 0 or len(defenders) == 0:
        return None
    
    # Calculate minimum distance from WR to any defender across all frames
    separations = []
    for frame in target_wr['frame_id'].unique():
        wr_frame = target_wr[target_wr['frame_id'] == frame]
        def_frame = defenders[defenders['frame_id'] == frame]
        
        if len(wr_frame) > 0 and len(def_frame) > 0:
            wr_pos = wr_frame[['x', 'y']].values
            def_pos = def_frame[['x', 'y']].values
            
            # Calculate distances
            for wr_x, wr_y in wr_pos:
                distances = np.sqrt((def_pos[:, 0] - wr_x)**2 + (def_pos[:, 1] - wr_y)**2)
                min_dist = distances.min()
                separations.append(min_dist)
    
    if separations:
        return {
            'avg_separation': np.mean(separations),
            'min_separation': np.min(separations),
            'max_separation': np.max(separations)
        }
    return None

# Sample a subset of plays for separation analysis (computational efficiency)
sample_plays = tracking_df[['game_id', 'play_id']].drop_duplicates().sample(n=1000, random_state=42)
print(f"   Analyzing separation for {len(sample_plays)} sample plays...")

separation_results = []
for idx, (game_id, play_id) in enumerate(sample_plays.values):
    if idx % 200 == 0:
        print(f"   Progress: {idx}/{len(sample_plays)}")
    
    play_data = tracking_df[(tracking_df['game_id'] == game_id) & 
                             (tracking_df['play_id'] == play_id)]
    sep_metrics = calculate_separation_metrics(play_data)
    
    if sep_metrics:
        separation_results.append({
            'game_id': game_id,
            'play_id': play_id,
            **sep_metrics
        })

sep_df = pd.DataFrame(separation_results)
print(f"\n   Calculated separation for {len(sep_df)} plays")
print(f"   Average separation: {sep_df['avg_separation'].mean():.2f} yards")
print(f"   Average min separation: {sep_df['min_separation'].mean():.2f} yards")

# Merge with play outcomes
sep_df = sep_df.merge(
    supp_data[['game_id', 'play_id', 'is_completion', 'is_high_pressure', 
               'pass_result', 'yards_gained', 'expected_points_added']],
    on=['game_id', 'play_id'],
    how='left'
)

# Analyze separation by outcome
print("\n6. Separation analysis by play outcome...")
print(f"   Completions:")
completions = sep_df[sep_df['is_completion'] == True]
print(f"      Count: {len(completions)}")
if len(completions) > 0:
    print(f"      Avg separation: {completions['avg_separation'].mean():.2f} yards")
    print(f"      Min separation: {completions['min_separation'].mean():.2f} yards")

print(f"   Incompletions:")
incompletions = sep_df[sep_df['is_completion'] == False]
print(f"      Count: {len(incompletions)}")
if len(incompletions) > 0:
    print(f"      Avg separation: {incompletions['avg_separation'].mean():.2f} yards")
    print(f"      Min separation: {incompletions['min_separation'].mean():.2f} yards")

# High pressure situation analysis
print(f"\n   High Pressure Situations:")
high_pressure = sep_df[sep_df['is_high_pressure'] == True]
print(f"      Count: {len(high_pressure)}")
if len(high_pressure) > 0:
    print(f"      Avg separation: {high_pressure['avg_separation'].mean():.2f} yards")
    print(f"      Completion rate: {(high_pressure['is_completion'].sum() / len(high_pressure) * 100):.1f}%")

# Save player characteristics
print("\n7. Saving player characteristics data...")
wr_stats.to_csv(output_dir / 'wr_movement_characteristics.csv', index=False)
sep_df.to_csv(output_dir / 'separation_analysis.csv', index=False)

print(f"\n✓ Saved WR movement characteristics")
print(f"✓ Saved separation analysis")

print("\n" + "=" * 80)
print("PHASE 3 COMPLETE: Player characteristics analysis finished")
print("=" * 80)
