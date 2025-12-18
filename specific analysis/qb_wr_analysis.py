import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Disable pandas truncation
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("=" * 80)
print("QB-WR PAIR ANALYSIS FOR HIGH-PRESSURE SITUATIONS")
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
print(supp_data.head())

# Parse play descriptions to extract QB and targeted receiver
print("\n2. Extracting QB and targeted receiver from play descriptions...")

def extract_qb_and_receiver(description):
    """Extract QB name and receiver name from play description"""
    if pd.isna(description):
        return None, None
    
    # Pattern: "QB_NAME pass ... to RECEIVER_NAME"
    import re
    # Look for pattern: name pass ... to name
    pattern = r'([A-Z]\.[A-Za-z\-\']+)\s+(?:pass|sacked).*?to\s+([A-Z]\.[A-Za-z\-\']+)'
    match = re.search(pattern, description)
    if match:
        return match.group(1), match.group(2)
    
    # If no "to" found, might be incomplete/sacked
    pattern_qb = r'([A-Z]\.[A-Za-z\-\']+)\s+(?:pass|sacked)'
    match_qb = re.search(pattern_qb, description)
    if match_qb:
        return match_qb.group(1), None
    
    return None, None

# Apply extraction
supp_data[['qb_name', 'receiver_name']] = supp_data['play_description'].apply(
    lambda x: pd.Series(extract_qb_and_receiver(x))
)

print(f"   Plays with QB identified: {supp_data['qb_name'].notna().sum()}")
print(f"   Plays with receiver identified: {supp_data['receiver_name'].notna().sum()}")

# Define high-pressure situations
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

# Additional situational flags
supp_data['is_third_long'] = (supp_data['down'] == 3) & (supp_data['yards_to_go'] >= 7)
supp_data['is_goal_line'] = supp_data['yardline_number'] <= 5
supp_data['is_pressure'] = (supp_data['is_fourth_down'] | 
                              supp_data['is_red_zone'] | 
                              supp_data['is_two_minute_drill'] |
                              supp_data['is_third_long'])

print("\n3. Situational breakdown:")
print(f"   4th down plays: {supp_data['is_fourth_down'].sum()}")
print(f"   Red zone plays: {supp_data['is_red_zone'].sum()}")
print(f"   2-minute drill plays: {supp_data['is_two_minute_drill'].sum()}")
print(f"   3rd & long plays: {supp_data['is_third_long'].sum()}")
print(f"   Goal line plays: {supp_data['is_goal_line'].sum()}")
print(f"   Any high-pressure play: {supp_data['is_pressure'].sum()}")

# Filter for targeted passes only (where we have both QB and receiver)
targeted_passes = supp_data[
    (supp_data['qb_name'].notna()) & 
    (supp_data['receiver_name'].notna())
].copy()

print(f"\n4. Analyzing {len(targeted_passes)} targeted passes...")

# Create QB-WR pair identifier
targeted_passes['qb_wr_pair'] = (targeted_passes['qb_name'] + ' → ' + targeted_passes['receiver_name'])

# Analyze overall QB-WR performance
print("\n5. Overall QB-WR pair performance...")
overall_pairs = targeted_passes.groupby('qb_wr_pair').agg({
    'play_id': 'count',
    'is_completion': 'sum',
    'yards_gained': 'sum',
    'expected_points_added': 'sum'
}).reset_index()

overall_pairs.columns = ['qb_wr_pair', 'targets', 'completions', 'total_yards', 'total_epa']
overall_pairs['completion_rate'] = (overall_pairs['completions'] / overall_pairs['targets'] * 100).round(1)
overall_pairs['yards_per_target'] = (overall_pairs['total_yards'] / overall_pairs['targets']).round(1)
overall_pairs['epa_per_target'] = (overall_pairs['total_epa'] / overall_pairs['targets']).round(2)

# Filter for pairs with at least 10 targets
qualified_pairs = overall_pairs[overall_pairs['targets'] >= 10].copy()
qualified_pairs = qualified_pairs.sort_values('completion_rate', ascending=False)

print(f"   Total unique QB-WR pairs: {len(overall_pairs)}")
print(f"   Qualified pairs (10+ targets): {len(qualified_pairs)}")
print(f"\n   Top 10 QB-WR pairs by completion rate:")
print(qualified_pairs.head(10)[['qb_wr_pair', 'targets', 'completions', 'completion_rate', 'epa_per_target']])

# Analyze performance by situation
print("\n6. Analyzing QB-WR performance in HIGH-PRESSURE situations...")

situations = {
    'Fourth Down': 'is_fourth_down',
    'Red Zone': 'is_red_zone',
    'Two-Minute Drill': 'is_two_minute_drill',
    'Third & Long': 'is_third_long',
    'Goal Line': 'is_goal_line'
}

all_situational_results = []

for situation_name, situation_flag in situations.items():
    print(f"\n   {situation_name}:")
    
    situation_data = targeted_passes[targeted_passes[situation_flag]].copy()
    
    if len(situation_data) == 0:
        print(f"      No data for {situation_name}")
        continue
    
    situation_pairs = situation_data.groupby('qb_wr_pair').agg({
        'play_id': 'count',
        'is_completion': 'sum',
        'yards_gained': ['sum', 'mean'],
        'expected_points_added': ['sum', 'mean']
    }).reset_index()
    
    situation_pairs.columns = ['qb_wr_pair', 'targets', 'completions', 
                                'total_yards', 'avg_yards', 'total_epa', 'avg_epa']
    situation_pairs['completion_rate'] = (situation_pairs['completions'] / 
                                           situation_pairs['targets'] * 100).round(1)
    
    # Filter for at least 3 targets in this situation
    qualified = situation_pairs[situation_pairs['targets'] >= 3].copy()
    qualified = qualified.sort_values('completion_rate', ascending=False)
    
    print(f"      Total pairs in {situation_name}: {len(situation_pairs)}")
    print(f"      Qualified pairs (3+ targets): {len(qualified)}")
    
    if len(qualified) > 0:
        print(f"\n      Top 10 pairs in {situation_name}:")
        print(qualified.head(10)[['qb_wr_pair', 'targets', 'completions', 
                                    'completion_rate', 'avg_epa']])
        
        # Save to results
        qualified['situation'] = situation_name
        all_situational_results.append(qualified)

# Combine all situational results
if all_situational_results:
    all_situations_df = pd.concat(all_situational_results, ignore_index=True)
    all_situations_df.to_csv(output_dir / 'qb_wr_pairs_by_situation.csv', index=False)
    print(f"\n✓ Saved situational analysis to qb_wr_pairs_by_situation.csv")

# Save overall qualified pairs
qualified_pairs.to_csv(output_dir / 'qb_wr_pairs_overall.csv', index=False)
print(f"✓ Saved overall QB-WR pairs to qb_wr_pairs_overall.csv")

# Create summary of clutch performers
print("\n7. Identifying CLUTCH QB-WR pairs...")
print("   (High performance across multiple pressure situations)")

clutch_summary = []
for _, pair_row in qualified_pairs.head(30).iterrows():
    pair_name = pair_row['qb_wr_pair']
    pair_data = {
        'qb_wr_pair': pair_name,
        'overall_targets': pair_row['targets'],
        'overall_comp_rate': pair_row['completion_rate'],
        'overall_epa': pair_row['epa_per_target']
    }
    
    # Check performance in each situation
    for situation_name, situation_flag in situations.items():
        situation_plays = targeted_passes[
            (targeted_passes['qb_wr_pair'] == pair_name) & 
            (targeted_passes[situation_flag])
        ]
        
        if len(situation_plays) >= 3:
            comp_rate = (situation_plays['is_completion'].sum() / len(situation_plays) * 100)
            avg_epa = situation_plays['expected_points_added'].mean()
            pair_data[f'{situation_name}_targets'] = len(situation_plays)
            pair_data[f'{situation_name}_comp_rate'] = round(comp_rate, 1)
            pair_data[f'{situation_name}_epa'] = round(avg_epa, 2)
        else:
            pair_data[f'{situation_name}_targets'] = 0
            pair_data[f'{situation_name}_comp_rate'] = None
            pair_data[f'{situation_name}_epa'] = None
    
    clutch_summary.append(pair_data)

clutch_df = pd.DataFrame(clutch_summary)
clutch_df.to_csv(output_dir / 'clutch_qb_wr_pairs.csv', index=False)
print(f"\n✓ Saved clutch performers analysis to clutch_qb_wr_pairs.csv")

print("\n" + "=" * 80)
print("PHASE 2 COMPLETE: QB-WR pair analysis finished")
print("=" * 80)
