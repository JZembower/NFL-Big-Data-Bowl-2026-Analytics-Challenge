import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', None)

print("=" * 80)
print("SITUATIONAL PATTERNS & NEW INSIGHTS DISCOVERY")
print("=" * 80)


from pathlib import Path
import pandas as pd

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

supp_data['is_completion'] = supp_data['pass_result'] == 'C'
supp_data['is_fourth_down'] = supp_data['down'] == 4
supp_data['is_red_zone'] = supp_data['yardline_number'] <= 20
supp_data['is_two_minute_drill'] = supp_data.apply(is_two_minute_drill, axis=1)
supp_data['is_third_long'] = (supp_data['down'] == 3) & (supp_data['yards_to_go'] >= 7)
supp_data['is_goal_line'] = supp_data['yardline_number'] <= 5

print(f"   Total plays: {len(supp_data)}")

# ========================================================================
# INSIGHT 1: Coverage Type Success Rates in High-Pressure Situations
# ========================================================================
print("\n" + "=" * 80)
print("INSIGHT 1: Coverage Types vs Success in High-Pressure Situations")
print("=" * 80)

situations = {
    'Fourth Down': 'is_fourth_down',
    'Red Zone': 'is_red_zone',
    'Two-Minute Drill': 'is_two_minute_drill',
    'Third & Long': 'is_third_long',
    'Goal Line': 'is_goal_line'
}

coverage_insights = []

for situation_name, situation_flag in situations.items():
    situation_data = supp_data[supp_data[situation_flag]].copy()
    
    if len(situation_data) == 0:
        continue
    
    coverage_analysis = situation_data.groupby('team_coverage_type').agg({
        'play_id': 'count',
        'is_completion': ['sum', 'mean'],
        'yards_gained': 'mean',
        'expected_points_added': 'mean'
    }).reset_index()
    
    coverage_analysis.columns = ['coverage_type', 'plays', 'completions', 
                                   'completion_rate', 'avg_yards', 'avg_epa']
    coverage_analysis['completion_rate'] = (coverage_analysis['completion_rate'] * 100).round(1)
    coverage_analysis['avg_yards'] = coverage_analysis['avg_yards'].round(1)
    coverage_analysis['avg_epa'] = coverage_analysis['avg_epa'].round(3)
    coverage_analysis['situation'] = situation_name
    
    # Filter for coverages with at least 10 plays
    coverage_analysis = coverage_analysis[coverage_analysis['plays'] >= 10]
    coverage_analysis = coverage_analysis.sort_values('completion_rate', ascending=False)
    
    coverage_insights.append(coverage_analysis)
    
    print(f"\n{situation_name}:")
    print(coverage_analysis.head(10))

if coverage_insights:
    all_coverage_insights = pd.concat(coverage_insights, ignore_index=True)
    all_coverage_insights.to_csv(output_dir / 'coverage_type_success_by_situation.csv', index=False)
    print(f"\n✓ Saved coverage type insights")

# ========================================================================
# INSIGHT 2: Formation Success by Down & Distance
# ========================================================================
print("\n" + "=" * 80)
print("INSIGHT 2: Best Offensive Formations by Down & Distance")
print("=" * 80)

# Create down/distance categories
def categorize_distance(yards):
    if yards <= 3:
        return 'Short (1-3)'
    elif yards <= 6:
        return 'Medium (4-6)'
    else:
        return 'Long (7+)'

supp_data['distance_category'] = supp_data['yards_to_go'].apply(categorize_distance)
supp_data['down_distance'] = supp_data['down'].astype(str) + ' & ' + supp_data['distance_category']

formation_analysis = supp_data.groupby(['down_distance', 'offense_formation']).agg({
    'play_id': 'count',
    'is_completion': 'mean',
    'yards_gained': 'mean',
    'expected_points_added': 'mean'
}).reset_index()

formation_analysis.columns = ['down_distance', 'formation', 'plays', 
                                'comp_rate', 'avg_yards', 'avg_epa']
formation_analysis['comp_rate'] = (formation_analysis['comp_rate'] * 100).round(1)
formation_analysis['avg_yards'] = formation_analysis['avg_yards'].round(1)
formation_analysis['avg_epa'] = formation_analysis['avg_epa'].round(3)

# Filter for at least 20 plays
formation_analysis = formation_analysis[formation_analysis['plays'] >= 20]
formation_analysis = formation_analysis.sort_values(['down_distance', 'avg_epa'], ascending=[True, False])

print("\nTop formations by down & distance (sorted by EPA):")
for dd in formation_analysis['down_distance'].unique():
    print(f"\n{dd}:")
    print(formation_analysis[formation_analysis['down_distance'] == dd].head(5))

formation_analysis.to_csv(output_dir / 'formation_success_by_down_distance.csv', index=False)
print(f"\n✓ Saved formation analysis")

# ========================================================================
# INSIGHT 3: Route Effectiveness by Field Position
# ========================================================================
print("\n" + "=" * 80)
print("INSIGHT 3: Route Effectiveness by Field Position")
print("=" * 80)

# Categorize field position
def categorize_field_position(yardline):
    if yardline <= 10:
        return 'Own 10 or less'
    elif yardline <= 25:
        return 'Own 11-25'
    elif yardline <= 50:
        return 'Own 26-50'
    elif yardline <= 75:
        return 'Opp 49-25'
    elif yardline <= 90:
        return 'Opp 24-10'
    else:
        return 'Opp 9 or less'

supp_data['field_position'] = supp_data['yardline_number'].apply(categorize_field_position)

route_analysis = supp_data[supp_data['route_of_targeted_receiver'].notna()].groupby(
    ['field_position', 'route_of_targeted_receiver']
).agg({
    'play_id': 'count',
    'is_completion': 'mean',
    'yards_gained': 'mean',
    'expected_points_added': 'mean'
}).reset_index()

route_analysis.columns = ['field_position', 'route', 'targets', 
                          'comp_rate', 'avg_yards', 'avg_epa']
route_analysis['comp_rate'] = (route_analysis['comp_rate'] * 100).round(1)
route_analysis['avg_yards'] = route_analysis['avg_yards'].round(1)
route_analysis['avg_epa'] = route_analysis['avg_epa'].round(3)

# Filter for at least 15 targets
route_analysis = route_analysis[route_analysis['targets'] >= 15]
route_analysis = route_analysis.sort_values(['field_position', 'avg_epa'], ascending=[True, False])

print("\nMost effective routes by field position:")
for fp in ['Own 10 or less', 'Opp 24-10', 'Opp 9 or less']:
    if fp in route_analysis['field_position'].values:
        print(f"\n{fp}:")
        print(route_analysis[route_analysis['field_position'] == fp].head(5))

route_analysis.to_csv(output_dir / 'route_effectiveness_by_field_position.csv', index=False)
print(f"\n✓ Saved route analysis")

# ========================================================================
# INSIGHT 4: Pass Length Strategy in Critical Situations
# ========================================================================
print("\n" + "=" * 80)
print("INSIGHT 4: Pass Length Strategy in Critical Situations")
print("=" * 80)

# Categorize pass length
def categorize_pass_length(length):
    if pd.isna(length):
        return 'Unknown'
    length = abs(length)
    if length <= 5:
        return 'Short (0-5)'
    elif length <= 15:
        return 'Medium (6-15)'
    else:
        return 'Deep (16+)'

supp_data['pass_length_category'] = supp_data['pass_length'].apply(categorize_pass_length)

pass_length_insights = []

for situation_name, situation_flag in situations.items():
    situation_data = supp_data[supp_data[situation_flag]].copy()
    
    if len(situation_data) == 0:
        continue
    
    length_analysis = situation_data.groupby('pass_length_category').agg({
        'play_id': 'count',
        'is_completion': 'mean',
        'yards_gained': 'mean',
        'expected_points_added': 'mean'
    }).reset_index()
    
    length_analysis.columns = ['pass_length', 'plays', 'comp_rate', 'avg_yards', 'avg_epa']
    length_analysis['comp_rate'] = (length_analysis['comp_rate'] * 100).round(1)
    length_analysis['avg_yards'] = length_analysis['avg_yards'].round(1)
    length_analysis['avg_epa'] = length_analysis['avg_epa'].round(3)
    length_analysis['situation'] = situation_name
    
    pass_length_insights.append(length_analysis)
    
    print(f"\n{situation_name}:")
    print(length_analysis)

if pass_length_insights:
    all_pass_length = pd.concat(pass_length_insights, ignore_index=True)
    all_pass_length.to_csv(output_dir / 'pass_length_strategy_by_situation.csv', index=False)
    print(f"\n✓ Saved pass length insights")

# ========================================================================
# INSIGHT 5: Receiver Alignment Impact on Success
# ========================================================================
print("\n" + "=" * 80)
print("INSIGHT 5: Receiver Alignment Impact in Pressure Situations")
print("=" * 80)

alignment_insights = []

for situation_name, situation_flag in situations.items():
    situation_data = supp_data[
        (supp_data[situation_flag]) & 
        (supp_data['receiver_alignment'].notna())
    ].copy()
    
    if len(situation_data) == 0:
        continue
    
    alignment_analysis = situation_data.groupby('receiver_alignment').agg({
        'play_id': 'count',
        'is_completion': 'mean',
        'yards_gained': 'mean',
        'expected_points_added': 'mean'
    }).reset_index()
    
    alignment_analysis.columns = ['alignment', 'plays', 'comp_rate', 'avg_yards', 'avg_epa']
    alignment_analysis['comp_rate'] = (alignment_analysis['comp_rate'] * 100).round(1)
    alignment_analysis['avg_yards'] = alignment_analysis['avg_yards'].round(1)
    alignment_analysis['avg_epa'] = alignment_analysis['avg_epa'].round(3)
    alignment_analysis['situation'] = situation_name
    alignment_analysis = alignment_analysis[alignment_analysis['plays'] >= 10]
    
    alignment_insights.append(alignment_analysis)
    
    print(f"\n{situation_name}:")
    print(alignment_analysis.sort_values('avg_epa', ascending=False).head(5))

if alignment_insights:
    all_alignment = pd.concat(alignment_insights, ignore_index=True)
    all_alignment.to_csv(output_dir / 'receiver_alignment_by_situation.csv', index=False)
    print(f"\n✓ Saved alignment insights")

# ========================================================================
# INSIGHT 6: Man vs Zone Coverage Success
# ========================================================================
print("\n" + "=" * 80)
print("INSIGHT 6: Man vs Zone Coverage Success Rates")
print("=" * 80)

man_zone_analysis = supp_data[supp_data['team_coverage_man_zone'].notna()].groupby(
    ['team_coverage_man_zone', 'down']
).agg({
    'play_id': 'count',
    'is_completion': 'mean',
    'yards_gained': 'mean',
    'expected_points_added': 'mean'
}).reset_index()

man_zone_analysis.columns = ['coverage', 'down', 'plays', 'comp_rate', 'avg_yards', 'avg_epa']
man_zone_analysis['comp_rate'] = (man_zone_analysis['comp_rate'] * 100).round(1)
man_zone_analysis['avg_yards'] = man_zone_analysis['avg_yards'].round(1)
man_zone_analysis['avg_epa'] = man_zone_analysis['avg_epa'].round(3)

print("\nMan vs Zone by Down:")
print(man_zone_analysis)

man_zone_analysis.to_csv(output_dir / 'man_vs_zone_by_down.csv', index=False)
print(f"\n✓ Saved man vs zone analysis")

# ========================================================================
# INSIGHT 7: Play Action Impact in Different Situations
# ========================================================================
print("\n" + "=" * 80)
print("INSIGHT 7: Play Action Impact")
print("=" * 80)

play_action_insights = []

for situation_name, situation_flag in situations.items():
    situation_data = supp_data[supp_data[situation_flag]].copy()
    
    if len(situation_data) == 0:
        continue
    
    pa_analysis = situation_data.groupby('play_action').agg({
        'play_id': 'count',
        'is_completion': 'mean',
        'yards_gained': 'mean',
        'expected_points_added': 'mean'
    }).reset_index()
    
    pa_analysis.columns = ['play_action', 'plays', 'comp_rate', 'avg_yards', 'avg_epa']
    pa_analysis['comp_rate'] = (pa_analysis['comp_rate'] * 100).round(1)
    pa_analysis['avg_yards'] = pa_analysis['avg_yards'].round(1)
    pa_analysis['avg_epa'] = pa_analysis['avg_epa'].round(3)
    pa_analysis['situation'] = situation_name
    
    play_action_insights.append(pa_analysis)
    
    print(f"\n{situation_name}:")
    print(pa_analysis)

if play_action_insights:
    all_play_action = pd.concat(play_action_insights, ignore_index=True)
    all_play_action.to_csv(output_dir / 'play_action_impact_by_situation.csv', index=False)
    print(f"\n✓ Saved play action insights")

print("\n" + "=" * 80)
print("PHASE 4 COMPLETE: Situational patterns analysis finished")
print("=" * 80)
