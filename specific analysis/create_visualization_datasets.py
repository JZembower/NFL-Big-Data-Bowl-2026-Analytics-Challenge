import pandas as pd
import numpy as np
from pathlib import Path
import json

pd.set_option('display.max_columns', None)

print("=" * 80)
print("CREATING VISUALIZATION-READY DATASETS")
print("=" * 80)

# Directory of this script: .../NFL-Big-Data-Bowl-2026-Analytics-Challenge
SCRIPT_DIR = Path(__file__).resolve().parent

# Project root = go up one level from analysis/
PROJECT_ROOT = SCRIPT_DIR.parent

# Data directories
output_dir = PROJECT_ROOT / "specific analysis" / "data"

# Load key analysis files
qb_wr_overall = pd.read_csv(output_dir / 'qb_wr_pairs_overall.csv')
qb_wr_situations = pd.read_csv(output_dir / 'qb_wr_pairs_by_situation.csv')
coverage_insights = pd.read_csv(output_dir / 'coverage_type_success_by_situation.csv')
formation_insights = pd.read_csv(output_dir / 'formation_success_by_down_distance.csv')
pass_length_insights = pd.read_csv(output_dir / 'pass_length_strategy_by_situation.csv')
play_action = pd.read_csv(output_dir / 'play_action_impact_by_situation.csv')

print("\n1. Creating TOP 10 tables for each situation...")

# Top 10 QB-WR pairs by each situation
situations = ['Fourth Down', 'Red Zone', 'Two-Minute Drill', 'Third & Long', 'Goal Line']

for situation in situations:
    sit_data = qb_wr_situations[qb_wr_situations['situation'] == situation].copy()
    sit_data = sit_data.sort_values(['completion_rate', 'avg_epa'], ascending=[False, False]).head(10)
    
    filename = situation.lower().replace(' ', '_').replace('&', 'and') + '_top10.csv'
    sit_data.to_csv(output_dir / filename, index=False)
    print(f"   ✓ {filename}")

print("\n2. Creating coaching decision matrix...")

# Create a decision matrix: Situation x Coverage Type
decision_matrix = coverage_insights.pivot_table(
    index='situation',
    columns='coverage_type',
    values='completion_rate',
    aggfunc='first'
)
decision_matrix.to_csv(output_dir / 'decision_matrix_situation_vs_coverage.csv')
print("   ✓ decision_matrix_situation_vs_coverage.csv")

# EPA version
decision_matrix_epa = coverage_insights.pivot_table(
    index='situation',
    columns='coverage_type',
    values='avg_epa',
    aggfunc='first'
)
decision_matrix_epa.to_csv(output_dir / 'decision_matrix_situation_vs_coverage_epa.csv')
print("   ✓ decision_matrix_situation_vs_coverage_epa.csv")

print("\n3. Creating pass length comparison charts data...")

# Reshape pass length data for easy visualization
pass_length_viz = pass_length_insights.pivot_table(
    index='situation',
    columns='pass_length',
    values=['comp_rate', 'avg_epa'],
    aggfunc='first'
)
pass_length_viz.to_csv(output_dir / 'pass_length_comparison_by_situation.csv')
print("   ✓ pass_length_comparison_by_situation.csv")

print("\n4. Creating play action impact comparison...")

# Play action comparison
play_action_viz = play_action.pivot_table(
    index='situation',
    columns='play_action',
    values=['comp_rate', 'avg_epa'],
    aggfunc='first'
)
play_action_viz.to_csv(output_dir / 'play_action_comparison.csv')
print("   ✓ play_action_comparison.csv")

# Calculate improvement
play_action_impact = []
for situation in play_action['situation'].unique():
    sit_data = play_action[play_action['situation'] == situation]
    
    if len(sit_data) == 2:
        with_pa = sit_data[sit_data['play_action'] == True].iloc[0]
        without_pa = sit_data[sit_data['play_action'] == False].iloc[0]
        
        play_action_impact.append({
            'situation': situation,
            'comp_rate_improvement': with_pa['comp_rate'] - without_pa['comp_rate'],
            'epa_improvement': with_pa['avg_epa'] - without_pa['avg_epa'],
            'with_pa_comp_rate': with_pa['comp_rate'],
            'without_pa_comp_rate': without_pa['comp_rate'],
            'with_pa_epa': with_pa['avg_epa'],
            'without_pa_epa': without_pa['avg_epa']
        })

play_action_impact_df = pd.DataFrame(play_action_impact)
play_action_impact_df.to_csv(output_dir / 'play_action_impact_summary.csv', index=False)
print("   ✓ play_action_impact_summary.csv")

print("\n5. Creating formation effectiveness rankings...")

# Top 3 formations for each down/distance
formation_rankings = []
for dd in formation_insights['down_distance'].unique():
    dd_data = formation_insights[formation_insights['down_distance'] == dd].head(3)
    
    for idx, row in dd_data.iterrows():
        formation_rankings.append({
            'down_distance': dd,
            'rank': list(dd_data.index).index(idx) + 1,
            'formation': row['formation'],
            'plays': row['plays'],
            'comp_rate': row['comp_rate'],
            'avg_epa': row['avg_epa']
        })

formation_rankings_df = pd.DataFrame(formation_rankings)
formation_rankings_df.to_csv(output_dir / 'formation_rankings_by_down_distance.csv', index=False)
print("   ✓ formation_rankings_by_down_distance.csv")

print("\n6. Creating QB-WR pair comparison dataset...")

# Get top 20 overall pairs with their performance in each situation
top_20_pairs = qb_wr_overall.head(20)['qb_wr_pair'].tolist()

pair_comparison = []
for pair in top_20_pairs:
    pair_overall = qb_wr_overall[qb_wr_overall['qb_wr_pair'] == pair].iloc[0]
    
    pair_row = {
        'qb_wr_pair': pair,
        'overall_targets': pair_overall['targets'],
        'overall_comp_rate': pair_overall['completion_rate'],
        'overall_epa': pair_overall['epa_per_target']
    }
    
    # Add situational performance
    for situation in situations:
        sit_data = qb_wr_situations[
            (qb_wr_situations['qb_wr_pair'] == pair) & 
            (qb_wr_situations['situation'] == situation)
        ]
        
        if len(sit_data) > 0:
            sit_row = sit_data.iloc[0]
            pair_row[f'{situation}_targets'] = sit_row['targets']
            pair_row[f'{situation}_comp_rate'] = sit_row['completion_rate']
            pair_row[f'{situation}_epa'] = sit_row['avg_epa']
        else:
            pair_row[f'{situation}_targets'] = 0
            pair_row[f'{situation}_comp_rate'] = None
            pair_row[f'{situation}_epa'] = None
    
    pair_comparison.append(pair_row)

pair_comparison_df = pd.DataFrame(pair_comparison)
pair_comparison_df.to_csv(output_dir / 'top20_pairs_situational_comparison.csv', index=False)
print("   ✓ top20_pairs_situational_comparison.csv")

print("\n7. Creating coverage type heatmap data...")

# Reshape coverage data for heatmap
coverage_heatmap = coverage_insights.pivot_table(
    index='coverage_type',
    columns='situation',
    values='completion_rate',
    aggfunc='first'
)
coverage_heatmap.to_csv(output_dir / 'coverage_heatmap_completion_rate.csv')
print("   ✓ coverage_heatmap_completion_rate.csv")

coverage_heatmap_epa = coverage_insights.pivot_table(
    index='coverage_type',
    columns='situation',
    values='avg_epa',
    aggfunc='first'
)
coverage_heatmap_epa.to_csv(output_dir / 'coverage_heatmap_epa.csv')
print("   ✓ coverage_heatmap_epa.csv")

print("\n8. Creating summary statistics table...")

# Summary stats for quick reference
summary_stats = {
    'metric': [],
    'value': []
}

summary_stats['metric'].append('Total Plays Analyzed')
summary_stats['value'].append(len(qb_wr_overall))

summary_stats['metric'].append('Total QB-WR Pairs')
summary_stats['value'].append(len(qb_wr_overall))

summary_stats['metric'].append('Qualified Pairs (10+ targets)')
summary_stats['value'].append(len(qb_wr_overall[qb_wr_overall['targets'] >= 10]))

summary_stats['metric'].append('Average Completion Rate')
summary_stats['value'].append(f"{qb_wr_overall['completion_rate'].mean():.1f}%")

summary_stats['metric'].append('Average EPA per Target')
summary_stats['value'].append(f"{qb_wr_overall['epa_per_target'].mean():.3f}")

# Add situational counts
for situation in situations:
    sit_data = qb_wr_situations[qb_wr_situations['situation'] == situation]
    summary_stats['metric'].append(f'{situation} Pairs')
    summary_stats['value'].append(len(sit_data))
    
    summary_stats['metric'].append(f'{situation} Avg Completion Rate')
    summary_stats['value'].append(f"{sit_data['completion_rate'].mean():.1f}%")

summary_stats_df = pd.DataFrame(summary_stats)
summary_stats_df.to_csv(output_dir / 'summary_statistics.csv', index=False)
print("   ✓ summary_statistics.csv")

print("\n9. Creating key findings JSON for dashboards...")

key_findings = {
    "headline_stats": {
        "total_qb_wr_pairs": int(len(qb_wr_overall)),
        "highest_completion_rate": f"{qb_wr_overall['completion_rate'].max():.1f}%",
        "top_pair": qb_wr_overall.iloc[0]['qb_wr_pair'],
        "top_pair_targets": int(qb_wr_overall.iloc[0]['targets']),
        "top_pair_comp_rate": f"{qb_wr_overall.iloc[0]['completion_rate']:.1f}%"
    },
    "situational_leaders": {},
    "coverage_vulnerabilities": {},
    "formation_recommendations": {}
}

# Add situational leaders
for situation in situations:
    sit_data = qb_wr_situations[qb_wr_situations['situation'] == situation].sort_values('completion_rate', ascending=False)
    if len(sit_data) > 0:
        leader = sit_data.iloc[0]
        key_findings["situational_leaders"][situation] = {
            "pair": leader['qb_wr_pair'],
            "completion_rate": f"{leader['completion_rate']:.1f}%",
            "targets": int(leader['targets']),
            "avg_epa": float(leader['avg_epa'])
        }

# Add coverage vulnerabilities
for situation in situations:
    sit_coverage = coverage_insights[coverage_insights['situation'] == situation].sort_values('completion_rate', ascending=False)
    if len(sit_coverage) > 0:
        vulnerable = sit_coverage.iloc[0]
        key_findings["coverage_vulnerabilities"][situation] = {
            "most_vulnerable": vulnerable['coverage_type'],
            "completion_rate": f"{vulnerable['completion_rate']:.1f}%",
            "avg_epa": float(vulnerable['avg_epa'])
        }

# Add formation recommendations
down_distances = ['3 & Long (7+)', '4 & Short (1-3)', '2 & Short (1-3)']
for dd in down_distances:
    dd_data = formation_insights[formation_insights['down_distance'] == dd].sort_values('avg_epa', ascending=False)
    if len(dd_data) > 0:
        best = dd_data.iloc[0]
        key_findings["formation_recommendations"][dd] = {
            "formation": best['formation'],
            "comp_rate": f"{best['comp_rate']:.1f}%",
            "avg_epa": float(best['avg_epa'])
        }

with open(output_dir / 'key_findings_dashboard.json', 'w') as f:
    json.dump(key_findings, f, indent=2)
print("   ✓ key_findings_dashboard.json")

print("\n" + "=" * 80)
print("VISUALIZATION DATASETS COMPLETE")
print("=" * 80)
print(f"\nAll files saved to: {output_dir}")
print("\nVisualization-ready files created:")
print("  - Situation-specific top 10 tables (5 files)")
print("  - Decision matrices for coverage and situation")
print("  - Pass length and play action comparisons")
print("  - Formation rankings by down/distance")
print("  - Top 20 pairs situational comparison")
print("  - Coverage heatmap data")
print("  - Summary statistics table")
print("  - Key findings JSON for dashboards")
