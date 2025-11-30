
"""
Feature engineering functions for NFL Big Data Bowl
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import LabelEncoder


def add_kinematic_features(df):
    """Add smoothed speed, acceleration, jerk, and turn rate"""
    df = df.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id']).copy()

    df['s_smooth'] = df.groupby(['game_id', 'play_id', 'nfl_id'])['s'].transform(
        lambda x: x.rolling(window=3, min_periods=1, center=True).mean()
    )
    df['a_smooth'] = df.groupby(['game_id', 'play_id', 'nfl_id'])['a'].transform(
        lambda x: x.rolling(window=3, min_periods=1, center=True).mean()
    )
    df['jerk'] = df.groupby(['game_id', 'play_id', 'nfl_id'])['a'].diff()
    df['dir_change'] = df.groupby(['game_id', 'play_id', 'nfl_id'])['dir'].diff()
    df['dir_change'] = df['dir_change'].apply(
        lambda x: x - 360 if x > 180 else (x + 360 if x < -180 else x)
    )
    df['bearing_diff'] = df['o'] - df['dir']
    df['bearing_diff'] = df['bearing_diff'].apply(
        lambda x: x - 360 if x > 180 else (x + 360 if x < -180 else x)
    )

    return df


def add_relative_position_features(df):
    """Add distances and angles to key locations and players"""
    df = df.copy()

    df['dist_to_ball_land'] = np.sqrt(
        (df['x'] - df['ball_land_x'])**2 + (df['y'] - df['ball_land_y'])**2
    )
    df['angle_to_ball_land'] = np.arctan2(
        df['ball_land_y'] - df['y'], df['ball_land_x'] - df['x']
    ) * 180 / np.pi

    def calc_player_distances(group):
        qb = group[group['player_role'] == 'Passer']
        if len(qb) > 0:
            qb_x, qb_y = qb['x'].iloc[0], qb['y'].iloc[0]
            group['dist_to_qb'] = np.sqrt((group['x'] - qb_x)**2 + (group['y'] - qb_y)**2)
        else:
            group['dist_to_qb'] = np.nan

        target = group[group['player_role'] == 'Targeted Receiver']
        if len(target) > 0:
            target_x, target_y = target['x'].iloc[0], target['y'].iloc[0]
            group['dist_to_target'] = np.sqrt((group['x'] - target_x)**2 + (group['y'] - target_y)**2)
        else:
            group['dist_to_target'] = np.nan

        return group

    df = df.groupby(['game_id', 'play_id', 'frame_id'], group_keys=False).apply(calc_player_distances)

    df['dist_to_left_sideline'] = df['y']
    df['dist_to_right_sideline'] = 53.3 - df['y']
    df['dist_to_nearest_sideline'] = df[['dist_to_left_sideline', 'dist_to_right_sideline']].min(axis=1)

    df['dist_to_own_endzone'] = df.apply(
        lambda row: row['x'] if row['play_direction'] == 'right' else 120 - row['x'], axis=1
    )
    df['dist_to_opp_endzone'] = 120 - df['dist_to_own_endzone']

    return df


def add_coverage_features(df):
    """Add separation metrics for coverage analysis"""
    df = df.copy()

    def calc_nearest_players(group):
        offense = group[group['player_side'] == 'Offense'][['x', 'y', 'nfl_id']].values
        defense = group[group['player_side'] == 'Defense'][['x', 'y', 'nfl_id']].values

        nearest_opp_dist = []
        for _, row in group.iterrows():
            if row['player_side'] == 'Offense':
                opponents = defense
            else:
                opponents = offense

            if len(opponents) > 0:
                distances = [euclidean([row['x'], row['y']], opp[:2]) for opp in opponents]
                nearest_opp_dist.append(min(distances))
            else:
                nearest_opp_dist.append(np.nan)

        group['nearest_opponent_dist'] = nearest_opp_dist
        return group

    df = df.groupby(['game_id', 'play_id', 'frame_id'], group_keys=False).apply(calc_nearest_players)

    def calc_receiver_separation(group):
        target = group[group['player_role'] == 'Targeted Receiver']
        if len(target) > 0:
            target_x, target_y = target['x'].iloc[0], target['y'].iloc[0]
            defenders = group[group['player_side'] == 'Defense']
            if len(defenders) > 0:
                defender_dists = np.sqrt((defenders['x'] - target_x)**2 + (defenders['y'] - target_y)**2)
                min_separation = defender_dists.min()
                group['receiver_separation'] = min_separation
            else:
                group['receiver_separation'] = np.nan
        else:
            group['receiver_separation'] = np.nan
        return group

    df = df.groupby(['game_id', 'play_id', 'frame_id'], group_keys=False).apply(calc_receiver_separation)

    def calc_density(group):
        densities = []
        for _, row in group.iterrows():
            others = group[group['nfl_id'] != row['nfl_id']]
            distances = np.sqrt((others['x'] - row['x'])**2 + (others['y'] - row['y'])**2)
            density = (distances <= 5).sum()
            densities.append(density)
        group['player_density_5yd'] = densities
        return group

    df = df.groupby(['game_id', 'play_id', 'frame_id'], group_keys=False).apply(calc_density)

    return df


def add_play_context_features(df):
    """Add encoded play-level contextual features"""
    df = df.copy()

    categorical_cols = [
        'team_coverage_man_zone', 'team_coverage_type', 'dropback_type',
        'route_of_targeted_receiver', 'offense_formation', 'pass_result',
        'player_position', 'player_role', 'player_side'
    ]

    label_encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    df['score_differential'] = df['pre_snap_home_score'] - df['pre_snap_visitor_score']
    df['field_position_norm'] = df['absolute_yardline_number'] / 100
    df['yards_to_go_norm'] = df['yards_to_go'] / 10
    df['down_distance_ratio'] = df['yards_to_go'] / (df['down'] + 1)

    if 'game_clock' in df.columns:
        df['game_clock_seconds'] = df['game_clock'].apply(
            lambda x: int(str(x).split(':')[0]) * 60 + int(str(x).split(':')[1]) 
            if pd.notna(x) and ':' in str(x) else np.nan
        )

    if 'play_action' in df.columns:
        df['play_action_binary'] = df['play_action'].map({'Y': 1, 'N': 0})

    return df, label_encoders


def make_all_features(input_df, supp_df):
    """
    Main pipeline to create all features

    Args:
        input_df: Input tracking data
        supp_df: Supplementary play context data

    Returns:
        DataFrame with all engineered features
    """
    # Merge
    df = input_df.merge(supp_df, on=['game_id', 'play_id'], how='left')

    # Apply all feature engineering
    print("Adding kinematic features...")
    df = add_kinematic_features(df)

    print("Adding relative position features...")
    df = add_relative_position_features(df)

    print("Adding coverage features...")
    df = add_coverage_features(df)

    print("Adding play context features...")
    df, label_encoders = add_play_context_features(df)

    print("All features created!")

    return df, label_encoders
