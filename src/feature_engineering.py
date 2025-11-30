
"""
Feature engineering functions for NFL Big Data Bowl
Enhanced with game situation, route intelligence, and velocity/momentum features
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import LabelEncoder


# ============================================================================
# KINEMATIC FEATURES
# ============================================================================

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


# ============================================================================
# RELATIVE POSITION FEATURES
# ============================================================================

def add_relative_position_features(df):
    """Add distances and angles to key locations and players"""
    df = df.copy()

    # Distance and angle to ball landing position
    df['dist_to_ball_land'] = np.sqrt(
        (df['x'] - df['ball_land_x'])**2 + (df['y'] - df['ball_land_y'])**2
    )
    df['angle_to_ball_land'] = np.arctan2(
        df['ball_land_y'] - df['y'], df['ball_land_x'] - df['x']
    ) * 180 / np.pi

    # Distance to QB and target receiver
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

    # Distance to sidelines
    df['dist_to_left_sideline'] = df['y']
    df['dist_to_right_sideline'] = 53.3 - df['y']
    df['dist_to_nearest_sideline'] = df[['dist_to_left_sideline', 'dist_to_right_sideline']].min(axis=1)

    # Distance to endzones
    df['dist_to_own_endzone'] = df.apply(
        lambda row: row['x'] if row['play_direction'] == 'right' else 120 - row['x'], axis=1
    )
    df['dist_to_opp_endzone'] = 120 - df['dist_to_own_endzone']

    return df


# ============================================================================
# COVERAGE FEATURES
# ============================================================================

def add_coverage_features(df):
    """Add separation metrics for coverage analysis"""
    df = df.copy()

    # Nearest opponent distance
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

    # Receiver separation
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

    # Player density
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


# ============================================================================
# PLAY CONTEXT FEATURES
# ============================================================================

def add_play_context_features(df):
    """Add encoded play-level contextual features"""
    df = df.copy()

    # Encode categorical variables
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

    # Numerical context features
    df['score_differential'] = df['pre_snap_home_score'] - df['pre_snap_visitor_score']
    df['field_position_norm'] = df['absolute_yardline_number'] / 100
    df['yards_to_go_norm'] = df['yards_to_go'] / 10
    df['down_distance_ratio'] = df['yards_to_go'] / (df['down'] + 1)

    # Game clock
    if 'game_clock' in df.columns:
        df['game_clock_seconds'] = df['game_clock'].apply(
            lambda x: int(str(x).split(':')[0]) * 60 + int(str(x).split(':')[1]) 
            if pd.notna(x) and ':' in str(x) else np.nan
        )

    # Play action
    if 'play_action' in df.columns:
        df['play_action_binary'] = df['play_action'].map({'True': 1, 'False': 0})

    return df, label_encoders


# ============================================================================
# GAME SITUATION FEATURES (✨ NEW ✨)
# ============================================================================

def add_game_situation_features(df):
    """
    Add advanced game situation features based on game context

    Features:
    - Time pressure indicators (2-minute drill, end of quarter)
    - Score pressure categories (winning big, close game, losing)
    - Critical down situations (3rd/4th down, long distance)
    - Red zone indicators
    - Field position categories
    """
    df = df.copy()

    # ===== TIME PRESSURE FEATURES =====
    if 'game_clock_seconds' not in df.columns and 'game_clock' in df.columns:
        df['game_clock_seconds'] = df['game_clock'].apply(
            lambda x: int(str(x).split(':')[0]) * 60 + int(str(x).split(':')[1]) 
            if pd.notna(x) and ':' in str(x) else np.nan
        )

    # Two-minute warning
    df['two_minute_drill'] = (
        ((df['quarter'] == 2) | (df['quarter'] == 4)) & 
        (df['game_clock_seconds'] <= 120)
    ).astype(int)

    # End of quarter pressure
    df['end_of_quarter'] = (df['game_clock_seconds'] <= 300).astype(int)

    # Time remaining normalized
    df['time_remaining_norm'] = (
        (df['quarter'] - 1) * 900 + df['game_clock_seconds']
    ) / 3600

    # ===== SCORE PRESSURE FEATURES =====
    if 'score_differential' not in df.columns:
        df['score_differential'] = df['pre_snap_home_score'] - df['pre_snap_visitor_score']

    # Adjust for possession team perspective
    df['score_diff_possession'] = df.apply(
        lambda row: row['score_differential'] 
        if row['possession_team'] == row['home_team_abbr'] 
        else -row['score_differential'],
        axis=1
    )

    # Score pressure categories
    def categorize_score_pressure(diff):
        if diff >= 17: return 'winning_big'
        elif diff >= 7: return 'winning_comfortable'
        elif diff >= 3: return 'winning_close'
        elif diff >= -3: return 'tied_close'
        elif diff >= -7: return 'losing_close'
        elif diff >= -17: return 'losing_comfortable'
        else: return 'losing_big'

    df['score_pressure_category'] = df['score_diff_possession'].apply(categorize_score_pressure)

    # Binary indicators
    df['is_winning'] = (df['score_diff_possession'] > 0).astype(int)
    df['is_close_game'] = (df['score_diff_possession'].abs() <= 7).astype(int)
    df['is_blowout'] = (df['score_diff_possession'].abs() >= 17).astype(int)

    # ===== DOWN & DISTANCE FEATURES =====
    df['third_down_long'] = ((df['down'] == 3) & (df['yards_to_go'] >= 7)).astype(int)
    df['fourth_down'] = (df['down'] == 4).astype(int)
    df['passing_down'] = (
        ((df['down'] == 2) & (df['yards_to_go'] >= 8)) |
        ((df['down'] == 3) & (df['yards_to_go'] >= 5))
    ).astype(int)
    df['short_yardage'] = ((df['down'].isin([3, 4])) & (df['yards_to_go'] <= 2)).astype(int)

    # Down category
    df['down_category'] = df['down'].map({
        1: 'first_down', 2: 'second_down', 3: 'third_down', 4: 'fourth_down'
    })

    # Distance category
    def categorize_distance(yards):
        if yards <= 3: return 'short'
        elif yards <= 7: return 'medium'
        else: return 'long'

    df['distance_category'] = df['yards_to_go'].apply(categorize_distance)

    # ===== FIELD POSITION FEATURES =====
    df['red_zone'] = (df['absolute_yardline_number'] <= 20).astype(int)
    df['green_zone'] = (df['absolute_yardline_number'] <= 10).astype(int)
    df['goal_line'] = (df['absolute_yardline_number'] <= 5).astype(int)
    df['own_territory'] = (df['absolute_yardline_number'] >= 50).astype(int)

    # Field position categories
    def categorize_field_position(yardline):
        if yardline <= 10: return 'goal_line'
        elif yardline <= 20: return 'red_zone'
        elif yardline <= 40: return 'opponent_territory'
        elif yardline <= 60: return 'midfield'
        elif yardline <= 80: return 'own_territory'
        else: return 'backed_up'

    df['field_position_category'] = df['absolute_yardline_number'].apply(categorize_field_position)

    # ===== COMBINED SITUATION FEATURES =====
    df['high_pressure_situation'] = (
        (df['is_close_game'] == 1) & 
        (df['two_minute_drill'] == 1) & 
        (df['down'] >= 3)
    ).astype(int)

    df['desperation_situation'] = (
        (df['score_diff_possession'] < -7) & 
        (df['two_minute_drill'] == 1) & 
        (df['yards_to_go'] >= 10)
    ).astype(int)

    df['conservative_situation'] = (
        (df['score_diff_possession'] > 7) & 
        (df['end_of_quarter'] == 1)
    ).astype(int)

    return df


# ============================================================================
# ROUTE INTELLIGENCE FEATURES (✨ NEW ✨)
# ============================================================================

def add_route_intelligence_features(df):
    """
    Add route-based intelligence features

    Features:
    - Historical route completion rates
    - Dropback depth categories (3-step, 5-step, 7-step)
    - Route depth indicators
    - Route complexity metrics
    """
    df = df.copy()

    # ===== ROUTE COMPLETION RATES =====
    if 'route_of_targeted_receiver' in df.columns and 'pass_result' in df.columns:
        route_completion = df.groupby('route_of_targeted_receiver')['pass_result'].apply(
            lambda x: (x == 'C').sum() / len(x) if len(x) > 0 else 0.5
        ).to_dict()
        df['route_completion_rate'] = df['route_of_targeted_receiver'].map(route_completion).fillna(0.5)

    # ===== DROPBACK DEPTH CATEGORIES =====
    def categorize_dropback(distance):
        if pd.isna(distance): return 'unknown'
        elif distance < 3: return 'quick_release'
        elif distance < 4.5: return 'three_step'
        elif distance < 6: return 'five_step'
        elif distance < 8: return 'seven_step'
        else: return 'deep_drop'

    if 'dropback_distance' in df.columns:
        df['dropback_category'] = df['dropback_distance'].apply(categorize_dropback)
        df['is_quick_release'] = (df['dropback_distance'] < 3).astype(int)
        df['is_three_step'] = ((df['dropback_distance'] >= 3) & (df['dropback_distance'] < 4.5)).astype(int)
        df['is_five_step'] = ((df['dropback_distance'] >= 4.5) & (df['dropback_distance'] < 6)).astype(int)
        df['is_seven_step'] = ((df['dropback_distance'] >= 6) & (df['dropback_distance'] < 8)).astype(int)

    # ===== ROUTE DEPTH & COMPLEXITY =====
    def categorize_route_depth(length):
        if pd.isna(length): return 'unknown'
        elif length < 0: return 'behind_los'
        elif length < 5: return 'short'
        elif length < 10: return 'intermediate'
        elif length < 20: return 'medium'
        else: return 'deep'

    if 'pass_length' in df.columns:
        df['route_depth_category'] = df['pass_length'].apply(categorize_route_depth)
        df['is_screen_pass'] = (df['pass_length'] < 0).astype(int)
        df['is_short_route'] = ((df['pass_length'] >= 0) & (df['pass_length'] < 10)).astype(int)
        df['is_deep_route'] = (df['pass_length'] >= 20).astype(int)

    # ===== ROUTE TYPE COMPLEXITY =====
    route_complexity = {
        'FLAT': 'simple', 'HITCH': 'simple', 'SLANT': 'simple', 'OUT': 'simple',
        'IN': 'intermediate', 'CROSS': 'intermediate', 'ANGLE': 'intermediate',
        'CORNER': 'complex', 'POST': 'complex', 'GO': 'complex', 'WHEEL': 'complex'
    }

    if 'route_of_targeted_receiver' in df.columns:
        df['route_complexity'] = df['route_of_targeted_receiver'].map(route_complexity).fillna('unknown')
        df['is_simple_route'] = (df['route_complexity'] == 'simple').astype(int)
        df['is_complex_route'] = (df['route_complexity'] == 'complex').astype(int)

    # ===== DROPBACK TYPE INDICATORS =====
    if 'dropback_type' in df.columns:
        df['is_traditional_dropback'] = (df['dropback_type'] == 'TRADITIONAL').astype(int)
        df['is_rollout'] = df['dropback_type'].str.contains('ROLLOUT', na=False).astype(int)
        df['is_scramble'] = df['dropback_type'].str.contains('SCRAMBLE', na=False).astype(int)
        df['is_designed_run'] = df['dropback_type'].str.contains('DESIGNED_RUN|QB_DRAW', na=False).astype(int)

    # ===== ROUTE-DROPBACK ALIGNMENT =====
    if 'dropback_distance' in df.columns and 'pass_length' in df.columns:
        df['dropback_route_mismatch'] = 0
        df.loc[(df['is_three_step'] == 1) & (df['pass_length'] > 10), 'dropback_route_mismatch'] = 1
        df.loc[(df['is_seven_step'] == 1) & (df['pass_length'] < 10), 'dropback_route_mismatch'] = 1

    return df


# ============================================================================
# VELOCITY & MOMENTUM FEATURES (✨ NEW ✨)
# ============================================================================

def add_velocity_momentum_features(df):
    """
    Add velocity vectors and momentum-based features

    Features:
    - Velocity components (vx, vy)
    - Acceleration components (ax, ay)
    - Momentum proxies
    - Trajectory curvature
    """
    df = df.copy()

    # ===== VELOCITY VECTORS =====
    df['dir_rad'] = np.deg2rad(df['dir'])
    df['velocity_x'] = df['s'] * np.cos(df['dir_rad'])
    df['velocity_y'] = df['s'] * np.sin(df['dir_rad'])
    df['velocity_magnitude'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)

    # ===== ACCELERATION VECTORS =====
    df['accel_x'] = df['a'] * np.cos(df['dir_rad'])
    df['accel_y'] = df['a'] * np.sin(df['dir_rad'])
    df['accel_magnitude'] = np.sqrt(df['accel_x']**2 + df['accel_y']**2)

    # ===== MOMENTUM PROXIES =====
    if 'player_weight' in df.columns:
        df['weight_norm'] = df['player_weight'] / 250
        df['momentum_x'] = df['weight_norm'] * df['velocity_x']
        df['momentum_y'] = df['weight_norm'] * df['velocity_y']
        df['momentum_magnitude'] = np.sqrt(df['momentum_x']**2 + df['momentum_y']**2)
        df['kinetic_energy'] = 0.5 * df['weight_norm'] * (df['s'] ** 2)

    # ===== TRAJECTORY FEATURES =====
    def calc_trajectory_features(group):
        group = group.sort_values('frame_id')
        group['velocity_change_rate'] = group['s'].diff() / 0.1
        group['direction_change_rate'] = group['dir'].diff() / 0.1
        group['trajectory_curvature'] = group['direction_change_rate'] / (group['s'] + 1e-6)
        group['jerk_x'] = group['accel_x'].diff() / 0.1
        group['jerk_y'] = group['accel_y'].diff() / 0.1
        return group

    df = df.groupby(['game_id', 'play_id', 'nfl_id'], group_keys=False).apply(calc_trajectory_features)

    # ===== MOVEMENT EFFICIENCY =====
    if 'ball_land_x' in df.columns and 'ball_land_y' in df.columns:
        df['dir_to_ball_rad'] = np.arctan2(
            df['ball_land_y'] - df['y'],
            df['ball_land_x'] - df['x']
        )
        df['velocity_towards_ball'] = (
            df['velocity_x'] * np.cos(df['dir_to_ball_rad']) +
            df['velocity_y'] * np.sin(df['dir_to_ball_rad'])
        )
        df['movement_efficiency'] = (df['velocity_towards_ball'] / (df['s'] + 1e-6)).clip(-1, 1)

    # ===== SPEED CATEGORIES =====
    def categorize_speed(speed):
        if speed < 2: return 'stationary'
        elif speed < 5: return 'jogging'
        elif speed < 8: return 'running'
        elif speed < 12: return 'sprinting'
        else: return 'max_speed'

    df['speed_category'] = df['s'].apply(categorize_speed)
    df['is_stationary'] = (df['s'] < 2).astype(int)
    df['is_sprinting'] = (df['s'] >= 8).astype(int)
    df['is_max_speed'] = (df['s'] >= 12).astype(int)

    # ===== ACCELERATION CATEGORIES =====
    df['is_accelerating'] = (df['a'] > 1).astype(int)
    df['is_decelerating'] = (df['a'] < -1).astype(int)
    df['is_constant_speed'] = (df['a'].abs() <= 1).astype(int)

    return df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

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

    print("Adding game situation features...")
    df = add_game_situation_features(df)

    print("Adding route intelligence features...")
    df = add_route_intelligence_features(df)

    print("Adding velocity & momentum features...")
    df = add_velocity_momentum_features(df)

    print("✅ All features created!")

    return df, label_encoders
