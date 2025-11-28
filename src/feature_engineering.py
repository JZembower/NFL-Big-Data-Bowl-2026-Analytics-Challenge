"""
Feature Engineering Module
Creates features for NFL player trajectory prediction
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create features for trajectory prediction"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize feature engineer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.feature_config = self.config['features']
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all configured features"""
        logger.info("Starting feature engineering...")
        
        df = df.copy()
        
        # Sort data for time-series operations
        df = df.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
        
        # High priority features
        if self.feature_config['velocity_components']:
            df = self.create_velocity_features(df)
        
        if self.feature_config['circular_encoding']:
            df = self.create_circular_encoding(df)
        
        if self.feature_config['distance_features']:
            df = self.create_distance_features(df)
        
        if self.feature_config['ball_relative_features']:
            df = self.create_ball_relative_features(df)
        
        if self.feature_config['frame_changes']:
            df = self.create_frame_change_features(df)
        
        if self.feature_config['rolling_windows']:
            df = self.create_rolling_features(df)
        
        # Medium priority features
        if self.feature_config['player_interactions']:
            df = self.create_player_interaction_features(df)
        
        if self.feature_config['team_aggregations']:
            df = self.create_team_aggregation_features(df)
        
        if self.feature_config['play_context']:
            df = self.create_play_context_features(df)
        
        # Advanced features
        if self.feature_config['trajectory_features']:
            df = self.create_trajectory_features(df)
        
        # Player-specific features
        df = self.create_player_features(df)
        
        logger.info(f"Feature engineering complete. Final shape: {df.shape}")
        
        return df
    
    def create_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create velocity component features"""
        logger.info("Creating velocity features...")
        
        # Convert direction to radians
        dir_rad = np.radians(df['dir'])
        
        # Velocity components
        df['velocity_x'] = df['s'] * np.cos(dir_rad)
        df['velocity_y'] = df['s'] * np.sin(dir_rad)
        
        # Velocity magnitude (same as speed, but useful for validation)
        df['velocity_magnitude'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)
        
        return df
    
    def create_circular_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create circular encoding for angles"""
        logger.info("Creating circular encoding features...")
        
        # Direction encoding
        df['dir_sin'] = np.sin(np.radians(df['dir']))
        df['dir_cos'] = np.cos(np.radians(df['dir']))
        
        # Orientation encoding
        df['o_sin'] = np.sin(np.radians(df['o']))
        df['o_cos'] = np.cos(np.radians(df['o']))
        
        # Direction-orientation alignment
        df['dir_o_alignment'] = np.cos(np.radians(df['dir'] - df['o']))
        
        return df
    
    def create_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create distance-based features"""
        logger.info("Creating distance features...")
        
        # Distance to line of scrimmage
        df['dist_to_los'] = np.abs(df['x'] - df['absolute_yardline_number'])
        df['x_from_los'] = df['x'] - df['absolute_yardline_number']
        df['is_behind_los'] = (df['x_from_los'] < 0).astype(int)
        
        # Distance to endzone (assuming normalized to left direction)
        df['dist_to_endzone'] = 110 - df['x']
        df['is_in_red_zone'] = (df['dist_to_endzone'] <= 20).astype(int)
        
        # Distance to sidelines
        df['dist_to_sideline'] = np.minimum(df['y'], 53.3 - df['y'])
        df['dist_to_center'] = np.abs(df['y'] - 26.65)
        
        # Lateral position
        df['lateral_position'] = pd.cut(df['y'], 
                                        bins=[0, 17.77, 35.53, 53.3],
                                        labels=['left', 'center', 'right'])
        
        # Field zones
        df['field_zone'] = pd.cut(df['x'],
                                  bins=[0, 20, 50, 80, 110, 120],
                                  labels=['own_endzone', 'own_territory', 
                                         'midfield', 'opp_territory', 'opp_endzone'])
        
        return df
    
    def create_ball_relative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features relative to ball landing position"""
        logger.info("Creating ball-relative features...")
        
        # Distance to ball landing position
        df['dist_to_ball_land'] = np.sqrt(
            (df['x'] - df['ball_land_x'])**2 + 
            (df['y'] - df['ball_land_y'])**2
        )
        
        # Angle to ball landing position
        df['angle_to_ball'] = np.arctan2(
            df['ball_land_y'] - df['y'],
            df['ball_land_x'] - df['x']
        )
        
        # Ball position differences
        df['ball_x_diff'] = df['ball_land_x'] - df['x']
        df['ball_y_diff'] = df['ball_land_y'] - df['y']
        
        # Moving toward ball
        angle_diff = df['angle_to_ball'] - np.radians(df['dir'])
        df['moving_toward_ball'] = (np.cos(angle_diff) > 0).astype(int)
        
        # Facing ball
        angle_diff_o = df['angle_to_ball'] - np.radians(df['o'])
        df['facing_ball'] = (np.cos(angle_diff_o) > 0).astype(int)
        
        # Ball travel distance
        df['ball_travel_distance'] = df['ball_land_x'] - df['absolute_yardline_number']
        
        return df
    
    def create_frame_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create frame-to-frame change features"""
        logger.info("Creating frame change features...")
        
        groupby_cols = ['game_id', 'play_id', 'nfl_id']
        
        # Speed changes
        df['speed_change'] = df.groupby(groupby_cols)['s'].diff()
        df['accel_change'] = df.groupby(groupby_cols)['a'].diff()
        
        # Direction changes
        df['dir_change'] = df.groupby(groupby_cols)['dir'].diff()
        df['o_change'] = df.groupby(groupby_cols)['o'].diff()
        
        # Handle circular nature of angles (wrap around 360)
        df['dir_change'] = df['dir_change'].apply(
            lambda x: x - 360 if x > 180 else (x + 360 if x < -180 else x)
        )
        df['o_change'] = df['o_change'].apply(
            lambda x: x - 360 if x > 180 else (x + 360 if x < -180 else x)
        )
        
        # Position changes
        df['x_change'] = df.groupby(groupby_cols)['x'].diff()
        df['y_change'] = df.groupby(groupby_cols)['y'].diff()
        
        # Distance traveled
        df['distance_traveled'] = np.sqrt(df['x_change']**2 + df['y_change']**2)
        
        # Acceleration indicators
        df['is_accelerating'] = (df['a'] > 0.5).astype(int)
        df['is_decelerating'] = (df['a'] < -0.5).astype(int)
        
        # Cutting/turning indicators
        df['is_cutting'] = (np.abs(df['dir_change']) > 45).astype(int)
        df['is_running_straight'] = (np.abs(df['dir_change']) < 10).astype(int)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features"""
        logger.info("Creating rolling features...")
        
        groupby_cols = ['game_id', 'play_id', 'nfl_id']
        windows = self.feature_config['rolling_windows']
        
        for window in windows:
            # Rolling speed
            df[f'speed_ma{window}'] = df.groupby(groupby_cols)['s'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            
            # Rolling acceleration
            df[f'accel_ma{window}'] = df.groupby(groupby_cols)['a'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            
            # Rolling distance to ball
            df[f'dist_to_ball_ma{window}'] = df.groupby(groupby_cols)['dist_to_ball_land'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        
        return df
    
    def create_player_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on player interactions"""
        logger.info("Creating player interaction features...")
        
        radius = self.feature_config['interaction_radius']
        
        # Group by game, play, and frame
        interaction_features = []
        
        for (game_id, play_id, frame_id), frame_df in df.groupby(['game_id', 'play_id', 'frame_id']):
            frame_features = []
            
            for idx, player in frame_df.iterrows():
                # Calculate distances to all other players
                other_players = frame_df[frame_df['nfl_id'] != player['nfl_id']]
                
                distances = np.sqrt(
                    (other_players['x'] - player['x'])**2 + 
                    (other_players['y'] - player['y'])**2
                )
                
                # Distance to nearest opponent
                if player['player_side'] == 'Offense':
                    opponents = other_players[other_players['player_side'] == 'Defense']
                else:
                    opponents = other_players[other_players['player_side'] == 'Offense']
                
                if len(opponents) > 0:
                    opp_distances = np.sqrt(
                        (opponents['x'] - player['x'])**2 + 
                        (opponents['y'] - player['y'])**2
                    )
                    nearest_opp_dist = opp_distances.min()
                    num_opponents_nearby = (opp_distances <= radius).sum()
                else:
                    nearest_opp_dist = np.nan
                    num_opponents_nearby = 0
                
                # Distance to passer
                passer = frame_df[frame_df['player_role'] == 'Passer']
                if len(passer) > 0:
                    dist_to_passer = np.sqrt(
                        (passer['x'].iloc[0] - player['x'])**2 + 
                        (passer['y'].iloc[0] - player['y'])**2
                    )
                else:
                    dist_to_passer = np.nan
                
                # Distance to targeted receiver
                target = frame_df[frame_df['player_role'] == 'Targeted Receiver']
                if len(target) > 0:
                    dist_to_target = np.sqrt(
                        (target['x'].iloc[0] - player['x'])**2 + 
                        (target['y'].iloc[0] - player['y'])**2
                    )
                else:
                    dist_to_target = np.nan
                
                frame_features.append({
                    'game_id': game_id,
                    'play_id': play_id,
                    'frame_id': frame_id,
                    'nfl_id': player['nfl_id'],
                    'dist_to_nearest_opponent': nearest_opp_dist,
                    'num_opponents_within_radius': num_opponents_nearby,
                    'dist_to_passer': dist_to_passer,
                    'dist_to_targeted_receiver': dist_to_target,
                })
            
            interaction_features.extend(frame_features)
        
        # Merge interaction features back
        interaction_df = pd.DataFrame(interaction_features)
        df = df.merge(interaction_df, on=['game_id', 'play_id', 'frame_id', 'nfl_id'], how='left')
        
        return df
    
    def create_team_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create team-level aggregation features"""
        logger.info("Creating team aggregation features...")
        
        # Group by game, play, frame, and side
        team_aggs = df.groupby(['game_id', 'play_id', 'frame_id', 'player_side']).agg({
            's': ['mean', 'std'],
            'x': ['mean', 'std'],
            'y': ['mean', 'std', 'min', 'max']
        }).reset_index()
        
        # Flatten column names
        team_aggs.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                            for col in team_aggs.columns.values]
        
        # Rename for clarity
        team_aggs = team_aggs.rename(columns={
            's_mean': 'team_avg_speed',
            's_std': 'team_speed_std',
            'x_mean': 'team_center_x',
            'x_std': 'team_spread_x',
            'y_mean': 'team_center_y',
            'y_std': 'team_spread_y',
            'y_min': 'team_y_min',
            'y_max': 'team_y_max'
        })
        
        # Formation width
        team_aggs['team_formation_width'] = team_aggs['team_y_max'] - team_aggs['team_y_min']
        
        # Merge back
        df = df.merge(team_aggs, on=['game_id', 'play_id', 'frame_id', 'player_side'], how='left')
        
        return df
    
    def create_play_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create play context features from supplementary data"""
        logger.info("Creating play context features...")
        
        # Down indicators
        if 'down' in df.columns:
            df['is_1st_down'] = (df['down'] == 1).astype(int)
            df['is_2nd_down'] = (df['down'] == 2).astype(int)
            df['is_3rd_down'] = (df['down'] == 3).astype(int)
            df['is_4th_down'] = (df['down'] == 4).astype(int)
        
        # Yardage situation
        if 'yards_to_go' in df.columns:
            df['is_short_yardage'] = (df['yards_to_go'] <= 3).astype(int)
            df['is_long_yardage'] = (df['yards_to_go'] >= 10).astype(int)
        
        # Coverage type
        if 'team_coverage_man_zone' in df.columns:
            df['is_man_coverage'] = (df['team_coverage_man_zone'] == 'Man').astype(int)
            df['is_zone_coverage'] = (df['team_coverage_man_zone'] == 'Zone').astype(int)
        
        # Play action
        if 'play_action' in df.columns:
            df['play_action'] = df['play_action'].fillna(0).astype(int)
        
        return df
    
    def create_trajectory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create trajectory-based features"""
        logger.info("Creating trajectory features...")
        
        groupby_cols = ['game_id', 'play_id', 'nfl_id']
        lookback = self.feature_config['lookback_frames']
        
        # Displacement over lookback window
        df[f'x_displacement_{lookback}'] = df.groupby(groupby_cols)['x'].transform(
            lambda x: x - x.shift(lookback)
        )
        df[f'y_displacement_{lookback}'] = df.groupby(groupby_cols)['y'].transform(
            lambda x: x - x.shift(lookback)
        )
        
        # Total distance traveled
        df['cumulative_distance'] = df.groupby(groupby_cols)['distance_traveled'].transform('cumsum')
        
        # Path efficiency (straight line distance / actual distance)
        df['path_efficiency'] = df.groupby(groupby_cols).apply(
            lambda g: np.sqrt(
                (g['x'] - g['x'].iloc[0])**2 + 
                (g['y'] - g['y'].iloc[0])**2
            ) / (g['cumulative_distance'] + 1e-6)
        ).reset_index(level=[0, 1, 2], drop=True)
        
        return df
    
    def create_player_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create player-specific features"""
        logger.info("Creating player-specific features...")
        
        # Parse height to inches
        if 'player_height' in df.columns:
            df['height_inches'] = df['player_height'].apply(self._parse_height)
        
        # BMI
        if 'player_weight' in df.columns and 'height_inches' in df.columns:
            df['player_bmi'] = (df['player_weight'] / (df['height_inches']**2)) * 703
        
        # Role indicators
        df['is_targeted_receiver'] = (df['player_role'] == 'Targeted Receiver').astype(int)
        df['is_passer'] = (df['player_role'] == 'Passer').astype(int)
        df['is_defensive_coverage'] = (df['player_role'] == 'Defensive Coverage').astype(int)
        df['is_other_route_runner'] = (df['player_role'] == 'Other Route Runner').astype(int)
        
        # Side indicator
        df['is_offense'] = (df['player_side'] == 'Offense').astype(int)
        df['is_defense'] = (df['player_side'] == 'Defense').astype(int)
        
        return df
    
    @staticmethod
    def _parse_height(height_str: str) -> float:
        """Parse height string (e.g., '6-2') to inches"""
        try:
            feet, inches = height_str.split('-')
            return int(feet) * 12 + int(inches)
        except:
            return np.nan
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get list of engineered feature names"""
        # Exclude ID columns and original raw features
        exclude_cols = [
            'game_id', 'play_id', 'nfl_id', 'frame_id', 'week',
            'player_name', 'player_birth_date', 'player_height',
            'player_position', 'player_role', 'player_side',
            'play_direction', 'lateral_position', 'field_zone'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        return feature_cols


def main():
    """Example usage of feature engineering"""
    from data_loader import NFLDataLoader
    
    # Load data
    loader = NFLDataLoader()
    input_df, _ = loader.load_processed_data()
    
    # Create features
    engineer = FeatureEngineer()
    input_df = engineer.create_all_features(input_df)
    
    # Get feature names
    features = engineer.get_feature_names(input_df)
    print(f"\nCreated {len(features)} features:")
    for i, feat in enumerate(features[:20], 1):
        print(f"  {i}. {feat}")
    if len(features) > 20:
        print(f"  ... and {len(features) - 20} more")
    
    # Save features
    output_path = Path("data/processed/input_with_features.parquet")
    input_df.to_parquet(output_path, index=False)
    print(f"\nSaved features to {output_path}")


if __name__ == "__main__":
    main()