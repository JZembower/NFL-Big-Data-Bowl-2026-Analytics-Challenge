"""
Data Loader Module
Handles loading and merging of NFL tracking data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NFLDataLoader:
    """Load and merge NFL Big Data Bowl tracking data"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize data loader with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_dir = Path(self.config['data']['raw_dir'])
        self.processed_dir = Path(self.config['data']['processed_dir'])
        self.splits_dir = Path(self.config['data']['splits_dir'])
        
        # Create directories if they don't exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)
    
    def load_week_data(self, week: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load input and output data for a specific week"""
        input_file = self.raw_dir / f"input_2023_w{week:02d}.csv"
        output_file = self.raw_dir / f"output_2023_w{week:02d}.csv"
        
        if not input_file.exists() or not output_file.exists():
            logger.warning(f"Week {week} data not found")
            return None, None
        
        logger.info(f"Loading week {week} data...")
        input_df = pd.read_csv(input_file)
        output_df = pd.read_csv(output_file)
        
        # Add week identifier
        input_df['week'] = week
        output_df['week'] = week
        
        return input_df, output_df
    
    def load_all_weeks(self, weeks: List[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and concatenate data from multiple weeks"""
        if weeks is None:
            weeks = self.config['processing']['weeks']
        
        input_dfs = []
        output_dfs = []
        
        for week in weeks:
            input_df, output_df = self.load_week_data(week)
            if input_df is not None:
                input_dfs.append(input_df)
                output_dfs.append(output_df)
        
        logger.info(f"Loaded {len(input_dfs)} weeks of data")
        
        input_combined = pd.concat(input_dfs, ignore_index=True)
        output_combined = pd.concat(output_dfs, ignore_index=True)
        
        logger.info(f"Combined input shape: {input_combined.shape}")
        logger.info(f"Combined output shape: {output_combined.shape}")
        
        return input_combined, output_combined
    
    def load_supplementary_data(self) -> pd.DataFrame:
        """Load supplementary play-level data"""
        supp_file = self.raw_dir / self.config['data']['supplementary_file']
        
        if not supp_file.exists():
            logger.warning("Supplementary data not found")
            return None
        
        logger.info("Loading supplementary data...")
        supp_df = pd.read_csv(supp_file)
        logger.info(f"Supplementary data shape: {supp_df.shape}")
        
        return supp_df
    
    def merge_with_supplementary(self, 
                                 input_df: pd.DataFrame, 
                                 supp_df: pd.DataFrame) -> pd.DataFrame:
        """Merge input data with supplementary play information"""
        logger.info("Merging with supplementary data...")
        
        # Select relevant columns from supplementary data
        supp_cols = [
            'game_id', 'play_id', 'down', 'yards_to_go', 'quarter',
            'pass_result', 'offense_formation', 'receiver_alignment',
            'route_of_targeted_receiver', 'play_action', 'dropback_type',
            'dropback_distance', 'pass_location_type', 'defenders_in_the_box',
            'team_coverage_man_zone', 'team_coverage_type', 'pass_length',
            'expected_points', 'expected_points_added', 'yards_gained'
        ]
        
        # Keep only columns that exist
        supp_cols = [col for col in supp_cols if col in supp_df.columns]
        supp_subset = supp_df[supp_cols].copy()
        
        # Merge on game_id and play_id
        merged_df = input_df.merge(
            supp_subset,
            on=['game_id', 'play_id'],
            how='left'
        )
        
        logger.info(f"Merged data shape: {merged_df.shape}")
        logger.info(f"Merge success rate: {(1 - merged_df['down'].isna().mean())*100:.2f}%")
        
        return merged_df
    
    def normalize_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize coordinates based on play direction"""
        if not self.config['processing']['normalize_by_play_direction']:
            return df
        
        logger.info("Normalizing coordinates by play direction...")
        
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Flip x coordinates for plays going right
        right_plays = df['play_direction'] == 'right'
        df.loc[right_plays, 'x'] = 120 - df.loc[right_plays, 'x']
        df.loc[right_plays, 'ball_land_x'] = 120 - df.loc[right_plays, 'ball_land_x']
        
        # Flip direction angles for plays going right
        df.loc[right_plays, 'dir'] = (df.loc[right_plays, 'dir'] + 180) % 360
        df.loc[right_plays, 'o'] = (df.loc[right_plays, 'o'] + 180) % 360
        
        # Set all plays to 'left' direction after normalization
        df['play_direction'] = 'left'
        
        logger.info("Coordinate normalization complete")
        
        return df
    
    def create_train_val_test_splits(self, 
                                     input_df: pd.DataFrame,
                                     output_df: pd.DataFrame) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create train/validation/test splits by week"""
        logger.info("Creating train/val/test splits...")
        
        train_weeks = self.config['processing']['train_weeks']
        val_weeks = self.config['processing']['val_weeks']
        test_weeks = self.config['processing']['test_weeks']
        
        splits = {}
        
        # Training set
        train_input = input_df[input_df['week'].isin(train_weeks)].copy()
        train_output = output_df[output_df['week'].isin(train_weeks)].copy()
        splits['train'] = (train_input, train_output)
        logger.info(f"Train: {len(train_input)} input frames, {len(train_output)} output frames")
        
        # Validation set
        val_input = input_df[input_df['week'].isin(val_weeks)].copy()
        val_output = output_df[output_df['week'].isin(val_weeks)].copy()
        splits['val'] = (val_input, val_output)
        logger.info(f"Val: {len(val_input)} input frames, {len(val_output)} output frames")
        
        # Test set
        test_input = input_df[input_df['week'].isin(test_weeks)].copy()
        test_output = output_df[output_df['week'].isin(test_weeks)].copy()
        splits['test'] = (test_input, test_output)
        logger.info(f"Test: {len(test_input)} input frames, {len(test_output)} output frames")
        
        return splits
    
    def save_processed_data(self, 
                           input_df: pd.DataFrame, 
                           output_df: pd.DataFrame,
                           suffix: str = ""):
        """Save processed data to disk"""
        logger.info(f"Saving processed data{' (' + suffix + ')' if suffix else ''}...")
        
        input_file = self.processed_dir / f"input_processed{suffix}.parquet"
        output_file = self.processed_dir / f"output_processed{suffix}.parquet"
        
        input_df.to_parquet(input_file, index=False)
        output_df.to_parquet(output_file, index=False)
        
        logger.info(f"Saved to {input_file} and {output_file}")
    
    def load_processed_data(self, suffix: str = "") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load processed data from disk"""
        logger.info(f"Loading processed data{' (' + suffix + ')' if suffix else ''}...")
        
        input_file = self.processed_dir / f"input_processed{suffix}.parquet"
        output_file = self.processed_dir / f"output_processed{suffix}.parquet"
        
        if not input_file.exists() or not output_file.exists():
            raise FileNotFoundError("Processed data not found. Run data processing first.")
        
        input_df = pd.read_parquet(input_file)
        output_df = pd.read_parquet(output_file)
        
        logger.info(f"Loaded input shape: {input_df.shape}")
        logger.info(f"Loaded output shape: {output_df.shape}")
        
        return input_df, output_df
    
    def get_data_summary(self, input_df: pd.DataFrame, output_df: pd.DataFrame) -> Dict:
        """Get summary statistics of the data"""
        summary = {
            'input_shape': input_df.shape,
            'output_shape': output_df.shape,
            'num_games': input_df['game_id'].nunique(),
            'num_plays': input_df.groupby(['game_id', 'play_id']).ngroups,
            'num_players': input_df['nfl_id'].nunique(),
            'num_weeks': input_df['week'].nunique() if 'week' in input_df.columns else None,
            'players_to_predict': input_df['player_to_predict'].sum(),
            'avg_input_frames_per_player': input_df.groupby(['game_id', 'play_id', 'nfl_id']).size().mean(),
            'avg_output_frames_per_player': output_df.groupby(['game_id', 'play_id', 'nfl_id']).size().mean(),
        }
        
        return summary


def main():
    """Example usage of data loader"""
    # Initialize loader
    loader = NFLDataLoader()
    
    # Load all weeks
    input_df, output_df = loader.load_all_weeks()
    
    # Load supplementary data
    supp_df = loader.load_supplementary_data()
    
    # Merge with supplementary
    if supp_df is not None:
        input_df = loader.merge_with_supplementary(input_df, supp_df)
    
    # Normalize coordinates
    input_df = loader.normalize_coordinates(input_df)
    
    # Get summary
    summary = loader.get_data_summary(input_df, output_df)
    print("\nData Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Create splits
    splits = loader.create_train_val_test_splits(input_df, output_df)
    
    # Save processed data
    loader.save_processed_data(input_df, output_df)
    
    # Save splits
    for split_name, (split_input, split_output) in splits.items():
        loader.save_processed_data(split_input, split_output, suffix=f"_{split_name}")
    
    print("\nData loading and processing complete!")


if __name__ == "__main__":
    main()