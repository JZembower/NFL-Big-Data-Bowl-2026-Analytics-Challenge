"""
Visualization Module
Visualize trajectory predictions and model performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class TrajectoryVisualizer:
    """Visualize trajectory predictions"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize visualizer"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.viz_config = self.config['evaluation']
        self.output_dir = Path(self.config['output']['visualizations_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_field(self, ax=None):
        """Draw NFL field"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        # Field dimensions
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 53.3)
        
        # Draw field lines
        for x in range(10, 111, 10):
            ax.axvline(x, color='white', linewidth=0.5, alpha=0.5)
        
        # Endzones
        ax.axvspan(0, 10, alpha=0.2, color='red', label='Endzone')
        ax.axvspan(110, 120, alpha=0.2, color='red')
        
        # Sidelines
        ax.axhline(0, color='white', linewidth=2)
        ax.axhline(53.3, color='white', linewidth=2)
        
        # Set background
        ax.set_facecolor('#2E7D32')
        
        ax.set_xlabel('X Position (yards)', fontsize=12)
        ax.set_ylabel('Y Position (yards)', fontsize=12)
        
        return ax
    
    def plot_single_play(self,
                        game_id: int,
                        play_id: int,
                        input_df: pd.DataFrame,
                        predictions: pd.DataFrame,
                        ground_truth: pd.DataFrame = None,
                        save_path: str = None):
        """Visualize a single play with predictions"""
        logger.info(f"Plotting play {game_id}-{play_id}...")
        
        # Filter data for this play
        play_input = input_df[
            (input_df['game_id'] == game_id) &
            (input_df['play_id'] == play_id)
        ]
        
        play_pred = predictions[
            (predictions['game_id'] == game_id) &
            (predictions['play_id'] == play_id)
        ]
        
        if ground_truth is not None:
            play_truth = ground_truth[
                (ground_truth['game_id'] == game_id) &
                (ground_truth['play_id'] == play_id)
            ]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 8))
        ax = self.plot_field(ax)
        
        # Plot each player
        for nfl_id in play_input['nfl_id'].unique():
            player_input = play_input[play_input['nfl_id'] == nfl_id]
            player_pred = play_pred[play_pred['nfl_id'] == nfl_id]
            
            # Get player info
            player_role = player_input['player_role'].iloc[0]
            player_side = player_input['player_side'].iloc[0]
            
            # Color by side
            color = 'blue' if player_side == 'Offense' else 'red'
            
            # Plot input trajectory
            ax.plot(player_input['x'], player_input['y'], 
                   color=color, linewidth=2, alpha=0.6, label=f'{player_role} (input)')
            
            # Plot starting position
            ax.scatter(player_input['x'].iloc[0], player_input['y'].iloc[0],
                      color=color, s=100, marker='o', edgecolors='black', linewidth=2)
            
            # Plot predicted trajectory
            if len(player_pred) > 0:
                ax.plot(player_pred['x'], player_pred['y'],
                       color=color, linewidth=2, linestyle='--', alpha=0.8,
                       label=f'{player_role} (predicted)')
                
                # Plot ending position
                ax.scatter(player_pred['x'].iloc[-1], player_pred['y'].iloc[-1],
                          color=color, s=100, marker='s', edgecolors='black', linewidth=2)
            
            # Plot ground truth if available
            if ground_truth is not None:
                player_truth = play_truth[play_truth['nfl_id'] == nfl_id]
                if len(player_truth) > 0:
                    ax.plot(player_truth['x'], player_truth['y'],
                           color='green', linewidth=2, linestyle=':', alpha=0.8,
                           label=f'{player_role} (actual)')
        
        # Plot ball landing position
        ball_x = play_input['ball_land_x'].iloc[0]
        ball_y = play_input['ball_land_y'].iloc[0]
        ax.scatter(ball_x, ball_y, color='yellow', s=200, marker='*',
                  edgecolors='black', linewidth=2, label='Ball Landing', zorder=10)
        
        # Add title
        ax.set_title(f'Play {game_id}-{play_id}\nCircle=Start, Square=End, Star=Ball',
                    fontsize=14, fontweight='bold')
        
        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_error_distribution(self,
                               predictions: pd.DataFrame,
                               ground_truth: pd.DataFrame,
                               save_path: str = None):
        """Plot distribution of prediction errors"""
        logger.info("Plotting error distribution...")
        
        # Merge predictions with ground truth
        merged = predictions.merge(
            ground_truth,
            on=['game_id', 'play_id', 'nfl_id', 'frame_id'],
            suffixes=('_pred', '_true')
        )
        
        # Calculate errors
        merged['error_x'] = merged['x_pred'] - merged['x_true']
        merged['error_y'] = merged['y_pred'] - merged['y_true']
        merged['displacement_error'] = np.sqrt(
            merged['error_x']**2 + merged['error_y']**2
        )
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # X error distribution
        axes[0, 0].hist(merged['error_x'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_title('X Position Error Distribution')
        axes[0, 0].set_xlabel('Error (yards)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].text(0.05, 0.95, f'Mean: {merged["error_x"].mean():.3f}\nStd: {merged["error_x"].std():.3f}',
                       transform=axes[0, 0].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Y error distribution
        axes[0, 1].hist(merged['error_y'], bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_title('Y Position Error Distribution')
        axes[0, 1].set_xlabel('Error (yards)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].text(0.05, 0.95, f'Mean: {merged["error_y"].mean():.3f}\nStd: {merged["error_y"].std():.3f}',
                       transform=axes[0, 1].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Displacement error distribution
        axes[1, 0].hist(merged['displacement_error'], bins=50, edgecolor='black', alpha=0.7, color='green')
        axes[1, 0].set_title('Displacement Error Distribution')
        axes[1, 0].set_xlabel('Error (yards)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].text(0.05, 0.95, f'Mean: {merged["displacement_error"].mean():.3f}\nMedian: {merged["displacement_error"].median():.3f}',
                       transform=axes[1, 0].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Error by frame
        frame_errors = merged.groupby('frame_id')['displacement_error'].mean()
        axes[1, 1].plot(frame_errors.index, frame_errors.values, marker='o', linewidth=2)
        axes[1, 1].set_title('Average Error by Frame')
        axes[1, 1].set_xlabel('Frame ID')
        axes[1, 1].set_ylabel('Average Displacement Error (yards)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_error_by_role(self,
                          predictions: pd.DataFrame,
                          ground_truth: pd.DataFrame,
                          input_df: pd.DataFrame,
                          save_path: str = None):
        """Plot errors by player role"""
        logger.info("Plotting errors by role...")
        
        # Get player roles
        player_roles = input_df[['game_id', 'play_id', 'nfl_id', 'player_role']].drop_duplicates()
        
        # Merge predictions with ground truth
        merged = predictions.merge(
            ground_truth,
            on=['game_id', 'play_id', 'nfl_id', 'frame_id'],
            suffixes=('_pred', '_true')
        )
        
        # Add player roles
        merged = merged.merge(player_roles, on=['game_id', 'play_id', 'nfl_id'])
        
        # Calculate errors
        merged['displacement_error'] = np.sqrt(
            (merged['x_pred'] - merged['x_true'])**2 +
            (merged['y_pred'] - merged['y_true'])**2
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Box plot by role
        roles = merged['player_role'].unique()
        data = [merged[merged['player_role'] == role]['displacement_error'].values 
                for role in roles]
        
        bp = ax.boxplot(data, labels=roles, patch_artist=True)
        
        # Color boxes
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title('Displacement Error by Player Role', fontsize=14, fontweight='bold')
        ax.set_ylabel('Displacement Error (yards)', fontsize=12)
        ax.set_xlabel('Player Role', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_feature_importance(self,
                               importance_df: pd.DataFrame,
                               top_n: int = 20,
                               save_path: str = None):
        """Plot feature importance"""
        logger.info("Plotting feature importance...")
        
        # Get top features
        top_features = importance_df.head(top_n)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Horizontal bar plot
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_features['importance_avg'], alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_sample_visualizations(self,
                                    input_df: pd.DataFrame,
                                    predictions: pd.DataFrame,
                                    ground_truth: pd.DataFrame,
                                    n_samples: int = None):
        """Create visualizations for sample plays"""
        if n_samples is None:
            n_samples = self.viz_config['visualization_samples']
        
        logger.info(f"Creating {n_samples} sample visualizations...")
        
        # Get random sample of plays
        plays = input_df[['game_id', 'play_id']].drop_duplicates().sample(n=n_samples, random_state=42)
        
        for idx, (_, row) in enumerate(plays.iterrows(), 1):
            save_path = self.output_dir / f"play_{row['game_id']}_{row['play_id']}.png"
            
            self.plot_single_play(
                row['game_id'],
                row['play_id'],
                input_df,
                predictions,
                ground_truth,
                save_path=str(save_path)
            )
        
        logger.info(f"Created {n_samples} visualizations in {self.output_dir}")


def main():
    """Example usage of visualizer"""
    visualizer = TrajectoryVisualizer()
    
    # This would typically be called after making predictions
    # visualizer.create_sample_visualizations(input_df, predictions, ground_truth)
    
    print("Visualizer initialized. Use with actual predictions.")


if __name__ == "__main__":
    main()