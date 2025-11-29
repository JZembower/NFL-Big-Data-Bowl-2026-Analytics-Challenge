"""
Baseline Models Module
Simple models for trajectory prediction (XGBoost, LightGBM, Linear Regression)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import yaml
import logging
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineModel:
    """Baseline model for trajectory prediction"""
    
    def __init__(self, config_path: str = "configs/config.yaml", model_type: str = "xgboost"):
        """Initialize baseline model"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_type = model_type
        self.model_config = self.config['models']['baseline']
        self.separate_xy = self.model_config['separate_xy_models']
        
        self.model_x = None
        self.model_y = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_data(self, 
                     input_df: pd.DataFrame, 
                     output_df: pd.DataFrame,
                     feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training"""
        logger.info("Preparing data for baseline model...")
        
        # Get last frame for each player in each play
        last_frames = input_df.groupby(['game_id', 'play_id', 'nfl_id'])['frame_id'].max().reset_index()
        last_frames.columns = ['game_id', 'play_id', 'nfl_id', 'last_frame_id']
        
        # Merge to get last frame data
        input_last = input_df.merge(last_frames, on=['game_id', 'play_id', 'nfl_id'])
        input_last = input_last[input_last['frame_id'] == input_last['last_frame_id']]
        
        # Get first frame from output (immediate next position)
        output_first = output_df[output_df['frame_id'] == 1].copy()
        
        # Merge input and output
        merged = input_last.merge(
            output_first[['game_id', 'play_id', 'nfl_id', 'x', 'y']],
            on=['game_id', 'play_id', 'nfl_id'],
            suffixes=('', '_target')
        )
        
        # Calculate displacement (target)
        merged['target_x'] = merged['x_target'] - merged['x']
        merged['target_y'] = merged['y_target'] - merged['y']
        
        # Extract features and targets
        X = merged[feature_cols].values
        y_x = merged['target_x'].values
        y_y = merged['target_y'].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        logger.info(f"Prepared data shape: X={X.shape}, y_x={y_x.shape}, y_y={y_y.shape}")
        
        return X, y_x, y_y
    
    def train(self, X_train: np.ndarray, y_x_train: np.ndarray, y_y_train: np.ndarray,
              X_val: np.ndarray = None, y_x_val: np.ndarray = None, y_y_val: np.ndarray = None):
        """Train baseline model"""
        logger.info(f"Training {self.model_type} baseline model...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        
        # Train X model
        logger.info("Training X displacement model...")
        if self.model_type == "xgboost":
            self.model_x = xgb.XGBRegressor(**self.model_config['params'])
            
            if X_val is not None:
                self.model_x.fit(
                    X_train_scaled, y_x_train,
                    eval_set=[(X_val_scaled, y_x_val)],
                    verbose=False
                )
            else:
                self.model_x.fit(X_train_scaled, y_x_train)
                
        elif self.model_type == "lightgbm":
            self.model_x = lgb.LGBMRegressor(**self.model_config['params'])
            
            if X_val is not None:
                self.model_x.fit(
                    X_train_scaled, y_x_train,
                    eval_set=[(X_val_scaled, y_x_val)],
                    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(False)]
                )
            else:
                self.model_x.fit(X_train_scaled, y_x_train)
                
        elif self.model_type == "ridge":
            self.model_x = Ridge(alpha=1.0)
            self.model_x.fit(X_train_scaled, y_x_train)
        
        # Train Y model
        logger.info("Training Y displacement model...")
        if self.model_type == "xgboost":
            self.model_y = xgb.XGBRegressor(**self.model_config['params'])
            
            if X_val is not None:
                self.model_y.fit(
                    X_train_scaled, y_y_train,
                    eval_set=[(X_val_scaled, y_y_val)],
                    verbose=False
                )
            else:
                self.model_y.fit(X_train_scaled, y_y_train)
                
        elif self.model_type == "lightgbm":
            self.model_y = lgb.LGBMRegressor(**self.model_config['params'])
            
            if X_val is not None:
                self.model_y.fit(
                    X_train_scaled, y_y_train,
                    eval_set=[(X_val_scaled, y_y_val)],
                    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(False)]
                )
            else:
                self.model_y.fit(X_train_scaled, y_y_train)
                
        elif self.model_type == "ridge":
            self.model_y = Ridge(alpha=1.0)
            self.model_y.fit(X_train_scaled, y_y_train)
        
        logger.info("Training complete!")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict X and Y displacements"""
        X_scaled = self.scaler.transform(X)
        
        pred_x = self.model_x.predict(X_scaled)
        pred_y = self.model_y.predict(X_scaled)
        
        return pred_x, pred_y
    
    def evaluate(self, X: np.ndarray, y_x: np.ndarray, y_y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        pred_x, pred_y = self.predict(X)
        
        # Calculate metrics
        rmse_x = np.sqrt(mean_squared_error(y_x, pred_x))
        rmse_y = np.sqrt(mean_squared_error(y_y, pred_y))
        rmse_total = np.sqrt(mean_squared_error(
            np.column_stack([y_x, y_y]),
            np.column_stack([pred_x, pred_y])
        ))
        
        mae_x = mean_absolute_error(y_x, pred_x)
        mae_y = mean_absolute_error(y_y, pred_y)
        
        # Average displacement error
        displacement_errors = np.sqrt((y_x - pred_x)**2 + (y_y - pred_y)**2)
        ade = displacement_errors.mean()
        
        metrics = {
            'rmse_x': rmse_x,
            'rmse_y': rmse_y,
            'rmse_total': rmse_total,
            'mae_x': mae_x,
            'mae_y': mae_y,
            'ade': ade
        }
        
        return metrics
    
    def get_feature_importance(self, feature_names: List[str], top_n: int = 20) -> pd.DataFrame:
        """Get feature importance from tree-based models"""
        if self.model_type not in ["xgboost", "lightgbm"]:
            logger.warning("Feature importance only available for tree-based models")
            return None
        
        # Get importance from X model
        if hasattr(self.model_x, 'feature_importances_'):
            importance_x = self.model_x.feature_importances_
            importance_y = self.model_y.feature_importances_
            
            # Average importance
            importance_avg = (importance_x + importance_y) / 2
            
            # Create dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance_x': importance_x,
                'importance_y': importance_y,
                'importance_avg': importance_avg
            })
            
            importance_df = importance_df.sort_values('importance_avg', ascending=False)
            
            return importance_df.head(top_n)
        
        return None
    
    def save(self, path: str):
        """Save model to disk"""
        save_dict = {
            'model_x': self.model_x,
            'model_y': self.model_y,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk"""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.model_x = save_dict['model_x']
        self.model_y = save_dict['model_y']
        self.scaler = save_dict['scaler']
        self.model_type = save_dict['model_type']
        self.feature_names = save_dict['feature_names']
        
        logger.info(f"Model loaded from {path}")


class MultiFramePredictor:
    def __init__(self, model: BaselineModel):
        self.model = model

    def predict_trajectory(self, input_df: pd.DataFrame, feature_cols, num_frames: int = 10) -> pd.DataFrame:
        """
        Predict multiple future frames for all players in input_df.

        Assumes input_df contains at least:
          - game_id, play_id, nfl_id, frame_id, x, y
          - feature columns used by the model (feature_cols)
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Predicting {num_frames} future frames...")

        # Make a working copy we can append to
        current_df = input_df.copy()

        # We'll accumulate predictions here
        all_pred_rows = []

        # We will start predicting from the frame AFTER the max frame_id present per (game_id, play_id, nfl_id)
        for step in range(1, num_frames + 1):
            logger.info(f"  Step {step}/{num_frames}")

            # 1) Find the last frame per (game_id, play_id, nfl_id) in the current_df
            last_frames = (
                current_df
                .groupby(['game_id', 'play_id', 'nfl_id'])['frame_id']
                .max()
                .reset_index()
                .rename(columns={'frame_id': 'last_frame_id'})
            )

            # 2) Join back to get the full rows of the last frames
            current_last = current_df.merge(
                last_frames,
                on=['game_id', 'play_id', 'nfl_id'],
                how='inner'
            )
            # Keep only rows where frame_id == last_frame_id
            current_last = current_last[current_last['frame_id'] == current_last['last_frame_id']].copy()

            # 3) Ensure all feature columns exist and are numeric
            for col in feature_cols:
                if col not in current_last.columns:
                    current_last[col] = 0.0
                elif not pd.api.types.is_numeric_dtype(current_last[col]):
                    current_last[col] = pd.to_numeric(current_last[col], errors='coerce').fillna(0.0)

            # 4) Prepare feature matrix in the correct order
            X = current_last[feature_cols].astype(float).values
            X = np.nan_to_num(X, nan=0.0)

            # 5) Predict displacement
            dx, dy = self.model.predict(X)

            # 6) Create next frame rows
            next_frame_id = current_last['frame_id'] + 1

            next_rows = pd.DataFrame({
                'game_id': current_last['game_id'].values,
                'play_id': current_last['play_id'].values,
                'nfl_id': current_last['nfl_id'].values,
                'frame_id': next_frame_id.values,
                'x': current_last['x'].values + dx,
                'y': current_last['y'].values + dy,
            })

            # Store for output
            all_pred_rows.append(next_rows)

            # Append into current_df so the next iteration can build on these
            # (This assumes we want strictly-autoregressive predictions.)
            current_df = pd.concat([current_df, next_rows], ignore_index=True)

        # Concatenate all predicted rows
        predictions = pd.concat(all_pred_rows, ignore_index=True)

        return predictions


def main():
    """Example usage of baseline model"""
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from data_loader import NFLDataLoader
    from feature_engineering import FeatureEngineer
    
    # Load data
    loader = NFLDataLoader()
    train_input, train_output = loader.load_processed_data(suffix="_train")
    val_input, val_output = loader.load_processed_data(suffix="_val")
    
    # Create features
    engineer = FeatureEngineer()
    train_input = engineer.create_all_features(train_input)
    val_input = engineer.create_all_features(val_input)
    
    feature_cols = engineer.get_feature_names(train_input)
    
    # Initialize model
    model = BaselineModel(model_type="xgboost")
    
    # Prepare data
    X_train, y_x_train, y_y_train = model.prepare_data(train_input, train_output, feature_cols)
    X_val, y_x_val, y_y_val = model.prepare_data(val_input, val_output, feature_cols)
    
    # Train
    model.train(X_train, y_x_train, y_y_train, X_val, y_x_val, y_y_val)
    
    # Evaluate
    train_metrics = model.evaluate(X_train, y_x_train, y_y_train)
    val_metrics = model.evaluate(X_val, y_x_val, y_y_val)
    
    print("\nTraining Metrics:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nValidation Metrics:")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Feature importance
    importance = model.get_feature_importance(feature_cols)
    if importance is not None:
        print("\nTop 10 Features:")
        print(importance.head(10))
    
    # Save model
    model.save("checkpoints/baseline_xgboost.pkl")


if __name__ == "__main__":
    main()