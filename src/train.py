"""
Main Training Script
Orchestrates the entire training pipeline
"""

import argparse
import yaml
import logging
from pathlib import Path
import pandas as pd

from data_loader import NFLDataLoader
from feature_engineering import FeatureEngineer
from models.baseline import BaselineModel
from models.lstm import LSTMTrainer
from evaluate import TrajectoryEvaluator
from visualize import TrajectoryVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Complete training pipeline"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize pipeline"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.loader = NFLDataLoader(config_path)
        self.engineer = FeatureEngineer(config_path)
        self.evaluator = TrajectoryEvaluator(config_path)
        self.visualizer = TrajectoryVisualizer(config_path)
        
        # Create directories
        Path(self.config['training']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.config['training']['log_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.config['output']['predictions_dir']).mkdir(parents=True, exist_ok=True)
    
    def prepare_data(self, force_reload: bool = False):
        """Prepare and process data"""
        logger.info("="*80)
        logger.info("STEP 1: DATA PREPARATION")
        logger.info("="*80)
        
        processed_file = Path(self.config['data']['processed_dir']) / "input_processed.parquet"
        
        if processed_file.exists() and not force_reload:
            logger.info("Loading existing processed data...")
            input_df, output_df = self.loader.load_processed_data()
        else:
            logger.info("Processing raw data...")
            
            # Load all weeks
            input_df, output_df = self.loader.load_all_weeks()
            
            # Load and merge supplementary data
            supp_df = self.loader.load_supplementary_data()
            if supp_df is not None:
                input_df = self.loader.merge_with_supplementary(input_df, supp_df)
            
            # Normalize coordinates
            input_df = self.loader.normalize_coordinates(input_df)
            
            # Save processed data
            self.loader.save_processed_data(input_df, output_df)
            
            # Create splits
            splits = self.loader.create_train_val_test_splits(input_df, output_df)
            for split_name, (split_input, split_output) in splits.items():
                self.loader.save_processed_data(split_input, split_output, suffix=f"_{split_name}")
        
        # Get summary
        summary = self.loader.get_data_summary(input_df, output_df)
        logger.info("\nData Summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        
        return input_df, output_df
    
    def create_features(self, force_recreate: bool = False):
        """Create features for all splits"""
        logger.info("\n" + "="*80)
        logger.info("STEP 2: FEATURE ENGINEERING")
        logger.info("="*80)
        
        features_file = Path(self.config['data']['processed_dir']) / "input_with_features_train.parquet"
        
        if features_file.exists() and not force_recreate:
            logger.info("Loading existing features...")
            train_input = pd.read_parquet(features_file)
            val_input = pd.read_parquet(
                Path(self.config['data']['processed_dir']) / "input_with_features_val.parquet"
            )
            test_input = pd.read_parquet(
                Path(self.config['data']['processed_dir']) / "input_with_features_test.parquet"
            )
        else:
            logger.info("Creating features...")
            
            # Load splits
            train_input, train_output = self.loader.load_processed_data(suffix="_train")
            val_input, val_output = self.loader.load_processed_data(suffix="_val")
            test_input, test_output = self.loader.load_processed_data(suffix="_test")
            
            # Create features
            train_input = self.engineer.create_all_features(train_input)
            val_input = self.engineer.create_all_features(val_input)
            test_input = self.engineer.create_all_features(test_input)
            
            # Save features
            train_input.to_parquet(features_file, index=False)
            val_input.to_parquet(
                Path(self.config['data']['processed_dir']) / "input_with_features_val.parquet",
                index=False
            )
            test_input.to_parquet(
                Path(self.config['data']['processed_dir']) / "input_with_features_test.parquet",
                index=False
            )
        
        # Get feature names
        feature_cols = self.engineer.get_feature_names(train_input)
        logger.info(f"\nCreated {len(feature_cols)} features")
        
        return train_input, val_input, test_input, feature_cols
    
    def train_baseline(self, train_input, val_input, feature_cols, model_type="xgboost"):
        """Train baseline model"""
        logger.info("\n" + "="*80)
        logger.info(f"STEP 3: TRAINING BASELINE MODEL ({model_type.upper()})")
        logger.info("="*80)
        
        # Load output data
        _, train_output = self.loader.load_processed_data(suffix="_train")
        _, val_output = self.loader.load_processed_data(suffix="_val")
        
        # Initialize model
        model = BaselineModel(model_type=model_type)
        
        # Prepare data
        X_train, y_x_train, y_y_train = model.prepare_data(train_input, train_output, feature_cols)
        X_val, y_x_val, y_y_val = model.prepare_data(val_input, val_output, feature_cols)
        
        # Train
        model.train(X_train, y_x_train, y_y_train, X_val, y_x_val, y_y_val)
        
        # Evaluate
        train_metrics = model.evaluate(X_train, y_x_train, y_y_train)
        val_metrics = model.evaluate(X_val, y_x_val, y_y_val)
        
        logger.info("\nTraining Metrics:")
        for metric, value in train_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        logger.info("\nValidation Metrics:")
        for metric, value in val_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Feature importance
        importance = model.get_feature_importance(feature_cols)
        if importance is not None:
            logger.info("\nTop 10 Features:")
            for idx, row in importance.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance_avg']:.4f}")
            
            # Save importance
            importance.to_csv(
                Path(self.config['training']['log_dir']) / f'{model_type}_feature_importance.csv',
                index=False
            )
            
            # Visualize
            self.visualizer.plot_feature_importance(
                importance,
                save_path=str(Path(self.config['output']['visualizations_dir']) / 
                            f'{model_type}_feature_importance.png')
            )
        
        # Save model
        model_path = Path(self.config['training']['checkpoint_dir']) / f'baseline_{model_type}.pkl'
        model.save(str(model_path))
        
        return model
    
    def train_lstm(self, train_input, val_input, feature_cols):
        """Train LSTM model"""
        logger.info("\n" + "="*80)
        logger.info("STEP 3: TRAINING LSTM MODEL")
        logger.info("="*80)
        
        # Load output data
        _, train_output = self.loader.load_processed_data(suffix="_train")
        _, val_output = self.loader.load_processed_data(suffix="_val")
        
        # Initialize trainer
        trainer = LSTMTrainer()
        
        # Prepare sequences
        train_sequences = trainer.prepare_sequences(train_input, train_output, feature_cols)
        val_sequences = trainer.prepare_sequences(val_input, val_output, feature_cols)
        
        # Train
        trainer.train(train_sequences, val_sequences)
        
        return trainer
    
    def evaluate_model(self, model, test_input, feature_cols, model_name="baseline"):
        """Evaluate model on test set"""
        logger.info("\n" + "="*80)
        logger.info(f"STEP 4: EVALUATING {model_name.upper()} MODEL")
        logger.info("="*80)
        
        # Load test output
        _, test_output = self.loader.load_processed_data(suffix="_test")
        
        # Make predictions
        if model_name == "baseline":
            X_test, y_x_test, y_y_test = model.prepare_data(test_input, test_output, feature_cols)
            pred_x, pred_y = model.predict(X_test)
            
            # Create predictions dataframe
            # (This is simplified - you'd need to map back to game/play/player IDs)
            predictions = pd.DataFrame({
                'x': pred_x,
                'y': pred_y
            })
        
        # Generate evaluation report
        # report = self.evaluator.generate_report(predictions, test_output, test_input)
        
        # Create visualizations
        # self.visualizer.create_sample_visualizations(test_input, predictions, test_output)
        
        logger.info("Evaluation complete!")
    
    def run_full_pipeline(self, model_type="xgboost", force_reload=False):
        """Run complete training pipeline"""
        logger.info("="*80)
        logger.info("NFL BIG DATA BOWL 2026 - TRAINING PIPELINE")
        logger.info("="*80)
        
        # Step 1: Prepare data
        input_df, output_df = self.prepare_data(force_reload=force_reload)
        
        # Step 2: Create features
        train_input, val_input, test_input, feature_cols = self.create_features(
            force_recreate=force_reload
        )
        
        # Step 3: Train model
        if model_type in ["xgboost", "lightgbm", "ridge"]:
            model = self.train_baseline(train_input, val_input, feature_cols, model_type)
        elif model_type == "lstm":
            model = self.train_lstm(train_input, val_input, feature_cols)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Step 4: Evaluate
        self.evaluate_model(model, test_input, feature_cols, model_name=model_type)
        
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETE!")
        logger.info("="*80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train NFL trajectory prediction model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--model', type=str, default='xgboost',
                       choices=['xgboost', 'lightgbm', 'ridge', 'lstm'],
                       help='Model type to train')
    parser.add_argument('--force-reload', action='store_true',
                       help='Force reload and reprocess data')
    parser.add_argument('--step', type=str, default='all',
                       choices=['data', 'features', 'train', 'evaluate', 'all'],
                       help='Which step to run')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = TrainingPipeline(args.config)
    
    # Run pipeline
    if args.step == 'all':
        pipeline.run_full_pipeline(model_type=args.model, force_reload=args.force_reload)
    elif args.step == 'data':
        pipeline.prepare_data(force_reload=args.force_reload)
    elif args.step == 'features':
        pipeline.create_features(force_recreate=args.force_reload)
    elif args.step == 'train':
        train_input, val_input, test_input, feature_cols = pipeline.create_features()
        if args.model in ["xgboost", "lightgbm", "ridge"]:
            pipeline.train_baseline(train_input, val_input, feature_cols, args.model)
        else:
            pipeline.train_lstm(train_input, val_input, feature_cols)
    elif args.step == 'evaluate':
        # Load model and evaluate
        logger.info("Evaluation step - implement model loading")


if __name__ == "__main__":
    main()