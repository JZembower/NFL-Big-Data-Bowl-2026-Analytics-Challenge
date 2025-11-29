import pandas as pd
from pathlib import Path

from data_loader import NFLDataLoader
from feature_engineering import FeatureEngineer
from models.baseline import BaselineModel, MultiFramePredictor

CONFIG_PATH = "configs/config.yaml"
MODEL_PATH = "checkpoints/baseline_xgboost.pkl"


def ensure_all_features(df, feature_cols):
    """
    Ensure all feature columns exist in df and are numeric.
    Missing columns are created and filled with 0.0.
    Non-numeric columns are coerced to numeric.
    """
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
        elif not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    return df


def main():
    loader = NFLDataLoader(CONFIG_PATH)
    engineer = FeatureEngineer(CONFIG_PATH)

    # 1. Load the data you want to submit predictions for.
    # For Kaggle, this would be their provided test (e.g. input_*.csv).
    # For now, we'll reuse our internal test split.
    print("Loading test data...")
    test_input, test_output = loader.load_processed_data(suffix="_test")

    # 2. Create features
    print("Creating features...")
    test_input = engineer.create_all_features(test_input)
    feature_cols = engineer.get_feature_names(test_input)

    # 3. Load trained baseline model
    print("Loading model...")
    model = BaselineModel(model_type="xgboost")
    model.load(MODEL_PATH)

    # Use saved feature names
    if model.feature_names is not None:
        feature_cols = model.feature_names
        print(f"Using {len(feature_cols)} features from saved model")

    # IMPORTANT: Ensure all required features exist
    print("Ensuring all features exist...")
    test_input = ensure_all_features(test_input, feature_cols)

    # 4. Multi-frame predictor
    predictor = MultiFramePredictor(model)

    # How many frames to predict?
    # For Kaggle, this must match the required future frames in sample_submission.
    # Here we'll just use the max output frame as a placeholder.
    num_frames = int(test_output['frame_id'].max())
    print(f"Predicting {num_frames} future frames for all plays/players in test split")

    # 5. Predict trajectories for ALL test_input
    print("Generating predictions (this may take a while)...")
    all_predictions = predictor.predict_trajectory(
        input_df=test_input,
        feature_cols=feature_cols,
        num_frames=num_frames,
    )

    # all_predictions columns: game_id, play_id, nfl_id, frame_id, x, y

    # 6. Join with meta (if needed)
    # For Kaggle, you'll need to match their required submission keys.
    # Example: suppose submission requires columns:
    #   game_id, play_id, nfl_id, frame_id, x, y
    submission = all_predictions.copy()

    # Optional: ensure correct dtypes / sorting
    submission = submission.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])

    # 7. Save as CSV
    out_path = Path("predictions") / "baseline_xgboost_submission.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    print(f"\n✓ Saved submission CSV to: {out_path}")
    print(f"  Shape: {submission.shape}")
    print(f"\n  First few rows:")
    print(submission.head(10))
    print(f"\n  Column names: {list(submission.columns)}")


if __name__ == "__main__":
    main()