import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from data_loader import NFLDataLoader
from feature_engineering import FeatureEngineer
from models.baseline import BaselineModel

CONFIG_PATH = "configs/config.yaml"
MODEL_PATH = "checkpoints/baseline_xgboost.pkl"


def load_test_with_features():
    loader = NFLDataLoader(CONFIG_PATH)
    engineer = FeatureEngineer(CONFIG_PATH)

    # Load processed test split
    test_input, test_output = loader.load_processed_data(suffix="_test")

    # If you already have feature parquet, you could read that instead
    # For robustness, we just recompute features here
    test_input = engineer.create_all_features(test_input)
    feature_cols = engineer.get_feature_names(test_input)

    return test_input, test_output, feature_cols


def pick_example_play(test_input, test_output):
    """
    Pick a single (game_id, play_id) pair that exists in both input and output.
    Here we just take the first one.
    """
    merged_ids = (
        test_output[['game_id', 'play_id']]
        .drop_duplicates()
        .merge(
            test_input[['game_id', 'play_id']].drop_duplicates(),
            on=['game_id', 'play_id'],
            how='inner'
        )
    )

    game_id, play_id = merged_ids.iloc[0][['game_id', 'play_id']]
    return int(game_id), int(play_id)


def get_play_data(input_df, output_df, game_id, play_id):
    play_input = input_df[(input_df['game_id'] == game_id) & (input_df['play_id'] == play_id)].copy()
    play_output = output_df[(output_df['game_id'] == game_id) & (output_df['play_id'] == play_id)].copy()
    return play_input, play_output


def prepare_first_frame_data(model, play_input, play_output, feature_cols):
    """
    Use BaselineModel.prepare_data logic but restricted to one play.
    This gives us:
      - last input frame per player
      - first output frame per player
      - their displacement targets
    """
    # Get last frame for each player in this play
    last_frames = (
        play_input
        .groupby(['game_id', 'play_id', 'nfl_id'])['frame_id']
        .max()
        .reset_index()
        .rename(columns={'frame_id': 'last_frame_id'})
    )

    input_last = play_input.merge(last_frames, on=['game_id', 'play_id', 'nfl_id'])
    input_last = input_last[input_last['frame_id'] == input_last['last_frame_id']]

    # First frame from output (frame_id == 1)
    output_first = play_output[play_output['frame_id'] == 1].copy()

    merged = input_last.merge(
        output_first[['game_id', 'play_id', 'nfl_id', 'x', 'y']],
        on=['game_id', 'play_id', 'nfl_id'],
        suffixes=('', '_target'),
    )

    # Targets (displacement) – just for inspection
    merged['target_x'] = merged['x_target'] - merged['x']
    merged['target_y'] = merged['y_target'] - merged['y']

    # Ensure all expected feature columns exist and are numeric
    all_features = list(feature_cols)

    for col in all_features:
        if col not in merged.columns:
            # Create missing columns as zeros
            merged[col] = 0.0
        # Force numeric dtype (if something slipped through as object)
        if not pd.api.types.is_numeric_dtype(merged[col]):
            merged[col] = pd.to_numeric(merged[col], errors='coerce').fillna(0.0)

    # Now build X with EXACTLY the same columns and order used at training
    X = merged[all_features].astype(float).values
    X = np.nan_to_num(X, nan=0.0)

    return merged, X


def plot_play_predictions(merged_df, pred_dx, pred_dy, title_suffix=""):
    """
    Plot:
     - Last input positions (current)
     - True next frame positions
     - Predicted next frame positions
    """
    # Field dimensions (normalized: 0–120 x, 0–53.3 y)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    ax.set_xlabel("X (yards)")
    ax.set_ylabel("Y (yards)")
    ax.set_title(f"Single-Frame Baseline Prediction {title_suffix}")

    # Current positions
    ax.scatter(merged_df['x'], merged_df['y'], c='blue', label='Last input frame', alpha=0.7, s=100)

    # True next
    true_x = merged_df['x_target']
    true_y = merged_df['y_target']
    ax.scatter(true_x, true_y, c='green', label='True next frame', alpha=0.7, s=100)

    # Predicted next
    pred_x = merged_df['x'] + pred_dx
    pred_y = merged_df['y'] + pred_dy
    ax.scatter(pred_x, pred_y, c='red', marker='x', s=100, label='Predicted next frame')

    # Draw arrows from current to true for a few players
    sample_size = min(10, len(merged_df))
    for _, row in merged_df.sample(sample_size, random_state=0).iterrows():
        # True arrow (green)
        ax.arrow(
            row['x'], row['y'],
            row['x_target'] - row['x'],
            row['y_target'] - row['y'],
            color='green', alpha=0.4, length_includes_head=True, head_width=0.5,
        )

    # Draw arrows from current to predicted for a few players
    for idx in merged_df.sample(sample_size, random_state=1).index:
        i = merged_df.index.get_loc(idx)
        row = merged_df.loc[idx]
        ax.arrow(
            row['x'], row['y'],
            pred_dx.iloc[i],
            pred_dy.iloc[i],
            color='red', alpha=0.4, length_includes_head=True, head_width=0.5,
        )

    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    # Load test data + features
    test_input, test_output, feature_cols = load_test_with_features()

    # Choose an example play
    game_id, play_id = pick_example_play(test_input, test_output)
    print(f"Plotting example play: game_id={game_id}, play_id={play_id}")

    play_input, play_output = get_play_data(test_input, test_output, game_id, play_id)

# Load model
    model = BaselineModel(model_type="xgboost")
    model.load(MODEL_PATH)

    # Use the same numeric features the model was trained on
    if model.feature_names is not None:
        used_feature_cols = model.feature_names
        print(f"Using {len(used_feature_cols)} features from saved model")
    else:
        used_feature_cols = feature_cols
        print(f"Warning: Model has no saved feature_names, using all {len(feature_cols)} features")

    merged_df, X = prepare_first_frame_data(model, play_input, play_output, used_feature_cols)

    # Predict displacement for that play
    pred_dx, pred_dy = model.predict(X)

    # Convert to pandas Series for easy alignment
    pred_dx = pd.Series(pred_dx, index=merged_df.index)
    pred_dy = pd.Series(pred_dy, index=merged_df.index)

    plot_play_predictions(
        merged_df,
        pred_dx,
        pred_dy,
        title_suffix=f"(game_id={game_id}, play_id={play_id})",
    )


if __name__ == "__main__":
    main()