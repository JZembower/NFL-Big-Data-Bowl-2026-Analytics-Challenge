import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from data_loader import NFLDataLoader
from feature_engineering import FeatureEngineer
from models.baseline import BaselineModel, MultiFramePredictor

CONFIG_PATH = "configs/config.yaml"
MODEL_PATH = "checkpoints/baseline_xgboost.pkl"


def load_test_with_features():
    loader = NFLDataLoader(CONFIG_PATH)
    engineer = FeatureEngineer(CONFIG_PATH)

    test_input, test_output = loader.load_processed_data(suffix="_test")
    test_input = engineer.create_all_features(test_input)
    feature_cols = engineer.get_feature_names(test_input)

    return test_input, test_output, feature_cols


def pick_example_play(test_input, test_output):
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


def plot_multiframe(play_input, play_output, pred_trajectory, title_prefix=""):
    """
    Plot a small grid of frames comparing actual vs predicted.
    We explicitly align on frame_ids that exist in BOTH actual and predicted.
    """
    # Frame IDs that exist in both actual and predicted
    actual_frames = set(play_output['frame_id'].unique())
    pred_frames   = set(pred_trajectory['frame_id'].unique())
    common_frames = sorted(list(actual_frames & pred_frames))

    if not common_frames:
        print("Warning: No common frame_ids between actual and predicted; cannot plot comparison.")
        return

    max_frames_to_plot = 4
    frames_to_plot = common_frames[:max_frames_to_plot]

    print(f"Plotting frames: {frames_to_plot}")
    print(f"Predicted frame_ids available: {sorted(pred_frames)[:20]} ...")

    fig, axes = plt.subplots(1, len(frames_to_plot), figsize=(5 * len(frames_to_plot), 6), sharey=True)

    if len(frames_to_plot) == 1:
        axes = [axes]

    for ax, f_id in zip(axes, frames_to_plot):
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 53.3)
        ax.set_title(f"{title_prefix}\nFrame {f_id}")
        ax.set_xlabel("X (yards)")
        ax.set_ylabel("Y (yards)")
        ax.grid(True, alpha=0.3)

        # Actual output frame
        actual = play_output[play_output['frame_id'] == f_id]
        if not actual.empty:
            ax.scatter(actual['x'], actual['y'], c='green', label='Actual', alpha=0.7, s=100)

        # Predicted frame
        pred = pred_trajectory[pred_trajectory['frame_id'] == f_id]
        if not pred.empty:
            ax.scatter(pred['x'], pred['y'], c='red', marker='x', s=100, label='Predicted')

        # Last input frame (for reference) – still show it on the first subplot only
        if f_id == frames_to_plot[0]:
            last_input_frame = play_input['frame_id'].max()
            input_last = play_input[play_input['frame_id'] == last_input_frame]
            if not input_last.empty:
                ax.scatter(input_last['x'], input_last['y'], c='blue', alpha=0.5, s=80, label='Last input')

        ax.legend()

    plt.tight_layout()
    plt.show()

def main():
    test_input, test_output, feature_cols = load_test_with_features()
    game_id, play_id = pick_example_play(test_input, test_output)
    print(f"Multi-frame example play: game_id={game_id}, play_id={play_id}")
    play_input, play_output = get_play_data(test_input, test_output, game_id, play_id)

    # Load baseline model
    model = BaselineModel(model_type="xgboost")
    model.load(MODEL_PATH)

    # Use saved feature names if available
    if model.feature_names is not None:
        feature_cols = model.feature_names
        print(f"Using {len(feature_cols)} features from saved model")

    # IMPORTANT: Ensure all required features exist in play_input
    play_input = ensure_all_features(play_input, feature_cols)

    # Use MultiFramePredictor
    predictor = MultiFramePredictor(model)

    # Decide how many frames to predict – e.g. up to number of frames in output
    num_frames = int(play_output['frame_id'].max())
    print(f"Predicting {num_frames} future frames")

    # Restrict input for this play only
    play_pred_traj = predictor.predict_trajectory(play_input, feature_cols, num_frames=num_frames)

    plot_multiframe(
        play_input,
        play_output,
        play_pred_traj,
        title_prefix=f"game_id={game_id}, play_id={play_id}",
    )


if __name__ == "__main__":
    main()