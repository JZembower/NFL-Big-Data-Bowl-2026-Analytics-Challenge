# NFL Big Data Bowl 2026 - Player Trajectory Prediction

A comprehensive machine learning pipeline for predicting NFL player trajectories after pass release using tracking data from the 2023 season.

## 📋 Project Overview

This project predicts future player positions (x, y coordinates) after a pass is thrown, using pre-snap and pre-throw tracking data. The challenge involves:

* **Input**: Player tracking data before pass release (\~28 frames per player)

* **Output**: Predicted player positions after pass release (\~12 frames per player)

* **Players**: Primarily defensive coverage and targeted receivers

## 🏗️ Project Structure

```
project/
├── data/
│   ├── raw/                    # Original CSV files (input_2023_w*.csv, output_2023_w*.csv)
│   ├── processed/              # Processed features (parquet files)
│   └── splits/                 # Train/val/test splits
├── src/
│   ├── data_loader.py          # Load and merge data
│   ├── feature_engineering.py  # Feature creation (140+ features)
│   ├── models/
│   │   ├── baseline.py         # XGBoost/LightGBM/Ridge models
│   │   ├── lstm.py             # LSTM sequence model
│   │   └── transformer.py      # Transformer model (TODO)
│   ├── train.py                # Main training pipeline
│   ├── evaluate.py             # Evaluation metrics (RMSE, ADE, FDE)
│   └── visualize.py            # Plot predictions on field
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory analysis
│   ├── 02_features.ipynb      # Feature engineering experiments
│   └── 03_modeling.ipynb      # Model experiments
├── configs/
│   └── config.yaml             # Hyperparameters and settings
├── checkpoints/                # Saved models
├── logs/                       # Training logs
├── predictions/                # Model predictions
├── visualizations/             # Generated plots
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1\. Installation

```bash
# Clone repository
git clone <your-repo-url>
cd nfl-big-data-bowl-2026

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2\. Data Setup

Place your data files in the `data/raw/` directory:

```
data/raw/
├── input_2023_w01.csv
├── input_2023_w02.csv
├── ...
├── output_2023_w01.csv
├── output_2023_w02.csv
├── ...
└── supplementary_data.csv
```

### 3\. Run Complete Pipeline

```bash
# Train XGBoost baseline model
python src/train.py --model xgboost

# Train LSTM model
python src/train.py --model lstm

# Force reload data
python src/train.py --model xgboost --force-reload
```

### 4\. Run Individual Steps

```bash
# Step 1: Data preparation
python src/train.py --step data

# Step 2: Feature engineering
python src/train.py --step features

# Step 3: Training only
python src/train.py --step train --model xgboost

# Step 4: Evaluation
python src/train.py --step evaluate
```

## 📊 Features

The pipeline creates **140+ engineered features** across 12 categories:

### High Priority Features

* **Velocity Components**: velocity_x, velocity_y from speed + direction

* **Circular Encoding**: sin/cos encoding of angles (dir, orientation)

* **Distance Features**: distance to ball, LOS, endzone, sideline

* **Ball-Relative**: angle to ball, moving toward ball, facing ball

* **Frame Changes**: speed_change, dir_change, acceleration changes

* **Rolling Averages**: 3-frame and 5-frame moving averages

### Medium Priority Features

* **Player Interactions**: distance to nearest opponent, defenders within radius

* **Team Aggregations**: avg team speed, formation width, spatial spread

* **Play Context**: down, yards_to_go, coverage type, red zone indicator

* **Route Patterns**: displacement, direction changes, path efficiency

### Advanced Features

* **Trajectory Features**: curvature, momentum, cumulative distance

* **Player-Specific**: BMI, age, position encoding, role indicators

## 🎯 Models

### 1\. Baseline Models (XGBoost/LightGBM)

* **Approach**: Predict displacement from last input frame

* **Pros**: Fast training, interpretable, good baseline

* **Cons**: No temporal modeling, single-step prediction

```python
from src.models.baseline import BaselineModel

model = BaselineModel(model_type="xgboost")
model.train(X_train, y_x_train, y_y_train)
predictions = model.predict(X_test)
```

### 2\. LSTM Model

* **Approach**: Sequence-to-sequence with bidirectional LSTM

* **Pros**: Models temporal dependencies, multi-step prediction

* **Cons**: Slower training, requires more data

```python
from src.models.lstm import LSTMTrainer

trainer = LSTMTrainer()
trainer.train(train_sequences, val_sequences)
predictions = trainer.predict(test_sequences)
```

### 3\. Transformer Model (TODO)

* **Approach**: Attention-based sequence modeling

* **Pros**: Captures long-range dependencies, parallel training

* **Cons**: More complex, requires careful tuning

## 📈 Evaluation Metrics

The pipeline calculates comprehensive metrics:

* **RMSE**: Root Mean Squared Error (X, Y, Total)

* **MAE**: Mean Absolute Error (X, Y)

* **ADE**: Average Displacement Error (Euclidean distance)

* **FDE**: Final Displacement Error (last frame only)

* **Per-Role Analysis**: Metrics broken down by player role

* **Per-Frame Analysis**: Error progression over time

* **Physical Plausibility**: Speed violations, out-of-bounds checks

```python
from src.evaluate import TrajectoryEvaluator

evaluator = TrajectoryEvaluator()
report = evaluator.generate_report(predictions, ground_truth, input_df)
```

## 📊 Visualization

Generate field visualizations with predicted trajectories:

```python
from src.visualize import TrajectoryVisualizer

visualizer = TrajectoryVisualizer()
visualizer.plot_single_play(game_id, play_id, input_df, predictions, ground_truth)
visualizer.plot_error_distribution(predictions, ground_truth)
visualizer.plot_error_by_role(predictions, ground_truth, input_df)
```

## ⚙️ Configuration

Edit `configs/config.yaml` to customize:

```yaml
# Data splits
processing:
  train_weeks: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
  val_weeks: [15, 16]
  test_weeks: [17, 18]

# Model hyperparameters
models:
  baseline:
    params:
      n_estimators: 500
      max_depth: 8
      learning_rate: 0.05
  
  lstm:
    params:
      hidden_size: 128
      num_layers: 2
      dropout: 0.2
      epochs: 50
```

## 📝 Usage Examples

### Example 1: Train and Evaluate Baseline

```python
from src.train import TrainingPipeline

pipeline = TrainingPipeline()
pipeline.run_full_pipeline(model_type="xgboost")
```

### Example 2: Custom Feature Engineering

```python
from src.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
input_df = engineer.create_all_features(input_df)
feature_cols = engineer.get_feature_names(input_df)
```

### Example 3: Load and Use Trained Model

```python
from src.models.baseline import BaselineModel

model = BaselineModel()
model.load("checkpoints/baseline_xgboost.pkl")
predictions = model.predict(X_test)
```

## 🔍 Key Insights from EDA

* **Data Quality**: No missing values, no duplicates, clean tracking data

* **Strong Signal**: Player X position highly correlates with ball landing X (r=0.875)

* **Role Matters**: Route runners move 2x faster than passers (4.0 vs 1.8 yards/sec)

* **Completion Rate**: 69.3% across all plays

* **Coverage**: Cover 3 Zone most common (31.4%), followed by Cover 1 Man (22.8%)

## 🎓 Best Practices

### Data Handling

* ✅ Split by game_id to avoid data leakage

* ✅ Normalize coordinates by play_direction

* ✅ Handle variable sequence lengths with padding

* ✅ Scale features before training

### Feature Engineering

* ✅ Start with high-priority features

* ✅ Use circular encoding for angles

* ✅ Create frame-to-frame changes

* ✅ Validate no future information leakage

### Model Training

* ✅ Use early stopping on validation set

* ✅ Monitor multiple metrics (RMSE, ADE, FDE)

* ✅ Check per-role performance

* ✅ Visualize predictions on sample plays

## 🐛 Troubleshooting

### Issue: Out of Memory

```python
# Reduce batch size in config.yaml
models:
  lstm:
    params:
      batch_size: 32  # Reduce from 64
```

### Issue: Poor Performance on Specific Roles

```python
# Train separate models per role
train_defense = train_df[train_df['player_side'] == 'Defense']
train_offense = train_df[train_df['player_side'] == 'Offense']
```

### Issue: Predictions Out of Bounds

```python
# Clip predictions to field boundaries
predictions['x'] = predictions['x'].clip(0, 120)
predictions['y'] = predictions['y'].clip(0, 53.3)
```

## 📚 Resources

* [NFL Big Data Bowl 2026](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026)

* [NFL Next Gen Stats](https://nextgenstats.nfl.com/)

* [Previous Winners' Solutions](https://www.kaggle.com/competitions/nfl-big-data-bowl-2025/discussion)

* [Social LSTM Paper](https://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf)

* [Trajectron++ Paper](https://arxiv.org/abs/2001.03093)

## 🤝 Contributing

Contributions welcome! Areas for improvement:

* \[ \] Implement Transformer model

* \[ \] Add Graph Neural Network approach

* \[ \] Multi-agent prediction (predict all players simultaneously)

* \[ \] Physics-based constraints

* \[ \] Ensemble methods

* \[ \] Real-time inference optimization

## 📄 License

MIT License - see LICENSE file for details

## 👥 Authors

Heinz Sports Analytics Club - NFL Big Data Bowl 2026 Team

## 🙏 Acknowledgments

* NFL for providing the tracking data

* Kaggle for hosting the competition

* Previous Big Data Bowl participants for inspiration

---

**Good luck with the challenge! 🏈**