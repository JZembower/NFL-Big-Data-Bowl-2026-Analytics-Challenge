# Quick Reference Guide

## 🚀 Common Commands

### Setup

```bash
# Initial setup
chmod +x setup.sh && ./setup.sh

# Activate environment
source venv/bin/activate

# Install/update dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train XGBoost baseline
python src/train.py --model xgboost

# Train LightGBM
python src/train.py --model lightgbm

# Train LSTM
python src/train.py --model lstm

# Force reload data
python src/train.py --model xgboost --force-reload
```

### Individual Steps

```bash
# Data preparation only
python src/train.py --step data

# Feature engineering only
python src/train.py --step features

# Training only
python src/train.py --step train --model xgboost

# Evaluation only
python src/train.py --step evaluate
```

## 📝 Code Snippets

### Load Data

```python
from src.data_loader import NFLDataLoader

loader = NFLDataLoader()
input_df, output_df = loader.load_all_weeks()
supp_df = loader.load_supplementary_data()
input_df = loader.merge_with_supplementary(input_df, supp_df)
```

### Create Features

```python
from src.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
input_df = engineer.create_all_features(input_df)
feature_cols = engineer.get_feature_names(input_df)
print(f"Created {len(feature_cols)} features")
```

### Train Baseline Model

```python
from src.models.baseline import BaselineModel

model = BaselineModel(model_type="xgboost")
X_train, y_x_train, y_y_train = model.prepare_data(train_input, train_output, feature_cols)
model.train(X_train, y_x_train, y_y_train)
metrics = model.evaluate(X_test, y_x_test, y_y_test)
model.save("checkpoints/my_model.pkl")
```

### Train LSTM Model

```python
from src.models.lstm import LSTMTrainer

trainer = LSTMTrainer()
train_seq = trainer.prepare_sequences(train_input, train_output, feature_cols)
val_seq = trainer.prepare_sequences(val_input, val_output, feature_cols)
trainer.train(train_seq, val_seq)
```

### Evaluate Predictions

```python
from src.evaluate import TrajectoryEvaluator

evaluator = TrajectoryEvaluator()
metrics = evaluator.calculate_metrics(predictions, ground_truth)
role_metrics = evaluator.evaluate_by_role(predictions, ground_truth, input_df)
report = evaluator.generate_report(predictions, ground_truth, input_df)
```

### Visualize Results

```python
from src.visualize import TrajectoryVisualizer

viz = TrajectoryVisualizer()
viz.plot_single_play(game_id, play_id, input_df, predictions, ground_truth)
viz.plot_error_distribution(predictions, ground_truth)
viz.plot_error_by_role(predictions, ground_truth, input_df)
viz.create_sample_visualizations(input_df, predictions, ground_truth, n_samples=10)
```

## 🔧 Configuration Quick Edit

Edit `configs/config.yaml`:

```yaml
# Change train/val/test split
processing:
  train_weeks: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  val_weeks: [13, 14, 15]
  test_weeks: [16, 17, 18]

# Toggle features
features:
  velocity_components: true
  player_interactions: false  # Disable slow features

# Adjust model hyperparameters
models:
  baseline:
    params:
      n_estimators: 1000  # More trees
      max_depth: 10       # Deeper trees
      learning_rate: 0.03 # Slower learning
```

## 📊 Key Metrics

### Baseline Performance Targets

* **RMSE**: < 3.0 yards

* **ADE**: < 4.0 yards

* **FDE**: < 5.0 yards

### LSTM Performance Targets

* **RMSE**: < 2.5 yards

* **ADE**: < 3.0 yards

* **FDE**: < 4.0 yards

## 🐛 Troubleshooting

### Issue: Module not found

```bash
# Make sure you're in the right directory
cd /path/to/project
source venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Issue: Out of memory

```python
# Reduce batch size in config.yaml
models:
  lstm:
    params:
      batch_size: 32  # Reduce from 64
```

### Issue: Slow feature engineering

```python
# Disable expensive features in config.yaml
features:
  player_interactions: false  # This is slow
  trajectory_features: false  # This too
```

### Issue: Poor predictions

```python
# Check feature importance
importance = model.get_feature_importance(feature_cols)
print(importance.head(20))

# Visualize predictions
viz.plot_single_play(game_id, play_id, input_df, predictions, ground_truth)

# Check per-role performance
role_metrics = evaluator.evaluate_by_role(predictions, ground_truth, input_df)
```

## 📁 File Locations

### Input Data

* Raw CSVs: `data/raw/input_2023_w*.csv`

* Processed: `data/processed/input_processed.parquet`

* With features: `data/processed/input_with_features_train.parquet`

### Models

* Checkpoints: `checkpoints/baseline_xgboost.pkl`

* LSTM: `checkpoints/lstm_best.pth`

### Outputs

* Predictions: `predictions/test_predictions.csv`

* Visualizations: `visualizations/*.png`

* Logs: `logs/*.csv`

## 🎯 Feature Categories

### High Priority (Start Here)

1. velocity_x, velocity_y

2. dist_to_ball_land

3. dir_sin, dir_cos, o_sin, o_cos

4. speed_change, dir_change

5. is_targeted_receiver, is_defense

### Medium Priority

1. speed_ma3, speed_ma5

2. dist_to_los, dist_to_endzone

3. team_avg_speed, team_spread_y

4. down, yards_to_go, is_man_coverage

### Advanced

1. dist_to_nearest_opponent

2. path_efficiency, cumulative_distance

3. Custom features

## 📈 Performance Optimization

### Speed up data loading

```python
# Use parquet instead of CSV
loader.save_processed_data(input_df, output_df)
input_df, output_df = loader.load_processed_data()  # Much faster
```

### Speed up feature engineering

```python
# Cache features
features_file = "data/processed/input_with_features.parquet"
if Path(features_file).exists():
    input_df = pd.read_parquet(features_file)
else:
    input_df = engineer.create_all_features(input_df)
    input_df.to_parquet(features_file)
```

### Speed up training

```python
# Use GPU for LSTM
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Reduce data for quick experiments
train_input_sample = train_input.sample(frac=0.1)
```

## 🔍 Debugging Tips

### Check data shapes

```python
print(f"Input shape: {input_df.shape}")
print(f"Output shape: {output_df.shape}")
print(f"Features: {len(feature_cols)}")
print(f"Unique plays: {input_df.groupby(['game_id', 'play_id']).ngroups}")
```

### Check for NaN values

```python
print(input_df.isnull().sum())
print(input_df[feature_cols].isnull().sum().sort_values(ascending=False).head(10))
```

### Visualize feature distributions

```python
import matplotlib.pyplot as plt
input_df[feature_cols[:10]].hist(figsize=(15, 10), bins=50)
plt.tight_layout()
plt.show()
```

### Check predictions

```python
print(f"Predictions shape: {predictions.shape}")
print(f"X range: [{predictions['x'].min():.2f}, {predictions['x'].max():.2f}]")
print(f"Y range: [{predictions['y'].min():.2f}, {predictions['y'].max():.2f}]")
print(f"Out of bounds: {((predictions['x'] < 0) | (predictions['x'] > 120)).sum()}")
```

## 💡 Pro Tips

1. **Start simple**: Train baseline first, then add complexity

2. **Visualize early**: Look at predictions on field before optimizing

3. **Check per-role**: Different roles may need different approaches

4. **Feature importance**: Let the model tell you what matters

5. **Validate assumptions**: Check coordinate normalization worked

6. **Save checkpoints**: Don't lose progress from long training runs

7. **Use config file**: Easier to track experiments than code changes

8. **Version control**: Git commit after each successful experiment

## 📚 Resources

* **Documentation**: See [README.md](http://README.md)

* **Project Summary**: See PROJECT_SUMMARY.md

* **Configuration**: configs/config.yaml

* **Examples**: notebooks/\*.ipynb

---

**Keep this file handy for quick reference during development!**