"""
LSTM Model Module
Sequence-to-sequence model for trajectory prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrajectoryDataset(Dataset):
    """Dataset for trajectory sequences"""
    
    def __init__(self, input_sequences: np.ndarray, output_sequences: np.ndarray):
        """
        Args:
            input_sequences: (N, seq_len, features)
            output_sequences: (N, output_len, 2) for (x, y)
        """
        self.input_sequences = torch.FloatTensor(input_sequences)
        self.output_sequences = torch.FloatTensor(output_sequences)
    
    def __len__(self):
        return len(self.input_sequences)
    
    def __getitem__(self, idx):
        return self.input_sequences[idx], self.output_sequences[idx]


class LSTMTrajectoryModel(nn.Module):
    """LSTM model for trajectory prediction"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 output_frames: int = 10,
                 dropout: float = 0.2,
                 bidirectional: bool = True):
        """Initialize LSTM model"""
        super(LSTMTrajectoryModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_frames = output_frames
        self.bidirectional = bidirectional
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate LSTM output size
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Decoder layers
        self.fc1 = nn.Linear(lstm_output_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_frames * 2)  # Predict x, y for each frame
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            (batch_size, output_frames, 2)
        """
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        # Decode
        out = self.relu(self.fc1(hidden))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        
        # Reshape to (batch_size, output_frames, 2)
        out = out.view(-1, self.output_frames, 2)
        
        return out


class LSTMTrainer:
    """Trainer for LSTM trajectory model"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize trainer"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['models']['lstm']['params']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
    
    def prepare_sequences(self,
                         input_df: pd.DataFrame,
                         output_df: pd.DataFrame,
                         feature_cols: List[str],
                         seq_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequence data for LSTM"""
        logger.info("Preparing sequences for LSTM...")
        
        input_sequences = []
        output_sequences = []
        
        # Group by player in each play
        for (game_id, play_id, nfl_id), group in input_df.groupby(['game_id', 'play_id', 'nfl_id']):
            # Sort by frame
            group = group.sort_values('frame_id')
            
            # Get input sequence (last seq_length frames)
            if len(group) >= seq_length:
                input_seq = group[feature_cols].tail(seq_length).values
            else:
                # Pad if not enough frames
                input_seq = group[feature_cols].values
                padding = np.zeros((seq_length - len(input_seq), len(feature_cols)))
                input_seq = np.vstack([padding, input_seq])
            
            # Get output sequence
            output_group = output_df[
                (output_df['game_id'] == game_id) &
                (output_df['play_id'] == play_id) &
                (output_df['nfl_id'] == nfl_id)
            ].sort_values('frame_id')
            
            if len(output_group) > 0:
                # Get x, y positions
                output_seq = output_group[['x', 'y']].values
                
                input_sequences.append(input_seq)
                output_sequences.append(output_seq)
        
        # Convert to arrays
        input_sequences = np.array(input_sequences)
        
        # Pad output sequences to same length
        max_output_len = max(len(seq) for seq in output_sequences)
        padded_output = np.zeros((len(output_sequences), max_output_len, 2))
        
        for i, seq in enumerate(output_sequences):
            padded_output[i, :len(seq), :] = seq
        
        # Handle NaN values
        input_sequences = np.nan_to_num(input_sequences, nan=0.0)
        padded_output = np.nan_to_num(padded_output, nan=0.0)
        
        logger.info(f"Prepared {len(input_sequences)} sequences")
        logger.info(f"Input shape: {input_sequences.shape}")
        logger.info(f"Output shape: {padded_output.shape}")
        
        return input_sequences, padded_output
    
    def create_model(self, input_size: int, output_frames: int):
        """Create LSTM model"""
        self.model = LSTMTrajectoryModel(
            input_size=input_size,
            hidden_size=self.model_config['hidden_size'],
            num_layers=self.model_config['num_layers'],
            output_frames=output_frames,
            dropout=self.model_config['dropout'],
            bidirectional=self.model_config['bidirectional']
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.model_config['learning_rate']
        )
        
        logger.info(f"Created LSTM model with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Calculate loss (only on non-padded frames)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_errors = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                # Calculate displacement errors
                errors = torch.sqrt(((outputs - targets) ** 2).sum(dim=2))
                all_errors.append(errors.cpu().numpy())
        
        all_errors = np.concatenate(all_errors, axis=0)
        
        metrics = {
            'val_loss': total_loss / len(val_loader),
            'ade': all_errors.mean(),  # Average Displacement Error
            'fde': all_errors[:, -1].mean()  # Final Displacement Error
        }
        
        return metrics
    
    def train(self,
              train_sequences: Tuple[np.ndarray, np.ndarray],
              val_sequences: Tuple[np.ndarray, np.ndarray],
              epochs: int = None,
              batch_size: int = None):
        """Train LSTM model"""
        if epochs is None:
            epochs = self.model_config['epochs']
        if batch_size is None:
            batch_size = self.model_config['batch_size']
        
        # Create datasets
        train_dataset = TrajectoryDataset(*train_sequences)
        val_dataset = TrajectoryDataset(*val_sequences)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model if not exists
        if self.model is None:
            input_size = train_sequences[0].shape[2]
            output_frames = train_sequences[1].shape[1]
            self.create_model(input_size, output_frames)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config['training']['early_stopping_patience']
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"ADE: {val_metrics['ade']:.4f}, "
                f"FDE: {val_metrics['fde']:.4f}"
            )
            
            # Early stopping
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                patience_counter = 0
                self.save_checkpoint('checkpoints/lstm_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        logger.info("Training complete!")
    
    def predict(self, input_sequences: np.ndarray) -> np.ndarray:
        """Predict trajectories"""
        self.model.eval()
        
        dataset = TrajectoryDataset(input_sequences, np.zeros((len(input_sequences), 1, 2)))
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        predictions = []
        
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                predictions.append(outputs.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model_config
        }, path)
        
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Recreate model if needed
        if self.model is None:
            # You'll need to pass the correct dimensions
            raise ValueError("Model must be created before loading checkpoint")
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Checkpoint loaded from {path}")


def main():
    """Example usage of LSTM model"""
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
    
    # Initialize trainer
    trainer = LSTMTrainer()
    
    # Prepare sequences
    train_sequences = trainer.prepare_sequences(train_input, train_output, feature_cols)
    val_sequences = trainer.prepare_sequences(val_input, val_output, feature_cols)
    
    # Train
    trainer.train(train_sequences, val_sequences)
    
    print("\nLSTM training complete!")


if __name__ == "__main__":
    main()