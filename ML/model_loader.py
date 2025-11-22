"""
Model loader module for histogram prediction.
This module can be imported into main.py for inference.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
from pathlib import Path


# ============================================================================
# Model Architecture
# ============================================================================

class MultiheadAttentionBlock(nn.Module):
    """Multi-head attention block."""
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, context=None):
        if context is None:
            context = x
        attn_out, _ = self.attention(x, context, context)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class SetTransformerEncoder(nn.Module):
    """Set Transformer Encoder with Inducing Points."""
    def __init__(self, dim, num_heads, num_inducers, dropout=0.1):
        super().__init__()
        self.inducing_points = nn.Parameter(torch.randn(1, num_inducers, dim))
        self.mab1 = MultiheadAttentionBlock(dim, num_heads, dropout)
        self.mab2 = MultiheadAttentionBlock(dim, num_heads, dropout)

    def forward(self, x):
        batch_size = x.size(0)
        I = self.inducing_points.expand(batch_size, -1, -1)
        H = self.mab1(I, x)
        return self.mab2(x, H)


class FlexibleHistogramPredictor(nn.Module):
    """SetTransformer-based histogram predictor."""
    def __init__(self, hidden_dim=64, num_bins=10, num_heads=4, num_inducers=16, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, hidden_dim)
        self.encoder = SetTransformerEncoder(hidden_dim, num_heads, num_inducers, dropout)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_bins),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.input_proj(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.decoder(x)


# ============================================================================
# Model Loader
# ============================================================================

class HistogramPredictor:
    """
    Wrapper class for histogram prediction model.
    """

    def __init__(self, model_path: str = "ML/best_model_nnj359uw.pt", device: str = None):
        """
        Initialize the predictor.

        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path

        # Load model
        self.model, self.checkpoint = self._load_model()

    def _load_model(self) -> Tuple[nn.Module, dict]:
        """Load the trained model from checkpoint."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Extract config
        config = checkpoint['config']

        # Create model
        model = FlexibleHistogramPredictor(
            hidden_dim=config['hidden_dim'],
            num_bins=10,
            num_heads=config['num_heads'],
            num_inducers=config['num_inducers'],
            dropout=config['dropout']
        )

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        return model, checkpoint

    def predict(self, scores: List[float]) -> dict:
        """
        Predict histogram distribution from sample scores.

        Args:
            scores: List of sample scores (0-100 range)

        Returns:
            Dictionary with histogram and statistics
        """
        # Validate input
        if not scores:
            raise ValueError("Scores list cannot be empty")

        if not all(0 <= s <= 100 for s in scores):
            raise ValueError("All scores must be in range [0, 100]")

        # Preprocess
        scores_array = np.array(scores, dtype=np.float32)
        scores_sorted = np.sort(scores_array)
        scores_norm = scores_sorted / 100.0

        # Convert to tensor
        x = torch.tensor(scores_norm, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            predicted_hist = self.model(x)

        # Convert to dict format
        histogram_values = predicted_hist.cpu().numpy()[0]

        # Create result dictionary
        result = {
            "0-10": float(histogram_values[0]),
            "10-20": float(histogram_values[1]),
            "20-30": float(histogram_values[2]),
            "30-40": float(histogram_values[3]),
            "40-50": float(histogram_values[4]),
            "50-60": float(histogram_values[5]),
            "60-70": float(histogram_values[6]),
            "70-80": float(histogram_values[7]),
            "80-90": float(histogram_values[8]),
            "90-100": float(histogram_values[9]),
        }

        return result

    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_type": "SetTransformer",
            "validation_loss": float(self.checkpoint['val_loss']),
            "epoch": int(self.checkpoint['epoch']),
            "config": self.checkpoint['config'],
            "device": self.device
        }


# ============================================================================
# For testing
# ============================================================================

if __name__ == "__main__":
    # Test the predictor
    predictor = HistogramPredictor()

    # Sample scores
    test_scores = [75, 82, 68, 91, 77, 85, 73, 80, 88, 79]

    # Predict
    result = predictor.predict(test_scores)

    print("Predicted histogram:")
    for bin_range, prob in result.items():
        print(f"  {bin_range}: {prob:.4f}")

    print("\nModel info:")
    info = predictor.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")