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
        # x: [batch, sample_size] (0~1로 정규화된 점수)
        x = x.unsqueeze(-1)  # [batch, sample_size, 1]
        x = self.input_proj(x)  # [batch, sample_size, hidden_dim]
        x = self.encoder(x)  # [batch, sample_size, hidden_dim]
        x = x.mean(dim=1)  # [batch, hidden_dim]
        return self.decoder(x)  # [batch, num_bins], 확률 분포


# ============================================================================
# Synthetic 데이터 생성 & 지표 함수
# ============================================================================

NUM_STUDENTS = 30  # 한 반 학생 수
NUM_BINS = 10  # 히스토그램 bin 수 (0-10, 10-20, ...)


def generate_class_scores() -> Tuple[np.ndarray, str]:
    """
    한 반(30명)의 점수를 생성하고, 어떤 타입인지도 같이 반환.
    easy / normal / hard / bimodal 네 가지 타입 중 하나.
    """
    class_type = np.random.choice(["easy", "normal", "hard", "bimodal"])

    if class_type == "easy":
        mu = np.random.uniform(75, 90)
        sigma = np.random.uniform(5, 10)
        scores = np.random.normal(mu, sigma, NUM_STUDENTS)
    elif class_type == "normal":
        mu = np.random.uniform(60, 80)
        sigma = np.random.uniform(8, 15)
        scores = np.random.normal(mu, sigma, NUM_STUDENTS)
    elif class_type == "hard":
        mu = np.random.uniform(40, 65)
        sigma = np.random.uniform(8, 15)
        scores = np.random.normal(mu, sigma, NUM_STUDENTS)
    elif class_type == "bimodal":
        mu1 = np.random.uniform(40, 60)
        mu2 = np.random.uniform(70, 90)
        sigma = np.random.uniform(5, 10)
        scores1 = np.random.normal(mu1, sigma, NUM_STUDENTS // 2)
        scores2 = np.random.normal(mu2, sigma, NUM_STUDENTS - NUM_STUDENTS // 2)
        scores = np.concatenate([scores1, scores2])
    else:
        raise ValueError("Unknown class type")

    scores = np.clip(scores, 0, 100)
    return scores.astype(np.float32), class_type


def scores_to_hist(scores: np.ndarray, num_bins: int = NUM_BINS) -> np.ndarray:
    """
    점수 배열을 [0,100] 구간 num_bins개로 나눈 히스토그램 (비율)로 변환.
    """
    bins = np.linspace(0, 100, num_bins + 1)
    hist, _ = np.histogram(scores, bins=bins)
    hist_prob = hist.astype(np.float32) / len(scores)
    return hist_prob


def js_divergence_1d(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    1D JS divergence for two histograms.
    p, q: [num_bins] 형태의 텐서 (합이 1인 확률분포)
    """
    p = torch.clamp(p, min=0.0) + eps
    q = torch.clamp(q, min=0.0) + eps
    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum()
    kl_qm = (q * (q / m).log()).sum()
    return 0.5 * (kl_pm + kl_qm)


def wasserstein_1d(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    1D Earth Mover's Distance (Wasserstein-1).
    p, q: [num_bins] 확률분포
    """
    cdf_p = torch.cumsum(p, dim=-1)
    cdf_q = torch.cumsum(q, dim=-1)
    emd = torch.mean(torch.abs(cdf_p - cdf_q))
    return emd


def evaluate_on_synthetic_data(predictor: "HistogramPredictor",
                               num_classes: int = 200,
                               sample_size: int = 10) -> dict:
    """
    시뮬레이션으로 num_classes개의 반을 생성해서
    - MSE
    - JS divergence
    - EMD(1-Wasserstein)
    평균값을 계산해주는 함수.
    """
    device = predictor.device
    model = predictor.model
    model.eval()

    mse_list = []
    js_list = []
    emd_list = []

    for _ in range(num_classes):
        # 1) 한 반 생성 (30명)
        scores_all, _ctype = generate_class_scores()

        # 2) 그 중 sample_size명만 뽑아서 모델 입력으로 사용
        idx = np.random.choice(len(scores_all), sample_size, replace=False)
        sample_scores = np.sort(scores_all[idx])
        sample_scores_norm = sample_scores / 100.0  # 0~1 정규화

        # 3) GT 히스토그램 (확률분포)
        true_hist = scores_to_hist(scores_all, num_bins=NUM_BINS)  # numpy [10]
        true_hist_t = torch.tensor(true_hist, dtype=torch.float32, device=device)

        # 4) 모델 예측 (확률분포)
        x = torch.tensor(sample_scores_norm, dtype=torch.float32, device=device).unsqueeze(0)  # [1, sample_size]
        with torch.no_grad():
            pred_hist = model(x)[0]  # [num_bins]
        # pred_hist는 Softmax 거친 확률분포

        # 5) 지표 계산
        mse = torch.mean((true_hist_t - pred_hist) ** 2).item()
        js = js_divergence_1d(true_hist_t, pred_hist).item()
        emd = wasserstein_1d(true_hist_t, pred_hist).item()

        mse_list.append(mse)
        js_list.append(js)
        emd_list.append(emd)

    results = {
            "num_classes": num_classes,
            "MSE"        : float(np.mean(mse_list)),
            "JS"         : float(np.mean(js_list)),
            "EMD"        : float(np.mean(emd_list)),
    }
    return results


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

    def predict(self, scores: List[float], total_students: int = None) -> dict:
        """
        Predict histogram distribution from sample scores.

        Args:
            scores: List of sample scores (0-100 range)
            total_students: Total number of students in class (for denormalization)
                          If None, returns probabilities (0-1 range)

        Returns:
            Dictionary with histogram (either probabilities or student counts)
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

        # Denormalize to student counts if total_students is provided
        if total_students is not None:
            histogram_values = histogram_values * total_students
            # Round to nearest integer
            histogram_values = np.round(histogram_values).astype(int)

        # Create result dictionary
        result = {
                "0-10"  : int(histogram_values[0]) if total_students else float(histogram_values[0]),
                "10-20" : int(histogram_values[1]) if total_students else float(histogram_values[1]),
                "20-30" : int(histogram_values[2]) if total_students else float(histogram_values[2]),
                "30-40" : int(histogram_values[3]) if total_students else float(histogram_values[3]),
                "40-50" : int(histogram_values[4]) if total_students else float(histogram_values[4]),
                "50-60" : int(histogram_values[5]) if total_students else float(histogram_values[5]),
                "60-70" : int(histogram_values[6]) if total_students else float(histogram_values[6]),
                "70-80" : int(histogram_values[7]) if total_students else float(histogram_values[7]),
                "80-90" : int(histogram_values[8]) if total_students else float(histogram_values[8]),
                "90-100": int(histogram_values[9]) if total_students else float(histogram_values[9]),
        }

        return result

    def get_model_info(self) -> dict:
        """Get model information."""
        return {
                "model_type"     : "SetTransformer",
                "validation_loss": float(self.checkpoint['val_loss']),
                "epoch"          : int(self.checkpoint['epoch']),
                "config"         : self.checkpoint['config'],
                "device"         : self.device
        }


# ============================================================================
# For testing
# ============================================================================

if __name__ == "__main__":
    # Test the predictor
    predictor = HistogramPredictor("best_model_nnj359uw.pt")

    # Sample scores
    test_scores = [75, 82, 68, 91, 77, 85, 73, 80, 88, 79]

    # Predict
    result = predictor.predict(test_scores)

    print("Predicted histogram (probabilities):")
    for bin_range, prob in result.items():
        print(f"  {bin_range}: {prob:.4f}")

    print("\nModel info:")
    info = predictor.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\nEvaluating on synthetic data...")
    metrics = evaluate_on_synthetic_data(predictor, num_classes=200, sample_size=len(test_scores))
    print("Synthetic evaluation results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
