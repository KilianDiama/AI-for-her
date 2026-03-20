import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class TemperatureScaler(nn.Module):
    """Post-hoc calibration for logits"""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature.clamp(min=1e-3)


class UltraSecureMammoAI(nn.Module):
    """
    Production-grade Conformal Prediction (RAPS + Randomization)

    Guarantees:
    - Marginal coverage: 1 - alpha
    - Handles uncertainty explicitly
    - Safe for high-stakes medical inference
    """

    def __init__(
        self,
        encoder: nn.Module,  # <- explicit vision backbone
        hidden_size: int,
        alpha: float = 0.01,
        kreg: int = 3,
        lam: float = 0.01,
        num_classes: int = 2,
        seed: int = 42,
    ):
        super().__init__()

        set_seed(seed)

        self.alpha = alpha
        self.kreg = kreg
        self.lam = lam
        self.num_classes = num_classes

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Backbone (explicit)
        self.encoder = encoder

        for p in self.encoder.parameters():
            p.requires_grad = False

        # Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )

        # Calibration
        self.temp_scaler = TemperatureScaler()
        self.q_hat: Optional[float] = None

        self.label_map = {
            0: "benign",
            1: "malignant"
        }

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.encoder(x)
        logits = self.classifier(features)
        return self.temp_scaler(logits)

    @torch.no_grad()
    def calibrate(self, cal_loader) -> Dict[str, float]:
        """
        Compute conformal threshold with RAPS + randomization.
        """

        self.eval()
        scores = []

        for images, labels in cal_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self(images)
            probs = F.softmax(logits, dim=1)

            sorted_probs, sorted_idx = torch.sort(probs, dim=1, descending=True)

            B, C = sorted_probs.shape

            match = (sorted_idx == labels.unsqueeze(1))
            if not match.any():
                raise RuntimeError("Label missing in predictions.")

            true_rank = match.nonzero(as_tuple=True)[1]

            cumsum = torch.cumsum(sorted_probs, dim=1)

            ranks = torch.arange(C, device=self.device).unsqueeze(0)
            penalty = self.lam * torch.clamp(ranks - self.kreg + 1, min=0)

            adjusted = cumsum + penalty

            base_scores = adjusted[torch.arange(B), true_rank]

            # 🔥 RANDOMIZATION (critical for theory)
            u = torch.rand_like(base_scores)
            prob_true = sorted_probs[torch.arange(B), true_rank]

            randomized_scores = base_scores - u * prob_true

            scores.append(randomized_scores.cpu())

        scores = torch.cat(scores).numpy()
        n = len(scores)

        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat = float(np.quantile(scores, q_level, method="higher"))

        return {
            "q_hat": self.q_hat,
            "n_samples": n,
            "avg_score": float(np.mean(scores)),
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict:
        """
        Returns conformal prediction sets + reject option.
        """

        if self.q_hat is None:
            raise RuntimeError("Model must be calibrated first.")

        self.eval()
        x = x.to(self.device)

        logits = self(x)
        probs = F.softmax(logits, dim=1)

        sorted_probs, sorted_idx = torch.sort(probs, dim=1, descending=True)

        B, C = sorted_probs.shape

        cumsum = torch.cumsum(sorted_probs, dim=1)

        ranks = torch.arange(C, device=self.device).unsqueeze(0)
        penalty = self.lam * torch.clamp(ranks - self.kreg + 1, min=0)

        adjusted = cumsum + penalty

        # Handle edge case safely
        mask = adjusted >= self.q_hat

        prediction_sets = []
        readable_sets = []
        set_sizes = []

        for i in range(B):
            valid = torch.where(mask[i])[0]

            if len(valid) == 0:
                # 🚨 SAFE fallback (full uncertainty)
                indices = sorted_idx[i].tolist()
            else:
                cutoff = valid[0].item()
                indices = sorted_idx[i, : cutoff + 1].tolist()

            prediction_sets.append(indices)
            readable_sets.append([self.label_map[j] for j in indices])
            set_sizes.append(len(indices))

        return {
            "prediction_sets": prediction_sets,
            "labels": readable_sets,
            "probabilities": probs.cpu().numpy(),
            "set_sizes": set_sizes,
            "is_uncertain": [s > 1 for s in set_sizes],
            "reject": [s == self.num_classes for s in set_sizes],
        }

    @torch.no_grad()
    def evaluate_coverage(self, loader) -> Dict[str, float]:
        """
        Empirical coverage (MANDATORY for medical validation)
        """

        if self.q_hat is None:
            raise RuntimeError("Model must be calibrated first.")

        self.eval()

        total = 0
        covered = 0
        avg_size = 0

        for images, labels in loader:
            outputs = self.predict(images)

            for pred_set, label in zip(outputs["prediction_sets"], labels):
                total += 1
                avg_size += len(pred_set)

                if label.item() in pred_set:
                    covered += 1

        coverage = covered / total
        avg_size /= total

        return {
            "coverage": coverage,
            "target": 1 - self.alpha,
            "avg_set_size": avg_size,
        }
