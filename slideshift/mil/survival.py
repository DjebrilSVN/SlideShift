"""Survival analysis for MIL: Cox Proportional Hazard loss and C-index.

Used for predicting time-to-event outcomes (e.g. overall survival, recurrence)
from TCGA clinical data. The MIL slide embedding is passed to a risk-score head
trained with the Cox partial likelihood loss.

References:
  - Cox (1972) "Regression models and life tables".
  - Katzman et al., "DeepSurv", 2018.
  - Chen et al., "Pathomic Fusion", 2022.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = hasattr(torch, "__version__") and hasattr(nn, "Module")
except (ImportError, AttributeError):
    torch = None
    nn = None
    F = None
    HAS_TORCH = False


def concordance_index(
    risk_scores: np.ndarray,
    survival_times: np.ndarray,
    event_indicators: np.ndarray,
) -> float:
    """Compute Harrell's concordance index (C-index).

    C-index = P(risk_i > risk_j | T_i < T_j, event_i = 1)

    Args:
        risk_scores:      (N,) predicted risk (higher = worse prognosis).
        survival_times:   (N,) observed survival times.
        event_indicators: (N,) binary; 1 = event occurred, 0 = censored.

    Returns:
        C-index in [0, 1]; 0.5 = random, 1.0 = perfect.
    """
    n = len(risk_scores)
    concordant = 0
    discordant = 0
    tied_risk = 0

    for i in range(n):
        if event_indicators[i] != 1:
            continue
        for j in range(n):
            if survival_times[j] <= survival_times[i]:
                continue
            if risk_scores[i] > risk_scores[j]:
                concordant += 1
            elif risk_scores[i] < risk_scores[j]:
                discordant += 1
            else:
                tied_risk += 1

    total_pairs = concordant + discordant + tied_risk
    if total_pairs == 0:
        return 0.5
    return float((concordant + 0.5 * tied_risk) / total_pairs)


def vectorised_concordance_index(
    risk_scores: np.ndarray,
    survival_times: np.ndarray,
    event_indicators: np.ndarray,
) -> float:
    """Vectorised C-index (faster for N > 100)."""
    order = np.argsort(survival_times)
    t = survival_times[order]
    r = risk_scores[order]
    e = event_indicators[order].astype(bool)

    concordant = 0.0
    discordant = 0.0
    tied_risk = 0.0

    for i in np.where(e)[0]:
        # Compare only to cases with longer survival time
        j_mask = t > t[i]
        r_j = r[j_mask]
        if len(r_j) == 0:
            continue
        concordant += np.sum(r[i] > r_j)
        discordant += np.sum(r[i] < r_j)
        tied_risk += np.sum(r[i] == r_j)

    total = concordant + discordant + tied_risk
    if total == 0:
        return 0.5
    return float((concordant + 0.5 * tied_risk) / total)


if HAS_TORCH:
    def cox_ph_loss(
        risk_scores: "torch.Tensor",
        survival_times: "torch.Tensor",
        event_indicators: "torch.Tensor",
    ) -> "torch.Tensor":
        """Breslow approximation of the Cox partial likelihood loss.

        Args:
            risk_scores:      (N,) predicted log-risk scores.
            survival_times:   (N,) observed times (may be censored).
            event_indicators: (N,) 1 = event, 0 = censored.

        Returns:
            Scalar loss (negative partial log-likelihood).
        """
        # Sort by descending survival time
        order = torch.argsort(survival_times, descending=True)
        risk = risk_scores[order]
        event = event_indicators[order].float()

        # log-sum-exp over risk set (Breslow)
        log_cum_hazard = torch.logcumsumexp(risk, dim=0)

        # Cox partial log-likelihood: sum over events
        uncensored_loss = event * (risk - log_cum_hazard)
        n_events = event.sum()
        if n_events == 0:
            return torch.tensor(0.0, requires_grad=True)

        return -uncensored_loss.sum() / n_events

    class CoxSurvivalHead(nn.Module):
        """Survival prediction head appended to a MIL slide embedding.

        Args:
            input_dim:   Slide embedding size from the MIL aggregator.
            hidden_dim:  Hidden layer size (0 for linear head).
            dropout:     Dropout probability.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 128,
            dropout: float = 0.25,
        ):
            super().__init__()
            if hidden_dim > 0:
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )
            else:
                self.net = nn.Linear(input_dim, 1)

        def forward(self, slide_embedding: "torch.Tensor") -> "torch.Tensor":
            """Returns (N, 1) log-risk scores."""
            return self.net(slide_embedding)

    class SurvivalMILTrainer:
        """Thin trainer wrapper for survival analysis with a MIL backbone.

        Expects the backbone to already be trained (or jointly fine-tuned).
        """

        def __init__(
            self,
            survival_head: CoxSurvivalHead,
            device: str = "cpu",
            lr: float = 1e-4,
            weight_decay: float = 1e-4,
        ):
            self.head = survival_head
            dev = torch.device(device if HAS_TORCH and torch.cuda.is_available() else "cpu")
            self.head = self.head.to(dev)
            self.device = dev
            self.optimizer = torch.optim.AdamW(
                self.head.parameters(), lr=lr, weight_decay=weight_decay
            )

        def train_step(
            self,
            embeddings: np.ndarray,
            times: np.ndarray,
            events: np.ndarray,
        ) -> float:
            """Single gradient step on a batch of slide embeddings."""
            self.head.train()
            emb_t = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
            t_t = torch.tensor(times, dtype=torch.float32, device=self.device)
            e_t = torch.tensor(events, dtype=torch.float32, device=self.device)

            self.optimizer.zero_grad()
            risk = self.head(emb_t).squeeze(-1)
            loss = cox_ph_loss(risk, t_t, e_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.head.parameters(), 1.0)
            self.optimizer.step()
            return float(loss.item())

        def predict(self, embeddings: np.ndarray) -> np.ndarray:
            """Predict log-risk scores (higher = worse prognosis)."""
            self.head.eval()
            with torch.no_grad():
                emb_t = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
                risk = self.head(emb_t).squeeze(-1)
            return risk.cpu().numpy().astype(np.float32)

        def evaluate(
            self,
            embeddings: np.ndarray,
            times: np.ndarray,
            events: np.ndarray,
        ) -> Dict[str, float]:
            """Report C-index and loss on a validation set."""
            risk = self.predict(embeddings)
            c_idx = vectorised_concordance_index(risk, times, events)

            with torch.no_grad():
                emb_t = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
                t_t = torch.tensor(times, dtype=torch.float32, device=self.device)
                e_t = torch.tensor(events, dtype=torch.float32, device=self.device)
                risk_t = self.head(emb_t).squeeze(-1)
                loss = float(cox_ph_loss(risk_t, t_t, e_t).item())

            return {"c_index": c_idx, "cox_loss": loss}

else:
    def cox_ph_loss(risk_scores, survival_times, event_indicators):  # type: ignore
        """Numpy fallback — returns zero (no gradient)."""
        return 0.0

    class CoxSurvivalHead:  # type: ignore
        def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.25):
            self.input_dim = input_dim

        def __call__(self, embedding: np.ndarray) -> np.ndarray:
            return np.zeros(embedding.shape[0], dtype=np.float32)

    class SurvivalMILTrainer:  # type: ignore
        def __init__(self, survival_head, device="cpu", lr=1e-4, weight_decay=1e-4):
            self.head = survival_head

        def train_step(self, embeddings, times, events):
            return 0.0

        def predict(self, embeddings):
            return np.zeros(len(embeddings), dtype=np.float32)

        def evaluate(self, embeddings, times, events):
            return {"c_index": 0.5, "cox_loss": 0.0}

