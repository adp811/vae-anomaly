"""Composite VAE loss: Huber + categorical cross-entropy + KL.

Training minimizes all three. Scoring reuses the Huber and cross-entropy
terms only. KL is a training regularizer on q(z | x); it is not part of the
anomaly score. The weights and the Huber delta live on one frozen config so
the training job and the scorer cannot drift apart inside a process.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class LossConfig:
    """Knobs that have to be stored with the trained weights.

    lambda_cont, lambda_cat, and beta scale the three training terms.
    huber_delta is part of the definition of the continuous error, so a
    scorer that uses a different delta is a different model.
    """

    lambda_cont: float = 1.0
    lambda_cat: float = 1.0
    beta: float = 0.1
    huber_delta: float = 1.0

    def __post_init__(self) -> None:
        if self.huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        if self.lambda_cont < 0 or self.lambda_cat < 0 or self.beta < 0:
            raise ValueError("loss weights must be non-negative")


class VAELoss(nn.Module):
    """total = lambda_cont * Huber + lambda_cat * CE + beta * KL.

    Reductions used for the training objective (stable gradients, comparable
    across batch sizes):

    - Huber: mean over the batch and over continuous features.
    - Cross-entropy: mean over the batch for each head, then summed across
      heads so every categorical feature contributes.
    - KL: closed form between q(z|x)=N(mu, diag(exp(logvar))) and N(0, I),
      summed over latent dimensions and then averaged over the batch.

    Per-row scoring uses a different reduction (sum within the row) but the
    same functions, the same delta, and the same lambda_cont / lambda_cat.
    The threshold is calibrated in that per-row scale, not in this mean scale.
    """

    def __init__(self, config: LossConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else LossConfig()

    def forward(
        self,
        recon_continuous: torch.Tensor,
        continuous: torch.Tensor,
        categorical_logits: list[torch.Tensor],
        categorical: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return total, huber, cross_entropy, kl.

        The three components are the unweighted reductions. ``total`` applies
        the config weights. Logging the components separately is how a run
        is checked for one term drowning the others.
        """

        huber = F.huber_loss(
            recon_continuous,
            continuous,
            delta=self.config.huber_delta,
            reduction="mean",
        )
        n_columns = int(categorical.shape[1]) if categorical.ndim == 2 else 0
        if len(categorical_logits) != n_columns:
            raise ValueError(
                f"got {len(categorical_logits)} categorical heads for "
                f"{n_columns} categorical columns"
            )
        # No heads: the categorical term is exactly zero, not a dummy class.
        if len(categorical_logits) == 0:
            cross_entropy = recon_continuous.new_zeros(())
        else:
            cross_entropy = torch.stack(
                [
                    F.cross_entropy(logits, categorical[:, index], reduction="mean")
                    for index, logits in enumerate(categorical_logits)
                ]
            ).sum()
        # -0.5 * sum_j (1 + logvar_j - mu_j^2 - exp(logvar_j)), then mean
        # over the batch. That is the usual diagonal-Gaussian KL, normalized
        # by batch size rather than left as a sum that scales with the batch.
        kl_per_row = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl = kl_per_row.mean()

        total = (
            self.config.lambda_cont * huber
            + self.config.lambda_cat * cross_entropy
            + self.config.beta * kl
        )
        return total, huber, cross_entropy, kl
