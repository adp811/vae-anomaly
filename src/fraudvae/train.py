"""Offline training loop. This module is not on the request path.

A later training job would call VAETrainer.fit on the normal-only loader,
then hand the state_dict, the preprocessor, the loss config, and a threshold
fit elsewhere to the scorer. Nothing here loads a request or computes the
deployment threshold.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from fraudvae.loss import VAELoss
from fraudvae.model import TabularVAE


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for a CPU run.

    The synthetic generator also takes ``seed`` on its own Generator, and
    the training DataLoader takes its own torch.Generator. Setting the
    globals here covers model initialization and the reparameterization
    noise, which both read PyTorch's RNG.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class VAETrainer:
    """Adam on the normal-only loader, printing the three loss terms each epoch.

    The hidden label yielded by the dataset is ignored. Fitting on fraud, or
    even filtering with the label inside the loss, would no longer be the
    zero-shot setup.
    """

    def __init__(
        self,
        model: TabularVAE,
        loss_fn: VAELoss,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        if device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError(f"requested device {device!r} but CUDA is not available")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def fit(self, loader: DataLoader, epochs: int) -> list[dict[str, float]]:
        """Train and return one metrics dict per epoch.

        Each dict has total, huber, cross_entropy, and kl as sample-weighted
        means of the batch reductions, so a short last batch does not count
        as much as a full one. The printout is the check that the weighted
        terms are in the same ballpark.
        """

        if epochs < 1:
            raise ValueError("epochs must be positive")
        history: list[dict[str, float]] = []
        self.model.train()
        for epoch in range(1, epochs + 1):
            total_sum = 0.0
            huber_sum = 0.0
            cross_entropy_sum = 0.0
            kl_sum = 0.0
            seen = 0
            for continuous, categorical, _labels in loader:
                # _labels is the hidden fraud flag. It is not an input.
                continuous = continuous.to(self.device)
                categorical = categorical.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                recon_continuous, categorical_logits, mu, logvar = self.model(continuous, categorical)
                total, huber, cross_entropy, kl = self.loss_fn(
                    recon_continuous,
                    continuous,
                    categorical_logits,
                    categorical,
                    mu,
                    logvar,
                )
                total.backward()
                self.optimizer.step()

                batch_size = int(continuous.shape[0])
                total_sum += float(total.item()) * batch_size
                huber_sum += float(huber.item()) * batch_size
                cross_entropy_sum += float(cross_entropy.item()) * batch_size
                kl_sum += float(kl.item()) * batch_size
                seen += batch_size

            if seen == 0:
                raise RuntimeError("training loader produced no rows")
            row: dict[str, Any] = {
                "epoch": epoch,
                "total": total_sum / seen,
                "huber": huber_sum / seen,
                "cross_entropy": cross_entropy_sum / seen,
                "kl": kl_sum / seen,
            }
            history.append(row)
            print(
                f"epoch {epoch:02d}  "
                f"total={row['total']:.4f}  "
                f"huber={row['huber']:.4f}  "
                f"cross_entropy={row['cross_entropy']:.4f}  "
                f"kl={row['kl']:.4f}"
            )
        return history
