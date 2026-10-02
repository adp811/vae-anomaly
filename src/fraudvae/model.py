"""Tabular VAE with a shared trunk and branched reconstruction heads.

The same module is used in the training job and, later, in the scorer.
Nothing in here knows whether a row is fraud. The encoder sees continuous
features concatenated with one-hot categoricals; the decoder reconstructs
the continuous columns from one linear head and emits raw logits for each
categorical column from its own linear head.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TabularVAE(nn.Module):
    """Encoder–latent–branched-decoder for mixed tabular transactions.

    Parameters
    ----------
    n_continuous:
        Number of scaled continuous inputs. Defaults to the five transaction
        fields, but the width is a constructor argument so a later schema
        change does not require an architecture rewrite.
    cat_cardinalities:
        Class counts in categorical-feature order. Merchant then device is
        (10, 3) for the synthetic schema. One-hot width is the sum.
    latent_dim:
        Size of z. Default is 6. The normal table is generated from about
        three factors, so this is still a bottleneck on the 18-wide
        continuous-plus-one-hot input.
    hidden_dim:
        Width of the encoder and decoder trunks.
    """

    def __init__(
        self,
        n_continuous: int,
        cat_cardinalities: tuple[int, ...] | list[int],
        latent_dim: int = 6,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if n_continuous < 1:
            raise ValueError("n_continuous must be positive")
        if not cat_cardinalities or any(card < 2 for card in cat_cardinalities):
            raise ValueError("each categorical feature needs at least 2 classes")

        self.n_continuous = n_continuous
        self.cat_cardinalities = [int(card) for card in cat_cardinalities]
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        input_dim = n_continuous + sum(self.cat_cardinalities)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # One linear head for every continuous column. No activation: the
        # targets are z-scored log1p features, which are unbounded in theory
        # and O(1) on normal rows.
        self.continuous_head = nn.Linear(hidden_dim, n_continuous)
        # One linear head per categorical feature. Outputs are raw logits.
        # Softmax is intentionally not applied; cross-entropy does that
        # internally, and applying it here would break the loss.
        self.categorical_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, card) for card in self.cat_cardinalities]
        )

    def encode(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map a batch to posterior parameters q(z | x) = N(mu, diag(exp(logvar))).

        One-hot blocks are built here, in categorical column order, class
        index 0..K-1. Scoring must call this method (or an equivalent that
        uses the same order) rather than assembling its own concatenation.
        """

        one_hots = [
            F.one_hot(categorical[:, index], num_classes=card).to(dtype=continuous.dtype)
            for index, card in enumerate(self.cat_cardinalities)
        ]
        encoded = torch.cat([continuous, *one_hots], dim=-1)
        hidden = self.encoder(encoded)
        mu = self.fc_mu(hidden)
        # Clamp before exp(logvar) in the KL term. A healthy train run stays
        # far inside this range; the clamp only stops a divergent step from
        # overflowing.
        logvar = self.fc_logvar(hidden).clamp(-10.0, 10.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + eps * sigma, eps ~ N(0, I). Standard Gaussian trick.

        Used on the training path so the KL term has a gradient through the
        sample. Scoring decodes mu directly so a transaction does not get a
        new score every time it is sent.
        """

        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def decode(self, latent: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Branched reconstruction. Continuous head, then one logit head each."""

        hidden = self.decoder(latent)
        recon_continuous = self.continuous_head(hidden)
        categorical_logits = [head(hidden) for head in self.categorical_heads]
        return recon_continuous, categorical_logits

    def forward(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Training forward: sample z, reconstruct, and return posterior params.

        Returns recon_continuous, categorical logit heads, mu, logvar.
        """

        mu, logvar = self.encode(continuous, categorical)
        latent = self.reparameterize(mu, logvar)
        recon_continuous, categorical_logits = self.decode(latent)
        return recon_continuous, categorical_logits, mu, logvar
