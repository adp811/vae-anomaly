"""Zero-shot fraud labeling with a tabular variational autoencoder.

The package is split the way a later service would be split:

- dataset: synthetic transactions, the normal-only / calibration / validation
  cut, and the log1p + z-score preprocessor
- model: TabularVAE, shared by training and scoring
- loss: Huber + categorical cross-entropy + KL
- train: offline loop, not part of a request path
- score: reconstruction-error scoring and the frozen-threshold decision

Training never sees fraud labels. A row is flagged when its reconstruction
error exceeds a threshold fit on a pure-normal calibration set.
"""

from fraudvae.dataset import (
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    DEVICE_VOCAB,
    MERCHANT_VOCAB,
    ContinuousPreprocessor,
    FeatureSchema,
    TransactionDataModule,
    TransactionDataset,
    generate_synthetic_transactions,
)
from fraudvae.loss import LossConfig, VAELoss
from fraudvae.model import TabularVAE
from fraudvae.score import ReconstructionScorer, detection_report
from fraudvae.train import VAETrainer, set_seed

__all__ = [
    "CATEGORICAL_FEATURES",
    "CONTINUOUS_FEATURES",
    "DEVICE_VOCAB",
    "MERCHANT_VOCAB",
    "ContinuousPreprocessor",
    "FeatureSchema",
    "LossConfig",
    "ReconstructionScorer",
    "TabularVAE",
    "TransactionDataModule",
    "TransactionDataset",
    "VAELoss",
    "VAETrainer",
    "detection_report",
    "generate_synthetic_transactions",
    "set_seed",
]
