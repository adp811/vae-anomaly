"""Reconstruction-error scoring with a frozen normal-only threshold.

This is the function surface a thin HTTP service would call. Training does
not happen here. KL is not part of the score. The threshold is an attribute
set from a pure-normal calibration set; request handling only reads it.

Online contract (one transaction)
---------------------------------
score(transaction) -> {
    reconstruction_error, threshold, high_risk,
    continuous_error,
    categorical_error: {merchant_id, device_type}
}

``high_risk`` is true when the reconstruction error is at or above the
calibrated threshold. Those rows are the "High-Risk Anomaly" flags.

``score_batch`` accepts a list and returns a list by calling the same
path as ``score``. It does not refit the threshold. ``score_frame`` is the
dataframe form of that same reconstruction, used by the notebook to score
the validation table without building one dict per row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report

from fraudvae.dataset import (
    ContinuousPreprocessor,
    FeatureSchema,
    encode_categoricals,
)
from fraudvae.loss import LossConfig
from fraudvae.model import TabularVAE


class ReconstructionScorer:
    """Score transactions with a trained TabularVAE and a frozen threshold."""

    def __init__(
        self,
        model: TabularVAE,
        preprocessor: ContinuousPreprocessor,
        loss_config: LossConfig,
        schema: FeatureSchema,
        threshold: float | None = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.preprocessor = preprocessor
        self.loss_config = loss_config
        self.schema = schema
        self.threshold = threshold
        self.threshold_percentile: float | None = None

    def fit_threshold(self, frame: pd.DataFrame, percentile: float = 95.0) -> float:
        """Set the threshold to a percentile of reconstruction error.

        ``frame`` must be pure normal and must not be the training rows or
        the rows that will be reported as metrics. The default 95th
        percentile flags the noisiest 5% of normal calibration traffic.
        The value is stored on this object and reused for every later call.
        """

        if not 0.0 < percentile < 100.0:
            raise ValueError("percentile must be between 0 and 100")
        if "label" in frame.columns and (frame["label"] != 0).any():
            raise ValueError(
                "refusing to fit the threshold on a frame that contains fraud rows"
            )
        # A few hundred normal rows is the floor for a 95th percentile that
        # is not just one noisy order statistic. Below that, refuse to publish.
        if len(frame) < 200:
            raise ValueError(
                "calibration set is too small for a stable percentile threshold "
                f"({len(frame)} rows; need at least 200 normal rows)"
            )
        scores = self.reconstruction_errors(frame)["reconstruction_error"].to_numpy()
        self.threshold = float(np.percentile(scores, percentile))
        self.threshold_percentile = float(percentile)
        return self.threshold

    def score(self, transaction: dict) -> dict:
        """Score one transaction dict. This is the request-path function.

        Required keys: amount, fee, rolling_spend_24h, distance_from_home_km,
        seconds_since_last_txn, merchant_id, device_type. merchant_id is a
        vocab token such as ``m3``. device_type is ``chip``, ``contactless``,
        or ``online``. Missing fields, non-positive continuous values, and
        unseen categories raise ValueError. They are not coerced.
        """

        return self.score_batch([transaction])[0]

    def score_batch(self, transactions: list[dict]) -> list[dict]:
        """Score a list of transactions and return a list of result dicts.

        Same reconstruction as ``score``. An empty list returns an empty
        list and does not touch the threshold.
        """

        if not isinstance(transactions, list):
            raise TypeError("score_batch expects a list of transaction dicts")
        if len(transactions) == 0:
            return []
        require_positive = bool(getattr(self.preprocessor, "require_positive", True))
        frame = pd.concat(
            [
                _transaction_frame(item, self.schema, require_positive=require_positive)
                for item in transactions
            ],
            ignore_index=True,
        )
        scored = self.score_frame(frame)
        return [self._result_from_row(scored.iloc[index]) for index in range(len(scored))]

    def _result_from_row(self, row: pd.Series) -> dict:
        return {
            "reconstruction_error": float(row["reconstruction_error"]),
            "threshold": float(self.threshold),
            "high_risk": bool(row["high_risk"]),
            "continuous_error": float(row["continuous_error"]),
            "categorical_error": {
                name: float(row[f"categorical_error_{name}"])
                for name in self.schema.categorical_names
            },
        }

    def score_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Score a dataframe with the same error ``score`` uses.

        Does not refit the threshold. Rows at or above the threshold are
        flagged ``High-Risk Anomaly``.
        """

        self._require_threshold()
        scored = self.reconstruction_errors(frame)
        scored["threshold"] = float(self.threshold)
        scored["high_risk"] = scored["reconstruction_error"] >= float(self.threshold)
        scored["flag"] = np.where(scored["high_risk"], "High-Risk Anomaly", "Within threshold")
        return scored

    def reconstruction_errors(self, frame: pd.DataFrame, batch_size: int = 1024) -> pd.DataFrame:
        """Per-row Huber sum + categorical cross-entropy sum. No KL.

        Decoding uses z = mu rather than a reparameterized sample, so the
        same row always produces the same error. Rows are chunked only to
        keep a backfill from holding the whole table as one tensor; the
        math does not depend on the chunk size.
        """

        self.model.eval()
        continuous = self.preprocessor.transform(frame)
        categorical = encode_categoricals(frame, self.schema)
        pieces: list[pd.DataFrame] = []
        for start in range(0, len(frame), batch_size):
            stop = min(start + batch_size, len(frame))
            pieces.append(
                self._score_numpy(continuous[start:stop], categorical[start:stop])
            )
        scored = pd.concat(pieces, ignore_index=True)
        scored.index = frame.index
        return scored

    def _require_threshold(self) -> None:
        if self.threshold is None:
            raise RuntimeError(
                "anomaly threshold is not set; call fit_threshold on a pure-normal "
                "calibration set before scoring"
            )

    @torch.no_grad()
    def _score_numpy(self, continuous: np.ndarray, categorical: np.ndarray) -> pd.DataFrame:
        continuous_tensor = torch.as_tensor(continuous, dtype=torch.float32, device=self.device)
        categorical_tensor = torch.as_tensor(categorical, dtype=torch.long, device=self.device)
        mu, _logvar = self.model.encode(continuous_tensor, categorical_tensor)
        recon_continuous, categorical_logits = self.model.decode(mu)

        # Sum within the row. Training logs the mean reduction; the threshold
        # lives in this per-row sum, weighted by the same lambdas.
        continuous_error = F.huber_loss(
            recon_continuous,
            continuous_tensor,
            delta=self.loss_config.huber_delta,
            reduction="none",
        ).sum(dim=-1)
        if categorical_logits:
            per_head = [
                F.cross_entropy(logits, categorical_tensor[:, index], reduction="none")
                for index, logits in enumerate(categorical_logits)
            ]
            categorical_matrix = torch.stack(per_head, dim=-1)
            categorical_error = categorical_matrix.sum(dim=-1)
        else:
            categorical_error = torch.zeros(continuous_tensor.shape[0], device=self.device)
            categorical_matrix = categorical_error.new_zeros((continuous_tensor.shape[0], 0))
        reconstruction_error = (
            self.loss_config.lambda_cont * continuous_error
            + self.loss_config.lambda_cat * categorical_error
        )

        data = {
            "reconstruction_error": reconstruction_error.detach().cpu().numpy(),
            "continuous_error": continuous_error.detach().cpu().numpy(),
            "categorical_error": categorical_error.detach().cpu().numpy(),
        }
        per_head_numpy = categorical_matrix.detach().cpu().numpy()
        for index, name in enumerate(self.schema.categorical_names):
            data[f"categorical_error_{name}"] = per_head_numpy[:, index]
        return pd.DataFrame(data)

    def continuous_feature_huber(self, frame: pd.DataFrame, batch_size: int = 1024) -> pd.DataFrame:
        """Per-feature Huber terms that sum to ``continuous_error``.

        Same ``decode(mu)`` path as the score. Columns follow the schema's
        continuous order. ``lambda_cont`` is not applied here; with the
        default weight of 1 these terms are the continuous contribution.
        """

        self.model.eval()
        continuous = self.preprocessor.transform(frame)
        categorical = encode_categoricals(frame, self.schema)
        names = list(self.schema.continuous_features)
        pieces: list[pd.DataFrame] = []
        for start in range(0, len(frame), batch_size):
            stop = min(start + batch_size, len(frame))
            pieces.append(self._feature_huber_numpy(continuous[start:stop], categorical[start:stop], names))
        scored = pd.concat(pieces, ignore_index=True)
        scored.index = frame.index
        return scored

    @torch.no_grad()
    def _feature_huber_numpy(
        self,
        continuous: np.ndarray,
        categorical: np.ndarray,
        names: list[str],
    ) -> pd.DataFrame:
        continuous_tensor = torch.as_tensor(continuous, dtype=torch.float32, device=self.device)
        categorical_tensor = torch.as_tensor(categorical, dtype=torch.long, device=self.device)
        mu, _logvar = self.model.encode(continuous_tensor, categorical_tensor)
        recon_continuous, _logits = self.model.decode(mu)
        per_feature = F.huber_loss(
            recon_continuous,
            continuous_tensor,
            delta=self.loss_config.huber_delta,
            reduction="none",
        )
        return pd.DataFrame(per_feature.detach().cpu().numpy(), columns=names)


def detection_report(labels: np.ndarray, high_risk: np.ndarray) -> str:
    """Precision, recall, and F1 against the hidden fraud label.

    The positive class is fraud (label 1). This is an offline report, not a
    claim that the threshold is a production operating point.
    """

    predicted = np.asarray(high_risk, dtype=int)
    truth = np.asarray(labels, dtype=int)
    return classification_report(
        truth,
        predicted,
        labels=[0, 1],
        target_names=["Normal", "Fraud"],
        digits=3,
        zero_division=0,
    )


def format_score_stats(name: str, scores: np.ndarray) -> str:
    """One-line distribution summary. Used instead of a plotting dependency."""

    if len(scores) == 0:
        return f"{name}: no rows"
    percentiles = np.percentile(scores, [5, 50, 95])
    return (
        f"{name}: n={len(scores)}  mean={scores.mean():.3f}  std={scores.std():.3f}  "
        f"p05={percentiles[0]:.3f}  p50={percentiles[1]:.3f}  p95={percentiles[2]:.3f}  "
        f"min={scores.min():.3f}  max={scores.max():.3f}"
    )


def text_histogram(scores: np.ndarray, bins: int = 12, width: int = 40) -> str:
    """Fixed-width histogram so the notebook can show the score distribution."""

    if len(scores) == 0:
        return "(empty)"
    counts, edges = np.histogram(scores, bins=bins)
    peak = max(int(counts.max()), 1)
    lines = []
    for count, left, right in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(round(width * (count / peak)))
        lines.append(f"{left:7.2f} – {right:7.2f} | {bar:<{width}} {int(count)}")
    return "\n".join(lines)


def _transaction_frame(
    transaction: dict,
    schema: FeatureSchema,
    require_positive: bool,
) -> pd.DataFrame:
    required = (*schema.continuous_features, *schema.categorical_names)
    missing = [name for name in required if name not in transaction]
    if missing:
        raise ValueError(f"transaction is missing required fields: {missing}")
    row: dict[str, object] = {}
    for name in schema.continuous_features:
        value = transaction[name]
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
        if not np.isfinite(numeric) or (require_positive and numeric <= 0):
            expectation = "a finite number > 0" if require_positive else "a finite number"
            raise ValueError(f"{name} must be {expectation}, got {value!r}")
        row[name] = numeric
    for name in schema.categorical_names:
        row[name] = transaction[name]
    return pd.DataFrame([row])
