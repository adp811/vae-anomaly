"""Load and prepare the Kaggle Credit Card Fraud Detection table.

The public file is ``creditcard.csv`` from ``mlg-ulb/creditcardfraud``.
Columns are ``Time``, ``V1``–``V28``, ``Amount``, and ``Class``. ``V1``–``V28``
are anonymized principal components shipped with the dataset. This module
does not refit that PCA.

This path is normal-only / semi-supervised anomaly detection. ``Class`` is
used to keep fraud out of the training and calibration windows, and to
score the final holdout. It is not an argument to ``VAELoss``.

Splits follow time. After a stable sort on ``Time`` (row order breaks ties),
the first 60% of rows is the candidate training window, the next 20% is
calibration, and the last 20% is the mixed evaluation window. Fraud inside
the first two windows is dropped, not moved into evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from fraudvae.dataset import FeatureSchema, TransactionDataset

EXPECTED_COLUMNS: tuple[str, ...] = (
    "Time",
    *(f"V{index}" for index in range(1, 29)),
    "Amount",
    "Class",
)
FEATURE_COLUMNS: tuple[str, ...] = tuple(name for name in EXPECTED_COLUMNS if name != "Class")
# Amount is right-skewed and can be zero, so log1p is defined and useful.
# Time is a clock that starts at 0. V1–V28 are already signed PCA scores.
LOG1P_COLUMNS: tuple[str, ...] = ("Amount",)
PRIMARY_THRESHOLD_PERCENTILE = 99.0
SENSITIVITY_PERCENTILES: tuple[float, ...] = (95.0, 97.0, 99.0, 99.5)


def creditcard_schema() -> FeatureSchema:
    """Continuous-only schema. No categorical vocabularies, no dummy head."""

    return FeatureSchema(continuous_features=FEATURE_COLUMNS, categorical_vocabs=())


def validate_creditcard_frame(frame: pd.DataFrame) -> None:
    """Raise ValueError unless ``frame`` has the Kaggle schema."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("credit-card table must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("credit-card table is empty")
    if not frame.columns.is_unique:
        raise ValueError(f"duplicate column names: {list(frame.columns)}")
    missing = [name for name in EXPECTED_COLUMNS if name not in frame.columns]
    extra = [name for name in frame.columns if name not in EXPECTED_COLUMNS]
    if missing or extra:
        raise ValueError(
            "credit-card schema mismatch. "
            f"Missing {missing or '[]'}. Unexpected {extra or '[]'}. "
            f"Expected exactly {list(EXPECTED_COLUMNS)}."
        )
    values = frame.loc[:, list(EXPECTED_COLUMNS)]
    if values.isna().any().any():
        raise ValueError("credit-card table contains missing values")
    numeric = values.to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(numeric).all():
        raise ValueError("credit-card table contains non-finite values")
    classes = frame["Class"].to_numpy(dtype=np.float64, copy=False)
    if not np.isin(classes, (0.0, 1.0)).all() or not np.equal(classes, np.round(classes)).all():
        raise ValueError("Class must be 0 (normal) or 1 (fraud) on every row")


def load_creditcard_csv(path: str | Path) -> pd.DataFrame:
    """Read ``creditcard.csv`` and return columns in the canonical order."""

    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"credit-card CSV not found at {csv_path}. "
            "Download mlg-ulb/creditcardfraud and place creditcard.csv there. "
            "See the README. Do not commit the file or a Kaggle token."
        )
    frame = pd.read_csv(csv_path)
    validate_creditcard_frame(frame)
    ordered = frame.loc[:, list(EXPECTED_COLUMNS)].copy()
    ordered["Class"] = ordered["Class"].astype(np.int64)
    return ordered


@dataclass(frozen=True)
class CreditCardSplits:
    """Chronological windows after the normal-only filter.

    ``train_normal`` and ``calibration_normal`` contain ``Class=0`` only.
    ``evaluation`` is the last time window with both classes left in place.
    ``subsampled_train`` is true only when ``max_train_normals`` dropped
    training normals. Evaluation fraud is never capped.
    """

    train_normal: pd.DataFrame
    calibration_normal: pd.DataFrame
    evaluation: pd.DataFrame
    train_fraud_excluded: int
    calibration_fraud_excluded: int
    evaluation_fraud: int
    subsampled_train: bool
    max_train_normals: int | None
    n_ordered: int


def chronological_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.6,
    calibration_fraction: float = 0.2,
    max_train_normals: int | None = None,
    seed: int = 42,
) -> CreditCardSplits:
    """Cut the table by time, then drop fraud from train and calibration.

    The optional cap draws a seeded sample from the training-window normals
    and then restores time order. It does not touch calibration or evaluation.
    """

    if not 0.0 < train_fraction < 1.0 or not 0.0 < calibration_fraction < 1.0:
        raise ValueError("train and calibration fractions must be between 0 and 1")
    if train_fraction + calibration_fraction >= 1.0:
        raise ValueError("train_fraction + calibration_fraction must leave an evaluation window")
    if max_train_normals is not None and max_train_normals < 1:
        raise ValueError("max_train_normals must be positive when set")

    helpers = [column for column in ("label", "row_id") if column in frame.columns]
    raw = frame.drop(columns=helpers)
    validate_creditcard_frame(raw)
    raw = raw.loc[:, list(EXPECTED_COLUMNS)].copy()

    ordered = raw.sort_values("Time", kind="mergesort").reset_index(drop=True)
    ordered["row_id"] = np.arange(len(ordered), dtype=np.int64)
    ordered["label"] = ordered["Class"].astype(np.int64)

    n_rows = len(ordered)
    train_end = int(n_rows * train_fraction)
    calibration_end = int(n_rows * (train_fraction + calibration_fraction))
    if not (0 < train_end < calibration_end < n_rows):
        raise ValueError(
            "split fractions produced an empty window for a table of "
            f"{n_rows} rows"
        )

    train_window = ordered.iloc[:train_end]
    calibration_window = ordered.iloc[train_end:calibration_end]
    evaluation = ordered.iloc[calibration_end:].reset_index(drop=True)

    train_normal = train_window.loc[train_window["Class"] == 0].reset_index(drop=True)
    calibration_normal = calibration_window.loc[calibration_window["Class"] == 0].reset_index(drop=True)
    train_fraud_excluded = int((train_window["Class"] == 1).sum())
    calibration_fraud_excluded = int((calibration_window["Class"] == 1).sum())
    evaluation_fraud = int((evaluation["Class"] == 1).sum())

    if train_normal.empty or calibration_normal.empty:
        raise ValueError("a normal-only window is empty after dropping fraud")
    if (evaluation["Class"] == 0).sum() == 0 or evaluation_fraud == 0:
        raise ValueError(
            "evaluation window must contain both normal and fraud rows; "
            f"found {int((evaluation['Class'] == 0).sum())} normal and {evaluation_fraud} fraud"
        )

    subsampled = False
    if max_train_normals is not None and len(train_normal) > max_train_normals:
        train_normal = (
            train_normal.sample(n=max_train_normals, random_state=seed)
            .sort_values(["Time", "row_id"], kind="mergesort")
            .reset_index(drop=True)
        )
        subsampled = True

    return CreditCardSplits(
        train_normal=train_normal,
        calibration_normal=calibration_normal,
        evaluation=evaluation,
        train_fraud_excluded=train_fraud_excluded,
        calibration_fraud_excluded=calibration_fraud_excluded,
        evaluation_fraud=evaluation_fraud,
        subsampled_train=subsampled,
        max_train_normals=max_train_normals,
        n_ordered=n_rows,
    )


class CreditCardPreprocessor:
    """Standardize 30 continuous columns. ``log1p`` is applied to Amount only.

    Mean and population standard deviation are fit on the normal training
    rows actually passed to ``fit`` (after any ``MAX_TRAIN_NORMALS`` cap).
    ``V1``–``V28`` are already PCA components; standardizing them still puts
    every Huber term on a common scale. ``Time`` is standardized as a clock,
    not logged. ``require_positive`` is false so ``Time=0``, ``Amount=0``,
    and signed PCA scores are legal.
    """

    require_positive = False

    def __init__(
        self,
        columns: tuple[str, ...] = FEATURE_COLUMNS,
        log1p_columns: tuple[str, ...] = LOG1P_COLUMNS,
    ) -> None:
        unknown = [name for name in log1p_columns if name not in columns]
        if unknown:
            raise ValueError(f"log1p columns not in the feature list: {unknown}")
        self.columns = tuple(columns)
        self.log1p_columns = tuple(log1p_columns)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.n_fit_rows_: int | None = None

    def fit(self, frame: pd.DataFrame) -> CreditCardPreprocessor:
        transformed = self._apply_log1p(frame)
        self.mean_ = transformed.mean(axis=0)
        self.std_ = np.clip(transformed.std(axis=0), 1e-6, None)
        self.n_fit_rows_ = int(transformed.shape[0])
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("CreditCardPreprocessor.fit must be called before transform")
        transformed = self._apply_log1p(frame)
        scaled = (transformed - self.mean_) / self.std_
        return scaled.astype(np.float32)

    def describe(self) -> str:
        """Short definition of the map that was fit. Safe to print."""

        if self.mean_ is None or self.std_ is None or self.n_fit_rows_ is None:
            return "CreditCardPreprocessor is not fit"
        logged = ", ".join(self.log1p_columns) if self.log1p_columns else "(none)"
        plain = ", ".join(name for name in self.columns if name not in self.log1p_columns)
        return (
            f"fit on {self.n_fit_rows_} normal training rows\n"
            f"log1p then z-score: {logged}\n"
            f"z-score only: {plain}\n"
            "mean_ and std_ are population statistics (ddof=0) of that transformed matrix"
        )

    def _apply_log1p(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [name for name in self.columns if name not in frame.columns]
        if missing:
            raise ValueError(f"frame is missing feature columns: {missing}")
        values = frame.loc[:, list(self.columns)].to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(values).all():
            raise ValueError("features must be finite")
        for name in self.log1p_columns:
            index = self.columns.index(name)
            column = values[:, index]
            if (column < 0).any():
                raise ValueError(f"{name} must be >= 0 before log1p")
            values[:, index] = np.log1p(column)
        return values


def encode_mu(model, scaled: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Deterministic latent means. Does not draw epsilon."""

    import torch

    if scaled.ndim != 2:
        raise ValueError("scaled features must be a 2D array")
    device = next(model.parameters()).device
    model.eval()
    pieces: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(scaled), batch_size):
            batch = torch.as_tensor(scaled[start : start + batch_size], dtype=torch.float32, device=device)
            categorical = torch.zeros(batch.shape[0], 0, dtype=torch.long, device=device)
            mu, _logvar = model.encode(batch, categorical)
            pieces.append(mu.detach().cpu().numpy())
    if not pieces:
        return np.zeros((0, model.latent_dim), dtype=np.float32)
    return np.concatenate(pieces, axis=0)


def tensor_dataset(scaled: np.ndarray, labels: np.ndarray) -> TransactionDataset:
    """Continuous matrix plus a width-0 categorical tensor for ``TabularVAE``."""

    categorical = np.zeros((scaled.shape[0], 0), dtype=np.int64)
    return TransactionDataset(scaled, categorical, np.asarray(labels, dtype=np.int64))


def detection_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    """Holdout metrics for one frozen threshold.

    Scores are ranked so that larger means more anomalous. Precision, recall,
    and F1 use fraud as the positive class. PR-AUC and ROC-AUC do not use
    the threshold. Accuracy is returned only so a caller can show why it is
    a poor summary under this base rate.
    """

    truth = np.asarray(labels, dtype=int)
    ranking = np.asarray(scores, dtype=float)
    if truth.shape[0] != ranking.shape[0]:
        raise ValueError("labels and scores must have the same length")
    if len(truth) == 0:
        raise ValueError("metrics require at least one row")
    if len(np.unique(truth)) < 2:
        raise ValueError("metrics require both classes in the evaluation window")

    predicted = (ranking >= threshold).astype(int)
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    precision, recall, f1, _support = precision_recall_fscore_support(
        truth,
        predicted,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    support_normal = int((truth == 0).sum())
    support_fraud = int((truth == 1).sum())
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "pr_auc": float(average_precision_score(truth, ranking)),
        "roc_auc": float(roc_auc_score(truth, ranking)),
        "normal_fpr": float(fp / support_normal) if support_normal else float("nan"),
        "flagged_fraction": float(predicted.mean()),
        "support_normal": support_normal,
        "support_fraud": support_fraud,
        "accuracy": float((predicted == truth).mean()),
    }


def threshold_sensitivity(
    calibration_scores: np.ndarray,
    evaluation_labels: np.ndarray,
    evaluation_scores: np.ndarray,
    percentiles: tuple[float, ...] = SENSITIVITY_PERCENTILES,
) -> pd.DataFrame:
    """Metrics at percentiles of the normal calibration scores.

    Every threshold is an order statistic of ``calibration_scores``. Evaluation
    labels are used only after the threshold exists.
    """

    calibration = np.asarray(calibration_scores, dtype=float)
    rows = []
    for percentile in percentiles:
        if not 0.0 < percentile < 100.0:
            raise ValueError(f"percentile out of range: {percentile}")
        threshold = float(np.percentile(calibration, percentile))
        metrics = detection_metrics(evaluation_labels, evaluation_scores, threshold)
        rows.append({"percentile": percentile, **metrics})
    return pd.DataFrame(rows)


def fit_isolation_forest(
    train_scaled: np.ndarray,
    seed: int,
    n_estimators: int = 100,
) -> IsolationForest:
    """Lightweight forest on the same scaled normal training matrix as the VAE."""

    if train_scaled.ndim != 2 or train_scaled.shape[0] < 2:
        raise ValueError("IsolationForest needs a 2D normal training matrix")
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples="auto",
        contamination="auto",
        random_state=seed,
        n_jobs=1,
    )
    model.fit(train_scaled)
    return model


def isolation_anomaly_scores(model: IsolationForest, scaled: np.ndarray) -> np.ndarray:
    """Higher means more anomalous. sklearn's decision function points the other way."""

    return -model.decision_function(scaled)


def fit_visualization_pca(
    normal_train_mu: np.ndarray,
    max_rows: int = 8000,
    seed: int = 42,
    n_components: int = 2,
) -> PCA:
    """2D PCA fit on normal-training latent means only.

    This is a plot projection. It is not the dataset's ``V1``–``V28``, and it
    is not part of the anomaly score. ``max_rows`` caps the fit sample.
    """

    if normal_train_mu.ndim != 2:
        raise ValueError("latent means must be a 2D array")
    if normal_train_mu.shape[0] < n_components:
        raise ValueError("not enough normal latent rows to fit visualization PCA")
    if max_rows < n_components:
        raise ValueError("max_rows is smaller than n_components")
    sample = normal_train_mu
    if len(sample) > max_rows:
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(len(sample), size=max_rows, replace=False))
        sample = sample[chosen]
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(sample)
    return pca


def format_metrics(name: str, metrics: dict[str, float | int]) -> str:
    """One block of holdout numbers. Accuracy is printed last, with the caveat nearby."""

    return "\n".join(
        [
            f"{name}",
            f"threshold={metrics['threshold']:.4f}",
            f"precision={metrics['precision']:.4f}  recall={metrics['recall']:.4f}  f1={metrics['f1']:.4f}",
            (
                f"confusion tn={metrics['tn']} fp={metrics['fp']} "
                f"fn={metrics['fn']} tp={metrics['tp']}"
            ),
            f"pr_auc={metrics['pr_auc']:.4f}  roc_auc={metrics['roc_auc']:.4f}",
            (
                f"normal_fpr={metrics['normal_fpr']:.4f}  "
                f"flagged_fraction={metrics['flagged_fraction']:.4f}"
            ),
            f"support normal={metrics['support_normal']}  fraud={metrics['support_fraud']}",
            f"accuracy={metrics['accuracy']:.4f}  (misleading at this base rate)",
        ]
    )
