"""Synthetic mixed-type transactions and the tensors the VAE trains on.

Generative assumptions (normal class)
--------------------------------------
Normal transactions lie near a three-factor manifold. Every continuous
column is positive and right-skewed: it is exp(gaussian) in log space.

1. Spend factor. ``amount`` is lognormal. ``fee`` tracks it tightly
   (log fee ≈ log amount + constant + small noise), the way a card fee
   tracks ticket size. Merchant id is a soft function of spend: high spend
   prefers premium merchants (m0, m1), low spend prefers everyday
   merchants, and m8/m9 stay rare.
2. Recency factor. ``seconds_since_last_txn`` is lognormal. ``rolling_spend_24h``
   moves with amount and falls as the gap grows: a long silence means less
   spend has piled up in the last day.
3. Distance factor. ``distance_from_home_km`` is lognormal and small.
   Device type follows it. Order is chip, contactless, online. Chip and
   contactless are local; online takes over as distance grows.

Generative assumptions (fraud class)
-------------------------------------
Fraud is a minority (~5% at the default sizes) and is not a label the
model trains on. Each fraud row is shifted off the manifold: larger
amount, a fee that does not grow with that amount, a long gap together
with a high rolling 24h spend, a far distance still presented as chip,
and a high amount at a rare merchant (m7–m9). Those breaks are a property
of this generator so reconstruction error can beat chance. They are not
something a production scorer would know in advance.

Leakage
-------
Callers fit the preprocessor and the VAE on the normal training split only.
The 95th-percentile threshold comes from a disjoint pure-normal calibration
split. Precision and recall are computed on a third, mixed validation split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Names and vocabs are the feature schema. Order is the one-hot contract:
# continuous columns are concatenated in this order, then one-hot blocks in
# CATEGORICAL_FEATURES order, and class index i is vocab position i.
CONTINUOUS_FEATURES: tuple[str, ...] = (
    "amount",
    "fee",
    "rolling_spend_24h",
    "distance_from_home_km",
    "seconds_since_last_txn",
)
CATEGORICAL_FEATURES: tuple[str, ...] = ("merchant_id", "device_type")
# m0..m9 so the serve example "m3" is a real class, not m03.
MERCHANT_VOCAB: tuple[str, ...] = tuple(f"m{i}" for i in range(10))
# One-hot order. Index 0 is chip, 1 is contactless, 2 is online.
DEVICE_VOCAB: tuple[str, ...] = ("chip", "contactless", "online")


@dataclass(frozen=True)
class FeatureSchema:
    """Frozen description of the columns the network was built for.

    Persisting this next to the state_dict is what keeps train-time one-hot
    order identical to serve-time one-hot order. Vocab tuples are ordered;
    position is the class index.
    """

    continuous_features: tuple[str, ...]
    categorical_vocabs: tuple[tuple[str, tuple[str, ...]], ...]

    @property
    def categorical_names(self) -> tuple[str, ...]:
        return tuple(name for name, _vocab in self.categorical_vocabs)

    @property
    def cardinalities(self) -> tuple[int, ...]:
        return tuple(len(vocab) for _name, vocab in self.categorical_vocabs)

    def vocab(self, name: str) -> tuple[str, ...]:
        for feature, vocab in self.categorical_vocabs:
            if feature == name:
                return vocab
        raise KeyError(f"unknown categorical feature {name!r}")

    def class_index(self, name: str, value: str) -> int:
        vocab = self.vocab(name)
        try:
            return vocab.index(value)
        except ValueError as exc:
            raise ValueError(
                f"{name} value {value!r} is not in the training vocabulary {vocab}. "
                "Unseen categories are refused rather than silently mapped, "
                "because a made-up one-hot slot would not match the trained heads."
            ) from exc


DEFAULT_SCHEMA = FeatureSchema(
    continuous_features=CONTINUOUS_FEATURES,
    categorical_vocabs=(
        ("merchant_id", MERCHANT_VOCAB),
        ("device_type", DEVICE_VOCAB),
    ),
)


class ContinuousPreprocessor:
    """log1p, then z-score, fit on normal training rows only.

    Raw amounts and the other continuous columns are right-skewed. Huber
    loss with delta=1 would then be dominated by a handful of legitimately
    large transactions, and fraud-sized tails would explode the gradient.
    log1p compresses that tail. Z-scoring with the training mean and std
    puts the Huber kink at one standard deviation of normal log-traffic,
    which is the scale the rest of the model assumes.

    The stored mean_ and std_ are serve-time artifacts. They are not
    recomputed per request, and they are not fit on fraud or on the
    validation rows.
    """

    def __init__(self, columns: tuple[str, ...] = CONTINUOUS_FEATURES) -> None:
        self.columns = columns
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> ContinuousPreprocessor:
        logged = self._log1p(frame)
        # Population std (ddof=0). With thousands of rows the difference
        # from the sample std is negligible and the divisor is explicit.
        self.mean_ = logged.mean(axis=0)
        self.std_ = np.clip(logged.std(axis=0), 1e-6, None)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("ContinuousPreprocessor.fit must be called before transform")
        logged = self._log1p(frame)
        scaled = (logged - self.mean_) / self.std_
        return scaled.astype(np.float32)

    def _log1p(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.loc[:, list(self.columns)].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("continuous features must be finite before log1p scaling")
        # Non-positive inputs are rejected. log1p(0) is defined, but a zero
        # amount or fee is not a valid transaction in this schema, and a
        # silent clamp would disagree with the serve contract.
        if (values <= 0).any():
            raise ValueError(
                "continuous features must be finite and > 0; non-positive inputs are rejected"
            )
        return np.log1p(values)


class TransactionDataset(Dataset):
    """One transaction per row: scaled continuous features, class indices, label.

    The label is the hidden fraud flag (0 normal, 1 fraud). It is yielded so
    callers can audit splits. The training loop must not pass it into the
    model or the loss.
    """

    def __init__(
        self,
        continuous: np.ndarray,
        categorical: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        if continuous.shape[0] != categorical.shape[0] or continuous.shape[0] != labels.shape[0]:
            raise ValueError("continuous, categorical, and labels must have the same number of rows")
        # Copy so a read-only NumPy view (pandas) does not become a tensor
        # PyTorch refuses to write through.
        self.continuous = torch.tensor(np.array(continuous, dtype=np.float32, copy=True))
        self.categorical = torch.tensor(np.array(categorical, dtype=np.int64, copy=True))
        self.labels = torch.tensor(np.array(labels, dtype=np.int64, copy=True))

    def __len__(self) -> int:
        return int(self.continuous.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.continuous[index], self.categorical[index], self.labels[index]


class TransactionDataModule:
    """Build the three disjoint splits and the loaders the notebook trains with.

    - train loader: normal rows only. This is the only data the VAE fits.
    - calibration frame: pure normal, held out from fitting, used once to
      set the 95th-percentile reconstruction threshold.
    - validation loader: held-out normals plus every fraud row. Metrics are
      computed here and nowhere else.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        n_train: int,
        n_calibration: int,
        n_val_normal: int,
        batch_size: int,
        seed: int,
        schema: FeatureSchema = DEFAULT_SCHEMA,
    ) -> None:
        self.schema = schema
        self.batch_size = batch_size
        self.seed = seed
        self.n_train = n_train
        self.n_calibration = n_calibration
        self.n_val_normal = n_val_normal

        normal = frame.loc[frame["label"] == 0].copy()
        fraud = frame.loc[frame["label"] == 1].copy()
        expected_normal = n_train + n_calibration + n_val_normal
        if len(normal) != expected_normal:
            raise ValueError(
                f"expected {expected_normal} normal rows "
                f"(train {n_train} + calibration {n_calibration} + val {n_val_normal}), "
                f"found {len(normal)}"
            )
        if fraud.empty:
            raise ValueError("frame has no fraud rows; validation metrics would be undefined")
        if "row_id" not in frame.columns:
            raise ValueError("frame is missing row_id; generate it with generate_synthetic_transactions")

        # pandas samples with its own RandomState(seed), independent of the
        # global NumPy seed, so the split stays fixed if another library
        # draws from the global generator first.
        shuffled = normal.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        self.train_frame = shuffled.iloc[:n_train].reset_index(drop=True)
        self.calibration_frame = shuffled.iloc[n_train : n_train + n_calibration].reset_index(drop=True)
        val_normal = shuffled.iloc[n_train + n_calibration :].reset_index(drop=True)
        self.validation_frame = (
            pd.concat([val_normal, fraud], ignore_index=True)
            .sample(frac=1.0, random_state=seed)
            .reset_index(drop=True)
        )
        self._assert_splits()

        # Fit scaling on training normals only, then apply that same map to
        # calibration and validation. Refitting on either would leak.
        self.preprocessor = ContinuousPreprocessor(schema.continuous_features).fit(self.train_frame)
        self.train_dataset = self._make_dataset(self.train_frame)
        self.calibration_dataset = self._make_dataset(self.calibration_frame)
        self.validation_dataset = self._make_dataset(self.validation_frame)

        generator = torch.Generator()
        generator.manual_seed(seed)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        self.calibration_loader = DataLoader(
            self.calibration_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        self.validation_loader = DataLoader(
            self.validation_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

    def summary(self) -> str:
        def describe(name: str, frame: pd.DataFrame) -> str:
            n_fraud = int((frame["label"] == 1).sum())
            n_normal = len(frame) - n_fraud
            return f"{name}: {len(frame)} rows ({n_normal} normal, {n_fraud} fraud)"

        return "\n".join(
            [
                describe("train", self.train_frame),
                describe("calibration", self.calibration_frame),
                describe("validation", self.validation_frame),
            ]
        )

    def _make_dataset(self, frame: pd.DataFrame) -> TransactionDataset:
        continuous = self.preprocessor.transform(frame)
        categorical = encode_categoricals(frame, self.schema)
        labels = frame["label"].to_numpy(dtype=np.int64)
        return TransactionDataset(continuous, categorical, labels)

    def _assert_splits(self) -> None:
        if (self.train_frame["label"] != 0).any():
            raise RuntimeError("training split contains fraud rows")
        if (self.calibration_frame["label"] != 0).any():
            raise RuntimeError("calibration split contains fraud rows")
        if (self.validation_frame["label"] == 1).sum() == 0:
            raise RuntimeError("validation split has no fraud rows")
        if (self.validation_frame["label"] == 0).sum() == 0:
            raise RuntimeError("validation split has no normal rows")
        train_ids = set(self.train_frame["row_id"].tolist())
        calibration_ids = set(self.calibration_frame["row_id"].tolist())
        validation_ids = set(self.validation_frame["row_id"].tolist())
        if train_ids & calibration_ids or train_ids & validation_ids or calibration_ids & validation_ids:
            raise RuntimeError("split leakage: a row_id appears in more than one split")


def encode_categoricals(frame: pd.DataFrame, schema: FeatureSchema = DEFAULT_SCHEMA) -> np.ndarray:
    """Map categorical columns to class indices in schema order.

    Index 0 is the first vocab entry. The encoder one-hots these indices
    with that same order, so a swap here silently trains a different model
    than the one the scorer expects.
    """

    columns: list[np.ndarray] = []
    for name in schema.categorical_names:
        vocab = schema.vocab(name)
        lookup = {token: index for index, token in enumerate(vocab)}
        encoded = []
        for value in frame[name].tolist():
            try:
                encoded.append(lookup[value])
            except KeyError as exc:
                raise ValueError(
                    f"{name} value {value!r} is not in the training vocabulary {vocab}"
                ) from exc
        columns.append(np.asarray(encoded, dtype=np.int64))
    return np.column_stack(columns)


def generate_synthetic_transactions(
    n_normal: int = 7600,
    n_fraud: int = 400,
    seed: int = 42,
) -> pd.DataFrame:
    """Draw a mixed table of normal and fraud transactions.

    The returned frame includes a hidden ``label`` column (0 normal, 1 fraud)
    and a ``row_id`` used to prove the splits are disjoint. The default
    400 fraud rows out of 8,000 is 5%.

    The NumPy Generator is seeded locally so the table does not depend on
    whatever else has drawn from the global NumPy RNG.
    """

    if n_normal <= 0 or n_fraud <= 0:
        raise ValueError("n_normal and n_fraud must both be positive")
    rng = np.random.default_rng(seed)
    normal = _sample_normal(n_normal, rng)
    fraud = _sample_fraud(n_fraud, rng)
    frame = pd.concat([normal, fraud], ignore_index=True)
    frame.insert(0, "row_id", np.arange(len(frame), dtype=np.int64))
    return frame


def _sample_normal(n: int, rng: np.random.Generator) -> pd.DataFrame:
    log_amount = rng.normal(3.2, 0.50, n)
    # Fee rides on amount. The tight noise is the link fraud breaks.
    log_fee = log_amount - 3.9 + rng.normal(0.0, 0.12, n)
    log_seconds = rng.normal(8.8, 0.50, n)
    log_distance = rng.normal(1.1, 0.50, n)
    # Longer gaps line up with a smaller rolling 24h spend.
    log_rolling = log_amount + rng.normal(0.85, 0.18, n) - 0.40 * (log_seconds - 8.8)

    merchant = _sample_categorical(_softmax(_merchant_logits(log_amount)), rng)
    device = _sample_categorical(_softmax(_device_logits(log_distance)), rng)
    return _frame_from_parts(
        log_amount, log_fee, log_rolling, log_distance, log_seconds, merchant, device, label=0
    )


def _sample_fraud(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Draw rows that are jointly unlikely under the normal process.

    Margins are shifted, and the normal correlations are reversed rather
    than merely resampled. Larger amounts and rarer merchant/device pairs
    are deliberate so a reconstruction model fit only on normals can beat
    chance. This is a property of the toy generator, not a real fraud rule.
    """

    log_amount = rng.normal(5.6, 0.28, n)
    # Fee stays small instead of scaling with the large amount.
    log_fee = rng.normal(0.3, 0.15, n)
    # Long silence, but rolling 24h spend is still high. Normals cannot do both.
    log_seconds = rng.normal(11.2, 0.22, n)
    log_rolling = rng.normal(6.8, 0.22, n)
    log_distance = rng.normal(4.4, 0.22, n)

    # Rare tail of the merchant vocab (m7–m9), not the premium ids a high
    # amount would take under the normal softmax.
    merchant = rng.integers(7, 10, size=n)
    # Chip is the local device. Fraud is far from home and still uses it.
    # A thin contactless share keeps the column from being a single constant.
    device = np.where(rng.random(n) < 0.85, 0, 1).astype(np.int64)
    return _frame_from_parts(
        log_amount, log_fee, log_rolling, log_distance, log_seconds, merchant, device, label=1
    )


def _merchant_logits(log_amount: np.ndarray) -> np.ndarray:
    """Linear softmax scores for P(merchant | log amount).

    Slopes are positive for premium merchants (m0, m1) and negative for
    everyday and rare ones (m5–m9), so the same spend factor that sets the
    amount also picks the merchant. That correlation is what fraud violates.
    """

    centered = (log_amount - 3.2)[:, None]
    slope = np.array([2.4, 2.4, 0.3, 0.3, 0.3, -1.6, -1.6, -1.8, -2.6, -2.6])
    bias = np.array([-0.4, -0.4, 0.55, 0.55, 0.55, 0.65, 0.65, 0.2, -1.5, -1.5])
    return centered * slope + bias


def _device_logits(log_distance: np.ndarray) -> np.ndarray:
    """Linear softmax scores for P(device | log distance).

    Class order is chip, contactless, online. Chip's slope is steeply
    negative, so a far-from-home chip transaction is rare here.
    """

    centered = (log_distance - 1.1)[:, None]
    slope = np.array([-1.6, -0.3, 1.9])
    bias = np.array([0.6, 1.15, -0.35])
    return centered * slope + bias


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def _sample_categorical(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample one class per row from an (n, k) probability table."""

    cdf = np.cumsum(probs, axis=1)
    cdf[:, -1] = 1.0
    draws = rng.random(probs.shape[0])
    return (draws[:, None] >= cdf).sum(axis=1).astype(np.int64)


def _frame_from_parts(
    log_amount: np.ndarray,
    log_fee: np.ndarray,
    log_rolling: np.ndarray,
    log_distance: np.ndarray,
    log_seconds: np.ndarray,
    merchant: np.ndarray,
    device: np.ndarray,
    label: int,
) -> pd.DataFrame:
    merchant_tokens = np.asarray(MERCHANT_VOCAB, dtype=object)[merchant]
    device_tokens = np.asarray(DEVICE_VOCAB, dtype=object)[device]
    return pd.DataFrame(
        {
            "amount": np.exp(log_amount),
            "fee": np.exp(log_fee),
            "rolling_spend_24h": np.exp(log_rolling),
            "distance_from_home_km": np.exp(log_distance),
            "seconds_since_last_txn": np.exp(log_seconds),
            "merchant_id": merchant_tokens,
            "device_type": device_tokens,
            "label": np.full(log_amount.shape[0], label, dtype=np.int64),
        }
    )
