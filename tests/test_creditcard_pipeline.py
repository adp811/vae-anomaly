"""Chronological Kaggle splits, continuous-only scoring, and the mixed schema."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from fraudvae.dataset import (
    DEFAULT_SCHEMA,
    ContinuousPreprocessor,
    TransactionDataModule,
    encode_categoricals,
    generate_synthetic_transactions,
)
from fraudvae.loss import LossConfig, VAELoss
from fraudvae.model import TabularVAE
from fraudvae.real_data import (
    FEATURE_COLUMNS,
    CreditCardPreprocessor,
    chronological_split,
    creditcard_schema,
    detection_metrics,
    encode_mu,
    fit_isolation_forest,
    fit_visualization_pca,
    isolation_anomaly_scores,
    tensor_dataset,
    threshold_sensitivity,
    validate_creditcard_frame,
)
from fraudvae.score import ReconstructionScorer
from fraudvae.train import VAETrainer, set_seed


def _frame(n: int = 1200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shuffled_time = rng.permutation(n).astype(np.float64)
    data = {
        "Time": shuffled_time,
        "Amount": rng.uniform(0.0, 50.0, n),
        "Class": np.zeros(n, dtype=np.int64),
    }
    for index in range(1, 29):
        data[f"V{index}"] = rng.normal(loc=0.0, scale=1.0, size=n)
    # Time value 1 sits in the training window and is not one of the planted fraud times.
    zero_amount_at = int(np.where(data["Time"] == 1.0)[0][0])
    data["Amount"][zero_amount_at] = 0.0
    negative_at = int(np.where(data["Time"] == 2.0)[0][0])
    data["V1"][negative_at] = -2.5
    # One fraud row in each window of a 60/20/20 cut, plus a second in evaluation.
    for fraction in (0.10, 0.70, 0.90, 0.95):
        row = int(n * fraction)
        # Fraud is planted on the time value, not the shuffled position,
        # so the chronological cut is what the assertions check.
        time_value = float(row)
        position = int(np.where(data["Time"] == time_value)[0][0])
        data["Class"][position] = 1
        data["V1"][position] = 6.0
    return pd.DataFrame(data)[list(FEATURE_COLUMNS) + ["Class"]]


def test_schema_rejects_missing_extra_and_bad_class() -> None:
    frame = _frame(n=40)
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_creditcard_frame(frame.drop(columns=["V28"]))
    extra = frame.copy()
    extra["Merchant"] = "m0"
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_creditcard_frame(extra)
    bad = frame.copy()
    bad.loc[0, "Class"] = 2
    with pytest.raises(ValueError, match="Class must be 0"):
        validate_creditcard_frame(bad)


def test_chronological_split_is_disjoint_and_normal_only() -> None:
    frame = _frame()
    splits = chronological_split(frame, seed=7)
    train_ids = set(splits.train_normal["row_id"])
    calibration_ids = set(splits.calibration_normal["row_id"])
    evaluation_ids = set(splits.evaluation["row_id"])
    assert train_ids.isdisjoint(calibration_ids)
    assert train_ids.isdisjoint(evaluation_ids)
    assert calibration_ids.isdisjoint(evaluation_ids)
    assert splits.train_normal["Time"].is_monotonic_increasing
    assert splits.calibration_normal["Time"].is_monotonic_increasing
    assert splits.evaluation["Time"].is_monotonic_increasing
    assert splits.train_normal["Time"].max() < splits.calibration_normal["Time"].min()
    assert splits.calibration_normal["Time"].max() < splits.evaluation["Time"].min()
    assert (splits.train_normal["Class"] == 0).all()
    assert (splits.calibration_normal["label"] == 0).all()
    assert (splits.calibration_normal["Class"] == 0).all()
    assert splits.train_fraud_excluded == 1
    assert splits.calibration_fraud_excluded == 1
    assert splits.evaluation_fraud == 2
    assert (splits.evaluation["Class"] == 1).sum() == 2
    assert not splits.subsampled_train


def test_train_cap_does_not_drop_evaluation_fraud() -> None:
    frame = _frame()
    capped = chronological_split(frame, max_train_normals=25, seed=3)
    assert capped.subsampled_train
    assert len(capped.train_normal) == 25
    assert (capped.train_normal["Class"] == 0).all()
    assert capped.evaluation_fraud == 2
    assert (capped.evaluation["Class"] == 1).sum() == 2
    idle = chronological_split(frame, max_train_normals=10_000, seed=3)
    assert not idle.subsampled_train
    assert len(idle.train_normal) == len(chronological_split(frame).train_normal)


def test_scaler_uses_train_normals_only_and_logs_amount() -> None:
    frame = _frame(seed=1)
    splits = chronological_split(frame, seed=1)
    preprocessor = CreditCardPreprocessor().fit(splits.train_normal)
    amount_index = FEATURE_COLUMNS.index("Amount")
    time_index = FEATURE_COLUMNS.index("Time")
    expected_amount = np.log1p(splits.train_normal["Amount"].to_numpy(dtype=np.float64)).mean()
    expected_time = splits.train_normal["Time"].to_numpy(dtype=np.float64).mean()
    assert preprocessor.mean_[amount_index] == pytest.approx(expected_amount)
    assert preprocessor.mean_[time_index] == pytest.approx(expected_time)
    # A huge calibration amount must not move the stored mean.
    poisoned = splits.calibration_normal.copy()
    poisoned.loc[0, "Amount"] = 1e9
    assert CreditCardPreprocessor().fit(splits.train_normal).mean_[amount_index] == pytest.approx(expected_amount)
    scaled = preprocessor.transform(splits.evaluation)
    assert scaled.shape == (len(splits.evaluation), 30)
    assert np.isfinite(scaled).all()
    zero_amount = splits.train_normal.loc[splits.train_normal["Time"] == 1.0]
    assert float(zero_amount["Amount"].iloc[0]) == 0.0
    assert float(splits.train_normal.loc[splits.train_normal["Time"] == 2.0, "V1"].iloc[0]) < 0
    preprocessor.transform(zero_amount)


def test_continuous_only_shapes_and_deterministic_score() -> None:
    set_seed(5)
    frame = _frame()
    splits = chronological_split(frame, seed=5)
    preprocessor = CreditCardPreprocessor().fit(splits.train_normal)
    model = TabularVAE(n_continuous=30, cat_cardinalities=(), latent_dim=6, hidden_dim=16)
    assert len(model.categorical_heads) == 0
    assert model.encoder[0].in_features == 30
    scorer = ReconstructionScorer(
        model,
        preprocessor,
        LossConfig(),
        creditcard_schema(),
        threshold=1.0,
    )
    first = scorer.reconstruction_errors(splits.evaluation)
    second = scorer.reconstruction_errors(splits.evaluation)
    assert len(first) == len(splits.evaluation)
    assert list(first.columns)[:3] == ["reconstruction_error", "continuous_error", "categorical_error"]
    assert np.allclose(first["reconstruction_error"], second["reconstruction_error"])
    assert np.allclose(first["categorical_error"], 0.0)
    features = scorer.continuous_feature_huber(splits.evaluation)
    assert features.shape == (len(splits.evaluation), 30)
    assert np.allclose(features.to_numpy().sum(axis=1), first["continuous_error"].to_numpy())

    scaled = preprocessor.transform(splits.train_normal.iloc[:32])
    labels = splits.train_normal["label"].to_numpy()[:32]
    loader = DataLoader(tensor_dataset(scaled, labels), batch_size=16, shuffle=False)
    continuous, categorical, _batch_labels = next(iter(loader))
    assert categorical.shape == (16, 0)
    recon, logits, mu, logvar = model(continuous, categorical)
    assert recon.shape == (16, 30)
    assert logits == []
    assert mu.shape == (16, 6)
    assert logvar.shape == (16, 6)
    _total, _huber, cross_entropy, _kl = VAELoss()(recon, continuous, logits, categorical, mu, logvar)
    assert float(cross_entropy) == 0.0
    mu_np = encode_mu(model, scaled[:20])
    assert mu_np.shape == (20, 6)
    assert np.allclose(mu_np, encode_mu(model, scaled[:20]))

    history = VAETrainer(model, VAELoss(), lr=1e-3).fit(loader, epochs=1)
    assert history[0]["cross_entropy"] == 0.0

    row = splits.evaluation.iloc[0]
    payload = {name: float(row[name]) for name in FEATURE_COLUMNS}
    result = scorer.score(payload)
    assert result["categorical_error"] == {}
    assert result["high_risk"] in (True, False)
    again = scorer.score(payload)
    assert again["reconstruction_error"] == pytest.approx(result["reconstruction_error"])


def test_threshold_comes_from_calibration_normals() -> None:
    frame = _frame()
    splits = chronological_split(frame, seed=2)
    preprocessor = CreditCardPreprocessor().fit(splits.train_normal)
    model = TabularVAE(30, ())
    scorer = ReconstructionScorer(model, preprocessor, LossConfig(), creditcard_schema())
    threshold = scorer.fit_threshold(splits.calibration_normal, percentile=99.0)
    calibration_scores = scorer.reconstruction_errors(splits.calibration_normal)["reconstruction_error"].to_numpy()
    assert threshold == pytest.approx(float(np.percentile(calibration_scores, 99)))
    dirty = splits.calibration_normal.copy()
    dirty.loc[0, "label"] = 1
    with pytest.raises(ValueError, match="fraud"):
        scorer.fit_threshold(dirty, percentile=99.0)

    evaluation_scores = scorer.reconstruction_errors(splits.evaluation)["reconstruction_error"].to_numpy()
    table = threshold_sensitivity(
        calibration_scores,
        splits.evaluation["label"].to_numpy(),
        evaluation_scores,
    )
    assert list(table["percentile"]) == [95.0, 97.0, 99.0, 99.5]
    flipped = splits.evaluation["label"].to_numpy().copy()
    flipped[:] = 1 - flipped
    # Thresholds ignore evaluation labels. The percentile column is identical.
    flipped_table = threshold_sensitivity(calibration_scores, flipped, evaluation_scores)
    assert np.allclose(table["threshold"], flipped_table["threshold"])


def test_isolation_forest_and_visualization_pca_use_train_normals() -> None:
    frame = _frame(n=400, seed=4)
    splits = chronological_split(frame, seed=4)
    # 400 rows: calibration normals are under the scorer's 200-row floor.
    # The forest itself has no such floor.
    preprocessor = CreditCardPreprocessor().fit(splits.train_normal)
    train_x = preprocessor.transform(splits.train_normal)
    calibration_x = preprocessor.transform(splits.calibration_normal)
    evaluation_x = preprocessor.transform(splits.evaluation)
    forest = fit_isolation_forest(train_x, seed=4, n_estimators=10)
    calibration_scores = isolation_anomaly_scores(forest, calibration_x)
    evaluation_scores = isolation_anomaly_scores(forest, evaluation_x)
    assert calibration_scores.shape == (len(splits.calibration_normal),)
    assert evaluation_scores.shape == (len(splits.evaluation),)
    threshold = float(np.percentile(calibration_scores, 99))
    metrics = detection_metrics(splits.evaluation["label"].to_numpy(), evaluation_scores, threshold)
    assert metrics["support_fraud"] == splits.evaluation_fraud
    assert metrics["support_normal"] + metrics["support_fraud"] == len(splits.evaluation)

    mu = np.random.default_rng(0).normal(size=(50, 6))
    pca = fit_visualization_pca(mu, max_rows=12, seed=0)
    assert pca.n_samples_ == 12
    assert pca.components_.shape == (2, 6)
    projected = pca.transform(mu[:5])
    assert projected.shape == (5, 2)


def test_mixed_schema_still_trains_and_scores_categories() -> None:
    set_seed(9)
    frame = generate_synthetic_transactions(n_normal=300, n_fraud=40, seed=9)
    module = TransactionDataModule(
        frame,
        n_train=180,
        n_calibration=60,
        n_val_normal=60,
        batch_size=32,
        seed=9,
    )
    assert module.preprocessor.require_positive is True
    model = TabularVAE(n_continuous=5, cat_cardinalities=DEFAULT_SCHEMA.cardinalities, latent_dim=6, hidden_dim=16)
    assert model.encoder[0].in_features == 18
    assert [head.out_features for head in model.categorical_heads] == [10, 3]
    encoded = encode_categoricals(module.train_frame)
    assert encoded.shape == (len(module.train_frame), 2)
    history = VAETrainer(model, VAELoss(), lr=1e-3).fit(module.train_loader, epochs=1)
    assert "cross_entropy" in history[0]
    scorer = ReconstructionScorer(
        model,
        module.preprocessor,
        LossConfig(),
        DEFAULT_SCHEMA,
        threshold=0.0,
    )
    scored = scorer.score_frame(module.validation_frame.iloc[:8])
    assert "categorical_error_merchant_id" in scored.columns
    assert "categorical_error_device_type" in scored.columns
    result = scorer.score(
        {
            "amount": 24.5,
            "fee": 0.5,
            "rolling_spend_24h": 57.0,
            "distance_from_home_km": 3.0,
            "seconds_since_last_txn": 6600.0,
            "merchant_id": "m3",
            "device_type": "contactless",
        }
    )
    assert set(result["categorical_error"]) == {"merchant_id", "device_type"}
    with pytest.raises(ValueError, match="> 0"):
        scorer.score(
            {
                "amount": 0.0,
                "fee": 0.5,
                "rolling_spend_24h": 57.0,
                "distance_from_home_km": 3.0,
                "seconds_since_last_txn": 6600.0,
                "merchant_id": "m3",
                "device_type": "contactless",
            }
        )


def test_empty_categoricals_reject_a_real_category_tensor() -> None:
    model = TabularVAE(4, ())
    continuous = torch.zeros(3, 4)
    categorical = torch.zeros(3, 1, dtype=torch.long)
    with pytest.raises(ValueError, match="continuous-only"):
        model.encode(continuous, categorical)
