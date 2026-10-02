# Zero-shot fraud labeler

A small PyTorch package and one notebook. A tabular variational autoencoder is trained only on normal synthetic card transactions. Fraud is flagged later by reconstruction error. No fraud labels enter the training loop.

The notebook is the thing to run. The package (`dataset`, `model`, `loss`, `train`, `score`) is the same code a later scoring service would import. `ReconstructionScorer.score(transaction)` is that function.

This is a synthetic baseline. It is not a production fraud model. The deployment map is `vae-fraud-plan.md` in the project docs, not in this repo.

## Layout

- `src/fraudvae/` — data, `TabularVAE`, loss, training loop, `score`
- `notebooks/zero_shot_fraud_vae.ipynb` — generate data, train, set the threshold, print precision / recall / F1

## Run the notebook

CPU is enough. The default run is 8,000 rows and 25 epochs.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/zero_shot_fraud_vae.ipynb
```

From the repo root, without opening the UI:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/zero_shot_fraud_vae.ipynb
```

The notebook adds `src/` to `sys.path` itself, whether you start it from the repo root or from `notebooks/`.

## Defaults the notebook uses

| Choice | Value |
| --- | --- |
| Rows | 7,600 normal + 400 fraud (5%) |
| Seed | 42 |
| Continuous preprocessing | `log1p`, then standardize, fit on train normals only |
| Latent size | 6 |
| Hidden width | 64 |
| Epochs | 25 |
| Loss weights | `lambda_cont=1`, `lambda_cat=1`, `beta=0.1` |
| Huber delta | 1.0 in the scaled space |
| Threshold | 95th percentile on a pure-normal calibration holdout |
| Normal split | 5,200 train / 1,200 calibration / 1,200 validation |
