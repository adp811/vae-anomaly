# Zero-shot fraud labeler

`TabularVAE` is a variational autoencoder for mixed tabular card transactions. Training uses normal rows only. A later row is flagged when the model cannot reconstruct it. The fraud label never enters the loss.

This is a synthetic baseline, not a production fraud model. The equations, dimensions, and a row-by-row walkthrough are in the LaTeX note, which is the place to read the math. GitHub's Markdown viewer does not render fenced LaTeX, and the PDF viewer does.

- [VAE theory (PDF)](docs/vae_theory.pdf)
- [LaTeX source](docs/vae_theory.tex)

## How to run

CPU is enough. The notebook uses 8,000 rows (7,600 normal, 400 fraud), seed 42, and 25 epochs. It adds `src/` to `sys.path` whether you start it from the repo root or from `notebooks/`.

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

To rebuild the theory PDF after editing the source:

```bash
pdflatex -interaction=nonstopmode -output-directory docs docs/vae_theory.tex
pdflatex -interaction=nonstopmode -output-directory docs docs/vae_theory.tex
```

## Architecture summary

`src/fraudvae/` is the library. The notebook only orchestrates it.

- Continuous fields, all required to be positive: `amount`, `fee`, `rolling_spend_24h`, `distance_from_home_km`, `seconds_since_last_txn`. `ContinuousPreprocessor` applies `log1p`, then a z-score fit on the 5,200 train normals only.
- Categorical fields: `merchant_id` (`m0`–`m9`) and `device_type` (`chip`, `contactless`, `online`). One-hot order is merchant, then device. The encoder input is 18 wide.
- `TabularVAE`: encoder 18 → 64 → 64 → `mu` and `logvar` (each 6). Training samples `z = mu + sigma * epsilon`. Scoring decodes `mu`.
- Decoder 6 → 64 → 64, then a linear head of width 5 and two logit heads of widths 10 and 3. No softmax in the module.
- `VAELoss`: mean Huber (delta 1) + summed per-head mean cross-entropy + batch-mean KL, with weights `lambda_cont=1`, `lambda_cat=1`, `beta=0.1`.
- `ReconstructionScorer.score` sums the five Huber terms and the two cross-entropies for that row. KL is not in the score. `high_risk` is true when the score is at or above a threshold frozen from the 1,200 pure-normal calibration rows.

## Observed metrics

On the seed-42 notebook run the calibration threshold printed as 2.1162. Validation was 1,200 held-out normals plus all 400 fraud rows.

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Normal | 1.000 | 0.956 | 0.977 | 1200 |
| Fraud | 0.883 | 1.000 | 0.938 | 400 |

Accuracy was 0.967. About 4.4% of validation normals were flagged, and every synthetic fraud row was. Epoch 25 printed `total=0.8791`, `huber=0.2116`, `cross_entropy=0.0861`, `kl=5.8136`. With `beta=0.1`, the weighted KL term (0.581) was larger than Huber plus cross-entropy (0.298). The score still ignores KL.

## Limitations

Fraud rows in this generator are deliberately off the normal manifold (shifted amounts, a fee that does not track amount, a long gap with high rolling spend, a far distance on `chip`, rare merchants). Recall 1.0 checks that the reconstruction score notices that synthetic shift. It does not transfer to a real book of transactions.

Unknown merchant or device tokens are rejected. The threshold is not recomputed per call. There is no auth, no feature store, and no online update of the weights in this repo.

## Layout

- `src/fraudvae/` — `dataset.py`, `model.py`, `loss.py`, `train.py`, `score.py`
- `notebooks/zero_shot_fraud_vae.ipynb` — the run
- `docs/vae_theory.tex`, `docs/vae_theory.pdf` — the math writeup
