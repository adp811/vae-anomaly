# Zero-shot fraud labeler

This model answers one question about a transaction: is this row harder for a network trained only on normal traffic to rebuild than almost all other normal rows? The answer is a reconstruction score and a boolean, `high_risk`. It is not a probability of fraud, and the fraud label is never an input to training.

The table is synthetic. `generate_synthetic_transactions` in `src/fraudvae/dataset.py` draws credit-card-like rows: amounts, fees, distances, merchant ids, device types. It is not an external or real card-fraud dataset, and nothing in this repo was checked against real payments. Normal rows follow a few tight links (fee tracks amount, rolling spend falls as the gap since the last transaction grows, device follows distance, merchant follows spend). Fraud rows are drawn to break those links by a wide margin. On that generator the notebook's fraud recall is 1.0. That shows the pipeline fires when the assumptions hold. It does not show that the model would catch fraud in production.

The derivation, with dimensions at each step, is in [docs/vae_theory.pdf](docs/vae_theory.pdf). Source: [docs/vae_theory.tex](docs/vae_theory.tex). GitHub will open the PDF in its viewer. The equations live there, not in this file.

## What happens to one row

| Stage | What is computed | Width |
| --- | --- | --- |
| Raw row | `amount`, `fee`, `rolling_spend_24h`, `distance_from_home_km`, `seconds_since_last_txn`, plus `merchant_id` (`m0`–`m9`) and `device_type` (`chip`, `contactless`, `online`) | 5 numbers + 2 labels |
| Preprocess | `log1p`, then z-score. Mean and std are fit on the 5,200 training normals only. Categories become one-hots in vocab order. | 18 |
| Encoder | 18 → 64 → 64 → `mu` and `logvar` | 6 and 6 |
| Latent draw | Training uses `z = mu + sigma * epsilon`. Scoring decodes `mu` so the same row does not jitter. | 6 |
| Decoder | 6 → 64 → 64, then a linear head for the 5 scaled numbers and logit heads of size 10 and 3. No softmax in the module. | 5 + 10 + 3 |
| Score | Sum of the five Huber terms and the two cross-entropies. KL is not included. | 1 |
| Decision | `high_risk` if the score is at or above the frozen 95th percentile of 1,200 calibration normals. | boolean |

An ordinary autoencoder can park unusual rows on their own latent islands and still reconstruct them. The KL term pressures `q(z|x)` toward a unit Gaussian, so nearby normal examples share one region and interpolation in that region stays on normal-looking outputs. That is the reason to use a VAE here. It is not a claim that a VAE always beats an autoencoder at anomaly detection.

Merchant and device are categories, not positions on a line. Reconstructing a merchant id with squared error would pretend that `m3` is "between" `m2` and `m4`. Huber is for the scaled numeric residuals, where a large legitimate amount should not explode the gradient. Cross-entropy is for which class was actually present. The heads are separate because those are different kinds of mistakes.

The score measures how far the row sits from structure the decoder learned on normals. A fraud row can still reconstruct cleanly if it looks like normal traffic in these seven fields. A legitimate row can score high if it is merely new: a merchant the training window barely saw, a distance the normal sample never reached. Do not read the score as `P(fraud)`.

KL is in the training loss because that is what shapes the latent region. It is absent from `ReconstructionScorer` because this project chose reconstruction difficulty as the signal. A large KL means the encoder is uncertain or far from the prior. That can be an anomaly signal in other systems. It is not the one implemented in `score.py`.

Training averages Huber across the batch and the five columns, and averages each cross-entropy over the batch, so the optimizer sees a stable scale. Scoring sums those same pieces inside one row, with the same Huber delta and the same `lambda_cont` and `lambda_cat`, so one transaction has one severity. `beta=0.1` keeps the KL from swallowing the objective while the decoder is still learning. By epoch 25 of the recorded run the weighted KL (0.581) was already larger than Huber plus cross-entropy (0.298). The score still ignores KL. The theory note is the place for that arithmetic.

## How to run

CPU is enough. The notebook uses 8,000 rows (7,600 normal and 400 fraud), seed 42, and 25 epochs. It adds `src/` to `sys.path` from the repo root or from `notebooks/`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/zero_shot_fraud_vae.ipynb
```

Without the UI, from the repo root:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/zero_shot_fraud_vae.ipynb
```

Rebuild the theory PDF after editing the source:

```bash
pdflatex -interaction=nonstopmode -output-directory docs docs/vae_theory.tex
pdflatex -interaction=nonstopmode -output-directory docs docs/vae_theory.tex
```

`ReconstructionScorer.score` takes one dict with the seven fields above and returns `reconstruction_error`, `threshold`, `high_risk`, `continuous_error`, and `categorical_error`. Unknown merchant or device tokens raise. Non-positive amounts, fees, distances, or times raise. The threshold is whatever `fit_threshold` stored. It is not recomputed on the request.

## Recorded run

Seed 42. Threshold from 1,200 pure-normal calibration rows, printed as 2.1162. Validation was a different 1,200 normals plus all 400 fraud rows. Those calibration rows were not used to fit weights, and metrics were not computed on them.

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Normal | 1.000 | 0.956 | 0.977 | 1200 |
| Fraud | 0.883 | 1.000 | 0.938 | 400 |

Accuracy was 0.967. Every synthetic fraud row scored above the cutoff (recall 100%, precision 88.3%). Validation normals were flagged at 4.4%, not 5%. The cutoff is the 95th percentile of the calibration sample. The validation normals are another draw of the same size. Their own 95th percentile was 2.040, a bit under 2.1162, so slightly fewer than one in twenty of them crossed a line that was fit on the other sample.

Epoch 25 printed `total=0.8791`, `huber=0.2116`, `cross_entropy=0.0861`, `kl=5.8136`.

## Limits of this setup

The fraud shift is large on purpose. On the validation set the mean scaled fraud vector was about `(4.9, 2.3, 4.8, 7.9, 4.7)` against a normal mean near zero. A detector that cannot separate that gap is broken. Separating it does not mean the same weights would separate subtle fraud.

The merchant and device vocabularies are closed. An unseen id is a schema error, not a fraud flag. There is no cardholder id, no merchant graph, and no sequence model. `seconds_since_last_txn` and `rolling_spend_24h` are columns on one row, not a learned history.

If the normal training rows contain fraud, the decoder learns to rebuild those patterns and the score goes quiet. The notebook split rejects that by construction. A real extract would not. The frozen threshold also goes stale when normal behavior drifts. Replacing 2.1162 requires a new normal calibration window, not a per-request percentile. And the score is not a calibrated probability. A value of 2.2 means "above the cutoff we stored," not "92% fraud."

## Layout

- `src/fraudvae/dataset.py` — generator, splits, scaler, class indices
- `src/fraudvae/model.py` — `TabularVAE`
- `src/fraudvae/loss.py` — `LossConfig`, `VAELoss`
- `src/fraudvae/train.py` — `VAETrainer`, which drops the label
- `src/fraudvae/score.py` — `ReconstructionScorer.score`
- `notebooks/zero_shot_fraud_vae.ipynb` — the run that produced the numbers above
