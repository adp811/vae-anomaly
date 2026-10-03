# Zero-shot fraud labeler

This model answers one question about a transaction: is this row harder for a network trained only on normal traffic to rebuild than almost all other normal rows? The answer is a reconstruction score and a boolean, `high_risk`. It is not a probability of fraud, and the fraud label is never an input to training.

The first notebook is synthetic. `generate_synthetic_transactions` in `src/fraudvae/dataset.py` draws credit-card-like rows: amounts, fees, distances, merchant ids, device types. It is not an external card-fraud dataset. Normal rows follow a few tight links (fee tracks amount, rolling spend falls as the gap since the last transaction grows, device follows distance, merchant follows spend). Fraud rows are drawn to break those links by a wide margin. On that generator the notebook's fraud recall is 1.0. That shows the pipeline fires when the assumptions hold. It does not show that the model would catch fraud in production. A second notebook runs the continuous-only model on the public Kaggle card file. That run is a different split and a different feature set; its numbers are in the section below.

The notes in [docs/vae_theory.pdf](docs/vae_theory.pdf) walk through both notebooks in plain language: what the model flags, what the knobs do, and what the recorded runs showed. The derivations are an appendix. Source: [docs/vae_theory.tex](docs/vae_theory.tex).

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

## Kaggle credit-card notebook

[`notebooks/kaggle_credit_card_vae.ipynb`](notebooks/kaggle_credit_card_vae.ipynb) uses the public Kaggle Credit Card Fraud Detection file (`mlg-ulb/creditcardfraud`, `creditcard.csv`). The CSV is not in git. Place it at `data/creditcard.csv`. If it is missing, the notebook downloads it with `kagglehub`. If that call asks for a login, accept the dataset rules on Kaggle and put the token in `~/.kaggle/kaggle.json`. Do not commit the token or the CSV.

This path is continuous only: `Time`, `V1`–`V28`, and `Amount`. There are no merchant or device heads and no categorical cross-entropy. `V1`–`V28` arrived as anonymized PCA components; this repo does not refit them. `Amount` is `log1p` then z-scored. `Time` and the PCA columns are z-scored without `log1p` (`Time` is a clock starting at 0, and the PCA scores are signed). The scaler is fit on the normal rows that enter training.

The split follows `Time`, not a shuffle. The first 60% of rows is the training window, the next 20% is calibration, and the last 20% is the mixed holdout. Training and calibration keep `Class = 0`. Fraud in those windows is dropped, not moved into the holdout. Labels are used to build that normal reference and to evaluate. They are not in the loss. That is normal-only / semi-supervised anomaly detection.

The score is the sum of the 30 Huber terms. KL is not included. The operating threshold is the 99th percentile of calibration scores. A 95th percentile would aim at flagging about 5% of normals, which is a lot of alarms at this base rate. The notebook also prints 95, 97, 99, and 99.5, all taken from calibration scores. An IsolationForest fit on the same scaled training normals, and calibrated on the same window, is the baseline.

`MAX_TRAIN_NORMALS` was `None` for the recorded run, so all 170,524 training-window normals were used. Setting it to an integer subsamples those normals only. Evaluation fraud is never subsampled. The trunk is width 128 and `beta` is 0.01. The synthetic settings (`beta=0.1`, width 64) collapse on this table: the decoder rebuilds the training mean and the 6-dimensional code goes unused. Twelve epochs, seed 42, latent size 6. Final training Huber was 0.1382, against 0.3469 for predicting the scaled mean.

On the chronological holdout (56,887 normal, 75 fraud; 360 fraud rows excluded from training and 57 from calibration):

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | Normal FPR |
| --- | --- | --- | --- | --- | --- | --- |
| VAE, 99th pct, threshold 34.7925 | 0.0998 | 0.5333 | 0.1681 | 0.0809 | 0.9384 | 0.0063 |
| IsolationForest, 99th pct | 0.0470 | 0.2267 | 0.0778 | 0.0382 | 0.9528 | 0.0061 |

VAE confusion on that holdout: TN 56526, FP 361, FN 35, TP 40. Flagged fraction 0.0070. Accuracy was 0.993. Calling every row normal is already 56887/56962 ≈ 0.9987, so accuracy is the wrong summary. Training took 14.9 seconds. The notebook's own timer for the full run was 18.3 seconds. These figures belong to this split of this 2013 file.

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/kaggle_credit_card_vae.ipynb
```

## Layout

- `src/fraudvae/dataset.py` — synthetic generator, random normal-only splits, log1p scaler
- `src/fraudvae/real_data.py` — Kaggle schema, chronological split, continuous scaler
- `src/fraudvae/model.py` — `TabularVAE`, including the continuous-only case
- `src/fraudvae/loss.py` — `LossConfig`, `VAELoss`
- `src/fraudvae/train.py` — `VAETrainer`, which drops the label
- `src/fraudvae/score.py` — `ReconstructionScorer.score`
- `notebooks/zero_shot_fraud_vae.ipynb` — synthetic mixed-data run
- `notebooks/kaggle_credit_card_vae.ipynb` — public card-file run
- `tests/test_creditcard_pipeline.py` — split, scaler, scoring, and mixed-schema checks
