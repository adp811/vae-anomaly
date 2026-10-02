# Zero-shot fraud labeler

`TabularVAE` is a variational autoencoder for mixed tabular card transactions. It is trained only on normal rows. A later row is flagged when the model cannot reconstruct it. The fraud label never enters the loss.

The implementation lives in `src/fraudvae/` (`dataset.py`, `model.py`, `loss.py`, `train.py`, `score.py`). `notebooks/zero_shot_fraud_vae.ipynb` is the run: it builds the splits, trains for 25 epochs, freezes a threshold, and calls `ReconstructionScorer.score`.

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

## Generative model and the ELBO

A transaction is a mixed vector $x$: five positive continuous fields (`amount`, `fee`, `rolling_spend_24h`, `distance_from_home_km`, `seconds_since_last_txn`) and two categoricals (`merchant_id` with 10 classes `m0`–`m9`, `device_type` with `chip`, `contactless`, `online`). The model treats $x$ as produced from a latent code $z$ of dimension 6.

The decoder is $p(x \mid z)$. The encoder is an approximate posterior $q(z \mid x) = \mathcal{N}(\mu, \operatorname{diag}(\exp(\mathrm{logvar})))$. The prior is $p(z) = \mathcal{N}(0, I)$. The marginal $\log p(x)$ integrates the decoder over every $z$, which this network cannot compute. Training optimizes the evidence lower bound instead:

```latex
\operatorname{ELBO}(x) = \mathbb{E}_{q(z \mid x)}[\log p(x \mid z)] - \mathrm{KL}\big(q(z \mid x) \parallel p(z)\big)
```

`VAELoss` minimizes a weighted negative of that bound.

The reconstruction term is the expectation. It forces the decoder, given $z$, to put the observed row back together: the five scaled continuous values, the merchant class, and the device class. If $z$ does not carry what the decoder needs, or the decoder never learned this kind of row, that term stays large.

The KL term forces $q(z \mid x)$ toward $\mathcal{N}(0, I)$. Without it the encoder can park every row in its own far-away code and the decoder can memorize those codes. With it, normal rows have to share one region of latent space, which is the region the decoder actually learns to read.

In this repo the reconstruction piece is not a single Gaussian log-likelihood. `VAELoss` uses Huber loss on the continuous columns and cross-entropy on each categorical head. There is no learned observation variance. The KL piece is the closed form for a diagonal Gaussian against $\mathcal{N}(0, I)$, multiplied by $\beta$.

## Encoder

`ContinuousPreprocessor` runs before the network. On each continuous column it applies `log1p`, then subtracts the training mean and divides by the training standard deviation (`ddof=0`). `fit` is called on the normal training split only. Calibration rows, validation rows, and `score` all reuse that stored `mean_` and `std_`. Non-positive inputs are rejected.

`TabularVAE.encode` one-hots the class indices and concatenates them with the five scaled values. Order is fixed by `FeatureSchema`: continuous columns in the order above, then `merchant_id` (index 0 is `m0`), then `device_type` (index 0 is `chip`). The input width is `5 + 10 + 3 = 18`.

The trunk is `Linear(18, 64)`, ReLU, `Linear(64, 64)`, ReLU. Two linear maps then emit $\mu$ and $\mathrm{logvar}$, each of size 6 (`latent_dim=6`, `hidden_dim=64`). $\mathrm{logvar}$ is clamped to $[-10, 10]$ so $\exp(\mathrm{logvar})$ cannot overflow. That clamp is a guard; a healthy run stays well inside it.

## Reparameterization

Training has to draw $z$ from $q(z \mid x)$ and still backpropagate into $\mu$ and $\mathrm{logvar}$. `TabularVAE.reparameterize` uses the Gaussian trick:

```latex
\begin{aligned}
\sigma &= \exp(0.5 \cdot \mathrm{logvar}) \\
z &= \mu + \sigma \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
\end{aligned}
```

`TabularVAE.forward` is the training path: encode, sample $z$, decode. The sample is what lets the reconstruction term send a gradient through a random code while the KL term, which depends only on $\mu$ and $\mathrm{logvar}$, keeps that code near the prior.

Scoring does not sample. `ReconstructionScorer` calls `encode`, discards $\mathrm{logvar}$ for the draw, and runs `decode(mu)`. The same transaction then produces the same reconstruction error on every call. A sampled $z$ would move the score whenever $\epsilon$ moved, and the frozen threshold would not mean the same thing from request to request.

## Branched decoder

`decode` maps $z$ through `Linear(6, 64)`, ReLU, `Linear(64, 64)`, ReLU, then splits.

`continuous_head` is one `Linear(64, 5)` with no activation. Its five outputs line up with the scaled `amount`, `fee`, `rolling_spend_24h`, `distance_from_home_km`, and `seconds_since_last_txn`. Those targets are z-scored, so they are centered near 0 on normal rows and the head is left linear.

`categorical_heads` is a `ModuleList` of two linear layers, `Linear(64, 10)` for `merchant_id` and `Linear(64, 3)` for `device_type`. Both return raw logits. The module does not apply softmax. `F.cross_entropy` applies the log-softmax itself; a softmax in the head would feed it probabilities and break the loss.

## Loss

`LossConfig` fixes the weights: `lambda_cont=1`, `lambda_cat=1`, `beta=0.1`, `huber_delta=1.0`. `VAELoss.forward` returns the weighted total and the three unweighted pieces. `VAETrainer` prints them at the end of every epoch.

Training reductions, chosen so the numbers do not grow just because the batch got bigger:

- Huber: `F.huber_loss` with `delta=1.0`, mean over the batch and over the five continuous columns.
- Cross-entropy: `F.cross_entropy` on each head against the class index, mean over the batch, then summed across the merchant head and the device head.
- KL: summed over the latent dimensions of each row, then averaged over the batch.

```latex
\begin{aligned}
\mathrm{KL}_i &= -\frac{1}{2} \sum_j \left(1 + \mathrm{logvar}_{i,j} - \mu_{i,j}^{2} - \exp(\mathrm{logvar}_{i,j})\right) \\
\mathrm{KL} &= \frac{1}{B} \sum_i \mathrm{KL}_i
\end{aligned}
```

```latex
\mathrm{total} = 1.0 \cdot \mathrm{Huber} + 1.0 \cdot (\mathrm{CE}_{\mathrm{merchant}} + \mathrm{CE}_{\mathrm{device}}) + 0.1 \cdot \mathrm{KL}
```

Huber is used because the raw fields are right-skewed money, distance, and time quantities. `log1p` and standardization pull normal rows into a unit-scale space, and delta 1.0 sits at one training standard deviation of that space. Inside the delta the penalty is quadratic. Outside it the slope is constant. Squared error would keep growing with the square of a residual, so a few legitimately large normal amounts would dominate the gradient. Huber still penalizes those residuals, with a slope that stays bounded.

$\beta$ is $0.1$ rather than $1$ so the KL term does not take the whole objective on the first epochs, when cross-entropy starts near $\log 10 + \log 3$ and the decoder still has to learn the normal manifold. Small $\beta$ leaves room for Huber and cross-entropy to fall.

They do fall, and the balance still shifts. On the notebook run, epoch 25 printed:

```
total=0.8791  huber=0.2116  cross_entropy=0.0861  kl=5.8136
```

With the weights applied, that is Huber $0.2116$, cross-entropy $0.0861$, and $\beta \cdot \mathrm{KL} = 0.5814$. The weighted KL term was larger than Huber and cross-entropy together. The reconstruction terms had already dropped (Huber from about 0.42 at epoch 1, cross-entropy from about 3.27). The KL remained a training regularizer. The anomaly score below does not include it.

## Training on normal rows only

`generate_synthetic_transactions` draws both classes, and `TransactionDataModule` then cuts them apart. The default normal split is 5,200 train, 1,200 calibration, 1,200 validation. All 400 fraud rows go to validation. The module checks that the training frame and the calibration frame are entirely label 0, and that no `row_id` is shared across the three splits.

`VAETrainer.fit` reads `(continuous, categorical, labels)` from the normal-only loader and drops `labels`. `TabularVAE` and `VAELoss` never see the fraud flag. The only distribution the ELBO is fit to is normal traffic: fee tracks amount, rolling 24h spend falls as `seconds_since_last_txn` grows, device type follows distance, merchant follows spend.

That is what makes the detector zero-shot. The decoder's parameters are a description of normal rows in the latent region the KL allows. A fraud row breaks those links (larger amount, a fee that does not follow it, a long gap with a high rolling spend, a far distance with `chip`, a rare merchant). The encoder still proposes a $z$, and the decoder still emits its best reconstruction, but that reconstruction was learned on normals, so the Huber and cross-entropy terms come out higher. The hidden label is used only after the fact, in `detection_report`, to compute precision, recall, and F1 on the validation mix.

## Anomaly score and threshold

`ReconstructionScorer.reconstruction_errors` scores one row at a time in the reconstruction scale, with KL omitted. Huber is summed over the five scaled columns (`reduction="none"`, then sum). Each categorical head contributes its own cross-entropy for that row's true class, and those two numbers are summed. The same weights as training are applied:

```latex
\mathrm{score} = \lambda_{\mathrm{cont}} \cdot \mathrm{Huber}_{\mathrm{sum}} + \lambda_{\mathrm{cat}} \cdot (\mathrm{CE}_{\mathrm{merchant}} + \mathrm{CE}_{\mathrm{device}})
```

With the default weights that is the sum of the continuous Huber and the two cross-entropies. Training logged batch means; the threshold lives in this per-row sum. The functions, the Huber delta, and the lambdas are the ones on `LossConfig`.

`fit_threshold` computes that score on the pure-normal calibration holdout and stores the 95th percentile. Those rows were not used to fit the weights, and they are not the rows the notebook reports metrics on. The value stays on the scorer. Later calls read it.

`high_risk` is true when `reconstruction_error >= threshold`. On the dataframe path those rows are labeled `High-Risk Anomaly`. `score(transaction)` returns `reconstruction_error`, `threshold`, `high_risk`, `continuous_error`, and `categorical_error` as a dict with `merchant_id` and `device_type`.

KL is left out of the score because it does not ask whether the row was rebuilt. It asks how far $q(z \mid x)$ sits from $\mathcal{N}(0, I)$. By epoch 25 that term already outweighed the reconstruction terms in the training objective. Folding it into the score would rank rows by posterior geometry, including rows the decoder reconstructed cleanly, and the calibration percentile would no longer be a percentile of reconstruction error. The detector's claim is the reconstruction claim: a row is unusual when the normal-trained decoder cannot put it back.

## Train and serve parity

`ReconstructionScorer` is built with the trained `TabularVAE`, the `ContinuousPreprocessor` that was fit on train normals, the same `LossConfig`, and the same `FeatureSchema`. `score` does not construct a second encoder or a second loss.

- Preprocess is `log1p` then `(value - mean_) / std_` with the training statistics. Raw amounts are not passed into a network that was trained on standardized values.
- One-hot order is `encode` plus `encode_categoricals`: merchant block, then device block, class index equal to vocab position. An unknown `merchant_id` or `device_type` raises. It is not turned into a zero vector.
- `lambda_cont`, `lambda_cat`, and `huber_delta` come from the `LossConfig` instance the trainer used. `beta` stays on that config for training and is not multiplied into the score.
- The threshold is the float stored by `fit_threshold`. `score` and `score_batch` read it. They do not recompute a percentile on the request.
