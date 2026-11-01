# Target geometry with a source tolerance

This implements the two-view candidate in `Target_Geometry_Fingerprint_Proposal.md`.
The implementation lives in `regularizers/target_geometry.py`; its trainer and
ablation runner are separate modules. The existing transformation regularizer,
VAE, ELBO, LoRA layers, trainer and transformation configurations are unchanged.

For a batch of **real images**, flatten and center across its members and form
`G(X) = centered(X) @ centered(X).T`. The dimensionless distortion is exactly

```
d(X, h) = sum((G(h(X)) - G(X))**2) / sum(G(X)**2)
```

There is no output trace normalization, division by output dimension or Gram
entry averaging. The denominator uses real input geometry only. Batches with
nonfinite or near-zero input Gram energy raise an error.

The source tolerance `c_s` is the mean distortion of the frozen source
autoencoder on independently sampled batches of held-out source images. Its
encoder uses posterior means and its decoder uses deterministic image means.
The held-out split is taken from the **source checkpoint's config and seed**.
The source decoder is an independent frozen copy, with LoRA removed.

On target batches, `F(x) = D_theta(mu_theta(x))` and
`H(x) = D_0(mu_theta(x))`. The full method uses
`a(X) = max(d(X,F), d(X,H))`, with the maximum inside each batch, and objective

```
ordinary_target_VAE_loss + eta * max(E_target_batches[a(X)] - c_s, 0)
```

Gradients pass through the frozen decoder to the adapted encoder. The adapted
decoder receives a regularizer gradient when its view attains the maximum.
Improving its geometry below the source-chart view does not increase the penalty.
Evaluation uses deterministic posterior means, independently of the sampled
latents used by the ordinary VAE loss. Bernoulli image means are sigmoid logits;
continuous Bernoulli uses the distribution's actual mean in this new regularizer.

## The expectation and stochastic optimization

Applying a hinge to every training batch would implement a different objective.
Instead, use the exact identity, valid for the proposal's finite soft weight:

```
eta * max(E[a] - c_s, 0)
    = max_{0 <= lambda <= eta} lambda * (E[a] - c_s).
```

The new trainer takes an ordinary model optimizer step on
`VAE_loss + lambda * (a_batch - c_s)` and a projected ascent step using the
same pre-update batch distortion. Write `q = lambda / eta`; initially `q = 0`:

```
q <- clip(q + dual_learning_rate * (a_batch - c_s), 0, 1)
lambda = eta * q
```

For a fixed multiplier, the batch violation and its model gradient estimate
the corresponding expectation without a per-batch hinge. The upper bound
`lambda <= eta` represents the stated **soft** penalty; removing it would give
the hard-constraint variant discussed in the proposal. No moving average,
extra tolerance margin, latent-distance loss or transformation bank is added.

The objective is algebraically identical, but finite-step nonconvex saddle
optimization need not solve its inner maximization or reach an optimum. The
dual step size can affect training. In particular, `eta` is a cap on the current
multiplier; the instantaneous coefficient can be smaller or zero. Inspect the
logged multiplier and violation before interpreting a sweep.

Validation holds model parameters fixed, averages batch maxima over a fixed,
independently sampled validation bank, **then** takes one hinge. This finite
Monte Carlo estimate has sampling uncertainty. Calibration's estimated standard
error is saved, but is not added to `c_s` and does not establish feasibility.
Source and target geometry batches use the same uniform sampling rule and B,
with distinct members within each batch and independent draws between batches.

The ordinary VAE train loader, its final partial batch, likelihood, beta and
optimizer are retained. Each step additionally draws a full-size target geometry
batch. Epoch training loss reports the VAE mean plus the hinge of the trajectory
mean distortion; validation is the fixed-parameter objective estimate used by
the existing scheduler/checkpoint logic.

## Parameters specific to this regularizer

| Parameter | Meaning and useful variation |
| --- | --- |
| `eta` | The single regularizer weight, equivalently the largest allowed dual multiplier. Larger values permit stronger geometric pressure relative to the unchanged VAE loss. The sweep uses 10, 100 and 1000. Weights from other Gram regularizers are not comparable. |
| `geometry_batch_size` | B, the number of connected real images in each geometry batch. It changes the geometry being compared and its statistical coverage, not the VAE minibatch size. It is shared by calibration, training and validation; `c_s` is recalibrated for every B. The sweep uses 64 and 128. Gram memory is O(B²), computation O(B² × image dimension), in addition to model evaluations. |
| `dual_learning_rate` | Projected ascent step size in normalized coordinate q. Larger values activate/deactivate pressure faster, with potentially noisier dynamics. The base config uses 0.01; the ablation crosses 0.001, 0.01 and 0.1 with every eta and geometry batch size. This is an optimization setting, not another term in the mathematical loss. |
| `view` | `full` implements the proposal. `target_only` removes H and retains target reconstruction geometry with the **same calibrated source tolerance**. `source_only` retains H and supplies only an encoder regularizer gradient. These are mechanism controls, not interchangeable definitions of the full method. |
| `calibration_batches` | Number of independent source batches for estimating `c_s`. Default 128. More batches reduce conditional Monte Carlo error; this is an estimation setting, not a strength parameter. At least two are required. |
| `validation_batches` | Number of fixed validation geometry batches used for the reported mean and hinge. Default 64. Increase for a more stable validation estimate, at extra evaluation cost. |
| `energy_floor` | Fixed minimum allowed squared input Gram norm. Default `1e-12`. Degenerate batches are rejected rather than rescaled. This is a numerical guard and should remain fixed in the ablation. |

`c_s` is a measured property of the source model and sampler, **not a tunable
slack parameter**. Calibration is rejected if `c_s >= 1`, because collapse
would satisfy that tolerance; a tolerance near 1 raises a warning. The existing
transformation parameters M, s, anchor counts, response/decoder weights and
trace normalization do not apply and are rejected if supplied here.

For orientation only, calibration of the existing checkpoints using B=64,
128 source batches, adaptation seed 42 and CPU execution measured:

| Dataset | `c_s` | Conditional Monte Carlo standard error |
| --- | ---: | ---: |
| MNIST | 0.0412654 | 0.0005873 |
| Ellipses | 0.00195168 | 0.0000380 |

These are startup calibration checks, not trained target results. Each actual
run recalibrates and records its own values. MNIST retains your existing 45°
interpolated rotation, for which exact orthogonal-transfer compatibility is
not assured. Ellipses retain the existing 90° rotation. No target transformation
is used to canonicalize training images for the regularizer.

## Run the ablations

From the repository root, using the project's Python environment:

```bash
python scripts/run_ablation.py config/ablation_vae_mnist_target_geometry.yaml
python scripts/run_ablation.py config/ablation_vae_ellipsis_target_geometry.yaml
```

The corresponding base LoRA files are
`config/lora_vae_mnist_target_geometry.yaml` and
`config/lora_vae_ellipsis_target_geometry.yaml`. To train one base config:

```bash
python scripts/run_target_geometry.py config/lora_vae_mnist_target_geometry.yaml
python scripts/run_target_geometry.py config/lora_vae_ellipsis_target_geometry.yaml
```

Each ablation contains **63 runs**: for each of seeds 42, 43 and 44, one ordinary
LoRA baseline, eighteen full-view grid points (3 eta × 2 B × 3 dual learning
rates), and two one-view controls at eta=100/B=64/dual rate=0.01. The controls
share the source calibration and target sampling draws of their matched full
run. All VAE/LoRA architecture, beta, training epochs, model learning rate,
training batch size and target shift match the existing dataset config. Only
regularizer-specific settings vary, with seed repetitions across all variants.
The YAML defaults to two parallel training workers; tune this execution setting
to available hardware. Calibration, model copies and extra forwards add cost.

The runner reuses the existing training workers, completed-checkpoint detection,
reconstruction scorers and FID implementation. It writes distinct model IDs,
per-run metrics, across-seed mean/std summaries and geometry-specific figures.
Target reconstruction is primary: the full-view model with minimum validation
target MSE is selected for qualitative inspection. Test FID is reported as a
secondary metric and is not used to choose that model. Compare seed-aggregated
results in `results/summary.csv` when drawing experimental conclusions.

## Result figures

Each ablation automatically generates the following files in `figures/` after
training and scoring; these are generated from the actual saved models/metrics:

- `eta_retention_gain_dual_0.001.png`, `eta_retention_gain_dual_0.01.png`, and
  `eta_retention_gain_dual_0.1.png`: **two panels per dual learning rate**, source
  retention and target gain versus eta. Geometry batch sizes use distinct,
  consistent line colors. Each point is the across-seed mean with ±1 sample
  standard deviation error bars. Only full two-view runs enter these curves;
  controls remain visible in the overview and CSV summaries.
- `figure_5_source_reconstructions.png`: held-out source inputs, source VAE
  reconstructions, ordinary LoRA reconstructions and selected target-geometry
  reconstructions of exactly the same images.
- `figure_6_target_reconstructions.png`: the equivalent comparison on held-out
  target images. `target_reconstructions.png` is retained as an alias.
- `figure_7_generations.png`: prior generations from the source VAE, ordinary
  LoRA and selected target-geometry model using **the same N(0,I) latent vectors**.
- `target_geometry.png`: the overview of target reconstruction MSE, source MSE
  increase and target FID, including the one-view controls.

The eta plots follow the existing regularizer's per-seed normalization convention
for target gain, and express source retention as 100 minus its normalized source
degradation. With S=source VAE, L=ordinary LoRA and P=plotted model:

```
source_degradation(P) = MSE_source(P) - MSE_source(S)
target_gain(P)        = MSE_target(S) - MSE_target(P)

Source retention (%) = 100 * (1 - source_degradation(P) / source_degradation(L))
Target gain (%)      = 100 * target_gain(P) / target_gain(L)
```

Thus source VAE has retention=100% and gain=0%; ordinary LoRA has retention=0%
and gain=100%. Both axes prefer higher values. Values are not clipped: negative
retention means worse forgetting than ordinary LoRA, and retention above 100%
means source reconstruction improved beyond the source VAE. Each model is
normalized against the ordinary LoRA run **with the same seed before averaging**.
This is a measure of recovering LoRA's source degradation, not pixel accuracy.

If any ordinary LoRA seed has a zero or negative reference change for a metric,
that metric uses raw MSE changes for all dual-rate figures instead of dividing
by the invalid reference or reversing the direction. Raw source retention is
`MSE_source(S)-MSE_source(P)`, with zero representing source VAE performance;
raw target gain is `MSE_target(S)-MSE_target(P)`. The axis and formula captions
state the units used. Error bars describe variation across seeds, not confidence
intervals or calibration uncertainty.

Qualitative examples are sampled reproducibly from held-out test datasets using
`visual_seed`; the same indices are used in the source and transported target
datasets. All reconstruction columns share the same inputs. All generation
columns share the same latent bank. `num_qualitative_images` controls examples
per reconstruction grid; optional `num_generation_images` controls the generation
grid independently and defaults to the same count. The selected model's eta, B,
dual rate and seed are shown on each figure. The runner saves these exact image
banks and latent vectors in `results/qualitative_examples.pt` for reproducibility.
The source VAE and ordinary LoRA comparison uses the selected model's seed.

Each run records `target_geometry_calibration.yaml`. TensorBoard includes the
two distortions (whichever views are enabled), mean worst distortion, source
tolerance, constraint violation, aggregate hinge and effective dual multiplier.
Every model checkpoint also has a `best.regularizer.pt` or
`last.regularizer.pt` sidecar with calibration, dual state and geometry sampler
state. Model checkpoints retain their existing format. As in the existing
ablation runner, incomplete runs restart from the source; sidecars do not add
automatic interrupted-training continuation.

Check that the constraint is active and gradients are useful before expanding
the sweep. The implementation's tests establish mathematical witnesses and
gradient plumbing; they do not establish target improvement, arbitrary-domain
feasibility, escape from exact collapse or rank-8 realizability.
