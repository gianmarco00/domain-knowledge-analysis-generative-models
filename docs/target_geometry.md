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
run records its calibration, reusing an identical cached calibration on CUDA.
MNIST retains your existing 45°
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

Every seed × eta × geometry batch size × dual learning rate combination
trains **both full and target-only** models. One ordinary LoRA baseline per seed
and the explicitly configured source-only controls are retained. Matching
controls deduplicate by training ID; off-grid full/target-only controls are also
paired. The current YAML files produce 146 runs for rotated MNIST, 38 for
ellipses, and 146 for MNIST→EMNIST. Source calibration and sampling streams are
shared across matched modes; each mode keeps its own model and dual state.

Existing full models, baselines and the former single target-only control keep
their training IDs, config paths and checkpoint directories. Rerunning trains
only missing models, reuses valid cached scores/FID, and recalculates all figures.
VAE/LoRA architecture, loss, training settings and the regularizer's math are
unchanged by the expanded grid. Parallelism remains an execution setting in
`parallel_models`.

Qualitative selection now has four independent roles:

- Full reconstructions: lowest **validation target MSE**, with lower test FID
  breaking an exact tie.
- Target-only reconstructions and the comparison star in figures 1–2: maximize
  the equal-weight mean of **test target MSE improvement (%)** and **test target
  FID improvement (%)**, each relative to ordinary LoRA with the same seed.
  Ties prefer lower absolute FID, then lower target MSE, then a stable model ID.
- Full generations: lowest **test target FID** among full models.
- Target-only generations: lowest **test target FID** among target-only models.

Both generation selections break FID ties with lower test target MSE. The
balanced target-only score is `(target_MSE_improvement + FID_improvement) / 2`;
it does not maximize raw error. The selected reconstruction and generation
models can differ, and each figure identifies the model it actually displays.
`results/selected_models.yaml` saves all four choices, their rules and the
balanced score. `selected_model.yaml` retains the full reconstruction selection
for compatibility. Across-seed mean/std metrics remain in `results/summary.csv`.

## Result figures

Each ablation generates ten figures in `figures/`:

- `figure_1_eta_all.png`: full-mode grid with one row per dual learning rate
  and target MSE, source MSE and target FID columns. Source-only controls remain
  visible; the single balanced target-only selection is marked by a star at
  its actually measured eta/LR, rather than copying it into other LR panels.
- `figure_2_eta_zoom.png`: the corresponding automatic zoom.
- `figure_3_target_only_eta_all.png`: the entire target-only grid with the same
  normalization, LR layout, batch-size colors and individual seed trajectories.
- `figure_4_target_only_eta_zoom.png`: its automatic zoom, using the exact same
  window, eta selection, clipping and dotted-connection rules.
- `figure_5_source_reconstruction_full.png`
- `figure_6_target_reconstruction_full.png`
- `figure_7_generations_full.png`
- `figure_8_source_reconstruction_target_only.png`
- `figure_9_target_reconstruction_target_only.png`
- `figure_10_generations_target_only.png`

The two reconstruction figures for each mode use that mode's reconstruction
selection. Generation figures 7 and 10 independently use that mode's lowest-FID
model. Every qualitative grid compares source VAE, same-seed ordinary LoRA and
the displayed selected model. Reconstruction grids also show the real input.

The eta plots match the output-Gram local-sweep convention:

```
improvement (%) = 100 * (ordinary_LoRA_score - model_score) / ordinary_LoRA_score
score = test target MSE, test source MSE, or test target FID
```

Unregularized LoRA is **0%**, positive is better, and negative is worse. Each
point is normalized against ordinary LoRA with the **same seed**. Color identifies
geometry batch size B; marker/line style identifies the decoder view. Individual
seed trajectories remain separate, as in the output-Gram parameter sweep.
A nonpositive/nonfinite baseline score is rejected because the requested
percentage is undefined; no artificial denominator or alternative metric is used.
Source-only controls and the selected target-only comparison appear only at their actually evaluated dual learning rates.

For each metric, the zoom first averages improvements across seeds for every
(rate, view, B, eta) setting and finds the best mean. Its symmetric half-range
is `abs(best_mean_improvement) + 5` **percentage points**, shared across LR rows
for consistent comparison. For example, FID 20 reduced to 17 is a 15% improvement,
so the range is -20% to +20%, equivalent to raw FID 16–24. If every setting is
worse than LoRA, the least-bad mean and the same padding determine the range;
flat scores produce a minimum range of ±5 percentage points.

Eta selection is independent for each (LR, metric) panel: an eta is retained
iff at least one individual sample, including controls, lies inside or exactly
on that panel's limits. Etas with every sample outside are removed from the
points and ticks. At retained etas, off-scale samples use outward triangles
at the boundary, and their connecting segments are dotted in the original
batch-size color. A connection that skips an omitted, off-scale eta is also
dotted. Dots and boundary markers indicate clipped display values; the CSV
scores and the all-model figure retain the full values.

Rerunning the normal ablation skips completed best+last checkpoints and reuses
valid cached metrics/FID; plot-only code changes do not change model IDs or the
FID scoring fingerprint. To regenerate figures entirely from saved metrics
and cached qualitative banks, with no model loading, training or scoring:

```bash
python scripts/run_ablation.py config/ablation_vae_mnist_target_geometry.yaml --plots-only
python scripts/run_ablation.py config/ablation_vae_ellipsis_target_geometry.yaml --plots-only
```

Plot-only mode uses all rows already saved in that output directory, regardless
of later sweep YAML edits. The new independent reconstruction/generation
selections require refreshed image banks: run the normal ablation once after
upgrading, then use `--plots-only` for later style changes. Normal reruns reuse
completed models and train only the new target-only grid entries. A missing or
stale bank fails explicitly instead of displaying a different checkpoint's
images. Obsolete figure names are removed after their replacements render.

Qualitative inputs are sampled reproducibly with `visual_seed`. Both modes use
**exactly the same source images, target images and prior latent vectors**, even
if their selected models have different training seeds. Each displayed model
uses the ordinary LoRA baseline with its own training seed. The shared prior
bank follows N(0,I); reconstructions use posterior means. `num_qualitative_images`
sets reconstruction examples and `num_generation_images` optionally sets the
number of generations (default: the same count).

`results/qualitative_examples_full.pt` and
`results/qualitative_examples_target_only.pt` save these exact tensors plus
separate reconstruction/generation selections and baseline metadata. Matching
banks are reused on normal reruns as well as plot-only runs. Checkpoint IDs and
hashes for both roles, baseline hashes, visual seed/counts, and shared input
banks are checked before reuse. Image creation loads each needed model once
per mode and computes only the outputs belonging to its selected role.

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

## Nonfinite training failures

The pulled Thor events and checkpoint have now been inspected. See
[the LR=0.01 investigation](target_geometry_lr001_failure_analysis.md) for the
measured evidence, reconstructed failure-window denominator audit and operation
tracing. The finite epoch-32 checkpoint and successful local
continuation do not establish the exact cause of the epoch-34 GPU failure.

For the MNIST→EMNIST Bernoulli configuration, a `Nonfinite target geometry
distortion` error with valid input energy means the adapted reconstruction
contains NaNs. Real inputs and sigmoid decoder means lie in [0,1]. With B=64
and 784 pixels, the numerator is bounded by `4 * B**2 * 784**2`; divided by
energy strictly above `1e-12`, the distortion is below about `1.01e22`, far below
the float32 limit. Finite normalized image means therefore cannot explain that
failure through Gram overflow. The failing run's actual tensors are needed to
identify the first nonfinite network operation.

One reproducible upstream mechanism is ordinary VAE numerical divergence:
`KL = .5 * sum(exp(log_variance) + mean**2 - 1 - log_variance)`. In float32,
`exp(log_variance)` overflows around 88.72, while posterior means and the
deterministic geometry can still be finite. Previously the geometry-only
check allowed an invalid VAE loss/gradient to reach Adam; a subsequent batch
then failed the geometry check because its weights/outputs were already NaN.
A finite loss can also have nonfinite backward gradients, so forward checks
alone are insufficient. Tests reproduce this causal chain; they do not claim
to recover the missing intermediate tensors from the reported Thor run.

The target-geometry trainer now checks the actual VAE and primal objectives,
both geometry values, and every computed trainable gradient **before** the
optimizer/dual update. Healthy updates retain their existing arithmetic.
The guard checks elementwise finiteness, not a float32 gradient norm that could
overflow for otherwise finite gradients. It does not clip gradients, replace
NaNs, change Gram denominators, skip batches or alter eta, q, c_s or the loss.
Gradient checks run on the tensor device and transfer only a scalar validity
flag on the healthy path; detailed CPU diagnostics run only after a failure.
Parameters and Adam state are also checked immediately after the step, before
dual ascent. Reusable device buffers preserve the pre-step values for replay.

On failure, `checkpoints/nonfinite.pt` separately records model/source-decoder
weights, gradients, optimizer and regularizer states, the failing image batches,
actual VAE forward outputs, RNG/sampler state and a `report`. The report names
the stage, epoch/batch, nonfinite parameters/Adam state, posterior variance,
decoder logits, image means and first nonfinite geometry activation. Existing
best/last checkpoints are preserved. This is forensic state, not an automatic
resume checkpoint. Pre-forward RNG state permits replay of the stochastic VAE
pass on copied modules. Failure-only ATen tracing reports the first nonfinite
forward/backward operation and its inputs. An optimizer failure instead replays
Adam from the captured pre-step values. Replays preserve the live RNG, model
and gradients and explicitly report when a failure does not reproduce.

The reported run used Adam LR **0.01**, ten times the original MNIST→EMNIST
YAML's **0.001**, and failed before its scheduler starts reducing LR after
epoch 50. The standalone target-only configuration retains the requested
**0.01** learning rate. Run it from the original finite source checkpoint:

```bash
python scripts/run_target_geometry.py config/lora_vae_mnist_emnist_target_only.yaml
```

It retains eta=100, B=64, dual LR=0.01, both calibration/validation counts and
the complete target-only regularizer, using model Adam LR=0.01. The checks
detect invalid objectives/gradients before the update; they do not stabilize
Adam, automatically lower its learning rate, or guarantee that the run will
complete. When all values are finite, the loss, gradients, Adam state, dual
update and sampler follow the original update rules, including at LR=0.01.
An unstable run still stops and preserves its failure evidence. No clipping,
loss/variance clamping or denominator epsilon has been introduced. The loss
numbers on a progress bar describe the previous
completed epoch, not the failing minibatch.

## Performance and equivalence audit

The target-geometry trainer has execution optimizations that preserve its
objective, sampler and ordinary VAE updates:

- On CPU, geometry batches backed by an ordinary `TensorDataset` (including nested
  `Subset` views) use indexed tensor gathering. They use the **same CPU randperm
  calls and generator state** as before. Arbitrary datasets and subclasses with
  custom `__getitem__` behavior retain their original access path; transforms
  are not cached or bypassed.
- Both real-input geometry denominators are the same tensor, computed and
  checked once. Both output distortions are still computed and checked for
  finiteness on every geometry batch.
- The trainer sends its scalar diagnostics to the host in one transfer, rather
  than separate reads for every loss component and distortion. Accumulation and
  the bounded dual update retain their original Python-double arithmetic.
- With a strict maximum, the regularizer's losing decoder branch has exactly
  zero gradient. The trainer backpropagates through the winning branch and
  releases the losing diagnostic graph. **Exact ties retain the original
  `Tensor.max` equal-share subgradient.** The ordinary VAE loss still reaches
  all adapted encoder and decoder parameters, retaining Adam's ordinary
  momentum updates even for the decoder whose regularizer view loses.
- When the dual multiplier is exactly zero, both views are evaluated for
  monitoring and dual ascent without building an unused geometry autograd
  graph. The next dual update still uses that batch's measured distortion.

On CUDA, the target-geometry experiment additionally keeps the MNIST/ellipse
image tensors in GPU memory. VAE minibatches use batched tensor indexing instead
of per-image Python collation; geometry permutations and gathers also run on
the GPU. The final partial VAE batch is retained. Held-out source MNIST's plain
`ToTensor` conversion is vectorized on the GPU. Arbitrary/custom dataset
transforms keep their original access path. An image-content-keyed cache reuses
up to 512 MiB of resident image tensors per ablation worker across runs.

Source calibration is cached under `.cache/target_geometry/calibration/` and
shared across workers with a file lock and atomic publication. Keys include
the checkpoint contents, held-out images, source config, sampling seed, B,
calibration count, energy floor, model precision, implementation and execution
backend. Eta, dual learning rate and view reuse this calibration because they
do not enter its calculation. Changed source weights/data/B/precision/backend
force recalibration. No adapted codes, outputs, losses or gradients are cached.

CUDA validation reuses its fixed image bank and checked input Grams/energies,
recomputes both model views each epoch, and accumulates metrics in device
float64 before one host transfer. Training combines output-finiteness checks
with the scalar transfer already needed for the dual update and active-view
selection. Input-energy checks and fail-before-update behavior are retained.
The dual update keeps Python-double arithmetic; model precision is unchanged.

CUDA generators retain uniform sampling without replacement and independent
training/calibration/validation streams. Their seeded sequences differ from
CPU generators; matched CUDA sweep runs still use the same streams. Existing
model IDs, completed-run detection, the ordinary LoRA baseline and the original
regularizer/trainer paths are unchanged. These are execution optimizations,
not CUDA speedup measurements; no AMP/TF32 policy or mathematical setting is
changed.

These paths introduce no new mathematical hyperparameters. Precision,
calibration/validation batch counts, target sampling, source tolerance, eta,
the dual learning rate and the likelihood/KL objective are unchanged. The
original transformation regularizer and trainer are not modified.

See [the measured CPU report and raw operator profiles](benchmarks/target_geometry_cpu/README.md).
The profiling script uses the actual source checkpoints, full target train and
validation datasets, and writes timing JSON plus PyTorch operator tables.
It can compare a historical Git implementation with the working tree:

```bash
python scripts/profile_target_geometry.py --reference-ref 775c21b \
  --device cpu --threads 4 --steps 40 --epochs 1 \
  --output /tmp/target_geometry_profile
```

To audit actual checkpoint-model parameters, gradients, Adam state, losses,
dual state and geometry sampler state after coupled noisy optimizer steps:

```bash
python scripts/profile_target_geometry.py --reference-ref 775c21b \
  --audit-only --audit-steps 5 --output /tmp/target_geometry_equivalence
```

Run timings on the device where the ablation will train. CPU timings do not
establish GPU speedups; device-to-host synchronization is especially
hardware-dependent. The benchmark excludes FID, figure generation and
TensorBoard/checkpoint I/O from epoch timings, so it isolates training and
validation rather than claiming an end-to-end ablation duration.
