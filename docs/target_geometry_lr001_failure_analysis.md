# Investigation of the Thor target-only failure at Adam LR 0.01

Inspected on 2026-11-03. No training hyperparameters, objective, gradients or
optimizer formulas have been changed. The earlier clipping proposal is withdrawn
because it does not meet the requirement of preserving the training updates.

## Artifacts and observations

Both runs are under
`runs/vae_mnist_lr_0.001_29_jul_1606_B_beta_1.7_LD_16/`:

- Failed: `REG_adapt_vae_mnist_emnist_finetuned_on_emnist_03_nov_1116_target_geometry_target_only_eta_100.0_B_64_rank_8_alpha_4`.
  Inspected its TensorBoard events, `best.pt`, `best.regularizer.pt` and calibration.
- Comparison: `adapt_vae_mnist_emnist_finetuned_on_emnist_03_nov_1037_rank_8_alpha_4`.
  Inspected its events and ordinary LoRA checkpoints.

The failed run contains a best checkpoint at **epoch 32**, 33 completed logged
epochs, and no last/failure checkpoint. The supplied traceback therefore points
to a failure inside epoch 34. The displayed losses describe epoch 33, not the
batch that failed.

| Evidence | Observation |
| --- | --- |
| Epoch-32 model and Adam buffers | Every tensor finite |
| TensorBoard scalars, epochs 1–33 | Every logged value finite; these are epoch summaries |
| Last training/validation loss | 151.4697 / 152.4673 |
| Last training reconstruction / weighted KL | 118.2653 / 32.6337 |
| Measured source tolerance | 0.041664074087748304 |
| Last training / validation distortion | 0.0473716 / 0.0509632 |
| Dual multiplier | At its cap of 100 by epoch 18; bounded, not exploding |
| Adam LR | 0.01 throughout the failed run; scheduler inactive in this window |
| Ordinary LoRA comparison | Completes 160 epochs; LR remains 0.01 through epoch 67, then decreases |

The ordinary run establishes that LR 0.01 can train this architecture and
dataset without the geometry term. It does not establish stability after
adding that term. No mixed-precision training is configured in these files;
the failed checkpoint's model and Adam moment tensors are float32.

## What the exception establishes

For this target-only Bernoulli run, valid images and sigmoid reconstruction
means are in [0,1]. With B=64 and P=784, the squared Gram-difference numerator
is at most `4 * B**2 * P**2`. The checked input energy exceeds `1e-12`, so a
finite reconstruction gives a distortion below approximately `1.01e22`, well
inside float32 range. The actual implementation checks that input energy
before checking the distortion.

Consequently, this exception establishes that **NaNs reached the target
reconstruction/geometry path**. It does not identify the first bad operation.
An earlier invalid VAE/geometry gradient can corrupt Adam and the parameters;
the next geometry forward can then be where the damage is detected. A bad
activation in the current forward is another possibility. Epoch averages and
an earlier finite checkpoint cannot distinguish these cases.

The earlier synthetic KL-overflow reproduction demonstrates a possible causal
chain, not the recovered cause of this Thor failure. In particular, there is
no evidence in the pulled checkpoint that its posterior variance is already
near overflow.

## Direct probes of the saved model

Local CPU probes used the actual epoch-32 model, saved Adam state and saved
geometry-sampler state. EMNIST tensors used the same class filtering, split,
pixel scaling and orientation correction as the training loader.

1. Scanned all 38,400 target training and 9,600 validation images. Posterior
   means, log-variances and KLs were all finite. The largest log-variance was
   **0.448685**; float32 exponential overflow is around 88.72. Maximum absolute
   posterior mean was 5.64852. Maximum per-image, unweighted KL was 43.1015.
2. On 64 independently drawn batch pairs, separately differentiated the VAE
   loss and `lambda * distortion`, with the saved lambda=100. Median gradient
   norms were 75.98 and 31.99 respectively. Their median cosine was **−0.917**;
   every sampled pair opposed one another. Median combined norm was 50.72.
   The geometry term materially changes optimization even though its scalar
   loss contribution is small. This observation does not prove divergence.
3. Continued the saved model for **900 updates** (three target-data epochs)
   using unchanged Adam at fixed LR 0.01, target-only eta=100, B=64 and dual
   LR=0.01. All objectives and gradients remained finite. Largest observed
   log-variance was 0.723855. Combined gradient norm had median 42.07, 95th
   percentile 106.26, 99th percentile 134.44 and maximum 189.16.

This continuation is a controlled probe, **not an exact replay of Thor**:
`best.pt` lacks the original global CPU/CUDA RNG and loader state, its epoch-33
updates are missing, and local CPU kernels differ from CUDA. It neither
reproduces the original failure nor proves that a 160-epoch GPU rerun succeeds.
The artifacts do not support an exact historical minibatch diagnosis.

## Division audit of the actual failure window

The saved geometry generator is a **private CPU generator**, even on Thor:
EMNIST's orientation transform keeps it on the ordinary dataset sampling path.
Starting from seed 27043, exactly `32 * 300` draws of `randperm(38400)` reproduce
the saved epoch-32 sampler state **byte for byte**. This verifies the local
sampling implementation against the saved run, rather than assuming a matching
seed is enough.

Starting at that state, the next **600 geometry batches** cover all of epoch 33
and all of epoch 34, including the failing batch. For these reconstructed real
input batches, float32 CPU Gram energy ranges from **612530.125 to 1098764.0**.
Every denominator is finite and positive, more than 17 orders of magnitude
above `energy_floor=1e-12`. These are recomputed input energies, not recorded
CUDA activation values. Reconstructing the input batches does not reconstruct
the missing adapted weights or posterior noise at failure.

An audit of the complete primal autograd graph also finds only one explicit
`DivBackward0`: the geometry ratio, with a denominator of 895515.375 on the
first batch after the saved state. Its denominator has `requires_grad=False`.
In particular, its backward does **not** differentiate through a denominator
that could depend on collapsed reconstructions.

| Division or potentially singular operation | Actual path |
| --- | --- |
| Geometry ratio | Divide by detached, checked **real-input** Gram energy |
| Batch centering | Mean over the fixed 64 batch members |
| VAE averaging | Mean over 128 examples; no empty minibatch |
| LoRA scaling | Fixed `alpha/rank = 4/8` |
| KL | `exp(log_variance)` and squares; no variance division |
| Bernoulli likelihood | `logsigmoid` of logits; no probability division |
| Adam | `sqrt(v / (1-beta2**t)) + 1e-8`; strictly positive for finite nonnegative v |
| Gaussian log-density's variance division | This function is **not called** by this VAE training loss |

The saved Adam denominators range from **5.798881466e-5 to 44.51733780**;
none is zero or nonfinite. These observations rule out zero input geometry as
the failure mechanism. They do not rule out an earlier overflow, nonfinite
backward operation, corrupted optimizer state or backend failure.

Writing `Y = center(output_images)`, `R = Y @ Y.T - G_input`, and
`E = sum(G_input**2)`, the output gradient before undoing centering is exactly
`4 * R @ Y / E`. Perfect reconstruction gives R=0, and collapsed output gives
Y=0. Neither creates a division by zero. Tests exercise perfect reconstruction,
collapse, saturated sigmoid logits and input energy just above the guard.

Adding epsilon to this already large denominator would change the objective
and would not repair NaNs that already entered the decoder. No such change has
been made. Float32 spacing at these energies is 0.0625–0.125; adding a typical
epsilon such as `1e-8` is rounded away and literally changes nothing. There is
no evidenced zero division here to stabilize.

## Locating the first invalid operation without altering updates

The target trainer now captures the RNG state **before** each stochastic VAE
forward and retains the pre-optimizer parameters and moments in reusable
device buffers. Checks immediately after Adam catch invalid moment/parameter
updates in the same batch, before dual ascent or another network forward.
The normal loss, backward, Adam step and dual formulas remain untouched.
CUDA tensors stay on CUDA; snapshots use device-to-device copies.

On a failure only, a copied computation is replayed with the original
pre-forward RNG. ATen operation tracing examines forward and backward outputs,
including divisions, while autograd records the forward origin of a failing
backward. If the optimizer caused the failure, it is replayed separately from
its captured pre-step parameters/moments and the original gradients. The trace
reports the first nonfinite operator and its input/output ranges, with phases
`vae_forward`, `geometry_forward`, `primal_backward` or `optimizer_step`.

`checkpoints/nonfinite.pt` includes `report.operation_trace`,
`pre_forward_rng_state`, and, for an optimizer failure, `pre_optimizer_state`.
Live gradients, model state and RNG are not changed by the forensic replay.
If the bad operation does not reproduce, the report explicitly says
`replay_finite`; it does not invent a diagnosis. GPU nondeterminism/backend
faults can limit replay, even with the same stochastic inputs.

Tests deliberately inject a backward zero division and a zero-epsilon Adam
division to verify that the tracer locates them, then verify that healthy
updates, optimizer state, dual state and sampling remain equivalent to the
original implementation, including LR 0.01. These synthetic cases are not
claimed to be the cause of the Thor run.

An exact fix must be derived from the first invalid operation. The old saved
model predates that event and cannot supply the missing operation inputs.
The added provenance makes an unchanged-parameter rerun informative; it is
not itself a numerical repair or a claim that LR 0.01 is now guaranteed stable.
