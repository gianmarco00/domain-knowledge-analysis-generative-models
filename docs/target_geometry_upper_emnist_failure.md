# Upper-EMNIST LR=0.01: confirmed variance overflow and a numerical proposal

The failure snapshot supplied on 2026-11-03 identifies the actual first bad
operation. This document supersedes speculation about the cause of this run;
it does not establish the cause of the earlier EMNIST run without a snapshot.
Production training code and all training/regularizer settings are unchanged
by this investigation. The proposed arithmetic is an **offline prototype**.

## Confirmed failure

Run: `target_geometry_full_seed_42_eta_10_B_64_3e0ea5bee6`, full view, eta=10,
B=64, dual LR=0.1, Adam LR=0.01. Failure: epoch 10, batch 368, after 5380
completed optimizer updates. Thor used PyTorch 2.14.0+cu130 / CUDA 13.0.

The recorded ATen trace stops at `models/vae/vae.py:38`:

```python
std = torch.exp(0.5 * log_variance)
```

| Captured quantity | Actual value |
| --- | ---: |
| Largest posterior log-variance | 448.25634765625 |
| Largest input to the first bad exponential | 224.128173828125 |
| Correct standard deviation at that input | 2.1758509e97 |
| Maximum finite float32 | approximately 3.4028235e38 |
| Infinite sampled-standard-deviation entries | 284 / 2048 |
| Infinite posterior-variance entries | 997 / 2048 |
| NaN decoder-logit entries | 45774 / 100352 |
| Real-input geometry denominator | 1165980.875 |
| Target / source-chart distortion | 0.7901147 / 0.7289440 |
| Nonfinite model parameters / Adam buffers | none / none |

Thus the chain is **excessive posterior log-variance -> overflowing standard
deviation -> nonfinite sampled latents/decoder activations -> NaN reconstruction
loss**, with the KL variance exponential also overflowing. The guard stopped
before this batch could corrupt Adam with its invalid loss/gradient.

The deterministic geometry uses posterior means, so it remains finite despite
the variance explosion. Its nonzero denominator is not implicated. Adding an
epsilon there cannot repair this failure; clamping the variance would change
the VAE objective/model and is not proposed.

## Evidence about how the bad state arose

CPU FP32 and FP64 evaluation of the saved encoder on the failing images both
produce a maximum log-variance near **448.23**. The extreme output persists
without concurrent models or CUDA, and is not merely FP32 exponential rounding.
This does not prove concurrency had no effect on the earlier trajectory.

The captured CPU RNG permits recovery of the epoch's shuffle seed. Rewinding
the two MT19937 words used for RandomSampler's seed draw recovers
7169526598771353922. Replaying that seed draw reproduces the saved RNG exactly;
the resulting permutation reproduces **every pixel of the failing batch**.
The two preceding input batches can therefore also be identified.

The last Adam update can be approximately inverted using its saved moments:
`theta_previous = theta_current + lr*m_hat/(sqrt(v_hat)+eps)`. FP32 rounding
prevents bitwise recovery of the old weights. The affected encoder moments also
have an unusually precise fingerprint: `m**2/v` is about **8.108108**, the value
for one dominant gradient spike followed by one ordinary momentum-decay step:

```
(.1**2 / .001) * (.9**2 / .999) = 8.108108...
```

Using that fingerprint to estimate the preceding moments gives this supported,
but approximate, reconstruction:

| Update / input batch | Posterior/gradient evidence |
| --- | --- |
| Before update 5379, epoch-10 batch 366 | Max log-variance approximately 36.74; weighted mean KL approximately 7.62e13, huge but still finite |
| Before update 5380, batch 367 | Max log-variance approximately 0.289; weighted mean KL approximately 31.22 |
| After update 5380, before failing batch 368 | Captured max log-variance 448.256; last parameter update is at most approximately 0.03474 per coordinate |

The KL gradients reconstructed for the spike batch predict the saved encoder
first moments with cosine similarity above **0.999999** and approximately
0.6% relative error. This strongly supports a finite KL-gradient spike followed
by **Adam momentum overshoot**, rather than a zero division or silently shared
model state. The reconstruction is not a claim to possess the original
intermediate CUDA tensors or exact previous gradients.

## The ordinary LoRA baseline fails too

The pulled events for
`target_geometry_ordinary_lora_seed_42_eta_0_B_0_fecb8f16bb` show finite training
loss 152.9097 at epoch 11, followed by NaN training, validation, reconstruction
and KL losses from **epoch 12** through its last logged epoch 16. This model has
**no geometry regularizer**. Its first bad operation was not captured by the
original generic Trainer, so its exact mechanism cannot be assumed identical.

The regularized failure run has finite epoch-9 training/validation losses
154.8003/155.1959 before the within-epoch event. Other workers had only reached
approximately nine completed epochs before the parent terminated the pool in
response to this exception. Their event files do not establish 160-epoch
stability. The supplied failure is a numerical check, not a reported CUDA OOM.

## A proposal that preserves the mathematical equations

Plain FP64 is insufficient for this captured state: evaluating the same
posterior in FP64 gives weighted mean KL approximately **3.14e192**. A complete
CPU FP64 primal/backward probe using the saved weights has a largest parameter
gradient approximately **6.32e195**. These gradients are finite, but squaring
them overflows FP64 Adam's `exp_avg_sq`.

A numerical range extension therefore needs both:

1. FP64 forward/backward where the posterior, sampled latents and gradients
   exceed FP32's representable range, without clamping or resampling the batch.
2. The **same Adam equations**, storing `r = sqrt(v)` and using a stable hypot
   operation instead of explicitly materializing squared huge gradients:

```
m_t = beta1*m_previous + (1-beta1)*g_t
r_t = hypot(sqrt(beta2)*r_previous, sqrt(1-beta2)*g_t)
theta_t = theta_previous - lr * (m_t / (1-beta1**t))
                          / (r_t / sqrt(1-beta2**t) + eps)
```

Squaring the `r_t` identity recovers exactly
`v_t = beta2*v_previous + (1-beta2)*g_t**2`. No LR, beta, epsilon, regularizer
coefficient, batch, or dual-update rule is changed. No gradient clipping,
reweighting or resampling is added. This changes numerical representation, not the
mathematical optimizer. It is **not bitwise identical** floating-point arithmetic.
The reference equations are in the [PyTorch Adam documentation](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html).

The offline prototype is in `scripts/analyze_target_geometry_failure.py` and
is not connected to `create_optimizer`, Trainer, Experiment or ablation configs.
Any production integration would need explicit numeric-policy metadata/cache
identity, compatible checkpoint handling, original-noise preservation, GPU
execution and the same policy for ordinary/full/target-only comparisons.

## What was tested, and what remains unresolved

On the captured weights/images, with fixed CPU FP32 Gaussian noise promoted to
FP64 (a controlled probe, **not the original CUDA draw**):

| Arithmetic | Forward/backward | Optimizer state |
| --- | --- | --- |
| FP32 | Overflows as captured | No invalid update applied |
| FP64 + ordinary Adam | Finite primal and gradients | Squared-gradient moments overflow in 12 encoder tensors |
| FP64 + hypot/RMS representation of Adam | Finite | Finite; next same-batch max log-variance approximately 0.643 |

The representation agrees with ordinary FP64 Adam in normal range to about
1e-14 over 1000 supplied-gradient updates. Tests also verify conversion from
existing Adam moments, unchanged gradients, step counters, and a finite exact
first update for gradients of magnitude 1e195 that break ordinary FP64 moments.

**This numerical repair does not remove the underlying optimization instability.**
Twenty further controlled updates on the same batch remained finite, but their
loss rose to approximately **1.37e117**. Repeated-batch probes are not a full
training experiment or a convergence test. The evidence therefore supports a
math-preserving way to evaluate this particular oversized Adam step, **not a
guarantee of stable, useful 160-epoch training at LR=0.01**. Removing actual
momentum overshoot while requiring every mathematical update to remain the
same is not possible; numerical representation and optimization stability are
separate issues.

Reproduce the offline investigation from the repository root:

```bash
python scripts/analyze_target_geometry_failure.py /path/to/nonfinite.pt \
  --recover-history --precision-probe --controlled-updates 20 \
  --output /tmp/upper_emnist_failure_analysis.json
python -m pytest tests/test_failure_precision_prototype.py -q
```

The optional `--device cuda` regenerates FP32 noise from the captured CUDA RNG
before promoting it, without changing the live default generator. Kernel and
version compatibility still affect replay. No CUDA execution was available for
this local investigation. The actual first-operation diagnosis comes directly
from the original recorded Thor tensors and trace, not a local CUDA claim.
