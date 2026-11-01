# Domain Knowledge Analysis for Generative Models

## LoRA Gram regularizer

In a LoRA YAML configuration, set the output Gram regularizer's trace normalization as follows:

```yaml
lora:
  regularizer:
    mode: output_gram
    trace_normalization: false
```

The default is `true`, including when the option is absent. Setting it to `false` compares the source and adapted Gram matrices without dividing each matrix by its trace, so the loss also reflects the magnitude of the responses. The input RMS scaling and averaging over output dimensions still apply. This can change the loss scale, so `eta` may need retuning. The option applies only to `output_gram`; `encoder_gram` already uses Gram matrices without trace normalization.

## Ablation FID

Run `python scripts/run_ablation.py` to train or reuse the ablation models and evaluate them. During training, the overall bar counts completed models (including reused checkpoints), and each worker has its own epoch bar. Detailed output is saved in each model directory as `training.log`. Ctrl+C stops the parallel workers. On a later run, models without both `best.pt` and the final `last.pt` are retrained from the source checkpoint; completed models are reused. The results include `test_target_fid` in `results/metrics.csv` and a plot in `figures/figure_9_target_fid.png`. Existing saved metrics are reused, and missing FID values are calculated from the saved checkpoints. The FID plot uses color for M and displays the selected model’s transformation strength s; ordinary LoRA and source-loss LoRA are shown separately.

FID compares generated images from each adapted decoder with the entire held-out rotated MNIST test set. It generates the same number of images as test examples, using the same prior latent samples for every model (`fid_seed` in the ablation config). Images are copied to three channels, and TorchMetrics resizes both sets to 299 × 299 for its standard 2048-feature Inception network. The network's weights are downloaded on first use. Inception runs in float32 on CUDA when available, with TF32 disabled; feature statistics and the FID calculation use float64. On Apple MPS, the decoder runs on MPS but FID runs on CPU. These float32 Inception scores can differ slightly from earlier scores calculated with float64 Inception.

The FID batch size is set separately as `fid_batch_size` (default 512). Reduce it if the GPU runs out of memory; the peak depends on available memory and CUDA libraries. CPU threads can be tuned for the entire run with `OMP_NUM_THREADS`, for example `OMP_NUM_THREADS=4 python scripts/run_ablation.py`. The Inception network and real-image statistics are reused across models, and real statistics are saved in `results/fid_reference` for later runs. The cache identifies the test images, weights, evaluation settings, FID-related code and library versions; changes to plots alone do not invalidate FID scores. Scores saved by the immediately preceding package-wide cache scheme remain reusable when their checkpoint and FID settings match. Saved scores also track the model checkpoint and FID seed. Older results without checkpoint fingerprints are reevaluated to ensure they match the current checkpoints.

Figures 1–4 express each model’s signed test reconstruction MSE change as a percentage of ordinary LoRA’s change for the same seed. Target gain is 100 × (source VAE target MSE − model target MSE) / (source VAE target MSE − ordinary LoRA target MSE). Source degradation is 100 × (model source MSE − source VAE source MSE) / (ordinary LoRA source MSE − source VAE source MSE). Ordinary LoRA therefore equals 100% for each metric when its change is nonzero. A zero denominator makes the percentage undefined, so the corresponding point is omitted. Negative reference changes reverse the direction of the percentage scale. The CSV and model selection retain the original absolute MSE differences.

Figure 1 has a pair of plots for each transformation strength. Figures 2 and 9 show the selected best-overall model’s strength, and figure 3 shows its regularization weight. If no best-overall model is eligible, they use the FID-eligible best-target-gain fallback and say so in their titles. Lines connect comparable runs within each seed; FID points are not connected.

An optional `ablation.local_sweep` defines an additional Cartesian grid of `eta_values`, `m_values`, and `s_values` for every configured seed. For example, the ellipse configuration compares η = 1, 5, 10, 20, 50, 100; M = 3, 5, 7, 10; and s = 0.75. Figure 10, `figures/figure_10_local_parameter_sweep.png`, shows one row of target MSE, source MSE, and target FID improvement plots per s. Each percentage is `100 × (ordinary LoRA score − regularized LoRA score) / ordinary LoRA score`, using test scores and the ordinary LoRA model for the same seed. Above zero means an improvement; a zero baseline makes the percentage undefined. Lines connect the same M and seed across η. Local-sweep models also participate in the other ablation plots and model selection.

The sweep uses the same model identities as the main grid. Overlapping runs reuse their existing completed checkpoints and compatible saved metrics and FIDs; only missing or invalidated results are computed. A changed training configuration still produces a different model identity. As a compatibility exception, an omitted `trace_normalization` and an explicit `true` both use the same default behavior, so either can reuse an existing completed run from the other form. Explicit `false` remains a different experiment.

Inception was trained on natural images, so these FID values are best interpreted as a relative comparison within this study rather than a definitive measure of image quality.

### Best-model selection

Automatic selection first excludes regularized models whose **test** target FID exceeds `max_fid_increase_fraction` above ordinary LoRA's FID for the same seed. The default is `0.12`: an ordinary LoRA FID of 20 permits FID values up to and including 22.4. Among the remaining models, best overall retains the existing validation target-gain requirement and minimizes validation source degradation. Best-target-gain and best-source-degradation summaries also use FID-eligible models. If none meets both rules, the FID-eligible best-target-gain model is used as a fallback. If none meets the FID limit, no model is selected: figures 2, 3, and 5–9 are not regenerated, and files from earlier runs may still be present. `results/best_models.yaml` records the thresholds, eligibility counts, and the actual selection.

You can choose a specific regularized model in the ablation YAML:

```yaml
ablation:
  max_fid_increase_fraction: 0.12
  best_model: {eta: 1000.0, m: 10, s: 0.1, seed: 42}
```

Set `best_model: null` for automatic selection. The override selects the specified existing run even when it fails the automatic FID or gain rules; the output records it as manual. `eta`, `m`, and `s` are required. Add `seed`, `mode`, `subset`, or `model_id` when needed to identify exactly one run; missing or ambiguous matches stop the run before model training. The filtered plots and qualitative figures use the manual model.

The FID criterion uses held-out test images, so choosing a model with it makes the test set part of model selection. Use a separate final evaluation set if an unbiased final test is needed.

## Lines datasets

To compare source images and training transformations, open `notebooks/visualize_transformed_dataset.ipynb` and set `DATASET` to `"mnist"`, `"lines_dataset"`, or `"vertical_lines_dataset"` in the settings cell. The same dataset choices are available in `notebooks/visualize_regularizer_strength.ipynb`; it reports any transformations that do not change the selected anchors and skips them during calibration.

Open `notebooks/visualize_lines_dataset.ipynb` to see the horizontal (`lines_dataset`) and vertical (`vertical_lines_dataset`) datasets. Each image is binary and 28 × 28 pixels, with a line 16–24 pixels long at a randomly sampled position. The underlying line angle is sampled uniformly from −7° to +7° around its orientation. Horizontal lines have a nominal thickness of 1–2 pixels and vertical lines 4–5 pixels. Pixel rounding produces small steps along tilted lines; adjacent steps are connected and line endpoints are kept clear.

The generator creates 5,000 training-pool images and 1,000 separate test draws per orientation. With the default 80/20 split, that gives 4,000 training and 1,000 validation examples. The draws and split are reproducible from `seed`. Some pixel patterns repeat within and across splits because the space of binary lines is small; the test set measures performance on new draws, not exclusively unseen pixel patterns.

### Train the lines source model and run the ablation

From the repository root, train the source VAE on horizontal lines:

```bash
./.venv/bin/python -c 'from domain_knowledge_analysis.experiments import Experiment; Experiment("config/vae_lines.yaml")'
```

This trains without running the scoring script and saves the best validation checkpoint under the `paths.run_dir` configured in `config/vae_lines.yaml`. Inspect the `Images/Fixed Latents`, `Images/Random Latents`, and `Images/Reconstructed Images` panels before committing to the full ablation:

```bash
./.venv/bin/tensorboard --logdir runs
```

Then run the lines ablation:

```bash
./.venv/bin/python scripts/run_ablation.py config/ablation_vae_lines.yaml
```

`config/lora_vae_lines.yaml` defines the source checkpoint and the 90° rotated horizontal target; that target keeps the horizontal lines' original thickness and is different from `vertical_lines_dataset`. Update `pretrained_model` in that file if you want to use a different source checkpoint. To train a single ordinary LoRA model, set `lora.regularizer` to `null`; restore the regularizer settings before running the ablation. The current ablation grid contains 36 regularized models and two baselines. It excludes contrast, gamma correction, binarization, and sharpening because they do not change the binary source images. The FID calculation uses the same Inception-based procedure as the MNIST ablation, requires pretrained weights on first use, and can be slow on CPU. Interpret FID cautiously for these simple images, especially with only 1,000 test examples.

The source run directory is fixed: retraining overwrites its checkpoints. Ablation results are reused by configuration and checkpoint path, not checkpoint contents, so choose a new `ablation.output_dir` before rerunning the study with a different source checkpoint at the same path. The model and training settings are a starting point; check generated samples as well as reconstruction quality before proceeding.

## Ellipsis dataset

`ellipsis_dataset` contains 28 × 28 grayscale ellipses with varying position, size, tilt, and thickness. Outer dimensions range from 10 × 5 to 22 × 11 pixels, with the same proportions. The slope is sampled uniformly from −5% to +5% (approximately −2.86° to +2.86°). Thickness ranges from 0.8 to 6 pixels; when it reaches the ellipse's vertical radius, the hole closes and the ellipse is filled. Every ellipse fits inside the image with a one-pixel margin. The outlines are drawn on a finer grid and averaged to produce smooth grayscale edges. The generator provides 5,000 training-pool images and 1,000 separately generated test images; the existing `train_split` divides the training pool. Some pixel patterns can repeat across splits.

Open `notebooks/visualize_ellipsis_dataset.ipynb` to see examples. In `notebooks/visualize_regularizer_strength.ipynb` or `notebooks/visualize_transformed_dataset.ipynb`, set `DATASET = "ellipsis_dataset"` to inspect its transformations.

To train the VAE on ellipses, run:

```bash
./.venv/bin/python -c 'from domain_knowledge_analysis.experiments import Experiment; Experiment("config/vae_ellipsis.yaml")'
```

The best checkpoint is saved at `runs/vae_ellipsis_source/checkpoints/best.pt`. After training, adapt it to 90-degree rotated ellipses with:

```bash
./.venv/bin/python -c 'from domain_knowledge_analysis.experiments import Experiment; Experiment("config/lora_vae_ellipsis.yaml").adapt()'
```

The LoRA configuration has its regularizer enabled for use by the ablation. To train a single ordinary LoRA model, set `lora.regularizer` to `null`, and restore it before running the ablation. The source run directory is fixed, so retraining overwrites its checkpoints. `ellipse_dataset` is also accepted as a dataset alias.

To run the ellipse ablation after training the source model:

```bash
./.venv/bin/python scripts/run_ablation.py config/ablation_vae_ellipsis.yaml
```

The ablation uses the same grid as the lines study, except it includes 20 transformations and tests M = 5, 10, and 20. The four additional intensity transformations act on the ellipses' grayscale pixels. Results are saved in `ablations/vae_ellipsis_rotated_ellipsis_output_gram`. Choose a new output directory when rerunning with a different source checkpoint at the same path, so that existing models and metrics are not reused.
