# Domain Knowledge Analysis for Generative Models

## Ablation FID

Run `python scripts/run_ablation.py` to train or reuse the ablation models and evaluate them. During training, the overall bar counts completed models (including reused checkpoints), and each worker has its own epoch bar. Detailed output is saved in each model directory as `training.log`. Ctrl+C stops the parallel workers. On a later run, models without both `best.pt` and the final `last.pt` are retrained from the source checkpoint; completed models are reused. The results include `test_target_fid` in `results/metrics.csv` and a plot in `figures/figure_9_target_fid.png`. Existing saved metrics are reused, and missing FID values are calculated from the saved checkpoints. The FID plot uses color for M and marker shape for transformation strength s; ordinary LoRA and source-loss LoRA are shown separately.

FID compares generated images from each adapted decoder with the entire held-out rotated MNIST test set. It generates the same number of images as test examples, using the same prior latent samples for every model (`fid_seed` in the ablation config). Images are copied to three channels, and TorchMetrics resizes both sets to 299 × 299 for its standard 2048-feature Inception network. The network's weights are downloaded on first use. Inception runs in float32 on CUDA when available, with TF32 disabled; feature statistics and the FID calculation use float64. On Apple MPS, the decoder runs on MPS but FID runs on CPU. These float32 Inception scores can differ slightly from earlier scores calculated with float64 Inception.

The FID batch size is set separately as `fid_batch_size` (default 512). Reduce it if the GPU runs out of memory; the peak depends on available memory and CUDA libraries. CPU threads can be tuned for the entire run with `OMP_NUM_THREADS`, for example `OMP_NUM_THREADS=4 python scripts/run_ablation.py`. The Inception network and real-image statistics are reused across models, and real statistics are saved in `results/fid_reference` for later runs. The cache identifies the test images, weights, evaluation settings, code and library versions. Saved scores also track the model checkpoint and FID seed. Older results without checkpoint fingerprints are reevaluated to ensure they match the current checkpoints.

Inception was trained on natural images, so these FID values are best interpreted as a relative comparison within this study rather than a definitive measure of MNIST image quality. Model selection still uses the validation reconstruction metrics.

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
