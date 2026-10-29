# Domain Knowledge Analysis for Generative Models

## Ablation FID

Run `python scripts/run_ablation.py` to train or reuse the ablation models and evaluate them. The results include `test_target_fid` in `results/metrics.csv` and a plot in `figures/figure_9_target_fid.png`. Existing saved metrics are reused, and missing FID values are calculated from the saved checkpoints. The FID plot uses color for M and marker shape for transformation strength s; ordinary LoRA and source-loss LoRA are shown separately.

FID compares generated images from each adapted decoder with the entire held-out rotated MNIST test set. It generates the same number of images as test examples, using the same prior latent samples for every model (`fid_seed` in the ablation config). Images are copied to three channels, and TorchMetrics resizes both sets to 299 × 299 for its standard 2048-feature Inception network. The network's weights are downloaded on first use. FID is computed in double precision on CPU or CUDA; on Apple MPS, the decoder still runs on MPS but FID runs on CPU. This can take considerable time for the full ablation.

Inception was trained on natural images, so these FID values are best interpreted as a relative comparison within this study rather than a definitive measure of MNIST image quality. Model selection still uses the validation reconstruction metrics.
