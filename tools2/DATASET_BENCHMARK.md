# Final FL benchmark dataset

The final thesis benchmark uses **cleaned real RuralIoT measurements**.
Synthetic/VAE data remains an optional platform capability but is not the source
of the default aggregation-method benchmark.

## Build

```bash
cd tools2
python3 05_prepare_fl_dataset.py --force
python3 06_validate_fl_dataset.py
```

Output: `data/fl_dataset_real/`.

## Method

1. Read `data/df_RuralIoT_*_CLEANED.csv`.
2. Find the longest complete, contiguous 10-minute interval for each sensor.
3. Centre-crop 960 real rows per client (about 6.7 days), giving all clients the
   same dataset size while avoiding interpolation across long acquisition gaps.
4. Chronologically split 70% train / 15% validation / 15% test.
5. Inject controlled, finite fault episodes **inside each split separately**.
6. Use an explicit injection mask as ground truth (`label`), not a floating-point
   difference threshold.
7. Locally forward-fill injected packet-loss gaps so neural networks receive no
   NaNs while the label still identifies the corrupted measurements.

Default class balance is 25% anomaly / 75% normal in every split for every
client.  Client non-IID heterogeneity comes from the underlying real sensor
distribution and the assigned fault profile:

- 001: temperature drift episodes
- 002: temperature bias episodes
- 003: single-feature packet-loss episodes
- 010: two-feature packet-loss episodes
- 21: burst-noise episodes
- 22: humidity flatline episodes
- 23: mixed flatline/dropout episodes

Each client directory contains `dataset_manifest.json`; the dataset root contains
`dataset_manifest.json` and `dataset_summary.csv` for reproducibility.

## Legacy synthetic path

The previous synthetic pipeline is preserved:

- `03_synthetic_generation.py`
- `04_error_injector.py`
- `05_prepare_fl_dataset_synthetic_legacy.py`
- existing `data/fl_dataset/`

It may still be used for platform demonstrations, but should not be the sole
source of final thesis claims about real measurement data.
