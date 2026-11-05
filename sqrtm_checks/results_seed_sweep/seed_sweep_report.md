# Seed sweep: R8 vs E34 vs G34, 10 independent trainings

## Test MSE

| Model | Test MSE (2000), mean ± SD | Min – max | Median | Test MSE (300), mean ± SD |
|---|---|---|---|---|
| R8 (5000 epochs) | 6.306e-05 ± 3.109e-05 | 3.143e-05 – 1.367e-04 | 5.987e-05 | — |
| R8 (10000 epochs) | 1.825e-05 ± 4.953e-06 | 1.131e-05 – 2.424e-05 | 2.000e-05 | 1.612e-05 ± 3.707e-06 |
| E34 after phase 1 | 1.087e-05 ± 2.195e-06 | 8.291e-06 – 1.450e-05 | 1.079e-05 | — |
| E34 + refit | 1.488e-05 ± 8.269e-06 | 3.780e-06 – 2.929e-05 | 1.208e-05 | 1.114e-05 ± 7.243e-06 |
| G34 after phase 1 | 2.303e-05 ± 3.603e-06 | 1.656e-05 – 2.880e-05 | 2.275e-05 | — |
| G34 + refit | 1.159e-05 ± 5.565e-06 | 4.752e-06 – 2.363e-05 | 1.153e-05 | 8.664e-06 ± 4.242e-06 |

10 trainings per model (SD with n − 1); only the initialisation seed differs (101–110).
Test set: 2000 new samples, uniform in [0, 1]², seed 2026, the same for all models and seeds;
the original 300 test samples only for the link to earlier results.

## Paired comparison (same seed)

- E34 + refit vs R8 (10000 epochs), same seed: better in 6 / 10 seeds; difference (variant − R8) mean -3.367e-06 ± 1.085e-05; ratio variant / R8 median 0.63 (range 0.16–2.34)
- G34 + refit vs R8 (10000 epochs), same seed: better in 8 / 10 seeds; difference (variant − R8) mean -6.660e-06 ± 7.148e-06; ratio variant / R8 median 0.51 (range 0.32–1.33)

## Activity

- R8: 8 per edge, same 8 in both layers (fixed)
- E34: per edge 21.8 (16.7–33.5); unique per layer 27.9 (21–34) / 33.0 (31–34)
- G34: per edge 18.1 (15.0–22.0); unique per layer 17.0 (13–20) / 20.3 (17–28)

Mask threshold in the largest gap of the sorted magnitudes; gap ratio E34 5.6–111.5,
G34 16.4–963.1. Effect of masking before the refit (test MSE masked / phase 1):
E34 1.0005 (median; 1.0000–1.0008),
G34 1.0003 (1.0001–1.0006): the gap threshold
only removes coefficients that are already close to zero.
Selected indices per edge / layer: `seed_sweep_selected_indices.csv`.

## Validity and stopping point

- Valid (u >= 0, P <= 60, finite) on training inputs, both test sets and a 201 x 201 grid over [0, 1]²:
  R8 10 / 10, E34 10 / 10, G34 10 / 10.
- Final value relative to the mean test MSE of the last 500 epochs of the final phase
  (1 = the run stopped at its plateau level): R8 0.980–0.995,
  E34 0.940–0.964, G34 0.943–0.983;
  relative SD within that window: max R8 1.2e-02, E34 3.7e-02,
  G34 3.5e-02.

## Settings

- sqrt(m) [2,2,1] KAN, power window [0.05, 8], layer-0 input range [-0.2, 1], shared layer-1 input range
  [-7.0080, 6.6222] (all variants and seeds); only expansion coefficients are trained.
- Adam, learning rate 1e-3 -> 1e-4 within each 5000-epoch phase, float64; 1000 training samples (data seed 42).
- R8: 2 x 5000 epochs (second phase continued, schedule restarted). E34: lambda 8e-05; G34: lambda
  0.0003; phase 1 (5000) -> mask -> refit (5000, initialisation seed + 1000, masked coefficients
  exactly zero and frozen, no penalty).
- Mapping, lambda and mask rule were fixed on training data only (`results_sparse_selection/`); the test sets
  were not used for any decision.

## Files

- `seed_sweep_runs.csv`: one row per run (30); `seed_sweep_summary.csv`; `seed_sweep_selected_indices.csv`
- `seed_sweep_test_mse.png`: test MSE per seed and model
- `runs/`: everything per run, incl. loss histories; `models/`: full forward export of every final model
