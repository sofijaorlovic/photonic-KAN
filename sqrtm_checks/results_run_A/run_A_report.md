# Run A: repeated training of the standard sqrt(m) [2,2,1] KAN

## Result

| | Mean ± SD | Min – max | Median |
|---|---|---|---|
| Train MSE | 4.628e-05 ± 2.736e-05 | 2.305e-05 – 1.030e-04 | 3.811e-05 |
| Test MSE | 4.032e-05 ± 2.515e-05 | 2.055e-05 – 1.006e-04 | 3.266e-05 |

10 independent trainings (SD with n − 1). Notebook run: test MSE 3.334e-05;
5 of 10 runs have a lower test MSE.
Calibration: 10 / 10 runs physically valid on the training inputs
(u >= 0, P <= 60 in layer 1), 10 / 10 also on the test inputs;
passes needed: 1–1 (mean 1.0).

## What varies, what is fixed

- Varies: only the seed of the weight initialisation (101, 102, 103, 104, 105, 106, 107, 108, 109, 110); hence also the
  calibrated layer-1 mapping of each run.
- Fixed: architecture, the 8 bases, sqrt(m) edge model, power window [0.05, 8], layer-0 input range,
  1000 training and 300 test samples (data seed 42), Adam, learning rate 0.001 -> 0.0001,
  5000 epochs, calibration procedure (probe 1000 epochs, margin 1.1), float64, CPU.
- The script was checked against the notebook: with the notebook's own random state it reproduces the
  exported model (`--reference`).

## Per run

| Seed | Train MSE | Test MSE | Passes | Layer-1 range | Valid (train / test) |
|---|---|---|---|---|---|
| 101 | 4.393e-05 | 4.083e-05 | 1 | [-5.281, 1.525] | yes / yes |
| 102 | 4.239e-05 | 3.696e-05 | 1 | [-5.628, 6.486] | yes / yes |
| 103 | 3.126e-05 | 2.649e-05 | 1 | [-7.109, 3.039] | yes / yes |
| 104 | 2.305e-05 | 2.055e-05 | 1 | [-4.545, 1.990] | yes / yes |
| 105 | 2.689e-05 | 2.189e-05 | 1 | [-5.118, 6.165] | yes / yes |
| 106 | 1.030e-04 | 1.006e-04 | 1 | [-6.008, 6.790] | yes / yes |
| 107 | 4.190e-05 | 3.747e-05 | 1 | [-4.794, 2.104] | yes / yes |
| 108 | 3.432e-05 | 2.835e-05 | 1 | [-6.276, 1.703] | yes / yes |
| 109 | 8.905e-05 | 6.665e-05 | 1 | [-5.802, 8.347] | yes / yes |
| 110 | 2.707e-05 | 2.340e-05 | 1 | [-5.651, 2.613] | yes / yes |

## Files

- `run_A_runs.csv`: one row per run (MSE, calibration, layer-1 mapping, validity, time)
- `run_A_summary.csv`: mean, SD, min, max, median
- `run_A_test_mse.png`: test MSE per run
- `runs/`, `models/`, `loss_curves/`: per-run details, full forward exports, loss curves
