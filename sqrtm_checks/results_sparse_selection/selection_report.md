# Sparse comparison: shared mapping and lambda (training data only)

Model: sqrt(m) [2,2,1] KAN, power window [0.05, 8], layer-0 input range [-0.2, 1], Adam, learning rate 1e-3 -> 1e-4 per 5000-epoch phase, float64. R8: 2 x 5000 epochs (second phase continued, schedule restarted); E34 / G34: phase 1 with penalty -> mask -> refit (new seeded initialisation, no penalty). Mask threshold: largest gap of the sorted magnitudes (|a| for E34, group norms per layer for G34). Trial seeds 1, 2, 3 (not among the final seeds 101-110). The test sets are not used.

## Step 1: shared layer-1 mapping

Trial runs of all three variants (lambda of the document: E34 8e-5, G34 8e-4) on the 1000 training samples, start range [-10, 10].

| Variant | Seed | Hidden z, end of phase 1 | Hidden z, final | Valid |
|---|---|---|---|---|
| R8 | 1 | [-4.284, 1.229] | [-4.159, 1.054] | yes |
| R8 | 2 | [-5.059, 5.613] | [-4.387, 4.977] | yes |
| R8 | 3 | [-5.050, 1.987] | [-4.664, 1.679] | yes |
| E34 | 1 | [-3.886, 2.903] | [-5.039, 3.882] | yes |
| E34 | 2 | [-4.554, 5.183] | [-6.388, 2.914] | yes |
| E34 | 3 | [-3.675, 3.046] | [-4.296, 3.657] | yes |
| G34 | 1 | [-3.846, 1.054] | [-4.623, 2.814] | yes |
| G34 | 2 | [-4.080, 4.279] | [-4.905, 6.003] | yes |
| G34 | 3 | [-3.565, 1.447] | [-4.380, 2.832] | yes |

Union [-6.3884, 6.0026], widened by 10 % -> **shared layer-1 input range [-7.0080, 6.6222]**, fixed for all variants and seeds.

## Step 2: lambda

Training samples split 800 fit / 200 validation (permutation seed 0); 3 trial seeds per lambda; validation MSE after refit. Rule fixed in advance: largest lambda whose mean validation MSE is within one standard error of the lowest one.

**E34**

| lambda | Validation MSE (mean ± SE) | Active per edge | Unique per layer | Smallest gap ratio | Valid |
|---|---|---|---|---|---|
| 1e-05 | 1.350e-05 ± 1.9e-06 | 32.4 | 34.0, 34.0 | 17.3 | yes |
| 3e-05 | 2.178e-05 ± 8.7e-06 | 27.1 | 34.0, 33.0 | 6.5 | yes |
| 8e-05 | 1.467e-05 ± 4.3e-07 | 20.2 | 24.7, 31.7 | 8.8 | yes |
| 0.0002 | 1.828e-05 ± 6.5e-06 | 20.9 | 22.3, 27.0 | 6.9 | yes |
| 0.0005 | 3.691e-05 ± 1.6e-05 | 11.2 | 16.3, 16.7 | 16.3 | yes |
| 0.001 | 3.917e-05 ± 2.3e-05 | 17.3 | 19.3, 20.7 | 6.5 | yes |

Lowest validation MSE at lambda = 1e-05; **chosen lambda = 8e-05**.

**G34**

| lambda | Validation MSE (mean ± SE) | Active per edge | Unique per layer | Smallest gap ratio | Valid |
|---|---|---|---|---|---|
| 0.0001 | 1.132e-05 ± 2.3e-06 | 26.2 | 24.0, 30.7 | 38.4 | yes |
| 0.0003 | 1.307e-05 ± 3.0e-06 | 18.9 | 17.7, 21.3 | 91.2 | yes |
| 0.0008 | 2.426e-05 ± 1.7e-06 | 14.9 | 14.3, 16.0 | 9.8 | yes |
| 0.002 | 2.554e-05 ± 1.3e-06 | 12.2 | 12.0, 12.7 | 158.5 | yes |
| 0.005 | 7.841e-05 ± 1.2e-05 | 10.8 | 11.0, 10.3 | 162.0 | yes |
| 0.01 | 7.566e-05 ± 1.4e-05 | 10.1 | 10.7, 9.0 | 80.0 | yes |

Lowest validation MSE at lambda = 0.0001; **chosen lambda = 0.0003**.

## Step 3: confirmation on all 1000 training samples

| Variant | lambda | Seed | Valid (phase 1 / final) | Hidden z, final | Layer-1 P | Train MSE |
|---|---|---|---|---|---|---|
| R8 | 0 | 1 | yes / yes | [-3.147, 0.282] | [0.925, 2.614] | 1.81e-05 |
| R8 | 0 | 2 | yes / yes | [-3.377, 4.759] | [0.842, 6.113] | 1.18e-05 |
| R8 | 0 | 3 | yes / yes | [-3.863, 1.000] | [0.680, 3.077] | 1.91e-05 |
| E34 | 8e-05 | 1 | yes / yes | [-3.749, 2.982] | [0.716, 4.549] | 9.84e-06 |
| E34 | 8e-05 | 2 | yes / yes | [-3.199, 5.680] | [0.905, 7.014] | 6.33e-06 |
| E34 | 8e-05 | 3 | yes / yes | [-3.350, 2.922] | [0.851, 4.500] | 1.02e-05 |
| G34 | 0.0003 | 1 | yes / yes | [-3.765, 2.937] | [0.711, 4.512] | 7.70e-06 |
| G34 | 0.0003 | 2 | yes / yes | [-3.963, 4.431] | [0.649, 5.807] | 1.95e-05 |
| G34 | 0.0003 | 3 | yes / yes | [-3.641, 2.943] | [0.752, 4.517] | 6.73e-06 |

## Fixed for the seed sweep

- Shared layer-1 input range [-7.0080, 6.6222] for R8, E34, G34
- lambda: E34 8e-05, G34 0.0003; mask threshold: largest gap
- Epochs: 2 x 5000 per variant
