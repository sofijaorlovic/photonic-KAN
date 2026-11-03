# Variant I: basis shift and coefficient error, sqrt(m) [2,2,1] KAN

## Summary

| Scenario | Median MSE | Median R | 90th perc. R | Valid / 30 |
|---|---|---|---|---|
| S0 | 3.334e-05 | 1.000 | — | — |
| S1 | 1.696e-03 | 50.868 | 164.802 | 30/30 |
| S2 | 4.631e-04 | 13.891 | 52.182 | 30/30 |
| S3 | 1.918e-03 | 57.525 | 259.133 | 30/30 |

R = MSE / MSE0; statistics over valid realizations only.
All realizations are valid.


## 1. Model and test set

- Model: `new_consistent_affine_2-2-1_sqrtm_exports/photonic_kan_2-2-1_sqrtm_full_forward_test.mat`, rebuilt from the exported coefficients
  (no notebook rerun); edge model phi(x) = sum_k a_k sqrt(b_k((alpha x + beta)^2 / m)), m = 8.
- Bases (Python index): [8, 10, 12, 14, 16, 18, 20, 22]; same 8 bases on every edge of both layers.
- Test set: the original 300 test samples, x in [0, 1]^2 (no further normalization).
- MSE0 = 3.334146e-05; the rebuild reproduces the exported PyTorch output to 1.39e-15
  and passes all 11 consistency checks of `rebuild.py`.

## 2. Valid domain, units, normalization

- Fit argument: FP-LD input power P = u^2 / m (units of `input_power` in the activation file).
- Valid domain: P in [0, 60], the measured input-power range of the FP-LD data
  the analytic fits were made on; the same for all 8 bases.
- Operating window: the input range in which each layer operates, i.e. its configured input range,
  which the layer's affine map sends onto the power window (power window of the layers' affine maps (same for every layer)):

  | Layer | Input range | alpha | beta | u window | P window | P on test set |
  |---|---|---|---|---|---|---|
  | 0 | [-0.2000, 1.0000] | 6.139620 | 1.860380 | [0.6325, 8.0000] | [0.05, 8] | [0.4440, 7.9637] |
  | 1 | [-7.3647, 3.0795] | 0.705422 | 5.827686 | [0.6325, 8.0000] | [0.05, 8] | [1.6662, 5.0350] |

  Operating window P in [0.05, 8], Delta_P = 7.95.
- Nominal arguments on the test set, all layers: P in [0.4440, 7.9637].

## 3. Common Delta_P

All 8 bases are fitted on the same measured power grid, so one common window and one Delta_P are used:
sigma_delta = 0.01 x 7.95 = 0.0795 (units of P).

## 4. Shift assumption

One delta_k per basis k and realization, applied to every occurrence of basis k (all 6 edges, both layers):
b_k(P) -> b_k(P - delta_k). This is a simplified model of a variation of the 8 basis profiles,
not a statement about the hardware topology.

## 5. Perturbed coefficients

48 learned expansion coefficients (layer 0: 32, layer 1: 16),
each with its own eps_i ~ N(0, 0.02^2): a_i -> a_i (1 + eps_i).
Zero coefficients: 0 (a zero coefficient stays zero under a relative error).
The 8 coefficients of each analytic fit are not changed (the float32-rounded values used by the
trained model, as exported, are kept).

## 6. Random numbers

NumPy `default_rng([2026, r])` (PCG64), one independent stream per realization r = 1..30;
order of draws: delta_1..delta_8, then eps of layer 0 and layer 1 in C order of coeffs[out, in, k].
The same delta and eps of realization r are used in S1, S2 and S3. All draws: `variant_I_draws.csv`.

## 7. Deviations from the instruction

None. Affine maps, inputs, all other functions and settings are fixed; no clipping, recalibration,
renormalization or retraining.

## Conclusion scope

The results describe only the sensitivity of the trained mathematical KAN model to the given static
modifications. The chosen percentages are not measured tolerances of FP-LD / SOA / VOA devices.
