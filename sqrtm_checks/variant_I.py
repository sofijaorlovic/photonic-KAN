"""
Variant I: horizontal shift of the FP-LD basis functions and error of the KAN expansion coefficients
====================================================================================================

Sensitivity test of the trained sqrt(m) [2,2,1] KAN (y = 0.3 x1^2 + cos(3 x2)), forward passes only:
no retraining, no change of architecture, data, affine maps or the 8 analytic fit coefficients.

    I  - basis shift:        b_k(P) -> b_k(P - delta_k),  delta_k ~ N(0, sigma_delta^2),
                             sigma_delta = 0.01 * Delta_P, Delta_P = width of the operating window.
                             One delta_k per basis and realization, used wherever basis k appears
                             (all edges, both layers). In this model the fit argument is the
                             FP-LD input power P = u^2/m, so delta is in units of P.
    II - coefficient error:  a_i -> a_i (1 + eps_i),  eps_i ~ N(0, 0.02^2), one eps_i per learned
                             expansion coefficient (both layers), fixed for the whole test set.

Scenarios: S0 reference, S1 shift only, S2 coefficients only, S3 both. For realization r the same
drawn delta and eps vectors are used in S1, S2, S3 (eps = 0 in S1, delta = 0 in S2).

Domain check: every argument P - delta_k, in every layer, must lie in the valid domain of the fit
(the measured input-power range of the FP-LD data). A realization with an argument outside it, or
any NaN / Inf, is marked invalid with its reason; it is not repaired (no clipping, no extrapolation,
no redraw). Inter-layer values are recomputed after the perturbation.

Usage:
    python variant_I.py [--window P_MIN P_MAX] [--out DIR]

    Default operating window: the power window onto which each layer's affine map sends the
    layer's configured input range ([0.05, 8] for both layers of this model).

Outputs (in DIR, default sqrtm_checks/results_variant_I/):
    variant_I_runs.csv      90 rows: scenario, realization, seed, valid, MSE, R, invalid reason, ...
    variant_I_draws.csv     the drawn delta_k and eps_i of every realization
    variant_I_summary.csv   summary table (median MSE, median R, 90th percentile R, valid / R)
    variant_I_report.md     short report with all settings and choices
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rebuild  # noqa: E402

SEED = 2026
N_REALIZATIONS = 30
SIGMA_DELTA_REL = 0.01      # sigma_delta = 1 % of the operating-window width
SIGMA_EPS = 0.02            # 2 % relative coefficient error
SCENARIOS = {               # (shift on, coefficient error on)
    "S1": (True, False),
    "S2": (False, True),
    "S3": (True, True),
}


def draw(model, r, sigma_delta):
    """delta (m,) and eps (one array per layer, coeffs shape) of realization r.
    Independent stream per realization: numpy default_rng([SEED, r]) (PCG64);
    order: delta_1..delta_m, then eps of layer 0, layer 1 (C order of coeffs[out, in, k])."""
    rng = np.random.default_rng([SEED, r])
    m = model["layers"][0]["m"]
    delta = rng.normal(0.0, sigma_delta, size=m)
    eps = [rng.normal(0.0, SIGMA_EPS, size=layer["coeffs"].shape) for layer in model["layers"]]
    return delta, eps


def argument_range(layers):
    return (float(min(np.nanmin(l["arg"]) for l in layers)),
            float(max(np.nanmax(l["arg"]) for l in layers)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", nargs=2, type=float, default=None, metavar=("P_MIN", "P_MAX"),
                    help="operating window of the FP-LD input power; Delta_P = P_MAX - P_MIN. "
                         "Default: the power window onto which every layer's affine map sends its "
                         "configured input range (must be the same for all layers).")
    ap.add_argument("--out", default=os.path.join(rebuild.HERE, "results_variant_I"))
    args = ap.parse_args()

    model = rebuild.load_model()
    x, y = model["x_test"], model["y_test"]
    m = model["layers"][0]["m"]
    domain = model["valid_domain"]

    # operating window: the input range each layer operates in, in units of the fit argument P.
    # Every layer's affine map sends its configured input range onto the layer's power window.
    windows = [tuple(l["power_window"]) for l in model["layers"]]
    if args.window is None:
        if len(set(windows)) != 1:
            sys.exit(f"Layers have different power windows {windows}; pass --window explicitly.")
        P_win_lo, P_win_hi = windows[0]
        window_source = "power window of the layers' affine maps (same for every layer)"
    else:
        P_win_lo, P_win_hi = args.window
        window_source = "set by --window"
    delta_P = P_win_hi - P_win_lo
    sigma_delta = SIGMA_DELTA_REL * delta_P
    zero_coeffs = int(sum((l["coeffs"] == 0).sum() for l in model["layers"]))
    n_coeffs = rebuild.num_coeffs(model)

    # step 0 checks: the rebuild must reproduce the exported model
    checks = rebuild.run_checks(model, verbose=False)
    failed = [c for c in checks if not c[1]]
    if failed:
        for name, _, detail in failed:
            print(f"[FAIL] {name}: {detail}")
        sys.exit("Rebuild checks failed; not running the sensitivity test.")

    # S0: nominal
    mse0, _, layers0, check0 = rebuild.mse(model, x, y)
    if not check0["valid"]:
        sys.exit("Nominal model leaves the valid domain: " + "; ".join(check0["issues"]))
    arg0 = argument_range(layers0)

    os.makedirs(args.out, exist_ok=True)
    runs, draws_rows = [], []
    for r in range(1, N_REALIZATIONS + 1):
        delta, eps = draw(model, r, sigma_delta)
        eps_flat = np.concatenate([e.ravel() for e in eps])
        draws_rows.append([r] + delta.tolist() + eps_flat.tolist())
        for name, (use_shift, use_eps) in SCENARIOS.items():
            mse_r, _, layers, check = rebuild.mse(
                model, x, y,
                delta=delta if use_shift else None,
                eps=eps if use_eps else None,
            )
            a_lo, a_hi = argument_range(layers)
            valid = check["valid"] and np.isfinite(mse_r)
            runs.append({
                "scenario": name, "realization": r, "seed": f"{SEED}/{r}",
                "valid": int(valid),
                "MSE": mse_r if valid else "", "R": mse_r / mse0 if valid else "",
                "invalid_reason": "" if valid else "; ".join(check["issues"]) or "non-finite MSE",
                "min_arg": a_lo, "max_arg": a_hi,
                "max_abs_delta": float(np.max(np.abs(delta))) if use_shift else 0.0,
                "max_abs_eps": float(np.max(np.abs(eps_flat))) if use_eps else 0.0,
            })

    # --- CSV: all runs
    runs_path = os.path.join(args.out, "variant_I_runs.csv")
    with open(runs_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0].keys()))
        w.writeheader()
        w.writerows(runs)   # 30 x 3 rows; S0 (MSE0) is in the summary table and the report

    # --- CSV: draws
    eps_names = [f"eps_L{L}_out{o}_in{i}_k{k}"
                 for L, l in enumerate(model["layers"])
                 for o in range(l["coeffs"].shape[0]) for i in range(l["coeffs"].shape[1]) for k in range(m)]
    with open(os.path.join(args.out, "variant_I_draws.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["realization"] + [f"delta_k{k}_idx{idx}" for k, idx in enumerate(model["selected_indices"])]
                   + eps_names)
        w.writerows(draws_rows)

    # --- summary
    summary = [{"scenario": "S0", "median_MSE": mse0, "median_R": 1.0, "p90_R": "", "valid": ""}]
    invalid_notes = []
    for name in SCENARIOS:
        rows = [q for q in runs if q["scenario"] == name]
        ok = [q for q in rows if q["valid"]]
        R = np.array([q["R"] for q in ok], dtype=float)
        M = np.array([q["MSE"] for q in ok], dtype=float)
        summary.append({
            "scenario": name,
            "median_MSE": float(np.median(M)) if ok else "",
            "median_R": float(np.median(R)) if ok else "",
            "p90_R": float(np.percentile(R, 90)) if ok else "",
            "valid": f"{len(ok)}/{len(rows)}",
        })
        for q in rows:
            if not q["valid"]:
                invalid_notes.append(f"{name}, r = {q['realization']}: {q['invalid_reason']}")

    with open(os.path.join(args.out, "variant_I_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # --- report
    def fmt(v, spec):
        return format(v, spec) if v != "" else "—"

    table = ["| Scenario | Median MSE | Median R | 90th perc. R | Valid / 30 |",
             "|---|---|---|---|---|"]
    for s in summary:
        table.append(f"| {s['scenario']} | {fmt(s['median_MSE'], '.3e')} | {fmt(s['median_R'], '.3f')} | "
                     f"{fmt(s['p90_R'], '.3f')} | {s['valid'] or '—'} |")

    L0, L1 = model["layers"]
    layer_rows = ["  | Layer | Input range | alpha | beta | u window | P window | P on test set |",
                  "  |---|---|---|---|---|---|---|"]
    for i, (l, ln) in enumerate(zip(model["layers"], layers0)):
        layer_rows.append(
            f"  | {i} | [{l['input_range'][0]:.4f}, {l['input_range'][1]:.4f}] | {l['alpha'][0]:.6f} | "
            f"{l['beta'][0]:.6f} | [{l['amplitude_window'][0]:.4f}, {l['amplitude_window'][1]:.4f}] | "
            f"[{l['power_window'][0]:g}, {l['power_window'][1]:g}] | "
            f"[{ln['P'].min():.4f}, {ln['P'].max():.4f}] |")
    report =f"""# Variant I: basis shift and coefficient error, sqrt(m) [2,2,1] KAN

## Summary

{chr(10).join(table)}

R = MSE / MSE0; statistics over valid realizations only.
{"All realizations are valid." if not invalid_notes else
 f"{len(invalid_notes)} invalid realization(s), excluded from the statistics (the statistics then do not describe the whole distribution):"}
{chr(10).join("- " + n for n in invalid_notes)}

## 1. Model and test set

- Model: `{os.path.relpath(model['path'], rebuild.PROJECT)}`, rebuilt from the exported coefficients
  (no notebook rerun); edge model phi(x) = sum_k a_k sqrt(b_k((alpha x + beta)^2 / m)), m = {m}.
- Bases (Python index): {model['selected_indices'].tolist()}; same 8 bases on every edge of both layers.
- Test set: the original {x.shape[0]} test samples, x in [0, 1]^2 (no further normalization).
- MSE0 = {mse0:.6e}; the rebuild reproduces the exported PyTorch output to {checks[0][2].split('= ')[1]}
  and passes all {len(checks)} consistency checks of `rebuild.py`.

## 2. Valid domain, units, normalization

- Fit argument: FP-LD input power P = u^2 / m (units of `input_power` in the activation file).
- Valid domain: P in [{domain[0]:g}, {domain[1]:g}], the measured input-power range of the FP-LD data
  the analytic fits were made on; the same for all 8 bases.
- Operating window: the input range in which each layer operates, i.e. its configured input range,
  which the layer's affine map sends onto the power window ({window_source}):

{chr(10).join(layer_rows)}

  Operating window P in [{P_win_lo:g}, {P_win_hi:g}], Delta_P = {delta_P:g}.
- Nominal arguments on the test set, all layers: P in [{arg0[0]:.4f}, {arg0[1]:.4f}].

## 3. Common Delta_P

All 8 bases are fitted on the same measured power grid, so one common window and one Delta_P are used:
sigma_delta = {SIGMA_DELTA_REL:g} x {delta_P:g} = {sigma_delta:.4g} (units of P).

## 4. Shift assumption

One delta_k per basis k and realization, applied to every occurrence of basis k (all 6 edges, both layers):
b_k(P) -> b_k(P - delta_k). This is a simplified model of a variation of the 8 basis profiles,
not a statement about the hardware topology.

## 5. Perturbed coefficients

{n_coeffs} learned expansion coefficients (layer 0: {L0['coeffs'].size}, layer 1: {L1['coeffs'].size}),
each with its own eps_i ~ N(0, {SIGMA_EPS:g}^2): a_i -> a_i (1 + eps_i).
Zero coefficients: {zero_coeffs} (a zero coefficient stays zero under a relative error).
The 8 coefficients of each analytic fit are not changed (the float32-rounded values used by the
trained model, as exported, are kept).

## 6. Random numbers

NumPy `default_rng([{SEED}, r])` (PCG64), one independent stream per realization r = 1..{N_REALIZATIONS};
order of draws: delta_1..delta_{m}, then eps of layer 0 and layer 1 in C order of coeffs[out, in, k].
The same delta and eps of realization r are used in S1, S2 and S3. All draws: `variant_I_draws.csv`.

## 7. Deviations from the instruction

None. Affine maps, inputs, all other functions and settings are fixed; no clipping, recalibration,
renormalization or retraining.

## Conclusion scope

The results describe only the sensitivity of the trained mathematical KAN model to the given static
modifications. The chosen percentages are not measured tolerances of FP-LD / SOA / VOA devices.
"""
    with open(os.path.join(args.out, "variant_I_report.md"), "w") as f:
        f.write(report)

    print(f"MSE0 = {mse0:.6e}, Delta_P = {delta_P:g}, sigma_delta = {sigma_delta:.4g}")
    print("\n".join(table))
    if invalid_notes:
        print("\nInvalid realizations:\n  " + "\n  ".join(invalid_notes))
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
