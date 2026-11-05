"""
Preparation of the sparse comparison: shared layer-1 mapping and lambda, decided on TRAINING data only
=====================================================================================================

Step 1  --mapping   Shared layer-1 mapping for R8, E34 and G34.
        Trial runs of the full pipeline of every variant (R8: 2 x 5000 epochs; E34 / G34: phase 1 + refit,
        lambda of the document 8e-5 / 8e-4, gap threshold) on the 1000 training samples, trial seeds
        1, 2, 3 (not among the final seeds 101-110), starting from a wide layer-1 range [-10, 10].
        Shared range = union of the hidden ranges z (training inputs) at the end of every phase of every
        trial, widened by 10 % around its centre. Fixed from here on.

Step 2  --lambda    lambda per sparse variant, under the shared mapping.
        Fixed split of the training samples: 800 fit / 200 validation (permutation seed 0).
        For each lambda of the grid and each trial seed 1, 2, 3: phase 1 + gap mask + refit on the
        800 samples; validation MSE of the refitted model on the 200.
        Rule (fixed in advance): lambda* = lowest mean validation MSE; the largest lambda whose mean
        validation MSE is within one standard error of lambda*'s is chosen (sparser model, same accuracy
        within noise). Validity of every run (u >= 0, P <= 60) is recorded; invalid runs are reported.

Step 3  --confirm   Validity of the chosen lambdas with the shared mapping on all 1000 training samples
        (trial seeds 1, 2, 3).

The test sets (original 300 and the new 2000) are not used anywhere in this script.

Usage:
    python sparse_select.py --mapping
    python sparse_select.py --lambda [--variant E34|G34] [--jobs N]
    python sparse_select.py --confirm
Results: sqrtm_checks/results_sparse_selection/
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "results_sparse_selection")

TRIAL_SEEDS = [1, 2, 3]
WIDE_RANGE = (-10.0, 10.0)
MAPPING_MARGIN = 1.10
DOC_LAMBDA = {"E34": 8e-5, "G34": 8e-4}
LAMBDA_GRID = {
    "E34": [1e-5, 3e-5, 8e-5, 2e-4, 5e-4, 1e-3],
    "G34": [1e-4, 3e-4, 8e-4, 2e-3, 5e-3, 1e-2],
}
EPOCHS = 5000
SPLIT_SEED = 0
N_FIT = 800


def _setup():
    import torch
    torch.set_num_threads(1)
    import sparse_sqrtm as S
    return torch, S


def _split(x, y):
    torch, _ = _setup()
    g = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(x.shape[0], generator=g)
    fit, val = perm[:N_FIT], perm[N_FIT:]
    return x[fit], y[fit], x[val], y[val]


def _summary(r):
    out = {"variant": r["variant"], "init_seed": r["init_seed"], "lam": r["lam"],
           "phase1_train_mse": r["phase1"]["train_mse"], "phase1_hidden": r["phase1"]["hidden_range"],
           "phase1_valid": r["phase1"]["validity"]["valid"],
           "final_train_mse": r["final"]["train_mse"], "final_hidden": r["final"]["hidden_range"],
           "final_valid": r["final"]["validity"]["valid"], "final_validity": r["final"]["validity"],
           "seconds": r["seconds"]}
    if "eval_mse" in r["final"]:
        out["val_mse"] = r["final"]["eval_mse"]
    if r["variant"] != "R8":
        out.update({"threshold": r["threshold_used"], "gap_ratio": r["gap_ratio"],
                    "masked_train_mse": r["masked"]["train_mse"],
                    "active_per_edge": r["mask"]["active_per_edge"],
                    "unique_per_layer": r["mask"]["unique_per_layer"],
                    "n_unique_per_layer": r["mask"]["n_unique_per_layer"],
                    "n_active_coeffs": r["mask"]["n_active_coeffs"]})
    return out


# ---------------------------------------------------------------------------
# Step 1: mapping
# ---------------------------------------------------------------------------

def _mapping_job(args):
    variant, seed = args
    torch, S = _setup()
    xt, yt, _, _ = S.notebook_data()
    r = S.run_variant(variant, WIDE_RANGE, seed, xt, yt, lam=DOC_LAMBDA.get(variant, 0.0),
                      threshold="gap", epochs=EPOCHS, refit_seed=seed + 1000, record_every=100)
    return _summary(r)


def step_mapping(jobs):
    tasks = [(v, s) for v in ("R8", "E34", "G34") for s in TRIAL_SEEDS]
    with ProcessPoolExecutor(jobs) as ex:
        res = list(ex.map(_mapping_job, tasks))
    lo = min(min(r["phase1_hidden"][0], r["final_hidden"][0]) for r in res)
    hi = max(max(r["phase1_hidden"][1], r["final_hidden"][1]) for r in res)
    c, h = 0.5 * (lo + hi), 0.5 * (hi - lo) * MAPPING_MARGIN
    shared = [c - h, c + h]
    out = {"trial_runs": res, "union_hidden_range": [lo, hi], "margin": MAPPING_MARGIN,
           "shared_layer1_range": shared, "start_range": list(WIDE_RANGE), "lambda_used": DOC_LAMBDA}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "step1_mapping.json"), "w") as f:
        json.dump(out, f, indent=1)
    for r in res:
        print(f"{r['variant']} seed {r['init_seed']}: hidden phase 1 [{r['phase1_hidden'][0]:.3f}, "
              f"{r['phase1_hidden'][1]:.3f}], final [{r['final_hidden'][0]:.3f}, {r['final_hidden'][1]:.3f}], "
              f"final train MSE {r['final_train_mse']:.2e}, valid {r['final_valid']}")
    print(f"\nUnion of hidden ranges: [{lo:.4f}, {hi:.4f}] -> shared layer-1 range "
          f"[{shared[0]:.4f}, {shared[1]:.4f}] (x {MAPPING_MARGIN})")


def shared_range():
    with open(os.path.join(OUT, "step1_mapping.json")) as f:
        return tuple(json.load(f)["shared_layer1_range"])


# ---------------------------------------------------------------------------
# Step 2: lambda
# ---------------------------------------------------------------------------

def _lambda_job(args):
    variant, lam, seed, rng = args
    torch, S = _setup()
    xt, yt, _, _ = S.notebook_data()
    xf, yf, xv, yv = _split(xt, yt)
    r = S.run_variant(variant, rng, seed, xf, yf, lam=lam, threshold="gap", epochs=EPOCHS,
                      refit_seed=seed + 1000, x_eval=xv, y_eval=yv, record_every=100)
    return _summary(r)


def choose_lambda(runs, variant):
    rows = []
    for lam in LAMBDA_GRID[variant]:
        rs = [r for r in runs if r["variant"] == variant and r["lam"] == lam]
        v = np.array([r["val_mse"] for r in rs])
        rows.append({"lam": lam, "n": len(rs), "mean_val_mse": float(v.mean()),
                     "se": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0,
                     "all_valid": all(r["final_valid"] and r["phase1_valid"] for r in rs),
                     "mean_active_per_edge": float(np.mean([np.mean(r["active_per_edge"]) for r in rs])),
                     "mean_unique_per_layer": np.mean([r["n_unique_per_layer"] for r in rs], axis=0).tolist(),
                     "min_gap_ratio": float(min(r["gap_ratio"] for r in rs))})
    best = min(rows, key=lambda q: q["mean_val_mse"])
    within = [q for q in rows if q["mean_val_mse"] <= best["mean_val_mse"] + best["se"]]
    chosen = max(within, key=lambda q: q["lam"])
    return rows, best["lam"], chosen["lam"]


def step_lambda(jobs, variants):
    rng = shared_range()
    tasks = [(v, lam, s, rng) for v in variants for lam in LAMBDA_GRID[v] for s in TRIAL_SEEDS]
    with ProcessPoolExecutor(jobs) as ex:
        runs = list(ex.map(_lambda_job, tasks))
    path = os.path.join(OUT, "step2_lambda.json")
    old = {}
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
    for v in variants:
        rows, best, chosen = choose_lambda(runs, v)
        old[v] = {"runs": [r for r in runs if r["variant"] == v], "table": rows,
                  "lambda_lowest_val_mse": best, "lambda_chosen": chosen,
                  "rule": "largest lambda with mean validation MSE <= min mean + its SE",
                  "split": {"n_fit": N_FIT, "n_val": 1000 - N_FIT, "perm_seed": SPLIT_SEED},
                  "layer1_range": list(rng)}
        print(f"\n{v} (layer-1 range [{rng[0]:.3f}, {rng[1]:.3f}])")
        print(f"{'lambda':>8} {'val MSE mean':>13} {'SE':>10} {'act/edge':>9} {'unique/layer':>14} "
              f"{'min gap':>8} valid")
        for q in rows:
            print(f"{q['lam']:>8.0e} {q['mean_val_mse']:>13.3e} {q['se']:>10.2e} {q['mean_active_per_edge']:>9.1f} "
                  f"{str([round(u, 1) for u in q['mean_unique_per_layer']]):>14} {q['min_gap_ratio']:>8.1f} "
                  f"{q['all_valid']}")
        print(f"lowest mean validation MSE at lambda = {best:g}; chosen lambda = {chosen:g}")
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w") as f:
        json.dump(old, f, indent=1)


def chosen_lambda():
    with open(os.path.join(OUT, "step2_lambda.json")) as f:
        d = json.load(f)
    return {v: d[v]["lambda_chosen"] for v in d}


# ---------------------------------------------------------------------------
# Step 3: confirm
# ---------------------------------------------------------------------------

def _confirm_job(args):
    variant, lam, seed, rng = args
    torch, S = _setup()
    xt, yt, _, _ = S.notebook_data()
    r = S.run_variant(variant, rng, seed, xt, yt, lam=lam, threshold="gap", epochs=EPOCHS,
                      refit_seed=seed + 1000, record_every=100)
    return _summary(r)


def step_confirm(jobs):
    rng = shared_range()
    lam = chosen_lambda()
    tasks = [(v, lam.get(v, 0.0), s, rng) for v in ("R8", "E34", "G34") for s in TRIAL_SEEDS]
    with ProcessPoolExecutor(jobs) as ex:
        res = list(ex.map(_confirm_job, tasks))
    with open(os.path.join(OUT, "step3_confirm.json"), "w") as f:
        json.dump({"layer1_range": list(rng), "lambda": lam, "runs": res}, f, indent=1)
    for r in res:
        v = r["final_validity"]
        print(f"{r['variant']} seed {r['init_seed']}: valid {r['phase1_valid']} / {r['final_valid']} | "
              f"hidden final [{r['final_hidden'][0]:.3f}, {r['final_hidden'][1]:.3f}] | "
              f"layer-1 P [{v['L1_min_P']:.3f}, {v['L1_max_P']:.3f}] | train MSE {r['final_train_mse']:.2e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mapping", action="store_true")
    ap.add_argument("--lambda", dest="lam", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--variant", nargs="+", default=["E34", "G34"])
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    if a.mapping:
        step_mapping(a.jobs)
    if a.lam:
        step_lambda(a.jobs, a.variant)
    if a.confirm:
        step_confirm(a.jobs)


if __name__ == "__main__":
    main()
