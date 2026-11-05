"""
Seed sweep: sparse comparison R8 vs E34 vs G34, 10 independent trainings (sparse comparison document, part 2)
==========================================================================================================

Variants (all sqrt(m) [2,2,1] KAN, sparse_sqrtm.py):
    R8   manual selection, 8 fixed bases [8, 10, ..., 22]; 2 x 5000 epochs (second phase continued with a
         restarted learning-rate schedule); MSE saved after 5000 and 10000 epochs
    E34  34 bases, per-edge L1 penalty, phase 1 (5000) -> gap mask per coefficient -> refit (5000)
    G34  34 bases, group-lasso penalty, phase 1 (5000) -> gap mask per layer -> refit (5000)
Fixed beforehand on training data only (sparse_select.py, results_sparse_selection/):
    shared layer-1 input range for all variants and seeds, lambda (E34, G34), mask rule (largest gap).
What varies: only the initialisation seed (101-110); the refit initialisation uses seed + 1000.
Same for every variant and seed: the 1000 training samples (data seed 42), optimizer, learning rate
1e-3 -> 1e-4 per phase, float64.

Evaluation (never used for any decision):
    new independent test set: 2000 samples, uniform in [0, 1]^2, seed TEST_SEED
    original 300 test samples (link to earlier results)
Per run: MSE after every phase, MSE before / after masking, active bases per edge and per layer,
validity (u >= 0, P <= 60) on training, test and a 201 x 201 grid over [0, 1]^2, the layer-1 mapping,
test-loss history of the final phase (every 10 epochs) for the stopping-point statistics.

Usage:
    python seed_sweep.py [--seeds ...] [--variants ...] [--jobs N]     # runs (skips finished ones)
    python seed_sweep.py --summarize
Results: sqrtm_checks/results_seed_sweep/
"""

import argparse
import csv
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "results_seed_sweep")
SEL = os.path.join(HERE, "results_sparse_selection")

SEEDS = list(range(101, 111))
VARIANTS = ["R8", "E34", "G34"]
TEST_SEED = 2026
N_TEST_NEW = 2000
EPOCHS = 5000
RECORD_EVERY = 10
LAST_WINDOW = 500          # epochs at the end of the final phase used for the stopping-point statistics


def settings():
    with open(os.path.join(SEL, "step1_mapping.json")) as f:
        rng = tuple(json.load(f)["shared_layer1_range"])
    with open(os.path.join(SEL, "step2_lambda.json")) as f:
        d = json.load(f)
    lam = {"R8": 0.0, "E34": d["E34"]["lambda_chosen"], "G34": d["G34"]["lambda_chosen"]}
    return rng, lam


def new_test_set():
    import torch
    import sparse_sqrtm as S
    g = torch.Generator().manual_seed(TEST_SEED)
    x = torch.rand(N_TEST_NEW, 2, generator=g, dtype=torch.float64)
    return x, S.target_fn(x)


def grid(n=201):
    import torch
    g = torch.linspace(0, 1, n, dtype=torch.float64)
    X1, X2 = torch.meshgrid(g, g, indexing="xy")
    return torch.stack([X1.ravel(), X2.ravel()], dim=1)


def _job(args):
    variant, seed = args
    import torch
    torch.set_num_threads(1)
    import sparse_sqrtm as S   # adds the project folder (kan.py) to the path
    import kan

    rng, lam = settings()
    xt, yt, x300, y300 = S.notebook_data()
    x2k, y2k = new_test_set()
    r = S.run_variant(variant, rng, seed, xt, yt, lam=lam[variant], threshold="gap", epochs=EPOCHS,
                      refit_seed=seed + 1000, x_eval=x2k, y_eval=y2k, record_every=RECORD_EVERY)

    final = r["final_model"]
    out = {"variant": variant, "init_seed": seed, "refit_seed": seed + 1000 if variant != "R8" else None,
           "lambda": lam[variant], "layer1_range": list(rng), "test_seed": TEST_SEED,
           "seconds": r["seconds"]}

    # MSE per stage
    out["phase1"] = {"train_mse": r["phase1"]["train_mse"], "test2000_mse": r["phase1"]["eval_mse"],
                     "valid_train": r["phase1"]["validity"]["valid"]}
    if variant == "R8":
        out["after_5000"] = out["phase1"]
    else:
        out["masked"] = {"train_mse": r["masked"]["train_mse"], "test2000_mse": r["masked"]["eval_mse"]}
        out["threshold"] = r["threshold_used"]
        out["gap_ratio"] = r["gap_ratio"]
        out["mask"] = r["mask"]
        out["masked_coeffs_exactly_zero"] = r["refit"]["masked_coeffs_exactly_zero"]
    out["final"] = {"train_mse": S.mse(final, xt, yt), "test2000_mse": S.mse(final, x2k, y2k),
                    "test300_mse": S.mse(final, x300, y300)}

    # stopping point: test-2000 history of the final phase
    h = np.array(r["final"]["hist"]["eval_mse"])
    k = LAST_WINDOW // RECORD_EVERY
    tail = h[-k:]
    out["stopping"] = {"last_window_epochs": LAST_WINDOW, "mean": float(tail.mean()), "sd": float(tail.std(ddof=1)),
                       "min": float(tail.min()), "max": float(tail.max()),
                       "final_over_mean": float(h[-1] / tail.mean())}
    out["history_final_phase"] = {"record_every": RECORD_EVERY, "test2000_mse": h.tolist(),
                                  "train_mse": r["final"]["hist"]["train_mse"]}
    out["history_phase1"] = {"record_every": RECORD_EVERY, "test2000_mse": r["phase1"]["hist"]["eval_mse"],
                             "train_mse": r["phase1"]["hist"]["train_mse"]}

    # validity on training inputs, both test sets and a dense grid
    out["validity"] = {name: S.validity(final, x) for name, x in
                       [("train", xt), ("test2000", x2k), ("test300", x300), ("grid201", grid())]}
    out["valid_all"] = all(v["valid"] for v in out["validity"].values())
    out["hidden_range_train"] = S.hidden_range(final, xt)
    L1 = final.layers[1]
    out["layer1_alpha"], out["layer1_beta"] = float(L1.alpha[0]), float(L1.beta[0])
    out["layer0_alpha"], out["layer0_beta"] = float(final.layers[0].alpha[0]), float(final.layers[0].beta[0])

    os.makedirs(os.path.join(OUT, "runs"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
    kan.export_full_forward_to_mat(
        final, x300,
        os.path.join(OUT, "models", f"photonic_kan_2-2-1_sqrtm_{variant}_seed_{seed}_full_forward_test.mat"),
        extra={"variant": variant, "init_seed": seed, "lambda": lam[variant],
               "selected_indices_python": np.array(S.library(variant)[1]),
               "power_window": np.array([S.P_MIN, S.P_MAX]), "x_test": x300, "y_test": y300,
               "test_mse": out["final"]["test300_mse"], "test2000_mse": out["final"]["test2000_mse"],
               "train_mse": out["final"]["train_mse"]})
    with open(os.path.join(OUT, "runs", f"{variant}_seed_{seed}.json"), "w") as f:
        json.dump(out, f, indent=1)
    msg = f"{variant} seed {seed}: test2000 {out['final']['test2000_mse']:.3e}"
    if variant != "R8":
        msg += f" | active/edge {r['mask']['active_per_edge']} unique/layer {r['mask']['n_unique_per_layer']}"
    print(msg + f" | valid {out['valid_all']} | {r['seconds']:.0f} s", flush=True)
    return msg


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def load_runs():
    runs = {}
    for p in glob.glob(os.path.join(OUT, "runs", "*.json")):
        with open(p) as f:
            r = json.load(f)
        runs[(r["variant"], r["init_seed"])] = r
    return runs


def summarize():
    runs = load_runs()
    rng, lam = settings()
    seeds = sorted({s for (_, s) in runs})

    def get(v, s, path):
        r = runs.get((v, s))
        if r is None:
            return np.nan
        for p in path:
            r = r[p]
        return r

    # per-run CSV
    with open(os.path.join(OUT, "seed_sweep_runs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "init_seed", "lambda", "train_mse_final", "test2000_mse_final", "test300_mse_final",
                    "test2000_after_5000_or_phase1", "test2000_masked_before_refit", "train_masked_before_refit",
                    "train_phase1", "threshold", "gap_ratio", "active_per_edge", "mean_active_per_edge",
                    "n_unique_layer0", "n_unique_layer1", "n_active_coeffs", "unique_layer0", "unique_layer1",
                    "valid_all", "layer1_alpha", "layer1_beta", "stop_last500_mean", "stop_last500_sd",
                    "stop_final_over_mean", "seconds"])
        for (v, s), r in sorted(runs.items(), key=lambda q: (VARIANTS.index(q[0][0]), q[0][1])):
            m = r.get("mask", {})
            w.writerow([v, s, r["lambda"], r["final"]["train_mse"], r["final"]["test2000_mse"],
                        r["final"]["test300_mse"], r["phase1"]["test2000_mse"],
                        r.get("masked", {}).get("test2000_mse", ""), r.get("masked", {}).get("train_mse", ""),
                        r["phase1"]["train_mse"], r.get("threshold", ""), r.get("gap_ratio", ""),
                        " ".join(map(str, m.get("active_per_edge", [8] * 6))),
                        np.mean(m.get("active_per_edge", [8] * 6)),
                        m.get("n_unique_per_layer", [8, 8])[0], m.get("n_unique_per_layer", [8, 8])[1],
                        m.get("n_active_coeffs", 48),
                        " ".join(map(str, m.get("unique_per_layer", [[8, 10, 12, 14, 16, 18, 20, 22]] * 2)[0])),
                        " ".join(map(str, m.get("unique_per_layer", [[8, 10, 12, 14, 16, 18, 20, 22]] * 2)[1])),
                        r["valid_all"], r["layer1_alpha"], r["layer1_beta"], r["stopping"]["mean"],
                        r["stopping"]["sd"], r["stopping"]["final_over_mean"], r["seconds"]])

    # per-edge selected indices of E34 / G34
    with open(os.path.join(OUT, "seed_sweep_selected_indices.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "init_seed", "layer", "out", "in", "n_active", "active_indices"])
        for (v, s), r in sorted(runs.items(), key=lambda q: (VARIANTS.index(q[0][0]), q[0][1])):
            if v == "R8":
                continue
            for e in r["mask"]["per_edge"]:
                w.writerow([v, s, e["layer"], e["out"], e["in"], len(e["active"]), " ".join(map(str, e["active"]))])

    def stat(v, path):
        x = np.array([get(v, s, path) for s in seeds], dtype=float)
        x = x[np.isfinite(x)]
        return {"n": len(x), "mean": x.mean(), "sd": x.std(ddof=1), "min": x.min(), "max": x.max(),
                "median": float(np.median(x))}

    rows = [("R8 (5000 epochs)", "R8", ["after_5000", "test2000_mse"]),
            ("R8 (10000 epochs)", "R8", ["final", "test2000_mse"]),
            ("E34 after phase 1", "E34", ["phase1", "test2000_mse"]),
            ("E34 + refit", "E34", ["final", "test2000_mse"]),
            ("G34 after phase 1", "G34", ["phase1", "test2000_mse"]),
            ("G34 + refit", "G34", ["final", "test2000_mse"])]
    summary = []
    for label, v, path in rows:
        st = stat(v, path)
        st300 = stat(v, ["final", "test300_mse"]) if path[0] == "final" else None
        summary.append((label, v, st, st300))

    def activity(v):
        if v == "R8":
            return "8 per edge, same 8 in both layers (fixed)"
        a = np.array([np.mean(get(v, s, ["mask", "active_per_edge"])) for s in seeds])
        u0 = np.array([get(v, s, ["mask", "n_unique_per_layer"])[0] for s in seeds])
        u1 = np.array([get(v, s, ["mask", "n_unique_per_layer"])[1] for s in seeds])
        return (f"per edge {a.mean():.1f} ({a.min():.1f}–{a.max():.1f}); unique per layer "
                f"{u0.mean():.1f} ({u0.min()}–{u0.max()}) / {u1.mean():.1f} ({u1.min()}–{u1.max()})")

    with open(os.path.join(OUT, "seed_sweep_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "n_runs", "test2000_mean", "test2000_sd", "test2000_min", "test2000_max",
                    "test2000_median", "test300_mean", "test300_sd"])
        for label, v, st, st300 in summary:
            w.writerow([label, st["n"], st["mean"], st["sd"], st["min"], st["max"], st["median"],
                        st300["mean"] if st300 else "", st300["sd"] if st300 else ""])

    # paired comparison with R8 (10000 epochs), same seed
    paired = []
    r8 = np.array([get("R8", s, ["final", "test2000_mse"]) for s in seeds])
    for v in ("E34", "G34"):
        x = np.array([get(v, s, ["final", "test2000_mse"]) for s in seeds])
        d = x - r8
        paired.append((v, d, int(np.sum(d < 0)), x / r8))

    # figure: test MSE per seed for the three final models
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"R8": "#525252", "E34": "#2171b5", "G34": "#d94801"}
    labels = {"R8": "R8 (manual, 8 bases)", "E34": "E34 + refit (per-edge L1)", "G34": "G34 + refit (group lasso)"}
    fig, ax = plt.subplots(figsize=(8, 4.4), constrained_layout=True)
    xs = np.arange(len(seeds))
    for j, v in enumerate(VARIANTS):
        y = [get(v, s, ["final", "test2000_mse"]) for s in seeds]
        ax.plot(xs + (j - 1) * 0.18, y, "o", ms=7, color=colors[v], mec="white", mew=1.2, label=labels[v], zorder=3)
        ax.axhline(np.nanmean(y), color=colors[v], lw=1.2, ls="--", alpha=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_yscale("log")
    ax.set_xlabel("initialisation seed")
    ax.set_ylabel("test MSE (2000 new samples)")
    ax.grid(axis="y", which="both", alpha=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Test MSE per training (dashed: mean over 10 trainings)", fontsize=11)
    fig.savefig(os.path.join(OUT, "seed_sweep_test_mse.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # report
    def ms(st):
        return f"{st['mean']:.3e} ± {st['sd']:.3e}"

    T = ["| Model | Test MSE (2000), mean ± SD | Min – max | Median | Test MSE (300), mean ± SD |",
         "|---|---|---|---|---|"]
    for label, v, st, st300 in summary:
        T.append(f"| {label} | {ms(st)} | {st['min']:.3e} – {st['max']:.3e} | {st['median']:.3e} | "
                 f"{ms(st300) if st300 else '—'} |")
    A = [f"- {v}: {activity(v)}" for v in VARIANTS]
    Pd = []
    for v, d, wins, ratio in paired:
        Pd.append(f"- {v} + refit vs R8 (10000 epochs), same seed: better in {wins} / {len(seeds)} seeds; "
                  f"difference (variant − R8) mean {d.mean():.3e} ± {d.std(ddof=1):.3e}; "
                  f"ratio variant / R8 median {np.median(ratio):.2f} (range {ratio.min():.2f}–{ratio.max():.2f})")
    n_valid = {v: sum(get(v, s, ["valid_all"]) for s in seeds) for v in VARIANTS}
    stop = {v: np.array([get(v, s, ["stopping", "final_over_mean"]) for s in seeds]) for v in VARIANTS}
    stop_sd = {v: np.array([get(v, s, ["stopping", "sd"]) / get(v, s, ["stopping", "mean"]) for s in seeds])
               for v in VARIANTS}
    masking = {v: np.array([get(v, s, ["masked", "test2000_mse"]) / get(v, s, ["phase1", "test2000_mse"])
                            for s in seeds]) for v in ("E34", "G34")}
    gaps = {v: np.array([get(v, s, ["gap_ratio"]) for s in seeds]) for v in ("E34", "G34")}

    report = f"""# Seed sweep: R8 vs E34 vs G34, 10 independent trainings

## Test MSE

{chr(10).join(T)}

10 trainings per model (SD with n − 1); only the initialisation seed differs ({seeds[0]}–{seeds[-1]}).
Test set: {N_TEST_NEW} new samples, uniform in [0, 1]², seed {TEST_SEED}, the same for all models and seeds;
the original 300 test samples only for the link to earlier results.

## Paired comparison (same seed)

{chr(10).join(Pd)}

## Activity

{chr(10).join(A)}

Mask threshold in the largest gap of the sorted magnitudes; gap ratio E34 {gaps['E34'].min():.1f}–{gaps['E34'].max():.1f},
G34 {gaps['G34'].min():.1f}–{gaps['G34'].max():.1f}. Effect of masking before the refit (test MSE masked / phase 1):
E34 {np.median(masking['E34']):.4f} (median; {masking['E34'].min():.4f}–{masking['E34'].max():.4f}),
G34 {np.median(masking['G34']):.4f} ({masking['G34'].min():.4f}–{masking['G34'].max():.4f}): the gap threshold
only removes coefficients that are already close to zero.
Selected indices per edge / layer: `seed_sweep_selected_indices.csv`.

## Validity and stopping point

- Valid (u >= 0, P <= 60, finite) on training inputs, both test sets and a 201 x 201 grid over [0, 1]²:
  R8 {n_valid['R8']} / {len(seeds)}, E34 {n_valid['E34']} / {len(seeds)}, G34 {n_valid['G34']} / {len(seeds)}.
- Final value relative to the mean test MSE of the last {LAST_WINDOW} epochs of the final phase
  (1 = the run stopped at its plateau level): R8 {stop['R8'].min():.3f}–{stop['R8'].max():.3f},
  E34 {stop['E34'].min():.3f}–{stop['E34'].max():.3f}, G34 {stop['G34'].min():.3f}–{stop['G34'].max():.3f};
  relative SD within that window: max R8 {stop_sd['R8'].max():.1e}, E34 {stop_sd['E34'].max():.1e},
  G34 {stop_sd['G34'].max():.1e}.

## Settings

- sqrt(m) [2,2,1] KAN, power window [0.05, 8], layer-0 input range [-0.2, 1], shared layer-1 input range
  [{rng[0]:.4f}, {rng[1]:.4f}] (all variants and seeds); only expansion coefficients are trained.
- Adam, learning rate 1e-3 -> 1e-4 within each 5000-epoch phase, float64; 1000 training samples (data seed 42).
- R8: 2 x 5000 epochs (second phase continued, schedule restarted). E34: lambda {lam['E34']:g}; G34: lambda
  {lam['G34']:g}; phase 1 (5000) -> mask -> refit (5000, initialisation seed + 1000, masked coefficients
  exactly zero and frozen, no penalty).
- Mapping, lambda and mask rule were fixed on training data only (`results_sparse_selection/`); the test sets
  were not used for any decision.

## Files

- `seed_sweep_runs.csv`: one row per run (30); `seed_sweep_summary.csv`; `seed_sweep_selected_indices.csv`
- `seed_sweep_test_mse.png`: test MSE per seed and model
- `runs/`: everything per run, incl. loss histories; `models/`: full forward export of every final model
"""
    with open(os.path.join(OUT, "seed_sweep_report.md"), "w") as f:
        f.write(report)
    print(report)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if not a.summarize:
        tasks = [(v, s) for v in a.variants for s in a.seeds
                 if not os.path.exists(os.path.join(OUT, "runs", f"{v}_seed_{s}.json"))]
        # longest jobs first
        tasks.sort(key=lambda t: t[0] == "R8")
        with ProcessPoolExecutor(a.jobs) as ex:
            list(ex.map(_job, tasks))
    summarize()


if __name__ == "__main__":
    main()
