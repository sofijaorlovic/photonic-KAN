"""
Run A: repeated training of the standard sqrt(m) [2,2,1] KAN
============================================================

Question: how much does the result of the standard model depend on the random weight
initialisation, i.e. is the single test MSE of the notebook typical?

Exactly the procedure of affine_kan_2-2-1_sqrtm.ipynb (the notebook itself is not run or changed):
    y = 0.3 x1^2 + cos(3 x2), x in [0, 1]^2; 1000 training and 300 test samples (data seed 42)
    8 bases [8, 10, 12, 14, 16, 18, 20, 22], field encoding sqrt(b(u^2/m)), power window [0.05, 8]
    layer 0 input range [-0.2, 1]
    hidden-range calibration: probe run (1000 epochs, layer-1 range +-10, constant lr 1e-3) ->
        layer-1 range = probe range x 1.1 -> full training -> check u >= 0, P <= 60 on the training
        inputs; if it fails, re-map to the measured range x 1.1 and retrain (max 10 passes)
    Adam, 5000 epochs, learning rate 1e-3 -> 1e-4 (exponential decay), float64, CPU

The ONLY thing that varies between runs: the seed of the weight initialisation. The notebook
initialises the model from the random state right after the data; here that state is replaced by
torch.manual_seed(seed). As in the notebook, the probe and every calibration pass of one run start
from the same initialisation. Training and test data are identical in every run.

Reference mode (--reference): uses the notebook's own random state (the state after the data) and
must reproduce the exported model; it checks the script against the notebook.

Usage:
    python run_A.py --reference                  # check the script against the exported model
    python run_A.py --seeds 101 102 ... 110      # the repetitions (default 101-110)
    python run_A.py --summarize                  # summary table and report from the finished runs

Outputs (sqrtm_checks/results_run_A/):
    runs/seed_<s>.json                     per run: settings, calibration log, MSE, validity, time
    models/photonic_kan_2-2-1_sqrtm_seed_<s>_full_forward_test.mat
                                           full forward export of every trained model (same format as
                                           the notebook export; rebuild.py / domain_check.py can read it)
    loss_curves/seed_<s>.mat               train / test loss and learning rate per epoch (final pass)
    run_A_runs.csv, run_A_summary.csv, run_A_test_mse.png, run_A_report.md   (after --summarize)
"""

import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)
sys.path.insert(0, HERE)

OUT = os.path.join(HERE, "results_run_A")
REFERENCE_EXPORT = os.path.join(PROJECT, "new_consistent_affine_2-2-1_sqrtm_exports",
                                "photonic_kan_2-2-1_sqrtm_full_forward_test.mat")

# ---- settings of affine_kan_2-2-1_sqrtm.ipynb (unchanged)
SELECTED_INDICES = [8, 10, 12, 14, 16, 18, 20, 22]
P_MIN, P_MAX = 0.05, 8.0
DIVIDE_BY_M = True
INPUT_NEG_MAX = [0.2, 3.0]
INPUT_POS_MAX = [1.0, 3.0]
PROBE_EPOCHS = 1000
PROBE_RANGE = 10.0
PROBE_MARGIN = 1.10
CALIB_MARGIN = 1.10
MAX_PASSES = 10
DATA_SEED = 42
N_TRAIN, N_TEST = 1000, 300
X_MIN, X_MAX = 0.0, 1.0
NUM_EPOCHS = 5000
LR, LR_FINAL = 1e-3, 1e-4

DEFAULT_SEEDS = list(range(101, 111))


# ---------------------------------------------------------------------------
# Training (torch only needed here)
# ---------------------------------------------------------------------------

def train_one(seed, reference=False, log_every=1000):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from scipy.io import savemat
    import kan
    from kan import KAN

    torch.set_num_threads(1)
    device = torch.device("cpu")
    t0 = time.time()

    b_coef, input_power, _, set_detuning = kan.load_photonic_activation_data(
        os.path.join(PROJECT, "activation_KAN_fit_B_Sofija_extend_positive_basisKAN.mat"))
    b_coef_selected = b_coef[SELECTED_INDICES]
    P_FIT_MAX = float(input_power.max())

    # data exactly as in the notebook
    torch.manual_seed(DATA_SEED)
    x_train = X_MIN + (X_MAX - X_MIN) * torch.rand(N_TRAIN, 2)
    x_test = X_MIN + (X_MAX - X_MIN) * torch.rand(N_TEST, 2)

    def target_fn(x):
        return 0.3 * x[:, 0:1] ** 2 + torch.cos(3.0 * x[:, 1:2])

    y_train, y_test = target_fn(x_train), target_fn(x_test)
    rng_state_after_data = torch.get_rng_state()
    x_train, y_train, x_test, y_test = [t.to(device=device, dtype=torch.float64)
                                        for t in (x_train, y_train, x_test, y_test)]

    # initialisation state of this run: the notebook's state after the data (reference) or the seed
    if reference:
        init_state = rng_state_after_data
    else:
        torch.manual_seed(seed)
        init_state = torch.get_rng_state()

    def build_model(layer1_range=None, margin=1.0):
        torch.set_rng_state(init_state)   # same initialisation in the probe and in every pass
        model = KAN(
            in_count=2, out_count=1, hidden_layer_sizes=[2],
            layer_type="photonic_interval_affine_clean",
            b_coef_selected=b_coef_selected,
            input_neg_max=INPUT_NEG_MAX, input_pos_max=INPUT_POS_MAX,
            basis_min=P_MIN, basis_max=P_MAX,
            dropout_prob=0.0, debug=False,
            field_encoding=True, divide_by_m=DIVIDE_BY_M,
        ).to(device=device, dtype=torch.float64)
        if layer1_range is not None:
            model.layers[1].set_input_range(*layer1_range, margin=margin)
        return model

    loss_fn = nn.MSELoss()

    def train_model(model):
        optimizer = optim.Adam(model.parameters(), lr=LR)
        gamma = (LR_FINAL / LR) ** (1.0 / max(NUM_EPOCHS - 1, 1))
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        train_losses, test_losses, lrs = [], [], []
        for epoch in range(NUM_EPOCHS):
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(model(x_train), y_train)
            loss.backward()
            optimizer.step()
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()
            model.eval()
            with torch.no_grad():
                test_loss = loss_fn(model(x_test), y_test)
            train_losses.append(loss.item())
            test_losses.append(test_loss.item())
            if log_every and (epoch % log_every == 0 or epoch == NUM_EPOCHS - 1):
                print(f"  [{tag}] epoch {epoch:5d} | train {loss.item():.3e} | test {test_loss.item():.3e}",
                      flush=True)
        return train_losses, test_losses, lrs

    def hidden_range(model, x):
        model.eval()
        with torch.no_grad():
            _, acts = model(x, return_all=True)
        return acts[1].min().item(), acts[1].max().item()

    def layer1_validity(model, x):
        model.eval()
        with torch.no_grad():
            _, stats = model(x, track_stats=True)
        return stats[1]["min_affine_input"].item(), stats[1]["max_P"].item()

    tag = "reference" if reference else f"seed {seed}"

    # 1. probe run
    probe = build_model((-PROBE_RANGE, PROBE_RANGE))
    opt = optim.Adam(probe.parameters(), lr=LR)
    for _ in range(PROBE_EPOCHS):
        probe.train()
        opt.zero_grad()
        loss = loss_fn(probe(x_train), y_train)
        loss.backward()
        opt.step()
    z_probe = hidden_range(probe, x_train)
    print(f"  [{tag}] probe: hidden z in [{z_probe[0]:.4f}, {z_probe[1]:.4f}]", flush=True)

    # 2.-4. calibration passes
    layer1_range, margin = z_probe, PROBE_MARGIN
    calibration_log = []
    valid = False
    for p in range(MAX_PASSES):
        model = build_model(layer1_range, margin)
        L1 = model.layers[1]
        train_losses, test_losses, lrs = train_model(model)
        z_min, z_max = hidden_range(model, x_train)
        u_min, P_max_obs = layer1_validity(model, x_train)
        valid = (u_min >= 0.0) and (P_max_obs <= P_FIT_MAX)
        calibration_log.append([p + 1, L1.input_min, L1.input_max, z_min, z_max, u_min, P_max_obs, test_losses[-1]])
        print(f"  [{tag}] pass {p + 1}: layer-1 range [{L1.input_min:.4f}, {L1.input_max:.4f}] -> "
              f"{'VALID' if valid else 'invalid, re-mapping'}", flush=True)
        if valid:
            break
        layer1_range, margin = (z_min, z_max), CALIB_MARGIN

    u_min_test, P_max_test = layer1_validity(model, x_test)
    test_valid = (u_min_test >= 0.0) and (P_max_test <= P_FIT_MAX)

    model.eval()
    with torch.no_grad():
        train_mse = loss_fn(model(x_train), y_train).item()
        test_mse = loss_fn(model(x_test), y_test).item()

    name = "reference" if reference else f"seed_{seed}"
    for sub in ("runs", "models", "loss_curves"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)

    calibration_log = np.array(calibration_log)
    export_path = os.path.join(OUT, "models", f"photonic_kan_2-2-1_sqrtm_{name}_full_forward_test.mat")
    err = kan.export_full_forward_to_mat(model, x_test, export_path, extra={
        "model_description": "[2,2,1] photonic KAN with affine mapping, field encoding: "
                             "edge = sum_k a_k sqrt(b_k((alpha*x+beta)^2/m)); Run A repetition",
        "target": "y = 0.3*x1^2 + cos(3*x2)",
        "power_window": np.array([P_MIN, P_MAX]),
        "selected_indices_python": np.array(SELECTED_INDICES),
        "selected_indices_matlab": np.array(SELECTED_INDICES) + 1,
        "selected_detunings": set_detuning[SELECTED_INDICES],
        "x_test": x_test, "y_test": y_test, "y_pred_test": model(x_test).detach(),
        "train_mse": train_mse, "test_mse": test_mse,
        "calibration_log": calibration_log, "calib_margin": CALIB_MARGIN,
        "probe_epochs": PROBE_EPOCHS, "probe_range": PROBE_RANGE, "probe_margin": PROBE_MARGIN,
        "probe_hidden_range": np.array(z_probe),
        "data_seed": DATA_SEED, "init_seed": -1 if reference else seed,
        "num_epochs": NUM_EPOCHS, "lr": LR, "lr_final": LR_FINAL,
    })
    assert err < 1e-8, "PyTorch and NumPy forward passes disagree"

    savemat(os.path.join(OUT, "loss_curves", f"{name}.mat"), {
        "epoch": np.arange(NUM_EPOCHS), "train_losses": np.array(train_losses),
        "test_losses": np.array(test_losses), "learning_rate": np.array(lrs),
        "init_seed": -1 if reference else seed,
    })

    L1 = model.layers[1]
    result = {
        "run": name, "init_seed": None if reference else seed, "data_seed": DATA_SEED,
        "train_mse": train_mse, "test_mse": test_mse,
        "passes": len(calibration_log), "train_valid": bool(valid), "test_valid": bool(test_valid),
        "probe_z_min": z_probe[0], "probe_z_max": z_probe[1],
        "layer1_input_min": float(L1.input_min), "layer1_input_max": float(L1.input_max),
        "layer1_alpha": float(L1.alpha[0]), "layer1_beta": float(L1.beta[0]),
        "hidden_min_train": float(calibration_log[-1][3]), "hidden_max_train": float(calibration_log[-1][4]),
        "layer1_min_u_train": float(calibration_log[-1][5]), "layer1_max_P_train": float(calibration_log[-1][6]),
        "layer1_min_u_test": u_min_test, "layer1_max_P_test": P_max_test,
        "best_test_mse_during_training": float(np.min(test_losses)),
        "calibration_log": calibration_log.tolist(),
        "torch_version": torch.__version__, "seconds": time.time() - t0,
    }
    with open(os.path.join(OUT, "runs", f"{name}.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(f"  [{tag}] done: train MSE {train_mse:.4e}, test MSE {test_mse:.4e}, "
          f"{len(calibration_log)} pass(es), {result['seconds']:.0f} s", flush=True)
    return result


def check_reference():
    """The reference run must reproduce the exported notebook model."""
    import rebuild
    ref = rebuild.load_model(REFERENCE_EXPORT)
    new = rebuild.load_model(os.path.join(OUT, "models", "photonic_kan_2-2-1_sqrtm_reference_full_forward_test.mat"))
    rows = []
    for L, (a, b) in enumerate(zip(ref["layers"], new["layers"])):
        for key in ("coeffs", "alpha", "beta"):
            rel = float(np.max(np.abs(a[key] - b[key])) / max(np.max(np.abs(a[key])), 1e-300))
            rows.append((f"layer {L} {key}", rel))
    mse_ref, mse_new = ref["test_mse_exported"], new["test_mse_exported"]
    rows.append(("test MSE", abs(mse_new - mse_ref) / mse_ref))
    print(f"Reference run vs exported notebook model (test MSE {mse_new:.6e} vs {mse_ref:.6e}):")
    for name, rel in rows:
        print(f"  {name:<18s} max relative difference {rel:.2e}")
    worst = max(r[1] for r in rows)
    print("  -> script reproduces the notebook" if worst < 1e-6 else
          "  -> DIFFERENCE: check the settings (or platform-dependent floating point)")
    return worst


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize():
    import rebuild
    runs = []
    for path in sorted(glob.glob(os.path.join(OUT, "runs", "seed_*.json"))):
        with open(path) as f:
            runs.append(json.load(f))
    runs.sort(key=lambda r: r["init_seed"])
    if not runs:
        sys.exit("No finished runs in " + os.path.join(OUT, "runs"))
    ref_mse = rebuild.load_model(REFERENCE_EXPORT)["test_mse_exported"]

    cols = ["init_seed", "train_mse", "test_mse", "passes", "train_valid", "test_valid",
            "probe_z_min", "probe_z_max", "layer1_input_min", "layer1_input_max", "layer1_alpha", "layer1_beta",
            "hidden_min_train", "hidden_max_train", "layer1_min_u_train", "layer1_max_P_train",
            "layer1_min_u_test", "layer1_max_P_test", "best_test_mse_during_training", "seconds"]
    with open(os.path.join(OUT, "run_A_runs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in runs:
            w.writerow([r[c] for c in cols])

    def stats(key):
        v = np.array([r[key] for r in runs], dtype=float)
        return {"mean": v.mean(), "sd": v.std(ddof=1) if len(v) > 1 else float("nan"),
                "min": v.min(), "max": v.max(), "median": float(np.median(v))}

    S = {k: stats(k) for k in ("train_mse", "test_mse")}
    test = np.array([r["test_mse"] for r in runs])
    n_better = int(np.sum(test < ref_mse))
    n_valid = sum(r["train_valid"] for r in runs)
    n_test_valid = sum(r["test_valid"] for r in runs)
    passes = [r["passes"] for r in runs]

    with open(os.path.join(OUT, "run_A_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "n_runs", "mean", "sd", "min", "max", "median"])
        for k, s in S.items():
            w.writerow([k, len(runs), s["mean"], s["sd"], s["min"], s["max"], s["median"]])

    # figure: test MSE per seed and the single notebook value
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    seeds = [r["init_seed"] for r in runs]
    ax.axhspan(S["test_mse"]["mean"] - S["test_mse"]["sd"], S["test_mse"]["mean"] + S["test_mse"]["sd"],
               color="#c6dbef", alpha=0.6, lw=0, label="mean ± SD")
    ax.axhline(S["test_mse"]["mean"], color="#2171b5", lw=1.5, label="mean")
    ax.axhline(ref_mse, color="#636363", lw=1.5, ls="--", label=f"notebook run ({ref_mse:.2e})")
    ax.plot(range(len(seeds)), test, "o", ms=8, color="#08519c", mec="white", mew=1.5, label="test MSE per run",
            zorder=3)
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("initialisation seed")
    ax.set_ylabel("test MSE")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(axis="y", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"Standard sqrt(m) [2,2,1] KAN: {len(runs)} trainings, only the initialisation differs", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.savefig(os.path.join(OUT, "run_A_test_mse.png"), dpi=150)
    plt.close(fig)

    def ms(s):
        return f"{s['mean']:.3e} ± {s['sd']:.3e}"

    lines = ["| Seed | Train MSE | Test MSE | Passes | Layer-1 range | Valid (train / test) |",
             "|---|---|---|---|---|---|"]
    for r in runs:
        lines.append(f"| {r['init_seed']} | {r['train_mse']:.3e} | {r['test_mse']:.3e} | {r['passes']} | "
                     f"[{r['layer1_input_min']:.3f}, {r['layer1_input_max']:.3f}] | "
                     f"{'yes' if r['train_valid'] else 'NO'} / {'yes' if r['test_valid'] else 'NO'} |")

    report = f"""# Run A: repeated training of the standard sqrt(m) [2,2,1] KAN

## Result

| | Mean ± SD | Min – max | Median |
|---|---|---|---|
| Train MSE | {ms(S['train_mse'])} | {S['train_mse']['min']:.3e} – {S['train_mse']['max']:.3e} | {S['train_mse']['median']:.3e} |
| Test MSE | {ms(S['test_mse'])} | {S['test_mse']['min']:.3e} – {S['test_mse']['max']:.3e} | {S['test_mse']['median']:.3e} |

{len(runs)} independent trainings (SD with n − 1). Notebook run: test MSE {ref_mse:.3e};
{n_better} of {len(runs)} runs have a lower test MSE.
Calibration: {n_valid} / {len(runs)} runs physically valid on the training inputs
(u >= 0, P <= 60 in layer 1), {n_test_valid} / {len(runs)} also on the test inputs;
passes needed: {min(passes)}–{max(passes)} (mean {np.mean(passes):.1f}).

## What varies, what is fixed

- Varies: only the seed of the weight initialisation ({', '.join(str(s) for s in seeds)}); hence also the
  calibrated layer-1 mapping of each run.
- Fixed: architecture, the 8 bases, sqrt(m) edge model, power window [0.05, 8], layer-0 input range,
  1000 training and 300 test samples (data seed {DATA_SEED}), Adam, learning rate {LR:g} -> {LR_FINAL:g},
  {NUM_EPOCHS} epochs, calibration procedure (probe {PROBE_EPOCHS} epochs, margin {PROBE_MARGIN}), float64, CPU.
- The script was checked against the notebook: with the notebook's own random state it reproduces the
  exported model (`--reference`).

## Per run

{chr(10).join(lines)}

## Files

- `run_A_runs.csv`: one row per run (MSE, calibration, layer-1 mapping, validity, time)
- `run_A_summary.csv`: mean, SD, min, max, median
- `run_A_test_mse.png`: test MSE per run
- `runs/`, `models/`, `loss_curves/`: per-run details, full forward exports, loss curves
"""
    with open(os.path.join(OUT, "run_A_report.md"), "w") as f:
        f.write(report)
    print(report)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()

    if args.reference:
        train_one(None, reference=True)
        check_reference()
    if args.seeds or not (args.reference or args.summarize):
        for s in args.seeds or DEFAULT_SEEDS:
            train_one(s)
    if args.summarize or not (args.reference or args.seeds):
        summarize()


if __name__ == "__main__":
    main()
