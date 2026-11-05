"""
Sparse basis selection for the sqrt(m) [2,2,1] KAN (port of kan_sparse_basis.py / sparse_basis_221_demo.ipynb)
==============================================================================================================

Same edge model and training style as the standard sqrt(m) model (affine_kan_2-2-1_sqrtm.ipynb, run_A.py):
    edge = sum_k a_k sqrt(b_k((alpha x + beta)^2 / m)), power window [0.05, 8], layer-0 input range [-0.2, 1],
    Adam, learning rate 1e-3 -> 1e-4 (exponential decay over each training phase), float64, CPU,
    y = 0.3 x1^2 + cos(3 x2), 1000 training samples (data seed 42).
Only the expansion coefficients are trained; alpha, beta are fixed.

Variants (sparse comparison document):
    R8   8 fixed bases [8, 10, ..., 22] on every edge
    E34  all 34 bases, per-edge L1 penalty  lam * sum |a|;           mask per coefficient: |a| > threshold
    G34  all 34 bases, group-lasso penalty  lam * sum_k ||a_.,.,k||;  mask per layer: basis k is kept on
         every edge of the layer if its group norm ||a_.,.,k|| (over all edges of the layer) > threshold
E34/G34: phase 1 training with the penalty -> mask (threshold in the largest gap of the sorted magnitudes)
-> refit: new seeded initialisation, mask fixed (masked coefficients exactly zero and frozen), no penalty.
R8: phase 1 -> phase 2 continued from the phase-1 coefficients with a restarted learning-rate schedule,
so all variants get 2 x 5000 epochs and the R8 value after 5000 epochs is a normal 5000-epoch result.

The layer-1 mapping is NOT calibrated per run: it is passed in (layer1_range) and is the same for all
variants and seeds (sparse comparison document, point 6).

With m bases per edge the splitter is 1:m (P = u^2/m); the affine map sends the layer input range onto the
amplitude window [sqrt(m P_min), sqrt(m P_max)], so the power window is [0.05, 8] for m = 8 and m = 34 alike.
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)

import kan  # noqa: E402
from kan_sparse_basis import SparseKAN  # noqa: E402

ACTIVATION_MAT = os.path.join(PROJECT, "activation_KAN_fit_B_Sofija_extend_positive_basisKAN.mat")

R8_INDICES = [8, 10, 12, 14, 16, 18, 20, 22]
P_MIN, P_MAX = 0.05, 8.0
DIVIDE_BY_M = True
LAYER0_RANGE = (-0.2, 1.0)
DATA_SEED = 42
N_TRAIN, N_TEST = 1000, 300
LR, LR_FINAL = 1e-3, 1e-4

_ACT = None


def activation_library():
    global _ACT
    if _ACT is None:
        b_coef, input_power, _, set_detuning = kan.load_photonic_activation_data(ACTIVATION_MAT)
        _ACT = {"b_coef": b_coef, "P_fit_max": float(input_power.max()), "set_detuning": set_detuning}
    return _ACT


def target_fn(x):
    return 0.3 * x[:, 0:1] ** 2 + torch.cos(3.0 * x[:, 1:2])


def notebook_data():
    """Training set and original 300-sample test set, exactly as in the notebooks (seed 42)."""
    torch.manual_seed(DATA_SEED)
    x_train = torch.rand(N_TRAIN, 2)
    x_test = torch.rand(N_TEST, 2)
    y_train, y_test = target_fn(x_train), target_fn(x_test)
    return [t.double() for t in (x_train, y_train, x_test, y_test)]


def library(variant):
    b = activation_library()["b_coef"]
    return (b[R8_INDICES], list(R8_INDICES)) if variant == "R8" else (b, list(range(b.shape[0])))


def build(variant, layer1_range, init_seed):
    """Model of a variant with the given fixed layer-1 input range; initialisation from init_seed."""
    b_sel, _ = library(variant)
    torch.manual_seed(init_seed)
    model = SparseKAN(
        in_count=2, out_count=1, hidden_layer_sizes=[2],
        layer_type="photonic_interval_affine_clean",
        b_coef_selected=b_sel,
        input_neg_max=[-LAYER0_RANGE[0], -layer1_range[0]],
        input_pos_max=[LAYER0_RANGE[1], layer1_range[1]],
        basis_min=P_MIN, basis_max=P_MAX,
        dropout_prob=0.0, debug=False,
        field_encoding=True, divide_by_m=DIVIDE_BY_M,
    ).double()
    for layer in model.layers:
        layer.warn_negative_u = False      # reported by the validity check instead
    return model


def penalty(model, variant):
    if variant == "E34":
        return model.l1_penalty()
    if variant == "G34":
        return model.group_penalty()
    return None


def train(model, x, y, epochs, variant=None, lam=0.0, x_eval=None, y_eval=None, record_every=1):
    """Adam, lr 1e-3 -> 1e-4 over `epochs`; loss = MSE + lam * penalty (penalty only if lam > 0).
    Returns per-epoch train MSE (without penalty) and, if given, evaluation MSE."""
    opt = optim.Adam(model.parameters(), lr=LR)
    gamma = (LR_FINAL / LR) ** (1.0 / max(epochs - 1, 1))
    sched = optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
    mse_fn = nn.MSELoss()
    hist = {"train_mse": [], "eval_mse": []}
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        mse = mse_fn(model(x), y)
        loss = mse + lam * penalty(model, variant) if lam > 0 else mse
        loss.backward()
        opt.step()
        sched.step()
        if epoch % record_every == 0 or epoch == epochs - 1:
            hist["train_mse"].append(mse.item())
            if x_eval is not None:
                model.eval()
                with torch.no_grad():
                    hist["eval_mse"].append(mse_fn(model(x_eval), y_eval).item())
    return hist


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------

def mask_values(model, variant):
    """The magnitudes the mask is decided on: |a| of every coefficient (E34) or the group norm of every
    basis in every layer (G34)."""
    if variant == "E34":
        return torch.cat([l.coeffs.detach().abs().flatten() for l in model.layers])
    return torch.cat([l.coeffs.detach().pow(2).sum(dim=(0, 1)).sqrt() for l in model.layers])


def gap_threshold(values):
    """Threshold in the largest gap of the sorted magnitudes (largest ratio between neighbours);
    returns (threshold = geometric mean of the two values at the gap, gap ratio)."""
    v = np.sort(np.asarray(values, dtype=float))
    v = np.maximum(v, 1e-300)
    ratios = v[1:] / v[:-1]
    i = int(np.argmax(ratios))
    return float(np.sqrt(v[i] * v[i + 1])), float(ratios[i])


def make_masks(model, variant, threshold):
    """E34: per coefficient |a| > threshold. G34: per layer, basis k kept on all edges of the layer if
    the group norm over the layer's edges > threshold. threshold = "gap": rule gap_threshold."""
    if threshold == "gap":
        threshold, _ = gap_threshold(mask_values(model, variant).cpu().numpy())
    masks = []
    for layer in model.layers:
        a = layer.coeffs.detach()
        if variant == "E34":
            masks.append((a.abs() > threshold).double())
        elif variant == "G34":
            keep_k = (a.pow(2).sum(dim=(0, 1)).sqrt() > threshold).double()      # (num_basis,)
            masks.append(keep_k.view(1, 1, -1).expand_as(a).clone())
        else:
            raise ValueError(variant)
    return masks


def apply_mask_copy(model, masks):
    """Set the masked coefficients of `model` to zero (in place)."""
    with torch.no_grad():
        for layer, m in zip(model.layers, masks):
            layer.coeffs.mul_(m)


def mask_summary(masks):
    """Active counts per edge, unique active bases per layer, total active coefficients."""
    per_edge, per_layer = [], []
    for L, m in enumerate(masks):
        O, I, K = m.shape
        for o in range(O):
            for i in range(I):
                per_edge.append({"layer": L, "out": o, "in": i,
                                 "active": torch.nonzero(m[o, i] > 0).flatten().tolist()})
        per_layer.append(torch.nonzero(m.sum(dim=(0, 1)) > 0).flatten().tolist())
    return {
        "per_edge": per_edge,
        "active_per_edge": [len(e["active"]) for e in per_edge],
        "unique_per_layer": per_layer,
        "n_unique_per_layer": [len(u) for u in per_layer],
        "n_active_coeffs": int(sum(m.sum().item() for m in masks)),
    }


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def validity(model, x):
    """u >= 0 and P within [0, P_fit_max] in every layer; finite output."""
    model.eval()
    with torch.no_grad():
        y, stats = model(x, track_stats=True)
    P_fit = activation_library()["P_fit_max"]
    out = {"finite": bool(torch.isfinite(y).all())}
    for L, s in enumerate(stats):
        out[f"L{L}_min_u"] = s["min_affine_input"].item()
        out[f"L{L}_min_P"] = s["min_P"].item()
        out[f"L{L}_max_P"] = s["max_P"].item()
    out["valid"] = out["finite"] and all(out[f"L{L}_min_u"] >= 0 and out[f"L{L}_max_P"] <= P_fit
                                         for L in range(len(stats)))
    return out


def hidden_range(model, x):
    model.eval()
    with torch.no_grad():
        _, acts = model(x, return_all=True)
    return acts[1].min().item(), acts[1].max().item()


def mse(model, x, y):
    model.eval()
    with torch.no_grad():
        return nn.functional.mse_loss(model(x), y).item()


# ---------------------------------------------------------------------------
# One full run of a variant
# ---------------------------------------------------------------------------

def run_variant(variant, layer1_range, init_seed, x_train, y_train, lam=0.0, threshold=None,
                epochs=5000, refit_seed=None, x_eval=None, y_eval=None, record_every=1):
    """
    R8:      phase 1 (`epochs`) -> phase 2: training continued from the phase-1 coefficients for another
             `epochs` with a restarted learning-rate schedule (1e-3 -> 1e-4); no penalty.
    E34/G34: phase 1 (`epochs`, penalty lam) -> mask (threshold, or "gap") -> refit (`epochs`, new
             initialisation refit_seed, masked coefficients frozen at zero, no penalty).
    Returns a dict with models, masks, MSEs before/after masking, histories and validity.
    """
    t0 = time.time()
    res = {"variant": variant, "init_seed": init_seed, "layer1_range": list(layer1_range), "lam": lam,
           "threshold": threshold}
    model = build(variant, layer1_range, init_seed)
    h1 = train(model, x_train, y_train, epochs, variant, lam, x_eval, y_eval, record_every)
    res["phase1"] = {"hist": h1, "train_mse": mse(model, x_train, y_train),
                     "validity": validity(model, x_train), "hidden_range": hidden_range(model, x_train)}
    if x_eval is not None:
        res["phase1"]["eval_mse"] = mse(model, x_eval, y_eval)
    res["phase1_model"] = model

    if variant == "R8":
        h2 = train(model, x_train, y_train, epochs, None, 0.0, x_eval, y_eval, record_every)
        res["phase2"] = {"hist": h2, "train_mse": mse(model, x_train, y_train),
                         "validity": validity(model, x_train), "hidden_range": hidden_range(model, x_train)}
        if x_eval is not None:
            res["phase2"]["eval_mse"] = mse(model, x_eval, y_eval)
        res["phase1_model"] = None          # phase-1 values are in res["phase1"]; model continued
        res["final_model"] = model

    if variant != "R8":
        thr = threshold
        if threshold == "gap":
            thr, ratio = gap_threshold(mask_values(model, variant).cpu().numpy())
            res["gap_ratio"] = ratio
        res["threshold_used"] = thr
        masks = make_masks(model, variant, thr)
        res["mask"] = mask_summary(masks)
        masked = build(variant, layer1_range, init_seed)
        masked.load_state_dict(model.state_dict())
        apply_mask_copy(masked, masks)
        res["masked"] = {"train_mse": mse(masked, x_train, y_train)}
        if x_eval is not None:
            res["masked"]["eval_mse"] = mse(masked, x_eval, y_eval)

        refit = build(variant, layer1_range, refit_seed if refit_seed is not None else init_seed + 1000)
        refit.freeze_mask_(masks)                    # masked coefficients zero, gradient masked
        h2 = train(refit, x_train, y_train, epochs, None, 0.0, x_eval, y_eval, record_every)
        zero_ok = all(bool((l.coeffs.detach()[m == 0] == 0).all()) for l, m in zip(refit.layers, masks))
        res["refit"] = {"hist": h2, "train_mse": mse(refit, x_train, y_train),
                        "validity": validity(refit, x_train), "hidden_range": hidden_range(refit, x_train),
                        "masked_coeffs_exactly_zero": zero_ok}
        if x_eval is not None:
            res["refit"]["eval_mse"] = mse(refit, x_eval, y_eval)
        res["refit_model"] = refit
        res["final_model"] = refit
    res["final"] = res["phase2"] if variant == "R8" else res["refit"]
    res["seconds"] = time.time() - t0
    return res
