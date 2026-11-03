"""
Rebuild of the trained sqrt(m) photonic KAN from its full-forward-pass export
=============================================================================

The model is rebuilt from the exported .mat file only (no notebook rerun, no PyTorch):

    u    = alpha * h + beta                  affine map of the layer input h
    P    = u^2 / m                           FP-LD input power (1:m splitter)
    b_k  = b(P - delta_k; b_coef_k)          analytic FP-LD activation (delta = 0: nominal)
    s_k  = sqrt(b_k)                         field amplitude of basis k
    phi  = sum_k a_qpk * (1 + eps_qpk) * s_k edge function (eps = 0: nominal)
    z_q  = sum_p phi_qp                      node value

delta (horizontal shift of each basis, in units of P) and eps (relative coefficient error)
are optional and are used only by the sensitivity test (variant_I.py).

Valid domain of the activation: the measured input-power range of the FP-LD data the
fit was made on (activation .mat, field input_power), common to all bases.

Only NumPy / SciPy are needed.
"""

import os

import numpy as np
from scipy.io import loadmat

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)

ACTIVATION_MAT = os.path.join(PROJECT, "activation_KAN_fit_B_Sofija_extend_positive_basisKAN.mat")
EXPORT_221 = os.path.join(PROJECT, "new_consistent_affine_2-2-1_sqrtm_exports",
                          "photonic_kan_2-2-1_sqrtm_full_forward_test.mat")


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------

def activation(P, b_coef):
    """b(P) for every basis. P: (..., 1) or broadcastable to (..., M); b_coef: (M, 8) -> (..., M)."""
    b1, b2, b3, b4, b5, b6, b7, b8 = [b_coef[:, k] for k in range(8)]
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        return (b1 * np.log(1.0 + b2 * np.log1p(np.expm1(b3 * P) ** b4))
                + b5 * P + b6 * P**2 + b7 * P**3 + b8 * P**4)


def load_activation(path=ACTIVATION_MAT):
    a = loadmat(path, squeeze_me=True)
    input_power = np.asarray(a["input_power"], dtype=float)
    return {
        "b_coef_all": np.asarray(a["b_coef"], dtype=float),          # (34, 8)
        "input_power": input_power,                                   # measured P grid
        "set_detuning": np.asarray(a["set_detuning"], dtype=float),
        "valid_domain": (float(input_power.min()), float(input_power.max())),
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(path=EXPORT_221, activation_path=ACTIVATION_MAT):
    """Reads layers, test set and reference values from a full-forward export."""
    d = loadmat(path, squeeze_me=True)
    act = load_activation(activation_path)
    n_layers = int(d["num_layers"])
    sizes = np.atleast_1d(d["layer_sizes"]).astype(int)

    layers = []
    for L in range(n_layers):
        I, O = sizes[L], sizes[L + 1]
        b_coef = np.atleast_2d(d[f"layer{L}_b_coef"]).astype(float)
        m = b_coef.shape[0]
        layer = {
            "b_coef": b_coef,
            "m": m,
            "alpha": np.atleast_1d(d[f"layer{L}_alpha"]).astype(float),
            "beta": np.atleast_1d(d[f"layer{L}_beta"]).astype(float),
            "coeffs": np.asarray(d[f"layer{L}_coeffs"], dtype=float).reshape(O, I, m),
            "divide_by_m": bool(d[f"layer{L}_divide_by_m"]),
            "field_encoding": bool(d[f"layer{L}_field_encoding"]),
        }
        for key in ("input_range", "power_window", "amplitude_window"):
            if f"layer{L}_{key}" in d:
                layer[key] = np.atleast_1d(d[f"layer{L}_{key}"]).astype(float)
        layers.append(layer)

    return {
        "path": path,
        "layers": layers,
        "sizes": sizes,
        "selected_indices": np.atleast_1d(d["selected_indices_python"]).astype(int),
        "valid_domain": act["valid_domain"],
        "activation": act,
        "x_test": np.atleast_2d(d["x_test"]).astype(float),
        "y_test": np.asarray(d["y_test"], dtype=float).reshape(-1, sizes[-1]),
        "y_pytorch": np.asarray(d["y_pytorch"], dtype=float).reshape(-1, sizes[-1]),
        "test_mse_exported": float(d["test_mse"]),
        "export": d,
    }


def num_coeffs(model):
    return int(sum(l["coeffs"].size for l in model["layers"]))


def forward(model, x, delta=None, eps=None, domain=None):
    """
    Forward pass with optional perturbations and a domain check of every basis argument.

    Args:
        x:      (N, in_count) inputs
        delta:  (m,) horizontal shift of basis k, in units of P; the same delta_k is used
                wherever basis k appears (every edge, every layer). None -> 0.
        eps:    list with one array per layer, shape = coeffs shape, relative coefficient
                error a * (1 + eps). None -> 0.
        domain: (P_lo, P_hi) valid argument interval of the activation; default: model's.

    Returns:
        y (N, out_count), list of per-layer dicts (input, u, P, arg = P - delta, b, basis,
        coeffs, edge_values, output) and a dict of domain violations.
    """
    h = np.asarray(x, dtype=float)
    lo, hi = model["valid_domain"] if domain is None else domain
    out_layers = []
    issues = []

    for L, layer in enumerate(model["layers"]):
        m = layer["m"]
        d_k = np.zeros(m) if delta is None else np.asarray(delta, dtype=float)
        coeffs = layer["coeffs"] if eps is None else layer["coeffs"] * (1.0 + eps[L])

        u = layer["alpha"][None, :] * h + layer["beta"][None, :]             # (N, I)
        P = u**2 / m if layer["divide_by_m"] else u**2                        # (N, I)
        arg = P[..., None] - d_k[None, None, :]                               # (N, I, m)
        b = activation(arg, layer["b_coef"])
        with np.errstate(invalid="ignore"):
            basis = np.sqrt(b)
        edge_values = np.einsum("oik,nik->noi", coeffs, basis)                # (N, O, I)
        out = edge_values.sum(axis=-1)                                        # (N, O)

        n_neg_u = int((u < 0).sum())
        n_below = int((arg < lo).sum())
        n_above = int((arg > hi).sum())
        n_nonfinite = int((~np.isfinite(b)).sum() + (~np.isfinite(basis)).sum() + (~np.isfinite(out)).sum())
        n_neg_b = int((b < 0).sum())
        if n_neg_u:
            issues.append(f"layer {L}: u < 0 at {n_neg_u} entries")
        if n_below:
            issues.append(f"layer {L}: P - delta < {lo:g} at {n_below} entries (min {np.nanmin(arg):.4g})")
        if n_above:
            issues.append(f"layer {L}: P - delta > {hi:g} at {n_above} entries (max {np.nanmax(arg):.4g})")
        if n_neg_b:
            issues.append(f"layer {L}: b < 0 at {n_neg_b} entries")
        if n_nonfinite:
            issues.append(f"layer {L}: NaN/Inf at {n_nonfinite} entries")

        out_layers.append({"input": h, "u": u, "P": P, "arg": arg, "b": b, "basis": basis,
                           "coeffs": coeffs, "edge_values": edge_values, "output": out})
        h = out

    return h, out_layers, {"valid": not issues, "issues": issues}


def mse(model, x, y, **kw):
    y_hat, layers, check = forward(model, x, **kw)
    return float(np.mean((y_hat - y) ** 2)), y_hat, layers, check


# ---------------------------------------------------------------------------
# Consistency checks of the rebuild (step 0)
# ---------------------------------------------------------------------------

def run_checks(model, tol=1e-10, verbose=True):
    """
    Returns a list of (name, passed, detail). Checks:
      1. rebuilt output == exported PyTorch output, and test MSE == exported test MSE
      2. every intermediate stage (u, P, b, basis, edge values, output) == exported one
      3. exported b_coef == rows selected_indices of the activation file
      4. affine map consistent with the layer's input range and power window
      5. every selected basis: b(P) >= 0 on the measured P grid and b(0) = 0
      6. nominal forward pass on the test set stays inside the valid domain, no NaN/Inf
    """
    d = model["export"]
    results = []

    def add(name, ok, detail):
        results.append((name, bool(ok), detail))

    y, layers, check = forward(model, model["x_test"])

    err = float(np.max(np.abs(y - model["y_pytorch"])))
    add("output vs exported PyTorch output", err < tol, f"max |diff| = {err:.2e}")
    test_mse = float(np.mean((y - model["y_test"]) ** 2))
    rel = abs(test_mse - model["test_mse_exported"]) / model["test_mse_exported"]
    add("test MSE vs exported test MSE", rel < 1e-8, f"MSE0 = {test_mse:.6e} (exported {model['test_mse_exported']:.6e})")

    for L, lay in enumerate(layers):
        worst = 0.0
        for key, exp_key in [("input", "input"), ("u", "u"), ("P", "P"), ("b", "b"),
                             ("basis", "basis"), ("edge_values", "edge_values"), ("output", "output")]:
            ours = lay[key]
            ref = np.asarray(d[f"layer{L}_{exp_key}"], dtype=float).reshape(ours.shape)
            worst = max(worst, float(np.max(np.abs(ours - ref)) / max(np.max(np.abs(ref)), 1e-300)))
        add(f"layer {L}: all intermediate stages vs export", worst < tol, f"max relative diff = {worst:.2e}")

    # kan.py stores b_coef as float32 before the model is cast to float64, so the trained model
    # uses the float32-rounded fit coefficients; the rebuild uses exactly those (exported) values.
    b_all = model["activation"]["b_coef_all"][model["selected_indices"]]
    b_all_f32 = b_all.astype(np.float32).astype(float)
    for L, layer in enumerate(model["layers"]):
        exact = np.array_equal(layer["b_coef"], b_all)
        as_f32 = np.array_equal(layer["b_coef"], b_all_f32)
        rel = float(np.max(np.abs(layer["b_coef"] - b_all) / np.maximum(np.abs(b_all), 1e-300)))
        add(f"layer {L}: b_coef == activation file rows {model['selected_indices'].tolist()}", exact or as_f32,
            "identical" if exact else
            (f"identical after float32 rounding (max relative diff {rel:.1e})" if as_f32 else f"max relative diff {rel:.1e}"))

    for L, layer in enumerate(model["layers"]):
        if "input_range" not in layer:
            continue
        in_lo, in_hi = layer["input_range"]
        P_lo, P_hi = layer["power_window"]
        m = layer["m"]
        k = m if layer["divide_by_m"] else 1.0
        u_target = np.sqrt(k * np.array([P_lo, P_hi]))
        u_got = np.array([layer["alpha"] * in_lo + layer["beta"], layer["alpha"] * in_hi + layer["beta"]])
        diff = float(np.max(np.abs(u_got - u_target[:, None])))
        # alpha, beta of layer 0 are float32 in kan.py (set before the cast to float64): allow float32 rounding
        tol_affine = 1e-6 * float(np.max(u_target))
        add(f"layer {L}: input range [{in_lo:.4g}, {in_hi:.4g}] -> P window [{P_lo:g}, {P_hi:g}]",
            diff < tol_affine, f"max |u - sqrt(m P)| at the range ends = {diff:.1e}"
            + (" (float32 alpha, beta)" if diff > 1e-12 else ""))

    lo, hi = model["valid_domain"]
    grid = np.linspace(lo, hi, 60001)[:, None]
    b_grid = activation(grid, b_all)
    b_min = float(np.min(b_grid))
    add(f"selected bases: b(P) >= 0 on [{lo:g}, {hi:g}] (60001 points)", b_min >= 0.0, f"min b = {b_min:.3e}")
    b0 = float(np.max(np.abs(activation(np.zeros((1, 1)), b_all))))
    add("selected bases: b(0) = 0", b0 < 1e-12, f"max |b(0)| = {b0:.1e}")

    add("nominal test-set pass inside the valid domain, finite", check["valid"],
        "; ".join(check["issues"]) or "no violations")

    if verbose:
        width = max(len(r[0]) for r in results)
        for name, ok, detail in results:
            print(f"[{'PASS' if ok else 'FAIL'}] {name:<{width}}  {detail}")
    return results


if __name__ == "__main__":
    model = load_model()
    print(f"Model: {os.path.relpath(model['path'], PROJECT)}")
    print(f"Layers {model['sizes'].tolist()}, m = {model['layers'][0]['m']}, "
          f"bases (python index) {model['selected_indices'].tolist()}, {num_coeffs(model)} expansion coefficients")
    print(f"Valid activation domain (measured input power): P in {model['valid_domain']}\n")
    results = run_checks(model)
    print(f"\n{sum(r[1] for r in results)} / {len(results)} checks passed")
