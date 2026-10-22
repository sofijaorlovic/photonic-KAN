from __future__ import annotations

import torch

from kan import (
    KAN,
    PhotonicBasisActivationLayerIntervalAffineClean,
    load_photonic_activation_data,
)


def load_full_basis_library(mat_path: str):
    """
    Parameters
    ----------
    mat_path : str
        Path to the .mat file.

    Returns
    -------
    tuple
        (b_coef, input_power, peakout_data, set_detuning), as returned
        by kan.load_photonic_activation_data, unfiltered (b_coef has
        all 34 rows).
    """
    return load_photonic_activation_data(mat_path)


class SparseKAN(KAN):

    # ------------------------------------------------------------------
    # Training-time penalties
    # ------------------------------------------------------------------

    def l1_penalty(self) -> torch.Tensor:
        """
        Returns
        -------
        torch.Tensor
            Scalar: sum(|a_qpk|) over every edge and basis curve, in
            every layer.
        """
        penalty = None
        for layer in self.layers:
            coeffs = self._get_layer_coeffs(layer)
            term = coeffs.abs().sum()
            penalty = term if penalty is None else penalty + term
        if penalty is None:
            raise RuntimeError("SparseKAN has no layers to penalize.")
        return penalty

    def group_penalty(self) -> torch.Tensor:
        """
        Returns
        -------
        torch.Tensor
            Scalar: sum_k sqrt(sum_{q,p} a_qpk^2), grouping coefficients
            by basis index k across all edges (q, p) within each layer.
        """
        penalty = None
        for layer in self.layers:
            coeffs = self._get_layer_coeffs(layer)  # (out_count, in_count, num_basis)
            per_basis_norm = coeffs.pow(2).sum(dim=(0, 1)).sqrt()  # (num_basis,)
            term = per_basis_norm.sum()
            penalty = term if penalty is None else penalty + term
        if penalty is None:
            raise RuntimeError("SparseKAN has no layers to penalize.")
        return penalty

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def active_basis_report(self, threshold: float = 1e-3):
        """
        Parameters
        ----------
        threshold : float, optional
            Minimum |coefficient| for a basis curve to count as active.
            Default 1e-3.

        Returns
        -------
        list of dict
            One dict per (layer, out_idx, in_idx) edge, with keys
            "layer", "out_idx", "in_idx", "num_active", "num_total",
            "active_indices", "max_abs_coeff".
        """
        report = []
        for layer_idx, layer in enumerate(self.layers):
            coeffs = self._get_layer_coeffs(layer).detach()
            out_count, in_count, num_basis = coeffs.shape
            for out_idx in range(out_count):
                for in_idx in range(in_count):
                    edge_coeffs = coeffs[out_idx, in_idx]
                    active = torch.nonzero(edge_coeffs.abs() > threshold, as_tuple=True)[0]
                    report.append(
                        {
                            "layer": layer_idx,
                            "out_idx": out_idx,
                            "in_idx": in_idx,
                            "num_active": int(active.numel()),
                            "num_total": int(num_basis),
                            "active_indices": active.tolist(),
                            "max_abs_coeff": float(edge_coeffs.abs().max().item()),
                        }
                    )
        return report

    def summarize_active_basis(self, threshold: float = 1e-3):
        """
        Parameters
        ----------
        threshold : float, optional
            Minimum |coefficient| for a basis curve to count as active.
            Default 1e-3.

        Returns
        -------
        list of dict
            Same as active_basis_report(threshold).
        """
        report = self.active_basis_report(threshold=threshold)

        print(f"{'layer':>5} {'out':>4} {'in':>4} {'active/total':>13}  active basis indices")
        for r in report:
            idx_str = ",".join(str(i) for i in r["active_indices"])
            print(
                f"{r['layer']:>5} {r['out_idx']:>4} {r['in_idx']:>4} "
                f"{r['num_active']:>6}/{r['num_total']:<5} [{idx_str}]"
            )

        counts = [r["num_active"] for r in report]
        if counts:
            print(
                f"\nActive basis count per edge (threshold={threshold}): "
                f"min={min(counts)}, mean={sum(counts) / len(counts):.2f}, max={max(counts)}"
            )
        return report

    # ------------------------------------------------------------------
    # Optional hard pruning / debiasing
    # ------------------------------------------------------------------

    def freeze_mask_(self, keep_masks):
        """
        Parameters
        ----------
        keep_masks : sequence of torch.Tensor
            One per layer, in `self.layers` order, each the same shape
            as that layer's `.coeffs` (`(out_count, in_count, num_basis)`).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `len(keep_masks) != len(self.layers)`.
        AttributeError
            If a layer does not expose a plain `.coeffs` parameter.
        """
        if len(keep_masks) != len(self.layers):
            raise ValueError(
                f"keep_masks has {len(keep_masks)} entries, expected one per "
                f"layer ({len(self.layers)})."
            )
        for layer, keep_mask in zip(self.layers, keep_masks):
            if not hasattr(layer, "coeffs"):
                raise AttributeError(
                    "freeze_mask_ only supports layers with a plain 'coeffs' "
                    f"parameter; got {type(layer).__name__}."
                )
            keep_mask = keep_mask.to(device=layer.coeffs.device, dtype=layer.coeffs.dtype)
            with torch.no_grad():
                layer.coeffs.mul_(keep_mask)
            layer.register_buffer("active_mask", keep_mask, persistent=False)
            layer.coeffs.register_hook(lambda grad, m=keep_mask: grad * m)


# ----------------------------------------------------------------------
# Calibrating input_neg_max / input_pos_max for a multi-layer SparseKAN
# ----------------------------------------------------------------------


def estimate_hidden_ranges(
    build_kwargs: dict,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    loss_fn,
    lam: float = 0.0,
    probe_epochs: int = 200,
    probe_range: float = 10.0,
    margin: float = 1.5,
    lr: float = 1e-3,
    seed: int = None,
):
    """
    Parameters
    ----------
    build_kwargs : dict
        Kwargs for kan.KAN / SparseKAN. Only `input_neg_max[0]` /
        `input_pos_max[0]` are used as is; every later entry is a
        placeholder replaced by this function's recommendation.
    x_probe : torch.Tensor
        Probe input data.
    y_probe : torch.Tensor
        Probe target data.
    loss_fn : callable
        Loss function, e.g. `nn.MSELoss()`.
    lam : float, optional
        Sparsity penalty weight for the probe run. Default 0.0.
    probe_epochs : int, optional
        Number of optimizer steps for the probe run. Default 200.
    probe_range : float, optional
        Placeholder `input_neg_max`/`input_pos_max` for every
        non-first layer during the probe only. Default 10.0.
    margin : float, optional
        Safety factor applied to the largest |output| observed.
        Default 1.5.
    lr : float, optional
        Probe optimizer learning rate. Default 1e-3.
    seed : int, optional
        Random seed for the probe model. Default None.

    Returns
    -------
    tuple of list
        `(input_neg_max_by_layer, input_pos_max_by_layer)`.
    """
    probe_kwargs = dict(build_kwargs)
    hidden_layer_sizes = build_kwargs.get("hidden_layer_sizes") or []
    num_layers = len(hidden_layer_sizes) + 1

    neg0 = build_kwargs["input_neg_max"]
    pos0 = build_kwargs["input_pos_max"]
    if isinstance(neg0, (list, tuple)):
        neg0 = neg0[0]
    if isinstance(pos0, (list, tuple)):
        pos0 = pos0[0]

    probe_kwargs["input_neg_max"] = [neg0] + [probe_range] * (num_layers - 1)
    probe_kwargs["input_pos_max"] = [pos0] + [probe_range] * (num_layers - 1)

    if seed is not None:
        torch.manual_seed(seed)

    probe_model = KAN(**probe_kwargs).to(x_probe.dtype)
    optimizer = torch.optim.Adam(probe_model.parameters(), lr=lr)
    max_abs_per_layer = [0.0] * num_layers

    for _ in range(probe_epochs):
        probe_model.train()
        optimizer.zero_grad()

        x = x_probe
        layer_outputs = []
        for layer in probe_model.layers:
            x = layer(x)
            layer_outputs.append(x)

        mse = loss_fn(x, y_probe)
        if lam:
            l1 = sum(probe_model._get_layer_coeffs(l).abs().sum() for l in probe_model.layers)
            loss = mse + lam * l1
        else:
            loss = mse

        loss.backward()
        optimizer.step()

        if torch.isnan(loss):
            # The placeholder range itself was too tight for this
            # architecture/data -- stop early and use whatever signal
            # was gathered so far; consider raising probe_range.
            break

        with torch.no_grad():
            for i, out in enumerate(layer_outputs):
                max_abs_per_layer[i] = max(max_abs_per_layer[i], out.abs().max().item())

    input_neg_max_by_layer = [neg0] + [margin * max_abs_per_layer[i] for i in range(num_layers - 1)]
    input_pos_max_by_layer = [pos0] + [margin * max_abs_per_layer[i] for i in range(num_layers - 1)]

    return input_neg_max_by_layer, input_pos_max_by_layer
