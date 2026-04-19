import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class TanhBasisActivationLayer(nn.Module):
    """
    KAN layer based on positive tanh-step basis functions.

    For each input feature x_i and output neuron j:

        phi_ij(x_i) = sum_m a_ijm * s_im(x_i)

    where

        s_im(x_i) = 0.5 * (1 + tanh(gamma_i * (x_i - c_im)))

    The layer output is

        y_j = sum_i phi_ij(x_i)

    Shapes:
        input:  (batch, in_count)
        output: (batch, out_count)
    """

    def __init__(
        self,
        in_count: int,
        out_count: int,
        num_basis: int = 8,
        x_min: float = 0.0,
        x_max: float = 1.0,
        gamma_scale: float = 3.0,
        debug: bool = False,
    ):
        super().__init__()

        if num_basis < 2:
            raise ValueError("num_basis must be at least 2.")
        if x_max <= x_min:
            raise ValueError("x_max must be greater than x_min.")

        self.in_count = in_count
        self.out_count = out_count
        self.num_basis = num_basis
        self.debug = debug

        centers_1d = torch.linspace(x_min, x_max, num_basis)         # (M,)
        centers = centers_1d.unsqueeze(0).repeat(in_count, 1)        # (in_count, M)

        delta = (x_max - x_min) / (num_basis - 1)
        gamma = gamma_scale / delta
        slopes = torch.full((in_count, num_basis), gamma)            # (in_count, M)

        self.register_buffer("centers", centers)
        self.register_buffer("slopes", slopes)

        # a_ijm: unrestricted trainable coefficients
        self.coeffs = nn.Parameter(
            0.01 * torch.randn(out_count, in_count, num_basis)
        )

    def forward(self, x, track_stats: bool = False):
        if x.dim() != 2:
            raise ValueError(f"Expected x shape (batch, in_count), got {tuple(x.shape)}")
        if x.size(1) != self.in_count:
            raise ValueError(
                f"Expected input with {self.in_count} features, got {x.size(1)}"
            )

        # (batch, 1, in_count, 1)
        x_expanded = x.unsqueeze(1).unsqueeze(-1)

        # (1, 1, in_count, num_basis)
        centers = self.centers.unsqueeze(0).unsqueeze(0)
        slopes = self.slopes.unsqueeze(0).unsqueeze(0)

        # Positive basis outputs in (0, 1)
        basis = 0.5 * (1.0 + torch.tanh(slopes * (x_expanded - centers)))
        # basis shape: (batch, 1, in_count, num_basis)

        # (1, out_count, in_count, num_basis)
        coeffs = self.coeffs.unsqueeze(0)

        # Sum over basis index m
        edge_values = (coeffs * basis).sum(dim=-1)
        # shape: (batch, out_count, in_count)

        # Sum over input index i
        y = edge_values.sum(dim=-1)
        # shape: (batch, out_count)

        if self.debug:
            print(f"[TanhBasisActivationLayer] x: {x.shape}")
            print(f"[TanhBasisActivationLayer] basis: {basis.shape}")
            print(f"[TanhBasisActivationLayer] coeffs: {coeffs.shape}")
            print(f"[TanhBasisActivationLayer] edge_values: {edge_values.shape}")
            print(f"[TanhBasisActivationLayer] y: {y.shape}")

        if not track_stats:
            return y

        stats = {
            "min_input": x.min().detach(),
            "max_input": x.max().detach(),
            "min_output": y.min().detach(),
            "max_output": y.max().detach(),
        }
        return y, stats


class TanhBasisActivationLayerAffine(nn.Module):
    """
    KAN layer with affine input transform inside the basis:
        u_i = alpha_i * x_i + beta_i
        s_im(x_i) = 0.5 * (1 + tanh(gamma_i * (u_i - c_im)))
    """

    def __init__(
        self,
        in_count: int,
        out_count: int,
        num_basis: int = 8,
        x_min: float = 0.0,
        x_max: float = 1.0,
        gamma_scale: float = 3.0,
        learn_affine: bool = True,
        debug: bool = False,
    ):
        super().__init__()

        if num_basis < 2:
            raise ValueError("num_basis must be at least 2.")
        if x_max <= x_min:
            raise ValueError("x_max must be greater than x_min.")

        self.in_count = in_count
        self.out_count = out_count
        self.num_basis = num_basis
        self.debug = debug
        self.learn_affine = learn_affine

        centers_1d = torch.linspace(x_min, x_max, num_basis)
        centers = centers_1d.unsqueeze(0).repeat(in_count, 1)   # (in_count, M)

        delta = (x_max - x_min) / (num_basis - 1)
        gamma = gamma_scale / delta
        slopes = torch.full((in_count, num_basis), gamma)       # (in_count, M)

        self.register_buffer("centers", centers)
        self.register_buffer("slopes", slopes)

        self.coeffs = nn.Parameter(
            0.01 * torch.randn(out_count, in_count, num_basis)
        )

        if learn_affine:
            # alpha > 0 enforced through softplus
            self.raw_alpha = nn.Parameter(torch.zeros(in_count))   # softplus(0) ~ 0.693
            self.beta = nn.Parameter(torch.zeros(in_count))
        else:
            self.register_buffer("fixed_alpha", torch.ones(in_count))
            self.register_buffer("fixed_beta", torch.zeros(in_count))

    def get_alpha(self):
        if self.learn_affine:
            return F.softplus(self.raw_alpha) + 1e-6
        return self.fixed_alpha

    def get_beta(self):
        if self.learn_affine:
            return self.beta
        return self.fixed_beta

    def forward(self, x, track_stats: bool = False):
        if x.dim() != 2:
            raise ValueError(f"Expected x shape (batch, in_count), got {tuple(x.shape)}")
        if x.size(1) != self.in_count:
            raise ValueError(
                f"Expected input with {self.in_count} features, got {x.size(1)}"
            )

        alpha = self.get_alpha()                      # (in_count,)
        beta = self.get_beta()                        # (in_count,)

        # affine-transformed inputs: u_i = alpha_i x_i + beta_i
        u = alpha.unsqueeze(0) * x + beta.unsqueeze(0)   # (batch, in_count)

        x_expanded = u.unsqueeze(1).unsqueeze(-1)        # (batch, 1, in_count, 1)
        centers = self.centers.unsqueeze(0).unsqueeze(0) # (1, 1, in_count, M)
        slopes = self.slopes.unsqueeze(0).unsqueeze(0)   # (1, 1, in_count, M)

        basis = 0.5 * (1.0 + torch.tanh(slopes * (x_expanded - centers)))
        coeffs = self.coeffs.unsqueeze(0)

        edge_values = (coeffs * basis).sum(dim=-1)   # (batch, out_count, in_count)
        y = edge_values.sum(dim=-1)                  # (batch, out_count)

        if not track_stats:
            return y

        stats = {
            "min_input": x.min().detach(),
            "max_input": x.max().detach(),
            "min_affine_input": u.min().detach(),
            "max_affine_input": u.max().detach(),
            "min_output": y.min().detach(),
            "max_output": y.max().detach(),
        }
        return y, stats


class KAN(nn.Module):
    """
    Multi-layer KAN using either:
      - TanhBasisActivationLayer
      - TanhBasisActivationLayerAffine
    """

    def __init__(
        self,
        in_count: int,
        out_count: int,
        hidden_layer_sizes=None,
        num_basis: int = 8,
        x_min: float = 0.0,
        x_max: float = 1.0,
        gamma_scale: float = 3.0,
        dropout_prob: float = 0.0,
        debug: bool = False,
        layer_type: str = "standard",   # "standard" or "affine"
        learn_affine: bool = True,      # only used if layer_type == "affine"
    ):
        super().__init__()

        if hidden_layer_sizes is None:
            hidden_layer_sizes = []

        self.in_count = in_count
        self.out_count = out_count
        self.hidden_layer_sizes = list(hidden_layer_sizes)
        self.num_basis = num_basis
        self.x_min = x_min
        self.x_max = x_max
        self.gamma_scale = gamma_scale
        self.debug = debug
        self.layer_type = layer_type
        self.learn_affine = learn_affine

        layer_sizes = [in_count] + self.hidden_layer_sizes + [out_count]

        if layer_type == "standard":
            layer_cls = TanhBasisActivationLayer
        elif layer_type == "affine":
            layer_cls = TanhBasisActivationLayerAffine
        else:
            raise ValueError("layer_type must be either 'standard' or 'affine'")

        layers = []
        for i in range(len(layer_sizes) - 1):
            if layer_type == "standard":
                layer = layer_cls(
                    in_count=layer_sizes[i],
                    out_count=layer_sizes[i + 1],
                    num_basis=num_basis,
                    x_min=x_min,
                    x_max=x_max,
                    gamma_scale=gamma_scale,
                    debug=debug,
                )
            else:
                layer = layer_cls(
                    in_count=layer_sizes[i],
                    out_count=layer_sizes[i + 1],
                    num_basis=num_basis,
                    x_min=x_min,
                    x_max=x_max,
                    gamma_scale=gamma_scale,
                    learn_affine=learn_affine,
                    debug=debug,
                )
            layers.append(layer)

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else None

    def _validate_layer(self, layer_idx: int):
        layer = self.layers[layer_idx]

        if not isinstance(layer, (TanhBasisActivationLayer, TanhBasisActivationLayerAffine)):
            raise TypeError("Selected layer is not a supported KAN layer.")
        return layer

    def _compute_basis_on_grid(self, layer, in_idx: int, x: torch.Tensor):
        """
        Returns basis values of shape (R, M) for a 1D input grid x of shape (R,).
        Supports both standard and affine layer types.
        """
        centers = layer.centers[in_idx]   # (M,)
        slopes = layer.slopes[in_idx]     # (M,)

        if isinstance(layer, TanhBasisActivationLayerAffine):
            alpha = layer.get_alpha()[in_idx]
            beta = layer.get_beta()[in_idx]
            x_eff = alpha * x + beta
        else:
            x_eff = x

        basis = 0.5 * (
            1.0 + torch.tanh(
                slopes.unsqueeze(0) * (x_eff.unsqueeze(1) - centers.unsqueeze(0))
            )
        )
        return basis
    
    def forward(self, x, track_stats: bool = False, return_all: bool = False):
        stats = []
        activations = [x] if return_all else None

        for layer_idx, layer in enumerate(self.layers):
            if track_stats:
                x, layer_stats = layer(x, track_stats=True)
                layer_stats["layer"] = layer_idx
                stats.append(layer_stats)
            else:
                x = layer(x)

            if self.dropout is not None and layer_idx < len(self.layers) - 1:
                x = self.dropout(x)

            if return_all:
                activations.append(x)

        if track_stats and return_all:
            return x, stats, activations
        if track_stats:
            return x, stats
        if return_all:
            return x, activations
        return x
    
    def plot_edge_function(
        self,
        layer_idx: int,
        out_idx: int,
        in_idx: int,
        x_range=None,
        resolution: int = 400,
        show_basis: bool = False,
    ):
        layer = self._validate_layer(layer_idx)

        if not (0 <= out_idx < layer.out_count):
            raise ValueError(f"out_idx must be in [0, {layer.out_count - 1}]")
        if not (0 <= in_idx < layer.in_count):
            raise ValueError(f"in_idx must be in [0, {layer.in_count - 1}]")

        if x_range is None:
            xmin, xmax = self.x_min, self.x_max
        else:
            xmin, xmax = x_range

        x = torch.linspace(xmin, xmax, resolution)

        coeffs = layer.coeffs[out_idx, in_idx]   # (M,)
        basis = self._compute_basis_on_grid(layer, in_idx, x)   # (R, M)
        contributions = basis * coeffs.unsqueeze(0)
        phi = contributions.sum(dim=1)

        x_np = x.detach().cpu().numpy()
        phi_np = phi.detach().cpu().numpy()

        plt.figure(figsize=(8, 5))
        plt.plot(x_np, phi_np, label=f"$\\phi_{{{in_idx},{out_idx}}}(x)$")

        if show_basis:
            for m in range(layer.num_basis):
                contrib_np = contributions[:, m].detach().cpu().numpy()
                plt.plot(x_np, contrib_np, linestyle="--", alpha=0.7, label=f"basis {m}")

        plt.xlabel(f"Input feature x[{in_idx}]")
        plt.ylabel("Edge response")
        plt.title(f"Layer {layer_idx}: edge function from input {in_idx} to output {out_idx}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_all_incoming_edges(
        self,
        layer_idx: int,
        out_idx: int,
        x_range=None,
        resolution: int = 400,
        max_inputs: int = None,
    ):
        layer = self._validate_layer(layer_idx)

        if not (0 <= out_idx < layer.out_count):
            raise ValueError(f"out_idx must be in [0, {layer.out_count - 1}]")

        if x_range is None:
            xmin, xmax = self.x_min, self.x_max
        else:
            xmin, xmax = x_range

        input_count = layer.in_count if max_inputs is None else min(layer.in_count, max_inputs)

        x = torch.linspace(xmin, xmax, resolution)
        x_np = x.detach().cpu().numpy()

        plt.figure(figsize=(9, 6))

        for in_idx in range(input_count):
            coeffs = layer.coeffs[out_idx, in_idx]
            basis = self._compute_basis_on_grid(layer, in_idx, x)
            phi = (basis * coeffs.unsqueeze(0)).sum(dim=1)

            plt.plot(x_np, phi.detach().cpu().numpy(), label=f"in {in_idx}")

        plt.xlabel("Input value")
        plt.ylabel("Edge response")
        plt.title(f"Layer {layer_idx}: all incoming edge functions to output neuron {out_idx}")
        plt.grid(True)
        plt.tight_layout()

        if input_count <= 12:
            plt.legend()

        plt.show()

    def plot_basis_functions(
        self,
        layer_idx: int,
        in_idx: int,
        x_range=None,
        resolution: int = 400,
    ):
        layer = self._validate_layer(layer_idx)

        if not (0 <= in_idx < layer.in_count):
            raise ValueError(f"in_idx must be in [0, {layer.in_count - 1}]")

        if x_range is None:
            xmin, xmax = self.x_min, self.x_max
        else:
            xmin, xmax = x_range

        x = torch.linspace(xmin, xmax, resolution)
        centers = layer.centers[in_idx]
        basis = self._compute_basis_on_grid(layer, in_idx, x)

        x_np = x.detach().cpu().numpy()
        basis_np = basis.detach().cpu().numpy()
        centers_np = centers.detach().cpu().numpy()

        plt.figure(figsize=(8, 5))

        for m in range(layer.num_basis):
            plt.plot(x_np, basis_np[:, m], label=f"basis {m}")

        if isinstance(layer, TanhBasisActivationLayerAffine):
            alpha = layer.get_alpha()[in_idx].detach().cpu().item()
            beta = layer.get_beta()[in_idx].detach().cpu().item()

            # positions in original x-space where alpha*x + beta = center
            transformed_centers = (centers.detach().cpu().numpy() - beta) / alpha
            for c in transformed_centers:
                plt.axvline(c, linestyle="--", alpha=0.4)
            title_suffix = f" (affine: alpha={alpha:.3f}, beta={beta:.3f})"
        else:
            for c in centers_np:
                plt.axvline(c, linestyle="--", alpha=0.4)
            title_suffix = ""

        plt.xlabel(f"Input feature x[{in_idx}]")
        plt.ylabel("Basis value")
        plt.title(f"Layer {layer_idx}: basis functions for input {in_idx}{title_suffix}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def print_coeff_stats(self, layer_idx: int):
        layer = self._validate_layer(layer_idx)

        coeffs = layer.coeffs.detach()

        print(f"Layer {layer_idx} coefficient tensor shape: {tuple(coeffs.shape)}")
        print(f"  min  = {coeffs.min().item():.6f}")
        print(f"  max  = {coeffs.max().item():.6f}")
        print(f"  mean = {coeffs.mean().item():.6f}")
        print(f"  std  = {coeffs.std().item():.6f}")

        if isinstance(layer, TanhBasisActivationLayerAffine):
            alpha = layer.get_alpha().detach()
            beta = layer.get_beta().detach()
            print(f"  alpha min/max = {alpha.min().item():.6f} / {alpha.max().item():.6f}")
            print(f"  beta  min/max = {beta.min().item():.6f} / {beta.max().item():.6f}")

    def plot_coefficients(
        self,
        layer_idx: int,
        out_idx: int,
        figsize=(8, 5),
        cmap="coolwarm",
    ):
        layer = self._validate_layer(layer_idx)

        if not (0 <= out_idx < layer.out_count):
            raise ValueError(f"out_idx must be in [0, {layer.out_count - 1}]")

        coeffs = layer.coeffs[out_idx].detach().cpu().numpy()

        plt.figure(figsize=figsize)
        plt.imshow(coeffs, aspect="auto", cmap=cmap)
        plt.colorbar(label="Coefficient value")
        plt.xlabel("Basis index m")
        plt.ylabel("Input feature i")
        plt.title(f"Layer {layer_idx}: coefficients for output neuron {out_idx}")
        plt.tight_layout()
        plt.show()

    def plot_coefficient_vector(
        self,
        layer_idx: int,
        out_idx: int,
        in_idx: int,
    ):
        layer = self._validate_layer(layer_idx)

        if not (0 <= out_idx < layer.out_count):
            raise ValueError(f"out_idx must be in [0, {layer.out_count - 1}]")
        if not (0 <= in_idx < layer.in_count):
            raise ValueError(f"in_idx must be in [0, {layer.in_count - 1}]")

        coeffs = layer.coeffs[out_idx, in_idx].detach().cpu().numpy()

        plt.figure(figsize=(7, 4))
        plt.plot(range(layer.num_basis), coeffs, marker="o")
        plt.xlabel("Basis index m")
        plt.ylabel("Coefficient value")
        plt.title(
            f"Layer {layer_idx}: coefficient vector for edge input {in_idx} -> output {out_idx}"
        )
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_all_coefficient_vectors(
        self,
        layer_idx: int,
        out_idx: int,
        max_inputs: int = None,
    ):
        layer = self._validate_layer(layer_idx)

        if not (0 <= out_idx < layer.out_count):
            raise ValueError(f"out_idx must be in [0, {layer.out_count - 1}]")

        input_count = layer.in_count if max_inputs is None else min(layer.in_count, max_inputs)

        plt.figure(figsize=(8, 5))

        for in_idx in range(input_count):
            coeffs = layer.coeffs[out_idx, in_idx].detach().cpu().numpy()
            plt.plot(range(layer.num_basis), coeffs, marker="o", label=f"in {in_idx}")

        plt.xlabel("Basis index m")
        plt.ylabel("Coefficient value")
        plt.title(f"Layer {layer_idx}: coefficient vectors for output neuron {out_idx}")
        plt.grid(True)
        if input_count <= 12:
            plt.legend()
        plt.tight_layout()
        plt.show()

    def get_layer_outputs(self, x):
        outputs = []
        h = x
        for layer in self.layers:
            h = layer(h)
            outputs.append(h.detach().cpu())
        return outputs

    