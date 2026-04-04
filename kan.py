import torch
import torch.nn as nn
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


class KAN(nn.Module):
    """
    Multi-layer KAN using only TanhBasisActivationLayer.

    Example:
        model = KAN(
            in_count=784,
            out_count=10,
            hidden_layer_sizes=[128, 64],
            num_basis=8,
            x_min=0.0,
            x_max=1.0,
            gamma_scale=3.0,
            debug=False
        )
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

        layer_sizes = [in_count] + self.hidden_layer_sizes + [out_count]

        self.layers = nn.ModuleList([
            TanhBasisActivationLayer(
                in_count=layer_sizes[i],
                out_count=layer_sizes[i + 1],
                num_basis=num_basis,
                x_min=x_min,
                x_max=x_max,
                gamma_scale=gamma_scale,
                debug=debug,
            )
            for i in range(len(layer_sizes) - 1)
        ])

        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else None

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
        """
        Plot a learned edge function phi_ij(x) for one layer.

        Args:
            layer_idx: which KAN layer to inspect
            out_idx: output neuron index j within that layer
            in_idx: input feature index i within that layer
            x_range: tuple (xmin, xmax); if None, uses layer defaults
            resolution: number of points in the plot
            show_basis: if True, also plot individual basis contributions
        """
        layer = self.layers[layer_idx]

        if not isinstance(layer, TanhBasisActivationLayer):
            raise TypeError("Selected layer is not a TanhBasisActivationLayer.")

        if not (0 <= out_idx < layer.out_count):
            raise ValueError(f"out_idx must be in [0, {layer.out_count - 1}]")
        if not (0 <= in_idx < layer.in_count):
            raise ValueError(f"in_idx must be in [0, {layer.in_count - 1}]")

        if x_range is None:
            xmin, xmax = self.x_min, self.x_max
        else:
            xmin, xmax = x_range

        x = torch.linspace(xmin, xmax, resolution)  # (R,)

        centers = layer.centers[in_idx]   # (M,)
        slopes = layer.slopes[in_idx]     # (M,)
        coeffs = layer.coeffs[out_idx, in_idx]  # (M,)

        # basis: (R, M)
        basis = 0.5 * (1.0 + torch.tanh(slopes.unsqueeze(0) * (x.unsqueeze(1) - centers.unsqueeze(0))))

        # contributions: (R, M)
        contributions = basis * coeffs.unsqueeze(0)

        # phi: (R,)
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
        """
        Plot phi_ij(x) for multiple input features i feeding one output neuron j.
        """
        layer = self.layers[layer_idx]

        if not isinstance(layer, TanhBasisActivationLayer):
            raise TypeError("Selected layer is not a TanhBasisActivationLayer.")

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
            centers = layer.centers[in_idx]
            slopes = layer.slopes[in_idx]
            coeffs = layer.coeffs[out_idx, in_idx]

            basis = 0.5 * (1.0 + torch.tanh(slopes.unsqueeze(0) * (x.unsqueeze(1) - centers.unsqueeze(0))))
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
        """
        Plot all tanh-step basis functions s_im(x) for one input feature
        in a selected layer.
        """
        layer = self.layers[layer_idx]

        if not isinstance(layer, TanhBasisActivationLayer):
            raise TypeError("Selected layer is not a TanhBasisActivationLayer.")

        if not (0 <= in_idx < layer.in_count):
            raise ValueError(f"in_idx must be in [0, {layer.in_count - 1}]")

        if x_range is None:
            xmin, xmax = self.x_min, self.x_max
        else:
            xmin, xmax = x_range

        x = torch.linspace(xmin, xmax, resolution)  # (R,)
        centers = layer.centers[in_idx]             # (M,)
        slopes = layer.slopes[in_idx]               # (M,)

        # basis: (R, M)
        basis = 0.5 * (
            1.0 + torch.tanh(
                slopes.unsqueeze(0) * (x.unsqueeze(1) - centers.unsqueeze(0))
            )
        )

        x_np = x.detach().cpu().numpy()
        basis_np = basis.detach().cpu().numpy()
        centers_np = centers.detach().cpu().numpy()

        plt.figure(figsize=(8, 5))

        for m in range(layer.num_basis):
            plt.plot(x_np, basis_np[:, m], label=f"basis {m}")

        for c in centers_np:
            plt.axvline(c, linestyle="--", alpha=0.4)

        plt.xlabel(f"Input feature x[{in_idx}]")
        plt.ylabel("Basis value")
        plt.title(f"Layer {layer_idx}: basis functions for input {in_idx}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def print_coeff_stats(self, layer_idx: int):
        """
        Print summary statistics of the learned coefficient tensor a_ijm
        for a selected layer.
        """
        layer = self.layers[layer_idx]

        if not isinstance(layer, TanhBasisActivationLayer):
            raise TypeError("Selected layer is not a TanhBasisActivationLayer.")

        coeffs = layer.coeffs.detach()

        print(f"Layer {layer_idx} coefficient tensor shape: {tuple(coeffs.shape)}")
        print(f"  min  = {coeffs.min().item():.6f}")
        print(f"  max  = {coeffs.max().item():.6f}")
        print(f"  mean = {coeffs.mean().item():.6f}")
        print(f"  std  = {coeffs.std().item():.6f}")

    def plot_coefficients(
        self,
        layer_idx: int,
        out_idx: int,
        figsize=(8, 5),
        cmap="coolwarm",
    ):
        """
        Plot the coefficient matrix a_ijm for a fixed output neuron j
        in the selected layer.

        Rows: input features i
        Columns: basis index m
        """
        layer = self.layers[layer_idx]

        if not isinstance(layer, TanhBasisActivationLayer):
            raise TypeError("Selected layer is not a TanhBasisActivationLayer.")

        if not (0 <= out_idx < layer.out_count):
            raise ValueError(f"out_idx must be in [0, {layer.out_count - 1}]")

        coeffs = layer.coeffs[out_idx].detach().cpu().numpy()  # (in_count, num_basis)

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
        """
        Plot the coefficient vector a_ijm across basis index m
        for one specific edge (i -> j).
        """
        layer = self.layers[layer_idx]

        if not isinstance(layer, TanhBasisActivationLayer):
            raise TypeError("Selected layer is not a TanhBasisActivationLayer.")

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
        """
        Plot coefficient vectors a_ijm for all or several input features i
        feeding one output neuron j in the selected layer.
        """
        layer = self.layers[layer_idx]

        if not isinstance(layer, TanhBasisActivationLayer):
            raise TypeError("Selected layer is not a TanhBasisActivationLayer.")

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
        """
        Return a list of activations after each layer.
        """
        outputs = []
        h = x
        for layer in self.layers:
            h = layer(h)
            outputs.append(h.detach().cpu())
        return outputs