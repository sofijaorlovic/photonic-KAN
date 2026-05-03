import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import scipy.io


def load_photonic_activation_data(mat_path: str):
    """
    Loads photonic activation data from the .mat file.

    Expected variables:
        b_coef:       (num_activations, 8)
        input_power:  (num_points,)
        peakout_data: (num_points, num_activations)
        set_detuning: (num_activations,)
    """
    mat = scipy.io.loadmat(mat_path, squeeze_me=True)

    required_keys = ["b_coef", "input_power", "peakout_data", "set_detuning"]
    for key in required_keys:
        if key not in mat:
            raise KeyError(f"Missing key '{key}' in .mat file.")

    b_coef = torch.tensor(mat["b_coef"], dtype=torch.float32)
    input_power = torch.tensor(mat["input_power"], dtype=torch.float32)
    peakout_data = torch.tensor(mat["peakout_data"], dtype=torch.float32)
    set_detuning = torch.tensor(mat["set_detuning"], dtype=torch.float32)

    return b_coef, input_power, peakout_data, set_detuning


def select_equidistant_basis_indices(total_basis: int, num_selected: int = 12):
    """
    Selects approximately equidistant basis indices from the available activations.
    """
    if num_selected > total_basis:
        raise ValueError(
            f"num_selected={num_selected} cannot be larger than total_basis={total_basis}."
        )

    indices = np.linspace(0, total_basis - 1, num_selected)
    indices = np.round(indices).astype(int)
    indices = np.unique(indices)

    return indices


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
    KAN layer with fixed affine input transform inside the basis:

        x in [-input_abs_max, input_abs_max]
        u in [x_min, x_max]

        u_i = alpha_i * x_i + beta_i
        s_im(x_i) = 0.5 * (1 + tanh(gamma_i * (u_i - c_im)))

    No clamp.
    Fixed alpha and beta.
    Equivalent affine idea to the photonic affine layer.
    """

    def __init__(
        self,
        in_count: int,
        out_count: int,
        num_basis: int = 8,
        input_abs_max: float = 1.0,
        x_min: float = 0.0,
        x_max: float = 1.0,
        gamma_scale: float = 3.0,
        debug: bool = False,
    ):
        super().__init__()

        if num_basis < 2:
            raise ValueError("num_basis must be at least 2.")

        if input_abs_max <= 0:
            raise ValueError("input_abs_max must be positive.")

        if x_max <= x_min:
            raise ValueError("x_max must be greater than x_min.")

        self.in_count = in_count
        self.out_count = out_count
        self.num_basis = num_basis
        self.input_abs_max = float(input_abs_max)
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.debug = debug

        centers_1d = torch.linspace(x_min, x_max, num_basis)
        centers = centers_1d.unsqueeze(0).repeat(in_count, 1)

        delta = (x_max - x_min) / (num_basis - 1)
        gamma = gamma_scale / delta
        slopes = torch.full((in_count, num_basis), gamma)

        self.register_buffer("centers", centers)
        self.register_buffer("slopes", slopes)

        alpha_value = (self.x_max - self.x_min) / (2.0 * self.input_abs_max)
        beta_value = 0.5 * (self.x_min + self.x_max)

        self.register_buffer(
            "alpha",
            torch.full((in_count,), alpha_value, dtype=torch.float32),
        )

        self.register_buffer(
            "beta",
            torch.full((in_count,), beta_value, dtype=torch.float32),
        )

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

        u = self.alpha.unsqueeze(0) * x + self.beta.unsqueeze(0)

        x_expanded = u.unsqueeze(1).unsqueeze(-1)
        centers = self.centers.unsqueeze(0).unsqueeze(0)
        slopes = self.slopes.unsqueeze(0).unsqueeze(0)

        basis = 0.5 * (1.0 + torch.tanh(slopes * (x_expanded - centers)))

        coeffs = self.coeffs.unsqueeze(0)
        edge_values = (coeffs * basis).sum(dim=-1)
        y = edge_values.sum(dim=-1)

        if not track_stats:
            return y

        stats = {
            "min_input": x.min().detach(),
            "max_input": x.max().detach(),
            "min_affine_input": u.min().detach(),
            "max_affine_input": u.max().detach(),
            "min_basis": basis.min().detach(),
            "max_basis": basis.max().detach(),
            "min_output": y.min().detach(),
            "max_output": y.max().detach(),
        }

        return y, stats
    

class PhotonicBasisActivationLayer(nn.Module):
    """
    KAN layer using fixed photonic activation functions as basis functions.

    For each input feature x_i and output neuron j:

        phi_ij(x_i) = sum_m a_ijm * f_m(x_i)

    where f_m is one fixed photonic activation defined by b_coef[m, :].

    The layer output is:

        y_j = sum_i phi_ij(x_i)

    Trainable parameters:
        coeffs[j, i, m]

    Fixed parameters:
        b_coef[m, :]

    Important:
        This version does NOT clamp inputs or intermediate values.
        Therefore, the inputs must already be physically valid.
    """

    def __init__(
        self,
        in_count: int,
        out_count: int,
        b_coef_selected: torch.Tensor,
        x_min: float = 0.0,
        x_max: float = 60.0,
        normalize_basis: bool = False,
        check_input_range: bool = False,
        debug: bool = False,
    ):
        super().__init__()

        if b_coef_selected is None:
            raise ValueError("b_coef_selected must be provided for PhotonicBasisActivationLayer.")

        if not torch.is_tensor(b_coef_selected):
            b_coef_selected = torch.tensor(b_coef_selected, dtype=torch.float32)

        if b_coef_selected.dim() != 2:
            raise ValueError(
                f"b_coef_selected must have shape (num_basis, 8), got {tuple(b_coef_selected.shape)}"
            )

        if b_coef_selected.size(1) != 8:
            raise ValueError(
                f"Each photonic basis must have 8 coefficients, got {b_coef_selected.size(1)}."
            )

        if x_max <= x_min:
            raise ValueError("x_max must be greater than x_min.")

        self.in_count = in_count
        self.out_count = out_count
        self.num_basis = b_coef_selected.size(0)

        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.normalize_basis = normalize_basis
        self.check_input_range = check_input_range
        self.debug = debug

        # Fixed photonic activation coefficients.
        # These are NOT trained.
        self.register_buffer("b_coef", b_coef_selected.clone().float())

        # Trainable KAN coefficients.
        self.coeffs = nn.Parameter(
            0.01 * torch.randn(out_count, in_count, self.num_basis)
        )

    def _photonic_basis(self, x: torch.Tensor):
        """
        Computes all selected photonic basis functions.

        Args:
            x: shape (batch, 1, in_count, 1)

        Returns:
            basis: shape (batch, 1, in_count, num_basis)
        """

        if self.check_input_range:
            if torch.any(x < self.x_min) or torch.any(x > self.x_max):
                raise ValueError(
                    f"Input outside expected physical range "
                    f"[{self.x_min}, {self.x_max}]. "
                    f"Got min={x.min().item()}, max={x.max().item()}."
                )

        b = self.b_coef

        b1 = b[:, 0].view(1, 1, 1, -1)
        b2 = b[:, 1].view(1, 1, 1, -1)
        b3 = b[:, 2].view(1, 1, 1, -1)
        b4 = b[:, 3].view(1, 1, 1, -1)
        b5 = b[:, 4].view(1, 1, 1, -1)
        b6 = b[:, 5].view(1, 1, 1, -1)
        b7 = b[:, 6].view(1, 1, 1, -1)
        b8 = b[:, 7].view(1, 1, 1, -1)

        # Original photonic activation:
        #
        # b1 * log(1 + b2 * log(1 + (exp(b3*x) - 1)^b4))
        # + b5*x + b6*x^2 + b7*x^3 + b8*x^4

        inner_base = torch.expm1(b3 * x)
        powered = torch.pow(inner_base, b4)
        inner_log = torch.log1p(powered)
        outer_argument = 1.0 + b2 * inner_log

        y = (
            b1 * torch.log(outer_argument)
            + b5 * x
            + b6 * x**2
            + b7 * x**3
            + b8 * x**4
        )

        if self.normalize_basis:
            scale = self.x_max - self.x_min
            y = y / scale

        return y

    def forward(self, x, track_stats: bool = False):
        if x.dim() != 2:
            raise ValueError(f"Expected x shape (batch, in_count), got {tuple(x.shape)}")

        if x.size(1) != self.in_count:
            raise ValueError(
                f"Expected input with {self.in_count} features, got {x.size(1)}"
            )

        x_expanded = x.unsqueeze(1).unsqueeze(-1)
        # shape: (batch, 1, in_count, 1)

        basis = self._photonic_basis(x_expanded)
        # shape: (batch, 1, in_count, num_basis)

        coeffs = self.coeffs.unsqueeze(0)
        # shape: (1, out_count, in_count, num_basis)

        edge_values = (coeffs * basis).sum(dim=-1)
        # shape: (batch, out_count, in_count)

        y = edge_values.sum(dim=-1)
        # shape: (batch, out_count)

        if self.debug:
            print(f"[PhotonicBasisActivationLayer] x: {x.shape}")
            print(f"[PhotonicBasisActivationLayer] basis: {basis.shape}")
            print(f"[PhotonicBasisActivationLayer] coeffs: {coeffs.shape}")
            print(f"[PhotonicBasisActivationLayer] edge_values: {edge_values.shape}")
            print(f"[PhotonicBasisActivationLayer] y: {y.shape}")

        if not track_stats:
            return y

        stats = {
            "min_input": x.min().detach(),
            "max_input": x.max().detach(),
            "min_basis": basis.min().detach(),
            "max_basis": basis.max().detach(),
            "min_output": y.min().detach(),
            "max_output": y.max().detach(),
        }

        return y, stats


class PhotonicBasisActivationLayerIntervalAffineClean(nn.Module):
    """
    Photonic KAN layer with fixed affine input mapping:

        x in [-input_abs_max, input_abs_max]
        u in [basis_min, basis_max]

        u = alpha * x + beta

    No clamp.
    No basis normalization.
    """

    def __init__(
        self,
        in_count: int,
        out_count: int,
        b_coef_selected: torch.Tensor,
        input_abs_max: float,
        basis_min: float = 0.05,
        basis_max: float = 60.0,
        debug: bool = False,
    ):
        super().__init__()

        if b_coef_selected is None:
            raise ValueError("b_coef_selected must be provided.")

        if not torch.is_tensor(b_coef_selected):
            b_coef_selected = torch.tensor(b_coef_selected, dtype=torch.float32)

        if b_coef_selected.dim() != 2 or b_coef_selected.size(1) != 8:
            raise ValueError(
                f"b_coef_selected must have shape (num_basis, 8), got {tuple(b_coef_selected.shape)}"
            )

        if input_abs_max <= 0:
            raise ValueError("input_abs_max must be positive.")

        if basis_max <= basis_min:
            raise ValueError("basis_max must be greater than basis_min.")

        self.in_count = in_count
        self.out_count = out_count
        self.num_basis = b_coef_selected.size(0)

        self.input_abs_max = float(input_abs_max)
        self.basis_min = float(basis_min)
        self.basis_max = float(basis_max)
        self.debug = debug

        self.register_buffer("b_coef", b_coef_selected.clone().float())

        alpha_value = (self.basis_max - self.basis_min) / (2.0 * self.input_abs_max)
        beta_value = 0.5 * (self.basis_min + self.basis_max)

        self.register_buffer(
            "alpha",
            torch.full((in_count,), alpha_value, dtype=torch.float32),
        )
        self.register_buffer(
            "beta",
            torch.full((in_count,), beta_value, dtype=torch.float32),
        )

        self.coeffs = nn.Parameter(
            0.01 * torch.randn(out_count, in_count, self.num_basis)
        )

    def _photonic_basis_from_u(self, u: torch.Tensor):
        b = self.b_coef

        b1 = b[:, 0].view(1, 1, 1, -1)
        b2 = b[:, 1].view(1, 1, 1, -1)
        b3 = b[:, 2].view(1, 1, 1, -1)
        b4 = b[:, 3].view(1, 1, 1, -1)
        b5 = b[:, 4].view(1, 1, 1, -1)
        b6 = b[:, 5].view(1, 1, 1, -1)
        b7 = b[:, 6].view(1, 1, 1, -1)
        b8 = b[:, 7].view(1, 1, 1, -1)

        inner_base = torch.expm1(b3 * u)
        powered = torch.pow(inner_base, b4)
        inner_log = torch.log1p(powered)
        outer_argument = 1.0 + b2 * inner_log

        y = (
            b1 * torch.log(outer_argument)
            + b5 * u
            + b6 * u**2
            + b7 * u**3
            + b8 * u**4
        )

        return y

    def forward(self, x, track_stats: bool = False):
        if x.dim() != 2:
            raise ValueError(f"Expected x shape (batch, in_count), got {tuple(x.shape)}")

        if x.size(1) != self.in_count:
            raise ValueError(
                f"Expected input with {self.in_count} features, got {x.size(1)}"
            )

        u = self.alpha.unsqueeze(0) * x + self.beta.unsqueeze(0)

        u_expanded = u.unsqueeze(1).unsqueeze(-1)
        basis = self._photonic_basis_from_u(u_expanded)

        coeffs = self.coeffs.unsqueeze(0)
        edge_values = (coeffs * basis).sum(dim=-1)

        y = edge_values.sum(dim=-1)

        if not track_stats:
            return y

        stats = {
            "min_input": x.min().detach(),
            "max_input": x.max().detach(),
            "min_affine_input": u.min().detach(),
            "max_affine_input": u.max().detach(),
            "min_basis": basis.min().detach(),
            "max_basis": basis.max().detach(),
            "min_output": y.min().detach(),
            "max_output": y.max().detach(),
        }

        return y, stats

class KAN(nn.Module):
    """
    Multi-layer KAN using either:
      - TanhBasisActivationLayer
      - TanhBasisActivationLayerAffine
      - PhotonicBasisActivationLayer
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
        layer_type: str = "standard",   # "standard", "affine", or "photonic"
        b_coef_selected: torch.Tensor = None,  # only used if layer_type == "photonic"
        #clamp_input: bool = True,       # only used if layer_type == "photonic"
        normalize_basis: bool = True,   # only used if layer_type == "photonic"
        check_input_range: bool = False, # only used if layer_type == "photonic"
        basis_min: float = 0.05,
        basis_max: float = 60.0,
        input_abs_max: float = 10.0,    # only used for photonic layers with interval affine
        input_min_by_layer=None,
        input_max_by_layer=None,
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
        self.basis_min = basis_min
        self.basis_max = basis_max
        self.input_abs_max = input_abs_max
        self.input_min_by_layer = input_min_by_layer
        self.input_max_by_layer = input_max_by_layer

        layer_sizes = [in_count] + self.hidden_layer_sizes + [out_count]

        if layer_type == "standard":
            layer_cls = TanhBasisActivationLayer
        elif layer_type == "affine":
            layer_cls = TanhBasisActivationLayerAffine
        elif layer_type == "photonic":
            layer_cls = PhotonicBasisActivationLayer
        elif layer_type == "photonic_interval_affine":
            layer_cls = PhotonicBasisActivationLayerIntervalAffine
        elif layer_type == "photonic_interval_affine_clean":
            layer_cls = PhotonicBasisActivationLayerIntervalAffineClean

        else:
            raise ValueError("layer_type must be 'standard', 'affine', or 'photonic'.")

        if layer_type in ["photonic", "photonic_interval_affine_clean"]:
            if b_coef_selected is None:
                raise ValueError(f"b_coef_selected must be provided when layer_type='{layer_type}'.")

            if not torch.is_tensor(b_coef_selected):
                b_coef_selected = torch.tensor(b_coef_selected, dtype=torch.float32)

            if b_coef_selected.dim() != 2 or b_coef_selected.size(1) != 8:
                raise ValueError(
                    f"b_coef_selected must have shape (num_basis, 8), got {tuple(b_coef_selected.shape)}"
                )

            num_basis = b_coef_selected.size(0)
            
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

            elif layer_type == "affine":
                layer = layer_cls(
                    in_count=layer_sizes[i],
                    out_count=layer_sizes[i + 1],
                    num_basis=num_basis,
                    input_abs_max=input_abs_max,
                    x_min=x_min,
                    x_max=x_max,
                    gamma_scale=gamma_scale,
                    debug=debug,
                )

            elif layer_type == "photonic":
                layer = layer_cls(
                    in_count=layer_sizes[i],
                    out_count=layer_sizes[i + 1],
                    b_coef_selected=b_coef_selected,
                    x_min=x_min,
                    x_max=x_max,
                    normalize_basis=normalize_basis,
                    check_input_range=check_input_range,
                    debug=debug,
                )

            elif layer_type == "photonic_interval_affine_clean":
                layer = layer_cls(
                    in_count=layer_sizes[i],
                    out_count=layer_sizes[i + 1],
                    b_coef_selected=b_coef_selected,
                    input_abs_max=input_abs_max,
                    basis_min=basis_min,
                    basis_max=basis_max,
                    debug=debug,
                )
            
            layers.append(layer)

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else None
    
    def _validate_layer(self, layer_idx: int):
        layer = self.layers[layer_idx]

        supported_layers = (
            TanhBasisActivationLayer,
            TanhBasisActivationLayerAffine,
            PhotonicBasisActivationLayer,
            PhotonicBasisActivationLayerIntervalAffineClean,
        )

        if not isinstance(layer, supported_layers):
            raise TypeError(
                f"Selected layer is not a supported KAN layer. "
                f"Got layer type: {type(layer).__name__}"
            )

        return layer

    def _compute_basis_on_grid(self, layer, in_idx: int, x: torch.Tensor):
        """
        Returns basis values of shape (R, M) for a 1D input grid x of shape (R,).
        Supports standard, affine, and photonic layer types.
        """
        if isinstance(layer, PhotonicBasisActivationLayer):
            x_expanded = x.view(-1, 1, 1, 1)
            basis = layer._photonic_basis(x_expanded)
            basis = basis[:, 0, 0, :]
            return basis
        
        if isinstance(layer, PhotonicBasisActivationLayerIntervalAffineClean):
            alpha = layer.alpha[in_idx]
            beta = layer.beta[in_idx]
            x_eff = alpha * x + beta
            x_expanded = x_eff.view(-1, 1, 1, 1)
            basis = layer._photonic_basis_from_u(x_expanded)
            basis = basis[:, 0, 0, :]
            return basis

        centers = layer.centers[in_idx]
        slopes = layer.slopes[in_idx]

        if isinstance(layer, TanhBasisActivationLayerAffine):
            alpha = layer.alpha[in_idx]
            beta = layer.beta[in_idx]
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

        finite_basis = torch.isfinite(basis)
        finite_phi = torch.isfinite(phi)

        print("basis shape:", basis.shape)
        print("basis finite:", finite_basis.all().item())
        print("phi finite:", finite_phi.all().item())

        if not finite_phi.all():
            bad_idx = torch.where(~finite_phi)[0][0].item()
            print("First non-finite phi at:")
            print("  index:", bad_idx)
            print("  x:", x[bad_idx].item())
            print("  phi:", phi[bad_idx].item())

        if not finite_basis.all():
            bad_positions = torch.where(~finite_basis)
            first_bad_r = bad_positions[0][0].item()
            first_bad_m = bad_positions[1][0].item()

            print("First non-finite basis at:")
            print("  x index:", first_bad_r)
            print("  basis index:", first_bad_m)
            print("  x:", x[first_bad_r].item())
            print("  basis value:", basis[first_bad_r, first_bad_m].item())

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
        #plt.legend()
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
        basis = self._compute_basis_on_grid(layer, in_idx, x)

        x_np = x.detach().cpu().numpy()
        basis_np = basis.detach().cpu().numpy()

        plt.figure(figsize=(8, 5))

        for m in range(layer.num_basis):
            plt.plot(x_np, basis_np[:, m], label=f"basis {m}")

        title_suffix = ""

        if isinstance(layer, TanhBasisActivationLayerAffine):
            centers = layer.centers[in_idx]
            alpha = layer.alpha[in_idx].detach().cpu().item()
            beta = layer.beta[in_idx].detach().cpu().item()

            transformed_centers = (centers.detach().cpu().numpy() - beta) / alpha

            for c in transformed_centers:
                plt.axvline(c, linestyle="--", alpha=0.4)

            title_suffix = f" (affine: alpha={alpha:.3f}, beta={beta:.3f})"

        elif isinstance(layer, TanhBasisActivationLayer):
            centers = layer.centers[in_idx]
            centers_np = centers.detach().cpu().numpy()

            for c in centers_np:
                plt.axvline(c, linestyle="--", alpha=0.4)

        elif isinstance(layer, PhotonicBasisActivationLayer):
            title_suffix = " (photonic bases)"

        elif isinstance(layer, PhotonicBasisActivationLayerIntervalAffineClean):
            title_suffix = " (photonic interval affine bases)"

        plt.xlabel(f"Input feature x[{in_idx}]")
        plt.ylabel("Basis value")
        plt.title(f"Layer {layer_idx}: basis functions for input {in_idx}{title_suffix}")
        plt.grid(True)

        if layer.num_basis <= 12:
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

        if isinstance(layer, PhotonicBasisActivationLayer):
            print(f"  photonic b_coef shape = {tuple(layer.b_coef.shape)}")
            print(f"  trainable photonic b_coef = False")

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

    def inspect_forward_range(self, x):
        """
        Runs one forward pass and returns min/max statistics for each layer.

        This does not train the model.
        It is only a diagnostic check.
        """

        self.eval()

        with torch.no_grad():
            y, stats = self.forward(x, track_stats=True)

        print("Final output:")
        print(f"  shape: {tuple(y.shape)}")
        print(f"  min:   {y.min().item()}")
        print(f"  max:   {y.max().item()}")
        print()

        for layer_stats in stats:
            layer_idx = layer_stats["layer"]

            print(f"Layer {layer_idx}:")
            print(f"  input min:  {layer_stats['min_input'].item()}")
            print(f"  input max:  {layer_stats['max_input'].item()}")
            print(f"  basis min:  {layer_stats['min_basis'].item()}")
            print(f"  basis max:  {layer_stats['max_basis'].item()}")
            print(f"  output min: {layer_stats['min_output'].item()}")
            print(f"  output max: {layer_stats['max_output'].item()}")
            print()

        return y, stats
    
    def plot_target_approximation(
        self,
        target_fn,
        x_range,
        fixed_values=None,
        resolution: int = 400,
    ):
        """
        Plots model approximation vs target function by varying one input at a time.

        Assumes:
            input dimension = 2
            output dimension can be 1 or more

        For each plot:
            - vary x[0], keep x[1] fixed
            - vary x[1], keep x[0] fixed
        """

        if self.in_count != 2:
            raise ValueError("This plotting function currently assumes in_count=2.")

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        xmin, xmax = x_range
        x_grid = torch.linspace(xmin, xmax, resolution, device=device, dtype=dtype)

        if fixed_values is None:
            fixed_values = [
                0.5 * (xmin + xmax),
                0.5 * (xmin + xmax),
            ]

        self.eval()

        with torch.no_grad():
            for varied_idx in range(2):
                x_plot = torch.zeros(resolution, 2, device=device, dtype=dtype)

                if varied_idx == 0:
                    x_plot[:, 0] = x_grid
                    x_plot[:, 1] = fixed_values[1]
                    title_var = "x1 varied, x2 fixed"
                else:
                    x_plot[:, 0] = fixed_values[0]
                    x_plot[:, 1] = x_grid
                    title_var = "x2 varied, x1 fixed"

                y_true = target_fn(x_plot)
                y_pred = self(x_plot)

                y_true_np = y_true.detach().cpu().numpy()
                y_pred_np = y_pred.detach().cpu().numpy()
                x_np = x_grid.detach().cpu().numpy()

                out_count = y_true.shape[1]

                fig, axes = plt.subplots(1, out_count, figsize=(6 * out_count, 4))

                if out_count == 1:
                    axes = [axes]

                for j in range(out_count):
                    axes[j].plot(x_np, y_pred_np[:, j], label=f"model y{j+1}")
                    axes[j].plot(x_np, y_true_np[:, j], linestyle="--", label=f"target y{j+1}")

                    axes[j].set_xlabel(f"x[{varied_idx}]")
                    axes[j].set_ylabel(f"y{j+1}")
                    axes[j].set_title(f"{title_var} | output y{j+1}")
                    axes[j].legend()
                    axes[j].grid(True)

                plt.tight_layout()
                plt.show()

    def calibrate_input_ranges_by_layer(
        self,
        x: torch.Tensor,
        first_layer_input_min: float,
        first_layer_input_max: float,
        margin: float = 1.10,
        min_width: float = 1e-6,
        verbose: bool = True,
    ):
        """
        Calibrates interval-affine photonic layers using full observed input ranges.

        Layer 0 is fixed to the known original input range:
            [first_layer_input_min, first_layer_input_max]

        Later layers are calibrated from the observed output range of the previous layer.
        """

        self.eval()

        with torch.no_grad():
            h = x

            for layer_idx, layer in enumerate(self.layers):
                if isinstance(layer, PhotonicBasisActivationLayerIntervalAffineClean):
                    if layer_idx == 0:
                        input_min = float(first_layer_input_min)
                        input_max = float(first_layer_input_max)
                        layer_margin = 1.0
                    else:
                        input_min = h.min().item()
                        input_max = h.max().item()
                        layer_margin = margin

                    layer.set_input_range(
                        input_min=input_min,
                        input_max=input_max,
                        margin=layer_margin,
                        min_width=min_width,
                    )

                    if verbose:
                        print(
                            f"Layer {layer_idx}: calibrated input range "
                            f"[{layer.input_min:.6g}, {layer.input_max:.6g}]"
                        )

                h = layer(h)