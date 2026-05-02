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

class PhotonicBasisActivationLayerIntervalAffine(nn.Module):
    """
    KAN layer using fixed photonic activation functions as basis functions,
    with a fixed interval-based affine transform before the photonic basis.

    We assume each input feature x_i lies approximately in:

        x_i in [-h, h]

    and map it to a positive photonic interval:

        u_i in [basis_min, basis_max]

    using:

        u_i = alpha * x_i + beta

    where:

        alpha = (basis_max - basis_min) / (2*h)
        beta  = (basis_min + basis_max) / 2

    Trainable:
        coeffs[j, i, m]

    Fixed:
        b_coef[m, :]
        alpha[i]
        beta[i]
    """

    def __init__(
        self,
        in_count: int,
        out_count: int,
        b_coef_selected: torch.Tensor,
        input_abs_max: float,
        basis_min: float = 0.05,
        basis_max: float = 60.0,
        normalize_basis: bool = True,
        debug: bool = False,
    ):
        super().__init__()

        if b_coef_selected is None:
            raise ValueError("b_coef_selected must be provided.")

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

        if basis_max <= basis_min:
            raise ValueError("basis_max must be greater than basis_min.")

        if input_abs_max <= 0:
            raise ValueError("input_abs_max must be positive.")

        self.in_count = in_count
        self.out_count = out_count
        self.num_basis = b_coef_selected.size(0)

        self.input_abs_max = float(input_abs_max)
        self.basis_min = float(basis_min)
        self.basis_max = float(basis_max)
        self.normalize_basis = normalize_basis
        self.debug = debug

        # Fixed photonic activation coefficients.
        # Not trainable.
        self.register_buffer("b_coef", b_coef_selected.clone().float())

        # Fixed interval affine transform.
        alpha_value = (self.basis_max - self.basis_min) / (2.0 * self.input_abs_max)
        beta_value = (self.basis_min + self.basis_max) / 2.0

        alpha = torch.full((in_count,), alpha_value, dtype=torch.float32)
        beta = torch.full((in_count,), beta_value, dtype=torch.float32)

        self.register_buffer("alpha", alpha)
        self.register_buffer("beta", beta)

        # Trainable KAN combination coefficients.
        self.coeffs = nn.Parameter(
            0.01 * torch.randn(out_count, in_count, self.num_basis)
        )

    def _photonic_basis_from_u(self, u: torch.Tensor):
        """
        Computes selected photonic basis functions.

        Args:
            u: shape (batch, 1, in_count, 1)

        Returns:
            basis: shape (batch, 1, in_count, num_basis)
        """
        b = self.b_coef

        b1 = b[:, 0].view(1, 1, 1, -1)
        b2 = b[:, 1].view(1, 1, 1, -1)
        b3 = b[:, 2].view(1, 1, 1, -1)
        b4 = b[:, 3].view(1, 1, 1, -1)
        b5 = b[:, 4].view(1, 1, 1, -1)
        b6 = b[:, 5].view(1, 1, 1, -1)
        b7 = b[:, 6].view(1, 1, 1, -1)
        b8 = b[:, 7].view(1, 1, 1, -1)

        # Numerical safety for exp.
        exp_arg = torch.clamp(b3 * u, min=-50.0, max=50.0)

        inner_base = torch.exp(exp_arg) - 1.0

        # u should be positive by construction, but this protects against
        # tiny numerical issues.
        inner_base = torch.clamp(inner_base, min=1e-12)

        powered = torch.pow(inner_base, b4)

        inner_log = torch.log1p(powered)

        outer_argument = 1.0 + b2 * inner_log

        # Protect log argument.
        outer_argument = torch.clamp(outer_argument, min=1e-12)

        y = (
            b1 * torch.log(outer_argument)
            + b5 * u
            + b6 * u**2
            + b7 * u**3
            + b8 * u**4
        )

        if self.normalize_basis:
            y = y / (self.basis_max - self.basis_min + 1e-12)

        return y

    def forward(self, x, track_stats: bool = False):
        if x.dim() != 2:
            raise ValueError(f"Expected x shape (batch, in_count), got {tuple(x.shape)}")

        if x.size(1) != self.in_count:
            raise ValueError(
                f"Expected input with {self.in_count} features, got {x.size(1)}"
            )

        # Fixed interval affine transform:
        # x in [-input_abs_max, input_abs_max]
        # maps to u in [basis_min, basis_max]
        u = self.alpha.unsqueeze(0) * x + self.beta.unsqueeze(0)
        # shape: (batch, in_count)

        u_expanded = u.unsqueeze(1).unsqueeze(-1)
        # shape: (batch, 1, in_count, 1)

        basis = self._photonic_basis_from_u(u_expanded)
        # shape: (batch, 1, in_count, num_basis)

        coeffs = self.coeffs.unsqueeze(0)
        # shape: (1, out_count, in_count, num_basis)

        edge_values = (coeffs * basis).sum(dim=-1)
        # shape: (batch, out_count, in_count)

        y = edge_values.sum(dim=-1)
        # shape: (batch, out_count)

        if self.debug:
            print(f"[PhotonicBasisActivationLayerIntervalAffine] x: {x.shape}")
            print(f"[PhotonicBasisActivationLayerIntervalAffine] u: {u.shape}")
            print(f"[PhotonicBasisActivationLayerIntervalAffine] basis: {basis.shape}")
            print(f"[PhotonicBasisActivationLayerIntervalAffine] coeffs: {coeffs.shape}")
            print(f"[PhotonicBasisActivationLayerIntervalAffine] y: {y.shape}")

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
            "alpha_min": self.alpha.min().detach(),
            "alpha_max": self.alpha.max().detach(),
            "beta_min": self.beta.min().detach(),
            "beta_max": self.beta.max().detach(),
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
        learn_affine: bool = True,      # only used if layer_type == "affine"
        b_coef_selected: torch.Tensor = None,  # only used if layer_type == "photonic"
        #clamp_input: bool = True,       # only used if layer_type == "photonic"
        normalize_basis: bool = True,   # only used if layer_type == "photonic"
        check_input_range: bool = False, # only used if layer_type == "photonic"
        basis_min: float = 0.05,
        basis_max: float = 60.0,
        input_abs_max: float = 1.0,
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
        self.basis_min = basis_min
        self.basis_max = basis_max
        self.input_abs_max = input_abs_max

        layer_sizes = [in_count] + self.hidden_layer_sizes + [out_count]

        if layer_type == "standard":
            layer_cls = TanhBasisActivationLayer
        elif layer_type == "affine":
            layer_cls = TanhBasisActivationLayerAffine
        elif layer_type == "photonic":
            layer_cls = PhotonicBasisActivationLayer
        elif layer_type == "photonic_interval_affine":
            layer_cls = PhotonicBasisActivationLayerIntervalAffine

        else:
            raise ValueError("layer_type must be 'standard', 'affine', or 'photonic'.")

        if layer_type in ["photonic", "photonic_interval_affine"]:
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
                    x_min=x_min,
                    x_max=x_max,
                    gamma_scale=gamma_scale,
                    learn_affine=learn_affine,
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

            elif layer_type == "photonic_interval_affine":
                layer = layer_cls(
                    in_count=layer_sizes[i],
                    out_count=layer_sizes[i + 1],
                    b_coef_selected=b_coef_selected,
                    input_abs_max=input_abs_max,
                    basis_min=basis_min,
                    basis_max=basis_max,
                    normalize_basis=normalize_basis,
                    debug=debug,
                )

            layers.append(layer)

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else None

    def _validate_layer(self, layer_idx: int):
        layer = self.layers[layer_idx]

        if not isinstance(
            layer,
            (TanhBasisActivationLayer, TanhBasisActivationLayerAffine, PhotonicBasisActivationLayer),
        ):
            raise TypeError("Selected layer is not a supported KAN layer.")
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

        centers = layer.centers[in_idx]
        slopes = layer.slopes[in_idx]

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
        basis = self._compute_basis_on_grid(layer, in_idx, x)

        x_np = x.detach().cpu().numpy()
        basis_np = basis.detach().cpu().numpy()

        plt.figure(figsize=(8, 5))

        for m in range(layer.num_basis):
            plt.plot(x_np, basis_np[:, m], label=f"basis {m}")

        title_suffix = ""

        if isinstance(layer, TanhBasisActivationLayerAffine):
            centers = layer.centers[in_idx]
            alpha = layer.get_alpha()[in_idx].detach().cpu().item()
            beta = layer.get_beta()[in_idx].detach().cpu().item()

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