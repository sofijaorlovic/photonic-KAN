import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.io import savemat
from pathlib import Path


def _to_matlab_safe(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()

    if isinstance(value, np.ndarray):
        return value

    if isinstance(value, (float, int, bool, str)):
        return value

    if isinstance(value, list):
        return np.array(value, dtype=object)

    if isinstance(value, tuple):
        return np.array(value, dtype=object)

    if isinstance(value, dict):
        return {str(k): _to_matlab_safe(v) for k, v in value.items()}

    return value


def export_plot_data_to_mat(path: str, data: dict, export_dir: str = None):
    """
    Saves plot data to a .mat file.

    If export_dir is provided, the file is saved inside that folder.
    The folder is created automatically if it does not exist.
    """
    if path is None:
        return

    if export_dir is not None:
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        path = export_dir / path
    else:
        path = Path(path)

    matlab_data = {str(k): _to_matlab_safe(v) for k, v in data.items()}
    savemat(str(path), matlab_data)

    print(f"Saved MATLAB plot data to: {path}")


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


# ---------------------------------------------------------------------------
# Photonic activation and field (amplitude) encoding
# ---------------------------------------------------------------------------

def photonic_activation(P: torch.Tensor, b_coef: torch.Tensor):
    """
    Analytic FP-LD activation b(P), evaluated for every selected basis.

        b(P) = b1*log(1 + b2*log(1 + (exp(b3*P) - 1)^b4))
               + b5*P + b6*P^2 + b7*P^3 + b8*P^4

    Args:
        P:      tensor whose LAST dimension has size 1 (broadcast over bases),
                e.g. shape (batch, 1, in_count, 1).
        b_coef: (num_basis, 8)

    Returns:
        tensor with the last dimension of size num_basis.
    """
    shape = [1] * (P.dim() - 1) + [-1]
    b1, b2, b3, b4, b5, b6, b7, b8 = [b_coef[:, k].view(*shape) for k in range(8)]

    inner_base = torch.expm1(b3 * P)
    powered = torch.pow(inner_base, b4)
    inner_log = torch.log1p(powered)
    outer_argument = 1.0 + b2 * inner_log

    return (
        b1 * torch.log(outer_argument)
        + b5 * P
        + b6 * P**2
        + b7 * P**3
        + b8 * P**4
    )


def field_encoded_basis(
    u: torch.Tensor,
    b_coef: torch.Tensor,
    divide_by_m: bool = True,
):
    """
    Field (amplitude) encoding of one photonic edge, as in the updated model:

        1. u                       input amplitude (x, or alpha*x + beta)
        2. u / sqrt(m)             1:m splitter, m = number of bases per edge
        3. P = u^2 / m             optical power entering each FP-LD
        4. b(P)                    photonic activation (power domain)
        5. sqrt(b(P))              output amplitude of each basis
       (6. a_k * sqrt(b_k(P))      done by the layer)
       (7. sum over k and edges    done by the layer)

    No clamping anywhere (not realisable in hardware): b(P) >= 0 for all FP-LD bases on
    [0, 60] and b(0) = 0 exactly, so sqrt(b) is always defined. The derivative of sqrt is
    infinite only at b = 0, i.e. P = 0, which occurs only at layer inputs that need no gradient.

    Args:
        u: tensor with last dimension of size 1.

    Returns:
        dict with P, b, basis (= sqrt(b)), all with last dimension num_basis
        (P keeps last dimension 1).
    """
    m = b_coef.size(0)
    P = u**2 / m if divide_by_m else u**2
    b = photonic_activation(P, b_coef)
    basis = torch.sqrt(b)
    return {"P": P, "b": b, "basis": basis}


def _power_stats(inter: dict):
    """min/max of the photonic power argument P and of the raw activation b."""
    return {
        "min_P": inter["P"].min().detach(),
        "max_P": inter["P"].max().detach(),
        "min_b": inter["b"].min().detach(),
        "max_b": inter["b"].max().detach(),
    }


def amplitude_window_from_power_window(P_min: float, P_max: float, m: int, divide_by_m: bool = True):
    """
    Amplitude interval [u_min, u_max] such that u^2/m spans [P_min, P_max].
    (u^2 if divide_by_m is False.)
    """
    m_eff = m if divide_by_m else 1
    return float(np.sqrt(m_eff * P_min)), float(np.sqrt(m_eff * P_max))


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
        field_encoding: bool = False,
        divide_by_m: bool = True,
    ):
        super().__init__()

        if b_coef_selected is None:
            raise ValueError("b_coef_selected must be provided for PhotonicBasisActivationLayer.")

        if field_encoding and normalize_basis:
            raise ValueError("normalize_basis is not supported together with field_encoding.")

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

        # Field encoding: edge = sum_k a_k * sqrt(b_k(x^2/m)).
        # With field_encoding=True, [x_min, x_max] is the range of the input
        # AMPLITUDE x (the range check is applied to x, not to x^2/m).
        self.field_encoding = bool(field_encoding)
        self.divide_by_m = bool(divide_by_m)

        # Fixed photonic activation coefficients.
        # These are NOT trained.
        self.register_buffer("b_coef", b_coef_selected.clone().float())

        # Trainable KAN coefficients.
        self.coeffs = nn.Parameter(
            0.01 * torch.randn(out_count, in_count, self.num_basis)
        )

    def _edge_intermediates(self, x: torch.Tensor):
        """
        All intermediate quantities of the basis evaluation.

        Args:
            x: shape (..., 1), input amplitude (field encoding) or power (old mode)

        Returns:
            dict with u, P, b, basis
        """
        if not self.field_encoding:
            basis = self._photonic_basis(x)
            return {"u": x, "P": x, "b": basis, "basis": basis}

        if self.check_input_range:
            if torch.any(x < self.x_min) or torch.any(x > self.x_max):
                raise ValueError(
                    f"Input amplitude outside expected range [{self.x_min}, {self.x_max}]. "
                    f"Got min={x.min().item()}, max={x.max().item()}."
                )

        out = field_encoded_basis(x, self.b_coef, self.divide_by_m)
        out["u"] = x
        return out

    def _edge_basis(self, x: torch.Tensor):
        return self._edge_intermediates(x)["basis"]

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

        inter = self._edge_intermediates(x_expanded)
        basis = inter["basis"]
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
        stats.update(_power_stats(inter))

        return y, stats


class PhotonicBasisActivationLayerPositive(PhotonicBasisActivationLayer):
    """
    Photonic KAN layer with strictly positive trainable coefficients.

    Instead of training coeffs directly, this layer trains raw_coeffs and maps them as:

        coeffs = softplus(raw_coeffs) + coeff_eps

    This guarantees coeffs > 0 while keeping optimization unconstrained.
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
        coeff_eps: float = 1e-9,
        field_encoding: bool = False,
        divide_by_m: bool = True,
    ):
        super().__init__(
            in_count=in_count,
            out_count=out_count,
            b_coef_selected=b_coef_selected,
            x_min=x_min,
            x_max=x_max,
            normalize_basis=normalize_basis,
            check_input_range=check_input_range,
            debug=debug,
            field_encoding=field_encoding,
            divide_by_m=divide_by_m,
        )

        self.coeff_eps = float(coeff_eps)

        # Remove the unrestricted coeffs created by the parent class.
        del self.coeffs

        # Train unconstrained raw coefficients.
        self.raw_coeffs = nn.Parameter(
            0.01 * torch.randn(out_count, in_count, self.num_basis)
        )

    def get_coeffs(self):
        return F.softplus(self.raw_coeffs) + self.coeff_eps

    def forward(self, x, track_stats: bool = False):
        if x.dim() != 2:
            raise ValueError(f"Expected x shape (batch, in_count), got {tuple(x.shape)}")

        if x.size(1) != self.in_count:
            raise ValueError(
                f"Expected input with {self.in_count} features, got {x.size(1)}"
            )

        x_expanded = x.unsqueeze(1).unsqueeze(-1)

        inter = self._edge_intermediates(x_expanded)
        basis = inter["basis"]

        coeffs = self.get_coeffs().unsqueeze(0)

        edge_values = (coeffs * basis).sum(dim=-1)
        y = edge_values.sum(dim=-1)

        if self.debug:
            print(f"[PhotonicBasisActivationLayerPositive] x: {x.shape}")
            print(f"[PhotonicBasisActivationLayerPositive] basis: {basis.shape}")
            print(f"[PhotonicBasisActivationLayerPositive] coeffs: {coeffs.shape}")
            print(f"[PhotonicBasisActivationLayerPositive] edge_values: {edge_values.shape}")
            print(f"[PhotonicBasisActivationLayerPositive] y: {y.shape}")

        if not track_stats:
            return y

        stats = {
            "min_input": x.min().detach(),
            "max_input": x.max().detach(),
            "min_basis": basis.min().detach(),
            "max_basis": basis.max().detach(),
            "min_coeff": self.get_coeffs().min().detach(),
            "max_coeff": self.get_coeffs().max().detach(),
            "min_output": y.min().detach(),
            "max_output": y.max().detach(),
        }
        stats.update(_power_stats(inter))

        return y, stats


class PhotonicBasisActivationLayerIntervalAffineClean(nn.Module):
    """
    Photonic KAN layer with a fixed affine input mapping u = alpha * x + beta.

    The input interval x in [-input_neg_max, input_pos_max] is mapped linearly
    onto an interval of u. alpha and beta are buffers (not trained), shared by
    all edges leaving the same input node.

    Old mode (field_encoding=False):
        u in [basis_min, basis_max]
        edge = sum_k a_k * b_k(u)

    Field encoding (field_encoding=True):
        [basis_min, basis_max] is the POWER window P seen by the FP-LD.
        u is mapped onto the amplitude window
            [sqrt(m * basis_min), sqrt(m * basis_max)]
        so that P = u^2 / m spans exactly [basis_min, basis_max], and
        edge = sum_k a_k * sqrt(b_k(u^2 / m)).
        (with divide_by_m=False: window [sqrt(basis_min), sqrt(basis_max)], P = u^2;
         the basis sees identical values, so results do not change.)

    No clamp on u. No basis normalization.
    """

    def __init__(
        self,
        in_count: int,
        out_count: int,
        b_coef_selected: torch.Tensor,
        input_neg_max: float,
        input_pos_max: float,
        basis_min: float = 0.05,
        basis_max: float = 60.0,
        debug: bool = False,
        field_encoding: bool = False,
        divide_by_m: bool = True,
        warn_negative_u: bool = True,
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

        if basis_max <= basis_min:
            raise ValueError("basis_max must be greater than basis_min.")

        if field_encoding and basis_min < 0:
            raise ValueError("With field_encoding, basis_min is an optical power and must be >= 0.")

        self.in_count = in_count
        self.out_count = out_count
        self.num_basis = b_coef_selected.size(0)

        self.input_neg_max = float(input_neg_max)
        self.input_pos_max = float(input_pos_max)
        self.basis_min = float(basis_min)
        self.basis_max = float(basis_max)
        self.debug = debug

        self.field_encoding = bool(field_encoding)
        self.divide_by_m = bool(divide_by_m)
        self.warn_negative_u = bool(warn_negative_u)
        self._warned_negative_u = False

        self.register_buffer("b_coef", b_coef_selected.clone().float())
        self.register_buffer("alpha", torch.zeros(in_count, dtype=torch.float32))
        self.register_buffer("beta", torch.zeros(in_count, dtype=torch.float32))
        self._update_affine()

        self.coeffs = nn.Parameter(
            0.01 * torch.randn(out_count, in_count, self.num_basis)
        )

    # ----- affine mapping -------------------------------------------------

    @property
    def input_min(self):
        return -self.input_neg_max

    @property
    def input_max(self):
        return self.input_pos_max

    def amplitude_window(self):
        """Target interval of u = alpha*x + beta."""
        if not self.field_encoding:
            return self.basis_min, self.basis_max
        return amplitude_window_from_power_window(
            self.basis_min, self.basis_max, self.num_basis, self.divide_by_m
        )

    def _update_affine(self):
        width = self.input_pos_max + self.input_neg_max
        if width <= 0:
            raise ValueError(
                f"Empty input interval [{-self.input_neg_max}, {self.input_pos_max}]."
            )
        u_lo, u_hi = self.amplitude_window()
        alpha_value = (u_hi - u_lo) / width
        beta_value = u_lo + alpha_value * self.input_neg_max
        with torch.no_grad():
            self.alpha.fill_(alpha_value)
            self.beta.fill_(beta_value)

    def set_input_range(self, input_min: float, input_max: float, margin: float = 1.0, min_width: float = 1e-6):
        """
        Re-map the affine transformation to the input interval [input_min, input_max],
        widened symmetrically around its centre by `margin`.
        """
        center = 0.5 * (input_min + input_max)
        half = max(0.5 * (input_max - input_min), 0.5 * min_width) * margin
        self.input_neg_max = -(center - half)
        self.input_pos_max = center + half
        self._update_affine()
        self._warned_negative_u = False

    # ----- basis evaluation -----------------------------------------------

    def _photonic_basis_from_u(self, u: torch.Tensor):
        """Raw b(u) (old mode). Kept for backward compatibility."""
        return photonic_activation(u, self.b_coef)

    def _edge_intermediates_from_u(self, u: torch.Tensor):
        """
        Args:
            u: shape (..., 1), affine-transformed input.
        Returns:
            dict with u, P, b, basis
        """
        if not self.field_encoding:
            b = photonic_activation(u, self.b_coef)
            return {"u": u, "P": u, "b": b, "basis": b}

        if self.warn_negative_u and not self._warned_negative_u and torch.any(u < 0):
            self._warned_negative_u = True   # warn once per layer; see stats["min_affine_input"]
            warnings.warn(
                "Affine input u = alpha*x + beta became negative "
                f"(min u = {u.min().item():.4g}). Squaring folds it back to a positive power; "
                "widen the input range of this layer (set_input_range / input_neg_max)."
            )

        out = field_encoded_basis(u, self.b_coef, self.divide_by_m)
        out["u"] = u
        return out

    def _edge_basis_from_u(self, u: torch.Tensor):
        return self._edge_intermediates_from_u(u)["basis"]

    def forward(self, x, track_stats: bool = False):
        if x.dim() != 2:
            raise ValueError(f"Expected x shape (batch, in_count), got {tuple(x.shape)}")

        if x.size(1) != self.in_count:
            raise ValueError(
                f"Expected input with {self.in_count} features, got {x.size(1)}"
            )

        u = self.alpha.unsqueeze(0) * x + self.beta.unsqueeze(0)

        u_expanded = u.unsqueeze(1).unsqueeze(-1)
        inter = self._edge_intermediates_from_u(u_expanded)
        basis = inter["basis"]

        coeffs = self.coeffs.unsqueeze(0)
        edge_values = (coeffs * basis).sum(dim=-1)

        y = edge_values.sum(dim=-1)

        if not track_stats:
            return y

        u_lo, u_hi = self.amplitude_window()
        stats = {
            "min_input": x.min().detach(),
            "max_input": x.max().detach(),
            "min_affine_input": u.min().detach(),
            "max_affine_input": u.max().detach(),
            "frac_outside_window": ((u < u_lo - 1e-9) | (u > u_hi + 1e-9)).double().mean().detach(),
            "min_basis": basis.min().detach(),
            "max_basis": basis.max().detach(),
            "min_output": y.min().detach(),
            "max_output": y.max().detach(),
        }
        stats.update(_power_stats(inter))

        return y, stats

class KAN(nn.Module):
    """
    Multi-layer KAN using either:
      - TanhBasisActivationLayer
      - TanhBasisActivationLayerAffine
      - PhotonicBasisActivationLayer
      - PhotonicBasisActivationLayerIntervalAffineClean
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
        input_abs_max: float = None,      # old symmetric option, optional backward compatibility
        input_neg_max=None,               # new asymmetric negative-side range
        input_pos_max=None,               # new asymmetric positive-side range
        coeff_eps: float = 1e-9,
        field_encoding: bool = False,     # photonic layers: edge = sum a_k sqrt(b_k(u^2/m))
        divide_by_m: bool = True,         # field encoding: include the 1/sqrt(m) splitter
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
        self.input_neg_max = input_neg_max
        self.input_pos_max = input_pos_max
        self.coeff_eps = coeff_eps
        self.field_encoding = field_encoding
        self.divide_by_m = divide_by_m

        field_kwargs = dict(
            field_encoding=field_encoding,
            divide_by_m=divide_by_m,
        )

        layer_sizes = [in_count] + self.hidden_layer_sizes + [out_count]
        num_layers = len(layer_sizes) - 1

        def _expand_layer_values(value, num_layers, name, default=None):
            """
            Expands scalar/list/tuple values to a list with one entry per layer.
            Each entry can itself be a scalar or a per-input list/tensor.
            """
            if value is None:
                if default is None:
                    return None
                value = default

            if isinstance(value, (list, tuple)):
                if len(value) == num_layers:
                    return list(value)
                else:
                    # Treat as a single per-input vector shared across layers only if not layer-specific.
                    # Usually for multi-layer networks, user should pass one value per layer.
                    raise ValueError(
                        f"{name} must have one entry per layer. Expected {num_layers}, got {len(value)}."
                    )

            return [value] * num_layers


        # New asymmetric input range handling.
        # Backward compatibility:
        # if input_neg_max/input_pos_max are not provided, use symmetric input_abs_max.
        if input_neg_max is None and input_pos_max is None:
            if input_abs_max is None:
                input_abs_max = 10.0

            input_neg_max_by_layer = _expand_layer_values(
                input_abs_max, num_layers, "input_abs_max"
            )
            input_pos_max_by_layer = _expand_layer_values(
                input_abs_max, num_layers, "input_abs_max"
            )

        elif input_neg_max is not None and input_pos_max is not None:
            input_neg_max_by_layer = _expand_layer_values(
                input_neg_max, num_layers, "input_neg_max"
            )
            input_pos_max_by_layer = _expand_layer_values(
                input_pos_max, num_layers, "input_pos_max"
            )

        else:
            raise ValueError(
                "You must provide both input_neg_max and input_pos_max, or neither."
            )

        self.input_neg_max_by_layer = input_neg_max_by_layer
        self.input_pos_max_by_layer = input_pos_max_by_layer

        if layer_type == "standard":
            layer_cls = TanhBasisActivationLayer
        elif layer_type == "affine":
            layer_cls = TanhBasisActivationLayerAffine
        elif layer_type == "photonic":
            layer_cls = PhotonicBasisActivationLayer
        elif layer_type == "photonic_positive":
            layer_cls = PhotonicBasisActivationLayerPositive
        elif layer_type == "photonic_interval_affine_clean":
            layer_cls = PhotonicBasisActivationLayerIntervalAffineClean

        else:
            raise ValueError("layer_type must be 'standard', 'affine', or 'photonic', or 'photonic_positive', or 'photonic_interval_affine_clean'.")

        if layer_type in ["photonic", "photonic_positive", "photonic_interval_affine_clean"]:
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
            layer_input_neg_max = input_neg_max_by_layer[i]
            layer_input_pos_max = input_pos_max_by_layer[i]
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
                    input_neg_max=layer_input_neg_max,
                    input_pos_max=layer_input_pos_max,
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
                    **field_kwargs,
                )

            elif layer_type == "photonic_positive":
                layer = layer_cls(
                    in_count=layer_sizes[i],
                    out_count=layer_sizes[i + 1],
                    b_coef_selected=b_coef_selected,
                    x_min=x_min,
                    x_max=x_max,
                    normalize_basis=normalize_basis,
                    check_input_range=check_input_range,
                    debug=debug,
                    coeff_eps=coeff_eps,
                    **field_kwargs,
                )

            elif layer_type == "photonic_interval_affine_clean":
                layer = layer_cls(
                    in_count=layer_sizes[i],
                    out_count=layer_sizes[i + 1],
                    b_coef_selected=b_coef_selected,
                    input_neg_max=layer_input_neg_max,
                    input_pos_max=layer_input_pos_max,
                    basis_min=basis_min,
                    basis_max=basis_max,
                    debug=debug,
                    **field_kwargs,
                )
            
            layers.append(layer)

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else None
    
    def _get_layer_coeffs(self, layer):
        """
        Returns the effective coefficient tensor for any supported KAN layer.

        For normal layers:
            returns layer.coeffs

        For positive-coefficient layers:
            returns softplus(raw_coeffs) + eps
        """
        if hasattr(layer, "get_coeffs"):
            return layer.get_coeffs()

        if hasattr(layer, "coeffs"):
            return layer.coeffs

        raise AttributeError(
            f"Layer type {type(layer).__name__} does not expose coeffs or get_coeffs()."
        )

    def _validate_layer(self, layer_idx: int):
        layer = self.layers[layer_idx]

        supported_layers = (
            TanhBasisActivationLayer,
            TanhBasisActivationLayerAffine,
            PhotonicBasisActivationLayer,
            PhotonicBasisActivationLayerPositive,
            PhotonicBasisActivationLayerIntervalAffineClean,
        )

        if not isinstance(layer, supported_layers):
            raise TypeError(
                f"Selected layer is not a supported KAN layer. "
                f"Got layer type: {type(layer).__name__}"
            )

        return layer

    def _grid_intermediates(self, layer, in_idx: int, x: torch.Tensor):
        """
        Photonic layers only: all intermediate quantities on a 1D grid x of shape (R,).

        Returns dict with
            x:     (R,)   node variable (layer input)
            u:     (R,)   amplitude after the affine map (= x without affine map)
            P:     (R,)   argument of the FP-LD activation (u^2/m with field encoding)
            b:     (R, M) raw photonic activation b(P)
            basis: (R, M) basis value used in the edge (sqrt(b) with field encoding)
        """
        x = x.to(device=layer.b_coef.device, dtype=layer.b_coef.dtype)

        if isinstance(layer, PhotonicBasisActivationLayer):
            inter = layer._edge_intermediates(x.view(-1, 1, 1, 1))
        elif isinstance(layer, PhotonicBasisActivationLayerIntervalAffineClean):
            u = layer.alpha[in_idx] * x + layer.beta[in_idx]
            inter = layer._edge_intermediates_from_u(u.view(-1, 1, 1, 1))
        else:
            raise TypeError(f"{type(layer).__name__} is not a photonic layer.")

        return {
            "x": x,
            "u": inter["u"].reshape(-1),
            "P": inter["P"].reshape(-1),
            "b": inter["b"][:, 0, 0, :],
            "basis": inter["basis"][:, 0, 0, :],
        }

    def _compute_basis_on_grid(self, layer, in_idx: int, x: torch.Tensor):
        """
        Returns basis values of shape (R, M) for a 1D input grid x of shape (R,).
        Supports standard, affine, and photonic layer types.
        For photonic layers with field encoding this is sqrt(b(u^2/m)).
        """
        if isinstance(layer, (PhotonicBasisActivationLayer, PhotonicBasisActivationLayerIntervalAffineClean)):
            return self._grid_intermediates(layer, in_idx, x)["basis"]

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
        
    def _photonic_grid_export(self, layer, in_idx: int, x: torch.Tensor):
        """Extra MATLAB fields for photonic layers: u, P = u^2/m, raw b(P), m, flags."""
        if not isinstance(layer, (PhotonicBasisActivationLayer, PhotonicBasisActivationLayerIntervalAffineClean)):
            return {}
        inter = self._grid_intermediates(layer, in_idx, x)
        return {
            "field_encoding": bool(getattr(layer, "field_encoding", False)),
            "divide_by_m": bool(getattr(layer, "divide_by_m", True)),
            "num_basis_m": layer.num_basis,
            "u": inter["u"],
            "P": inter["P"],
            "b_raw": inter["b"],
        }

    def plot_edge_function(
        self,
        layer_idx: int,
        out_idx: int,
        in_idx: int,
        x_range=None,
        resolution: int = 400,
        show_basis: bool = False,
        mat_export_path: str = None,
        mat_export_dir: str = None,
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

        x = torch.linspace(xmin, xmax, resolution, dtype=torch.float64)

        coeffs = self._get_layer_coeffs(layer)[out_idx, in_idx]  # (M,)
        basis = self._compute_basis_on_grid(layer, in_idx, x)   # (R, M)
        x = x.to(device=basis.device, dtype=basis.dtype)
        coeffs = coeffs.to(dtype=basis.dtype)
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

        export_plot_data_to_mat(
            mat_export_path,
            {
                "plot_type": "edge_function",
                "layer_idx": layer_idx,
                "out_idx": out_idx,
                "in_idx": in_idx,
                "x_range": np.array([xmin, xmax]),
                "resolution": resolution,
                "x": x,
                "basis": basis,
                "coeffs": coeffs,
                "contributions": contributions,
                "phi": phi,
                "show_basis": show_basis,
                **self._photonic_grid_export(layer, in_idx, x),
            },
            export_dir=mat_export_dir,
        )

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

        x = torch.linspace(xmin, xmax, resolution, dtype=torch.float64)
        x_np = x.detach().cpu().numpy()

        plt.figure(figsize=(9, 6))

        for in_idx in range(input_count):
            coeffs = self._get_layer_coeffs(layer)[out_idx, in_idx]
            basis = self._compute_basis_on_grid(layer, in_idx, x)
            phi = (basis * coeffs.to(basis.dtype).unsqueeze(0)).sum(dim=1)

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
        mat_export_path: str = None,
        mat_export_dir: str = None,
    ):
        layer = self._validate_layer(layer_idx)

        if not (0 <= in_idx < layer.in_count):
            raise ValueError(f"in_idx must be in [0, {layer.in_count - 1}]")

        if x_range is None:
            xmin, xmax = self.x_min, self.x_max
        else:
            xmin, xmax = x_range

        x = torch.linspace(xmin, xmax, resolution, dtype=torch.float64)
        basis = self._compute_basis_on_grid(layer, in_idx, x)

        export_data = {
            "plot_type": "basis_functions",
            "layer_idx": layer_idx,
            "in_idx": in_idx,
            "x_range": np.array([xmin, xmax]),
            "resolution": resolution,
            "x": x,
            "basis": basis,
        }

        if hasattr(layer, "centers"):
            export_data["centers"] = layer.centers[in_idx]

        if hasattr(layer, "slopes"):
            export_data["slopes"] = layer.slopes[in_idx]

        if hasattr(layer, "alpha"):
            export_data["alpha"] = layer.alpha[in_idx]

        if hasattr(layer, "beta"):
            export_data["beta"] = layer.beta[in_idx]

        if hasattr(layer, "b_coef"):
            export_data["b_coef"] = layer.b_coef

        export_data.update(self._photonic_grid_export(layer, in_idx, x))

        export_plot_data_to_mat(
            mat_export_path,
            export_data,
            export_dir=mat_export_dir,
        )

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

        if getattr(layer, "field_encoding", False):
            title_suffix = " (field encoding: sqrt(b(u^2/m)))"

        elif isinstance(layer, PhotonicBasisActivationLayerPositive):
            title_suffix = " (positive photonic bases)"

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

    def plot_coefficient_vector(
        self,
        layer_idx: int,
        out_idx: int,
        in_idx: int,
        mat_export_path: str = None,
        mat_export_dir: str = None,
    ):
        layer = self._validate_layer(layer_idx)

        if not (0 <= out_idx < layer.out_count):
            raise ValueError(f"out_idx must be in [0, {layer.out_count - 1}]")
        if not (0 <= in_idx < layer.in_count):
            raise ValueError(f"in_idx must be in [0, {layer.in_count - 1}]")

        coeffs_tensor = self._get_layer_coeffs(layer)[out_idx, in_idx]
        coeffs = coeffs_tensor.detach().cpu().numpy()
        
        export_plot_data_to_mat(
            mat_export_path,
            {
                "plot_type": "coefficient_vector",
                "layer_idx": layer_idx,
                "out_idx": out_idx,
                "in_idx": in_idx,
                "basis_index": np.arange(layer.num_basis),
                "coeffs": coeffs,
            },
            export_dir=mat_export_dir,
        )

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
        mat_export_prefix: str = None,
        mat_export_dir: str = None,
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

                if mat_export_prefix is not None:
                    export_plot_data_to_mat(
                        f"{mat_export_prefix}_varied_x{varied_idx}.mat",
                        {
                            "plot_type": "target_approximation",
                            "varied_idx": varied_idx,
                            "x_range": np.array([xmin, xmax]),
                            "resolution": resolution,
                            "fixed_values": np.array(fixed_values),
                            "x_grid": x_grid,
                            "x_plot": x_plot,
                            "y_true": y_true,
                            "y_pred": y_pred,
                        },
                    export_dir=mat_export_dir,
                    )

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


    def print_photonic_affine_params(self):
        """
        Print alpha and beta values for all photonic interval affine layers.

        In PhotonicBasisActivationLayerIntervalAffineClean:
        - alpha has shape (in_count,)
        - beta has shape (in_count,)
        - alpha[in_idx], beta[in_idx] are shared across all outgoing edges
            from input feature in_idx to every output unit.
        """
        print("Photonic interval affine parameters:")

        for layer_idx, layer in enumerate(self.layers):
            if isinstance(layer, PhotonicBasisActivationLayerIntervalAffineClean):
                alpha_cpu = layer.alpha.detach().cpu()
                beta_cpu = layer.beta.detach().cpu()

                print(f"\nLayer {layer_idx}: {type(layer).__name__}")
                print(f"  in_count = {layer.in_count}")
                print(f"  out_count = {layer.out_count}")
                print(f"  input_neg_max = {layer.input_neg_max}")
                print(f"  input_pos_max = {layer.input_pos_max}")
                if layer.field_encoding:
                    u_lo, u_hi = layer.amplitude_window()
                    print(f"  field encoding: edge = sum_k a_k sqrt(b_k(u^2/m)), m = {layer.num_basis}, divide_by_m = {layer.divide_by_m}")
                    print(f"  power window P = [{layer.basis_min}, {layer.basis_max}]")
                    print(f"  amplitude window u = alpha*x+beta in [{u_lo:.6g}, {u_hi:.6g}]")
                else:
                    print(f"  basis interval = [{layer.basis_min}, {layer.basis_max}]")
                print(f"  alpha shape = {tuple(alpha_cpu.shape)}")
                print(f"  beta shape  = {tuple(beta_cpu.shape)}")
                print(f"  alpha values = {alpha_cpu.numpy()}")
                print(f"  beta values  = {beta_cpu.numpy()}")

                if alpha_cpu.numel() > 0:
                    print(f"  alpha unique = {torch.unique(alpha_cpu).numpy()}")
                    print(f"  beta unique  = {torch.unique(beta_cpu).numpy()}")

# ---------------------------------------------------------------------------
# Independent NumPy forward pass and full-pipeline MATLAB export
# ---------------------------------------------------------------------------

def _np_photonic_activation(P: np.ndarray, b_coef: np.ndarray):
    """NumPy re-implementation of b(P). P: (..., 1), b_coef: (M, 8) -> (..., M)."""
    b1, b2, b3, b4, b5, b6, b7, b8 = [b_coef[:, k] for k in range(8)]
    return (
        b1 * np.log(1.0 + b2 * np.log1p(np.expm1(b3 * P) ** b4))
        + b5 * P + b6 * P**2 + b7 * P**3 + b8 * P**4
    )


def manual_forward_numpy(model, x):
    """
    Re-computes the forward pass of a photonic KAN in NumPy, step by step,
    independently of the PyTorch layer code.

    Supported layers: PhotonicBasisActivationLayer(+Positive),
    PhotonicBasisActivationLayerIntervalAffineClean (old mode or field encoding).

    Returns:
        y (batch, out_count), list of per-layer dicts with every intermediate stage:
            input, alpha, beta, u, P, b, basis, coeffs, contributions, edge_values, output
    """
    h = x.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(x) else np.asarray(x, dtype=np.float64)
    layers_out = []

    for layer in model.layers:
        b_coef = layer.b_coef.detach().cpu().numpy().astype(np.float64)
        m = b_coef.shape[0]
        coeffs = model._get_layer_coeffs(layer).detach().cpu().numpy().astype(np.float64)  # (O, I, M)
        field = bool(getattr(layer, "field_encoding", False))
        divide_by_m = bool(getattr(layer, "divide_by_m", True))

        if isinstance(layer, PhotonicBasisActivationLayerIntervalAffineClean):
            alpha = layer.alpha.detach().cpu().numpy().astype(np.float64)
            beta = layer.beta.detach().cpu().numpy().astype(np.float64)
        elif isinstance(layer, PhotonicBasisActivationLayer):
            if getattr(layer, "normalize_basis", False):
                raise NotImplementedError("normalize_basis is not supported in manual_forward_numpy.")
            alpha = np.ones(layer.in_count)
            beta = np.zeros(layer.in_count)
        else:
            raise TypeError(f"Unsupported layer {type(layer).__name__}")

        u = alpha[None, :] * h + beta[None, :]                       # (N, I)
        if field:
            P = u**2 / m if divide_by_m else u**2                      # (N, I)
            b = _np_photonic_activation(P[..., None], b_coef)          # (N, I, M)
            basis = np.sqrt(b)
        else:
            P = u
            b = _np_photonic_activation(P[..., None], b_coef)
            basis = b

        contributions = coeffs[None, :, :, :] * basis[:, None, :, :]  # (N, O, I, M)
        edge_values = contributions.sum(axis=-1)                       # (N, O, I)
        out = edge_values.sum(axis=-1)                                 # (N, O)

        layers_out.append({
            "field_encoding": field, "divide_by_m": divide_by_m, "num_basis_m": m,
            "b_coef": b_coef, "alpha": alpha, "beta": beta, "coeffs": coeffs,
            "input": h, "u": u, "P": P, "b": b, "basis": basis,
            "contributions": contributions, "edge_values": edge_values, "output": out,
        })
        h = out

    return h, layers_out


def export_full_forward_to_mat(model, x, path, extra: dict = None, include_contributions: bool = False):
    """
    Saves the complete forward pass for inputs x to one .mat file:

        layer{L}_input          (N, I)    node variables entering layer L
        layer{L}_alpha/_beta    (I,)      affine map (1 / 0 without affine map)
        layer{L}_u              (N, I)    u = alpha*x + beta
        layer{L}_P              (N, I)    FP-LD input power u^2/m (= u in old mode)
        layer{L}_b              (N, I, M) raw photonic activation b(P)
        layer{L}_basis          (N, I, M) sqrt(b) (field encoding) or b (old mode)
        layer{L}_coeffs         (O, I, M) effective expansion coefficients a
        layer{L}_edge_values    (N, O, I) phi_{o,i} = sum_m a_{o,i,m} * basis_{i,m}
        layer{L}_output         (N, O)    sum_i phi_{o,i}
        + min/max of each, and the PyTorch vs NumPy check.

    Returns:
        max |model(x) - NumPy forward| relative to max |y|.
    """
    model.eval()
    with torch.no_grad():
        y_torch = model(x).detach().cpu().numpy().astype(np.float64)

    y_np, layers_out = manual_forward_numpy(model, x)
    max_abs_err = float(np.max(np.abs(y_torch - y_np)))
    rel_err = max_abs_err / max(float(np.max(np.abs(y_np))), 1e-300)

    data = {
        "num_layers": len(model.layers),
        "layer_sizes": np.array([model.in_count] + list(model.hidden_layer_sizes) + [model.out_count]),
        "layer_type": model.layer_type,
        "y_pytorch": y_torch,
        "y_manual_numpy": y_np,
        "max_abs_error_pytorch_vs_numpy": max_abs_err,
        "rel_error_pytorch_vs_numpy": rel_err,
    }

    for L, d in enumerate(layers_out):
        p = f"layer{L}"
        data[f"{p}_field_encoding"] = int(d["field_encoding"])
        data[f"{p}_divide_by_m"] = int(d["divide_by_m"])
        data[f"{p}_num_basis_m"] = d["num_basis_m"]
        data[f"{p}_b_coef"] = d["b_coef"]
        data[f"{p}_alpha"] = d["alpha"]
        data[f"{p}_beta"] = d["beta"]
        data[f"{p}_coeffs"] = d["coeffs"]
        layer = model.layers[L]
        if isinstance(layer, PhotonicBasisActivationLayerIntervalAffineClean):
            data[f"{p}_input_range"] = np.array([layer.input_min, layer.input_max])
            data[f"{p}_power_window"] = np.array([layer.basis_min, layer.basis_max])
            data[f"{p}_amplitude_window"] = np.array(layer.amplitude_window())
        keys = ["input", "u", "P", "b", "basis", "edge_values", "output"]
        if include_contributions:
            keys.append("contributions")
        for k in keys:
            data[f"{p}_{k}"] = d[k]
            data[f"{p}_min_{k}"] = float(np.min(d[k]))
            data[f"{p}_max_{k}"] = float(np.max(d[k]))

    if extra:
        data.update({str(k): _to_matlab_safe(v) for k, v in extra.items()})

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(str(path), {k: _to_matlab_safe(v) for k, v in data.items()}, do_compression=True)
    print(f"Saved full forward pass to: {path}")
    print(f"Max |PyTorch - NumPy| = {max_abs_err:.3e}  (relative to max |y|: {rel_err:.3e})")
    return rel_err
