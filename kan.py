# kan.py
import torch
import torch.nn as nn
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt


def init_param(shape, low=-0.25, high=0.25):
    """Utility for initializing learnable parameters."""
    return nn.Parameter(torch.empty(shape).uniform_(low, high))


class TanhActivation(nn.Module):
    """Custom positive Tanh-based activation used as a building block in KAN."""

    def __init__(self, in_count: int, out_count: int, debug: bool = False, return_max: bool = False):
        super().__init__()
        self.log_alpha = init_param((out_count, in_count))
        self.log_beta = init_param((out_count, in_count))
        self.p = init_param((out_count, in_count))
        self.q = init_param((out_count, in_count))
        self.c = 5.0  # Can be made learnable if desired

        self.debug = debug
        self.return_max = return_max
        self.last_y_pre_sum = None

    def forward(self, x):
        # Enforce positive input
        x = torch.relu(x)
        if self.debug and (x < 0).any():
            raise ValueError("Negative input detected after ReLU!")

        # Shape: (batch, 1, in_count)
        x = x.unsqueeze(1)

        alpha = self.log_alpha
        beta = self.log_beta
        p = self.p
        q = self.q

        f = lambda z, a: 0.5 * (1 + torch.tanh(self.c * z - a))
        y = alpha * f(x, p) + beta * (1 - f(x, q))

        self.last_y_pre_sum = y
        max_last_column = y[:, -1].max().item()
        min_last_column = y[:, -1].min().item()

        # Sum over inputs to get shape (batch, out_count)
        y = y.sum(dim=-1)

        # Ensure positive output
        y = torch.nn.functional.softplus(y)
        if self.debug and (y <= 0).any():
            raise ValueError("Non-positive output detected after Softplus!")

        if self.return_max:
            return y, {"max": max_last_column, "min": min_last_column}
        return y


class BSplineActivationLayer(nn.Module):
    """Activation layer defined by B-splines over a learnable detuning parameter w."""

    def __init__(self, in_count: int, out_count: int, debug: bool = False, return_max: bool = False):
        super().__init__()
        self.in_count = in_count
        self.out_count = out_count
        self.debug = debug
        self.return_max = return_max

        self.raw_gamma = nn.Parameter(torch.zeros(out_count, in_count))
        self.fixed_downscale = 1 / float(out_count)

        self.output_min = None
        self.input_min = None
        self.input_max = None
        self.min_gamma = None
        self.max_gamma = None

        data = loadmat("activation_KAN_fit_B_Sofija_novo1.mat", simplify_cells=True)
        c_coef_spline = data["c_coef_spline"]
        self.splines = []
        for spline in c_coef_spline:
            breaks = torch.tensor(spline["breaks"], dtype=torch.float32)
            coefs = torch.tensor(spline["coefs"], dtype=torch.float32)
            self.splines.append((breaks, coefs))

        self.mu_detuning = torch.tensor(data["mu_detuning"], dtype=torch.float64)
        self.sigma_detuning = torch.tensor(data["sigma_detuning"], dtype=torch.float64)
        self.w = nn.Parameter(5 + (35.5 - 5) * torch.rand(out_count, in_count, dtype=torch.float64))

    def clamp_w(self):
        return torch.clamp(self.w, min=5.5, max=35.5)

    def normalize_w(self):
        return (self.w - self.mu_detuning) / self.sigma_detuning

    def clamp_normalized_w(self, w_norm: torch.Tensor) -> torch.Tensor:
        """Clamp a normalized detuning to the valid normalized interval."""
        norm_min = (5 - self.mu_detuning) / self.sigma_detuning
        norm_max = (35.5 - self.mu_detuning) / self.sigma_detuning
        return torch.clamp(w_norm, min=norm_min, max=norm_max)

    def _eval_bspline(self, w: torch.Tensor, breaks: torch.Tensor, coefs: torch.Tensor) -> torch.Tensor:
        """Evaluate cubic spline at value(s) w using piecewise polynomial (pp-form)."""
        w_clamped = torch.clamp(w, min=breaks[0].item(), max=breaks[-1].item() - 1e-6)
        interval_idx = torch.bucketize(w_clamped, breaks) - 1
        interval_idx = interval_idx.clamp(min=0, max=coefs.shape[0] - 1)
        a = coefs[interval_idx]  # (..., 4)
        a3, a2, a1, a0 = a.unbind(dim=-1)
        t = w_clamped - breaks[interval_idx]
        return a3 * t**3 + a2 * t**2 + a1 * t + a0

    def forward(self, x):
        current_min = x.min().detach()
        current_max = x.max().detach()
        x = torch.relu(x)

        if self.input_min is None or current_min < self.input_min:
            self.input_min = current_min
        if self.input_max is None or current_max > self.input_max:
            self.input_max = current_max

        w = self.clamp_w()
        w_norm = (w - self.mu_detuning) / self.sigma_detuning

        # Evaluate splines: b1..b5
        b_vals = [self._eval_bspline(w_norm, breaks, coefs) for (breaks, coefs) in self.splines]
        b1, b2, b3, b4, b5 = b_vals

        x = x.unsqueeze(1)  # (batch, 1, in_count)

        exp_input = b3 * x
        exp_result = torch.exp(exp_input)
        base = exp_result - 1
        base_log = base ** b4
        log_argument = 1 + base_log
        log1p_term = torch.log(1 + b2 * torch.log(log_argument))

        y_ = b1 * log1p_term + b5 * x

        gamma = torch.nn.functional.softplus(self.raw_gamma)
        self.real_gamma = gamma

        current_min_gamma = gamma.min().detach()
        current_max_gamma = gamma.max().detach()
        if self.min_gamma is None or current_min_gamma < self.min_gamma:
            self.min_gamma = current_min_gamma
        if self.max_gamma is None or current_max_gamma > self.max_gamma:
            self.max_gamma = current_max_gamma

        y = y_ * gamma * self.fixed_downscale
        max_last_column = y[:, -1].max().item()
        self.output_min = y[:, -1].min().item()
        self.last_y_ = y_

        y = y.sum(dim=-1)
        y = torch.nn.functional.softplus(y)

        if self.return_max:
            return y, max_last_column
        return y

    def __repr__(self):
        return f"BSplineActivationLayer(in_count={self.in_count}, out_count={self.out_count}, num_splines=5)"

    def plot_individual_activation_functions(self, input_indices, output_indices, x_range=(-5, 5), num_points=200):
        """Plot individual activation functions before summing over input features."""
        device = self.w.device
        x_vals = torch.linspace(x_range[0], x_range[1], num_points, device=device)

        with torch.no_grad():
            w_norm = self.normalize_w()
            w_norm = self.clamp_normalized_w(w_norm)

            b_vals = [self._eval_bspline(w_norm, breaks, coefs) for (breaks, coefs) in self.splines]
            b1, b2, b3, b4, b5 = b_vals

            gamma = self.raw_gamma

            for out_idx in output_indices:
                plt.figure(figsize=(8, 4))
                for in_idx in input_indices:
                    g = gamma[out_idx, in_idx]
                    b1_ = b1[out_idx, in_idx]
                    b2_ = b2[out_idx, in_idx]
                    b3_ = b3[out_idx, in_idx]
                    b4_ = b4[out_idx, in_idx]
                    b5_ = b5[out_idx, in_idx]

                    x_in = x_vals.unsqueeze(1)

                    exp_input = torch.clamp(b3_ * x_in, max=30.0)
                    exp_result = torch.exp(exp_input)
                    base_log = (exp_result - 1).clamp(min=1e-6) ** b4_
                    log_argument = 1 + base_log
                    log_argument = log_argument.clamp(min=1e-6)
                    log1p_term = torch.log(1 + b2_ * torch.log(log_argument).clamp(min=1e-6))
                    y_ = b1_ * log1p_term + b5_ * x_in
                    y_final = y_ * g

                    plt.plot(x_vals.cpu().numpy(), y_final.squeeze().cpu().numpy(),
                             label=f"Input {in_idx} → Output {out_idx}")
                plt.title(f"Activation functions to output node {out_idx}")
                plt.xlabel("Input feature value")
                plt.ylabel("Activation output (before sum)")
                plt.legend()
                plt.show()


class HybridActivationLayer(nn.Module):
    """Hybrid activation combining spline-defined log and polynomial regions."""

    def __init__(self, in_count: int, out_count: int, debug: bool = False, return_max: bool = False):
        super().__init__()
        self.in_count = in_count
        self.out_count = out_count
        self.debug = debug
        self.return_max = return_max

        self.raw_gamma = nn.Parameter(torch.zeros(out_count, in_count))
        self.fixed_downscale = 0.1

        data = loadmat("activation_KAN_fit_BP_Sofija_extend_splineup.mat", simplify_cells=True)

        self.b_splines = [
            (torch.tensor(s["breaks"], dtype=torch.float64),
             torch.tensor(s["coefs"], dtype=torch.float64))
            for s in data["c_coef_spline"]
        ]
        self.p_splines = [
            (torch.tensor(s["breaks"], dtype=torch.float64),
             torch.tensor(s["coefs"], dtype=torch.float64))
            for s in data["p_coef_spline"]
        ]

        self.mu_detuning = torch.tensor(data["mu_detuning"], dtype=torch.float64)
        self.sigma_detuning = torch.tensor(data["sigma_detuning"], dtype=torch.float64)
        self.w = nn.Parameter(5 + (35.5 - 5) * torch.rand(out_count, in_count, dtype=torch.float64))

        self.output_min = None
        self.input_min = None
        self.input_max = None
        self.min_gamma = None
        self.max_gamma = None

    def clamp_w(self):
        return torch.clamp(self.w, min=5.5, max=35.5)

    def normalize_w(self):
        return (self.w - self.mu_detuning) / self.sigma_detuning

    def _eval_bspline(self, w: torch.Tensor, breaks: torch.Tensor, coefs: torch.Tensor) -> torch.Tensor:
        w_clamped = torch.clamp(w, min=breaks[0].item(), max=breaks[-1].item() - 1e-6)
        interval_idx = torch.bucketize(w_clamped, breaks) - 1
        interval_idx = interval_idx.clamp(min=0, max=coefs.shape[0] - 1)
        a = coefs[interval_idx]
        a3, a2, a1, a0 = a.unbind(dim=-1)
        t = w_clamped - breaks[interval_idx]
        return a3 * t**3 + a2 * t**2 + a1 * t + a0

    def _eval_poly4_coeffs(self, w: torch.Tensor, breaks: torch.Tensor, coefs: torch.Tensor) -> torch.Tensor:
        """Return polynomial value for a given w (kept as in original implementation)."""
        w_clamped = torch.clamp(w, min=breaks[0].item(), max=breaks[-1].item() - 1e-6)
        interval_idx = torch.bucketize(w_clamped, breaks) - 1
        interval_idx = interval_idx.clamp(min=0, max=coefs.shape[0] - 1)
        a = coefs[interval_idx]
        a3, a2, a1, a0 = a.unbind(dim=-1)
        t = w_clamped - breaks[interval_idx]
        return a3 * t**3 + a2 * t**2 + a1 * t + a0

    def forward(self, x):
        current_min = x.min().detach()
        current_max = x.max().detach()
        x = torch.relu(x)

        if self.input_min is None or current_min < self.input_min:
            self.input_min = current_min
        if self.input_max is None or current_max > self.input_max:
            self.input_max = current_max

        batch_size = x.shape[0]
        x_exp = x.unsqueeze(1)  # (batch, 1, in_count)

        w = self.clamp_w()
        w_norm = (w - self.mu_detuning) / self.sigma_detuning

        y_ = torch.zeros((batch_size, self.out_count, self.in_count),
                         dtype=torch.float64, device=x.device)

        mask_b = (x_exp < 30)
        mask_p = ~mask_b
        mask_b = mask_b.expand(-1, self.out_count, -1)
        mask_p = mask_p.expand(-1, self.out_count, -1)

        if mask_b.any():
            b_vals = [self._eval_bspline(w_norm, br, cf) for (br, cf) in self.b_splines]
            b1, b2, b3, b4, b5 = b_vals
            exp_input = b3 * x_exp
            exp_result = torch.exp(exp_input)
            base_log = (exp_result - 1) ** b4
            log_argument = 1 + base_log
            log1p_term = torch.log(1 + b2 * torch.log(log_argument))
            y_b = b1 * log1p_term + b5 * x_exp
            y_[mask_b] = y_b[mask_b]

        if mask_p.any():
            p_vals = [self._eval_poly4_coeffs(w_norm, br, cf) for (br, cf) in self.p_splines]
            p1, p2, p3, p4, p5 = p_vals
            y_p = p1 * x_exp**4 + p2 * x_exp**3 + p3 * x_exp**2 + p4 * x_exp + p5
            y_[mask_p] = y_p[mask_p]

        gamma = torch.nn.functional.softplus(self.raw_gamma)
        self.real_gamma = gamma

        current_min_gamma = gamma.min().detach()
        current_max_gamma = gamma.max().detach()
        if self.min_gamma is None or current_min_gamma < self.min_gamma:
            self.min_gamma = current_min_gamma
        if self.max_gamma is None or current_max_gamma > self.max_gamma:
            self.max_gamma = current_max_gamma
        if self.debug:
            self.current_gamma = gamma.detach().cpu()

        y = y_ * gamma * self.fixed_downscale
        max_last_column = y[:, -1].max().item()
        self.output_min = y[:, -1].min().item()

        y = y.sum(dim=-1)

        if self.return_max:
            return y, max_last_column
        return y


class LogActivationLayer(nn.Module):
    """Final log-spline + polynomial activation used in the thesis."""

    def __init__(self, in_count: int, out_count: int, debug: bool = False, return_max: bool = False):
        super().__init__()
        self.in_count = in_count
        self.out_count = out_count
        self.debug = debug
        self.return_max = return_max

        self.raw_gamma = nn.Parameter(torch.zeros(out_count, in_count))
        self.fixed_downscale = 1 / float(out_count)

        data = loadmat("activation_KAN_fit_B_Sofija_extend_logpoly.mat", simplify_cells=True)

        self.b_splines = [
            (torch.tensor(s["breaks"], dtype=torch.float64),
             torch.tensor(s["coefs"], dtype=torch.float64))
            for s in data["c_coef_spline"]
        ]

        self.mu_detuning = torch.tensor(data["mu_detuning"], dtype=torch.float64)
        self.sigma_detuning = torch.tensor(data["sigma_detuning"], dtype=torch.float64)
        self.w = nn.Parameter(5 + (38.0 - 5) * torch.rand(out_count, in_count, dtype=torch.float64))

        self.output_min = None
        self.input_min = None
        self.input_max = None
        self.min_gamma = None
        self.max_gamma = None

    def clamp_w(self):
        return torch.clamp(self.w, min=-5.5, max=37.9)

    def normalize_w(self):
        return (self.w - self.mu_detuning) / self.sigma_detuning

    def _eval_bspline(self, w: torch.Tensor, breaks: torch.Tensor, coefs: torch.Tensor) -> torch.Tensor:
        w_clamped = torch.clamp(w, min=breaks[0].item(), max=breaks[-1].item() - 1e-6)
        interval_idx = torch.bucketize(w_clamped, breaks) - 1
        interval_idx = interval_idx.clamp(min=0, max=coefs.shape[0] - 1)
        a = coefs[interval_idx]
        a3, a2, a1, a0 = a.unbind(dim=-1)
        t = w_clamped - breaks[interval_idx]
        return a3 * t**3 + a2 * t**2 + a1 * t + a0

    def forward(self, x):
        current_min = x.min().detach()
        current_max = x.max().detach()
        x = torch.relu(x)

        if self.input_min is None or current_min < self.input_min:
            self.input_min = current_min
        if self.input_max is None or current_max > self.input_max:
            self.input_max = current_max

        x_exp = x.unsqueeze(1)

        w = self.clamp_w()
        w_norm = (w - self.mu_detuning) / self.sigma_detuning

        b_vals = [self._eval_bspline(w_norm, br, cf) for (br, cf) in self.b_splines]
        b1, b2, b3, b4, b5, b6, b7, b8 = b_vals

        exp_input = b3 * x_exp
        exp_result = torch.exp(exp_input)
        base_log = (exp_result - 1) ** b4
        log_argument = 1 + base_log
        log1p_term = torch.log(1 + b2 * torch.log(log_argument))
        y_ = (
            b1 * log1p_term
            + b5 * x_exp
            + b6 * x_exp**2
            + b7 * x_exp**3
            + b8 * x_exp**4
        )

        gamma = torch.nn.functional.softplus(self.raw_gamma)
        self.real_gamma = gamma

        current_min_gamma = gamma.min().detach()
        current_max_gamma = gamma.max().detach()
        if self.min_gamma is None or current_min_gamma < self.min_gamma:
            self.min_gamma = current_min_gamma
        if self.max_gamma is None or current_max_gamma > self.max_gamma:
            self.max_gamma = current_max_gamma

        y = y_ * gamma * self.fixed_downscale
        max_last_column = y[:, -1].max().item()
        self.output_min = y[:, -1].min().item()

        y = y.sum(dim=-1)

        if self.return_max:
            return y, max_last_column
        return y


class NegativeActivationLayer(nn.Module):
    """Extension to negative detuning region using spline-based activation."""

    def __init__(self, in_count: int, out_count: int, debug: bool = False, return_max: bool = False):
        super().__init__()
        self.in_count = in_count
        self.out_count = out_count
        self.debug = debug
        self.return_max = return_max

        self.raw_gamma = nn.Parameter(torch.zeros(out_count, in_count))
        self.fixed_downscale = 0.01 / float(out_count)
        self.a = torch.tensor(10.0)
        self.A = torch.tensor(38.0)

        data = loadmat("activation_KAN_fit_B_Sofija_extend_negative.mat", simplify_cells=True)

        self.b_splines = [
            (torch.tensor(s["breaks"], dtype=torch.float64),
             torch.tensor(s["coefs"], dtype=torch.float64))
            for s in data["c_coef_spline"]
        ]

        self.mu_detuning = torch.tensor(data["mu_detuning"], dtype=torch.float64)
        self.sigma_detuning = torch.tensor(data["sigma_detuning"], dtype=torch.float64)
        self.w = nn.Parameter(-5 + (0.0 + 5) * torch.rand(out_count, in_count, dtype=torch.float64))

        self.output_min = None
        self.input_min = None
        self.input_max = None
        self.min_gamma = None
        self.max_gamma = None

    def clamp_w(self):
        return torch.clamp(self.w, min=-5.5, max=37.9)

    def normalize_w(self):
        return (self.w - self.mu_detuning) / self.sigma_detuning

    def _eval_bspline(self, w: torch.Tensor, breaks: torch.Tensor, coefs: torch.Tensor) -> torch.Tensor:
        w_clamped = torch.clamp(w, min=breaks[0].item(), max=breaks[-1].item() - 1e-6)
        interval_idx = torch.bucketize(w_clamped, breaks) - 1
        interval_idx = interval_idx.clamp(min=0, max=coefs.shape[0] - 1)
        a = coefs[interval_idx]
        a3, a2, a1, a0 = a.unbind(dim=-1)
        t = w_clamped - breaks[interval_idx]
        return a3*t**3 + a2*t**2 + a1*t + a0

    def forward(self, x):
        current_min = x.min().detach()
        current_max = x.max().detach()
        x = torch.relu(x)

        if self.input_min is None or current_min < self.input_min:
            self.input_min = current_min
        if self.input_max is None or current_max > self.input_max:
            self.input_max = current_max

        x_exp = x.unsqueeze(1)

        w = self.clamp_w()
        w_norm = (w - self.mu_detuning) / self.sigma_detuning

        b_vals = [self._eval_bspline(w_norm, br, cf) for (br, cf) in self.b_splines]
        b1, b2, b3, b4, b5, b6, b7, b8 = b_vals

        exp_input = b3 * x_exp
        exp_result = torch.exp(exp_input)
        base_log = (exp_result - 1) ** b4
        log_argument = 1 + base_log
        log1p_term = torch.log(1 + b2 * torch.log(log_argument))
        f_ = (
            b1 * log1p_term
            + b5 * x_exp
            + b6 * x_exp**2
            + b7 * x_exp**3
            + b8 * x_exp**4
        )
        y_ = (self.A / 2) * (1 - torch.tanh(self.a * w_norm)) + torch.tanh(self.a * w_norm) * f_

        gamma = torch.nn.functional.softplus(self.raw_gamma)
        self.real_gamma = gamma

        current_min_gamma = gamma.min().detach()
        current_max_gamma = gamma.max().detach()
        if self.min_gamma is None or current_min_gamma < self.min_gamma:
            self.min_gamma = current_min_gamma
        if self.max_gamma is None or current_max_gamma > self.max_gamma:
            self.max_gamma = current_max_gamma

        y = y_ * gamma * self.fixed_downscale
        max_last_column = y[:, -1].max().item()
        self.output_min = y[:, -1].min().item()

        y = y.sum(dim=-1)

        if self.return_max:
            return y, max_last_column
        return y
    

class PosNegActivationLayer(nn.Module):
    """Activation with independent positive and negative branches:

       f(det1, det2, gamma1, gamma2, x) = gamma1 * fp(det1, x) + gamma2 * fn(det2, x)

       where:
           fp(det1, x) = spline+poly with detuning det1
           fn(det2, x) = A - spline+poly with detuning det2
    """

    def __init__(self, in_count: int, out_count: int,
                 debug: bool = False, return_max: bool = False):
        super().__init__()
        self.in_count = in_count
        self.out_count = out_count
        self.debug = debug
        self.return_max = return_max

        # Two positive scaling factors (gamma1, gamma2)
        self.raw_gamma_pos = nn.Parameter(torch.zeros(out_count, in_count))
        self.raw_gamma_neg = nn.Parameter(torch.zeros(out_count, in_count))

        # Simple downscale to avoid exploding magnitudes (same idea as before)
        self.fixed_downscale = 0.01 / float(out_count)

        # Load spline data and A from the same .mat file
        data = loadmat("activation_KAN_fit_B_Sofija_extend_positiveplusnegative.mat", simplify_cells=True)

        # List of (breaks, coefs) for the 8 spline-parameterized coefficients b1,...,b8
        self.b_splines = [
            (torch.tensor(s["breaks"], dtype=torch.float64),
             torch.tensor(s["coefs"], dtype=torch.float64))
            for s in data["c_coef_spline"]
        ]

        # Detuning normalization (same as before)
        self.mu_detuning = torch.tensor(data["mu_detuning"], dtype=torch.float64)
        self.sigma_detuning = torch.tensor(data["sigma_detuning"], dtype=torch.float64)

        # Two independent detuning parameters: det1, det2
        # Initialized in a moderate range [0, 30]; you can tighten this if needed.
        self.w_pos = nn.Parameter(30.0 * torch.rand(out_count, in_count, dtype=torch.float64))
        self.w_neg = nn.Parameter(30.0 * torch.rand(out_count, in_count, dtype=torch.float64))

        # Base A from the .mat file (assumed scalar or broadcastable)
        self.A = torch.tensor(data["A"], dtype=torch.float64).squeeze()

        # Trainable multiplier for A so you can increase it if needed
        # A_effective = A_base * softplus(log_A_scale)
        #self.log_A_scale = nn.Parameter(torch.zeros(1, dtype=torch.float64))

        # Debug / tracking
        self.output_min = None
        self.input_min = None
        self.input_max = None
        self.min_gamma_pos = None
        self.max_gamma_pos = None
        self.min_gamma_neg = None
        self.max_gamma_neg = None

    def clamp_w(self, w: torch.Tensor) -> torch.Tensor:
        # Detunings constrained roughly to [0, 30]
        return torch.clamp(w, min=0.0, max=30.0 - 1e-3)

    def _eval_bspline(self, w: torch.Tensor,
                      breaks: torch.Tensor,
                      coefs: torch.Tensor) -> torch.Tensor:
        # Same spline evaluation as in LogActivationLayer
        w_clamped = torch.clamp(w, min=breaks[0].item(), max=breaks[-1].item() - 1e-6)
        interval_idx = torch.bucketize(w_clamped, breaks) - 1
        interval_idx = interval_idx.clamp(min=0, max=coefs.shape[0] - 1)
        a = coefs[interval_idx]      # (..., 4) : cubic coefficients
        a3, a2, a1, a0 = a.unbind(dim=-1)
        t = w_clamped - breaks[interval_idx]
        return a3 * t**3 + a2 * t**2 + a1 * t + a0

    def _fp_core(self,
                 x_exp: torch.Tensor,
                 b1: torch.Tensor, b2: torch.Tensor, b3: torch.Tensor, b4: torch.Tensor,
                 b5: torch.Tensor, b6: torch.Tensor, b7: torch.Tensor, b8: torch.Tensor
                 ) -> torch.Tensor:
        """
        Core spline+polynomial:
        fp(det, x) = b1 * log(1 + b2 * log(1 + (exp(b3 * x) - 1)^b4))
                     + b5 * x + b6 * x^2 + b7 * x^3 + b8 * x^4
        Shapes:
            x_exp: (batch, 1, in)
            b_i:   (out, in)
        Result:
            (batch, out, in)
        """
        exp_input = b3 * x_exp
        exp_result = torch.exp(exp_input)
        base_log = (exp_result - 1) ** b4
        log_argument = 1 + base_log
        log1p_term = torch.log(1 + b2 * torch.log(log_argument))

        y = (
            b1 * log1p_term
            + b5 * x_exp
            + b6 * x_exp**2
            + b7 * x_exp**3
            + b8 * x_exp**4
        )
        return y

    def forward(self, x: torch.Tensor):
        # Track input range
        current_min = x.min().detach()
        current_max = x.max().detach()

        x = torch.relu(x)

        if self.input_min is None or current_min < self.input_min:
            self.input_min = current_min
        if self.input_max is None or current_max > self.input_max:
            self.input_max = current_max

        # (batch, in) -> (batch, 1, in) to broadcast over out dimension
        x_exp = x.unsqueeze(1)

        # --- Positive branch (det1, gamma1) ---

        w_pos = self.clamp_w(self.w_pos)
        w_pos_norm = (w_pos - self.mu_detuning) / self.sigma_detuning

        b_vals_pos = [self._eval_bspline(w_pos_norm, br, cf) for (br, cf) in self.b_splines]
        b1p, b2p, b3p, b4p, b5p, b6p, b7p, b8p = b_vals_pos

        fp_val = self._fp_core(x_exp, b1p, b2p, b3p, b4p, b5p, b6p, b7p, b8p)

        # --- Negative branch (det2, gamma2, A) ---

        w_neg = self.clamp_w(self.w_neg)
        w_neg_norm = (w_neg - self.mu_detuning) / self.sigma_detuning

        b_vals_neg = [self._eval_bspline(w_neg_norm, br, cf) for (br, cf) in self.b_splines]
        b1n, b2n, b3n, b4n, b5n, b6n, b7n, b8n = b_vals_neg

        fn_core = self._fp_core(x_exp, b1n, b2n, b3n, b4n, b5n, b6n, b7n, b8n)

        # Effective A (can be increased during training if helpful)
        # A_effective = self.A_base * torch.nn.functional.softplus(self.log_A_scale)
        fn_val = self.A - fn_core

        # --- Positive scaling coefficients gamma1, gamma2 ---

        gamma_pos = torch.nn.functional.softplus(self.raw_gamma_pos)
        gamma_neg = torch.nn.functional.softplus(self.raw_gamma_neg)

        # Keep for inspection if needed
        self.real_gamma_pos = gamma_pos
        self.real_gamma_neg = gamma_neg

        # Track gamma ranges
        current_min_gamma_pos = gamma_pos.min().detach()
        current_max_gamma_pos = gamma_pos.max().detach()
        current_min_gamma_neg = gamma_neg.min().detach()
        current_max_gamma_neg = gamma_neg.max().detach()

        if self.min_gamma_pos is None or current_min_gamma_pos < self.min_gamma_pos:
            self.min_gamma_pos = current_min_gamma_pos
        if self.max_gamma_pos is None or current_max_gamma_pos > self.max_gamma_pos:
            self.max_gamma_pos = current_max_gamma_pos

        if self.min_gamma_neg is None or current_min_gamma_neg < self.min_gamma_neg:
            self.min_gamma_neg = current_min_gamma_neg
        if self.max_gamma_neg is None or current_max_gamma_neg > self.max_gamma_neg:
            self.max_gamma_neg = current_max_gamma_neg

        # Combine branches:
        # f = gamma1 * fp(det1, x) + gamma2 * fn(det2, x)
        y = (gamma_pos * fp_val + gamma_neg * fn_val) * self.fixed_downscale

        # For debugging: look at last "input column" of the last output channel
        max_last_column = y[:, -1].max().item()
        self.output_min = y[:, -1].min().item()
        self.output_max = max_last_column

        # Sum over input dimension -> (batch, out)
        y = y.sum(dim=-1)

        if self.return_max:
            return y, max_last_column
        return y


class KAN(nn.Module):
    """Core KAN model that stacks activation layers to form an MLP-like architecture."""

    def __init__(
        self,
        in_count: int,
        out_count: int,
        hidden_layer_sizes,
        debug: bool = False,
        input_dropout_prob: float = 0.0,
        hidden_dropout_prob: float = 0.0,
        activation_cls=PosNegActivationLayer,
    ):
        """
        Parameters
        ----------
        in_count : int
            Number of input features.
        out_count : int
            Number of output features.
        hidden_layer_sizes : list[int]
            Sizes of hidden layers.
        activation_cls : nn.Module class
            Activation layer to use between layers (default: LogActivationLayer).
        """
        super().__init__()

        layers = []
        prev_size = in_count

        for i, size in enumerate(hidden_layer_sizes):
            layers.append(activation_cls(prev_size, size, debug=debug, return_max=True))

            if i == 0 and input_dropout_prob > 0:
                layers.append(nn.Dropout(input_dropout_prob))
            elif i > 0 and hidden_dropout_prob > 0:
                layers.append(nn.Dropout(hidden_dropout_prob))
            prev_size = size

        layers.append(activation_cls(prev_size, out_count, debug=debug, return_max=True))
        self.layers = nn.ModuleList(layers)
    '''
    def forward(self, x, return_all: bool = False, track_stats: bool = False):
        outputs = []
        stats = []

        for i, layer in enumerate(self.layers):
            if isinstance(layer, (LogActivationLayer, BSplineActivationLayer, HybridActivationLayer,
                                  NegativeActivationLayer, TanhActivation, PosNegActivationLayer)):
                out = layer(x)
                if isinstance(out, tuple):
                    x = out[0]
                    aux = out[1]
                else:
                    x = out
                    aux = None

                if track_stats and hasattr(layer, "input_min"):
                    stats.append(
                        {
                            "layer": i,
                            "max_input": layer.input_max,
                            "min_input": layer.input_min,
                            "min_output": getattr(layer, "output_min", None),
                            "max_output": getattr(layer, "output_max", None),
                            "max_gamma": layer.max_gamma.item() if hasattr(layer, "max_gamma") and layer.max_gamma is not None else None,
                            "min_gamma": layer.min_gamma.item() if hasattr(layer, "min_gamma") and layer.min_gamma is not None else None,
                            "aux": aux,
                        }
                    )
            else:
                x = layer(x)

            if return_all:
                outputs.append(x.detach().cpu().numpy())

        if track_stats:
            return (x, outputs, stats) if return_all else (x, stats)
        return (x, outputs) if return_all else x
    '''
    def forward(self, x, return_all: bool = False, track_stats: bool = False):
        """
        Generic forward:

        - Any layer is allowed to return either:
            * a tensor y, or
            * a tuple (y, aux)

        - We always unwrap tuples so that the *next* layer
          only ever sees a tensor.

        - If track_stats=True, we record statistics for layers
          that expose input_min / input_max / output_min / output_max
          and gamma-related attributes.
        """
        outputs = []
        stats = []

        for i, layer in enumerate(self.layers):
            # Run the layer
            out = layer(x)

            # Unwrap (y, aux) if needed
            if isinstance(out, tuple):
                x, aux = out
            else:
                x = out
                aux = None

            # Collect stats for layers that track them
            if track_stats and hasattr(layer, "input_min"):
                stats.append(
                    {
                        "layer": i,
                        "max_input": getattr(layer, "input_max", None),
                        "min_input": getattr(layer, "input_min", None),
                        "min_output": getattr(layer, "output_min", None),
                        "max_output": getattr(layer, "output_max", None),
                        # single-gamma layers
                        "max_gamma": layer.max_gamma.item()
                        if hasattr(layer, "max_gamma") and layer.max_gamma is not None
                        else None,
                        "min_gamma": layer.min_gamma.item()
                        if hasattr(layer, "min_gamma") and layer.min_gamma is not None
                        else None,
                        # PosNegActivationLayer dual gammas
                        "max_gamma_pos": layer.max_gamma_pos.item()
                        if hasattr(layer, "max_gamma_pos") and layer.max_gamma_pos is not None
                        else None,
                        "min_gamma_pos": layer.min_gamma_pos.item()
                        if hasattr(layer, "min_gamma_pos") and layer.min_gamma_pos is not None
                        else None,
                        "max_gamma_neg": layer.max_gamma_neg.item()
                        if hasattr(layer, "max_gamma_neg") and layer.max_gamma_neg is not None
                        else None,
                        "min_gamma_neg": layer.min_gamma_neg.item()
                        if hasattr(layer, "min_gamma_neg") and layer.min_gamma_neg is not None
                        else None,
                        "aux": aux,
                    }
                )

            if return_all:
                outputs.append(x.detach().cpu().numpy())

        if track_stats:
            return (x, outputs, stats) if return_all else (x, stats)
        return (x, outputs) if return_all else x

    
    def __repr__(self):
        return f"KAN({self.layers})"

    def get_learnable_params(self):
        return [p for layer in self.layers for p in layer.parameters() if p.requires_grad]

    def export_w_values(self, filename_base: str = "kan_weights"):
        """Export w (detuning) and gamma values from all BSpline-like layers into a .mat file."""

        w_values = []
        gamma_values = []
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "w") and hasattr(layer, "real_gamma"):
                w_np = layer.w.detach().cpu().numpy()
                gamma_np = layer.real_gamma.detach().cpu().numpy()
                w_values.append((i, w_np))
                gamma_values.append((i, gamma_np))

        export_dict = {}
        for idx, (layer_idx, w) in enumerate(w_values):
            export_dict[f"w_layer_{layer_idx}"] = w
        for idx, (layer_idx, g) in enumerate(gamma_values):
            export_dict[f"gamma_layer_{layer_idx}"] = g

        savemat(f"{filename_base}.mat", export_dict)
        print(f"Exported {len(w_values)} layers' weights and gamma values to {filename_base}.mat.")

    def export_single_sample_trace(
        self,
        sample_input: torch.Tensor,
        filename: str = "kan_single_sample_trace.mat",
    ):
        """
        Run a single input through the model and export:
          - input to each layer
          - output before summation (y_ * gamma * scale)
          - w and real_gamma for each activation layer
        """
        self.eval()
        export_dict = {}
        x = sample_input.clone().detach()

        for i, layer in enumerate(self.layers):
            # Store input to this layer
            export_dict[f"input_before_activation_layer_{i}"] = (
                x.squeeze(0).detach().cpu().numpy()
            )

            if isinstance(layer, TanhActivation):
                x, _ = layer(x)
                export_dict[f"output_before_sum_layer_{i}"] = (
                    layer.last_y_pre_sum.squeeze(0).detach().cpu().numpy()
                )

            elif isinstance(layer, HybridActivationLayer):
                x_relu = torch.relu(x)
                w = layer.clamp_w()
                w_norm = (w - layer.mu_detuning) / layer.sigma_detuning

                x_exp = x_relu.unsqueeze(1)
                mask_b = (x_exp < 30)
                mask_p = ~mask_b
                mask_b = mask_b.expand(-1, layer.out_count, -1)
                mask_p = mask_p.expand(-1, layer.out_count, -1)

                y_ = torch.zeros(
                    (x_exp.shape[0], layer.out_count, layer.in_count),
                    dtype=torch.float64,
                    device=x.device,
                )

                if mask_b.any():
                    b_vals = [
                        layer._eval_bspline(w_norm, br, cf)
                        for (br, cf) in layer.b_splines
                    ]
                    b1, b2, b3, b4, b5 = b_vals
                    exp_input = b3 * x_exp
                    exp_result = torch.exp(exp_input)
                    base_log = (exp_result - 1) ** b4
                    log_arg = 1 + base_log
                    log1p_term = torch.log(1 + b2 * torch.log(log_arg))
                    y_b = b1 * log1p_term + b5 * x_exp
                    y_[mask_b] = y_b[mask_b]

                if mask_p.any():
                    p_vals = [
                        layer._eval_poly4_coeffs(w_norm, br, cf)
                        for (br, cf) in layer.p_splines
                    ]
                    p1, p2, p3, p4, p5 = p_vals
                    y_p = (
                        p1 * x_exp**4
                        + p2 * x_exp**3
                        + p3 * x_exp**2
                        + p4 * x_exp
                        + p5
                    )
                    y_[mask_p] = y_p[mask_p]

                y_pre_sum = y_ * layer.real_gamma * layer.fixed_downscale
                export_dict[f"output_before_sum_layer_{i}"] = (
                    y_pre_sum.squeeze(0).detach().cpu().numpy()
                )
                export_dict[f"w_layer_{i}"] = layer.w.detach().cpu().numpy()
                export_dict[f"gamma_layer_{i}"] = layer.real_gamma.detach().cpu().numpy()

                x = y_pre_sum.sum(dim=-1)

            elif isinstance(layer, BSplineActivationLayer):
                x_relu = torch.relu(x)
                w_norm = (layer.clamp_w() - layer.mu_detuning) / layer.sigma_detuning

                b_vals = [
                    layer._eval_bspline(w_norm, b, c) for (b, c) in layer.splines
                ]
                b1, b2, b3, b4, b5 = b_vals

                x_expanded = x_relu.unsqueeze(1)
                exp_input = b3 * x_expanded
                exp_result = torch.exp(exp_input)
                base_log = (exp_result - 1) ** b4
                log_arg = 1 + base_log
                log1p_term = torch.log(1 + b2 * torch.log(log_arg))
                y_ = b1 * log1p_term + b5 * x_expanded

                y_pre_sum = y_ * layer.real_gamma * layer.fixed_downscale
                export_dict[f"output_before_sum_layer_{i}"] = (
                    y_pre_sum.squeeze(0).detach().cpu().numpy()
                )
                export_dict[f"w_layer_{i}"] = layer.w.detach().cpu().numpy()
                export_dict[f"gamma_layer_{i}"] = layer.real_gamma.detach().cpu().numpy()

                x = torch.nn.functional.softplus(y_pre_sum.sum(dim=-1))

            elif isinstance(layer, LogActivationLayer):
                x_relu = torch.relu(x)
                w_norm = (layer.clamp_w() - layer.mu_detuning) / layer.sigma_detuning

                b_vals = [
                    layer._eval_bspline(w_norm, b, c) for (b, c) in layer.b_splines
                ]
                b1, b2, b3, b4, b5, b6, b7, b8 = b_vals

                x_expanded = x_relu.unsqueeze(1)
                exp_input = b3 * x_expanded
                exp_result = torch.exp(exp_input)
                base_log = (exp_result - 1) ** b4
                log_arg = 1 + base_log
                log1p_term = torch.log(1 + b2 * torch.log(log_arg))
                y_ = (
                    b1 * log1p_term
                    + b5 * x_expanded
                    + b6 * x_expanded**2
                    + b7 * x_expanded**3
                    + b8 * x_expanded**4
                )

                y_pre_sum = y_ * layer.real_gamma * layer.fixed_downscale
                export_dict[f"output_before_sum_layer_{i}"] = (
                    y_pre_sum.squeeze(0).detach().cpu().numpy()
                )
                export_dict[f"w_layer_{i}"] = layer.w.detach().cpu().numpy()
                export_dict[f"gamma_layer_{i}"] = layer.real_gamma.detach().cpu().numpy()

                x = y_pre_sum.sum(dim=-1)

            elif isinstance(layer, NegativeActivationLayer):
                x_relu = torch.relu(x)
                w_norm = (layer.clamp_w() - layer.mu_detuning) / layer.sigma_detuning

                b_vals = [
                    layer._eval_bspline(w_norm, b, c) for (b, c) in layer.b_splines
                ]
                b1, b2, b3, b4, b5, b6, b7, b8 = b_vals

                x_expanded = x_relu.unsqueeze(1)
                exp_input = b3 * x_expanded
                exp_result = torch.exp(exp_input)
                base_log = (exp_result - 1) ** b4
                log_arg = 1 + base_log
                log1p_term = torch.log(1 + b2 * torch.log(log_arg))
                f_ = (
                    b1 * log1p_term
                    + b5 * x_expanded
                    + b6 * x_expanded**2
                    + b7 * x_expanded**3
                    + b8 * x_expanded**4
                )
                y_ = (layer.A / 2) * (1 - torch.tanh(layer.a * w_norm)) + torch.tanh(
                    layer.a * w_norm
                ) * f_

                y_pre_sum = y_ * layer.real_gamma * layer.fixed_downscale
                export_dict[f"output_before_sum_layer_{i}"] = (
                    y_pre_sum.squeeze(0).detach().cpu().numpy()
                )
                export_dict[f"w_layer_{i}"] = layer.w.detach().cpu().numpy()
                export_dict[f"gamma_layer_{i}"] = layer.real_gamma.detach().cpu().numpy()
                export_dict[f"a_layer_{i}"] = layer.a.detach().cpu().numpy()
                export_dict[f"A_layer_{i}"] = layer.A.detach().cpu().numpy()

                x = y_pre_sum.sum(dim=-1)

            else:
                # Non-activation layers (e.g., Dropout)
                x = layer(x)

        print("Export keys:", export_dict.keys())
        savemat(filename, export_dict)
        print(f"Exported trace to {filename}")
