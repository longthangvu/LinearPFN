import typing as T
import torch
import numpy as np
from synthetic_prior.misc import dotdict

def make_multiple_series(
    n_context: int,
    sequence_length: int,
    num_features: int,
    device: str,
    component_params: dotdict,
    return_components: bool = False,  # For debugging and visualization purposes
    scale_noise: bool = True,
    equal_spacing: bool = True,
):
    if not equal_spacing:
        x = (
            (
                torch.rand(n_context, sequence_length, num_features, device=device)
                * component_params.n_units.unsqueeze(-1).unsqueeze(-1)
            )
            .squeeze()
            .to(device)
        )

        x, _ = torch.sort(x, dim=1)
    else:
        x = torch.linspace(0, 1, sequence_length, device=device).unsqueeze(0).repeat(
            n_context, 1
        ).unsqueeze(-1) * component_params.n_units.unsqueeze(-1).unsqueeze(-1)
        x = x.squeeze().to(device)

    trend_comp_total, trend_comp_linear, trend_comp_exponential = (
        generate_trend_component(
            trend_linear_scaler=component_params.trend_lin,
            trend_exp_scaler=component_params.trend_exp,
            offset_linear=component_params.offset_lin,
            offset_exp=component_params.offset_exp,
            x=x,
        )
    )

    seasonal_components = generate_seasonal_component(
        annual_param=component_params.annual_param,
        monthly_param=component_params.monthly_param,
        weekly_param=component_params.weekly_param,
        x=x,
        n_units=component_params.n_units,
        n_harmonics=component_params.harmonics,
        device=device,
    )

    total_seasonality, annual_seasonality, monthly_seasonality, weekly_seasonality = (
        seasonal_components[:, :, 0],
        seasonal_components[:, :, 1],
        seasonal_components[:, :, 2],
        seasonal_components[:, :, 3],
    )

    if component_params.p_add:

        noisless_values= component_params.amplitude[:, None] * (component_params.trend_damp*trend_comp_total + total_seasonality)
    else:
        noisless_values= component_params.amplitude[:, None] * (trend_comp_total * total_seasonality)
    # print('p_add', component_params.p_add)
    noise_mean = torch.ones_like(component_params.noise_k)

    weibull_noise_term = generate_noise_component(
        k=component_params.noise_k,
        noise_mean=noise_mean,
        shape=(x.shape[0], x.shape[1]),
        device=device,
    )

    noise = 1 + component_params.noise_scale.unsqueeze(-1) * (
        weibull_noise_term - noise_mean.unsqueeze(-1)
    )

    if scale_noise:
        noise = noise * trend_comp_total

    if return_components:
        return (
            x,
            trend_comp_total,
            trend_comp_linear,
            trend_comp_exponential,
            total_seasonality,
            annual_seasonality,
            monthly_seasonality,
            weekly_seasonality,
            noise,
            noisless_values,
            component_params,
        )
    return x, noisless_values, noise


def shift_axis(distance_to_origin, scaler):
    if scaler is None:
        return distance_to_origin
    scaled_offset = (
        torch.mul(scaler, distance_to_origin[:, -1])
        .unsqueeze(-1)
        .expand(-1, distance_to_origin.shape[-1])
    )
    return torch.sub(distance_to_origin, scaled_offset)

def generate_trend_component(
    trend_linear_scaler: torch.Tensor,
    trend_exp_scaler: torch.Tensor,
    offset_linear: torch.Tensor,
    offset_exp: torch.Tensor,
    x: torch.Tensor,
    min_exp_scaler: float = 0.00001,
):
    """
    Method to generate trend component of the time series
    Args:
    trend_linear_scaler: Linear scaler for the trend
    trend_exp_scaler: Exponential scaler for the trend
    offset_linear: Offset for the linear trend
    offset_exp: Offset for the exponential trend
    x: Input tensor
    min_exp_scaler: Minimum value for the exponential scaler
    return: Tuple of values, linear trend and exponential trend
    """
    values = torch.ones_like(x)
    origin = x[:, 0].unsqueeze(-1).expand(-1, x.shape[-1])

    distance_to_origin = torch.sub(x, origin)
    if trend_linear_scaler is not None:
        trend_linear_scaler = trend_linear_scaler.unsqueeze(-1).expand(-1, x.shape[-1])
        linear_trend = torch.mul(
            shift_axis(distance_to_origin, offset_linear), trend_linear_scaler
        )
        values = torch.add(values, linear_trend)

    if trend_exp_scaler is not None:
        trend_exp_scaler = (
            trend_exp_scaler.clip(min=min_exp_scaler)
            .unsqueeze(-1)
            .expand(-1, x.shape[-1])
        )
        # exp_trend = torch.pow(
        #     trend_exp_scaler, shift_axis(distance_to_origin, offset_exp)
        # )
        # scale time to [0,1] per series to de-couple growth from n_units/sequence length
        span = (x[:, -1] - x[:, 0]).unsqueeze(-1).expand_as(x).clamp_min(1e-8)
        exp_trend = torch.pow(
            trend_exp_scaler, shift_axis(distance_to_origin, offset_exp) / span
        )

        values = torch.mul(values, exp_trend)

        return values, linear_trend, exp_trend

    return values, linear_trend, None


def get_freq_component(
    frequency_feature: torch.Tensor,
    n_harmonics: torch.Tensor,
    cycle: T.Union[int, float],
    device: str = "cpu",
):
    """
    Method to get systematic movement of values across time
    """

    harmonics = (
        torch.arange(1, n_harmonics.item() + 1)
        .unsqueeze(0)
        .expand(frequency_feature.shape[0], -1)
        .to(device)
    )
    sin_coef = torch.normal(mean=0, std=1 / harmonics)
    cos_coef = torch.normal(mean=0, std=1 / harmonics)

    # normalize the coefficients such that their sum of squares is 1
    coef_sq_sum = torch.sqrt(torch.sum(sin_coef**2) + torch.sum(cos_coef**2))
    sin_coef /= coef_sq_sum
    cos_coef /= coef_sq_sum

    # construct the result for systematic movement which
    # comprises of patterns of varying frequency
    freq_pattern = torch.div(frequency_feature, cycle)
    sin = (
        sin_coef.unsqueeze(-1)
        * torch.sin(2 * torch.pi * harmonics.unsqueeze(-1) * freq_pattern.unsqueeze(1))
    ).sum(1)
    cos = (
        cos_coef.unsqueeze(-1)
        * torch.cos(2 * torch.pi * harmonics.unsqueeze(-1) * freq_pattern.unsqueeze(1))
    ).sum(1)

    return torch.add(sin, cos)


def binning_function(
    x: torch.Tensor, bins: T.Union[int, float], cycle: float, n_units: torch.Tensor
):
    out = (x / cycle).floor() + 1
    mask = out > bins
    out[mask] = (out[mask] % bins) + 1
    return out


def generate_seasonal_component(
    annual_param: torch.Tensor,
    monthly_param: torch.Tensor,
    weekly_param: torch.Tensor,
    x: torch.Tensor,
    n_units: torch.Tensor,
    n_harmonics: torch.Tensor,
    device: str = "cpu",
):
    # write docstring for this function
    """
    Method to generate seasonal component of the time series
    Args:
    annual_param: Annual parameter
    monthly_param: Monthly parameter
    weekly_param: Weekly parameter
    x: Input tensor
    n_units: Number of units
    n_harmonics: Number of harmonics
    device: Device to run the code
    return: Seasonal component
    """

    seasonal = torch.ones(x.shape[0], x.shape[1], 4).to(device)

    if annual_param is not None:
        annual_component = 1 + annual_param.unsqueeze(-1) * get_freq_component(
            binning_function(x, 12, 30.417, n_units), n_harmonics[0], 12, device
        )
        seasonal[:, :, 1] = annual_component
        seasonal[:, :, 0] = torch.mul(seasonal[:, :, 0], annual_component)

    if monthly_param is not None:
        monthly_component = 1 + monthly_param.unsqueeze(-1) * get_freq_component(
            binning_function(x, 30.417, 1, n_units), n_harmonics[1], 30.417, device
        )
        seasonal[:, :, 2] = monthly_component
        seasonal[:, :, 0] = torch.mul(seasonal[:, :, 0], monthly_component)

    if weekly_param is not None:
        weekly_component = 1 + weekly_param.unsqueeze(-1) * get_freq_component(
            binning_function(x, 7, 1, n_units), n_harmonics[2], 7, device
        )
        seasonal[:, :, 3] = weekly_component
        seasonal[:, :, 0] = torch.mul(seasonal[:, :, 0], weekly_component)

    # seasonal dimensions = total_seasonality, annual, monthly, weekly
    return seasonal


def generate_noise_component(
    k: torch.Tensor,
    noise_mean: torch.Tensor,
    shape: T.Tuple[int, int],
    device: str = "cpu",
):
    """
    Method to generate noise component of the time series
    Args:
    k: Shape parameter for the weibull distribution
    noise_mean: Mean of the noise
    shape: Shape of the noise
    device: Device to run the code
    return: Noise component
    """
    lambda_ = noise_mean / (np.log(2) ** (1 / k))
    return torch.from_numpy(np.random.weibull(k.unsqueeze(-1).cpu(), size=shape)).to(
        device
    ) * lambda_.unsqueeze(-1)
# def generate_series(component_params, series_len=12000, num_samples=1, equal_spacing=True, device="cpu"):
#     n_units = torch.ceil(series_len / component_params.resolution).to(device)     # [num_samples]
#     n_units = n_units[:, None]                          # [num_samples, 1]
#     if equal_spacing:
#         # Evenly spaced grid in [0, 1]
#         base = torch.linspace(0, 1, series_len, device=device)     # [T]
#         time_points = base[None, :] * n_units # [num_samples, T]
#     else:
#         # Random positions in [0, n_units]
#         time_points = torch.rand(num_samples, series_len, device=device) * n_units    # [num_samples, T]
#         time_points, _ = torch.sort(time_points, dim=1) # Sort along time dimension
#     # 1)
#     trend_comps = generate_trend(trend_lin_scaler=component_params.trend_lin,offset_lin=component_params.offset_lin,
#                                  trend_exp_scaler=component_params.trend_exp,offset_exp=component_params.offset_exp,
#                                  p_strong_exp=component_params.p_strong_exp,
#                                  time_points=time_points,device=device)
#     total_trend, lin_trend, exp_trend = trend_comps
#     # 2)
#     seasonal_comps = generate_seasonality(annual_param=component_params.annual_param,
#                                           monthly_param=component_params.monthly_param,
#                                           weekly_param=component_params.weekly_param,
#                                           n_harmonics=component_params.harmonics,
#                                           time_points=time_points,device=device)
#     total_seasonality, annual_seasonality, monthly_seasonality, weekly_seasonality = seasonal_comps
    
#     amp = component_params.amplitude[:, None]       # [B, 1]
#     p_add = component_params.p_add[:, None]         # [B, 1]
#     additive = component_params.trend_damp[:, None] * total_trend + total_seasonality
#     multiplicative = total_trend * total_seasonality
#     # print(p_add)
#     combined = torch.where(p_add, additive, multiplicative)
#     noiseless_series = amp * combined
    
#     # 3)
#     noises = 1 + generate_noise(noise_k=component_params.noise_k,
#                                 noise_scale=component_params.noise_scale,
#                                 series_len=series_len, device=device)     # [num_samples, T]
#     noises = torch.where(component_params.p_noise_trend_scaling[:, None], noises * total_trend, noises)   # scale noise
    
#     final_series = noiseless_series * noises
#     return final_series, noiseless_series

# def generate_trend(trend_lin_scaler: torch.Tensor | None, trend_exp_scaler: torch.Tensor | None,
#                    offset_lin: torch.Tensor, offset_exp: torch.Tensor, p_strong_exp: torch.Tensor | None,
#                    time_points: torch.Tensor, device: str = "cpu", min_exp_scaler: float = 1e-5):
#     B, T = time_points.shape
#     # Distance from origin (t - t0)
#     origin = time_points[:, :1]                       # [B, 1]
#     dt = time_points - origin                         # [B, T]

#     values = torch.ones_like(time_points)
#     linear_trend, exp_trend = None, None
    
#     if trend_lin_scaler is not None:
#         slope = trend_lin_scaler.to(device)[:, None]  # [B, 1]
#         linear_trend = _shift_axis(dt, offset_lin) * slope
#         values = values + linear_trend
#     if trend_exp_scaler is not None:
#         base = trend_exp_scaler.to(device).clamp_min(min_exp_scaler)[:, None]   # [B, 1]
#         boost = torch.where(p_strong_exp[:, None], 1000.0, 1.0)
#         # Normalize by span to keep exponent stable
#         span = (time_points[:, -1] - time_points[:, 0]).clamp_min(1e-8)[:, None]  # [B, 1]
#         exponent = boost * _shift_axis(dt, offset_exp) / span                              # [B, T]
#         exp_trend = torch.pow(base, exponent)
#         values = values * exp_trend

#     return values, linear_trend, exp_trend

# def generate_seasonality(annual_param,monthly_param,weekly_param,
#                          n_harmonics,time_points,cap=0.95,device="cpu"):
#     B, T = time_points.shape
#     total = torch.ones(B, T, device=device)

#     annual, monthly, weekly = None, None, None

#     if annual_param is not None:
#         annual = _get_season_comp(annual_param, time_points, 365.25, int(n_harmonics[0]))
#         total = total * annual

#     if monthly_param is not None:
#         monthly = _get_season_comp(monthly_param, time_points, 30.417, int(n_harmonics[1]))
#         total = total * monthly

#     if weekly_param is not None:
#         weekly = _get_season_comp(weekly_param, time_points, 7.0, int(n_harmonics[2]))
#         total = total * weekly

#     return total, annual, monthly, weekly

# def generate_noise(noise_k: torch.Tensor, noise_scale: torch.Tensor,
#                              num_samples=1,series_len=12000, device="cpu"):
#     noise_k = noise_k.to(device)[:, None]          # [B, 1]
#     scale   = noise_scale.to(device)[:, None]      # [B, 1]

#     U = torch.rand(num_samples, series_len, device=device).clamp_min(1e-8)
#     noise = scale * (-torch.log1p(-U)) ** (1.0 / noise_k)

#     return noise

# def _get_freq_component(phase: torch.Tensor, n_harmonics: int, device="cpu"):
#     """
#     Fourier mixture basis evaluated at phase (cycles).
#     Returns: [B, T]
#     """
#     B, T = phase.shape
#     harmonics = torch.arange(1, n_harmonics + 1, device=device)[None, :]   # [1, H]

#     # Random coefficients per batch
#     std = 1.0 / harmonics
#     sin_coef = torch.randn(B, n_harmonics, device=device) * std
#     cos_coef = torch.randn(B, n_harmonics, device=device) * std

#     # Normalize per sample
#     coef_norm = torch.sqrt(
#         (sin_coef**2).sum(dim=1, keepdim=True) +
#         (cos_coef**2).sum(dim=1, keepdim=True)
#     ).clamp_min(1e-8)

#     sin_coef /= coef_norm
#     cos_coef /= coef_norm

#     # Evaluate Fourier series
#     angle = 2 * torch.pi * harmonics[:, :, None] * phase[:, None, :]  # [B, H, T]

#     sin = (sin_coef[:, :, None] * torch.sin(angle)).sum(dim=1)
#     cos = (cos_coef[:, :, None] * torch.cos(angle)).sum(dim=1)

#     return (sin + cos) / (2.0 * n_harmonics) ** 0.5

# def _bounded_part(phi, amp, cap=0.4, amp_scale=3.0):
#     # amp_scale larger => less saturation => more small amplitudes
#     phi_b = torch.tanh(phi)                         # [-1,1]
#     a = torch.tanh(amp / amp_scale) * cap           # [-cap,cap]
#     return 1.0 + a[:, None] * phi_b                 # positive if cap<1


# def _get_season_comp(comp_param, time_points, num_days, n_harmonic):
#     phase = time_points / num_days
#     phi = _get_freq_component(phase, n_harmonic)
#     return _bounded_part(phi, comp_param)

# def _shift_axis(distance_to_origin,scaler):
#     if scaler is None:
#         return distance_to_origin
#     # Total span per sample: [B, 1]
#     span = distance_to_origin[:, -1:]         
#     # Offset: [B, 1] (broadcasts over T automatically)
#     offset = scaler[:, None] * span
#     return distance_to_origin - offset