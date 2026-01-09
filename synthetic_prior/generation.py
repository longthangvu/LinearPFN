import torch

def generate_series(component_params, series_len=12000, num_samples=1, equal_spacing=True, device="cpu"):
    n_units = torch.ceil(series_len / component_params.resolution).to(device)     # [num_samples]
    n_units = n_units[:, None]                          # [num_samples, 1]
    if equal_spacing:
        # Evenly spaced grid in [0, 1]
        base = torch.linspace(0, 1, series_len, device=device)     # [T]
        time_points = base[None, :] * n_units # [num_samples, T]
    else:
        # Random positions in [0, n_units]
        time_points = torch.rand(num_samples, series_len, device=device) * n_units    # [num_samples, T]
        time_points, _ = torch.sort(time_points, dim=1) # Sort along time dimension
    # 1)
    trend_comps = generate_trend(trend_lin_scaler=component_params.trend_lin,offset_lin=component_params.offset_lin,
                                 trend_exp_scaler=component_params.trend_exp,offset_exp=component_params.offset_exp,
                                 p_strong_exp=component_params.p_strong_exp,
                                 time_points=time_points,device=device)
    total_trend, lin_trend, exp_trend = trend_comps
    # 2)
    seasonal_comps = generate_seasonality(annual_param=component_params.annual_param,
                                          monthly_param=component_params.monthly_param,
                                          weekly_param=component_params.weekly_param,
                                          n_harmonics=component_params.harmonics,
                                          time_points=time_points,device=device)
    total_seasonality, annual_seasonality, monthly_seasonality, weekly_seasonality = seasonal_comps
    
    amp = component_params.amplitude[:, None]       # [B, 1]
    p_add = component_params.p_add[:, None]         # [B, 1]
    additive = component_params.trend_damp[:, None] * total_trend + total_seasonality
    multiplicative = total_trend * total_seasonality
    # print(p_add)
    combined = torch.where(p_add, additive, multiplicative)
    noiseless_series = amp * combined
    
    # 3)
    noises = 1 + generate_noise(noise_k=component_params.noise_k,
                                noise_scale=component_params.noise_scale,
                                series_len=series_len, device=device)     # [num_samples, T]
    noises = torch.where(component_params.p_noise_trend_scaling[:, None], noises * total_trend, noises)   # scale noise
    
    final_series = noiseless_series * noises
    return final_series, noiseless_series

def generate_trend(trend_lin_scaler: torch.Tensor | None, trend_exp_scaler: torch.Tensor | None,
                   offset_lin: torch.Tensor, offset_exp: torch.Tensor, p_strong_exp: torch.Tensor | None,
                   time_points: torch.Tensor, device: str = "cpu", min_exp_scaler: float = 1e-5):
    B, T = time_points.shape
    # Distance from origin (t - t0)
    origin = time_points[:, :1]                       # [B, 1]
    dt = time_points - origin                         # [B, T]

    values = torch.ones_like(time_points)
    linear_trend, exp_trend = None, None
    
    if trend_lin_scaler is not None:
        slope = trend_lin_scaler.to(device)[:, None]  # [B, 1]
        linear_trend = _shift_axis(dt, offset_lin) * slope
        values = values + linear_trend
    if trend_exp_scaler is not None:
        base = trend_exp_scaler.to(device).clamp_min(min_exp_scaler)[:, None]   # [B, 1]
        boost = torch.where(p_strong_exp[:, None], 1000.0, 1.0)
        # Normalize by span to keep exponent stable
        span = (time_points[:, -1] - time_points[:, 0]).clamp_min(1e-8)[:, None]  # [B, 1]
        exponent = boost * _shift_axis(dt, offset_exp) / span                              # [B, T]
        exp_trend = torch.pow(base, exponent)
        values = values * exp_trend

    return values, linear_trend, exp_trend

def generate_seasonality(annual_param,monthly_param,weekly_param,
                         n_harmonics,time_points,cap=0.95,device="cpu"):
    B, T = time_points.shape
    total = torch.ones(B, T, device=device)

    annual, monthly, weekly = None, None, None

    if annual_param is not None:
        annual = _get_season_comp(annual_param, time_points, 365.25, int(n_harmonics[0]))
        total = total * annual

    if monthly_param is not None:
        monthly = _get_season_comp(monthly_param, time_points, 30.417, int(n_harmonics[1]))
        total = total * monthly

    if weekly_param is not None:
        weekly = _get_season_comp(weekly_param, time_points, 7.0, int(n_harmonics[2]))
        total = total * weekly

    return total, annual, monthly, weekly

def generate_noise(noise_k: torch.Tensor, noise_scale: torch.Tensor,
                             num_samples=1,series_len=12000, device="cpu"):
    noise_k = noise_k.to(device)[:, None]          # [B, 1]
    scale   = noise_scale.to(device)[:, None]      # [B, 1]

    U = torch.rand(num_samples, series_len, device=device).clamp_min(1e-8)
    noise = scale * (-torch.log1p(-U)) ** (1.0 / noise_k)

    return noise

def _get_freq_component(phase: torch.Tensor, n_harmonics: int, device="cpu"):
    """
    Fourier mixture basis evaluated at phase (cycles).
    Returns: [B, T]
    """
    B, T = phase.shape
    harmonics = torch.arange(1, n_harmonics + 1, device=device)[None, :]   # [1, H]

    # Random coefficients per batch
    std = 1.0 / harmonics
    sin_coef = torch.randn(B, n_harmonics, device=device) * std
    cos_coef = torch.randn(B, n_harmonics, device=device) * std

    # Normalize per sample
    coef_norm = torch.sqrt(
        (sin_coef**2).sum(dim=1, keepdim=True) +
        (cos_coef**2).sum(dim=1, keepdim=True)
    ).clamp_min(1e-8)

    sin_coef /= coef_norm
    cos_coef /= coef_norm

    # Evaluate Fourier series
    angle = 2 * torch.pi * harmonics[:, :, None] * phase[:, None, :]  # [B, H, T]

    sin = (sin_coef[:, :, None] * torch.sin(angle)).sum(dim=1)
    cos = (cos_coef[:, :, None] * torch.cos(angle)).sum(dim=1)

    return (sin + cos) / (2.0 * n_harmonics) ** 0.5

def _bounded_part(phi, amp, cap=0.4, amp_scale=3.0):
    # amp_scale larger => less saturation => more small amplitudes
    phi_b = torch.tanh(phi)                         # [-1,1]
    a = torch.tanh(amp / amp_scale) * cap           # [-cap,cap]
    return 1.0 + a[:, None] * phi_b                 # positive if cap<1


def _get_season_comp(comp_param, time_points, num_days, n_harmonic):
    phase = time_points / num_days
    phi = _get_freq_component(phase, n_harmonic)
    return _bounded_part(phi, comp_param)

def _shift_axis(distance_to_origin,scaler):
    if scaler is None:
        return distance_to_origin
    # Total span per sample: [B, 1]
    span = distance_to_origin[:, -1:]         
    # Offset: [B, 1] (broadcasts over T automatically)
    offset = scaler[:, None] * span
    return distance_to_origin - offset