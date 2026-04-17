from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Patch

from synthetic_prior.generation import generate_series
from synthetic_prior.misc import dotdict
from utils.build_model import build_model_from_hf


@dataclass(frozen=True)
class SplitConfig:
    series_len: int = 1000
    lookback: int = 96
    horizon: int = 96

    @property
    def context_end(self) -> int:
        return self.series_len - (self.lookback + self.horizon)

    @property
    def query_start(self) -> int:
        return self.context_end

    @property
    def query_end(self) -> int:
        return self.query_start + self.lookback

    @property
    def target_start(self) -> int:
        return self.query_end

    @property
    def target_end(self) -> int:
        return self.series_len

    @property
    def query_endpoint(self) -> int:
        return self.query_end - 1

    def as_dict(self) -> dict[str, tuple[int, int] | int]:
        return {
            "series_len": self.series_len,
            "context": (0, self.context_end),
            "query": (self.query_start, self.query_end),
            "target": (self.target_start, self.target_end),
        }


def set_qualitative_plot_style() -> None:
    plt.style.use("seaborn-v0_8-white")
    plt.rcParams["axes.grid"] = False


def make_component_params(
    annual_param=0.0,
    monthly_param=0.0,
    weekly_param=0.0,
    harmonics=(1, 1, 1),
    trend_lin=0.0,
    trend_exp=1.0,
    offset_lin=0.0,
    offset_exp=0.0,
    noise_k=2.5,
    noise_scale=0.02,
    amplitude=1.0,
    additive=True,
    trend_damp=0.25,
    trend_scaled_noise=False,
    strong_exp=False,
    resolution=8.0,
):
    def scalar(value, dtype=torch.float32):
        return torch.tensor([value], dtype=dtype)

    return dotdict(
        {
            "annual_param": scalar(annual_param),
            "monthly_param": scalar(monthly_param),
            "weekly_param": scalar(weekly_param),
            "harmonics": torch.tensor(harmonics, dtype=torch.long),
            "trend_lin": scalar(trend_lin),
            "trend_exp": scalar(trend_exp),
            "offset_lin": scalar(offset_lin),
            "offset_exp": scalar(offset_exp),
            "noise_k": scalar(noise_k),
            "noise_scale": scalar(noise_scale),
            "amplitude": scalar(amplitude),
            "p_add": torch.tensor([additive], dtype=torch.bool),
            "trend_damp": scalar(trend_damp),
            "p_noise_trend_scaling": torch.tensor([trend_scaled_noise], dtype=torch.bool),
            "p_strong_exp": torch.tensor([strong_exp], dtype=torch.bool),
            "resolution": scalar(resolution),
        }
    )


def default_series_specs() -> list[dict]:
    return [
        {
            "name": "1. Easy trend + seasonality",
            "notes": "1. Mild linear trend, single seasonal component, very low noise.",
            "params": make_component_params(
                weekly_param=1.4,
                harmonics=(1, 1, 1),
                trend_lin=0.003,
                noise_scale=0.02,
                resolution=8.0,
                additive=True,
            ),
        },
        {
            "name": "2. Stronger seasonality",
            "notes": "2. Same basic pattern, richer seasonal structure.",
            "params": make_component_params(
                weekly_param=2.2,
                monthly_param=1.2,
                harmonics=(1, 1, 2),
                trend_lin=0.003,
                noise_scale=0.03,
                resolution=8.0,
                additive=True,
            ),
        },
        {
            "name": "3. More harmonics",
            "notes": "3. More complex seasonality and increased noise.",
            "params": make_component_params(
                weekly_param=2.0,
                monthly_param=1.5,
                harmonics=(1, 3, 5),
                trend_lin=0.003,
                noise_scale=0.04,
                resolution=8.0,
                additive=True,
            ),
        },
        {
            "name": "4. Larger trend scale",
            "notes": "4. Trend dominates more of the trajectory.",
            "params": make_component_params(
                weekly_param=2.0,
                monthly_param=1.4,
                harmonics=(1, 3, 5),
                trend_lin=0.007,
                offset_lin=0.15,
                noise_scale=0.05,
                resolution=8.0,
                additive=True,
            ),
        },
        {
            "name": "5. Exponential drift",
            "notes": "5. New exponential trend growth on top of the seasonal structure.",
            "params": make_component_params(
                weekly_param=2.0,
                monthly_param=1.4,
                harmonics=(1, 3, 5),
                trend_lin=0.006,
                trend_exp=1.01,
                offset_exp=0.1,
                noise_scale=0.06,
                resolution=8.0,
                additive=True,
            ),
        },
        {
            "name": "6. Multiplicative interaction",
            "notes": "6. Trend and seasonality interact,\nchanging seasonal amplitude over time.",
            "params": make_component_params(
                weekly_param=2.3,
                monthly_param=1.6,
                harmonics=(2, 4, 5),
                trend_lin=0.007,
                trend_exp=1.015,
                offset_exp=0.2,
                noise_scale=0.07,
                resolution=8.0,
                additive=False,
            ),
        },
        {
            "name": "7. Moderate noise",
            "notes": "7. Stronger stochastic variation.",
            "params": make_component_params(
                weekly_param=2.5,
                monthly_param=1.8,
                harmonics=(2, 4, 6),
                trend_lin=0.008,
                trend_exp=1.02,
                offset_exp=0.25,
                noise_k=1.6,
                noise_scale=0.14,
                resolution=7.0,
                additive=False,
                trend_scaled_noise=True,
            ),
        },
        {
            "name": "8. Hardest mixed case",
            "notes": "8. Higher harmonics, stronger drift,\nlower effective resolution, and high noise.",
            "params": make_component_params(
                annual_param=1.5,
                monthly_param=2.0,
                weekly_param=2.8,
                harmonics=(3, 5, 7),
                trend_lin=0.010,
                trend_exp=1.03,
                offset_lin=0.2,
                offset_exp=0.35,
                noise_k=1.1,
                noise_scale=0.24,
                amplitude=1.2,
                resolution=5.5,
                additive=False,
                trend_scaled_noise=True,
            ),
        },
    ]


def generate_qualitative_examples(
    series_specs: list[dict] | None = None,
    *,
    split: SplitConfig | None = None,
    seed: int = 7,
) -> tuple[list[dict], pd.DataFrame]:
    split = split or SplitConfig()
    series_specs = series_specs or default_series_specs()

    torch.manual_seed(seed)
    np.random.seed(seed)

    examples = []
    for spec in series_specs:
        y, noiseless = generate_series(spec["params"], series_len=split.series_len, num_samples=1)
        examples.append(
            {
                "name": spec["name"],
                "notes": spec["notes"],
                "series": y[0].cpu().numpy(),
                "noiseless": noiseless[0].cpu().numpy(),
                "params": spec["params"],
            }
        )

    summary = pd.DataFrame(
        [
            {
                "name": example["name"],
                "notes": example["notes"],
                "resolution": float(example["params"].resolution.item()),
                "harmonics": tuple(int(x) for x in example["params"].harmonics.tolist()),
                "trend_lin": float(example["params"].trend_lin.item()),
                "trend_exp": float(example["params"].trend_exp.item()),
                "noise_scale": float(example["params"].noise_scale.item()),
                "additive": bool(example["params"].p_add.item()),
            }
            for example in examples
        ]
    )
    return examples, summary


def create_patches(series: np.ndarray | torch.Tensor, lookback: int, horizon: int):
    series_t = torch.as_tensor(series, dtype=torch.float32)
    endpoints = torch.arange(lookback - 1, len(series_t) - horizon, dtype=torch.long)
    patch_x = torch.stack([series_t[t - lookback + 1 : t + 1] for t in endpoints], dim=0)
    patch_z = torch.stack([series_t[t + 1 : t + 1 + horizon] for t in endpoints], dim=0)
    return patch_x, patch_z, endpoints


def compute_robust_scaler(patches_x: torch.Tensor):
    mu = patches_x.mean()
    med = patches_x.median()
    mad = (patches_x - med).abs().median()
    sigma = (1.4826 * mad).clamp_min(0.10)
    return mu, sigma


def prepare_linearpfn_task(series: np.ndarray, *, split: SplitConfig | None = None, num_context: int = 100) -> dict:
    split = split or SplitConfig()
    patch_x, patch_z, endpoints = create_patches(series, split.lookback, split.horizon)

    endpoints_np = endpoints.cpu().numpy()
    ctx_candidates = np.nonzero((endpoints_np + 1 + split.horizon) <= split.context_end)[0]
    if len(ctx_candidates) == 0:
        raise ValueError("No valid context patches found for the requested split.")

    num_context = min(int(num_context), len(ctx_candidates))
    ctx_idx = np.linspace(0, len(ctx_candidates) - 1, num=num_context, dtype=int)
    ctx_idx = ctx_candidates[ctx_idx]

    qry_matches = np.nonzero(endpoints_np == split.query_endpoint)[0]
    if len(qry_matches) != 1:
        raise ValueError(f"Expected one query patch at endpoint {split.query_endpoint}, found {len(qry_matches)}.")
    qry_idx = int(qry_matches[0])

    mu, sigma = compute_robust_scaler(patch_x[ctx_idx])
    x_norm = torch.clamp((patch_x - mu) / sigma, -10.0, 10.0)
    z_norm = torch.clamp((patch_z - mu) / sigma, -10.0, 10.0)

    return {
        "ctx_x": x_norm[ctx_idx].unsqueeze(0),
        "ctx_z": z_norm[ctx_idx].unsqueeze(0),
        "qry_x": x_norm[qry_idx : qry_idx + 1].unsqueeze(0),
        "qry_z": z_norm[qry_idx : qry_idx + 1].unsqueeze(0),
        "t_ctx": endpoints[ctx_idx].unsqueeze(0),
        "t_qry": endpoints[qry_idx : qry_idx + 1].unsqueeze(0),
        "stats": {"mu": mu, "sigma": sigma},
        "indices": {"ctx": ctx_idx, "qry": qry_idx},
        "endpoints": {"ctx": endpoints_np[ctx_idx], "qry": int(endpoints_np[qry_idx])},
        "raw": {
            "query_history": np.asarray(series[split.query_start : split.query_end]),
            "target": np.asarray(series[split.target_start : split.target_end]),
        },
    }

@torch.no_grad()
def run_linearpfn_on_series(
    model: torch.nn.Module,
    series: np.ndarray,
    *,
    split: SplitConfig | None = None,
    num_context: int = 100,
    device: torch.device | None = None,
) -> dict:
    split = split or SplitConfig()
    device = device or next(model.parameters()).device
    task = prepare_linearpfn_task(series, split=split, num_context=num_context)

    pred_norm = model(
        task["ctx_x"].to(device),
        task["ctx_z"].to(device),
        task["qry_x"].to(device),
        task["t_ctx"].to(device),
        task["t_qry"].to(device),
    ).squeeze(0).squeeze(0).detach().cpu()

    mu = task["stats"]["mu"].detach().cpu()
    sigma = task["stats"]["sigma"].detach().cpu()
    pred = (pred_norm * sigma + mu).numpy()
    target = task["raw"]["target"]
    mae = float(np.mean(np.abs(pred - target)))

    return {
        "prediction": pred,
        "target": target,
        "query_history": task["raw"]["query_history"],
        "mae": mae,
        "task": task,
    }


def run_linearpfn_on_examples(
    examples: list[dict],
    *,
    split: SplitConfig | None = None,
    num_context: int = 100,
    device: torch.device | None = None,
) -> list[dict]:
    _, model = build_model_from_hf()
    model.to(device)

    results = []
    for example in examples:
        result = run_linearpfn_on_series(
            model,
            example["series"],
            split=split,
            num_context=num_context,
            device=device,
        )
        results.append(
            {
                "name": example["name"],
                "notes": example["notes"],
                "series": example["series"],
                "noiseless": example["noiseless"],
                **result,
            }
        )
    return results


def _draw_split_regions(ax, split: SplitConfig) -> None:
    ax.axvspan(0, split.context_end, color="#d3f9d8", alpha=0.35)
    ax.axvspan(split.query_start, split.query_end, color="#fff3bf", alpha=0.45)
    ax.axvspan(split.target_start, split.target_end, color="#ffc9c9", alpha=0.35)
    ax.axvline(split.query_start, color="#2b8a3e", linestyle="--", linewidth=1.2)
    ax.axvline(split.target_start, color="#c92a2a", linestyle="--", linewidth=1.2)
    ax.grid(False)


def plot_examples_overview(examples: list[dict], *, split: SplitConfig | None = None):
    split = split or SplitConfig()
    set_qualitative_plot_style()

    fig, axes = plt.subplots(4, 2, figsize=(18, 16), sharex=True)
    axes = axes.flatten()
    x = np.arange(split.series_len)

    for ax, example in zip(axes, examples):
        ax.plot(x, example["noiseless"], color="#9aa0a6", linewidth=1.4, alpha=0.9, label="Noiseless base")
        ax.plot(x, example["series"], color="#0b7285", linewidth=1.8, label="Observed series")
        _draw_split_regions(ax, split)
        ax.set_title(example["name"], fontsize=12, loc="left")
        ax.text(
            0.01,
            0.98,
            example["notes"],
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 3},
        )
        ax.set_ylabel("value")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Eight synthetic series with increasing qualitative difficulty", fontsize=18, y=0.995)
    fig.supxlabel("time index")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig, axes


def plot_single_example(example: dict, *, split: SplitConfig | None = None):
    split = split or SplitConfig()
    set_qualitative_plot_style()

    fig, ax = plt.subplots(figsize=(16, 4.5))
    ax.plot(example["series"], color="#0b7285", linewidth=2, label="Observed series")
    ax.axvspan(0, split.context_end, color="#d3f9d8", alpha=0.35, label="Context")
    ax.axvspan(split.query_start, split.query_end, color="#fff3bf", alpha=0.45, label="Query")
    ax.axvspan(split.target_start, split.target_end, color="#ffc9c9", alpha=0.35, label="Target")
    ax.axvline(split.query_start, color="#2b8a3e", linestyle="--", linewidth=1.2)
    ax.axvline(split.target_start, color="#c92a2a", linestyle="--", linewidth=1.2)
    ax.grid(False)
    ax.set_title(example["name"])
    ax.set_xlabel("time index")
    ax.set_ylabel("value")
    ax.legend(frameon=False, ncol=4)
    return fig, ax


def plot_forecast_overview(results: list[dict], *, split: SplitConfig | None = None):
    split = split or SplitConfig()
    set_qualitative_plot_style()

    fig, axes = plt.subplots(4, 2, figsize=(18, 16), sharex=True)
    axes = axes.flatten()
    x = np.arange(split.series_len)
    horizon_x = np.arange(split.target_start, split.target_end)

    for ax, result in zip(axes, results):
        ax.plot(x, result["series"], color="#7ba2c8df", linewidth=1.3, alpha=0.9, label="Observed series")
        ax.plot(horizon_x, result["target"], color="#c92a2a", linewidth=2.0, label="Ground truth")
        ax.plot(horizon_x, result["prediction"], color="#1864ab", linewidth=2.0, label="LinearPFN forecast")
        _draw_split_regions(ax, split)
        ax.set_title(f"{result['name']}  |  MAE={result['mae']:.3f}", fontsize=12, loc="left")
        ax.text(
            0.01,
            0.98,
            result["notes"],
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 3},
        )
        ax.set_ylabel("value")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("LinearPFN forecasts on the qualitative synthetic examples", fontsize=18, y=0.995)
    fig.supxlabel("time index")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig, axes


def plot_forecast_paper_figure(results: list[dict], *, split: SplitConfig | None = None):
    split = split or SplitConfig()
    set_qualitative_plot_style()

    fig, axes = plt.subplots(4, 2, figsize=(14, 6), sharex=True)
    axes = axes.flatten()
    x = np.arange(split.series_len)
    horizon_x = np.arange(split.target_start, split.target_end)

    for ax, result in zip(axes, results):
        ax.plot(x, result["series"], color="#7590ad", linewidth=1.0, alpha=0.9, label="Observed series")
        ax.plot(horizon_x, result["target"], color="#1864ab", linewidth=1.6, label="Ground truth")
        ax.plot(horizon_x, result["prediction"], color="#c92a2a", linewidth=1.6, label="LinearPFN forecast")

        ax.axvspan(0, split.context_end, color="#d3f9d8", alpha=0.25)
        ax.axvspan(split.query_start, split.query_end, color="#fff3bf", alpha=0.3)
        ax.axvspan(split.target_start, split.target_end, color="#ffc9c9", alpha=0.22)
        ax.axvline(split.query_start, color="#2b8a3e", linestyle="--", linewidth=0.9)
        ax.axvline(split.target_start, color="#c92a2a", linestyle="--", linewidth=0.9)
        ax.grid(False)

        ax.text(
            0.015,
            0.96,
            # f"{result['name']}\n{result['notes']}",
            f"{result['notes']}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=12,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.5},
        )

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        ax.set_xlim(0, split.series_len - 1)
        ax.tick_params(axis="both", labelsize=8, length=2.5)
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.tick_params(axis="x", labelsize=16)

    for ax in axes[len(results) :]:
        ax.axis("off")

    line_handles, line_labels = axes[0].get_legend_handles_labels()
    color_handles = [
        Patch(facecolor="#d3f9d8", edgecolor="#6d9272", alpha=0.25, label="Context"),
        Patch(facecolor="#fff3bf", edgecolor="#9b926a", alpha=0.3, label="Query"),
        Patch(facecolor="#ffc9c9", edgecolor="#a87373", alpha=0.22, label="Target"),
    ]
    handles = line_handles + color_handles
    labels = line_labels + [handle.get_label() for handle in color_handles]
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.975], pad=0.7, h_pad=0.5, w_pad=0.45)
    return fig, axes


def plot_forecast_detail(result: dict, *, split: SplitConfig | None = None):
    split = split or SplitConfig()
    set_qualitative_plot_style()

    fig, ax = plt.subplots(figsize=(16, 4.5))
    horizon_x = np.arange(split.target_start, split.target_end)
    ax.plot(result["series"], color="#adb5bd", linewidth=1.3, alpha=0.9, label="Observed series")
    ax.plot(horizon_x, result["target"], color="#c92a2a", linewidth=2.0, label="Ground truth")
    ax.plot(horizon_x, result["prediction"], color="#1864ab", linewidth=2.0, label="LinearPFN forecast")
    _draw_split_regions(ax, split)
    ax.set_title(f"{result['name']}  |  MAE={result['mae']:.3f}")
    ax.set_xlabel("time index")
    ax.set_ylabel("value")
    ax.legend(frameon=False, ncol=3)
    return fig, ax
