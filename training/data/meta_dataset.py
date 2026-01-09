import math, os, glob, random
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import torch

class _ShardSeriesDataset:
    """
    Lightweight series provider over *.npy shards.
    Expected file shape per shard: (k, T) with dtype float32/float64.
      - v: torch tensor shaped [1, T, 1] to satisfy downstream indexing v[0, :, 0]
    """
    def __init__(self, shards_dir: str, mmap: bool = True):
        self.shards_dir = Path(shards_dir)
        if not self.shards_dir.exists():
            raise FileNotFoundError(f"Shards directory not found: {self.shards_dir}")

        self._paths: List[Path] = sorted(self.shards_dir.glob("y_shard_*.npy"))
        if not self._paths:
            raise FileNotFoundError(f"No shard files matched 'y_shard_*.npy' in {self.shards_dir}")

        # Index shard sizes without loading into RAM
        self._sizes: List[int] = []
        self._lengths: List[int] = []
        self._mmap = mmap
        for p in self._paths:
            arr = np.load(p, mmap_mode="r" if mmap else None)
            if arr.ndim != 2:
                raise ValueError(f"Shard {p} must have shape (k, T); got {arr.shape}")
            k, T = arr.shape
            self._sizes.append(k)
            self._lengths.append(T)
        if len(set(self._lengths)) != 1:
            raise ValueError(f"All shards must have identical series length T. Found: {set(self._lengths)}")
        self.T = self._lengths[0]
        # Sampling weights proportional to number of series per shard
        total = sum(self._sizes)
        self._weights = [s / total for s in self._sizes]

    def _load_shard(self, idx: int):
        return np.load(self._paths[idx], mmap_mode="r" if self._mmap else None)

    def get_a_context(self) -> Tuple[Optional[torch.Tensor], torch.Tensor, dict]:
        shard_idx = random.choices(range(len(self._paths)), weights=self._weights, k=1)[0]
        shard = self._load_shard(shard_idx)
        row = random.randrange(self._sizes[shard_idx])
        y_np = shard[row]  # shape (T,)

        v = torch.tensor(y_np, dtype=torch.float32).reshape(1, -1, 1)
        meta = {"source_shard": str(self._paths[shard_idx]), "row": row, "T": self.T}
        return v, meta


class VariableMetaDataset:
    """ Variable-C/Q meta-dataset that draws raw series from pre-generated shards. """
    def __init__(self, shards_dir: str, L=50, H=20, C_range=(4, 256), Q_range=(1, 16), 
                 device="cpu", sample_log_uniform=True):
        self.device = device
        self.L, self.H = L, H
        self.C_range, self.Q_range = C_range, Q_range
        self.sample_log_uniform = sample_log_uniform
        self.base_dataset = _ShardSeriesDataset(shards_dir)

        print("VariableMetaDataset:")
        print(f"  L={L}, H={H}")
        print(f"  C range: {C_range} ({'log-uniform' if sample_log_uniform else 'uniform'})")
        print(f"  Q range: {Q_range}")
        print(f"  Shards dir: {Path(shards_dir).resolve()} | T={self.base_dataset.T}")

    def _sample_C(self):
        min_C, max_C = self.C_range
        if self.sample_log_uniform:
            log_min, log_max = math.log(min_C), math.log(max_C)
            log_C = np.random.uniform(log_min, log_max)
            return min(max_C, max(min_C, int(math.exp(log_C))))
        else:
            return np.random.randint(min_C, max_C + 1)

    def _sample_Q(self):
        return np.random.randint(self.Q_range[0], self.Q_range[1] + 1)

    def sample_context_query_split(self, endpoints: np.ndarray, series_len: int, C=512, Q=8):
        split = int(0.8 * series_len)
        ctx_cand = np.nonzero((endpoints + self.H) <= split)[0]
        qry_cand = np.nonzero((endpoints - self.L + 1) >= split)[0]

        ctx_idx = np.random.choice(ctx_cand, size=C, replace=False)
        qry_idx = np.random.choice(qry_cand, size=Q, replace=False)

        return ctx_idx, qry_idx

    def create_patches(self, series: torch.Tensor):
        T = len(series)
        endpoints = np.arange(self.L - 1, T - self.H, dtype=int)
        patch_X = torch.stack([series[t - self.L + 1 : t + 1] for t in endpoints], axis=0)
        patch_Y = torch.stack([series[t + 1 : t + 1 + self.H] for t in endpoints], axis=0)
        return patch_X, patch_Y, endpoints

    def compute_scaler(self, patches_x: torch.Tensor, ctx_idx=None):
        if ctx_idx is None:
            split = int(0.8 * patches_x.shape[0])
            x_ctx = patches_x[:split]
        else:
            x_ctx = patches_x[ctx_idx]
        mu = x_ctx.mean()
        med = x_ctx.median()
        mad = (x_ctx - med).abs().median()
        sigma = (1.4826 * mad).clamp_min(0.10)
        return mu, sigma

    def create_meta_task(self):
        C = self._sample_C()
        Q = self._sample_Q()

        v, meta = self.base_dataset.get_a_context()
        y_series = v[0, :, 0].to(self.device)

        patches_x, patches_z, endpoints = self.create_patches(y_series)
        endpoints_np = np.asarray(endpoints, dtype=int)

        mu, sigma = self.compute_scaler(patches_x)
        x_norm = torch.clamp((patches_x - mu) / sigma, -10.0, 10.0)
        z_norm = torch.clamp((patches_z - mu) / sigma, -10.0, 10.0)

        ctx_idx, qry_idx = self.sample_context_query_split(endpoints_np, len(y_series), C, Q)

        meta_task = {
            "ctx_x": x_norm[ctx_idx],
            "ctx_z": z_norm[ctx_idx],
            "qry_x": x_norm[qry_idx],
            "qry_z": z_norm[qry_idx],
            "stats": {"mu": mu, "sigma": sigma},
            "raw_series": y_series,
            "endpoints": {
                "ctx": [endpoints[i] for i in ctx_idx],
                "qry": [endpoints[i] for i in qry_idx],
            },
            "source": meta,
        }
        return meta_task
