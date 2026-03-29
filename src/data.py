"""Data generation, loading, and transformation utilities."""

# ~~ Imports ~~ ────────────────────────────────────────────────────────────────
import math

import torch
from torch import Tensor

# ~~ Exports ~~ ────────────────────────────────────────────────────────────────
__all__: list[str] = [
    "generate_normal_from_teacher",
    "generate_normal_meas_steps",
    "generate_normal_rank_qr",
    "generate_sparse_coeffs",
    "generate_step_signal",
    "generate_uniform_sparse_signal",
]


# ~~ Functions ~~ ──────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_normal_rank_qr(
    n: int,
    m: int,
    sigma: list[float],
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    rank: int = len(sigma)
    if rank > min(n, m):
        raise ValueError(f"Rank {rank} is greater than the minimum of {n} and {m}")
    u: Tensor = torch.linalg.qr(torch.randn(m, rank, device=device, dtype=dtype))[0]
    v: Tensor = torch.linalg.qr(torch.randn(n, rank, device=device, dtype=dtype))[0]
    return u @ torch.diag(torch.tensor(sigma, device=device, dtype=dtype)) @ v.T


@torch.no_grad()
def generate_normal_from_teacher(
    shape: torch.Size,
    teacher: torch.nn.Module,
    noise_std: float = 1.0,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x: torch.Tensor = torch.randn(shape, device=device, dtype=dtype)
    y: torch.Tensor = teacher(x)
    y = y + torch.randn_like(y) * noise_std
    return x, y


@torch.no_grad()
def generate_sparse_coeffs(
    n: int,
    n_nonzero: int,
    mag_range: tuple[float, float] = (1.0, 2.0),
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if not (0 <= n_nonzero <= n - 1):
        raise ValueError(f"n_nonzero must be in [0, {n - 1}], got {n_nonzero}")
    w: torch.Tensor = torch.zeros(n, device=device, dtype=dtype)
    k: int = n_nonzero
    chosen: torch.Tensor = torch.randperm(n - 1, device=device)[:k] + 1
    mags: torch.Tensor = torch.rand(k, device=device, dtype=dtype) * (mag_range[1] - mag_range[0]) + mag_range[0]
    signs: torch.Tensor = torch.randint(0, 2, (k,), device=device, dtype=torch.int) * 2 - 1
    signs = signs.to(dtype=dtype)
    w[chosen] = signs * mags
    return w


@torch.no_grad()
def _generate_uniform_design_cosine(
    n: int,
    bs: int,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    t_manual: Tensor | None = None,
) -> torch.Tensor:
    if t_manual is not None:
        if t_manual.shape[0] != bs:
            raise ValueError(f"t_manual batch size {t_manual.shape[0]} != expected batch size {bs}")
        t = t_manual.to(device=device, dtype=dtype)
    else:
        t = torch.rand(bs, 1, device=device, dtype=dtype)
    ks: torch.Tensor = torch.arange(n, device=device, dtype=dtype).view(1, -1)
    return torch.cos(2 * torch.pi * t * ks)


@torch.no_grad()
def generate_uniform_sparse_signal(
    n: int,
    bs: int,
    w_true: torch.Tensor,
    noise_std: float,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    t_manual: Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x: Tensor = _generate_uniform_design_cosine(n, bs, dtype, device, t_manual)
    y: Tensor = x @ w_true.to(device, dtype=dtype)
    y = y + torch.randn_like(y) * noise_std
    return x, y


@torch.no_grad()
def generate_step_signal(
    segments: tuple[tuple[float, int], ...] = ((1.0, 50), (-1.5, 70), (0.5, 40), (2.0, 40)),
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    values: Tensor = torch.tensor([v for v, _ in segments], dtype=dtype, device=device)
    lengths: Tensor = torch.tensor([ll for _, ll in segments], dtype=torch.long, device=device)
    return torch.repeat_interleave(values, lengths)


@torch.no_grad()
def generate_normal_meas_steps(
    true_signal: torch.Tensor,
    m: int = 60,
    noise_std: float = 0.05,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    slen: int = true_signal.shape[0]
    if m <= 0 or m >= slen:
        raise ValueError(f"Number of measurements m={m} must be positive and less than the signal length.")
    a: Tensor = torch.randn(m, slen, dtype=dtype, device=device) / math.sqrt(m)
    y: Tensor = a @ true_signal.to(device=device, dtype=dtype)
    y = y + torch.randn_like(input=y, dtype=dtype, device=device) * noise_std
    return a, y


# ~~ Classes ~~ ────────────────────────────────────────────────────────────────

# ~~ Main Execution ~~ ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Do nothing
    pass
