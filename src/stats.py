"""Statistical and diagnostic utilities for experiment analysis."""

# ~~ Imports ~~ ────────────────────────────────────────────────────────────────
import torch
from models import BiaslessReluNet
from torch import Tensor

# ~~ Exports ~~ ────────────────────────────────────────────────────────────────
__all__: list[str] = [
    "eigenvalues_gram_pq",
    "empirical_columnwise_vw_norm",
    "empirical_qk_norm_ratio",
    "empirical_rankcomp_ib",
    "norm_ratio_pq",
]


# ~~ Functions ~~ ──────────────────────────────────────────────────────────────
def empirical_rankcomp_ib(p: Tensor, q: Tensor) -> Tensor:
    """Compute 2 * svdvals(p @ q), sorted descending."""
    return 2 * torch.linalg.svdvals(p @ q).sort(descending=True).values


def eigenvalues_gram_pq(p: Tensor, q: Tensor) -> tuple[Tensor, Tensor]:
    """Return sorted (descending) singular values of P and Q."""
    sv_p = torch.linalg.svdvals(p).sort(descending=True).values
    sv_q = torch.linalg.svdvals(q).sort(descending=True).values
    return sv_p, sv_q


def empirical_qk_norm_ratio(q: Tensor, k: Tensor, eps: float = 1e-12) -> Tensor:
    """Column-wise L2 norm ratio ||Q[:, j]|| / ||K[:, j]||."""
    return torch.linalg.vector_norm(q, dim=0) / (torch.linalg.vector_norm(k, dim=0) + eps)


def norm_ratio_pq(p: Tensor, q: Tensor, eps: float = 1e-12) -> Tensor:
    """Per-mode L2 norm ratio ||p[:, i]||₂ / ||q[i, :]||₂ for p (out×rank), q (rank×in)."""
    p_norms = torch.linalg.vector_norm(p, dim=0)  # (rank,)
    q_norms = torch.linalg.vector_norm(q, dim=1)  # (rank,)
    return p_norms / (q_norms + eps)


def empirical_columnwise_vw_norm(model: BiaslessReluNet, eps: float = 1e-12) -> Tensor:
    """Empirical column-wise norm ratio ||v[:, i]||₂ / ||w[i, :]||₂ for v (out×rank), w (rank×in)."""
    if len(model.layers) != 2:
        raise ValueError("Model must have exactly 2 layers.")
    v_norms: Tensor = torch.linalg.vector_norm(model.layers[-1].weight.detach().clone(), dim=0)  # type: ignore
    w_norms: Tensor = torch.linalg.vector_norm(model.layers[0].weight.detach().clone(), dim=-1)  # type: ignore
    return v_norms / (w_norms + eps)


# ~~ Classes ~~ ────────────────────────────────────────────────────────────────

# ~~ Main Execution ~~ ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Do nothing
    pass
