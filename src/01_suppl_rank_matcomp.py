"""Low-rank matrix completion experiment."""

# ~~ Imports ~~ ────────────────────────────────────────────────────────────────
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import simple_parsing
import torch
from data import generate_normal_rank_qr
from models import MatrixFactorization
from numpy.typing import NDArray
from optim import easy_sgd
from plotting import custom_plot_setup
from plotting import plot_out
from plotting import set_petroff_2021_colors
from safetensors.torch import save_file as save_safetensors
from seeder import seed_everything
from stats import eigenvalues_gram_pq
from stats import empirical_rankcomp_ib
from stats import norm_ratio_pq
from torch.optim import SGD
from tqdm.auto import trange
from utils import config_tag
from utils import resolve_device
from utils import torch_set_hiperf_precision


# ~~ Configuration ~~ ──────────────────────────────────────────────────────────
@dataclass
class Config:
    """Configuration for the low-rank matrix completion experiment."""

    seed: int = 0
    n_size: int = 20
    m_size: int = 20
    model_rank: int = 20
    observed_fraction: float = 0.2
    batching_ratio: float = 0.5
    true_sigma: tuple[float, ...] = (15.0, 5.0)
    num_steps: int = 500_000
    learning_rate: float = 1e-3
    log_every: int = 1000
    device: str = "auto"


if __name__ == "__main__":
    cfg = simple_parsing.parse(Config)
    tag: str = config_tag(cfg)

    # ~~ Problem Properties ~~ ─────────────────────────────────────────────────
    true_ib: list[float] = [float(2 * v) for v in sorted(cfg.true_sigma, reverse=True)]

    # ~~ Extra Configuration ~~ ────────────────────────────────────────────────
    device: torch.device = resolve_device(cfg.device)
    torch_set_hiperf_precision(newapi=True)

    # ~~ Training setup ~~ ─────────────────────────────────────────────────────
    seed_everything(seed=cfg.seed)

    m_true: torch.Tensor = generate_normal_rank_qr(
        n=cfg.n_size,
        m=cfg.m_size,
        sigma=cfg.true_sigma,
        device=device,
    )

    obs_mask: torch.Tensor = torch.rand(size=(cfg.m_size, cfg.n_size), device=device) < cfg.observed_fraction

    model: MatrixFactorization = MatrixFactorization(
        in_features=cfg.n_size,
        out_features=cfg.m_size,
        rank=cfg.model_rank,
        device=device,
        init_std=0.1,
    )

    optimizer: SGD = easy_sgd(params=model.parameters(), lr=cfg.learning_rate)

    # ~~ Training loop ~~ ──────────────────────────────────────────────────────
    nsteps: list[int] = []
    n_logged: int = (
        cfg.num_steps // cfg.log_every
        + (1 if cfg.log_every > 1 else 0)
        + (1 if cfg.num_steps % cfg.log_every != 0 else 0)
    )
    losses: torch.Tensor = torch.empty(n_logged, device=device)
    log_idx: int = 0
    emp_ib: list[torch.Tensor] = []
    p_eigs: list[torch.Tensor] = []
    q_eigs: list[torch.Tensor] = []
    pq_norm_ratios: list[torch.Tensor] = []

    for step in trange(1, cfg.num_steps + 1):
        m_pred: torch.Tensor = model()
        if cfg.batching_ratio == 0.0:
            obs_indices = obs_mask.nonzero()
            chosen = obs_indices[torch.randint(len(obs_indices), (1,))]
            batch_mask = torch.zeros_like(obs_mask)
            batch_mask[chosen[0, 0], chosen[0, 1]] = True
        elif cfg.batching_ratio < 1.0:
            batch_mask = obs_mask & (torch.rand_like(obs_mask, dtype=torch.float) < cfg.batching_ratio)
        else:
            batch_mask = obs_mask
        loss: torch.Tensor = ((m_true - m_pred) * batch_mask).pow(exponent=2).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if step % cfg.log_every == 0 or step == 1 or step == cfg.num_steps:
                rankcomp_stats: torch.Tensor = empirical_rankcomp_ib(
                    p=model.p.detach().clone(),
                    q=model.q.detach().clone(),
                )
                nsteps.append(step)
                losses[log_idx] = loss.detach()
                log_idx += 1
                emp_ib.append(rankcomp_stats.cpu())
                eig_p, eig_q = eigenvalues_gram_pq(
                    p=model.p.detach().clone(),
                    q=model.q.detach().clone(),
                )
                p_eigs.append(eig_p.cpu())
                q_eigs.append(eig_q.cpu())
                pq_norm_ratios.append(norm_ratio_pq(p=model.p.detach(), q=model.q.detach()).cpu())

    # ~~ Stats manipulation ~~ ─────────────────────────────────────────────────
    np_steps: NDArray = np.array(nsteps)
    np_losses: NDArray = losses.cpu().numpy()
    # np_emp_ib shape: (n_modes, n_logged) — each row is one SV trace over time
    np_emp_ib: NDArray = np.stack([t.numpy() for t in emp_ib], axis=0).T
    np_p_eigs: NDArray = np.stack([t.numpy() for t in p_eigs], axis=0).T
    np_q_eigs: NDArray = np.stack([t.numpy() for t in q_eigs], axis=0).T
    np_pq_norm_ratios: NDArray = np.stack([t.numpy() for t in pq_norm_ratios], axis=0).T

    n_true: int = len(cfg.true_sigma)
    true_sigma_sorted: NDArray = np.array(sorted(cfg.true_sigma, reverse=True))

    # ~~ Save plot data ~~ ────────────────────────────────────────────────────
    save_safetensors(
        tensors={
            "steps": torch.from_numpy(np_steps).cpu().contiguous(),
            "losses": torch.from_numpy(np_losses).cpu().contiguous(),
            "emp_ib": torch.from_numpy(np_emp_ib).cpu().contiguous(),
            "pq_norm_ratios": torch.from_numpy(np_pq_norm_ratios).cpu().contiguous(),
        },
        filename=str(Path(__file__).resolve().parent.parent / "saved" / f"fig_02_matcomp_{tag}.safetensors"),
        metadata={
            **{k: str(v) for k, v in dataclasses.asdict(cfg).items()},
            **{f"true_ib_{i}": str(v) for i, v in enumerate(true_ib)},
        },
    )

    # ~~ Printout ~~ ───────────────────────────────────────────────────────────
    print(f"Final loss after {cfg.num_steps} steps: {np_losses[-1]:.6f}")

    # ~~ Plotting ~~ ───────────────────────────────────────────────────────────
    custom_plot_setup()
    set_petroff_2021_colors()

    fig, axes = plt.subplots(3, 2, figsize=(14, 15), constrained_layout=True)

    # -- Plot 1 (top-left): Mode energies — theoretical vs empirical --
    ax1 = axes[0, 0]
    for i in range(n_true):
        ax1.plot(np_steps, np_emp_ib[i], linewidth=2, color=f"C{i + 1}", label=rf"$S_{{{i + 1}}}$")
        ax1.axhline((true_ib[i] / 2.0), linestyle="--", color=f"C{i + 1}", label=rf"$2\sigma_{{{i + 1}}}^\star$")
    ax1.set_xlim(0, cfg.num_steps)
    ax1.set_title(r"Mode energies: theoretical vs empirical")
    ax1.set_xlabel("SGD iteration")
    ax1.set_ylabel(r"$S_i$")
    ax1.grid(True)
    ax1.legend(fontsize="small", ncol=2)
    ax1.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0), useMathText=True)

    # -- Plot 2 (top-right): All non-zero empirical SVs --
    ax2 = axes[0, 1]
    threshold: float = float(1e-2 * np.max(np_emp_ib[:, -1]))
    n_modes_total: int = np_emp_ib.shape[0]
    for i in range(n_modes_total):
        if np.max(np_emp_ib[i]) < threshold:
            continue
        if i < n_true:
            ax2.plot(np_steps, np_emp_ib[i], linewidth=2, color=f"C{i + 1}", label=rf"$S_{{{i + 1}}}$")
        else:
            ax2.plot(
                np_steps,
                np_emp_ib[i],
                linewidth=1,
                color="gray",
                alpha=0.5,
                label="Extra modes" if i == n_true else None,
            )
    for i in range(n_true):
        ax2.axhline((true_ib[i] / 2.0), linestyle="--", color=f"C{i + 1}")
    ax2.set_xlim(0, cfg.num_steps)
    ax2.set_title("All significant empirical SVs")
    ax2.set_xlabel("SGD iteration")
    ax2.set_ylabel(r"$S_i$")
    ax2.grid(True)
    ax2.legend(fontsize="small", ncol=2)
    ax2.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0), useMathText=True)

    # -- Plot 3 (bottom-left): Log-sum quantities --
    ax3 = axes[1, 0]

    def _log_sum_terms(sigmas: NDArray) -> tuple[float, float, float]:
        """Compute Term A, B, C from an array of singular values."""
        sigmas = sigmas[sigmas > 0]
        term_a = float(np.sum(np.log(sigmas)))
        term_b = 0.0
        for ii in range(len(sigmas)):
            for jj in range(ii + 1, len(sigmas)):
                term_b += float(np.log(sigmas[ii] + sigmas[jj]))
        return term_a, term_b, term_a + term_b

    # Theoretical values
    theory_a, theory_b, theory_c = _log_sum_terms(true_sigma_sorted)

    # Empirical traces — 3a (all significant SVs) and 3b (top-k only)
    n_steps_logged: int = np_emp_ib.shape[1]
    emp_a_all = np.empty(n_steps_logged)
    emp_b_all = np.empty(n_steps_logged)
    emp_c_all = np.empty(n_steps_logged)
    emp_a_topk = np.empty(n_steps_logged)
    emp_b_topk = np.empty(n_steps_logged)
    emp_c_topk = np.empty(n_steps_logged)

    for t in range(n_steps_logged):
        # All significant SVs (half the mode energy = singular value)
        all_svs = np_emp_ib[:, t] / 2.0
        sig_mask = all_svs > threshold / 2.0
        sig_svs = all_svs[sig_mask]
        emp_a_all[t], emp_b_all[t], emp_c_all[t] = (
            _log_sum_terms(sig_svs) if len(sig_svs) > 0 else (np.nan, np.nan, np.nan)
        )

        # Top-k only
        topk_svs = all_svs[:n_true]
        emp_a_topk[t], emp_b_topk[t], emp_c_topk[t] = (
            _log_sum_terms(topk_svs) if np.all(topk_svs > 0) else (np.nan, np.nan, np.nan)
        )

    colors_3 = ["C3", "C4", "C5"]
    labels_3 = [r"$\sum_i \log \sigma_i$", r"$\sum_{i<j} \log(\sigma_i+\sigma_j)$", "A + B"]

    for vals_all, vals_topk, hval, col, lab in zip(
        [emp_a_all, emp_b_all, emp_c_all],
        [emp_a_topk, emp_b_topk, emp_c_topk],
        [theory_a, theory_b, theory_c],
        colors_3,
        labels_3,
        strict=True,
    ):
        ax3.plot(np_steps, vals_all, linewidth=1.5, color=col, label=f"{lab} (all)")
        ax3.plot(np_steps, vals_topk, linewidth=1.5, color=col, linestyle="--", label=f"{lab} (top-k)")
        ax3.axhline(hval, linestyle=":", color=col, alpha=0.6)

    ax3.set_xlim(0, cfg.num_steps)
    ax3.set_title("Log-sum diagnostic quantities")
    ax3.set_xlabel("SGD iteration")
    ax3.grid(True)
    ax3.legend(fontsize="x-small", ncol=2)
    ax3.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0), useMathText=True)

    # -- Plot 4 (bottom-right): Eigenvalue ratios of P^TP vs QQ^T --
    ax4 = axes[1, 1]
    n_gram_modes: int = np_p_eigs.shape[0]
    for i in range(n_gram_modes):
        num = np_p_eigs[i]
        den = np_q_eigs[i]
        # Suppress where both are negligible
        both_small = (num < threshold) & (den < threshold)
        ratio = np.where(both_small, np.nan, num / (den + 1e-30))
        if np.all(np.isnan(ratio)):
            continue
        ax4.plot(np_steps, ratio, linewidth=1.5, color=f"C{i + 1}", label=rf"Mode {i + 1}")

    ax4.set_xlim(0, cfg.num_steps)
    ax4.set_title(r"Eigenvalue ratios: $\mathrm{eig}(P^\top P)_i \,/\, \mathrm{eig}(QQ^\top)_i$")
    ax4.set_xlabel("SGD iteration")
    ax4.set_ylabel("Ratio")
    ax4.grid(True)
    ax4.legend(fontsize="small", ncol=2)
    ax4.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0), useMathText=True)

    # -- Plot 5 (bottom-bottom-left): L2 norm ratios ||p_col_i|| / ||q_row_i|| --
    ax5 = axes[2, 0]
    for i in range(np_pq_norm_ratios.shape[0]):
        ax5.plot(np_steps, np_pq_norm_ratios[i], linewidth=1.5, color=f"C{i + 1}", label=rf"Mode {i + 1}")
    ax5.set_xlim(0, cfg.num_steps)
    ax5.set_title(r"L2 norm ratio: $\|p_i\| \,/\, \|q_i\|$")
    ax5.set_xlabel("SGD iteration")
    ax5.set_ylabel("Ratio")
    ax5.grid(True)
    ax5.legend(fontsize="small", ncol=2)
    ax5.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0), useMathText=True)

    axes[2, 1].set_visible(False)

    plot_out(str(Path(__file__).resolve().parent.parent / "figures" / f"fig_02_matcomp_{tag}.png"))
