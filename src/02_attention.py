"""Self-attention Q/K norm balancing experiment."""

# ~~ Imports ~~ ────────────────────────────────────────────────────────────────
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import simple_parsing
import torch
from data import generate_normal_from_teacher
from models import SingleHeadAttention
from models import unbalance_qk
from numpy.typing import NDArray
from optim import easy_sgd
from plotting import custom_plot_setup
from plotting import plot_out
from plotting import set_petroff_2021_colors
from safetensors.torch import save_file as save_safetensors
from seeder import seed_everything
from stats import empirical_qk_norm_ratio
from torch.optim import SGD
from tqdm.auto import trange
from utils import config_tag
from utils import resolve_device
from utils import torch_set_hiperf_precision


# ~~ Configuration ~~ ──────────────────────────────────────────────────────────
@dataclass
class Config:
    """Configuration for the attention Q/K balancing experiment."""

    seed: int = 0
    sequence_length: int = 8
    emb_dim: int = 16
    head_dim: int = 8
    output_noise_std: float = 0.02
    teacher_init_std: float = 0.75
    num_steps: int = 50_000
    num_steps_for_acc: int = 300_000
    batch_size: int = 16
    learning_rate: float = 1e-3
    student_init_std: float = 0.1
    discard_first_points: int = 5
    log_every: int = 100
    device: str = "auto"


if __name__ == "__main__":
    cfg = simple_parsing.parse(Config)
    tag: str = config_tag(cfg, exclude=("device", "num_steps_for_acc"))

    # ~~ Extra Configuration ~~ ────────────────────────────────────────────────
    device: torch.device = resolve_device(cfg.device)
    torch_set_hiperf_precision(newapi=True)

    # ~~ Training setup ~~ ─────────────────────────────────────────────────────
    seed_everything(seed=cfg.seed)

    teacher: SingleHeadAttention = SingleHeadAttention(
        d_emb=cfg.emb_dim,
        d_head=cfg.head_dim,
        wo_eye=True,
        pool_seq=False,
        device=device,
        init_std=cfg.teacher_init_std,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student: SingleHeadAttention = unbalance_qk(
        model=SingleHeadAttention(
            d_emb=cfg.emb_dim,
            d_head=cfg.head_dim,
            wo_eye=True,
            pool_seq=False,
            device=device,
            init_std=cfg.student_init_std,
        ),
        log_scale=1.2,
    )

    with torch.no_grad():
        teacher.wq.data.mul_(4 + abs(torch.randn(1).item()) * 4)
        student.wq.data.mul_(1 / 4 + abs(torch.randn(1).item()) / 4)

    optimizer: SGD = easy_sgd(params=student.parameters(), lr=cfg.learning_rate)

    # ~~ Training loop ~~ ──────────────────────────────────────────────────────
    nsteps: list[int] = []
    n_logged: int = (
        cfg.num_steps // cfg.log_every
        + (1 if cfg.log_every > 1 else 0)
        + (1 if cfg.num_steps % cfg.log_every != 0 else 0)
    )
    losses: torch.Tensor = torch.empty(n_logged, device=device)
    log_idx: int = 0
    norm_ratios: list[torch.Tensor] = []
    frob_ratios: list[float] = []
    nuc_ratios: list[float] = []

    for step in trange(1, cfg.num_steps + 1):
        x: torch.Tensor
        y: torch.Tensor
        x, y = generate_normal_from_teacher(
            shape=torch.Size((cfg.batch_size, cfg.sequence_length, cfg.emb_dim)),
            teacher=teacher,
            noise_std=cfg.output_noise_std,
            device=device,
        )

        y_pred: torch.Tensor = student(x)
        loss: torch.Tensor = torch.mean((y - y_pred) ** 2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if step % cfg.log_every == 0 or step == 1 or step == cfg.num_steps:
                wq = student.wq.detach().clone()
                wk = student.wk.detach().clone()

                norm_ratio: torch.Tensor = empirical_qk_norm_ratio(q=wq, k=wk)

                eps: float = 1e-12
                frob_ratio: float = (
                    torch.linalg.matrix_norm(wq, ord="fro") / (torch.linalg.matrix_norm(wk, ord="fro") + eps)
                ).item()
                nuc_ratio: float = (
                    torch.linalg.matrix_norm(wq, ord="nuc") / (torch.linalg.matrix_norm(wk, ord="nuc") + eps)
                ).item()

                nsteps.append(step)
                losses[log_idx] = loss.detach()
                log_idx += 1
                norm_ratios.append(norm_ratio.cpu())
                frob_ratios.append(frob_ratio)
                nuc_ratios.append(nuc_ratio)

    # ~~ Extended training for accuracy ~~ ─────────────────────────────────────
    total_steps: int = max(cfg.num_steps_for_acc, cfg.num_steps)
    extra_steps: int = total_steps - cfg.num_steps

    with torch.no_grad():
        x_eval, y_eval = generate_normal_from_teacher(
            shape=torch.Size((cfg.batch_size * 100, cfg.sequence_length, cfg.emb_dim)),
            teacher=teacher,
            noise_std=cfg.output_noise_std,
            device=device,
        )
        plot_eval_loss: float = torch.mean((y_eval - student(x_eval)) ** 2).item()

    if extra_steps > 0:
        for _step in trange(extra_steps, desc="Extended training"):
            x, y = generate_normal_from_teacher(
                shape=torch.Size((cfg.batch_size, cfg.sequence_length, cfg.emb_dim)),
                teacher=teacher,
                noise_std=cfg.output_noise_std,
                device=device,
            )
            y_pred = student(x)
            loss = torch.mean((y - y_pred) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        final_eval_loss: float = torch.mean((y_eval - student(x_eval)) ** 2).item()

    # ~~ Stats manipulation ~~ ─────────────────────────────────────────────────
    np_steps: NDArray = np.array(nsteps)[cfg.discard_first_points :]
    np_losses: NDArray = losses.cpu().numpy()[cfg.discard_first_points :]
    np_norm_ratios: NDArray = np.stack([nr.numpy() for nr in norm_ratios], axis=0)[cfg.discard_first_points :]
    np_frob_ratios: NDArray = np.array(frob_ratios)[cfg.discard_first_points :]
    np_nuc_ratios: NDArray = np.array(nuc_ratios)[cfg.discard_first_points :]

    # ~~ Save plot data ~~ ────────────────────────────────────────────────────
    save_safetensors(
        tensors={
            "steps": torch.from_numpy(np_steps).cpu().contiguous(),
            "losses": torch.from_numpy(np_losses).cpu().contiguous(),
            "norm_ratios": torch.from_numpy(np_norm_ratios).cpu().contiguous(),
            "frob_ratios": torch.from_numpy(np_frob_ratios).cpu().contiguous(),
            "nuc_ratios": torch.from_numpy(np_nuc_ratios).cpu().contiguous(),
        },
        filename=str(Path(__file__).resolve().parent.parent / "saved" / f"fig_03_attention_{tag}.safetensors"),
        metadata={k: str(v) for k, v in dataclasses.asdict(cfg).items()},
    )

    # ~~ Printout ~~ ───────────────────────────────────────────────────────────
    print(
        f"[ACC] Train MSE @ {cfg.num_steps} steps (plot): {plot_eval_loss:.6f} (train loss, large-batch eval; no test set)"
    )
    print(
        f"[ACC] Train MSE @ {total_steps} steps (final): {final_eval_loss:.6f} (train loss, large-batch eval; no test set)"
    )

    # ~~ Plotting ~~ ───────────────────────────────────────────────────────────
    custom_plot_setup()
    set_petroff_2021_colors()
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)

    for j in range(cfg.head_dim):
        ax.plot(np_steps, np_norm_ratios[:, j], linewidth=1.5)
    ax.axhline(
        1.0,
        linestyle="--",
        color="k",
        linewidth=1,
        label="Balanced ratio (i.e. ratio = 1)",
    )
    ax.set_title(r"Self-attention: K/Q column-wise norm balancing")
    ax.set_xlabel("SGD iteration")
    ax.set_ylabel(r"$\|W_Q[:,j]\| / \|W_K[:,j]\|$ ratio (per-column)")
    ax.set_xlim(cfg.discard_first_points * cfg.log_every, cfg.num_steps)
    ax.legend()
    ax.grid(True)

    ax.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0), useMathText=True)

    plot_out(str(Path(__file__).resolve().parent.parent / "figures" / f"fig_03_attention_{tag}.png"))

    # ~~ Plotting (matrix norm ratios) ~~ ──────────────────────────────────────
    custom_plot_setup()
    set_petroff_2021_colors()
    fig2, ax2 = plt.subplots(figsize=(8, 4), constrained_layout=True)

    ax2.plot(np_steps, np_frob_ratios, linewidth=1.5, label=r"Frobenius norm ratio")
    ax2.plot(np_steps, np_nuc_ratios, linewidth=1.5, label=r"Nuclear norm ratio")
    ax2.axhline(
        1.0,
        linestyle="--",
        color="k",
        linewidth=1,
        label="Balanced ratio (i.e. ratio = 1)",
    )
    ax2.set_title(r"Self-attention: Q/K matrix norm balancing")
    ax2.set_xlabel("SGD iteration")
    ax2.set_ylabel(r"$\|W_Q\| / \|W_K\|$ ratio")
    ax2.set_xlim(cfg.discard_first_points * cfg.log_every, cfg.num_steps)
    ax2.legend()
    ax2.grid(True)

    ax2.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0), useMathText=True)

    plot_out(str(Path(__file__).resolve().parent.parent / "figures" / f"fig_03_attention_matrixnorms_{tag}.png"))
