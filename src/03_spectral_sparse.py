"""Spectral-sparse signal recovery experiment."""

# ~~ Imports ~~ ────────────────────────────────────────────────────────────────
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import simple_parsing
import torch
import torch.nn.functional as F
from data import generate_sparse_coeffs
from data import generate_uniform_sparse_signal
from models import Hadamard1DLinear
from models import Simple1DLinear
from numpy.typing import NDArray
from optim import easy_sgd
from plotting import custom_plot_setup
from plotting import plot_out
from plotting import set_petroff_2021_colors
from safetensors.torch import save_file as save_safetensors
from seeder import seed_everything
from tqdm.auto import trange
from training import train_on_spectral_sparse_experiment
from utils import config_tag
from utils import resolve_device
from utils import torch_set_hiperf_precision


# ~~ Configuration ~~ ──────────────────────────────────────────────────────────
@dataclass
class Config:
    """Configuration for the spectral-sparse experiment."""

    seed: int = 456
    n_spectral_modes: int = 64
    n_spectral_modes_nonzero: int = 3
    spectral_magnitude_range: tuple[float, float] = (1.0, 2.0)
    train_samples: int = 32
    test_samples: int = 10_000
    output_noise_std: float = 0.1
    num_steps: int = 50_000
    num_steps_for_acc: int = 0
    weight_init_scale: float = 0.0001
    learning_rate: float = 1e-3
    ellone_regularization_strength: float = 0.1
    device: str = "auto"


# ~~ Main Execution ~~ ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = simple_parsing.parse(Config)
    tag: str = config_tag(cfg, exclude=("device", "num_steps_for_acc"))

    # ~~ Extra Configuration ~~ ────────────────────────────────────────────────
    device: torch.device = resolve_device(cfg.device)
    torch_set_hiperf_precision(newapi=True)

    # ~~ Data generation ~~ ────────────────────────────────────────────────────
    seed_everything(seed=cfg.seed)

    t_train: torch.Tensor = torch.rand(cfg.train_samples, 1, device=device)
    w_true: torch.Tensor = generate_sparse_coeffs(
        n=cfg.n_spectral_modes,
        n_nonzero=cfg.n_spectral_modes_nonzero,
        mag_range=cfg.spectral_magnitude_range,
        device=device,
    )

    x_train: torch.Tensor
    y_train: torch.Tensor
    x_train, y_train = generate_uniform_sparse_signal(
        n=cfg.n_spectral_modes,
        bs=cfg.train_samples,
        w_true=w_true,
        noise_std=cfg.output_noise_std,
        device=device,
        t_manual=t_train,
    )

    x_test: torch.Tensor
    y_test: torch.Tensor
    x_test, y_test = generate_uniform_sparse_signal(
        n=cfg.n_spectral_modes,
        bs=cfg.test_samples,
        w_true=w_true,
        noise_std=0.0,
        device=device,
    )

    # ~~ Training setup ~~ ─────────────────────────────────────────────────────

    naive_model_ur: Simple1DLinear = Simple1DLinear(
        features=cfg.n_spectral_modes, device=device, init_std=cfg.weight_init_scale
    )

    naive_model_rr: Simple1DLinear = Simple1DLinear(
        features=cfg.n_spectral_modes, device=device, init_std=cfg.weight_init_scale
    )

    hadamard_model_ur: Hadamard1DLinear = Hadamard1DLinear(
        features=cfg.n_spectral_modes, device=device, init_std=cfg.weight_init_scale
    )

    # ~~ Training ~~ ───────────────────────────────────────────────────────────
    naive_train_losses_ur: torch.Tensor
    naive_test_losses_ur: torch.Tensor
    naive_train_losses_ur, naive_test_losses_ur = train_on_spectral_sparse_experiment(
        experiment_name="naive-unreg",
        model=naive_model_ur,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        num_steps=cfg.num_steps,
        learning_rate=cfg.learning_rate,
        ellone_regularization_strength=0.0,
        device=device,
    )

    naive_train_losses_rr: torch.Tensor
    naive_test_losses_rr: torch.Tensor
    naive_train_losses_rr, naive_test_losses_rr = train_on_spectral_sparse_experiment(
        experiment_name="naive-reg",
        model=naive_model_rr,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        num_steps=cfg.num_steps,
        learning_rate=cfg.learning_rate,
        ellone_regularization_strength=cfg.ellone_regularization_strength,
        device=device,
    )

    hadamard_train_losses_ur: torch.Tensor
    hadamard_test_losses_ur: torch.Tensor
    hadamard_train_losses_ur, hadamard_test_losses_ur = train_on_spectral_sparse_experiment(
        experiment_name="hadamard-unreg",
        model=hadamard_model_ur,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        num_steps=cfg.num_steps,
        learning_rate=cfg.learning_rate,
        ellone_regularization_strength=0.0,
        device=device,
    )

    # ~~ Stats manipulation ~~ ─────────────────────────────────────────────────
    w_naive_ur: torch.Tensor = naive_model_ur.w.detach().clone().cpu()
    w_naive_rr: torch.Tensor = naive_model_rr.w.detach().clone().cpu()
    w_hadamard_ur: torch.Tensor = hadamard_model_ur.w.detach().clone().cpu()

    # ~~ Reconstruction computation ~~ ──────────────────────────────────────
    t_grid: torch.Tensor = torch.linspace(0, 1, cfg.test_samples // 10, device=device)

    x_plot_true: torch.Tensor
    y_plot_true: torch.Tensor
    x_plot_true, y_plot_true = generate_uniform_sparse_signal(
        n=cfg.n_spectral_modes,
        bs=cfg.test_samples // 10,
        w_true=w_true,
        noise_std=0.0,
        device=device,
        t_manual=t_grid.view(-1, 1),
    )
    x_plot_true = x_plot_true.cpu()
    with torch.no_grad():
        y_plot_naive: torch.Tensor = x_plot_true @ w_naive_ur
        y_plot_pq: torch.Tensor = x_plot_true @ w_hadamard_ur
        y_plot_naive_rr: torch.Tensor = x_plot_true @ w_naive_rr

    # ~~ Extended training for accuracy ~~ ─────────────────────────────────────
    total_steps: int = max(cfg.num_steps_for_acc, cfg.num_steps)
    extra_steps: int = total_steps - cfg.num_steps

    with torch.no_grad():
        plot_test_mse_ur: float = F.mse_loss(naive_model_ur(x_test), y_test).item()
        plot_test_mse_rr: float = F.mse_loss(naive_model_rr(x_test), y_test).item()
        plot_test_mse_h: float = F.mse_loss(hadamard_model_ur(x_test), y_test).item()

    if extra_steps > 0:
        models_and_names: list[tuple[str, Simple1DLinear | Hadamard1DLinear, float]] = [
            ("naive-unreg", naive_model_ur, 0.0),
            ("naive-reg", naive_model_rr, cfg.ellone_regularization_strength),
            ("hadamard-unreg", hadamard_model_ur, 0.0),
        ]
        for ext_name, ext_model, ext_l1 in models_and_names:
            ext_optimizer = easy_sgd(ext_model.parameters(), lr=cfg.learning_rate)
            ext_model.train()
            for _step in trange(extra_steps, desc=f"Extended training ({ext_name})"):
                mse_loss = F.mse_loss(ext_model(x_train), y_train)
                train_loss = mse_loss + ext_l1 * ext_model.w.abs().sum() if ext_l1 > 0.0 else mse_loss
                ext_optimizer.zero_grad(set_to_none=True)
                train_loss.backward()
                ext_optimizer.step()

    with torch.no_grad():
        final_test_mse_ur: float = F.mse_loss(naive_model_ur(x_test), y_test).item()
        final_test_mse_rr: float = F.mse_loss(naive_model_rr(x_test), y_test).item()
        final_test_mse_h: float = F.mse_loss(hadamard_model_ur(x_test), y_test).item()

    # ~~ Save plot data ~~ ──────────────────────────────────────────────────
    save_safetensors(
        tensors={
            "naive_train_losses_ur": naive_train_losses_ur.cpu().contiguous(),
            "naive_test_losses_ur": naive_test_losses_ur.cpu().contiguous(),
            "naive_train_losses_rr": naive_train_losses_rr.cpu().contiguous(),
            "naive_test_losses_rr": naive_test_losses_rr.cpu().contiguous(),
            "hadamard_train_losses_ur": hadamard_train_losses_ur.cpu().contiguous(),
            "hadamard_test_losses_ur": hadamard_test_losses_ur.cpu().contiguous(),
            "w_true": w_true.cpu().contiguous(),
            "w_naive_ur": w_naive_ur.contiguous(),
            "w_naive_rr": w_naive_rr.contiguous(),
            "w_hadamard_ur": w_hadamard_ur.contiguous(),
            "t_grid": t_grid.cpu().contiguous(),
            "t_train": t_train.cpu().contiguous(),
            "y_train": y_train.cpu().contiguous(),
            "x_plot_true": x_plot_true.cpu().contiguous(),
            "y_plot_true": y_plot_true.cpu().contiguous(),
            "y_plot_naive": y_plot_naive.cpu().contiguous(),
            "y_plot_pq": y_plot_pq.cpu().contiguous(),
            "y_plot_naive_rr": y_plot_naive_rr.cpu().contiguous(),
        },
        filename=str(Path(__file__).resolve().parent.parent / "saved" / f"fig_04_spectralsparse_{tag}.safetensors"),
        metadata={k: str(v) for k, v in dataclasses.asdict(cfg).items()},
    )

    # ~~ Printout ~~ ───────────────────────────────────────────────────────────
    print(f"[ACC] Test MSE @ {cfg.num_steps} steps (plot):")
    print(f"  naive-unreg     | {plot_test_mse_ur:.6f}")
    print(f"  naive-reg       | {plot_test_mse_rr:.6f}")
    print(f"  hadamard-unreg  | {plot_test_mse_h:.6f}")
    print(f"[ACC] Test MSE @ {total_steps} steps (final):")
    print(f"  naive-unreg     | {final_test_mse_ur:.6f}")
    print(f"  naive-reg       | {final_test_mse_rr:.6f}")
    print(f"  hadamard-unreg  | {final_test_mse_h:.6f}")

    # ~~ Plotting ~~ ───────────────────────────────────────────────────────────
    custom_plot_setup()
    set_petroff_2021_colors()

    plt.rcParams.update({
        "lines.linewidth": 1.0,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), constrained_layout=True)
    fig.set_constrained_layout_pads(hspace=0.1)  # type: ignore

    # Train/Test loss evolution
    ax0 = axes[0]
    (line1,) = ax0.plot(naive_train_losses_ur, label="Vanilla model (ur), train")
    (line2,) = ax0.plot(hadamard_train_losses_ur, label="Hadamard model, train")
    (line3,) = ax0.plot(naive_train_losses_rr, label="Vanilla model (rr), train")
    ax0.plot(
        naive_test_losses_ur,
        "--",
        color=line1.get_color(),
        label="Vanilla model (ur), test",
    )
    ax0.plot(
        hadamard_test_losses_ur,
        "--",
        color=line2.get_color(),
        label="Hadamard model, test",
    )
    ax0.plot(
        naive_test_losses_rr,
        "--",
        color=line3.get_color(),
        label="Vanilla model (rr), test",
    )
    ax0.set_xlabel("SGD iteration")
    ax0.set_yscale("log")
    ax0.set_title("Loss evolution")
    ax0.set_ylabel("MSE (log scale)")
    ax0.legend()
    ax0.grid(True)
    ax0.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0), useMathText=True)
    ax0.set_xlim(0, cfg.num_steps)

    color_vanilla_ur = line1.get_color()
    color_hadamard_ur = line2.get_color()
    color_vanilla_rr = line3.get_color()

    # Spectral coefficients
    ax1 = axes[1]
    ks: NDArray = np.arange(cfg.n_spectral_modes)

    stem_true = ax1.stem(ks, w_true.cpu().numpy(), linefmt="-", markerfmt="o", basefmt="k-", label="true")
    stem_true.stemlines.set_color("black")
    stem_true.markerline.set_color("black")

    stem_vanilla = ax1.stem(
        ks,
        w_naive_ur.numpy(),
        linefmt="-",
        markerfmt="s",
        basefmt="k-",
        label="vanilla (ur)",
    )
    stem_vanilla.stemlines.set_color(color_vanilla_ur)
    stem_vanilla.markerline.set_color(color_vanilla_ur)

    stem_hadamard = ax1.stem(
        ks,
        w_hadamard_ur.numpy(),
        linefmt="--",
        markerfmt="^",
        basefmt="k-",
        label="Hadamard",
    )
    stem_hadamard.stemlines.set_color(color_hadamard_ur)
    stem_hadamard.markerline.set_color(color_hadamard_ur)

    stem_vanilla_rr = ax1.stem(
        ks,
        w_naive_rr.numpy(),
        linefmt="--",
        markerfmt="d",
        basefmt="k-",
        label="vanilla (rr)",
    )
    stem_vanilla_rr.stemlines.set_color(color_vanilla_rr)
    stem_vanilla_rr.markerline.set_color(color_vanilla_rr)

    for line in ax1.lines:
        line.set_linewidth(0.8)
    for coll in ax1.collections:
        coll.set_linewidth(0.8)

    ax1.set_title("Spectral coefficients")
    ax1.set_ylabel("$w_k$")
    ax1.set_xlabel("Frequency $k$")
    ax1.legend()
    ax1.grid(True)
    ax1.set_xlim(0, 64)

    # Signal reconstruction
    ax2 = axes[2]

    ax2.plot(
        t_grid.cpu().numpy(),
        y_plot_true.cpu().numpy(),
        color="black",
        label="true",
        lw=1.2,
    )
    ax2.plot(
        t_grid.cpu().numpy(),
        y_plot_naive.cpu().numpy(),
        color=color_vanilla_ur,
        label="vanilla (ur)",
    )
    ax2.plot(
        t_grid.cpu().numpy(),
        y_plot_pq.cpu().numpy(),
        color=color_hadamard_ur,
        label="Hadamard",
        linestyle="--",
    )
    ax2.plot(
        t_grid.cpu().numpy(),
        y_plot_naive_rr.cpu().numpy(),
        color=color_vanilla_rr,
        label="vanilla (rr)",
        linestyle="--",
    )

    ax2.scatter(
        t_train.cpu().numpy(),
        y_train.cpu().numpy(),
        s=30,
        marker="x",
        color="black",
        linewidths=1.5,
        label="train data",
    )

    ax2.set_xlabel("t")
    ax2.set_ylabel("f(t)")
    ax2.set_title("Function fit")
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlim(0, 1)

    plot_out(str(Path(__file__).resolve().parent.parent / "figures" / f"fig_04_spectralsparse_{tag}.png"))
