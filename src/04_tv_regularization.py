"""Total-variation regularization via implicit bias experiment."""

# ~~ Imports ~~ ────────────────────────────────────────────────────────────────
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import simple_parsing
import torch
import torch.nn.functional as F
from data import generate_normal_meas_steps
from data import generate_step_signal
from models import CumsumSignalRegressor
from models import Simple1DLinear
from models import total_variation
from optim import easy_sgd
from plotting import custom_plot_setup
from plotting import plot_out
from plotting import set_petroff_2021_colors
from safetensors.torch import save_file as save_safetensors
from seeder import seed_everything
from tqdm.auto import trange
from training import train_on_step_signal_experiment
from utils import config_tag
from utils import resolve_device
from utils import torch_set_hiperf_precision


# ~~ Configuration ~~ ──────────────────────────────────────────────────────────
@dataclass
class Config:
    """Configuration for the TV regularization experiment."""

    seed: int = 0
    signal_length: int = 200
    signal_segments: tuple[tuple[float, int], ...] = (
        (1.0, 50),
        (-1.5, 70),
        (0.5, 40),
        (2.0, 40),
    )
    observed_measurements: int = 60
    output_noise_std: float = 0.1
    num_steps: int = 500_000
    num_steps_for_acc: int = 0
    weight_init_scale_simple: float = 0.0001
    weight_init_scale_cumsum: float = 0.3
    learning_rate_simple: float = 0.01
    learning_rate_cumsum: float = 0.001
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
    true_signal: torch.Tensor = generate_step_signal(segments=cfg.signal_segments, device=device)

    ameas: torch.Tensor
    ymeas: torch.Tensor
    ameas, ymeas = generate_normal_meas_steps(
        true_signal=true_signal,
        m=cfg.observed_measurements,
        noise_std=cfg.output_noise_std,
        device=device,
    )

    # ~~ Training setup ~~ ─────────────────────────────────────────────────────
    simple_model: Simple1DLinear = Simple1DLinear(
        features=cfg.signal_length,
        device=device,
        init_std=cfg.weight_init_scale_simple,
    )

    cumsum_model: CumsumSignalRegressor = CumsumSignalRegressor(
        features=cfg.signal_length,
        device=device,
        init_std=cfg.weight_init_scale_cumsum,
    )

    # ~~ Training ~~ ───────────────────────────────────────────────────────────
    simple_train_losses: torch.Tensor = train_on_step_signal_experiment(
        experiment_name="naive-ur",
        model=simple_model,
        ameas=ameas,
        ymeas=ymeas,
        num_steps=cfg.num_steps,
        learning_rate=cfg.learning_rate_simple,
        device=device,
        print_every=None,
    )

    cumsum_train_losses: torch.Tensor = train_on_step_signal_experiment(
        experiment_name="cumsum-ur",
        model=cumsum_model,
        ameas=ameas,
        ymeas=ymeas,
        num_steps=cfg.num_steps,
        learning_rate=cfg.learning_rate_cumsum,
        device=device,
        print_every=None,
    )

    # ~~ Eval metrics at plot stage ~~ ─────────────────────────────────────────
    with torch.no_grad():
        plot_recon_mse_simple: float = F.mse_loss(simple_model.w, true_signal).item()
        plot_recon_mse_cumsum: float = F.mse_loss(cumsum_model.w, true_signal).item()
        plot_tv_simple: float = total_variation(simple_model.w).item()
        plot_tv_cumsum: float = total_variation(cumsum_model.w).item()
        plot_fit_mse_simple: float = F.mse_loss(simple_model(ameas), ymeas).item()
        plot_fit_mse_cumsum: float = F.mse_loss(cumsum_model(ameas), ymeas).item()

    # ~~ Extended training for accuracy ~~ ─────────────────────────────────────
    total_steps: int = max(cfg.num_steps_for_acc, cfg.num_steps)
    extra_steps: int = total_steps - cfg.num_steps

    if extra_steps > 0:
        models_and_names: list[tuple[str, Simple1DLinear | CumsumSignalRegressor, float]] = [
            ("naive-ur", simple_model, cfg.learning_rate_simple),
            ("cumsum-ur", cumsum_model, cfg.learning_rate_cumsum),
        ]
        for ext_name, ext_model, ext_lr in models_and_names:
            ext_optimizer = easy_sgd(ext_model.parameters(), lr=ext_lr)
            ext_model.train()
            for _step in trange(extra_steps, desc=f"Extended training ({ext_name})"):
                mse_loss = F.mse_loss(ext_model(ameas), ymeas)
                ext_optimizer.zero_grad(set_to_none=True)
                mse_loss.backward()
                ext_optimizer.step()

    # ~~ Eval metrics at final stage ~~ ────────────────────────────────────────
    with torch.no_grad():
        final_recon_mse_simple: float = F.mse_loss(simple_model.w, true_signal).item()
        final_recon_mse_cumsum: float = F.mse_loss(cumsum_model.w, true_signal).item()
        final_tv_simple: float = total_variation(simple_model.w).item()
        final_tv_cumsum: float = total_variation(cumsum_model.w).item()
        final_fit_mse_simple: float = F.mse_loss(simple_model(ameas), ymeas).item()
        final_fit_mse_cumsum: float = F.mse_loss(cumsum_model(ameas), ymeas).item()

    # ~~ Printout ~~ ───────────────────────────────────────────────────────────
    print(f"[ACC] Reconstruction MSE @ {cfg.num_steps} steps (plot):")
    print(f"  simple  | {plot_recon_mse_simple:.6f}")
    print(f"  cumsum  | {plot_recon_mse_cumsum:.6f}")
    print(f"[ACC] Reconstruction MSE @ {total_steps} steps (final):")
    print(f"  simple  | {final_recon_mse_simple:.6f}")
    print(f"  cumsum  | {final_recon_mse_cumsum:.6f}")
    print(f"[ACC] Total Variation @ {cfg.num_steps} steps (plot):")
    print(f"  simple  | {plot_tv_simple:.6f}")
    print(f"  cumsum  | {plot_tv_cumsum:.6f}")
    print(f"[ACC] Total Variation @ {total_steps} steps (final):")
    print(f"  simple  | {final_tv_simple:.6f}")
    print(f"  cumsum  | {final_tv_cumsum:.6f}")
    print(f"[ACC] Measurement Fit MSE @ {cfg.num_steps} steps (plot):")
    print(f"  simple  | {plot_fit_mse_simple:.6f}")
    print(f"  cumsum  | {plot_fit_mse_cumsum:.6f}")
    print(f"[ACC] Measurement Fit MSE @ {total_steps} steps (final):")
    print(f"  simple  | {final_fit_mse_simple:.6f}")
    print(f"  cumsum  | {final_fit_mse_cumsum:.6f}")

    # ~~ Save plot data ~~ ─────────────────────────────────────────────────────
    save_safetensors(
        tensors={
            "true_signal": true_signal.detach().clone().cpu().contiguous(),
            "w_simple": simple_model.w.detach().clone().cpu().contiguous(),
            "w_cumsum": cumsum_model.w.detach().clone().cpu().contiguous(),
            "ameas": ameas.detach().clone().cpu().contiguous(),
            "ymeas": ymeas.detach().clone().cpu().contiguous(),
            "simple_train_losses": simple_train_losses.cpu().contiguous(),
            "cumsum_train_losses": cumsum_train_losses.cpu().contiguous(),
        },
        filename=str(Path(__file__).resolve().parent.parent / "saved" / f"fig_05_tv_{tag}.safetensors"),
        metadata={k: str(v) for k, v in dataclasses.asdict(cfg).items()},
    )

    # ~~ Plotting ~~ ───────────────────────────────────────────────────────────
    custom_plot_setup()
    set_petroff_2021_colors()
    plt.figure(figsize=(10, 4))

    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

    # Get colors from the current cycle for consistency across plots
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    color_gt = colors[0]
    color_base = colors[1]
    color_inv = colors[2]

    plt.plot(
        true_signal.detach().clone().cpu().numpy(),
        label="Ground truth",
        linewidth=2,
        color=color_gt,
    )
    plt.plot(
        simple_model.w.detach().clone().cpu().numpy(),
        label="Vanilla model",
        color=color_base,
    )
    plt.plot(
        cumsum_model.w.detach().clone().cpu().numpy(),
        label="Biased model",
        color=color_inv,
    )
    plt.xlim(0, 2e2)
    plt.title(r"Recovery of structure via inverse-designed TV-minimizing implicit bias")
    plt.xlabel("Independent variable")
    plt.ylabel("Signal (dependent variable)")
    plt.legend()
    plt.tight_layout()
    plot_out(str(Path(__file__).resolve().parent.parent / "figures" / f"fig_05_tv_{tag}.png"))
