"""Recreate experiment figures using NMI publication template."""

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import nmi
import numpy as np
from matplotlib import patheffects as pe
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerPathCollection
from safetensors import safe_open


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUT_FMT: str = "png"
OUT_DPI: int = 300
SAVED_DIR: Path = Path(__file__).resolve().parent.parent / "saved"
FIGURES_DIR: Path = Path(__file__).resolve().parent.parent / "figures"

# Type alias for plot functions
PlotFn = Callable[[dict[str, np.ndarray], dict[str, str], Path], None]


# ---------------------------------------------------------------------------
# Custom legend handler: right-align scatter markers
# ---------------------------------------------------------------------------


class _RightAlignedScatter(HandlerPathCollection):
    """Position scatter marker at right edge of legend handle."""

    def get_xdata(self, legend, xdescent, ydescent, width, height, fontsize):
        xdata = [width - xdescent]
        return xdata, xdata


_SCATTER_HANDLER = {PathCollection: _RightAlignedScatter()}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_safetensor(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Load a safetensors file, returning numpy arrays and metadata."""
    with safe_open(str(path), framework="numpy") as f:
        meta = f.metadata() or {}
        data = {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118
    return data, meta


# ---------------------------------------------------------------------------
# Fig 01 — Shallow ReLU V/W norm ratio
# ---------------------------------------------------------------------------


def plot_fig_01_shallowrelu(
    data: dict[str, np.ndarray],
    meta: dict[str, str],
    out_path: Path,
) -> None:
    """V/W per-hidden-unit norm ratio evolution."""
    steps = data["steps"]
    norm_ratios = data["norm_ratios"]  # shape (N, hidden_features)

    fig, ax = nmi.subplots(1, 1, width="1.5", aspect=0.65)

    for j in range(norm_ratios.shape[1]):
        ax.plot(steps, norm_ratios[:, j])

    ax.axhline(
        1.0,
        color="k",
        **nmi.scaled_preset(nmi.DASHED_REF),
        label="Theor. balance (ratio: 1)",
    )

    ax.set_xlim(steps[0], steps[-1])
    ax.set_xlabel("Opt. iteration")
    ax.set_ylabel(r"${}^{|\mathbf{v}_j|} \! / \! {}_{\|\mathrm{W}_{[\cdot,j]}\|_{2}}$")
    ax.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0), useMathText=True)
    ax.legend()

    nmi.savefig(fig, out_path, dpi=OUT_DPI, fmt=OUT_FMT)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Fig 02 — Matrix completion mode energy
# ---------------------------------------------------------------------------


def plot_fig_02_matcomp(
    data: dict[str, np.ndarray],
    meta: dict[str, str],
    out_path: Path,
) -> None:
    """Mode energy decomposition and P/Q norm ratios for low-rank matrix completion."""
    steps = data["steps"]
    emp_ib = data["emp_ib"]  # shape (2, N)
    norm_ratios = data["pq_norm_ratios"]  # shape (model_rank, N)
    true_ib_0 = float(meta["true_ib_0"])
    true_ib_1 = float(meta["true_ib_1"])
    num_steps = int(meta["num_steps"])

    fig, axes = nmi.subplots(2, 1, width="1.5", aspect=0.65)
    ax0, ax1 = axes[0], axes[1]

    # --- Panel a: Mode energies ---
    ax0.plot(steps, emp_ib[0], color=nmi.Palette.teal[1], label=r"Est. $\sigma_1$")
    ax0.plot(steps, emp_ib[1], color=nmi.Palette.blue[1], label=r"Est. $\sigma_2$")
    ax0.axhline(
        true_ib_0 / 2.0,
        color=nmi.Palette.teal[1],
        **nmi.scaled_preset(nmi.DASHED_REF),
        label=r"Theor. $\sigma_1^\star$",
    )
    ax0.axhline(
        true_ib_1 / 2.0,
        color=nmi.Palette.blue[1],
        **nmi.scaled_preset(nmi.DASHED_REF),
        label=r"Theor. $\sigma_2^\star$",
    )

    ax0.set_xlim(0, num_steps)
    ax0.set_xlabel("Opt. iteration")
    ax0.set_ylabel(r"Singular value(s)")
    ax0.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0), useMathText=True)
    ax0.legend(ncol=2)

    # --- Panel b: P/Q norm ratios ---
    for j in range(norm_ratios.shape[0]):
        ax1.plot(steps, norm_ratios[j])

    ax1.axhline(
        1.0,
        color="k",
        **nmi.scaled_preset(nmi.DASHED_REF),
        label="Theor. balance (ratio: 1)",
    )

    ax1.set_xlim(0, num_steps)
    ax1.set_xlabel("Opt. iteration")
    ax1.set_ylabel(r"${}^{\|\mathbf{U}_j\|_{2}} \! / \! {}_{\|\mathbf{V}_j\|_{2}}$")
    ax1.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0), useMathText=True)
    ax1.legend()

    nmi.label_panels(axes)

    nmi.savefig(fig, out_path, dpi=OUT_DPI, fmt=OUT_FMT)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Fig 03 — Attention K/Q norm balancing
# ---------------------------------------------------------------------------


def plot_fig_03_attention(
    data: dict[str, np.ndarray],
    meta: dict[str, str],
    out_path: Path,
) -> None:
    """Q/K column-wise norm ratio evolution."""
    steps = data["steps"]
    norm_ratios = data["norm_ratios"]  # shape (N, head_dim)
    head_dim = int(meta["head_dim"])

    fig, ax = nmi.subplots(1, 1, width="1.5", aspect=0.65)

    for j in range(head_dim):
        ax.plot(steps, norm_ratios[:, j])

    ax.axhline(
        1.0,
        color="k",
        **nmi.scaled_preset(nmi.DASHED_REF),
        label="Theor. balance (ratio: 1)",
    )

    ax.set_xlim(steps[0], 4 * steps[-1] / 5)
    ax.set_xlabel("Opt. iteration")
    ax.set_ylabel(r"${}^{\|\mathrm{Q}_{[\cdot,j]}\|_{2}} \! / \! {}_{\|\mathrm{K}_{[\cdot,j]}\|_{2}}$")
    ax.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0), useMathText=True)
    ax.legend()

    nmi.savefig(fig, out_path, dpi=OUT_DPI, fmt=OUT_FMT)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Fig 04 — Spectral-sparse recovery (3 panels)
# ---------------------------------------------------------------------------


def plot_fig_04_spectralsparse(
    data: dict[str, np.ndarray],
    meta: dict[str, str],
    out_path: Path,
) -> None:
    """Two-panel figure: spectral coefficients, function fit."""
    n_modes = int(meta["n_spectral_modes"])

    c_vanilla_ur = nmi.Palette.teal[2]
    c_hadamard = nmi.Palette.cool_warm[3]
    c_true = "black"

    with nmi.context(scale=nmi.ONE_HALF_COL / nmi.DOUBLE_COL / 0.715):
        fig, axes = nmi.subplots(2, 1, width=nmi.DOUBLE_COL * 0.715, aspect=0.45)

        # --- Panel a: Spectral coefficients (stem) ---
        ax1 = axes[0]
        ks = np.arange(n_modes)

        stem_true = ax1.stem(
            ks,
            data["w_true"].ravel(),
            linefmt="-",
            markerfmt="o",
            basefmt="k-",
            label="True",
        )
        stem_true.stemlines.set_color(c_true)
        stem_true.markerline.set_color(c_true)

        stem_vu = ax1.stem(
            ks,
            data["w_naive_ur"].ravel(),
            linefmt="-",
            markerfmt="s",
            basefmt="k-",
            label="Vanilla",
        )
        stem_vu.stemlines.set_color(c_vanilla_ur)
        stem_vu.markerline.set_color(c_vanilla_ur)

        stem_h = ax1.stem(
            ks,
            data["w_hadamard_ur"].ravel(),
            linefmt="--",
            markerfmt="^",
            basefmt="k-",
            label="Designed",
        )
        stem_h.stemlines.set_color(c_hadamard)
        stem_h.markerline.set_color(c_hadamard)

        for line in ax1.lines:
            line.set_linewidth(0.6)
        for coll in ax1.collections:
            coll.set_linewidth(0.6)

        ax1.set_xlabel("Frequency")
        ax1.set_ylabel("Amplitude")
        ax1.set_xlim(0, n_modes)
        ax1.legend(ncol=3)

        # --- Panel b: Function fit ---
        ax2 = axes[1]
        t_grid = data["t_grid"].ravel()

        ax2.plot(t_grid, data["y_plot_true"].ravel(), color=c_true, label="True", lw=1.0)
        ax2.plot(
            t_grid,
            data["y_plot_naive"].ravel(),
            color=c_vanilla_ur,
            label="Vanilla",
        )
        ax2.plot(
            t_grid,
            data["y_plot_pq"].ravel(),
            color=c_hadamard,
            label="Designed",
            linestyle="--",
        )
        ax2.scatter(
            data["t_train"].ravel(),
            data["y_train"].ravel(),
            s=15,
            marker="x",
            color=c_true,
            linewidths=1.0,
            label="Data",
            zorder=5,
        )

        ax2.set_xlabel("$t$")
        ax2.set_ylabel("$y$")
        ax2.set_xlim(0, 1)
        ax2.legend(ncol=4, handler_map=_SCATTER_HANDLER)

        nmi.label_panels(axes)

        nmi.savefig(fig, out_path, dpi=OUT_DPI, fmt=OUT_FMT)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Fig 05 — TV regularization recovery
# ---------------------------------------------------------------------------


def plot_fig_05(
    data: dict[str, np.ndarray],
    meta: dict[str, str],
    out_path: Path,
) -> None:
    """TV-regularized signal recovery."""
    true_signal = data["true_signal"].ravel()
    w_simple = data["w_simple"].ravel()
    w_cumsum = data["w_cumsum"].ravel()
    signal_length = int(meta["signal_length"])

    fig, ax = nmi.subplots(1, 1, width="1.5", aspect=0.5)

    ax.plot(true_signal, color="black", lw=1.5, label="True")
    ax.plot(w_simple, color=nmi.Palette.teal[1], label="Vanilla")
    ax.plot(w_cumsum, color=nmi.Palette.blue[1], label="Designed")

    ax.set_xlim(0, signal_length)
    ax.set_xlabel("t")
    ax.set_ylabel("$y$")
    ax.legend(ncol=3)

    nmi.savefig(fig, out_path, dpi=OUT_DPI, fmt=OUT_FMT)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Hyperbolas gauge — analytic (u, v) parametrisation plot
# ---------------------------------------------------------------------------


def plot_hyperbolas_gauge(out_path: Path) -> None:
    """Two-panel hyperbola curves in the (u, v) parametrisation space."""

    def _draw_hyperbolas(ax, *, gauge: bool = False) -> None:
        """Render hyperbola iso-curves (and optionally the gauge diagonal)."""
        u = np.linspace(0.02, 10.0, 4000)
        levels = np.array([0.45, 0.8, 1.4, 2.3, 3.8, 6.5])

        base_rgb = np.array([int(c, 16) / 255 for c in ("3b", "7d", "d8")])
        shadow_rgb = np.array([0.72, 0.76, 0.82])

        if gauge:
            diag_u = np.linspace(0.0, 10.0, 400)
            ax.plot(
                diag_u,
                diag_u,
                color=(0.25, 0.25, 0.25),
                lw=0.9,
                alpha=0.7,
                linestyle="--",
                zorder=5,
            )
            ax.text(
                5.2,
                5.2,
                r"$\chi(u,v)$",
                rotation=45,
                rotation_mode="anchor",
                ha="left",
                va="bottom",
                color=(0.22, 0.22, 0.22),
                fontsize=6.5,
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": (1, 1, 1, 0.75),
                    "edgecolor": "none",
                },
                zorder=30,
            )

        for i, w in enumerate(levels):
            v = w / u
            mask = (v >= 0.02) & (v <= 10.0)
            uu, vv = u[mask], v[mask]
            if len(uu) < 3:
                continue

            t = i / (len(levels) - 1)
            color = tuple((1 - 0.60 * t) * base_rgb + 0.60 * t * np.ones(3))
            lw = 1.8 - 0.7 * t
            alpha = 0.96 - 0.18 * t

            ax.plot(
                uu + 0.05,
                vv - 0.05,
                color=shadow_rgb,
                lw=lw + 0.4,
                alpha=0.35,
                solid_capstyle="round",
                zorder=1 + i,
            )
            (line,) = ax.plot(
                uu,
                vv,
                color=color,
                lw=lw,
                alpha=alpha,
                solid_capstyle="round",
                zorder=20 + i,
            )
            line.set_path_effects([pe.Stroke(linewidth=lw + 0.35, foreground=(1, 1, 1, 0.40)), pe.Normal()])

        ax.set_xlim(0, 10.0)
        ax.set_ylim(0, 10.0)
        ax.set_xlabel(r"$u$")
        ax.set_ylabel(r"$v$")
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.set_yticks([0, 2, 4, 6, 8, 10])
        ax.set_aspect("equal", adjustable="datalim")

    import logging

    fig, axes = nmi.subplots(1, 2, width="double", aspect=1.0, gridspec_kw={"wspace": 0.05})

    _draw_hyperbolas(axes[0], gauge=False)
    _draw_hyperbolas(axes[1], gauge=True)

    nmi.label_panels(axes)

    mpl_logger = logging.getLogger("matplotlib")
    prev_level = mpl_logger.level
    mpl_logger.setLevel(logging.ERROR)
    nmi.savefig(fig, out_path, dpi=OUT_DPI, fmt=OUT_FMT)
    mpl_logger.setLevel(prev_level)

    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

PLOT_DISPATCH: dict[str, PlotFn] = {
    "fig_01_shallowrelu_*.safetensors": plot_fig_01_shallowrelu,
    "fig_02_matcomp_*.safetensors": plot_fig_02_matcomp,
    "fig_03_attention_*.safetensors": plot_fig_03_attention,
    "fig_04_spectralsparse_*.safetensors": plot_fig_04_spectralsparse,
    "fig_05_tv_*.safetensors": plot_fig_05,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Discover safetensors files and recreate all figures."""
    # Thinner defaults (propagates into nmi.context blocks via _scaled_rcparams)
    nmi.override_base(**{"lines.linewidth": 1.2, "lines.markersize": 4.0})

    nmi.use()

    # Override for non-context figures (use() loads mplstyle which resets to 1.5)
    plt.rcParams["lines.linewidth"] = 1.2
    plt.rcParams["lines.markersize"] = 4.0

    # Thinner dashed reference lines
    nmi.DASHED_REF["linewidth"] = 1.6

    for pattern, plot_fn in PLOT_DISPATCH.items():
        paths = sorted(SAVED_DIR.glob(pattern))
        if not paths:
            print(f"  No files matching {pattern} — skipping")
            continue
        for path in paths:
            print(f"Processing {path.name} ...")
            try:
                data, meta = load_safetensor(path)
                out_path = FIGURES_DIR / f"{path.stem}_nmi.{OUT_FMT}"
                plot_fn(data, meta, out_path)
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                print(f"  ERROR plotting {path.name}: {exc} — skipping")

    # Hyperbolas gauge figure (analytic, no safetensors data)
    out_path = FIGURES_DIR / f"fig_00_hyperbolas_gauge_nmi.{OUT_FMT}"
    print("Processing hyperbolas gauge plot ...")
    plot_hyperbolas_gauge(out_path)


if __name__ == "__main__":
    main()
