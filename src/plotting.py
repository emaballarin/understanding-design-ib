"""Plotting utilities for spectral-sparse experiments."""

# ~~ Imports ~~ ────────────────────────────────────────────────────────────────
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import Cycler
from matplotlib.rcsetup import cycler

# ~~ Exports ~~ ────────────────────────────────────────────────────────────────
__all__: list[str] = [
    "custom_plot_setup",
    "petroff_2021_color",
    "petroff_2021_cycler",
    "plot_out",
    "set_petroff_2021_colors",
]

# ~~ Constants ~~ ──────────────────────────────────────────────────────────────
petroff_2021_color: list[str] = [
    # After: M. A. Petroff, "Accessible Color Sequences for Data Visualization", 2021
    "#3f90da",
    "#ffa90e",
    "#bd1f01",
    "#94a4a2",
    "#832db6",
    "#a96b59",
    "#e76300",
    "#b9ac70",
    "#717581",
    "#92dadd",
]

petroff_2021_cycler: Cycler[Any, Any] = cycler(color=petroff_2021_color)


_BASE_FONT_SIZE: float | None = None


# ~~ Functions ~~ ──────────────────────────────────────────────────────────────
def custom_plot_setup(usetex: bool = True) -> None:
    """Apply the project's default matplotlib style (ggplot-based, enlarged font)."""
    global _BASE_FONT_SIZE
    plt.rcParams["text.usetex"] = usetex
    plt.style.use(style="ggplot")
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["grid.color"] = "gainsboro"
    plt.rcParams["axes.formatter.useoffset"] = False
    if _BASE_FONT_SIZE is None:
        _BASE_FONT_SIZE = plt.rcParams["font.size"]
    plt.rcParams["font.size"] = _BASE_FONT_SIZE * 1.2


def set_petroff_2021_colors() -> None:
    """Set the Petroff 2021 accessible colour cycle as the default axes cycle."""
    mpl.rcParams["axes.prop_cycle"] = petroff_2021_cycler


def plot_out(save_path: str | None = None) -> None:
    """Save figure to *save_path* or show interactively."""
    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
    else:
        plt.show()


# ~~ Main Execution ~~ ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Do nothing
    pass
