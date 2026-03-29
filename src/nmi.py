"""Nature Machine Intelligence figure style helpers.

Provides colour palettes, figure-sizing utilities, panel-label helpers,
and marker/line presets matching NMI publication conventions.

Scale compensation
------------------
If a figure will be rescaled in LaTeX (e.g.
``\\includegraphics[width=0.5\\textwidth]{fig.pdf}``), pass ``scale=0.5``
to ``use()``, ``context()``, or ``subplots()``.  All size-dependent
quantities (fonts, line widths, marker sizes, tick lengths, paddings)
are pre-divided by the scale factor so that *after* LaTeX shrinks the
figure, they land at the intended physical size.

Usage
-----
>>> import nmi
>>> nmi.use(scale=0.5)  # everything pre-compensated for 50 % scaling
>>> fig, axes = nmi.subplots(1, 3)  # full-width, 3 panels
>>> nmi.label_panels(axes)  # bold a, b, c labels
"""

import contextlib
from collections.abc import Generator
from pathlib import Path
from typing import Any
from typing import ClassVar
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = [
    "DASHED_REF",
    "DOUBLE_COL",
    "ONE_HALF_COL",
    "SINGLE_COL",
    "Palette",
    "context",
    "figsize",
    "label_panels",
    "override_base",
    "savefig",
    "scaled_preset",
    "subplots",
    "use",
]

# ---------------------------------------------------------------------------
# Style file path
# ---------------------------------------------------------------------------

_STYLE_PATH = Path(__file__).with_name("nmi.mplstyle")

# ---------------------------------------------------------------------------
# Base sizes (nominal values from nmi.mplstyle, used for scale compensation)
# ---------------------------------------------------------------------------

_BASE: dict[str, float] = {
    # Fonts (pt)
    "font.size": 7.0,
    "axes.titlesize": 8.0,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.0,
    # Axes geometry (pt)
    "axes.linewidth": 0.8,
    "axes.titlepad": 6.0,
    "axes.labelpad": 4.0,
    # Tick geometry (pt)
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "xtick.major.pad": 3.0,
    "ytick.major.pad": 3.0,
    # Lines & markers
    "lines.linewidth": 1.5,
    "lines.markersize": 5.0,
    "lines.markeredgewidth": 0.0,
    # Legend geometry
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.4,
    "legend.columnspacing": 1.0,
    "legend.labelspacing": 0.4,
    # Error bars
    "errorbar.capsize": 2.0,
    # Patches & hatches
    "patch.linewidth": 0.8,
    "hatch.linewidth": 0.5,
}


def override_base(**kwargs: float) -> None:
    """Override base size values used for scale compensation."""
    _BASE.update(kwargs)


def _scaled_rcparams(scale: float) -> dict[str, float]:
    """Return rcParams dict with all sizes divided by *scale*."""
    inv = 1.0 / scale
    return {k: v * inv for k, v in _BASE.items()}


# ---------------------------------------------------------------------------
# Current scale tracking (for functions with hardcoded sizes)
# ---------------------------------------------------------------------------

_current_scale: float = 1.0


def _s(value: float) -> float:
    """Compensate a hardcoded size by the active scale factor."""
    return value / _current_scale


# ---------------------------------------------------------------------------
# Style activation
# ---------------------------------------------------------------------------


def use(scale: float = 1.0) -> None:
    """Activate the NMI matplotlib style globally.

    Parameters
    ----------
    scale : float
        LaTeX scaling factor (e.g. 0.5 if the figure will be included
        at half-width).  All size-dependent rcParams are pre-compensated
        so fonts/lines appear at the correct physical size after scaling.
    """
    global _current_scale
    _current_scale = scale
    plt.style.use(str(_STYLE_PATH))
    if scale != 1.0:
        mpl.rcParams.update(_scaled_rcparams(scale))


@contextlib.contextmanager
def context(scale: float = 1.0) -> Generator[None]:
    """Temporarily activate the NMI style within a ``with`` block.

    Parameters
    ----------
    scale : float
        LaTeX scaling factor (see ``use``).

    Usage::

        with nmi.context(scale=0.5):
            fig, ax = plt.subplots()
            ...
    """
    global _current_scale
    prev_scale = _current_scale
    _current_scale = scale
    try:
        with plt.style.context(str(_STYLE_PATH)):
            if scale != 1.0:
                mpl.rcParams.update(_scaled_rcparams(scale))
            yield
    finally:
        _current_scale = prev_scale


# ---------------------------------------------------------------------------
# Nature figure dimensions (in inches, converted from mm)
# ---------------------------------------------------------------------------

MM_TO_IN = 1.0 / 25.4

SINGLE_COL = 88.0 * MM_TO_IN  # ~3.46 in
ONE_HALF_COL = 120.0 * MM_TO_IN  # ~4.72 in
DOUBLE_COL = 180.0 * MM_TO_IN  # ~7.08 in

ColumnWidth = Literal["single", "1.5", "double"]

_COL_MAP: dict[ColumnWidth, float] = {
    "single": SINGLE_COL,
    "1.5": ONE_HALF_COL,
    "double": DOUBLE_COL,
}


def figsize(
    width: ColumnWidth | float = "double",
    aspect: float = 0.65,
    *,
    nrows: int = 1,
    ncols: int = 1,
) -> tuple[float, float]:
    """Return (width, height) in inches for a given Nature column width.

    Parameters
    ----------
    width : {"single", "1.5", "double"} or float
        Column width preset, or an explicit width in inches.
    aspect : float
        Height / width ratio, applied per panel.
    nrows, ncols : int
        Grid dimensions.  *aspect* is applied to a single panel
        (width/ncols × aspect) and then multiplied by *nrows*.
    """
    w = _COL_MAP[width] if isinstance(width, str) else width
    panel_w = w / ncols
    return (w, panel_w * aspect * nrows)


def subplots(
    nrows: int = 1,
    ncols: int = 1,
    *,
    width: ColumnWidth | float = "double",
    aspect: float = 0.65,
    scale: float | None = None,
    **kwargs: Any,
) -> tuple[Figure, np.ndarray | Axes]:
    """Create a figure + axes sized for Nature publication.

    Parameters
    ----------
    nrows, ncols : int
        Grid dimensions.
    width : {"single", "1.5", "double"} or float
        Column width preset, or explicit width in inches.
    aspect : float
        Height / width ratio.
    scale : float, optional
        LaTeX scaling factor.  If given, overrides (and updates) the
        global scale set by ``use()`` / ``context()``.  All rcParams
        are re-compensated accordingly.
    **kwargs
        Forwarded to ``plt.subplots``.
    """
    if scale is not None:
        global _current_scale
        _current_scale = scale
        mpl.rcParams.update(_scaled_rcparams(scale))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize(width, aspect, nrows=nrows, ncols=ncols),
        **kwargs,
    )
    return fig, axes


# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------


class Palette:
    """Named colour sequences for NMI-style figures."""

    # 3-stop teal gradient (light -> dark)
    teal: ClassVar[list[str]] = ["#8cc5a0", "#3fb8a0", "#1a7a70"]

    # 3-stop blue gradient (light -> dark)
    blue: ClassVar[list[str]] = ["#6aaddb", "#3b7dd8", "#1a2f6e"]

    # 4-stop cool-to-warm, for moderate sweeps
    cool_warm: ClassVar[list[str]] = ["#2b4a9c", "#7c6cb8", "#8b5e5e", "#d08a40"]


# ---------------------------------------------------------------------------
# Line presets
# ---------------------------------------------------------------------------

# Dashed reference line (analytic baselines, asymptotic limits, …)
DASHED_REF: dict[str, Any] = {
    "linestyle": (0, (3, 2)),
    "linewidth": 2.0,
    "alpha": 0.85,
}


def scaled_preset(preset: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a line-style preset with linewidth compensated.

    Use this when drawing lines manually with ``DASHED_REF``
    so that their widths match the active scale::

        ax.axhline(0.5, color="k", **nmi.scaled_preset(nmi.DASHED_REF))
    """
    out = dict(preset)
    if "linewidth" in out:
        out["linewidth"] = out["linewidth"] / _current_scale
    return out


# ---------------------------------------------------------------------------
# Panel labels
# ---------------------------------------------------------------------------


def label_panels(
    axes: np.ndarray | list[Axes] | Axes,
    *,
    labels: str | list[str] | None = None,
    start: str = "a",
    x: float = -0.12,
    y: float = 1.08,
    fontsize: float = 10,
    fontweight: str = "bold",
) -> None:
    """Add bold lowercase panel labels (a, b, c, ...) to axes.

    Font size is automatically compensated by the active scale factor.

    Parameters
    ----------
    axes : array-like of Axes
        Axes produced by ``plt.subplots`` or ``nmi.subplots``.
    labels : str or list[str], optional
        Explicit labels.  Overrides *start* if given.
    start : str
        First label character when *labels* is None (default ``"a"``).
    x, y : float
        Label position in axes-fraction coordinates.
    fontsize, fontweight : formatting overrides.
        *fontsize* is the **target** size in pt (i.e. what you want after
        LaTeX scaling); it is automatically divided by the active scale.
    """
    flat = np.atleast_1d(np.asarray(axes, dtype=object)).ravel()
    if labels is None:
        labels = [chr(ord(start) + i) for i in range(len(flat))]
    elif isinstance(labels, str):
        labels = list(labels)

    for ax, lab in zip(flat, labels):
        ax.text(
            x,
            y,
            lab,
            transform=ax.transAxes,
            fontsize=_s(fontsize),
            fontweight=fontweight,
            va="top",
            ha="right",
        )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def savefig(
    fig: Figure,
    path: str | Path,
    *,
    dpi: int = 300,
    fmt: str | None = None,
) -> None:
    """Save figure with Nature-compliant defaults.

    For vector output, pass ``fmt="pdf"`` or use a ``.pdf`` extension;
    *dpi* is then ignored by matplotlib.

    Parameters
    ----------
    fig : Figure
    path : str or Path
        Output path.  Extension determines format unless *fmt* is given.
    dpi : int
        Raster resolution (Nature requires >= 300).
    fmt : str, optional
        Override format (e.g. ``"pdf"``, ``"png"``, ``"tiff"``).
    """
    fig.savefig(
        path,
        dpi=dpi,
        format=fmt,
        bbox_inches="tight",
        pad_inches=0.05,
    )
