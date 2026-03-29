"""General-purpose utilities: config tagging, device helpers, and precision setup."""

# ~~ Imports ~~ ────────────────────────────────────────────────────────────────
import dataclasses
import hashlib
import json
import os
import sys
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from typing import TextIO

import torch

# ~~ Exports ~~ ────────────────────────────────────────────────────────────────
__all__: list[str] = [
    "config_tag",
    "resolve_device",
    "suppress_std",
    "torch_set_hiperf_precision",
]


# ~~ Config tagging ~~ ─────────────────────────────────────────────────────────

_FIELD_ABBREV: dict[str, str] = {
    "seed": "s",
    "num_steps": "T",
    "learning_rate": "lr",
    "batch_size": "bs",
    "n_size": "n",
    "m_size": "m",
    "model_rank": "r",
    "observed_fraction": "of",
    "batching_ratio": "br",
    "true_sigma": "sig",
    "emb_dim": "ed",
    "head_dim": "hd",
    "output_noise_std": "ons",
    "n_spectral_modes": "nsm",
    "n_spectral_modes_nonzero": "nnz",
    "train_samples": "ntr",
    "test_samples": "nte",
    "weight_init_scale": "wis",
    "ellone_regularization_strength": "l1",
    "signal_length": "slen",
    "observed_measurements": "om",
    "learning_rate_simple": "lrs",
    "learning_rate_cumsum": "lrc",
    "weight_init_scale_simple": "wiss",
    "weight_init_scale_cumsum": "wisc",
    "sequence_length": "seqlen",
    "teacher_init_std": "tis",
    "student_init_std": "sis",
    "discard_first_points": "dfp",
    "log_every": "le",
    "spectral_magnitude_range": "smr",
    "signal_segments": "ss",
}


def _fmt_value(v: object) -> str:
    """Format a config value compactly for filename use."""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, tuple):
        return "-".join(_fmt_value(x) for x in v)
    return str(v)


def config_tag(cfg: object, *, exclude: tuple[str, ...] = ("device",)) -> str:
    """Return a ``{hash8}[_{diff}]`` tag for deterministic filename use.

    The hash covers all fields except those in *exclude*. The diff suffix
    shows all fields that differ from the dataclass defaults, using
    abbreviated field names.
    """
    fields = {k: v for k, v in dataclasses.asdict(cfg).items() if k not in exclude}
    blob = json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str)
    h = hashlib.sha256(blob.encode()).hexdigest()[:8]

    defaults = type(cfg)()
    diff_tokens: list[str] = []
    for f in dataclasses.fields(cfg):
        if f.name in exclude:
            continue
        cur = getattr(cfg, f.name)
        default = getattr(defaults, f.name)
        if cur != default:
            abbrev = _FIELD_ABBREV.get(f.name, f.name)
            diff_tokens.append(f"{abbrev}{_fmt_value(cur)}")

    parts = [h]
    parts.extend(diff_tokens)
    return "_".join(parts)


# ~~ Functions ~~ ──────────────────────────────────────────────────────────────
def resolve_device(device_str: str = "auto") -> torch.device:
    """Return a ``torch.device`` from a string, treating ``"auto"`` as CUDA-if-available."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


@contextmanager
def suppress_std(which: str = "all") -> Generator[None, Any]:
    """Context manager to suppress stdout, stderr, or both.

    Args:
        which: One of ``"none"``, ``"out"``, ``"err"``, ``"all"``.
    """
    if which not in ("none", "out", "err", "all"):
        raise ValueError("`which` must be either: 'none', 'out', 'err', 'all'")

    with open(file=os.devnull, mode="w") as devnull:
        if which in ("out", "all"):
            old_stdout: TextIO = sys.stdout
            sys.stdout = devnull
        if which in ("err", "all"):
            old_stderr: TextIO = sys.stderr
            sys.stderr = devnull

        try:
            yield
        finally:
            if which in ("out", "all"):
                sys.stdout = old_stdout  # type: ignore[possibly-undefined]
            if which in ("err", "all"):
                sys.stderr = old_stderr  # type: ignore[possibly-undefined]


def torch_set_hiperf_precision(newapi: bool = False, aggressive: bool = False, quiet: bool = False) -> None:
    """Configure PyTorch for high-performance reduced-precision computation.

    Args:
        newapi: Use the newer ``fp32_precision`` API instead of legacy flags.
        aggressive: Additionally allow FP16 accumulation and reduced-precision reductions.
        quiet: Suppress all stdout/stderr output from the configuration.
    """
    with suppress_std(which="all" if quiet else "none"):
        torch.backends.cudnn.benchmark = True
        if newapi:
            torch.backends.fp32_precision = "tf32"  # type: ignore[attr-defined]
            torch.backends.cudnn.fp32_precision = "tf32"  # type: ignore[attr-defined]
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            torch.backends.cudnn.conv.fp32_precision = "tf32"  # type: ignore[attr-defined]
            torch.backends.cudnn.rnn.fp32_precision = "tf32"  # type: ignore[attr-defined]
        else:
            torch.set_float32_matmul_precision(precision="high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if aggressive:
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True


# ~~ Classes ~~ ────────────────────────────────────────────────────────────────

# ~~ Main Execution ~~ ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Do nothing
    pass
