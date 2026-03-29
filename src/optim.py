"""Optimizer factories."""

# ~~ Imports ~~ ────────────────────────────────────────────────────────────────
from collections.abc import Iterable
from typing import Any

from torch import Tensor
from torch.optim import SGD

# ~~ Exports ~~ ────────────────────────────────────────────────────────────────
__all__: list[str] = [
    "easy_sgd",
]

# ~~ Type Aliases ~~ ───────────────────────────────────────────────────────────
type ParamsT = Iterable[Tensor] | Iterable[dict[str, Any]] | Iterable[tuple[str, Tensor]]


# ~~ Functions ~~ ──────────────────────────────────────────────────────────────
def easy_sgd(
    params: ParamsT,
    lr: float | Tensor = 1e-3,
) -> SGD:
    """Create a simple SGD optimizer with default parameters.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate.

    Returns:
        An SGD optimizer instance.
    """
    return SGD(params=params, lr=lr, momentum=0, weight_decay=0)


# ~~ Main Execution ~~ ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Do nothing
    pass
