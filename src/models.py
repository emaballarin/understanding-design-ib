#!/usr/bin/env python3
"""Model definitions."""

# ~~ Imports ~~ ────────────────────────────────────────────────────────────────
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter

# ~~ Exports ~~ ────────────────────────────────────────────────────────────────
__all__: list[str] = [
    "BiaslessReluNet",
    "CumsumSignalRegressor",
    "Hadamard1DLinear",
    "MatrixFactorization",
    "Simple1DLinear",
    "SingleHeadAttention",
    "total_variation",
    "unbalance_qk",
    "unbalance_vw",
]


# ~~ Classes ~~ ────────────────────────────────────────────────────────────────
class MatrixFactorization(nn.Module):
    """Low-rank matrix factorization module: M = P @ Q."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        init_std: float = 0.1,
    ) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if out_features <= 0:
            raise ValueError(f"out_features must be positive, got {out_features}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.rank: int = rank
        self.p: Parameter = Parameter(data=torch.empty(out_features, rank, device=device, dtype=dtype))
        self.q: Parameter = Parameter(data=torch.empty(rank, in_features, device=device, dtype=dtype))
        self.reset_parameters(std=init_std)

    def reset_parameters(self, std: float = 0.1) -> None:
        self.p.data.normal_(mean=0.0, std=std * 1.1)
        self.q.data.normal_(mean=0.0, std=std)

    def forward(self) -> Tensor:
        return self.p @ self.q


class SingleHeadAttention(nn.Module):
    def __init__(
        self,
        d_emb: int,
        d_head: int,
        wo_eye: bool = False,
        pool_seq: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        init_std: float = 0.1,
    ) -> None:
        super().__init__()
        if d_emb <= 0 or d_head <= 0:
            raise ValueError(f"d_emb and d_head must be positive; got d_emb={d_emb}, d_head={d_head}")
        if init_std <= 0.0:
            raise ValueError(f"init_std must be positive; got init_std={init_std}")
        self.d_emb: int = d_emb
        self.d_head: int = d_head
        self.wo_eye: bool = wo_eye
        self.pool_seq: bool = pool_seq
        self._scale: float = 1.0 / math.sqrt(d_head)

        self.wq = Parameter(torch.empty(d_emb, d_head, device=device, dtype=dtype))
        self.wk = Parameter(torch.empty(d_emb, d_head, device=device, dtype=dtype))
        self.wv = Parameter(torch.empty(d_emb, d_head, device=device, dtype=dtype))

        if not self.wo_eye:
            self.wo = Parameter(torch.empty(d_head, d_emb, device=device, dtype=dtype))

        self.reset_parameters(std=init_std)

    @property
    def d_out(self) -> int:
        return self.d_head if self.wo_eye else self.d_emb

    def reset_parameters(self, std: float = 0.1) -> None:
        self.wq.data.normal_(mean=0.0, std=std)
        self.wk.data.normal_(mean=0.0, std=std)
        self.wv.data.normal_(mean=0.0, std=std)
        if not self.wo_eye:
            self.wo.data.normal_(mean=0.0, std=std)

    def rescale_parameters(self, scale: float) -> None:
        self.wq.data.mul_(scale)
        self.wk.data.mul_(scale)
        self.wv.data.mul_(scale)
        if not self.wo_eye:
            self.wo.data.mul_(scale)

    def forward(self, x: Tensor) -> Tensor:
        q: Tensor = x @ self.wq
        k: Tensor = x @ self.wk
        v: Tensor = x @ self.wv

        h: Tensor = torch.nn.functional.scaled_dot_product_attention(query=q, key=k, value=v, scale=self._scale)

        if not self.wo_eye:
            h = h @ self.wo

        if self.pool_seq:
            h = h.mean(dim=-2)

        return h


class Simple1DLinear(nn.Module):
    """Linear layer with learnable 1D weight vector: y = x @ w."""

    def __init__(
        self,
        features: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        init_std: float = 0.1,
    ) -> None:
        super().__init__()
        if features <= 0:
            raise ValueError(f"features must be positive, got {features}")
        if init_std <= 0.0:
            raise ValueError(f"init_std must be positive, got {init_std}")
        self.w: Parameter = Parameter(torch.empty(features, device=device, dtype=dtype))
        self.reset_parameters(std=init_std)

    def reset_parameters(self, std: float = 0.1) -> None:
        self.w.data.normal_(mean=0.0, std=std)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.w


class Hadamard1DLinear(nn.Module):
    """Linear layer with Hadamard-factorized weights: y = x @ (p * q)."""

    def __init__(
        self,
        features: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        init_std: float = 0.1,
    ) -> None:
        super().__init__()
        if features <= 0:
            raise ValueError(f"features must be positive, got {features}")
        if init_std <= 0.0:
            raise ValueError(f"init_std must be positive, got {init_std}")
        self.p: Parameter = Parameter(torch.empty(features, device=device, dtype=dtype))
        self.q: Parameter = Parameter(torch.empty(features, device=device, dtype=dtype))
        self.reset_parameters(std=init_std)

    def reset_parameters(self, std: float = 0.1) -> None:
        # Use std**0.5 so that Var(p*q) = std², matching Simple1DLinear variance.
        self.p.data.normal_(mean=0.0, std=std**0.5)
        self.q.data.normal_(mean=0.0, std=std**0.5)

    @property
    def w(self) -> Tensor:
        return self.p * self.q

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.w


class CumsumSignalRegressor(nn.Module):
    """Signal regressor with cumulative-sum reparameterization: w = cumsum(w1 * w2)."""

    def __init__(
        self,
        features: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        init_std: float = 0.1,
    ) -> None:
        super().__init__()
        if features <= 0:
            raise ValueError(f"features must be positive, got {features}")
        if init_std <= 0.0:
            raise ValueError(f"init_std must be positive, got {init_std}")
        self.w1: Parameter = Parameter(torch.empty(features, device=device, dtype=dtype))
        self.w2: Parameter = Parameter(torch.empty(features, device=device, dtype=dtype))
        self.reset_parameters(std=init_std)

    def reset_parameters(self, std: float = 0.1) -> None:
        self.w1.data.normal_(mean=0.0, std=std)
        self.w2.data.normal_(mean=0.0, std=std)

    @property
    def w(self) -> Tensor:
        return torch.cumsum(self.w1 * self.w2, dim=0)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.w


class BiaslessReluNet(nn.Module):
    """Simple feedforward network with ReLU activations and no biases."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        depth: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        init_std: float = 0.1,
    ) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if out_features <= 0:
            raise ValueError(f"out_features must be positive, got {out_features}")
        if hidden_features <= 0:
            raise ValueError(f"hidden_features must be positive, got {hidden_features}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if init_std <= 0.0:
            raise ValueError(f"init_std must be positive, got {init_std}")
        self._depth: int = depth
        self._init_std: float = init_std
        self.layers: nn.ModuleList = nn.ModuleList()
        for i in range(depth):
            in_f = in_features if i == 0 else hidden_features
            out_f = out_features if i == depth - 1 else hidden_features
            self.layers.append(nn.Linear(in_f, out_f, bias=False, device=device, dtype=dtype))
        self.reset_parameters(std=init_std)

    def reset_parameters(self, std: float = 0.1) -> None:
        for layer in self.layers:
            layer.weight.data.normal_(mean=0.0, std=std)  # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self._depth - 1:
                x = F.relu(x)
        return x


# ~~ Functions ~~ ──────────────────────────────────────────────────────────────
@torch.no_grad()
def unbalance_qk(model: SingleHeadAttention, log_scale: float) -> SingleHeadAttention:
    """Scale Q/K weights inversely to create artificial norm imbalance."""
    s: Tensor = torch.exp(
        torch.linspace(-log_scale, log_scale, model.d_head, device=model.wq.device, dtype=model.wq.dtype)
    )
    model.wq.mul_(s)
    model.wk.div_(s)
    return model


@torch.no_grad()
def unbalance_vw(model: BiaslessReluNet, log_scale: float) -> BiaslessReluNet:
    """Scale layer-0 rows / layer-1 columns inversely to create artificial norm imbalance."""
    if len(model.layers) != 2:
        raise ValueError("Model must have exactly 2 layers.")
    h: int = model.layers[0].weight.shape[0]  # type: ignore[union-attr]
    s: Tensor = torch.exp(
        torch.linspace(
            -log_scale, log_scale, h, device=model.layers[0].weight.device, dtype=model.layers[0].weight.dtype
        )  # type: ignore[union-attr]
    )
    model.layers[0].weight.data.mul_(s.unsqueeze(1))  # type: ignore[union-attr]
    model.layers[1].weight.data.mul_(1.0 / s.unsqueeze(0))  # type: ignore[union-attr]
    return model


def total_variation(x: torch.Tensor) -> torch.Tensor:
    """Compute the total variation of a 1D signal."""
    return x.diff().abs().sum()


# ~~ Main Execution ~~ ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Do nothing
    pass
