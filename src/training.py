"""Training loops for spectral-sparse experiments."""

# ~~ Imports ~~ ────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F
from models import CumsumSignalRegressor
from models import Hadamard1DLinear
from models import Simple1DLinear
from models import total_variation
from optim import easy_sgd
from tqdm.auto import trange

# ~~ Exports ~~ ────────────────────────────────────────────────────────────────
__all__: list[str] = ["train_on_spectral_sparse_experiment", "train_on_step_signal_experiment"]


# ~~ Functions ~~ ──────────────────────────────────────────────────────────────
def train_on_spectral_sparse_experiment(
    experiment_name: str,
    model: Simple1DLinear | Hadamard1DLinear,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    num_steps: int,
    learning_rate: float,
    ellone_regularization_strength: float,
    device: torch.device,
    log_only_final: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Train a 1D linear model with optional L1 regularization."""
    if not isinstance(model, (Simple1DLinear, Hadamard1DLinear)):
        raise TypeError(f"model must be Simple1DLinear or Hadamard1DLinear, got {type(model)}")

    print(f"Running experiment '{experiment_name}'...")

    optimizer = easy_sgd(model.parameters(), lr=learning_rate)

    n_logged: int = 1 if log_only_final else num_steps
    train_losses: torch.Tensor = torch.empty(n_logged, device=device)
    test_losses: torch.Tensor = torch.empty(n_logged, device=device)
    log_idx: int = 0

    model.train()
    for iteration in trange(num_steps):
        mse_loss: torch.Tensor = F.mse_loss(model(x_train), y_train)
        train_loss: torch.Tensor = mse_loss
        if ellone_regularization_strength > 0.0:
            train_loss = mse_loss + ellone_regularization_strength * model.w.abs().sum()

        optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        optimizer.step()

        if not log_only_final or iteration >= num_steps - 1:
            with torch.no_grad():
                test_loss: torch.Tensor = F.mse_loss(input=model(x_test), target=y_test)
            train_losses[log_idx] = mse_loss.detach()
            test_losses[log_idx] = test_loss.detach()
            log_idx += 1

    print(f"Experiment '{experiment_name}' completed.\n")

    return train_losses.cpu(), test_losses.cpu()


def train_on_step_signal_experiment(
    experiment_name: str,
    model: Simple1DLinear | CumsumSignalRegressor,
    ameas: torch.Tensor,
    ymeas: torch.Tensor,
    num_steps: int,
    learning_rate: float,
    device: torch.device,
    print_every: int | None = None,
) -> torch.Tensor:
    """Train a model on the step signal regression task. Returns train losses."""
    if not isinstance(model, (Simple1DLinear, CumsumSignalRegressor)):
        raise TypeError(f"model must be Simple1DLinear or CumsumSignalRegressor, got {type(model)}")

    print(f"Running experiment '{experiment_name}'...")

    optimizer = easy_sgd(model.parameters(), lr=learning_rate)

    train_losses: torch.Tensor = torch.empty(num_steps, device=device)

    model.train()
    for iteration in trange(num_steps):
        mse_loss: torch.Tensor = F.mse_loss(model(ameas), ymeas)

        optimizer.zero_grad(set_to_none=True)
        mse_loss.backward()
        optimizer.step()

        train_losses[iteration] = mse_loss.detach()

        if print_every is not None and (iteration + 1 >= num_steps or (iteration + 1) % print_every == 0):
            with torch.no_grad():
                tv_val: float = total_variation(model.w).item()
            print(f"Iteration {iteration + 1}/{num_steps}, MSE Loss: {mse_loss.item():.6f}, TV: {tv_val:.6f}")

    print(f"Experiment '{experiment_name}' completed.\n")

    return train_losses.cpu()


# ~~ Main Execution ~~ ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Do nothing
    pass
