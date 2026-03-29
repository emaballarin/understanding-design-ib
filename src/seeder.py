#!/usr/bin/env python3
"""Global random seeder with deterministic sub-seeding across PyTorch, NumPy, and Python."""

import os
import random
from collections.abc import Generator
from contextlib import contextmanager
from warnings import warn

import numpy as np
import torch as th


__all__: list[str] = ["GlobalRandomSeeder", "seed_everything"]

MIN_SEED: int = 0
MAX_SEED: int = 4294967295  # NumPy's 32-bit constraint


class GlobalRandomSeeder:
    """
    Deterministic seed manager for multi-backend RNG synchronization.

    Provides counter-based seed derivation from a master seed, named tagging
    for checkpointing, LIFO-based save/restore navigation, and scoped context
    managers for temporary seed states.
    """

    __slots__: tuple[str, ...] = (
        "_counter",
        "_cuda_deterministic_algorithms",
        "_cudnn_benchmark",
        "_cudnn_deterministic",
        "_lifo_stack",
        "_master_seed",
        "_tag_order",
        "_tags",
    )

    def __init__(
        self,
        seed: int,
        *,
        cudnn_deterministic: bool = True,
        cudnn_benchmark: bool = False,
        cuda_deterministic_algorithms: bool = True,
    ) -> None:
        """
        Initialize the seeder with a master seed and optional CUDA determinism flags.

        Args:
            seed: Master seed for all RNG backends.
            cudnn_deterministic: Set torch.backends.cudnn.deterministic.
            cudnn_benchmark: Set torch.backends.cudnn.benchmark.
            cuda_deterministic_algorithms: Set torch.use_deterministic_algorithms.
        """
        self._master_seed: int = self._validate_seed(seed)
        self._counter: int = 0

        self._cudnn_deterministic: bool = cudnn_deterministic
        self._cudnn_benchmark: bool = cudnn_benchmark
        self._cuda_deterministic_algorithms: bool = cuda_deterministic_algorithms

        self._tags: dict[str, int] = {}
        self._tag_order: list[str] = []
        self._lifo_stack: list[int] = []

        self._sync()

    @staticmethod
    def _validate_seed(seed: int) -> int:
        if not (MIN_SEED <= seed <= MAX_SEED):
            warn(f"Seed {seed} out of bounds [{MIN_SEED}, {MAX_SEED}], clamping to 0.")
            return 0
        return seed

    def _current_seed(self) -> int:
        return self._validate_seed(self._master_seed + self._counter)

    def _sync(self) -> None:
        """Synchronize all RNG backends to current seed."""
        seed = self._current_seed()

        os.environ["EB_GLOBAL_SEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)

        if th.cuda.is_available():
            th.cuda.manual_seed_all(seed)

        th.backends.cudnn.deterministic = self._cudnn_deterministic
        th.backends.cudnn.benchmark = self._cudnn_benchmark

        if self._cuda_deterministic_algorithms:
            th.use_deterministic_algorithms(True, warn_only=True)

    def seed(self, value: int) -> int:
        """
        Set the seed to an explicit value and sync all backends.

        Args:
            value: The seed value to set.

        Returns:
            The validated seed that was set.
        """
        self._master_seed = self._validate_seed(value)
        self._counter = 0
        self._sync()
        return self._current_seed()

    def get_seed(self) -> int:
        """Return current seed without advancing or syncing."""
        return self._current_seed()

    def next_seed(self) -> int:
        """
        Increment counter, sync all backends, and return the new seed.

        Returns:
            The new seed as an integer.
        """
        self._counter += 1
        self._sync()
        return self._current_seed()

    def tag(self, name: str) -> None:
        """
        Save current counter under a named tag.

        Args:
            name: Tag name. Warns and generates unique key on duplicate.
        """
        if name in self._tags:
            warn(message=f"Tag '{name}' already exists, overwriting.")
            self._tag_order.remove(name)

        self._tags[name] = self._counter
        self._tag_order.append(name)

    def save(self, name: str) -> None:
        """
        Push current counter to LIFO stack, then restore to named tag and sync.

        Args:
            name: Tag name to restore to.

        Raises:
            KeyError: If tag does not exist.
        """
        if name not in self._tags:
            raise KeyError(f"Tag '{name}' does not exist.")

        self._lifo_stack.append(self._counter)
        self._counter = self._tags[name]
        self._sync()

    def restore(self) -> None:
        """
        Pop counter from LIFO stack and sync all backends.

        Raises:
            IndexError: If LIFO stack is empty.
        """
        if not self._lifo_stack:
            raise IndexError("Cannot restore: LIFO stack is empty.")

        self._counter = self._lifo_stack.pop()
        self._sync()

    @contextmanager
    def with_seed(self, value: int) -> Generator[int]:
        """
        Context manager for scoped execution with a specific seed.

        On entry: pushes current counter, sets seed to value, syncs.
        On exit: restores previous counter, syncs.

        Args:
            value: Seed value for the scoped context.

        Yields:
            The seed value used within the context.
        """
        saved_master: int = self._master_seed
        saved_counter: int = self._counter

        self._master_seed = self._validate_seed(value)
        self._counter = 0
        self._sync()

        try:
            yield self._current_seed()
        finally:
            self._master_seed = saved_master
            self._counter = saved_counter
            self._sync()

    @contextmanager
    def with_tag(self, name: str) -> Generator[int]:
        """
        Context manager for scoped execution rewound to a named tag.

        On entry: pushes current counter, restores to tag, syncs.
        On exit: restores previous counter, syncs.

        Args:
            name: Tag name to rewind to.

        Yields:
            The seed value at the tagged state.

        Raises:
            KeyError: If tag does not exist.
        """
        if name not in self._tags:
            raise KeyError(f"Tag '{name}' does not exist.")

        saved_counter: int = self._counter
        self._counter = self._tags[name]
        self._sync()

        try:
            yield self._current_seed()
        finally:
            self._counter = saved_counter
            self._sync()

    @property
    def master_seed(self) -> int:
        """The original master seed."""
        return self._master_seed

    @property
    def counter(self) -> int:
        """Current counter offset from master seed."""
        return self._counter

    @property
    def tags(self) -> dict[str, int]:
        """Copy of current tags mapping."""
        return dict(self._tags)

    @property
    def stack_depth(self) -> int:
        """Current depth of the LIFO stack."""
        return len(self._lifo_stack)


def seed_everything(seed: int | None = None) -> int:
    """
    Set seed for all RNG backends with full CUDA determinism.

    Convenience function for one-shot global seeding. For stateful seed
    management, use GlobalRandomSeeder directly.

    Args:
        seed: Seed value. Defaults to 0 if None.

    Returns:
        The seed that was set.
    """
    if seed is None:
        seed = 0
    seeder: GlobalRandomSeeder = GlobalRandomSeeder(seed)
    return seeder.get_seed()
