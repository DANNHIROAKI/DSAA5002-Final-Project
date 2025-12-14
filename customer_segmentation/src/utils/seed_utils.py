"""Random seed helpers to keep experiments reproducible.

The project has multiple stochastic components:
- train/val/test split,
- clustering initialisation (KMeans/GMM),
- EM initialisation for RAMoE/HyRAMoE.

To make comparisons fair (especially vs. baselines), each experiment should
set a global seed once at the start.

This module provides:
- ``set_global_seed``: seeds Python, NumPy, and (optionally) PyTorch.
- ``reproducible_numpy_rng``: returns a dedicated NumPy Generator.
- ``temp_seed``: context manager for local deterministic blocks.
"""

from __future__ import annotations

import importlib.util
import os
import random
from contextlib import contextmanager
from typing import Iterator, Optional

import numpy as np


def _seed_torch(seed: int, *, deterministic: bool = True) -> None:
    """Seed PyTorch RNGs if PyTorch is available.

    This is a no-op when PyTorch is not installed.

    Parameters
    ----------
    seed:
        Seed value.
    deterministic:
        When True, configure deterministic cuDNN behavior (if applicable).
    """
    if importlib.util.find_spec("torch") is None:
        return

    import torch  # type: ignore

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_global_seed(seed: int = 42, *, deterministic: bool = True) -> None:
    """Seed random number generators across supported libraries.

    Notes
    -----
    - Sets ``PYTHONHASHSEED`` for reproducible hashing.
    - scikit-learn typically relies on NumPy RNG, so NumPy seeding suffices.
    """
    os.environ["PYTHONHASHSEED"] = str(int(seed))

    random.seed(int(seed))
    np.random.seed(int(seed))

    _seed_torch(int(seed), deterministic=deterministic)


def reproducible_numpy_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Return a dedicated NumPy RNG seeded for reproducible sampling."""
    return np.random.default_rng(seed)


@contextmanager
def temp_seed(seed: int, *, deterministic: bool = True) -> Iterator[None]:
    """Temporarily set seeds for a local deterministic code block.

    Restores Python ``random`` and NumPy RNG states afterwards.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()

    set_global_seed(int(seed), deterministic=deterministic)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


__all__ = ["set_global_seed", "reproducible_numpy_rng", "temp_seed"]
