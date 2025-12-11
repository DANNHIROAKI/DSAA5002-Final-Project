"""Random seed helpers to keep experiments reproducible."""

from __future__ import annotations

import importlib.util
import os
import random
from typing import Optional

import numpy as np


def _seed_torch(seed: int) -> None:
    """Seed PyTorch RNGs if the library is available.

    This is a no-op when PyTorch is not installed. It also configures cuDNN
    to be deterministic when possible.
    """
    if importlib.util.find_spec("torch") is None:
        return

    import torch  # type: ignore[import]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_global_seed(seed: int = 42, *, deterministic: bool = True) -> None:
    """Seed random number generators across supported libraries.

    Parameters
    ----------
    seed :
        Seed value to apply across libraries. Defaults to 42.
    deterministic :
        When True, also configures deterministic behavior for supported
        backends (currently PyTorch cuDNN). Has no effect when the backend
        is absent.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if deterministic:
        _seed_torch(seed)


def reproducible_numpy_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Return a NumPy Generator seeded for reproducible sampling."""
    return np.random.default_rng(seed)


__all__ = ["set_global_seed", "reproducible_numpy_rng"]
