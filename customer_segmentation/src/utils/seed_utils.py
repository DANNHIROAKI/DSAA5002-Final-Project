"""Random seed helpers to keep experiments reproducible.

Upgrades for the new methodology:
- Provide a context manager `temp_seed` for local deterministic blocks.
- Keep `set_global_seed` and `reproducible_numpy_rng` fully backward compatible.

Call set_global_seed() once near the beginning of each experiment script
(e.g. run_baselines/run_rajc) so results are comparable across runs.
"""

from __future__ import annotations

import importlib.util
import os
import random
from contextlib import contextmanager
from typing import Optional, Iterator

import numpy as np


def _seed_torch(seed: int, *, deterministic: bool = True) -> None:
    """Seed PyTorch RNGs if the library is available.

    This is a no-op when PyTorch is not installed.

    Parameters
    ----------
    seed :
        Seed value for PyTorch's RNGs.
    deterministic :
        When True, configure cuDNN deterministic behavior if applicable.
    """
    if importlib.util.find_spec("torch") is None:
        return

    import torch  # type: ignore[import]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_global_seed(seed: int = 42, *, deterministic: bool = True) -> None:
    """Seed random number generators across supported libraries.

    Parameters
    ----------
    seed :
        Seed value to apply across libraries. Defaults to 42.
    deterministic :
        When True (default), also configures deterministic behavior for supported
        backends (currently PyTorch cuDNN). Has no effect when the backend
        is absent.

    Notes
    -----
    - This sets PYTHONHASHSEED to make Python's hash-based operations reproducible.
    - scikit-learn uses NumPy's RNG, so seeding NumPy is usually sufficient
      for reproducible clustering / model training.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    _seed_torch(seed, deterministic=deterministic)


def reproducible_numpy_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Return a NumPy Generator seeded for reproducible sampling.

    Parameters
    ----------
    seed :
        Optional seed for the Generator. If None, NumPy will draw from entropy
        sources but without affecting the global RNG.

    Returns
    -------
    np.random.Generator
        A dedicated RNG instance.
    """
    return np.random.default_rng(seed)


@contextmanager
def temp_seed(seed: int, *, deterministic: bool = True) -> Iterator[None]:
    """Temporarily set seeds for a local deterministic block.

    This is useful when you want a specific randomized procedure to be repeatable
    without permanently perturbing the global RNG states.

    Notes
    -----
    - Restores Python `random` and NumPy states after the context.
    - For PyTorch, we set the seed but do not attempt to restore previous states.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()

    set_global_seed(seed, deterministic=deterministic)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


__all__ = ["set_global_seed", "reproducible_numpy_rng", "temp_seed"]
