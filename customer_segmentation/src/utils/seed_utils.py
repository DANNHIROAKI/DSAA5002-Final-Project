"""Random seed helpers."""
import random
import numpy as np


def set_global_seed(seed: int = 42):
    """Set seeds for random, numpy, and (optionally) other frameworks."""
    random.seed(seed)
    np.random.seed(seed)
