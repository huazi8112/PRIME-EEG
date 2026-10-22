"""Numerical routines curated from the frozen research implementation."""


from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np


def persistent(mask: np.ndarray, count: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if count <= 1:
        return mask
    cumulative = np.cumsum(mask.astype(np.int32))
    totals = cumulative.copy()
    totals[count:] -= cumulative[:-count]
    return mask & (totals >= count)
