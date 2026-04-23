from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9


def mul(a: pd.Series, b: pd.Series) -> pd.Series:
    return a * b


def div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / (b + EPS)


def sub(a: pd.Series, b: pd.Series) -> pd.Series:
    return a - b


def absdiff(a: pd.Series, b: pd.Series) -> pd.Series:
    return np.abs(a - b)
