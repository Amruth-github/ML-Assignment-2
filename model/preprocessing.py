from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer


def load_and_clean(path) -> pd.DataFrame:
    raise NotImplementedError


def build_preprocessor() -> ColumnTransformer:
    raise NotImplementedError
