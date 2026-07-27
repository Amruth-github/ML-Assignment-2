from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model.config import (
    CATEGORICAL_FEATURES,
    FEATURES,
    ID_LIKE_COLUMNS,
    LEAKAGE_COLUMNS,
    MIN_CATEGORY_FREQUENCY,
    MISSING_LABEL,
    NUMERIC_FEATURES,
    RAW_DATA,
    TARGET,
)

ADR_CEILING = 1000.0


def _as_label(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        values = values.astype("Int64")
    return values.astype("string").fillna(MISSING_LABEL).astype(str)


def _arrival_weekday(df: pd.DataFrame) -> pd.Series:
    month = pd.to_datetime(df["arrival_date_month"], format="%B", errors="coerce").dt.month
    arrival = pd.to_datetime(
        {
            "year": df["arrival_date_year"],
            "month": month,
            "day": df["arrival_date_day_of_month"],
        },
        errors="coerce",
    )
    return arrival.dt.day_name().fillna(MISSING_LABEL).astype(str)


def load_and_clean(path=RAW_DATA) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in LEAKAGE_COLUMNS if c in df.columns])

    if "children" in df.columns:
        df["children"] = df["children"].fillna(0).astype(int)
    if "country" in df.columns:
        df["country"] = df["country"].fillna(MISSING_LABEL)

    for column, flag in zip(ID_LIKE_COLUMNS, ("has_agent", "has_company")):
        if column in df.columns:
            df[column] = _as_label(df[column])
            df[flag] = (df[column] != MISSING_LABEL).astype(int)

    df["arrival_weekday"] = _arrival_weekday(df)
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["is_family"] = ((df["children"] + df["babies"]) > 0).astype(int)

    df = df[df["total_guests"] > 0]
    df["adr"] = df["adr"].clip(lower=0.0, upper=ADR_CEILING)
    df["adr_per_guest"] = df["adr"] / df["total_guests"]

    prior = df["previous_cancellations"] + df["previous_bookings_not_canceled"]
    df["prev_cancel_ratio"] = (
        df["previous_cancellations"] / prior.replace(0, np.nan)
    ).fillna(0.0)

    keep = [c for c in FEATURES if c in df.columns]
    if TARGET in df.columns:
        keep.append(TARGET)
    return df[keep].drop_duplicates().reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=MIN_CATEGORY_FREQUENCY,
                    sparse_output=False,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )
