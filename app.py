import io

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from model.config import (
    FEATURES,
    METRICS_FILE,
    MODEL_NAMES,
    SAVED_MODELS_DIR,
    TARGET,
    TEST_DATA,
)
from model.evaluation import METRIC_COLUMNS, confusion, evaluate
from model.preprocessing import load_and_clean

st.set_page_config(
    page_title="Hotel Booking Cancellation Predictor",
    layout="centered",
)

CLASS_LABELS = ["Stayed", "Cancelled"]
TEAL = "#2f6f6b"
CLAY = "#c1666b"


@st.cache_resource(show_spinner=False)
def load_model(key: str):
    return joblib.load(SAVED_MODELS_DIR / f"{key}.joblib")


@st.cache_data(show_spinner=False)
def clean_bundled():
    return load_and_clean(TEST_DATA)


@st.cache_data(show_spinner=False)
def clean_uploaded(payload: bytes):
    return load_and_clean(io.BytesIO(payload))


@st.cache_data(show_spinner=False)
def read_comparison():
    if not METRICS_FILE.exists():
        return None
    return pd.read_csv(METRICS_FILE).set_index("ML Model Name")


def confusion_figure(matrix):
    shares = matrix / matrix.sum(axis=1, keepdims=True)
    annotations = np.array(
        [[f"{count:,}\n{share:.0%}" for count, share in zip(*rows)]
         for rows in zip(matrix, shares)]
    )

    figure, axis = plt.subplots(figsize=(4.6, 3.4))
    figure.patch.set_alpha(0)
    sns.heatmap(
        shares,
        annot=annotations,
        fmt="",
        cmap=sns.light_palette(TEAL, as_cmap=True),
        cbar=False,
        square=True,
        linewidths=3,
        linecolor="#faf8f5",
        vmin=0,
        vmax=1,
        annot_kws={"fontsize": 11},
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
        ax=axis,
    )
    axis.set_xlabel("Predicted", labelpad=8, fontsize=10)
    axis.set_ylabel("Actual", labelpad=8, fontsize=10)
    axis.tick_params(length=0, labelsize=10)
    return figure


available_models = {
    key: name
    for key, name in MODEL_NAMES.items()
    if (SAVED_MODELS_DIR / f"{key}.joblib").exists()
}

st.title("Will this booking be cancelled?")
st.markdown(
    "Six classification models trained on 86,000 hotel bookings, "
    "scored live on held-out test data."
)

if not available_models:
    st.error(
        "No trained models found in model/saved_models/. "
        "Run model/train_models.ipynb to generate them."
    )
    st.stop()

with st.sidebar:
    st.subheader("Controls")

    use_upload = st.radio(
        "Data to score",
        options=[False, True],
        format_func=lambda flag: "Upload my own CSV" if flag else "Bundled test split",
    )
    upload = st.file_uploader("Test data (CSV)", type="csv") if use_upload else None

    model_key = st.selectbox(
        "Model",
        options=list(available_models),
        format_func=lambda key: available_models[key],
    )

    st.caption(
        "Changing the data changes what is scored. The chosen model is already "
        "trained and is never refitted here."
    )
    st.divider()
    st.caption("Hotel Booking Demand dataset, Antonio, Almeida and Nunes (2019).")

if use_upload:
    if upload is None:
        st.info(
            "Choose a labelled CSV in the sidebar. The repository's test_data.csv "
            "works, and so does any file with the same columns."
        )
        st.stop()
    try:
        data = clean_uploaded(upload.getvalue())
    except Exception as error:
        st.error(
            "That CSV could not be processed. It needs the same columns as the "
            "repository's test_data.csv or the original hotel_bookings.csv. "
            f"({type(error).__name__}: {error})"
        )
        st.stop()
    source = upload.name
else:
    if not TEST_DATA.exists():
        st.error("test_data.csv is missing. Switch to upload, or run the training notebook.")
        st.stop()
    data = clean_bundled()
    source = f"{TEST_DATA.name} (bundled)"

if TARGET not in data.columns:
    st.error(f"That file has no '{TARGET}' column, so metrics cannot be computed.")
    st.stop()

missing = [column for column in FEATURES if column not in data.columns]
if missing:
    st.error(f"That file is missing {len(missing)} required column(s): {', '.join(missing[:6])}")
    st.stop()

if data.empty:
    st.error("No usable rows were found in that file.")
    st.stop()

X, y = data[FEATURES], data[TARGET]
model = load_model(model_key)

with st.spinner("Scoring bookings..."):
    scores = evaluate(model, X, y)
    matrix = confusion(model, X, y)

baseline = max(y.mean(), 1 - y.mean())

selected, everything = st.tabs([available_models[model_key], "All six models"])

with selected:
    st.caption(
        f"{len(data):,} bookings from {source} — {y.mean():.1%} were cancelled, "
        f"so guessing the majority class alone scores {baseline:.3f} accuracy."
    )

    with st.container(border=True):
        for row in (METRIC_COLUMNS[:3], METRIC_COLUMNS[3:]):
            for column, metric in zip(st.columns(3), row):
                column.metric(
                    metric,
                    f"{scores[metric]:.3f}",
                    delta=f"{scores[metric] - baseline:+.3f} vs baseline"
                    if metric == "Accuracy"
                    else None,
                )

    st.pyplot(confusion_figure(matrix))
    plt.close("all")
    st.caption("Cells show booking counts, with each row's share of actual outcomes.")

with everything:
    comparison = read_comparison()
    if comparison is None:
        st.info("Run the training notebook to generate model/metrics.csv.")
    else:
        st.caption("Recorded on the held-out test split during training.")
        st.dataframe(
            comparison.style.format("{:.4f}").background_gradient(
                cmap=sns.light_palette(TEAL, as_cmap=True), axis=0
            )
        )
        best = comparison["MCC"].idxmax()
        st.markdown(
            f"**{best}** leads on Matthews correlation coefficient, the most balanced "
            "single measure when only a quarter of bookings are cancelled."
        )
