# ML Assignment 2 — Hotel Booking Cancellation Prediction

Machine Learning Assignment 2, M.Tech (AIML/DSE), BITS Pilani WILP.

Six classification models are trained on the Hotel Booking Demand dataset to predict
whether a booking will be cancelled, and served through an interactive Streamlit app.

## a. Problem statement

Hotel booking cancellations are expensive. A room released close to the arrival date is
often impossible to resell, so cancellations distort revenue forecasting, staffing and
overbooking decisions. A hotel that can identify high-risk bookings early can respond
by confirming them, adjusting overbooking limits, or targeting retention offers.

This is treated as a **binary classification** problem: using only the attributes of a
booking that are known at the time it is made, predict whether it will eventually be
cancelled. The target is `is_canceled`, where 1 means the booking was cancelled and 0
means the guest stayed. Six models are trained on identical data and compared across
six evaluation metrics.

## b. Dataset description

| Property | Value |
|---|---|
| Source | Hotel Booking Demand, Kaggle (originally Antonio, Almeida and Nunes, 2019) |
| Task | Binary classification |
| Target | `is_canceled` (1 = cancelled, 0 = stayed) |
| Raw size | 119,390 rows, 32 columns |
| Size after cleaning | 85,984 rows, 36 features |
| Feature types | 24 numeric, 12 categorical |
| Class balance (raw) | 62.96% stayed, 37.04% cancelled |
| Class balance (cleaned) | 72.42% stayed, 27.58% cancelled |
| Train / test split | Stratified 80/20 — 68,787 train, 17,197 test |
| Encoded width | 219 columns after one-hot encoding |
| Assignment minimums | 36 features (min 12) and 85,984 instances (min 500) — both satisfied |

### Data preparation decisions

**Target leakage removed.** Three columns were dropped because they reveal the outcome:
`reservation_status` maps one-to-one onto the target (`Canceled` and `No-Show` rows are
always cancellations, `Check-Out` rows never are), `reservation_status_date` is only
known once a booking has resolved, and `assigned_room_type` is settled at check-in,
after cancellation is already known. Keeping any of them produces near-perfect but
meaningless scores.

**Duplicates removed.** 31,994 raw rows are exact duplicates. A random split would place
identical rows on both sides, letting the tree and nearest-neighbour models be scored on
rows they had memorised. De-duplication runs *after* feature engineering, because
reducing `agent` and `company` to categories merges rows that previously differed only
by identifier. 33,406 rows were removed in total, which shifts the cancellation rate
from 37.04% to 27.58%.

**Ambiguous rows kept.** 281 pairs of bookings are identical across every feature but
have opposite outcomes. These are retained because they represent irreducible error
rather than redundancy — and they are why no model here should be expected to approach
100% accuracy.

**Missing values.** `children` (4 rows) filled with zero; `country` (488 rows) labelled
`Unknown`; `agent` (16,340 rows) and `company` (112,593 rows) labelled `None`, with
additional `has_agent` and `has_company` indicator flags.

**High-cardinality identifiers kept.** `agent` and `company` were retained as categorical
features rather than discarded. Cancellation rates across the ten busiest agents range
from 0.07 to 0.73 in the raw data, and from 0.06 to 0.41 after de-duplication, so the
signal survives cleaning. Both columns are known at booking time, which makes them
legitimate predictors rather than leakage.

**Rare categories grouped.** Categories occurring fewer than 50 times collapse into a
single bucket. This stops the 177 `country` levels from dominating the feature space, and
lets the app handle a country it has never seen before.

**Outliers handled.** `adr` (average daily rate) runs from -6.38 to 5,400 in the raw data
and was clipped to 0–1,000. 180 bookings recording zero adults, children and babies were
removed.

**Engineered features.** `total_nights`, `total_guests`, `is_family`, `has_agent`,
`has_company`, `adr_per_guest`, `prev_cancel_ratio` (the share of a guest's earlier
bookings that were cancelled), and `arrival_weekday`, reconstructed from the separate
year, month and day columns because cancellation rates vary with the day of arrival —
from 0.32 on Sundays to 0.41 on Thursdays in the raw data, and from 0.24 on Tuesdays to
0.31 on Fridays after de-duplication.

**Pipeline.** Numeric columns are median-imputed and standardised; categorical columns
are mode-imputed and one-hot encoded. The transformer is fitted on the training split
only, so imputation medians, scaler statistics and category vocabularies never see the
test data.

## c. GitHub Repository Link

https://github.com/Amruth-github/ML-Assignment-2

## Live Streamlit App Link

https://hotel-booking-cancellation-assignment.streamlit.app/

## d. Models used

All six models were trained on the same features, through the same preprocessing
pipeline, and evaluated at the default 0.5 decision threshold on the same held-out test
split. Differences in the table are therefore attributable to the algorithms themselves.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8012 | 0.8587 | 0.6830 | 0.5210 | 0.5911 | 0.4703 |
| Decision Tree | 0.8269 | 0.8798 | 0.7139 | 0.6218 | 0.6646 | 0.5511 |
| kNN | 0.7906 | 0.8177 | 0.6774 | 0.4596 | 0.5477 | 0.4312 |
| Naive Bayes (Gaussian) | 0.5198 | 0.7105 | 0.3544 | 0.9022 | 0.5089 | 0.2700 |
| Random Forest (Ensemble) | 0.8346 | 0.9020 | 0.8017 | 0.5317 | 0.6394 | 0.5567 |
| Gradient Boosting (Ensemble) | 0.8478 | 0.9157 | 0.7484 | 0.6749 | 0.7098 | 0.6084 |

Because the cleaned data is imbalanced at roughly 72/28, predicting the majority class
for every booking already scores **0.7242 accuracy**. Accuracy alone is therefore
misleading here, and MCC and AUC are the more informative measures.

### Model configuration

| Model | Configuration |
|---|---|
| Logistic Regression | Up to 3,000 iterations on standardised features |
| Decision Tree | Max depth 16, minimum 10 samples per leaf |
| kNN | 25 neighbours, distance-weighted, fitted on a stratified 15,000-row subsample |
| Naive Bayes | Gaussian |
| Random Forest | 200 trees, max depth 20, minimum 2 samples per leaf |
| Gradient Boosting | Histogram-based, 300 iterations, learning rate 0.1 |

kNN is fitted on a subsample because it stores its training data inside the saved model
file, and the random forest is depth-capped, so that both stay small enough to deploy on
Streamlit Community Cloud.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Reaches 0.8012 accuracy, only about 8 points above the 0.7242 majority-class baseline, with MCC 0.4703. Its AUC of 0.8587 shows it ranks bookings by risk reasonably well, but recall of 0.5210 means it misses almost half of all cancellations. A single linear boundary cannot represent interactions such as deposit type combined with lead time. Valuable mainly as a fast, interpretable baseline. |
| Decision Tree | Improves clearly on the linear model at 0.8269 accuracy and 0.5511 MCC, because it captures the non-linear interactions logistic regression cannot. It beats the random forest on both F1 (0.6646) and recall (0.6218), since a single tree is more willing to predict the minority class. Depth had to be capped at 16 to control overfitting. The most directly interpretable of the strong models. |
| kNN | The weakest discriminative model at 0.7906 accuracy and 0.4312 MCC, with the lowest recall of any model (0.4596). Distances lose meaning in a 219-dimensional, mostly one-hot space — the curse of dimensionality striking the algorithm's core assumption. It is also the slowest at prediction time and the only model needing a subsample to stay deployable. Poorly matched to high-dimensional categorical data. |
| Naive Bayes (Gaussian) | The most unusual result: at 0.5198 accuracy it does worse than always predicting the majority class (0.7242), yet it has the highest recall of any model at 0.9022. It labels almost every booking a likely cancellation, giving precision of just 0.3544 and the lowest MCC at 0.2700. Its conditional-independence assumption is badly violated by 219 correlated one-hot columns. Only preferable where a missed cancellation costs far more than a false alarm. |
| Random Forest (Ensemble) | The strongest of the five models the assignment requires, with MCC 0.5567, AUC 0.9020, and the best precision of any model at 0.8017. Averaging 200 trees removes most of the single tree's variance, visible in the improved AUC and precision. It is conservative about positive predictions, with recall (0.5317) below the single tree's, so ranking it above the decision tree depends on preferring MCC and precision to recall and F1. Also the largest saved model at roughly 18 MB. |
| Gradient Boosting (Ensemble) | The best model overall, leading on accuracy (0.8478), AUC (0.9157), F1 (0.7098) and MCC (0.6084). Growing trees sequentially so each corrects its predecessors' errors captures the lead time, deposit type and market segment interactions more effectively than independently grown trees. It pairs strong precision (0.7484) with much better recall (0.6749) than the random forest, which is why its F1 and MCC lead. Compact on disk at under 1 MB. |
| **Overall Winner** | **Gradient Boosting**, leading on four of six metrics including MCC (0.6084) and AUC (0.9157) — the two most appropriate for imbalanced data. Among the five models the assignment lists explicitly, **Random Forest** wins on MCC (0.5567) and AUC (0.9020). Every model except Naive Bayes is conservative about predicting cancellations, a direct consequence of only 27.58% of bookings being cancelled combined with the default 0.5 threshold. |

## Streamlit app features

| Required feature | Implementation |
|---|---|
| Dataset upload option (CSV) | Sidebar switch between the bundled test split and an uploaded CSV; uploads pass through the identical cleaning pipeline used in training |
| Model selection dropdown | Sidebar dropdown over all six trained models, each loaded on demand and cached |
| Display of evaluation metrics | Accuracy, AUC, Precision, Recall, F1 and MCC as metric cards, with accuracy compared against the majority-class baseline |
| Confusion matrix | Heatmap annotated with both booking counts and each row's share of actual outcomes |

Results for all six models on the test data are shown together in a second tab.
`sample_upload.csv` is a 2,000-row stratified sample of the test split, provided for
trying the upload path.

## Project structure

```
Assignment_2/
├── app.py                        Streamlit application
├── requirements.txt              Pinned dependency versions
├── README.md
├── hotel_bookings.csv            Full raw dataset
├── test_data.csv                 Held-out test split (17,197 rows)
├── sample_upload.csv             2,000-row sample for demonstrating upload
├── .streamlit/config.toml        Application theme
├── .devcontainer/               Dev container definition for BITS Virtual Lab
└── model/
    ├── config.py                 Shared paths, column groups, constants
    ├── preprocessing.py          Cleaning and the ColumnTransformer
    ├── evaluation.py             The six evaluation metrics
    ├── train_models.ipynb        EDA, training, evaluation and export
    ├── metrics.csv               Generated comparison table
    └── saved_models/             Six fitted pipelines (*.joblib)
```

Code shared by training and inference lives in the modules; anything only training needs
lives in the notebook. The app therefore imports `config`, `preprocessing` and
`evaluation`, and never the training code — which is also why the app and the notebook
report identical metrics.

## Reproducing

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run `model/train_models.ipynb` top to bottom to regenerate `test_data.csv`,
`model/metrics.csv` and `model/saved_models/`, then launch the app:

```bash
streamlit run app.py
```

Running the notebook additionally requires Jupyter, which is not in `requirements.txt`
because the deployed app does not need it:

```bash
pip install jupyter
```

## Environment

Python 3.11 with scikit-learn 1.9.0, pandas 3.0.5, numpy 2.4.6, streamlit 1.60.0,
matplotlib 3.11.1, seaborn 0.13.2 and joblib 1.5.3.
