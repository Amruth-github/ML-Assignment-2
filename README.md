# ML Assignment 2 — Hotel Booking Cancellation Prediction

Machine Learning Assignment 2, M.Tech (AIML/DSE), BITS Pilani WILP.

## a. Problem statement

_TBD_

## b. Dataset description

_TBD — source, size, feature list, target definition, class balance, cleaning decisions._

## c. GitHub Repository Link

_TBD_

## Live Streamlit App Link

_TBD_

## d. Models used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| kNN | | | | | | |
| Naive Bayes | | | | | | |
| Random Forest (Ensemble) | | | | | | |
| Gradient Boosting (Ensemble) | | | | | | |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | |
| Decision Tree | |
| kNN | |
| Naive Bayes | |
| Random Forest (Ensemble) | |
| Gradient Boosting (Ensemble) | |
| **Overall Winner** | |

## Project structure

```
Assignment_2/
├── app.py                     Streamlit application
├── requirements.txt
├── README.md
├── hotel_bookings.csv         Full raw dataset
├── test_data.csv              Held-out test split used by the app
└── model/
    ├── config.py              Shared paths, column groups, constants
    ├── preprocessing.py       Cleaning and the ColumnTransformer
    ├── evaluation.py          The six evaluation metrics
    ├── train_models.ipynb     EDA, training, evaluation and export
    ├── metrics.csv            Generated comparison table
    └── saved_models/          Generated *.joblib pipelines
```

Code shared by training and inference lives in the modules; anything only
training needs lives in the notebook. The Streamlit app therefore imports
`config`, `preprocessing` and `evaluation`, and never the training code.

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
