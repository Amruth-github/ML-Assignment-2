from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = PROJECT_ROOT / "hotel_bookings.csv"
TEST_DATA = PROJECT_ROOT / "test_data.csv"
SAVED_MODELS_DIR = PROJECT_ROOT / "model" / "saved_models"
METRICS_FILE = PROJECT_ROOT / "model" / "metrics.csv"

TARGET = "is_canceled"

RANDOM_STATE = 42
TEST_SIZE = 0.2

LEAKAGE_COLUMNS = [
    "reservation_status",
    "reservation_status_date",
    "assigned_room_type",
]

ID_LIKE_COLUMNS = ["agent", "company"]

MISSING_LABEL = "None"

NUMERIC_FEATURES = [
    "lead_time",
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "total_nights",
    "total_guests",
    "is_family",
    "has_agent",
    "has_company",
    "adr_per_guest",
    "prev_cancel_ratio",
]

CATEGORICAL_FEATURES = [
    "hotel",
    "arrival_date_month",
    "arrival_weekday",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
    "agent",
    "company",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MIN_CATEGORY_FREQUENCY = 50

KNN_SUBSAMPLE_SIZE = 15_000

MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "kNN",
    "naive_bayes": "Naive Bayes (Gaussian)",
    "random_forest": "Random Forest (Ensemble)",
    "gradient_boosting": "Gradient Boosting (Ensemble)",
}
