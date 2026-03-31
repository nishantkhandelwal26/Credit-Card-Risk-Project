from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"

# Default path from legacy code; fallback to the data file in this repo
DATA_PATH = DATA_DIR / "german_credit_data.csv"
MODEL_PATH = MODEL_DIR / "credit_risk_pipeline.pkl"
METRICS_PATH = MODEL_DIR / "model_metrics.json"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.json"

TARGET_COLUMN = "credit_risk"

RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMERIC_FEATURES = [
    "duration",
    "amount",
    "age"
]

CATEGORICAL_FEATURES = [
    "status",
    "credit_history",
    "purpose",
    "savings",
    "employment_duration",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "present_residence",
    "property",
    "other_installment_plans",
    "housing",
    "number_credits",
    "job",
    "people_liable",
    "telephone",
    "foreign_worker"
]