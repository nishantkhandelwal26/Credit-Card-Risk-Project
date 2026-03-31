from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR/"data"
MODEL_DIR = BASE_DIR/"model"

DATA_PATH = DATA_DIR/"german_credit_data.csv"


TARGET_COLUMN = "credit_risk"
RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMERIC_FEATURES = [
    "duration",
    "amount",
    "age",
    "present_residence",
    "number_credits",
    "people_liable"
]

CATEGORICAL_FEATURES = [
    "status",
    "credit_history",
    "purpose",
    "savings",
    "employment_duration",
    "personal_status_sex",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker"
]