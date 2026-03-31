import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.config import(
    MODEL_DIR,
    MODEL_PATH,
    METRICS_PATH,
    FEATURE_COLUMNS_PATH,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES
)


from src.data_utils import load_data
from src.preprocess import build_preprocessor
from src.evaluate import evaluate_model

def train_and_select_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    required_coulumns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
    missing_columns = [col for col in required_coulumns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns in dataset: {missing_columns}")
    
    x = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


    preprocessor = build_preprocessor()

    candidate_models = {
        "logistic regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_split=10, random_state=RANDOM_STATE)
    }

    results = {}
    best_model_name = None
    best_pipeline = None
    best_score = -1

    for model_name, estimator in candidate_models.items():
        pipeline = Pipeline(
            steps = [
                ("preprocessor", preprocessor),
                ("model", estimator)
            ]
        )

        pipeline.fit(x_train, y_train)
        metrics = evaluate_model(pipeline, x_test, y_test)

        results[model_name] = metrics

        if metrics['f1_score'] > best_score:
            best_score = metrics['f1_score']
            best_model_name = model_name
            best_pipeline = pipeline

    joblib.dump(best_pipeline, MODEL_PATH)

    payload = {
        'best_model': best_model_name,
        'selection_metric': 'f1_score', 
        'results':results
    }

    with open(METRICS_PATH, 'w') as file:
        json.dump(payload, file, indent = 2)

    with open(FEATURE_COLUMNS_PATH, 'w') as file:
        json.dump({
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES
        }, file, indent = 2)

    print(f"Best model: {best_model_name}")
    print(f"Saved pipeline to: {MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")

if __name__ == "__main__":
    train_and_select_model()


