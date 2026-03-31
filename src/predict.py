import joblib
import pandas as pd
from src.config import MODEL_PATH


def load_pipeline():
    return joblib.load(MODEL_PATH)


def predict_single(input_data: dict) -> dict:
    model = load_pipeline()
    input_df = pd.DataFrame([input_data])

    raw_pred = str(model.predict(input_df)[0]).strip().lower()
    label_map = {"good": 0, "bad": 1}
    prediction = label_map.get(raw_pred, 0)

    proba = model.predict_proba(input_df)[0]
    classes = [str(c).strip().lower() for c in model.classes_]
    bad_idx = classes.index("bad") if "bad" in classes else 1
    probability_bad_risk = float(proba[bad_idx])

    return {
        "prediction": prediction,
        "prediction_label": raw_pred,
        "probability_bad_risk": probability_bad_risk
    }
