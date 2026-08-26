import os
import joblib
import pandas as pd


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(__file__)
            )
        )
    ),
    "ml",
    "model",
    "heart_disease_model.pkl"
)


# ============================================================
# FEATURE ORDER
# ============================================================

FEATURE_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"ML model not found at: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


model = load_model()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_risk(data: dict):

    # Create DataFrame using the exact feature order
    input_data = pd.DataFrame(
        [data],
        columns=FEATURE_NAMES
    )

    # Model prediction
    prediction = model.predict(input_data)[0]

    # Probability for class 1
    probability = model.predict_proba(input_data)[0][1]

    # Convert probability to percentage
    risk_percentage = round(
        float(probability) * 100,
        2
    )

    # Risk classification
    if risk_percentage < 30:

        risk_level = "Low"

    elif risk_percentage < 70:

        risk_level = "Moderate"

    else:

        risk_level = "High"

    return {
        "prediction": int(prediction),
        "risk_probability": round(
            float(probability),
            4
        ),
        "risk_percentage": risk_percentage,
        "risk_level": risk_level
    }