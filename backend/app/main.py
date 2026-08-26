from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.schemas import HealthData
from app.ml.predictor import predict_risk

from app.database import Base, engine, get_db
from app.models import PredictionHistory


# ============================================================
# DATABASE
# ============================================================

Base.metadata.create_all(bind=engine)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI Health Risk Prediction API",
    description="AI-powered health risk prediction and analytics system",
    version="1.0.0"
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

# Frontend is currently running on:
# http://127.0.0.1:5500
#
# Also allow localhost because browsers can treat
# localhost and 127.0.0.1 as different origins.

ALLOWED_ORIGINS = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():
    return {
        "message": "AI Health Risk Prediction API is running"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict(
    data: HealthData,
    db: Session = Depends(get_db)
):

    # Run machine-learning prediction
    result = predict_risk(
        data.model_dump()
    )

    # Create database record
    prediction_record = PredictionHistory(
        age=data.age,
        sex=data.sex,
        cp=data.cp,
        trestbps=data.trestbps,
        chol=data.chol,
        fbs=data.fbs,
        restecg=data.restecg,
        thalach=data.thalach,
        exang=data.exang,
        oldpeak=data.oldpeak,
        slope=data.slope,
        ca=data.ca,
        thal=data.thal,

        prediction=result["prediction"],
        risk_probability=result["risk_probability"],
        risk_percentage=result["risk_percentage"],
        risk_level=result["risk_level"]
    )

    # Save prediction to database
    db.add(prediction_record)
    db.commit()
    db.refresh(prediction_record)

    return result


# ============================================================
# PREDICTION HISTORY
# ============================================================

@app.get("/history")
def get_prediction_history(
    db: Session = Depends(get_db)
):

    records = (
        db.query(PredictionHistory)
        .order_by(PredictionHistory.id.desc())
        .all()
    )

    return [
        {
            "id": record.id,
            "age": record.age,
            "prediction": record.prediction,
            "risk_probability": record.risk_probability,
            "risk_percentage": record.risk_percentage,
            "risk_level": record.risk_level,
            "created_at": record.created_at
        }
        for record in records
    ]