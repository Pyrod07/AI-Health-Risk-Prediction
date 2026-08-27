import os

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.database import Base, engine, get_db
from app.ml.predictor import predict_risk
from app.models import PredictionHistory
from app.schemas import HealthData


Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Health Risk Prediction API",
    description="AI-powered health risk prediction and analytics system",
    version="1.0.0",
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://127.0.0.1:5500,http://localhost:5500",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "AI Health Risk Prediction API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: HealthData, db: Session = Depends(get_db)):
    result = predict_risk(data.model_dump())

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
        risk_level=result["risk_level"],
    )

    db.add(prediction_record)
    db.commit()
    db.refresh(prediction_record)

    return result


@app.get("/history")
def get_prediction_history(db: Session = Depends(get_db)):
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
            "created_at": record.created_at,
        }
        for record in records
    ]
import os

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.database import Base, engine, get_db
from app.ml.predictor import predict_risk
from app.models import PredictionHistory
from app.schemas import HealthData


Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Health Risk Prediction API",
    description="AI-powered health risk prediction and analytics system",
    version="1.0.0",
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://127.0.0.1:5500,http://localhost:5500",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "AI Health Risk Prediction API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: HealthData, db: Session = Depends(get_db)):
    result = predict_risk(data.model_dump())

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
        risk_level=result["risk_level"],
    )

    db.add(prediction_record)
    db.commit()
    db.refresh(prediction_record)

    return result


@app.get("/history")
def get_prediction_history(db: Session = Depends(get_db)):
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
            "created_at": record.created_at,
        }
        for record in records
    ]