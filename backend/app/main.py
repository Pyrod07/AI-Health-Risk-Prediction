from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

import os
import time
import uuid
import logging
import json
from datetime import datetime

from dotenv import load_dotenv

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.schemas import HealthData
from app.ml.predictor import predict_risk

from app.database import Base, engine, get_db
from app.models import PredictionHistory


# ============================================================
# 1. ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

ENVIRONMENT = os.getenv(
    "ENVIRONMENT",
    "development"
)

FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "http://localhost:5173"
)


# ============================================================
# 2. STRUCTURED LOGGER
# ============================================================

logger = logging.getLogger("health-risk-api")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)


def log_json(
    level,
    message,
    request_id=None,
    **kwargs
):
    """
    Create structured JSON logs.
    """

    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "level": level,
        "logger": "health-risk-api",
        "message": message
    }

    if request_id:
        log_data["request_id"] = request_id

    log_data.update(kwargs)

    print(
        json.dumps(
            log_data,
            default=str
        )
    )


# ============================================================
# 3. DATABASE
# ============================================================

Base.metadata.create_all(
    bind=engine
)


# ============================================================
# 4. RATE LIMITER
# ============================================================

limiter = Limiter(
    key_func=get_remote_address
)


# ============================================================
# 5. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI Health Risk Prediction API",
    description="AI-powered health risk prediction and analytics system",
    version="1.0.0"
)


# ============================================================
# 6. RATE LIMIT EXCEPTION HANDLER
# ============================================================

app.state.limiter = limiter

app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)


# ============================================================
# 7. CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        FRONTEND_URL,
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],

    allow_credentials=True,

    allow_methods=[
        "GET",
        "POST",
        "OPTIONS"
    ],

    allow_headers=[
        "Content-Type",
        "Authorization"
    ]
)


# ============================================================
# 8. REQUEST LOGGING + REQUEST ID
# ============================================================

@app.middleware("http")
async def request_logging_middleware(
    request: Request,
    call_next
):

    request_id = str(
        uuid.uuid4()
    )

    request.state.request_id = request_id

    start_time = time.time()

    try:

        response = await call_next(
            request
        )

        duration_ms = round(
            (time.time() - start_time) * 1000,
            2
        )

        response.headers[
            "X-Request-ID"
        ] = request_id

        log_json(
            "INFO",
            "Request completed",

            request_id=request_id,

            method=request.method,

            path=request.url.path,

            status_code=response.status_code,

            duration_ms=duration_ms
        )

        return response

    except Exception as exc:

        duration_ms = round(
            (time.time() - start_time) * 1000,
            2
        )

        log_json(
            "ERROR",
            "Unhandled request exception",

            request_id=request_id,

            method=request.method,

            path=request.url.path,

            duration_ms=duration_ms,

            error_type=type(exc).__name__,

            error=str(exc)
        )

        raise


# ============================================================
# 9. SECURITY HEADERS
# ============================================================

@app.middleware("http")
async def security_headers(
    request: Request,
    call_next
):

    response = await call_next(
        request
    )

    response.headers[
        "X-Content-Type-Options"
    ] = "nosniff"

    response.headers[
        "X-Frame-Options"
    ] = "DENY"

    response.headers[
        "Referrer-Policy"
    ] = "strict-origin-when-cross-origin"

    response.headers[
        "X-XSS-Protection"
    ] = "0"

    return response


# ============================================================
# 10. GLOBAL ERROR HANDLER
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception
):

    request_id = getattr(
        request.state,
        "request_id",
        "unknown"
    )

    log_json(
        "ERROR",
        "Unhandled application error",

        request_id=request_id,

        method=request.method,

        path=request.url.path,

        error_type=type(exc).__name__,

        error=str(exc)
    )

    return JSONResponse(
        status_code=500,

        content={
            "error": "Internal server error",
            "message": (
                "Something went wrong while "
                "processing your request."
            ),
            "request_id": request_id
        }
    )


# ============================================================
# 11. HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "AI Health Risk Prediction API is running",
        "environment": ENVIRONMENT,
        "version": "1.0.0"
    }


# ============================================================
# 12. HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy"
    }


# ============================================================
# 13. AI PREDICTION
# ============================================================

@app.post("/predict")
@limiter.limit("10/minute")
def predict(
    request: Request,
    data: HealthData,
    db: Session = Depends(get_db)
):

    request_id = getattr(
        request.state,
        "request_id",
        "unknown"
    )

    log_json(
        "INFO",
        "Prediction request received",

        request_id=request_id,

        method=request.method,

        path=request.url.path
    )

    try:

        # ----------------------------------------------------
        # Run machine-learning prediction
        # ----------------------------------------------------

        result = predict_risk(
            data.model_dump()
        )


        # ----------------------------------------------------
        # Create database record
        # ----------------------------------------------------

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

            risk_probability=result[
                "risk_probability"
            ],

            risk_percentage=result[
                "risk_percentage"
            ],

            risk_level=result[
                "risk_level"
            ]
        )


        # ----------------------------------------------------
        # Save prediction
        # ----------------------------------------------------

        db.add(
            prediction_record
        )

        db.commit()

        db.refresh(
            prediction_record
        )


        # ----------------------------------------------------
        # Prediction logging
        # ----------------------------------------------------

        log_json(
            "INFO",
            "Prediction completed",

            request_id=request_id,

            method=request.method,

            path=request.url.path,

            prediction=result["prediction"],

            risk_level=result["risk_level"],

            risk_percentage=result[
                "risk_percentage"
            ]
        )


        return result


    except Exception as exc:

        # Rollback database transaction
        db.rollback()

        log_json(
            "ERROR",
            "Prediction failed",

            request_id=request_id,

            method=request.method,

            path=request.url.path,

            error_type=type(exc).__name__,

            error=str(exc)
        )

        return JSONResponse(

            status_code=500,

            content={
                "error": "Prediction failed",

                "message": (
                    "Unable to process the "
                    "prediction request."
                ),

                "request_id": request_id
            }
        )


# ============================================================
# 14. PREDICTION HISTORY
# ============================================================

@app.get("/history")
def get_prediction_history(
    db: Session = Depends(get_db)
):

    records = (
        db.query(
            PredictionHistory
        )
        .order_by(
            PredictionHistory.id.desc()
        )
        .all()
    )

    return [

        {
            "id": record.id,

            "age": record.age,

            "prediction": record.prediction,

            "risk_probability":
                record.risk_probability,

            "risk_percentage":
                record.risk_percentage,

            "risk_level":
                record.risk_level,

            "created_at":
                record.created_at
        }

        for record in records
    ]