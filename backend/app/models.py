from sqlalchemy import Column, Integer, Float, String, DateTime
from datetime import datetime

from app.database import Base


class PredictionHistory(Base):

    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)

    age = Column(Integer)
    sex = Column(Integer)
    cp = Column(Integer)

    trestbps = Column(Float)
    chol = Column(Float)
    fbs = Column(Integer)
    restecg = Column(Integer)

    thalach = Column(Float)
    exang = Column(Integer)
    oldpeak = Column(Float)

    slope = Column(Integer)
    ca = Column(Integer)
    thal = Column(Integer)

    prediction = Column(Integer)

    risk_probability = Column(Float)
    risk_percentage = Column(Float)
    risk_level = Column(String)

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )