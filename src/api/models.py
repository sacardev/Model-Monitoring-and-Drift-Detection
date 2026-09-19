from sqlalchemy import Column, Integer, Float, Boolean, DateTime, JSON
from sqlalchemy.sql import func
from .database import Base

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    features = Column(JSON, nullable=False)
    prediction = Column(Integer, nullable=False)
    probability = Column(Float, nullable=False)

class DriftCheck(Base):
    __tablename__ = "drift_checks"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    dataset_drifted = Column(Boolean, nullable=False)
    drift_share = Column(Float, nullable=False)
    report = Column(JSON, nullable=False)
    rows_checked = Column(Integer, nullable=False)
