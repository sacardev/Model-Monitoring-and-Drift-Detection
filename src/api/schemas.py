from datetime import datetime
from pydantic import BaseModel

class CustomerFeatures(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    gender: str
    Contract: str
    InternetService: str
    PaymentMethod: str

class PredictRequest(BaseModel):
    records: list[CustomerFeatures]

class PredictionItem(BaseModel):
    prediction: int
    probability: float

class PredictResponse(BaseModel):
    results: list[PredictionItem]

class DriftFeatureResult(BaseModel):
    feature: str
    feature_type: str
    metric: str
    value: float
    drifted: bool

class DriftCheckResponse(BaseModel):
    id: int
    created_at: datetime
    dataset_drifted: bool
    drift_share: float
    rows_checked: int
    features: list[DriftFeatureResult]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool