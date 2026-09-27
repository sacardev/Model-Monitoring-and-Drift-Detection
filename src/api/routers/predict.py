from fastapi import APIRouter, Depends
import pandas as pd
from sqlalchemy.orm import Session

from ..schemas import PredictRequest, PredictResponse, PredictionItem
from ..database import get_db
from ..dependencies import get_pipeline
from ..models import PredictionLog

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, db: Session = Depends(get_db), pipeline = Depends(get_pipeline)):

    records = [r.model_dump() for r in payload.records]
    df = pd.DataFrame(records)

    probablities = pipeline.predict_proba(df)[:, 1]
    prediction = (probablities >= 0.5).astype(int)
    
    logs = [
        PredictionLog(feature = record, prediction = int(pred), probability = float(prob))
        for record, pred, prob in zip(records, prediction, probablities)
    ]

    db.bulk_save_objects(logs)
    db.commit()

    return PredictResponse(
        results = [
            PredictionItem(prediction = int(p), probability = float(pr))
            for p, pr in zip(prediction, probablities)
        ]
    )
    