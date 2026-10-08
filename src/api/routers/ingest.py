from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import pandas as pd

from ..database import get_db
from ..dependencies import get_pipeline
from ..models import PredictionLog
from ..schemas import PredictRequest

router = APIRouter()

@router.post("/ingest", status_code=201)
def ingest(payload: PredictRequest, db: Session = Depends(get_db), pipeline=Depends(get_pipeline)):
    
    records = [r.model_dump() for r in payload.records]
    df = pd.DataFrame(records)
    probabilities = pipeline.predict_proba(df)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    logs = [
        PredictionLog(features=record, prediction=int(pred), probability=float(prob))
        for record, pred, prob in zip(records, predictions, probabilities)
    ]
    db.bulk_save_objects(logs)
    db.commit()
    return {"ingested": len(logs)}