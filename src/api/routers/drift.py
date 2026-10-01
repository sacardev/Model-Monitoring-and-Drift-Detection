import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select
import pandas as pd

from ..database import get_db
from ..models import DriftCheck, PredictionLog
from ..schemas import DriftCheckResponse
from ..config import DRIFT_CHECK_MIN_ROWS
from ..dependencies import get_drift_detector, get_reference_data

router = APIRouter()

def _load_recent_features(db: Session, limit: int = 500) -> pd.DataFrame:
    rows = db.execute(
        select(PredictionLog.features).order_by(PredictionLog.id.desc()).limit(limit)
    ).scalars().all()

    return pd.DataFrame(rows)

@router.post("/drift-report", response_model=DriftCheckResponse)
def run_drift_check(db: Session = Depends(get_db), reference = Depends(get_reference_data), detector = Depends(get_drift_detector)):

    current = _load_recent_features(db)

    min_rows = int(DRIFT_CHECK_MIN_ROWS)
    if len(current) < min_rows:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {min_rows} logged rows, have {len(current)}.",
        )

    report = detector.run(reference, current)

    record = DriftCheck(
        dataset_drifted=report.dataset_drifted,
        drift_share=report.drift_share,
        report=report.to_dict(),
        rows_checked=len(current),
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    return DriftCheckResponse(
        id=record.id,
        created_at=record.created_at,
        dataset_drifted=record.dataset_drifted,
        drift_share=record.drift_share,
        rows_checked=record.rows_checked,
        features=[r.__dict__ for r in report.feature_result],
    )
    
@router.get("/drift-report/latest", response_model=DriftCheckResponse)
def latest_drift_check(db: Session = Depends(get_db)):
    record = db.execute(
        select(DriftCheck).order_by(DriftCheck.id.desc()).limit(1)
    ).scalar_one_or_none()

    if record is None:
        raise HTTPException(status_code=404, detail="No drift checks have been run yet.")

    report_data = record.report
    if isinstance(report_data, str):
        report_data = json.loads(report_data)
    elif not isinstance(report_data, dict):
        report_data = {}

    features = report_data.get("features") or report_data.get("feature_result") or []

    return DriftCheckResponse(
        id=record.id,
        created_at=record.created_at,
        dataset_drifted=record.dataset_drifted,
        drift_share=record.drift_share,
        rows_checked=record.rows_checked,
        features=features,
    )
