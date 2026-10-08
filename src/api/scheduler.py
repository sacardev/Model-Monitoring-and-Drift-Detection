import logging
from apscheduler.schedulers.background import BackgroundScheduler
import pandas as pd
from sqlalchemy import select

from .config import DRIFT_CHECK_INTERVAL_MINUTES, DRIFT_CHECK_MIN_ROWS
from .database import SessionLocal
from .dependencies import get_drift_detector, get_reference_data
from .models import DriftCheck, PredictionLog

logger = logging.getLogger(__name__)


def check_drift_job():
    db = SessionLocal()
    try:
        rows = (
            db.execute(
                select(PredictionLog.features).order_by(PredictionLog.id.desc()).limit(500)
            )
            .scalars()
            .all()
        )

        current = pd.DataFrame(rows)
        if len(current) < DRIFT_CHECK_MIN_ROWS:
            logger.info(
                "Skipping scheduled drift check: found %d rows, minimum is %d",
                len(current),
                DRIFT_CHECK_MIN_ROWS,
            )
            return

        detector = get_drift_detector()
        reference = get_reference_data()
        report = detector.run(reference, current)

        record = DriftCheck(
            dataset_drifted=report.dataset_drifted,
            drift_share=report.drift_share,
            report=report.to_dict(),
            rows_checked=len(current),
        )
        db.add(record)
        db.commit()
        logger.info(
            "Scheduled drift check completed: dataset_drifted=%s, drift_share=%.2f",
            record.dataset_drifted,
            record.drift_share,
        )
    except Exception as exc:
        db.rollback()
        logger.error("Error during scheduled drift check: %s", exc)
    finally:
        db.close()


def build_scheduler() -> BackgroundScheduler:
    """Build and configure the background APScheduler."""
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        check_drift_job,
        trigger="interval",
        minutes=DRIFT_CHECK_INTERVAL_MINUTES,
        id="periodic_drift_check",
        replace_existing=True,
    )
    return scheduler