import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATABASE_PATH = BASE_DIR / "monitoring.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

PIPELINE_PATH = BASE_DIR / "artifacts" / "model.pkl"
REFERENCE_DATA_PATH = BASE_DIR / "reference_data.csv"

DRIFT_SCHEDULER_ENABLED = os.getenv("DRIFT_SCHEDULER_ENABLED", "true").lower() == "true"
DRIFT_CHECK_INTERVAL_MINUTES = os.getenv("DRIFT_CHECK_INTERVAL_MINUTES", "30")
DRIFT_CHECK_MIN_ROWS = os.getenv("DRIFT_CHECK_MIN_ROWS", "50")