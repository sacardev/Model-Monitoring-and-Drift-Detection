from functools import lru_cache
import pandas as pd 
import joblib

from ..data_prep import NUMERICAL_COLS, CATEGORICAL_COLS
from ..drift.detector import DriftDetector
from .config import PIPELINE_PATH, REFERENCE_DATA_PATH

@lru_cache(maxsize=1)
def get_pipeline():
    return joblib.load(PIPELINE_PATH)

@lru_cache(maxsize=1)
def get_reference_data() -> pd.DataFrame:
    return pd.read_csv(REFERENCE_DATA_PATH)

@lru_cache(maxsize=1)
def get_drift_detector() -> DriftDetector:
    return DriftDetector(NUMERICAL_COLS, CATEGORICAL_COLS)