import os
from pathlib import Path
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
THRESHOLD_CONFIG_PATH = BASE_DIR / "threshold_config.yaml"
CACHE_TTL_SECONDS = int(os.getenv("DASHBOARD_CACHE_TTL", "15"))

def load_thresholds() -> dict:
    with open(THRESHOLD_CONFIG_PATH) as f:
        return yaml.safe_load(f)