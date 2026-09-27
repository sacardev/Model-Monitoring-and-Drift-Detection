from fastapi import APIRouter
from ..dependencies import get_pipeline
from ..schemas import HealthResponse

router = APIRouter()

@router.get("/health", response_model = HealthResponse)

def health_check():
    try:
        get_pipeline()
        return HealthResponse(status = "ok", model_loaded="true")
    except Exception:
        return HealthResponse(status = "degraded", model_loaded="false")