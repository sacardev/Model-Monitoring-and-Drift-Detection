from contextlib import asynccontextmanager
from fastapi import FastAPI

from .config import DRIFT_SCHEDULER_ENABLED
from .database import Base, engine
from .routers import drift, health, ingest, predict
from .scheduler import build_scheduler

# Create database tables if they do not exist
Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = None
    if DRIFT_SCHEDULER_ENABLED:
        scheduler = build_scheduler()
        scheduler.start()
    yield
    if scheduler and scheduler.running:
        scheduler.shutdown(wait=False)


app = FastAPI(
    title="Model Monitoring & Drift Detection API",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(ingest.router)
app.include_router(drift.router)