"""
FastAPI application — REST API for the Seller Prediction AI system.
Production-ready with CORS, compression, rate limiting, and error handling.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from src.data.schema import LeadTier
from src.db.models import PropertyRow
from src.db.repository import LeadRepository
from src.db.session import SessionLocal, init_db
from src.engine.lead_generator import LeadGenerator
from src.models.predictor import SellerPredictor

from .schemas import (
    BatchPredictRequest,
    HealthOut,
    LeadDetailOut,
    LeadListOut,
    LeadOut,
    PredictRequest,
    PredictionOut,
    request_to_bundle,
)

logger = logging.getLogger("seller_predict")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENVIRONMENT = os.environ.get("ENVIRONMENT", "production")
ALLOWED_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")
RATE_LIMIT_RPM = int(os.environ.get("RATE_LIMIT_RPM", "120"))

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

repo = LeadRepository()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    predictor = SellerPredictor()
    app.state.lead_generator = LeadGenerator(predictor=predictor)
    app.state.predictor_mode = predictor.mode
    app.state.start_time = time.time()
    logger.info(f"Seller Predict AI started — mode={predictor.mode}, env={ENVIRONMENT}")
    yield
    logger.info("Seller Predict AI shutting down")


app = FastAPI(
    title="Seller Predict AI",
    description="AI system that predicts which homeowners are likely to sell soon. "
                "Analyzes tax records, distress signals, vacancy indicators, equity levels, "
                "and probate filings to generate off-market property leads.",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

# CORS — allow configurable origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gzip compression for responses > 500 bytes
app.add_middleware(GZipMiddleware, minimum_size=500)

# Rate limiting state (in-memory, per-IP)
_rate_limit_store: dict[str, list[float]] = defaultdict(list)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple sliding-window rate limiter (per IP, per minute)."""
    if request.url.path in ("/health", "/docs", "/redoc", "/openapi.json", "/dashboard"):
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window = 60.0

    # Clean old entries
    _rate_limit_store[client_ip] = [
        t for t in _rate_limit_store[client_ip] if now - t < window
    ]

    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_RPM:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Try again in 60 seconds."},
            headers={"Retry-After": "60"},
        )

    _rate_limit_store[client_ip].append(now)
    return await call_next(request)


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    """Add response timing header."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    response.headers["X-Response-Time-Ms"] = f"{elapsed:.1f}"
    return response


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error on {request.method} {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

from .dashboard import router as dashboard_router
app.include_router(dashboard_router)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictionOut, tags=["Predictions"])
def predict_single(req: PredictRequest, db: Session = Depends(get_db)):
    """Score a single property and return seller probability with explainable factors."""
    bundle = request_to_bundle(req)
    engine: LeadGenerator = app.state.lead_generator

    original_tier = engine.min_tier
    engine.min_tier = LeadTier.COLD
    leads = engine.generate([bundle])
    engine.min_tier = original_tier

    if not leads:
        raise HTTPException(400, "Could not generate prediction for this property")

    lead = leads[0]
    repo.persist_leads(db, [lead])

    return PredictionOut(
        parcel_id=lead.parcel_id,
        sell_probability=lead.prediction.sell_probability,
        tier=lead.prediction.tier.value,
        top_factors=lead.prediction.top_factors,
        predicted_at=lead.prediction.predicted_at,
    )


@app.post("/predict/batch", response_model=list[PredictionOut], tags=["Predictions"])
def predict_batch(req: BatchPredictRequest, db: Session = Depends(get_db)):
    """Score a batch of properties (max 1000). Returns all results sorted by score."""
    bundles = [request_to_bundle(item) for item in req.items]
    engine: LeadGenerator = app.state.lead_generator

    original_tier = engine.min_tier
    engine.min_tier = LeadTier.COLD
    leads = engine.generate(bundles)
    engine.min_tier = original_tier

    repo.persist_leads(db, leads)

    return [
        PredictionOut(
            parcel_id=lead.parcel_id,
            sell_probability=lead.prediction.sell_probability,
            tier=lead.prediction.tier.value,
            top_factors=lead.prediction.top_factors,
            predicted_at=lead.prediction.predicted_at,
        )
        for lead in leads
    ]


@app.get("/leads", response_model=LeadListOut, tags=["Leads"])
def list_leads(
    tier: Optional[str] = Query(None, description="Filter by tier: hot, warm, cold"),
    min_score: Optional[float] = Query(None, ge=0, le=1, description="Minimum seller probability"),
    city: Optional[str] = Query(None, description="Filter by city name"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """List off-market leads with filtering and pagination. Sorted by score descending."""
    offset = (page - 1) * page_size
    rows, total = repo.list_leads(
        db, tier=tier, min_score=min_score, city=city,
        limit=page_size, offset=offset,
    )

    leads_out = []
    for row in rows:
        prop = db.get(PropertyRow, row.parcel_id)
        leads_out.append(LeadOut(
            parcel_id=row.parcel_id,
            address=prop.address if prop else "",
            city=prop.city if prop else "",
            state=prop.state if prop else "",
            sell_probability=row.sell_probability,
            tier=row.tier,
            offer_low=row.offer_low,
            offer_high=row.offer_high,
            equity_pct=row.equity_pct,
            ltv_ratio=row.ltv_ratio,
            generated_at=row.generated_at,
        ))

    return LeadListOut(leads=leads_out, total=total, page=page, page_size=page_size)


@app.get("/leads/{parcel_id}", response_model=LeadDetailOut, tags=["Leads"])
def get_lead_detail(parcel_id: str, db: Session = Depends(get_db)):
    """Get full lead detail including prediction history over time."""
    row = repo.get_lead(db, parcel_id)
    if row is None:
        raise HTTPException(404, f"No active lead found for parcel {parcel_id}")

    prop = db.get(PropertyRow, parcel_id)
    history = repo.get_prediction_history(db, parcel_id)
    top_factors = history[0].top_factors if history else []

    return LeadDetailOut(
        parcel_id=row.parcel_id,
        address=prop.address if prop else "",
        city=prop.city if prop else "",
        state=prop.state if prop else "",
        sell_probability=row.sell_probability,
        tier=row.tier,
        offer_low=row.offer_low,
        offer_high=row.offer_high,
        equity_pct=row.equity_pct,
        ltv_ratio=row.ltv_ratio,
        generated_at=row.generated_at,
        top_factors=top_factors,
        prediction_history=[
            PredictionOut(
                parcel_id=h.parcel_id,
                sell_probability=h.sell_probability,
                tier=h.tier,
                top_factors=h.top_factors,
                predicted_at=h.predicted_at,
            )
            for h in history
        ],
    )


@app.get("/health", response_model=HealthOut, tags=["System"])
def health_check(db: Session = Depends(get_db)):
    """System health check with uptime and lead count."""
    from sqlalchemy import func, select
    from src.db.models import LeadRow

    count = db.execute(select(func.count()).select_from(LeadRow)).scalar() or 0
    uptime = time.time() - getattr(app.state, "start_time", time.time())

    return HealthOut(
        status="ok",
        version="0.1.0",
        mode=app.state.predictor_mode,
        total_leads=count,
    )
