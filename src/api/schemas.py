"""
Pydantic request/response models for the REST API.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from src.data.schema import (
    DistressSignal, DistressType, EquityProfile, ProbateFiling,
    ProbateStatus, Property, TaxRecord, VacancyIndicator, VacancySignal,
)
from src.engine.lead_generator import PropertyBundle


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class PropertyIn(BaseModel):
    parcel_id: str
    address: str
    city: str
    state: str
    zip_code: str
    county: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class TaxRecordIn(BaseModel):
    assessed_value: float
    market_value: float
    annual_tax: float
    tax_delinquent: bool = False
    delinquent_amount: float = 0.0
    years_delinquent: int = 0
    last_sale_date: Optional[date] = None
    last_sale_price: Optional[float] = None
    owner_name: str = ""
    owner_occupied: bool = True


class DistressSignalIn(BaseModel):
    distress_type: str
    filing_date: date
    amount: Optional[float] = None
    resolved: bool = False
    source: str = ""

    @field_validator("distress_type")
    @classmethod
    def validate_distress_type(cls, v: str) -> str:
        valid = {e.value for e in DistressType}
        if v not in valid:
            raise ValueError(f"distress_type must be one of {valid}")
        return v


class VacancyIndicatorIn(BaseModel):
    signal: str
    detected_date: date
    confidence: float = 0.5
    source: str = ""

    @field_validator("signal")
    @classmethod
    def validate_signal(cls, v: str) -> str:
        valid = {e.value for e in VacancySignal}
        if v not in valid:
            raise ValueError(f"signal must be one of {valid}")
        return v


class EquityProfileIn(BaseModel):
    estimated_market_value: float
    outstanding_mortgage: float
    other_liens: float = 0.0


class ProbateFilingIn(BaseModel):
    decedent_name: str
    filing_date: date
    status: str
    executor_name: Optional[str] = None
    estimated_estate_value: Optional[float] = None
    num_heirs: Optional[int] = None
    contested: bool = False

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        valid = {e.value for e in ProbateStatus}
        if v not in valid:
            raise ValueError(f"status must be one of {valid}")
        return v


class PredictRequest(BaseModel):
    property: PropertyIn
    tax: Optional[TaxRecordIn] = None
    distress: Optional[list[DistressSignalIn]] = None
    vacancy: Optional[list[VacancyIndicatorIn]] = None
    equity: Optional[EquityProfileIn] = None
    probate: Optional[ProbateFilingIn] = None


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest] = Field(..., max_length=1000)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class PredictionOut(BaseModel):
    parcel_id: str
    sell_probability: float
    tier: str
    top_factors: list[str]
    predicted_at: datetime


class LeadOut(BaseModel):
    parcel_id: str
    address: str
    city: str
    state: str
    sell_probability: float
    tier: str
    offer_low: Optional[float] = None
    offer_high: Optional[float] = None
    equity_pct: Optional[float] = None
    ltv_ratio: Optional[float] = None
    generated_at: datetime


class LeadListOut(BaseModel):
    leads: list[LeadOut]
    total: int
    page: int
    page_size: int


class LeadDetailOut(LeadOut):
    top_factors: list[str] = []
    prediction_history: list[PredictionOut] = []


class HealthOut(BaseModel):
    status: str
    version: str
    mode: str
    total_leads: int = 0


# ---------------------------------------------------------------------------
# Converters: Pydantic request → domain dataclass
# ---------------------------------------------------------------------------

def request_to_bundle(req: PredictRequest) -> PropertyBundle:
    prop = Property(
        parcel_id=req.property.parcel_id,
        address=req.property.address,
        city=req.property.city,
        state=req.property.state,
        zip_code=req.property.zip_code,
        county=req.property.county,
        latitude=req.property.latitude,
        longitude=req.property.longitude,
    )

    tax = None
    if req.tax:
        tax = TaxRecord(
            parcel_id=prop.parcel_id,
            assessed_value=req.tax.assessed_value,
            market_value=req.tax.market_value,
            annual_tax=req.tax.annual_tax,
            tax_delinquent=req.tax.tax_delinquent,
            delinquent_amount=req.tax.delinquent_amount,
            years_delinquent=req.tax.years_delinquent,
            last_sale_date=req.tax.last_sale_date,
            last_sale_price=req.tax.last_sale_price,
            owner_name=req.tax.owner_name,
            owner_occupied=req.tax.owner_occupied,
        )

    distress = []
    if req.distress:
        for d in req.distress:
            distress.append(DistressSignal(
                parcel_id=prop.parcel_id,
                distress_type=DistressType(d.distress_type),
                filing_date=d.filing_date,
                amount=d.amount,
                resolved=d.resolved,
                source=d.source,
            ))

    vacancy = []
    if req.vacancy:
        for v in req.vacancy:
            vacancy.append(VacancyIndicator(
                parcel_id=prop.parcel_id,
                signal=VacancySignal(v.signal),
                detected_date=v.detected_date,
                confidence=v.confidence,
                source=v.source,
            ))

    equity = None
    if req.equity:
        equity = EquityProfile(
            parcel_id=prop.parcel_id,
            estimated_market_value=req.equity.estimated_market_value,
            outstanding_mortgage=req.equity.outstanding_mortgage,
            other_liens=req.equity.other_liens,
        )

    probate = None
    if req.probate:
        probate = ProbateFiling(
            parcel_id=prop.parcel_id,
            decedent_name=req.probate.decedent_name,
            filing_date=req.probate.filing_date,
            status=ProbateStatus(req.probate.status),
            executor_name=req.probate.executor_name,
            estimated_estate_value=req.probate.estimated_estate_value,
            num_heirs=req.probate.num_heirs,
            contested=req.probate.contested,
        )

    return PropertyBundle(
        property=prop,
        tax=tax,
        distress=distress,
        vacancy=vacancy,
        equity=equity,
        probate=probate,
    )
