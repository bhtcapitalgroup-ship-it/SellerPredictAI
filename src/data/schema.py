"""
Data models for the Seller Prediction AI system.
Defines all entities, data sources, and scoring outputs.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DistressType(enum.Enum):
    FORECLOSURE_FILING = "foreclosure_filing"
    TAX_LIEN = "tax_lien"
    CODE_VIOLATION = "code_violation"
    MECHANIC_LIEN = "mechanic_lien"
    UTILITY_SHUTOFF = "utility_shutoff"
    DIVORCE_FILING = "divorce_filing"
    BANKRUPTCY = "bankruptcy"


class VacancySignal(enum.Enum):
    MAIL_FORWARDING = "mail_forwarding"
    UTILITY_INACTIVE = "utility_inactive"
    OVERGROWN_VEGETATION = "overgrown_vegetation"
    NEIGHBOR_REPORT = "neighbor_report"
    NO_OCCUPANCY_PERMIT = "no_occupancy_permit"
    EXTENDED_ABSENCE = "extended_absence"


class ProbateStatus(enum.Enum):
    FILED = "filed"
    IN_PROGRESS = "in_progress"
    SETTLED = "settled"
    CONTESTED = "contested"


class LeadTier(enum.Enum):
    HOT = "hot"          # score >= 0.80
    WARM = "warm"        # score >= 0.55
    COLD = "cold"        # score <  0.55


# ---------------------------------------------------------------------------
# Core property record
# ---------------------------------------------------------------------------

@dataclass
class Property:
    parcel_id: str
    address: str
    city: str
    state: str
    zip_code: str
    county: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None


# ---------------------------------------------------------------------------
# Data-source records
# ---------------------------------------------------------------------------

@dataclass
class TaxRecord:
    parcel_id: str
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
    assessment_year: int = 2025


@dataclass
class DistressSignal:
    parcel_id: str
    distress_type: DistressType
    filing_date: date
    amount: Optional[float] = None
    resolved: bool = False
    resolution_date: Optional[date] = None
    source: str = ""


@dataclass
class VacancyIndicator:
    parcel_id: str
    signal: VacancySignal
    detected_date: date
    confidence: float = 0.5          # 0.0-1.0
    last_verified: Optional[date] = None
    source: str = ""


@dataclass
class EquityProfile:
    parcel_id: str
    estimated_market_value: float
    outstanding_mortgage: float
    other_liens: float = 0.0
    equity_amount: float = 0.0       # computed
    equity_pct: float = 0.0          # computed
    ltv_ratio: float = 0.0           # computed
    as_of_date: date = field(default_factory=date.today)

    def compute(self) -> None:
        self.equity_amount = self.estimated_market_value - self.outstanding_mortgage - self.other_liens
        if self.estimated_market_value > 0:
            self.equity_pct = self.equity_amount / self.estimated_market_value
            self.ltv_ratio = (self.outstanding_mortgage + self.other_liens) / self.estimated_market_value
        else:
            self.equity_pct = 0.0
            self.ltv_ratio = 0.0


@dataclass
class ProbateFiling:
    parcel_id: str
    decedent_name: str
    filing_date: date
    status: ProbateStatus
    executor_name: Optional[str] = None
    estimated_estate_value: Optional[float] = None
    num_heirs: Optional[int] = None
    contested: bool = False


# ---------------------------------------------------------------------------
# Unified feature vector (model input)
# ---------------------------------------------------------------------------

@dataclass
class PropertyFeatures:
    """Flat feature vector built from all data sources for a single parcel."""

    parcel_id: str

    # Tax features
    assessed_value: float = 0.0
    market_value: float = 0.0
    tax_delinquent: bool = False
    delinquent_amount: float = 0.0
    years_delinquent: int = 0
    owner_occupied: bool = True
    years_since_last_sale: float = 0.0

    # Distress features
    num_distress_signals: int = 0
    has_foreclosure: bool = False
    has_tax_lien: bool = False
    has_bankruptcy: bool = False
    has_divorce: bool = False
    total_distress_amount: float = 0.0
    days_since_latest_distress: float = 999.0

    # Vacancy features
    num_vacancy_signals: int = 0
    max_vacancy_confidence: float = 0.0
    has_utility_inactive: bool = False
    has_mail_forwarding: bool = False

    # Equity features
    equity_pct: float = 0.0
    ltv_ratio: float = 0.0
    is_underwater: bool = False
    high_equity: bool = False           # equity_pct > 0.50

    # Probate features
    has_probate: bool = False
    probate_contested: bool = False
    num_heirs: int = 0
    days_since_probate_filing: float = 999.0


# ---------------------------------------------------------------------------
# Prediction output
# ---------------------------------------------------------------------------

@dataclass
class SellerPrediction:
    parcel_id: str
    sell_probability: float             # 0.0-1.0
    tier: LeadTier
    top_factors: list[str] = field(default_factory=list)
    predicted_at: datetime = field(default_factory=datetime.utcnow)

    @staticmethod
    def tier_from_score(score: float) -> LeadTier:
        if score >= 0.80:
            return LeadTier.HOT
        elif score >= 0.55:
            return LeadTier.WARM
        return LeadTier.COLD


@dataclass
class OffMarketLead:
    parcel_id: str
    property: Property
    prediction: SellerPrediction
    equity_profile: Optional[EquityProfile] = None
    contact_info: Optional[dict] = None
    recommended_offer_range: Optional[tuple[float, float]] = None
    generated_at: datetime = field(default_factory=datetime.utcnow)
