"""
Transforms raw data-source records into a unified PropertyFeatures vector.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from .schema import (
    DistressSignal, DistressType, EquityProfile, ProbateFiling,
    PropertyFeatures, TaxRecord, VacancyIndicator, VacancySignal,
)


class FeatureBuilder:
    """Builds a PropertyFeatures vector from heterogeneous source records."""

    def __init__(self, reference_date: Optional[date] = None):
        self.reference_date = reference_date or date.today()

    def build(
        self,
        parcel_id: str,
        tax: Optional[TaxRecord] = None,
        distress: Optional[list[DistressSignal]] = None,
        vacancy: Optional[list[VacancyIndicator]] = None,
        equity: Optional[EquityProfile] = None,
        probate: Optional[ProbateFiling] = None,
    ) -> PropertyFeatures:
        feats = PropertyFeatures(parcel_id=parcel_id)
        self._apply_tax(feats, tax)
        self._apply_distress(feats, distress or [])
        self._apply_vacancy(feats, vacancy or [])
        self._apply_equity(feats, equity)
        self._apply_probate(feats, probate)
        return feats

    # ------------------------------------------------------------------

    def _days_between(self, d: date) -> float:
        return max((self.reference_date - d).days, 0)

    def _apply_tax(self, f: PropertyFeatures, tax: Optional[TaxRecord]) -> None:
        if tax is None:
            return
        f.assessed_value = tax.assessed_value
        f.market_value = tax.market_value
        f.tax_delinquent = tax.tax_delinquent
        f.delinquent_amount = tax.delinquent_amount
        f.years_delinquent = tax.years_delinquent
        f.owner_occupied = tax.owner_occupied
        if tax.last_sale_date:
            f.years_since_last_sale = self._days_between(tax.last_sale_date) / 365.25

    def _apply_distress(self, f: PropertyFeatures, signals: list[DistressSignal]) -> None:
        active = [s for s in signals if not s.resolved]
        f.num_distress_signals = len(active)
        f.has_foreclosure = any(s.distress_type == DistressType.FORECLOSURE_FILING for s in active)
        f.has_tax_lien = any(s.distress_type == DistressType.TAX_LIEN for s in active)
        f.has_bankruptcy = any(s.distress_type == DistressType.BANKRUPTCY for s in active)
        f.has_divorce = any(s.distress_type == DistressType.DIVORCE_FILING for s in active)
        f.total_distress_amount = sum(s.amount or 0 for s in active)
        if active:
            f.days_since_latest_distress = min(
                self._days_between(s.filing_date) for s in active
            )

    def _apply_vacancy(self, f: PropertyFeatures, signals: list[VacancyIndicator]) -> None:
        f.num_vacancy_signals = len(signals)
        if signals:
            f.max_vacancy_confidence = max(s.confidence for s in signals)
        f.has_utility_inactive = any(s.signal == VacancySignal.UTILITY_INACTIVE for s in signals)
        f.has_mail_forwarding = any(s.signal == VacancySignal.MAIL_FORWARDING for s in signals)

    def _apply_equity(self, f: PropertyFeatures, equity: Optional[EquityProfile]) -> None:
        if equity is None:
            return
        equity.compute()
        f.equity_pct = equity.equity_pct
        f.ltv_ratio = equity.ltv_ratio
        f.is_underwater = equity.equity_amount < 0
        f.high_equity = equity.equity_pct > 0.50

    def _apply_probate(self, f: PropertyFeatures, probate: Optional[ProbateFiling]) -> None:
        if probate is None:
            return
        f.has_probate = True
        f.probate_contested = probate.contested
        f.num_heirs = probate.num_heirs or 0
        f.days_since_probate_filing = self._days_between(probate.filing_date)
