"""
Lead generation engine.

Orchestrates the full pipeline:
  ingest → feature build → predict → rank → output off-market leads
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from src.data.schema import (
    DistressSignal, EquityProfile, LeadTier, OffMarketLead,
    ProbateFiling, Property, PropertyFeatures, SellerPrediction,
    TaxRecord, VacancyIndicator,
)
from src.data.feature_builder import FeatureBuilder
from src.models.predictor import SellerPredictor


class PropertyBundle:
    """All raw data for a single property, ready for feature extraction."""

    def __init__(
        self,
        property: Property,
        tax: Optional[TaxRecord] = None,
        distress: Optional[list[DistressSignal]] = None,
        vacancy: Optional[list[VacancyIndicator]] = None,
        equity: Optional[EquityProfile] = None,
        probate: Optional[ProbateFiling] = None,
    ):
        self.property = property
        self.tax = tax
        self.distress = distress or []
        self.vacancy = vacancy or []
        self.equity = equity
        self.probate = probate


class LeadGenerator:
    """
    End-to-end pipeline: raw property data → ranked off-market leads.
    """

    def __init__(
        self,
        predictor: Optional[SellerPredictor] = None,
        min_tier: LeadTier = LeadTier.WARM,
        reference_date: Optional[date] = None,
    ):
        self.predictor = predictor or SellerPredictor()
        self.min_tier = min_tier
        self.feature_builder = FeatureBuilder(reference_date=reference_date)
        self._tier_priority = {LeadTier.HOT: 0, LeadTier.WARM: 1, LeadTier.COLD: 2}

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def generate(self, bundles: list[PropertyBundle]) -> list[OffMarketLead]:
        """Run full pipeline on a batch of property bundles."""

        # 1. Build feature vectors
        features = [self._extract_features(b) for b in bundles]

        # 2. Score all properties
        predictions = self.predictor.predict_batch(features)

        # 3. Filter by minimum tier
        max_tier_rank = self._tier_priority[self.min_tier]
        qualified = [
            (bundle, pred)
            for bundle, pred in zip(bundles, predictions)
            if self._tier_priority[pred.tier] <= max_tier_rank
        ]

        # 4. Build leads with offer ranges
        leads = []
        for bundle, pred in qualified:
            offer_range = self._estimate_offer_range(bundle, pred)
            lead = OffMarketLead(
                parcel_id=bundle.property.parcel_id,
                property=bundle.property,
                prediction=pred,
                equity_profile=bundle.equity,
                recommended_offer_range=offer_range,
            )
            leads.append(lead)

        # 5. Sort by score descending
        leads.sort(key=lambda l: l.prediction.sell_probability, reverse=True)

        return leads

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self, leads: list[OffMarketLead]) -> dict:
        """Summary statistics for a batch of leads."""
        hot = [l for l in leads if l.prediction.tier == LeadTier.HOT]
        warm = [l for l in leads if l.prediction.tier == LeadTier.WARM]
        cold = [l for l in leads if l.prediction.tier == LeadTier.COLD]

        return {
            "total_leads": len(leads),
            "hot": len(hot),
            "warm": len(warm),
            "cold": len(cold),
            "avg_score": round(
                sum(l.prediction.sell_probability for l in leads) / max(len(leads), 1), 4
            ),
            "top_factors": self._aggregate_factors(leads),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_features(self, bundle: PropertyBundle) -> PropertyFeatures:
        return self.feature_builder.build(
            parcel_id=bundle.property.parcel_id,
            tax=bundle.tax,
            distress=bundle.distress,
            vacancy=bundle.vacancy,
            equity=bundle.equity,
            probate=bundle.probate,
        )

    def _estimate_offer_range(
        self, bundle: PropertyBundle, pred: SellerPrediction
    ) -> Optional[tuple[float, float]]:
        """Estimate a reasonable offer range based on equity and distress level."""
        if bundle.equity is None:
            if bundle.tax and bundle.tax.market_value > 0:
                mv = bundle.tax.market_value
                return (mv * 0.65, mv * 0.85)
            return None

        bundle.equity.compute()
        mv = bundle.equity.estimated_market_value

        # Higher distress → lower offer range (more negotiating leverage)
        if pred.sell_probability >= 0.80:
            discount = (0.25, 0.35)
        elif pred.sell_probability >= 0.55:
            discount = (0.15, 0.25)
        else:
            discount = (0.08, 0.18)

        low = mv * (1 - discount[1])
        high = mv * (1 - discount[0])
        return (round(low, 2), round(high, 2))

    @staticmethod
    def _aggregate_factors(leads: list[OffMarketLead], top_n: int = 5) -> list[str]:
        """Most common top factors across all leads."""
        counts: dict[str, int] = {}
        for lead in leads:
            for factor in lead.prediction.top_factors:
                counts[factor] = counts.get(factor, 0) + 1
        ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [f for f, _ in ranked[:top_n]]
