"""
Database repository — all CRUD operations for properties, predictions, and leads.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.data.schema import (
    OffMarketLead, Property, PropertyFeatures, SellerPrediction,
)
from .models import FeatureSnapshotRow, LeadRow, PredictionRow, PropertyRow


class LeadRepository:

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def upsert_property(self, session: Session, prop: Property) -> PropertyRow:
        row = session.get(PropertyRow, prop.parcel_id)
        if row is None:
            row = PropertyRow(
                parcel_id=prop.parcel_id,
                address=prop.address,
                city=prop.city,
                state=prop.state,
                zip_code=prop.zip_code,
                county=prop.county,
                latitude=prop.latitude,
                longitude=prop.longitude,
            )
            session.add(row)
        else:
            row.address = prop.address
            row.city = prop.city
            row.state = prop.state
            row.zip_code = prop.zip_code
            row.county = prop.county
            row.latitude = prop.latitude
            row.longitude = prop.longitude
        return row

    # ------------------------------------------------------------------
    # Predictions (append-only history)
    # ------------------------------------------------------------------

    def save_prediction(self, session: Session, pred: SellerPrediction) -> PredictionRow:
        row = PredictionRow(
            parcel_id=pred.parcel_id,
            sell_probability=pred.sell_probability,
            tier=pred.tier.value,
            predicted_at=pred.predicted_at,
        )
        row.set_top_factors(pred.top_factors)
        session.add(row)
        return row

    # ------------------------------------------------------------------
    # Leads (upsert — one active lead per parcel)
    # ------------------------------------------------------------------

    def upsert_lead(self, session: Session, lead: OffMarketLead) -> LeadRow:
        stmt = select(LeadRow).where(LeadRow.parcel_id == lead.parcel_id)
        row = session.execute(stmt).scalar_one_or_none()

        offer_low = lead.recommended_offer_range[0] if lead.recommended_offer_range else None
        offer_high = lead.recommended_offer_range[1] if lead.recommended_offer_range else None
        equity_pct = lead.equity_profile.equity_pct if lead.equity_profile else None
        ltv_ratio = lead.equity_profile.ltv_ratio if lead.equity_profile else None

        if row is None:
            row = LeadRow(
                parcel_id=lead.parcel_id,
                sell_probability=lead.prediction.sell_probability,
                tier=lead.prediction.tier.value,
                offer_low=offer_low,
                offer_high=offer_high,
                equity_pct=equity_pct,
                ltv_ratio=ltv_ratio,
                is_active=True,
            )
            session.add(row)
        else:
            row.sell_probability = lead.prediction.sell_probability
            row.tier = lead.prediction.tier.value
            row.offer_low = offer_low
            row.offer_high = offer_high
            row.equity_pct = equity_pct
            row.ltv_ratio = ltv_ratio
            row.is_active = True

        return row

    # ------------------------------------------------------------------
    # Feature snapshots
    # ------------------------------------------------------------------

    def save_feature_snapshot(self, session: Session, features: PropertyFeatures) -> FeatureSnapshotRow:
        d = asdict(features)
        row = FeatureSnapshotRow(
            parcel_id=features.parcel_id,
            snapshot_json=json.dumps(d),
        )
        session.add(row)
        return row

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_lead(self, session: Session, parcel_id: str) -> Optional[LeadRow]:
        stmt = select(LeadRow).where(
            LeadRow.parcel_id == parcel_id,
            LeadRow.is_active == True,
        )
        return session.execute(stmt).scalar_one_or_none()

    def list_leads(
        self,
        session: Session,
        tier: Optional[str] = None,
        min_score: Optional[float] = None,
        city: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[LeadRow], int]:
        """Returns (leads, total_count) with filtering and pagination."""
        stmt = select(LeadRow).where(LeadRow.is_active == True)

        if tier:
            stmt = stmt.where(LeadRow.tier == tier)
        if min_score is not None:
            stmt = stmt.where(LeadRow.sell_probability >= min_score)
        if city:
            stmt = stmt.join(PropertyRow).where(PropertyRow.city == city)

        # Count
        from sqlalchemy import func
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = session.execute(count_stmt).scalar() or 0

        # Paginated results
        stmt = stmt.order_by(LeadRow.sell_probability.desc()).offset(offset).limit(limit)
        leads = list(session.execute(stmt).scalars().all())

        return leads, total

    def get_prediction_history(self, session: Session, parcel_id: str) -> list[PredictionRow]:
        stmt = (
            select(PredictionRow)
            .where(PredictionRow.parcel_id == parcel_id)
            .order_by(PredictionRow.predicted_at.desc())
        )
        return list(session.execute(stmt).scalars().all())

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def persist_leads(
        self,
        session: Session,
        leads: list[OffMarketLead],
    ) -> None:
        """Persist a full batch of leads (property + prediction + lead row)."""
        for lead in leads:
            self.upsert_property(session, lead.property)
            self.save_prediction(session, lead.prediction)
            self.upsert_lead(session, lead)
        session.commit()
