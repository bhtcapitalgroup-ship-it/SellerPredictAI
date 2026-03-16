"""
SQLAlchemy ORM models for persistence layer.
"""

from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import (
    Boolean, Float, Index, Integer, String, Text, DateTime,
    ForeignKey, func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class PropertyRow(Base):
    __tablename__ = "properties"

    parcel_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    address: Mapped[str] = mapped_column(String(256))
    city: Mapped[str] = mapped_column(String(128))
    state: Mapped[str] = mapped_column(String(2))
    zip_code: Mapped[str] = mapped_column(String(10))
    county: Mapped[str] = mapped_column(String(128))
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    predictions: Mapped[list[PredictionRow]] = relationship(back_populates="prop")
    lead: Mapped[LeadRow | None] = relationship(back_populates="property", uselist=False)


class PredictionRow(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    parcel_id: Mapped[str] = mapped_column(String(64), ForeignKey("properties.parcel_id"))
    sell_probability: Mapped[float] = mapped_column(Float)
    tier: Mapped[str] = mapped_column(String(16))
    top_factors_json: Mapped[str] = mapped_column(Text, default="[]")
    predicted_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    prop: Mapped[PropertyRow] = relationship(back_populates="predictions")

    def get_top_factors(self) -> list[str]:
        return json.loads(self.top_factors_json)

    def set_top_factors(self, factors: list[str]) -> None:
        self.top_factors_json = json.dumps(factors)

    # Alias for convenience
    @property
    def top_factors(self) -> list[str]:
        return self.get_top_factors()

    __table_args__ = (
        Index("ix_predictions_parcel", "parcel_id"),
        Index("ix_predictions_time", "predicted_at"),
    )


class LeadRow(Base):
    __tablename__ = "leads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    parcel_id: Mapped[str] = mapped_column(String(64), ForeignKey("properties.parcel_id"), unique=True)
    sell_probability: Mapped[float] = mapped_column(Float)
    tier: Mapped[str] = mapped_column(String(16))
    offer_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    offer_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    equity_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    ltv_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    property: Mapped[PropertyRow] = relationship(back_populates="lead")

    __table_args__ = (
        Index("ix_leads_tier", "tier"),
        Index("ix_leads_score", "sell_probability"),
        Index("ix_leads_active", "is_active"),
    )


class FeatureSnapshotRow(Base):
    __tablename__ = "feature_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    parcel_id: Mapped[str] = mapped_column(String(64), ForeignKey("properties.parcel_id"))
    snapshot_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("ix_snapshots_parcel", "parcel_id"),
    )
