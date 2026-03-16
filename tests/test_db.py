"""
Tests for the database layer — uses in-memory SQLite.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Force in-memory DB before any imports touch session.py
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.schema import (
    EquityProfile, LeadTier, OffMarketLead, Property, SellerPrediction,
)
from src.db.models import Base, LeadRow, PredictionRow, PropertyRow
from src.db.repository import LeadRepository
from src.db.session import SessionLocal, engine


def setup():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    return SessionLocal(), LeadRepository()


def make_lead(parcel_id: str, score: float, city: str = "Austin") -> OffMarketLead:
    prop = Property(parcel_id, f"{parcel_id} St", city, "TX", "78701", "Travis")
    tier = SellerPrediction.tier_from_score(score)
    pred = SellerPrediction(
        parcel_id=parcel_id,
        sell_probability=score,
        tier=tier,
        top_factors=["Factor A", "Factor B"],
    )
    eq = EquityProfile(parcel_id, 300000, 200000)
    eq.compute()
    return OffMarketLead(
        parcel_id=parcel_id,
        property=prop,
        prediction=pred,
        equity_profile=eq,
        recommended_offer_range=(200000, 250000),
    )


def test_upsert_property():
    db, repo = setup()
    prop = Property("DB001", "100 Test St", "Austin", "TX", "78701", "Travis")

    row = repo.upsert_property(db, prop)
    db.commit()
    assert row.parcel_id == "DB001"
    assert row.address == "100 Test St"

    # Update
    prop.address = "100 Test St Updated"
    repo.upsert_property(db, prop)
    db.commit()
    row = db.get(PropertyRow, "DB001")
    assert row.address == "100 Test St Updated"

    db.close()
    print("[PASS] upsert_property")


def test_save_prediction():
    db, repo = setup()
    prop = Property("DB002", "200 Test St", "Dallas", "TX", "75201", "Dallas")
    repo.upsert_property(db, prop)

    pred = SellerPrediction("DB002", 0.85, LeadTier.HOT, ["Foreclosure", "Vacant"])
    row = repo.save_prediction(db, pred)
    db.commit()

    assert row.parcel_id == "DB002"
    assert row.sell_probability == 0.85
    assert row.tier == "hot"
    assert row.top_factors == ["Foreclosure", "Vacant"]

    # Append another prediction
    pred2 = SellerPrediction("DB002", 0.90, LeadTier.HOT, ["Foreclosure", "Probate"])
    repo.save_prediction(db, pred2)
    db.commit()

    history = repo.get_prediction_history(db, "DB002")
    assert len(history) == 2
    assert history[0].sell_probability >= history[1].sell_probability  # desc order

    db.close()
    print("[PASS] save_prediction + history")


def test_upsert_lead():
    db, repo = setup()
    lead = make_lead("DB003", 0.92, "Houston")
    repo.upsert_property(db, lead.property)
    repo.upsert_lead(db, lead)
    db.commit()

    row = repo.get_lead(db, "DB003")
    assert row is not None
    assert row.sell_probability == 0.92
    assert row.tier == "hot"
    assert row.offer_low == 200000
    assert row.offer_high == 250000

    # Update score
    lead2 = make_lead("DB003", 0.75, "Houston")
    repo.upsert_lead(db, lead2)
    db.commit()

    row = repo.get_lead(db, "DB003")
    assert row.sell_probability == 0.75
    assert row.tier == "warm"

    db.close()
    print("[PASS] upsert_lead")


def test_list_leads_with_filters():
    db, repo = setup()

    leads = [
        make_lead("FL001", 0.95, "Austin"),
        make_lead("FL002", 0.85, "Austin"),
        make_lead("FL003", 0.60, "Dallas"),
        make_lead("FL004", 0.40, "Dallas"),
        make_lead("FL005", 0.20, "Houston"),
    ]
    repo.persist_leads(db, leads)

    # All leads
    rows, total = repo.list_leads(db)
    assert total == 5

    # Filter by tier
    rows, total = repo.list_leads(db, tier="hot")
    assert total == 2
    assert all(r.tier == "hot" for r in rows)

    # Filter by min_score
    rows, total = repo.list_leads(db, min_score=0.55)
    assert total == 3

    # Filter by city
    rows, total = repo.list_leads(db, city="Dallas")
    assert total == 2

    # Pagination
    rows, total = repo.list_leads(db, limit=2, offset=0)
    assert len(rows) == 2
    assert total == 5
    assert rows[0].sell_probability >= rows[1].sell_probability  # sorted desc

    db.close()
    print("[PASS] list_leads with filters")


def test_persist_leads_batch():
    db, repo = setup()
    leads = [make_lead(f"BT{i:03d}", 0.5 + i * 0.05) for i in range(10)]
    repo.persist_leads(db, leads)

    rows, total = repo.list_leads(db, limit=100)
    assert total == 10

    db.close()
    print("[PASS] persist_leads batch")


if __name__ == "__main__":
    print("=" * 60)
    print("  Database Layer Tests")
    print("=" * 60)
    test_upsert_property()
    test_save_prediction()
    test_upsert_lead()
    test_list_leads_with_filters()
    test_persist_leads_batch()
    print("\n✓ All DB tests passed.")
