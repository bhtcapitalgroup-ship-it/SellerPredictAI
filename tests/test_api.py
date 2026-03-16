"""
Tests for the FastAPI REST API.
"""

import os
import sys
from pathlib import Path

os.environ["DATABASE_URL"] = "sqlite:///:memory:"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from src.api.app import app
from src.db.session import init_db
from src.engine.lead_generator import LeadGenerator
from src.models.predictor import SellerPredictor

# Setup: init DB and app state (lifespan may not fire in all TestClient modes)
init_db()
predictor = SellerPredictor()
app.state.lead_generator = LeadGenerator(predictor=predictor)
app.state.predictor_mode = predictor.mode

client = TestClient(app)


HOT_PROPERTY = {
    "property": {
        "parcel_id": "API001",
        "address": "123 API Test St",
        "city": "Austin",
        "state": "TX",
        "zip_code": "78701",
        "county": "Travis",
    },
    "tax": {
        "assessed_value": 280000,
        "market_value": 350000,
        "annual_tax": 7200,
        "tax_delinquent": True,
        "delinquent_amount": 14400,
        "years_delinquent": 2,
        "last_sale_date": "2010-06-15",
        "owner_occupied": False,
    },
    "distress": [
        {"distress_type": "foreclosure_filing", "filing_date": "2026-02-01", "amount": 320000},
        {"distress_type": "tax_lien", "filing_date": "2025-11-10", "amount": 14400},
    ],
    "vacancy": [
        {"signal": "utility_inactive", "detected_date": "2025-12-01", "confidence": 0.9},
        {"signal": "mail_forwarding", "detected_date": "2025-12-15", "confidence": 0.85},
    ],
    "equity": {
        "estimated_market_value": 350000,
        "outstanding_mortgage": 320000,
        "other_liens": 14400,
    },
    "probate": {
        "decedent_name": "John Doe Sr.",
        "filing_date": "2025-10-01",
        "status": "in_progress",
        "num_heirs": 3,
        "contested": True,
    },
}

COLD_PROPERTY = {
    "property": {
        "parcel_id": "API002",
        "address": "789 Quiet Lane",
        "city": "Houston",
        "state": "TX",
        "zip_code": "77001",
        "county": "Harris",
    },
    "tax": {
        "assessed_value": 320000,
        "market_value": 400000,
        "annual_tax": 8000,
        "last_sale_date": "2024-08-01",
        "owner_occupied": True,
    },
    "equity": {
        "estimated_market_value": 400000,
        "outstanding_mortgage": 350000,
    },
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"
    assert data["mode"] == "rules"
    print("[PASS] GET /health")


def test_predict_single_hot():
    r = client.post("/predict", json=HOT_PROPERTY)
    assert r.status_code == 200
    data = r.json()
    assert data["parcel_id"] == "API001"
    assert data["sell_probability"] >= 0.80
    assert data["tier"] == "hot"
    assert len(data["top_factors"]) > 0
    print(f"[PASS] POST /predict — HOT (score={data['sell_probability']})")


def test_predict_single_cold():
    r = client.post("/predict", json=COLD_PROPERTY)
    assert r.status_code == 200
    data = r.json()
    assert data["parcel_id"] == "API002"
    assert data["tier"] == "cold"
    print(f"[PASS] POST /predict — COLD (score={data['sell_probability']})")


def test_predict_batch():
    batch = {"items": [HOT_PROPERTY, COLD_PROPERTY]}
    r = client.post("/predict/batch", json=batch)
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 2
    scores = [d["sell_probability"] for d in data]
    assert max(scores) >= 0.80  # at least one hot
    print(f"[PASS] POST /predict/batch — {len(data)} results")


def test_list_leads():
    # Ensure some data exists
    client.post("/predict", json=HOT_PROPERTY)
    client.post("/predict", json=COLD_PROPERTY)

    r = client.get("/leads")
    assert r.status_code == 200
    data = r.json()
    assert "leads" in data
    assert "total" in data
    assert data["total"] >= 2
    # Check ordering (desc)
    leads = data["leads"]
    for i in range(len(leads) - 1):
        assert leads[i]["sell_probability"] >= leads[i + 1]["sell_probability"]
    print(f"[PASS] GET /leads — {data['total']} total")


def test_list_leads_filter_tier():
    r = client.get("/leads?tier=hot")
    assert r.status_code == 200
    data = r.json()
    for lead in data["leads"]:
        assert lead["tier"] == "hot"
    print(f"[PASS] GET /leads?tier=hot — {data['total']} results")


def test_list_leads_filter_city():
    r = client.get("/leads?city=Austin")
    assert r.status_code == 200
    data = r.json()
    for lead in data["leads"]:
        assert lead["city"] == "Austin"
    print(f"[PASS] GET /leads?city=Austin — {data['total']} results")


def test_lead_detail():
    client.post("/predict", json=HOT_PROPERTY)

    r = client.get("/leads/API001")
    assert r.status_code == 200
    data = r.json()
    assert data["parcel_id"] == "API001"
    assert data["tier"] == "hot"
    assert "top_factors" in data
    assert "prediction_history" in data
    assert len(data["prediction_history"]) >= 1
    print(f"[PASS] GET /leads/API001 — detail with {len(data['prediction_history'])} history entries")


def test_lead_detail_not_found():
    r = client.get("/leads/NONEXISTENT")
    assert r.status_code == 404
    print("[PASS] GET /leads/NONEXISTENT — 404")


def test_predict_minimal():
    """Property with no supplemental data should still score."""
    minimal = {
        "property": {
            "parcel_id": "API003",
            "address": "1 Bare Bones Rd",
            "city": "Dallas",
            "state": "TX",
            "zip_code": "75201",
            "county": "Dallas",
        }
    }
    r = client.post("/predict", json=minimal)
    assert r.status_code == 200
    data = r.json()
    assert 0 <= data["sell_probability"] <= 1
    print(f"[PASS] POST /predict — minimal input (score={data['sell_probability']})")


if __name__ == "__main__":
    print("=" * 60)
    print("  API Tests")
    print("=" * 60)
    test_health()
    test_predict_single_hot()
    test_predict_single_cold()
    test_predict_batch()
    test_list_leads()
    test_list_leads_filter_tier()
    test_list_leads_filter_city()
    test_lead_detail()
    test_lead_detail_not_found()
    test_predict_minimal()
    print("\n✓ All API tests passed.")
