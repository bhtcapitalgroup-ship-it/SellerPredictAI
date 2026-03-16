# SellerPredictAI

AI-powered system that identifies homeowners likely to sell before their property hits the market. Analyzes five public data sources — tax records, distress signals, vacancy indicators, equity levels, and probate filings — to generate scored off-market leads with recommended offer ranges.

Built for real estate investors, wholesalers, and acquisition teams who want a data-driven edge on off-market deal flow.

---

## How It Works

```
Tax Records ──┐
Distress    ──┤
Vacancy     ──┼──→ Feature Extraction ──→ AI Scoring ──→ Lead Ranking ──→ Off-Market Leads
Equity      ──┤         (25 features)      (0–100%)      HOT/WARM/COLD    + Offer Range
Probate     ──┘
```

Each property is scored on **17 weighted signals** across five categories. The system outputs a **seller probability score** (0–100%), assigns a **lead tier**, explains the **top contributing factors**, and recommends an **offer price range** based on equity and distress level.

### Example Output

```
[HOT  97.3%]  123 Elm St, Austin, TX
              Factors: Active foreclosure · Probate (contested, 3 heirs) · Tax lien
              Offer:   $227,500 – $262,500

[WARM 75.8%]  666 Ash Dr, San Antonio, TX
              Factors: Probate filing · Divorce · Recent distress event
              Offer:   $123,750 – $140,250

[COLD 10.9%]  789 Pine Dr, Houston, TX
              Factors: None — owner-occupied, recent purchase
              Offer:   $328,000 – $368,000
```

---

## Features

### Dual-Mode AI Predictor
- **Rule-based scoring** — works on day one with zero training data. Uses 17 configurable weighted signals with sigmoid normalization
- **ML mode** — train a gradient-boosted classifier (LightGBM / scikit-learn) on historical sales data for higher accuracy

### Five Data Source Integration
| Source | Signals Extracted |
|---|---|
| Tax Records | Assessed/market value, delinquency, ownership duration, absentee owner |
| Distress Signals | Foreclosure, tax lien, bankruptcy, divorce, code violations, mechanic liens |
| Vacancy Indicators | Utility shutoff, mail forwarding, overgrown vegetation, occupancy status |
| Equity Profiles | LTV ratio, underwater detection, high-equity identification |
| Probate Filings | Filing status, number of heirs, contested estates |

### Explainable Predictions
Every score includes the **top 5 contributing factors** in plain language — no black-box outputs.

### Smart Offer Pricing
Recommended offer ranges adjust dynamically based on:
- Property market value and equity position
- Distress severity (HOT leads get 25–35% discount, WARM 15–25%, COLD 8–18%)

### Production-Ready API
- FastAPI with Swagger docs (`/docs`) and ReDoc (`/redoc`)
- CORS, Gzip compression, per-IP rate limiting
- Response timing headers (`X-Response-Time-Ms`)

### Live Monitoring Dashboard
Real-time HTML dashboard at `/dashboard` showing tier distribution, top leads, city breakdown, and pipeline statistics.

### Historical Score Tracking
Every prediction is stored with a timestamp. Track how a property's sell probability changes over time as new data arrives.

---

## Quick Start

### Prerequisites
- Python 3.11+
- pip

### Installation

```bash
git clone https://github.com/bhtcapitalgroup-ship-it/SellerPredictAI.git
cd SellerPredictAI
pip install -e ".[dev]"
```

For ML model training (optional):
```bash
pip install -e ".[ml]"
```

### Run the Server

```bash
# Development
seller-predict serve --port 8000 --reload

# Production (Gunicorn with 8 workers)
gunicorn src.api.app:app -c gunicorn.conf.py
```

### Open in Browser

| Page | URL |
|---|---|
| API Docs | http://localhost:8000/docs |
| Dashboard | http://localhost:8000/dashboard |
| Health Check | http://localhost:8000/health |

---

## API Reference

### Score a Property

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "property": {
      "parcel_id": "P001",
      "address": "123 Elm St",
      "city": "Austin",
      "state": "TX",
      "zip_code": "78701",
      "county": "Travis"
    },
    "tax": {
      "assessed_value": 280000,
      "market_value": 350000,
      "annual_tax": 7200,
      "tax_delinquent": true,
      "delinquent_amount": 14400,
      "years_delinquent": 2,
      "last_sale_date": "2010-06-15",
      "owner_occupied": false
    },
    "distress": [
      {"distress_type": "foreclosure_filing", "filing_date": "2026-02-01", "amount": 320000}
    ],
    "vacancy": [
      {"signal": "utility_inactive", "detected_date": "2025-12-01", "confidence": 0.9}
    ],
    "equity": {
      "estimated_market_value": 350000,
      "outstanding_mortgage": 320000,
      "other_liens": 14400
    },
    "probate": {
      "decedent_name": "John Doe Sr.",
      "filing_date": "2025-10-01",
      "status": "in_progress",
      "num_heirs": 3,
      "contested": true
    }
  }'
```

**Response:**

```json
{
  "parcel_id": "P001",
  "sell_probability": 0.9726,
  "tier": "hot",
  "top_factors": [
    "Active foreclosure filing",
    "Property in probate",
    "Tax lien recorded",
    "Utilities inactive — likely vacant",
    "Contested probate"
  ],
  "predicted_at": "2026-03-16T19:37:41.031421"
}
```

### All Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict` | Score a single property |
| `POST` | `/predict/batch` | Score up to 1,000 properties |
| `GET` | `/leads` | List leads (filter by `tier`, `min_score`, `city`) |
| `GET` | `/leads/{parcel_id}` | Lead detail with prediction history |
| `GET` | `/health` | System health check |
| `GET` | `/dashboard` | Live monitoring dashboard |
| `GET` | `/docs` | Swagger API documentation |

---

## CLI Usage

### Batch Predict from CSV Files

```bash
seller-predict predict \
  --properties data/sample/properties.csv \
  --tax data/sample/tax_records.csv \
  --distress data/sample/distress_signals.csv \
  --vacancy data/sample/vacancy_indicators.csv \
  --equity data/sample/equity_profiles.csv \
  --probate data/sample/probate_filings.csv \
  --format json \
  -o results.json
```

### Export Leads

```bash
# Export hot leads to CSV
seller-predict export --tier hot --format csv -o hot_leads.csv

# Export all leads scoring above 60%
seller-predict export --min-score 0.6 --format json
```

### Train ML Model

```bash
# Generate synthetic training data
seller-predict generate-data --count 5000 --sell-rate 0.15 -o data/training

# Train the model
seller-predict train \
  --features data/training/features.csv \
  --labels data/training/labels.csv \
  --save models/model.pkl
```

---

## Project Structure

```
SellerPredictAI/
├── src/
│   ├── api/
│   │   ├── app.py              # FastAPI application (endpoints, middleware)
│   │   ├── dashboard.py        # Real-time monitoring dashboard
│   │   └── schemas.py          # Pydantic request/response models
│   ├── data/
│   │   ├── schema.py           # Domain models (Property, TaxRecord, etc.)
│   │   └── feature_builder.py  # 25-feature vector extraction
│   ├── models/
│   │   └── predictor.py        # Dual-mode AI predictor (rules + ML)
│   ├── engine/
│   │   └── lead_generator.py   # End-to-end scoring pipeline
│   ├── db/
│   │   ├── models.py           # SQLAlchemy ORM (4 tables)
│   │   ├── repository.py       # CRUD operations
│   │   └── session.py          # Database session management
│   ├── ingest/
│   │   └── csv_loader.py       # CSV data ingestion
│   ├── training/
│   │   └── data_generator.py   # Synthetic training data
│   └── cli.py                  # Click CLI commands
├── tests/
│   ├── test_pipeline.py        # Core pipeline tests
│   ├── test_api.py             # API endpoint tests
│   ├── test_db.py              # Database layer tests
│   └── test_csv_loader.py      # CSV ingestion tests
├── config/
│   └── weights.json            # Signal weight configuration
├── data/
│   └── sample/                 # Sample CSV datasets (10 properties)
├── gunicorn.conf.py            # Production server configuration
├── pyproject.toml              # Package metadata & dependencies
└── requirements.txt            # Pip dependencies
```

---

## Scoring Model

### Signal Weights (Rule-Based Mode)

| Signal | Weight | Category |
|---|---|---|
| Foreclosure filing | 0.18 | Distress |
| Probate filing | 0.15 | Probate |
| Bankruptcy | 0.12 | Distress |
| Underwater equity | 0.10 | Equity |
| Tax lien | 0.08 | Distress |
| Utility inactive | 0.08 | Vacancy |
| Delinquent taxes | 0.07 | Tax |
| Divorce filing | 0.06 | Distress |
| Mail forwarding | 0.06 | Vacancy |
| High equity (>50%) | 0.06 | Equity |
| Distress recency (<90d) | 0.06 | Distress |
| Vacancy confidence | 0.05 | Vacancy |
| Long hold (10+ years) | 0.05 | Ownership |
| Contested probate | 0.05 | Probate |
| Multiple distress signals | 0.04 | Distress |
| Multiple heirs | 0.04 | Probate |
| Absentee owner | 0.04 | Ownership |

Raw scores are compressed to 0–1 via a sigmoid function. Weights are fully configurable in `config/weights.json`.

### Lead Tiers

| Tier | Score | Meaning |
|---|---|---|
| **HOT** | >= 80% | High likelihood to sell — prioritize outreach |
| **WARM** | >= 55% | Moderate signals — worth pursuing |
| **COLD** | < 55% | Low signals — monitor for changes |

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///data/seller_predict.db` | Database connection string |
| `ENVIRONMENT` | `production` | Environment name |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `RATE_LIMIT_RPM` | `120` | Max requests per minute per IP |
| `PORT` | `8000` | Server port (Gunicorn) |
| `WEB_CONCURRENCY` | `auto` | Number of Gunicorn workers |
| `LOG_LEVEL` | `info` | Logging level |

---

## Testing

```bash
# Run all tests
python tests/test_pipeline.py && \
python tests/test_db.py && \
python tests/test_api.py && \
python tests/test_csv_loader.py

# Or with pytest
pytest tests/ -v
```

25 tests across 4 test suites covering the core pipeline, database layer, API endpoints, and CSV ingestion.

---

## Roadmap

### v0.2 — Data Enrichment
- [ ] MLS data integration for comp analysis
- [ ] Property condition scoring from permit history
- [ ] Owner contact info enrichment (skip tracing)
- [ ] Neighborhood trend analysis (appreciation/depreciation rates)

### v0.3 — Advanced ML
- [ ] Time-series scoring — predict *when* a property will sell, not just *if*
- [ ] Ensemble model with XGBoost, LightGBM, and neural network
- [ ] Auto-retraining pipeline with new sales data as feedback loop
- [ ] A/B testing framework for model comparison
- [ ] Feature importance drift detection

### v0.4 — Outreach Automation
- [ ] Direct mail campaign generator with personalized letters
- [ ] SMS/email drip sequence integration
- [ ] CRM integration (HubSpot, Podio, REI tools)
- [ ] Lead assignment and routing rules

### v0.5 — Scale & Intelligence
- [ ] PostgreSQL migration for multi-user deployment
- [ ] Redis caching for hot lead queries
- [ ] Webhook notifications when new HOT leads are detected
- [ ] Geographic clustering — identify distressed neighborhoods
- [ ] Comps-based automated offer pricing (replace discount-based model)
- [ ] Multi-state support with jurisdiction-aware data parsing

### v1.0 — Production Platform
- [ ] Multi-tenant SaaS architecture
- [ ] Role-based access control
- [ ] Scheduled batch scoring (daily/weekly cron)
- [ ] Audit logging and compliance features
- [ ] Mobile-responsive dashboard
- [ ] White-label API for partners

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI, Pydantic v2 |
| Server | Gunicorn + Uvicorn workers |
| Database | SQLAlchemy 2.0 + SQLite (PostgreSQL-ready) |
| ML | scikit-learn / LightGBM |
| CLI | Click |
| Testing | pytest, FastAPI TestClient |
| Language | Python 3.11+ |

---

## License

Proprietary. All rights reserved.

---

Built by [BHT Capital Group](https://github.com/bhtcapitalgroup-ship-it)
