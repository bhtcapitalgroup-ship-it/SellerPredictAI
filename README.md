<div align="center">

# SellerPredict AI

### Predictive Intelligence for Off-Market Real Estate Acquisitions

**Find motivated sellers before they list. Close deals before your competition knows they exist.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-proprietary-red?style=for-the-badge)](LICENSE)
[![Build](https://img.shields.io/badge/build-passing-brightgreen?style=for-the-badge)]()
[![Tests](https://img.shields.io/badge/tests-25%20passing-brightgreen?style=for-the-badge)]()

---

*Built for the Dallas-Fort Worth market. Engineered by BHT Capital Group.*

[Live Dashboard](#live-dashboard) · [API Docs](#api-reference) · [Getting Started](#getting-started) · [Roadmap](#roadmap)

</div>

---

## The Problem

Every day, thousands of DFW homeowners are one life event away from needing to sell — foreclosure, divorce, probate, tax delinquency, vacancy. By the time these properties hit the MLS, you're competing with every investor in the metroplex and paying retail.

**The deals that build portfolios are found before anyone else is looking.**

## The Solution

SellerPredict AI continuously analyzes **five public data streams** across Dallas, Fort Worth, Arlington, Plano, and the entire DFW metroplex to identify properties with the highest probability of an off-market sale. Each property receives a **seller probability score**, a **lead tier**, a plain-language explanation of **why** the owner is likely to sell, and a **recommended offer range** — all before a listing agent ever gets the call.

```
                                    SellerPredict AI Engine

  County Tax Records ──┐
  Foreclosure Filings ─┤
  Vacancy Signals ─────┼──→  25-Feature Extraction  ──→  AI Scoring  ──→  Ranked Lead Feed
  Mortgage & Equity ───┤          per parcel              (0-100%)        with Offer Pricing
  Probate Court Data ──┘
```

---

## Live Output

```
 TIER   SCORE   ADDRESS                          WHY                                          OFFER RANGE
 ────   ─────   ───────                          ───                                          ───────────
 HOT    97.3%   123 Elm St, Dallas, TX           Foreclosure + Probate + Tax lien             $227,500 – $262,500
 HOT    91.6%   222 Cedar Blvd, Fort Worth, TX   Foreclosure + Tax lien + 3 distress signals  $338,000 – $390,000
 WARM   75.8%   666 Ash Dr, Arlington, TX        Probate + Divorce + Contested estate         $123,750 – $140,250
 WARM   57.1%   444 Walnut St, Plano, TX         Tax lien + Vacant + Delinquent taxes         $165,000 – $187,000
 COLD   10.9%   789 Pine Dr, Irving, TX          No signals — owner-occupied, recent buy      $328,000 – $368,000
```

Every score is **explainable** — you see exactly which signals are driving the prediction. No black boxes.

---

## How It Works

### 1. Data Ingestion
The system pulls from five distinct public data sources across DFW counties (Dallas, Tarrant, Collin, Denton, Rockwall):

| Data Source | What We Extract | Why It Matters |
|---|---|---|
| **County Tax Records** | Assessed value, delinquency status, ownership duration, absentee flag | Tax-delinquent and long-hold absentee owners sell at 3x the base rate |
| **Distress Filings** | Foreclosure, tax liens, bankruptcy, divorce, code violations | Each filing is a ticking clock — the owner needs liquidity |
| **Vacancy Signals** | Utility shutoff, mail forwarding, overgrown vegetation reports | Vacant properties are 5x more likely to transact off-market |
| **Equity Profiles** | Mortgage balance, LTV ratio, underwater detection | High-equity owners are motivated by gains; underwater owners by relief |
| **Probate Filings** | Estate status, heir count, contested flag | Inherited properties with multiple heirs sell within 12 months 40% of the time |

### 2. Feature Engineering
Each property is transformed into a **25-dimensional feature vector** combining signals across all five data sources — distress count, recency, severity, vacancy confidence, equity position, probate complexity, and ownership profile.

### 3. AI Scoring
Two prediction modes, switchable based on your data maturity:

| Mode | How It Works | Best For |
|---|---|---|
| **Rule-Based** | 17 weighted signals with sigmoid normalization. Fully configurable. | Day-one deployment — no training data needed |
| **Machine Learning** | Gradient-boosted classifier (LightGBM / scikit-learn) trained on historical DFW sales | Maximum accuracy when you have 6+ months of labeled outcomes |

### 4. Lead Generation
Scored properties are ranked, tiered, and enriched with a **dynamic offer range** that adjusts based on equity position and distress severity:

| Tier | Score | Discount Range | Action |
|---|---|---|---|
| **HOT** | 80%+ | 25–35% below market | Immediate outreach — direct mail, door knock, skip trace |
| **WARM** | 55–79% | 15–25% below market | Add to drip campaign — monitor for escalation |
| **COLD** | < 55% | 8–18% below market | Watch list — re-score monthly |

---

## Features Implemented

### Core Intelligence
- **Dual-mode AI predictor** — rule-based scoring (works immediately) + trainable ML model (GBM/LightGBM)
- **25-feature unified vector** built from 5 public data sources
- **17 configurable signal weights** with sigmoid normalization to [0, 1]
- **Explainable predictions** — top 5 contributing factors in plain language per property
- **Dynamic offer pricing** — recommended offer ranges based on equity + distress level
- **Historical score tracking** — monitor how a property's sell probability evolves over time

### Data Pipeline
- **CSV batch ingestion** — load tax, distress, vacancy, equity, and probate data from CSV files
- **Multi-format date parsing** — handles `YYYY-MM-DD`, `MM/DD/YYYY`, and variants automatically
- **Graceful missing data handling** — every data source is optional; the system scores with whatever is available
- **Synthetic training data generator** — produce realistic labeled datasets for ML model training

### REST API (Production-Grade)
- **`POST /predict`** — score a single property with full explainability
- **`POST /predict/batch`** — score up to 1,000 properties in one request
- **`GET /leads`** — paginated lead retrieval with filters (tier, min score, city)
- **`GET /leads/{parcel_id}`** — full lead detail with prediction history
- **`GET /dashboard`** — real-time HTML monitoring dashboard
- **`GET /health`** — system health and uptime
- **Swagger UI** at `/docs` and **ReDoc** at `/redoc`

### Infrastructure
- **Gunicorn production server** — multi-worker (8 Uvicorn workers), auto-restart after 1,000 requests
- **CORS middleware** — configurable allowed origins
- **Gzip compression** — automatic response compression (> 500 bytes)
- **Rate limiting** — sliding-window per-IP throttle (default 120 req/min)
- **Response timing** — `X-Response-Time-Ms` header on every response
- **SQLite persistence** — SQLAlchemy 2.0 ORM, PostgreSQL-ready via connection string swap
- **4 indexed database tables** — properties, predictions (append-only), leads, feature snapshots

### CLI Tooling
- **`seller-predict predict`** — batch score from CSV, output JSON or CSV
- **`seller-predict export`** — pull leads from DB with tier/score/city filters
- **`seller-predict train`** — train ML model from labeled feature data
- **`seller-predict generate-data`** — create synthetic training datasets
- **`seller-predict serve`** — launch the API server

### Testing
- **25 automated tests** across 4 test suites — pipeline, database, API, and CSV ingestion
- Full endpoint coverage including error handling and edge cases

---

## Getting Started

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11 or higher |
| pip | Latest |
| Git | Any |

### 1. Clone the Repository

```bash
git clone https://github.com/bhtcapitalgroup-ship-it/SellerPredictAI.git
cd SellerPredictAI
```

### 2. Install Dependencies

```bash
# Core installation
pip install -e ".[dev]"

# With ML model training support (optional)
pip install -e ".[ml]"
```

### 3. Run the Sample Pipeline

Score the included 10-property DFW sample dataset to see the system in action:

```bash
seller-predict predict \
  --properties data/sample/properties.csv \
  --tax data/sample/tax_records.csv \
  --distress data/sample/distress_signals.csv \
  --vacancy data/sample/vacancy_indicators.csv \
  --equity data/sample/equity_profiles.csv \
  --probate data/sample/probate_filings.csv \
  --format json
```

### 4. Start the Server

```bash
# Development (with auto-reload)
seller-predict serve --port 8000 --reload

# Production
gunicorn src.api.app:app -c gunicorn.conf.py
```

### 5. Open the Dashboard

| Page | URL |
|---|---|
| **Monitoring Dashboard** | [http://localhost:8000/dashboard](http://localhost:8000/dashboard) |
| **Interactive API Docs** | [http://localhost:8000/docs](http://localhost:8000/docs) |
| **Health Check** | [http://localhost:8000/health](http://localhost:8000/health) |

### 6. Score Your First Property

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "property": {
      "parcel_id": "DAL-2024-001",
      "address": "4521 Gaston Ave",
      "city": "Dallas",
      "state": "TX",
      "zip_code": "75246",
      "county": "Dallas"
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
      {
        "distress_type": "foreclosure_filing",
        "filing_date": "2026-02-01",
        "amount": 320000
      }
    ],
    "vacancy": [
      {
        "signal": "utility_inactive",
        "detected_date": "2025-12-01",
        "confidence": 0.9
      }
    ]
  }'
```

**Response:**

```json
{
  "parcel_id": "DAL-2024-001",
  "sell_probability": 0.9312,
  "tier": "hot",
  "top_factors": [
    "Active foreclosure filing",
    "Utilities inactive — likely vacant",
    "Delinquent property taxes",
    "Recent distress event (<90 days)",
    "Absentee owner"
  ],
  "predicted_at": "2026-03-16T19:37:41.031421"
}
```

### 7. Train the ML Model (Optional)

```bash
# Generate 5,000 synthetic training properties
seller-predict generate-data --count 5000 --sell-rate 0.15 -o data/training

# Train
seller-predict train \
  --features data/training/features.csv \
  --labels data/training/labels.csv \
  --save models/model.pkl
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Score a single property — returns probability, tier, factors |
| `POST` | `/predict/batch` | Score up to 1,000 properties in one call |
| `GET` | `/leads` | Paginated lead list — filter by `tier`, `min_score`, `city` |
| `GET` | `/leads/{parcel_id}` | Full lead detail with prediction history over time |
| `GET` | `/dashboard` | Real-time monitoring dashboard (HTML) |
| `GET` | `/health` | System status, version, predictor mode, lead count |
| `GET` | `/docs` | Interactive Swagger API documentation |
| `GET` | `/redoc` | ReDoc API documentation |

---

## Scoring Model

### Signal Weights

Signals are ranked by predictive power for off-market sales in the DFW market:

| # | Signal | Weight | Category |
|---|---|---|---|
| 1 | Active foreclosure | 0.18 | Distress |
| 2 | Probate filing | 0.15 | Probate |
| 3 | Bankruptcy | 0.12 | Distress |
| 4 | Negative equity | 0.10 | Equity |
| 5 | Tax lien | 0.08 | Distress |
| 6 | Utility shutoff | 0.08 | Vacancy |
| 7 | Delinquent taxes | 0.07 | Tax |
| 8 | Divorce filing | 0.06 | Distress |
| 9 | Mail forwarding | 0.06 | Vacancy |
| 10 | High equity (>50%) | 0.06 | Equity |
| 11 | Recent distress (<90d) | 0.06 | Distress |
| 12 | Vacancy confidence | 0.05 | Vacancy |
| 13 | Contested probate | 0.05 | Probate |
| 14 | Long hold (10+ yr) | 0.05 | Ownership |
| 15 | Multiple distress | 0.04 | Distress |
| 16 | Multiple heirs | 0.04 | Probate |
| 17 | Absentee owner | 0.04 | Ownership |

All weights are configurable in `config/weights.json`. Raw scores are compressed to [0, 1] via a sigmoid function (midpoint=0.35, steepness=6.0).

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///data/seller_predict.db` | Database connection (SQLite or PostgreSQL) |
| `ENVIRONMENT` | `production` | Runtime environment |
| `CORS_ORIGINS` | `*` | Allowed CORS origins (comma-separated) |
| `RATE_LIMIT_RPM` | `120` | Rate limit per IP per minute |
| `PORT` | `8000` | Server port |
| `WEB_CONCURRENCY` | `auto` | Gunicorn worker count |
| `LOG_LEVEL` | `info` | Logging verbosity |

---

## Architecture

```
SellerPredictAI/
├── src/
│   ├── api/                     # REST API Layer
│   │   ├── app.py               #   FastAPI app, middleware, endpoints
│   │   ├── dashboard.py         #   Real-time monitoring dashboard
│   │   └── schemas.py           #   Pydantic request/response models
│   ├── data/                    # Domain Models
│   │   ├── schema.py            #   Property, TaxRecord, DistressSignal, etc.
│   │   └── feature_builder.py   #   25-feature vector extraction
│   ├── models/                  # AI Layer
│   │   └── predictor.py         #   Dual-mode predictor (rules + GBM)
│   ├── engine/                  # Orchestration
│   │   └── lead_generator.py    #   Full pipeline: ingest → score → rank → price
│   ├── db/                      # Persistence
│   │   ├── models.py            #   SQLAlchemy ORM (4 tables)
│   │   ├── repository.py        #   CRUD + filtered queries
│   │   └── session.py           #   Engine & session management
│   ├── ingest/                  # Data Ingestion
│   │   └── csv_loader.py        #   Multi-source CSV loader
│   ├── training/                # ML Training
│   │   └── data_generator.py    #   Synthetic data generation
│   └── cli.py                   # CLI Interface (Click)
├── tests/                       # 25 tests across 4 suites
├── config/weights.json          # Signal weight configuration
├── data/sample/                 # DFW sample dataset (10 properties)
├── gunicorn.conf.py             # Production server config
└── pyproject.toml               # Package metadata
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **API** | FastAPI + Pydantic v2 | Async-ready, automatic validation, auto-generated docs |
| **Server** | Gunicorn + Uvicorn | Multi-worker production server with async I/O |
| **Database** | SQLAlchemy 2.0 + SQLite | Zero-config local, swap to PostgreSQL for production |
| **ML** | scikit-learn / LightGBM | Industry-standard gradient boosting with feature importances |
| **CLI** | Click | Clean command-line interface for batch operations |
| **Testing** | pytest + FastAPI TestClient | Full coverage with in-memory DB for speed |

---

## Roadmap

### v0.2 — DFW Data Enrichment
- [ ] Live county tax API integration (Dallas, Tarrant, Collin, Denton)
- [ ] NTREIS MLS data feed for automated comp analysis
- [ ] Skip tracing integration for owner contact enrichment
- [ ] Neighborhood-level appreciation and crime trend overlays
- [ ] Property condition scoring from permit and inspection history

### v0.3 — Advanced ML
- [ ] Time-series model — predict *when* a property will sell, not just *if*
- [ ] Ensemble scoring (XGBoost + LightGBM + logistic regression)
- [ ] Auto-retraining pipeline triggered by new closed sales data
- [ ] A/B testing framework for model comparison
- [ ] Feature drift detection and alerting

### v0.4 — Outreach Automation
- [ ] Personalized direct mail campaign generator
- [ ] SMS and email drip sequence builder
- [ ] CRM integration — Podio, HubSpot, REsimpli, InvestorFuse
- [ ] Lead assignment rules (by zip, tier, or team member)
- [ ] Response tracking and conversion attribution

### v0.5 — Scale
- [ ] PostgreSQL + Redis for multi-user, high-throughput deployment
- [ ] Webhook alerts when new HOT leads are detected
- [ ] Geographic clustering — identify distressed neighborhoods on a map
- [ ] Comps-based offer engine (replace percentage-discount model)
- [ ] Multi-market expansion beyond DFW

### v1.0 — Platform
- [ ] Multi-tenant SaaS with role-based access
- [ ] Scheduled daily/weekly batch scoring
- [ ] Mobile-responsive dashboard
- [ ] Investor performance analytics (conversion rates by tier, market, strategy)
- [ ] White-label API for partners and brokerages
- [ ] SOC 2 compliance and audit logging

---

<div align="center">

## About

**SellerPredict AI** is built by **BHT Capital Group** — a DFW-based real estate investment firm using technology to find, analyze, and close off-market deals faster than traditional methods allow.

[GitHub](https://github.com/bhtcapitalgroup-ship-it) · Proprietary Software — All Rights Reserved

---

*"The best deal is the one nobody else knows about."*

</div>
