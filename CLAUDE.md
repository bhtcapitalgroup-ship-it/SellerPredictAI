# CLAUDE.md — SellerPredict AI

## What This Is
AI system that predicts which DFW homeowners are likely to sell soon. Generates off-market property leads from 5 public data sources: tax records, distress signals, vacancy indicators, equity levels, and probate filings.

Built by BHT Capital Group. GitHub: https://github.com/bhtcapitalgroup-ship-it/SellerPredictAI

## Tech Stack
- **Language:** Python 3.11+ (3.14 on this machine)
- **API:** FastAPI + Pydantic v2 (src/api/app.py)
- **Server:** Gunicorn + Uvicorn workers (gunicorn.conf.py)
- **Database:** SQLAlchemy 2.0 + SQLite (data/seller_predict.db), PostgreSQL-ready
- **ML:** scikit-learn GradientBoosting (LightGBM if installed)
- **CLI:** Click (src/cli.py)
- **Tests:** pytest / direct execution (tests/)

## Project Structure
```
src/
  api/app.py           — FastAPI endpoints + middleware (CORS, rate limit, gzip)
  api/dashboard.py     — HTML monitoring dashboard at /dashboard
  api/schemas.py       — Pydantic request/response models + converters
  data/schema.py       — Domain dataclasses (Property, TaxRecord, DistressSignal, etc.)
  data/feature_builder.py — Transforms raw records → 25-feature PropertyFeatures vector
  models/predictor.py  — Dual-mode: rule-based (17 weighted signals) + ML (GBM)
  engine/lead_generator.py — Full pipeline: ingest → features → score → rank → offer price
  db/models.py         — SQLAlchemy ORM: properties, predictions, leads, feature_snapshots
  db/repository.py     — All CRUD operations
  db/session.py        — Engine + session (DATABASE_URL env var)
  ingest/csv_loader.py — Load all 5 data sources from CSV, join on parcel_id
  training/data_generator.py — Synthetic labeled data for ML training
  cli.py               — Commands: predict, export, train, generate-data, serve
config/weights.json    — Signal weights (editable without code changes)
data/sample/           — 10-property DFW sample dataset (7 CSV files)
```

## Key Commands

### Run server
```bash
# Dev
seller-predict serve --port 8000 --reload
# Prod
gunicorn src.api.app:app -c gunicorn.conf.py
```

### Run tests (all 25)
```bash
python3 tests/test_pipeline.py && DATABASE_URL="sqlite:///:memory:" python3 tests/test_db.py && DATABASE_URL="sqlite:///:memory:" python3 tests/test_api.py && python3 tests/test_csv_loader.py
```

### Batch predict from CSVs
```bash
seller-predict predict --properties data/sample/properties.csv --tax data/sample/tax_records.csv --distress data/sample/distress_signals.csv --vacancy data/sample/vacancy_indicators.csv --equity data/sample/equity_profiles.csv --probate data/sample/probate_filings.csv --format json
```

### Deploy to internet
```bash
lsof -ti:8000 | xargs kill 2>/dev/null
gunicorn src.api.app:app -c gunicorn.conf.py &
sleep 3
ngrok http 8000
```

### Train ML model
```bash
seller-predict generate-data --count 5000 -o data/training
seller-predict train --features data/training/features.csv --labels data/training/labels.csv --save models/model.pkl
```

## API Endpoints
- POST /predict — single property scoring
- POST /predict/batch — up to 1,000 properties
- GET /leads — filtered/paginated lead list
- GET /leads/{parcel_id} — detail + prediction history
- GET /dashboard — monitoring dashboard
- GET /health — system health
- GET /docs — Swagger UI

## Important Notes
- DB tests and API tests MUST run with `DATABASE_URL="sqlite:///:memory:"` to avoid touching production data
- The predictor has two modes: "rules" (default, no training needed) and "ml" (requires trained model.pkl)
- Weights in config/weights.json are hot-configurable — no code change needed
- Gunicorn spawns workers based on CPU count (capped at 8). Each worker runs its own FastAPI instance
- The lead generator's min_tier filters results. API endpoints override to COLD to always return scores
- Sample data is synthetic DFW properties — safe to commit. Real data and .db files are gitignored
- python3 (not python) on this machine
