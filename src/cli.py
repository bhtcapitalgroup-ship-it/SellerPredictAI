"""
CLI interface for Seller Predict AI.

Usage:
    seller-predict predict --properties data/props.csv [--tax ...] [--output results.json]
    seller-predict export  --format csv --output leads.csv [--tier hot]
    seller-predict train   --features data/features.csv --labels data/labels.csv
    seller-predict serve   [--host 0.0.0.0] [--port 8000]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from src.data.schema import LeadTier
from src.db.repository import LeadRepository
from src.db.session import SessionLocal, init_db
from src.engine.lead_generator import LeadGenerator
from src.ingest.csv_loader import CSVLoader
from src.models.predictor import SellerPredictor


@click.group()
def cli():
    """Seller Predict AI — off-market lead generation system."""
    pass


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--properties", required=True, type=click.Path(exists=True), help="Properties CSV")
@click.option("--tax", type=click.Path(exists=True), default=None, help="Tax records CSV")
@click.option("--distress", type=click.Path(exists=True), default=None, help="Distress signals CSV")
@click.option("--vacancy", type=click.Path(exists=True), default=None, help="Vacancy indicators CSV")
@click.option("--equity", type=click.Path(exists=True), default=None, help="Equity profiles CSV")
@click.option("--probate", type=click.Path(exists=True), default=None, help="Probate filings CSV")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output file (default: stdout)")
@click.option("--format", "fmt", type=click.Choice(["json", "csv"]), default="json")
@click.option("--min-tier", type=click.Choice(["hot", "warm", "cold"]), default="cold")
@click.option("--persist/--no-persist", default=True, help="Save results to database")
def predict(properties, tax, distress, vacancy, equity, probate, output, fmt, min_tier, persist):
    """Run batch predictions from CSV files."""
    loader = CSVLoader()
    bundles = loader.build_bundles(
        properties_csv=Path(properties),
        tax_csv=Path(tax) if tax else None,
        distress_csv=Path(distress) if distress else None,
        vacancy_csv=Path(vacancy) if vacancy else None,
        equity_csv=Path(equity) if equity else None,
        probate_csv=Path(probate) if probate else None,
    )

    click.echo(f"Loaded {len(bundles)} properties", err=True)

    tier_map = {"hot": LeadTier.HOT, "warm": LeadTier.WARM, "cold": LeadTier.COLD}
    engine = LeadGenerator(min_tier=tier_map[min_tier])
    leads = engine.generate(bundles)
    summary = engine.summary(leads)

    # Persist to DB
    if persist:
        init_db()
        db = SessionLocal()
        try:
            repo = LeadRepository()
            repo.persist_leads(db, leads)
            click.echo(f"Persisted {len(leads)} leads to database", err=True)
        finally:
            db.close()

    # Summary to stderr
    click.echo(f"\n{'='*50}", err=True)
    click.echo(f"  Results: {summary['total_leads']} leads", err=True)
    click.echo(f"  HOT: {summary['hot']}  WARM: {summary['warm']}  COLD: {summary['cold']}", err=True)
    click.echo(f"  Avg score: {summary['avg_score']}", err=True)
    click.echo(f"  Top factors: {', '.join(summary['top_factors'][:3])}", err=True)
    click.echo(f"{'='*50}\n", err=True)

    # Output results
    results = []
    for lead in leads:
        results.append({
            "parcel_id": lead.parcel_id,
            "address": lead.property.address,
            "city": lead.property.city,
            "state": lead.property.state,
            "score": lead.prediction.sell_probability,
            "tier": lead.prediction.tier.value,
            "top_factors": lead.prediction.top_factors[:3],
            "offer_low": lead.recommended_offer_range[0] if lead.recommended_offer_range else None,
            "offer_high": lead.recommended_offer_range[1] if lead.recommended_offer_range else None,
        })

    if fmt == "json":
        text = json.dumps(results, indent=2)
    else:
        import csv
        import io
        buf = io.StringIO()
        if results:
            writer = csv.DictWriter(buf, fieldnames=results[0].keys())
            writer.writeheader()
            for r in results:
                r["top_factors"] = "; ".join(r["top_factors"])
                writer.writerow(r)
        text = buf.getvalue()

    if output:
        Path(output).write_text(text)
        click.echo(f"Written to {output}", err=True)
    else:
        click.echo(text)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--format", "fmt", type=click.Choice(["json", "csv"]), default="json")
@click.option("--output", "-o", type=click.Path(), default=None)
@click.option("--tier", type=click.Choice(["hot", "warm", "cold"]), default=None)
@click.option("--min-score", type=float, default=None)
@click.option("--city", type=str, default=None)
@click.option("--limit", type=int, default=200)
def export(fmt, output, tier, min_score, city, limit):
    """Export leads from the database."""
    init_db()
    db = SessionLocal()
    try:
        repo = LeadRepository()
        rows, total = repo.list_leads(db, tier=tier, min_score=min_score, city=city, limit=limit)

        results = []
        for row in rows:
            from src.db.models import PropertyRow
            prop = db.get(PropertyRow, row.parcel_id)
            results.append({
                "parcel_id": row.parcel_id,
                "address": prop.address if prop else "",
                "city": prop.city if prop else "",
                "state": prop.state if prop else "",
                "score": row.sell_probability,
                "tier": row.tier,
                "offer_low": row.offer_low,
                "offer_high": row.offer_high,
                "equity_pct": row.equity_pct,
            })

        click.echo(f"Exported {len(results)}/{total} leads", err=True)

        if fmt == "json":
            text = json.dumps(results, indent=2)
        else:
            import csv
            import io
            buf = io.StringIO()
            if results:
                writer = csv.DictWriter(buf, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            text = buf.getvalue()

        if output:
            Path(output).write_text(text)
            click.echo(f"Written to {output}", err=True)
        else:
            click.echo(text)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--features", required=True, type=click.Path(exists=True), help="Features CSV")
@click.option("--labels", required=True, type=click.Path(exists=True), help="Labels CSV (parcel_id, sold)")
@click.option("--save", type=click.Path(), default="models/model.pkl")
def train(features, labels, save):
    """Train ML model from labeled historical data."""
    import csv as csv_mod

    import numpy as np

    # Load features
    with open(features) as f:
        reader = csv_mod.DictReader(f)
        rows = list(reader)

    if not rows:
        click.echo("No feature rows found", err=True)
        sys.exit(1)

    def _to_float(val: str) -> float:
        if val in ("True", "true", "1"):
            return 1.0
        if val in ("False", "false", "0", ""):
            return 0.0
        return float(val)

    feature_cols = [k for k in rows[0].keys() if k != "parcel_id"]
    X = np.array([[_to_float(r.get(c, "0")) for c in feature_cols] for r in rows])

    # Load labels
    with open(labels) as f:
        reader = csv_mod.DictReader(f)
        label_map = {r["parcel_id"]: int(float(r["sold"])) for r in reader}

    y = np.array([label_map.get(r["parcel_id"], 0) for r in rows])

    click.echo(f"Training on {len(X)} samples, {len(feature_cols)} features", err=True)
    click.echo(f"Positive rate: {y.mean():.2%}", err=True)

    predictor = SellerPredictor()
    predictor.train(X, y, save_path=Path(save))

    click.echo(f"Model saved to {save}", err=True)

    # Basic eval
    try:
        from sklearn.metrics import accuracy_score, roc_auc_score
        preds = predictor._ml_model.predict(X)
        probas = predictor._ml_model.predict_proba(X)[:, 1]
        click.echo(f"Training accuracy: {accuracy_score(y, preds):.4f}", err=True)
        click.echo(f"Training AUC: {roc_auc_score(y, probas):.4f}", err=True)
    except ImportError:
        click.echo("Install scikit-learn for evaluation metrics", err=True)


# ---------------------------------------------------------------------------
# generate-data
# ---------------------------------------------------------------------------

@cli.command("generate-data")
@click.option("--output", "-o", type=click.Path(), default="data/training", help="Output directory")
@click.option("--count", "-n", type=int, default=5000, help="Number of properties to generate")
@click.option("--sell-rate", type=float, default=0.15, help="Fraction that are sellers (0-1)")
@click.option("--seed", type=int, default=42, help="Random seed")
def generate_data(output, count, sell_rate, seed):
    """Generate synthetic training data."""
    from src.training.data_generator import SyntheticDataGenerator

    gen = SyntheticDataGenerator(seed=seed)
    paths = gen.export_training_csvs(Path(output), n=count, sell_rate=sell_rate)

    click.echo(f"Generated {count} properties ({sell_rate:.0%} sellers)", err=True)
    click.echo(f"  Features: {paths['features']}", err=True)
    click.echo(f"  Labels:   {paths['labels']}", err=True)
    click.echo(f"  Properties: {paths['properties']}", err=True)


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=8000, type=int)
@click.option("--reload", "do_reload", is_flag=True, default=False)
def serve(host, port, do_reload):
    """Start the FastAPI server."""
    import uvicorn
    uvicorn.run("src.api.app:app", host=host, port=port, reload=do_reload)


if __name__ == "__main__":
    cli()
