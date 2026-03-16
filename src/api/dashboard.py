"""
Monitoring dashboard — serves an HTML dashboard with real-time lead analytics.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.db.models import LeadRow, PredictionRow, PropertyRow
from src.db.session import SessionLocal

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(db: Session = Depends(get_db)):
    """Serve the monitoring dashboard."""

    # Gather stats
    total_properties = db.execute(select(func.count()).select_from(PropertyRow)).scalar() or 0
    total_leads = db.execute(select(func.count()).select_from(LeadRow)).scalar() or 0
    active_leads = db.execute(
        select(func.count()).select_from(LeadRow).where(LeadRow.is_active == True)
    ).scalar() or 0
    total_predictions = db.execute(select(func.count()).select_from(PredictionRow)).scalar() or 0

    # Tier breakdown
    tier_counts = {}
    for tier in ["hot", "warm", "cold"]:
        count = db.execute(
            select(func.count()).select_from(LeadRow).where(
                LeadRow.tier == tier, LeadRow.is_active == True
            )
        ).scalar() or 0
        tier_counts[tier] = count

    # Avg score
    avg_score = db.execute(
        select(func.avg(LeadRow.sell_probability)).where(LeadRow.is_active == True)
    ).scalar() or 0

    # Top leads
    top_leads_rows = db.execute(
        select(LeadRow)
        .where(LeadRow.is_active == True)
        .order_by(LeadRow.sell_probability.desc())
        .limit(15)
    ).scalars().all()

    top_leads_html = ""
    for row in top_leads_rows:
        prop = db.get(PropertyRow, row.parcel_id)
        addr = prop.address if prop else row.parcel_id
        city = prop.city if prop else ""
        tier_class = row.tier
        offer = ""
        if row.offer_low and row.offer_high:
            offer = f"${row.offer_low:,.0f} – ${row.offer_high:,.0f}"
        top_leads_html += f"""
        <tr>
            <td><span class="tier-badge {tier_class}">{row.tier.upper()}</span></td>
            <td class="address">{addr}</td>
            <td>{city}</td>
            <td class="score">{row.sell_probability:.1%}</td>
            <td class="offer">{offer}</td>
        </tr>"""

    # City breakdown
    city_stats = db.execute(
        select(
            PropertyRow.city,
            func.count(LeadRow.id),
            func.avg(LeadRow.sell_probability),
        )
        .join(LeadRow, LeadRow.parcel_id == PropertyRow.parcel_id)
        .where(LeadRow.is_active == True)
        .group_by(PropertyRow.city)
        .order_by(func.count(LeadRow.id).desc())
        .limit(10)
    ).all()

    city_labels = [r[0] for r in city_stats]
    city_counts_data = [r[1] for r in city_stats]
    city_avg_scores = [round(r[2] * 100, 1) for r in city_stats]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Seller Predict AI — Dashboard</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #0a0e17;
        color: #e0e6ed;
        min-height: 100vh;
    }}
    .header {{
        background: linear-gradient(135deg, #1a1f35 0%, #0d1321 100%);
        border-bottom: 1px solid #1e2a3a;
        padding: 20px 32px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .header h1 {{
        font-size: 22px;
        font-weight: 600;
        color: #fff;
    }}
    .header h1 span {{ color: #4f8cff; }}
    .header .meta {{
        font-size: 13px;
        color: #6b7a8d;
    }}
    .grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        padding: 24px 32px;
    }}
    .card {{
        background: #111827;
        border: 1px solid #1e2a3a;
        border-radius: 12px;
        padding: 20px;
    }}
    .card .label {{
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #6b7a8d;
        margin-bottom: 8px;
    }}
    .card .value {{
        font-size: 32px;
        font-weight: 700;
        color: #fff;
    }}
    .card .value.hot {{ color: #ef4444; }}
    .card .value.warm {{ color: #f59e0b; }}
    .card .value.cold {{ color: #6b7280; }}
    .card .value.score {{ color: #4f8cff; }}

    .content {{
        display: grid;
        grid-template-columns: 2fr 1fr;
        gap: 16px;
        padding: 0 32px 32px;
    }}

    .table-card {{
        background: #111827;
        border: 1px solid #1e2a3a;
        border-radius: 12px;
        overflow: hidden;
    }}
    .table-card h2 {{
        padding: 16px 20px;
        font-size: 15px;
        font-weight: 600;
        border-bottom: 1px solid #1e2a3a;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
    }}
    th {{
        text-align: left;
        padding: 10px 16px;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #6b7a8d;
        border-bottom: 1px solid #1e2a3a;
    }}
    td {{
        padding: 10px 16px;
        font-size: 14px;
        border-bottom: 1px solid #0d1321;
    }}
    .address {{ font-weight: 500; }}
    .score {{ font-weight: 600; font-variant-numeric: tabular-nums; }}
    .offer {{ color: #6b7a8d; font-size: 13px; }}

    .tier-badge {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }}
    .tier-badge.hot {{ background: #ef44441a; color: #ef4444; }}
    .tier-badge.warm {{ background: #f59e0b1a; color: #f59e0b; }}
    .tier-badge.cold {{ background: #6b72801a; color: #6b7280; }}

    .sidebar-card {{
        background: #111827;
        border: 1px solid #1e2a3a;
        border-radius: 12px;
        padding: 20px;
    }}
    .sidebar-card h2 {{
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 16px;
    }}
    .tier-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid #1e2a3a;
    }}
    .tier-row:last-child {{ border: none; }}
    .tier-bar {{
        height: 8px;
        border-radius: 4px;
        margin-top: 6px;
    }}
    .tier-bar.hot {{ background: linear-gradient(90deg, #ef4444, #dc2626); }}
    .tier-bar.warm {{ background: linear-gradient(90deg, #f59e0b, #d97706); }}
    .tier-bar.cold {{ background: linear-gradient(90deg, #6b7280, #4b5563); }}

    .city-row {{
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        font-size: 14px;
        border-bottom: 1px solid #0d1321;
    }}
    .city-row:last-child {{ border: none; }}
    .city-count {{ color: #4f8cff; font-weight: 600; }}

    .pipeline-info {{
        margin-top: 20px;
        padding-top: 16px;
        border-top: 1px solid #1e2a3a;
    }}
    .pipeline-stat {{
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        font-size: 13px;
    }}
    .pipeline-stat .label {{ color: #6b7a8d; }}
</style>
</head>
<body>
    <div class="header">
        <h1><span>&#9679;</span> Seller Predict AI</h1>
        <div class="meta">Lead Generation Dashboard &middot; Real-time</div>
    </div>

    <div class="grid">
        <div class="card">
            <div class="label">Total Properties</div>
            <div class="value">{total_properties:,}</div>
        </div>
        <div class="card">
            <div class="label">Hot Leads</div>
            <div class="value hot">{tier_counts.get('hot', 0):,}</div>
        </div>
        <div class="card">
            <div class="label">Warm Leads</div>
            <div class="value warm">{tier_counts.get('warm', 0):,}</div>
        </div>
        <div class="card">
            <div class="label">Avg Score</div>
            <div class="value score">{avg_score:.1%}</div>
        </div>
    </div>

    <div class="content">
        <div class="table-card">
            <h2>Top Off-Market Leads</h2>
            <table>
                <thead>
                    <tr>
                        <th>Tier</th>
                        <th>Address</th>
                        <th>City</th>
                        <th>Score</th>
                        <th>Offer Range</th>
                    </tr>
                </thead>
                <tbody>
                    {top_leads_html}
                </tbody>
            </table>
        </div>

        <div>
            <div class="sidebar-card">
                <h2>Tier Distribution</h2>
                {"".join(f'''
                <div class="tier-row">
                    <div>
                        <span class="tier-badge {tier}">{tier.upper()}</span>
                        <div class="tier-bar {tier}" style="width: {max(count / max(active_leads, 1) * 100, 4):.0f}%"></div>
                    </div>
                    <span style="font-weight: 600; font-size: 20px;">{count}</span>
                </div>''' for tier, count in tier_counts.items())}
            </div>

            <div class="sidebar-card" style="margin-top: 16px;">
                <h2>Leads by City</h2>
                {"".join(f'''
                <div class="city-row">
                    <span>{city}</span>
                    <span class="city-count">{count}</span>
                </div>''' for city, count in zip(city_labels, city_counts_data))}
            </div>

            <div class="sidebar-card" style="margin-top: 16px;">
                <h2>Pipeline Stats</h2>
                <div class="pipeline-info">
                    <div class="pipeline-stat">
                        <span class="label">Properties Indexed</span>
                        <span>{total_properties:,}</span>
                    </div>
                    <div class="pipeline-stat">
                        <span class="label">Active Leads</span>
                        <span>{active_leads:,}</span>
                    </div>
                    <div class="pipeline-stat">
                        <span class="label">Total Predictions</span>
                        <span>{total_predictions:,}</span>
                    </div>
                    <div class="pipeline-stat">
                        <span class="label">Avg Lead Score</span>
                        <span>{avg_score:.1%}</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>"""

    return HTMLResponse(content=html)
