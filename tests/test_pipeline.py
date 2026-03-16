"""
End-to-end pipeline test with synthetic property data.
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.schema import (
    DistressSignal, DistressType, EquityProfile, ProbateFiling,
    ProbateStatus, Property, TaxRecord, VacancyIndicator, VacancySignal,
    LeadTier,
)
from src.data.feature_builder import FeatureBuilder
from src.models.predictor import SellerPredictor
from src.engine.lead_generator import LeadGenerator, PropertyBundle


REF_DATE = date(2026, 3, 16)


def make_bundles() -> list[PropertyBundle]:
    """Create synthetic test properties spanning hot/warm/cold tiers."""

    # --- HOT: foreclosure + vacant + probate ---
    p1 = Property("P001", "123 Elm St", "Austin", "TX", "78701", "Travis")
    t1 = TaxRecord("P001", 280_000, 350_000, 7_200, tax_delinquent=True,
                    delinquent_amount=14_400, years_delinquent=2,
                    last_sale_date=date(2010, 6, 15), owner_occupied=False)
    d1 = [
        DistressSignal("P001", DistressType.FORECLOSURE_FILING, date(2026, 2, 1), amount=320_000),
        DistressSignal("P001", DistressType.TAX_LIEN, date(2025, 11, 10), amount=14_400),
    ]
    v1 = [
        VacancyIndicator("P001", VacancySignal.UTILITY_INACTIVE, date(2025, 12, 1), confidence=0.9),
        VacancyIndicator("P001", VacancySignal.MAIL_FORWARDING, date(2025, 12, 15), confidence=0.85),
    ]
    e1 = EquityProfile("P001", 350_000, 320_000, other_liens=14_400)
    pr1 = ProbateFiling("P001", "John Doe Sr.", date(2025, 10, 1), ProbateStatus.IN_PROGRESS,
                        num_heirs=3, contested=True)

    # --- WARM: absentee owner, high equity, long hold ---
    p2 = Property("P002", "456 Oak Ave", "Dallas", "TX", "75201", "Dallas")
    t2 = TaxRecord("P002", 180_000, 240_000, 4_800,
                    last_sale_date=date(2005, 3, 20), owner_occupied=False)
    e2 = EquityProfile("P002", 240_000, 60_000)

    # --- COLD: owner-occupied, no distress, recent purchase ---
    p3 = Property("P003", "789 Pine Dr", "Houston", "TX", "77001", "Harris")
    t3 = TaxRecord("P003", 320_000, 400_000, 8_000,
                    last_sale_date=date(2024, 8, 1), owner_occupied=True)
    e3 = EquityProfile("P003", 400_000, 350_000)

    # --- WARM: probate only ---
    p4 = Property("P004", "101 Maple Ln", "San Antonio", "TX", "78201", "Bexar")
    t4 = TaxRecord("P004", 150_000, 190_000, 3_800,
                    last_sale_date=date(2015, 1, 10), owner_occupied=True)
    pr4 = ProbateFiling("P004", "Mary Smith", date(2026, 1, 15), ProbateStatus.FILED,
                        num_heirs=2)

    return [
        PropertyBundle(p1, t1, d1, v1, e1, pr1),
        PropertyBundle(p2, t2, equity=e2),
        PropertyBundle(p3, t3, equity=e3),
        PropertyBundle(p4, t4, probate=pr4),
    ]


def test_feature_builder():
    bundles = make_bundles()
    fb = FeatureBuilder(reference_date=REF_DATE)

    # Hot property
    f = fb.build("P001", bundles[0].tax, bundles[0].distress, bundles[0].vacancy,
                 bundles[0].equity, bundles[0].probate)
    assert f.has_foreclosure is True
    assert f.has_probate is True
    assert f.num_vacancy_signals == 2
    assert f.tax_delinquent is True
    assert f.is_underwater is False  # 350k - 320k - 14.4k = 15.6k (positive)
    print(f"[PASS] Feature builder — hot property")

    # Cold property
    f3 = fb.build("P003", bundles[2].tax, equity=bundles[2].equity)
    assert f3.has_foreclosure is False
    assert f3.has_probate is False
    assert f3.owner_occupied is True
    print(f"[PASS] Feature builder — cold property")


def test_predictor():
    bundles = make_bundles()
    fb = FeatureBuilder(reference_date=REF_DATE)
    predictor = SellerPredictor()

    features = [
        fb.build(b.property.parcel_id, b.tax, b.distress, b.vacancy, b.equity, b.probate)
        for b in bundles
    ]
    predictions = predictor.predict_batch(features)

    # P001 should be HOT
    assert predictions[0].tier == LeadTier.HOT, f"P001 tier={predictions[0].tier}, score={predictions[0].sell_probability}"
    assert predictions[0].sell_probability >= 0.80

    # P003 should be COLD
    assert predictions[2].tier == LeadTier.COLD, f"P003 tier={predictions[2].tier}, score={predictions[2].sell_probability}"

    for p in predictions:
        print(f"  {p.parcel_id}: score={p.sell_probability:.4f} tier={p.tier.value} factors={p.top_factors[:3]}")

    print(f"[PASS] Predictor scoring and tier assignment")


def test_lead_generator():
    bundles = make_bundles()
    engine = LeadGenerator(min_tier=LeadTier.COLD, reference_date=REF_DATE)

    leads = engine.generate(bundles)
    summary = engine.summary(leads)

    assert summary["total_leads"] >= 2  # at least P001 and one warm
    assert summary["hot"] >= 1
    assert leads[0].prediction.sell_probability >= leads[-1].prediction.sell_probability

    print(f"\n--- Lead Generation Summary ---")
    print(f"  Total leads: {summary['total_leads']}")
    print(f"  HOT: {summary['hot']}  WARM: {summary['warm']}  COLD: {summary['cold']}")
    print(f"  Avg score: {summary['avg_score']}")
    print(f"  Top factors: {summary['top_factors']}")

    print(f"\n--- Off-Market Leads ---")
    for lead in leads:
        p = lead.prediction
        offer = lead.recommended_offer_range
        offer_str = f"${offer[0]:,.0f} - ${offer[1]:,.0f}" if offer else "N/A"
        print(f"  [{p.tier.value.upper():4s}] {lead.property.address}, {lead.property.city}")
        print(f"         Score: {p.sell_probability:.4f}  Offer range: {offer_str}")
        print(f"         Why:   {', '.join(p.top_factors[:3])}")

    print(f"\n[PASS] Lead generator end-to-end")


if __name__ == "__main__":
    print("=" * 60)
    print("  Seller Prediction AI — Pipeline Tests")
    print("=" * 60)
    test_feature_builder()
    print()
    test_predictor()
    print()
    test_lead_generator()
    print("\n✓ All tests passed.")
