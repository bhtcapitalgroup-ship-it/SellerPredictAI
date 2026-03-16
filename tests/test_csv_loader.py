"""
Tests for the CSV ingestion module.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.schema import DistressType, ProbateStatus, VacancySignal
from src.ingest.csv_loader import CSVLoader


def _write_csv(content: str) -> Path:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    f.write(content)
    f.close()
    return Path(f.name)


def test_load_properties():
    csv = _write_csv(
        "parcel_id,address,city,state,zip_code,county,latitude,longitude\n"
        "P1,100 Main St,Austin,TX,78701,Travis,30.27,-97.74\n"
        "P2,200 Oak Ave,Dallas,TX,75201,Dallas,,\n"
    )
    loader = CSVLoader()
    props = loader.load_properties(csv)
    assert len(props) == 2
    assert props[0].parcel_id == "P1"
    assert props[0].latitude == 30.27
    assert props[1].latitude is None
    print("[PASS] load_properties")


def test_load_tax_records():
    csv = _write_csv(
        "parcel_id,assessed_value,market_value,annual_tax,tax_delinquent,delinquent_amount,years_delinquent,last_sale_date,owner_occupied\n"
        "P1,280000,350000,7200,true,14400,2,2010-06-15,false\n"
        "P2,180000,240000,4800,false,0,0,2020-01-01,true\n"
    )
    loader = CSVLoader()
    records = loader.load_tax_records(csv)
    assert len(records) == 2
    assert records["P1"].tax_delinquent is True
    assert records["P1"].delinquent_amount == 14400
    assert records["P2"].owner_occupied is True
    print("[PASS] load_tax_records")


def test_load_distress_signals():
    csv = _write_csv(
        "parcel_id,distress_type,filing_date,amount,resolved,source\n"
        "P1,foreclosure_filing,2026-02-01,320000,false,county\n"
        "P1,tax_lien,2025-11-10,14400,false,county\n"
        "P2,divorce_filing,2026-01-20,,false,court\n"
    )
    loader = CSVLoader()
    signals = loader.load_distress_signals(csv)
    assert len(signals) == 2  # 2 parcels
    assert len(signals["P1"]) == 2
    assert signals["P1"][0].distress_type == DistressType.FORECLOSURE_FILING
    assert signals["P2"][0].distress_type == DistressType.DIVORCE_FILING
    print("[PASS] load_distress_signals")


def test_load_vacancy_indicators():
    csv = _write_csv(
        "parcel_id,signal,detected_date,confidence,source\n"
        "P1,utility_inactive,2025-12-01,0.90,utility_co\n"
        "P1,mail_forwarding,2025-12-15,0.85,usps\n"
    )
    loader = CSVLoader()
    indicators = loader.load_vacancy_indicators(csv)
    assert len(indicators["P1"]) == 2
    assert indicators["P1"][0].signal == VacancySignal.UTILITY_INACTIVE
    assert indicators["P1"][1].confidence == 0.85
    print("[PASS] load_vacancy_indicators")


def test_load_equity_profiles():
    csv = _write_csv(
        "parcel_id,estimated_market_value,outstanding_mortgage,other_liens\n"
        "P1,350000,320000,14400\n"
        "P2,240000,60000,0\n"
    )
    loader = CSVLoader()
    profiles = loader.load_equity_profiles(csv)
    assert profiles["P1"].estimated_market_value == 350000
    assert profiles["P2"].outstanding_mortgage == 60000
    print("[PASS] load_equity_profiles")


def test_load_probate_filings():
    csv = _write_csv(
        "parcel_id,decedent_name,filing_date,status,num_heirs,contested\n"
        "P1,John Doe,2025-10-01,in_progress,3,true\n"
    )
    loader = CSVLoader()
    filings = loader.load_probate_filings(csv)
    assert filings["P1"].status == ProbateStatus.IN_PROGRESS
    assert filings["P1"].num_heirs == 3
    assert filings["P1"].contested is True
    print("[PASS] load_probate_filings")


def test_build_bundles():
    props_csv = _write_csv(
        "parcel_id,address,city,state,zip_code,county\n"
        "P1,100 Main,Austin,TX,78701,Travis\n"
        "P2,200 Oak,Dallas,TX,75201,Dallas\n"
    )
    tax_csv = _write_csv(
        "parcel_id,assessed_value,market_value,annual_tax,tax_delinquent,owner_occupied\n"
        "P1,280000,350000,7200,true,false\n"
    )
    distress_csv = _write_csv(
        "parcel_id,distress_type,filing_date,amount,resolved\n"
        "P1,foreclosure_filing,2026-02-01,320000,false\n"
    )

    loader = CSVLoader()
    bundles = loader.build_bundles(
        properties_csv=props_csv,
        tax_csv=tax_csv,
        distress_csv=distress_csv,
    )

    assert len(bundles) == 2
    # P1 has tax + distress
    b1 = bundles[0]
    assert b1.property.parcel_id == "P1"
    assert b1.tax is not None
    assert b1.tax.tax_delinquent is True
    assert len(b1.distress) == 1
    assert b1.equity is None  # no equity CSV provided

    # P2 has no supplemental data
    b2 = bundles[1]
    assert b2.tax is None
    assert len(b2.distress) == 0

    print("[PASS] build_bundles — join on parcel_id")


if __name__ == "__main__":
    print("=" * 60)
    print("  CSV Loader Tests")
    print("=" * 60)
    test_load_properties()
    test_load_tax_records()
    test_load_distress_signals()
    test_load_vacancy_indicators()
    test_load_equity_profiles()
    test_load_probate_filings()
    test_build_bundles()
    print("\n✓ All CSV loader tests passed.")
