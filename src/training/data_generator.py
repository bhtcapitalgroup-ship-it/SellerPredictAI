"""
Synthetic training data generator for the ML model.

Generates realistic property data with known sell/no-sell labels,
enabling model training without real historical data.
"""

from __future__ import annotations

import csv
import random
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from src.data.schema import (
    DistressSignal, DistressType, EquityProfile, ProbateFiling,
    ProbateStatus, Property, TaxRecord, VacancyIndicator, VacancySignal,
)
from src.data.feature_builder import FeatureBuilder
from src.engine.lead_generator import PropertyBundle

TEXAS_CITIES = [
    ("Austin", "Travis"), ("Dallas", "Dallas"), ("Houston", "Harris"),
    ("San Antonio", "Bexar"), ("Fort Worth", "Tarrant"), ("El Paso", "El Paso"),
    ("Arlington", "Tarrant"), ("Plano", "Collin"), ("Lubbock", "Lubbock"),
    ("Corpus Christi", "Nueces"),
]

STREETS = [
    "Main St", "Oak Ave", "Elm Dr", "Pine Rd", "Maple Ln", "Cedar Blvd",
    "Birch Way", "Walnut St", "Spruce Ave", "Ash Dr", "Willow Ct",
    "Pecan Pl", "Cypress Trl", "Magnolia Dr", "Hickory Ln",
]


class SyntheticDataGenerator:
    """Generates labeled synthetic property data for ML training."""

    def __init__(self, seed: int = 42, reference_date: Optional[date] = None):
        self.rng = random.Random(seed)
        self.reference_date = reference_date or date.today()
        self.feature_builder = FeatureBuilder(reference_date=self.reference_date)

    def generate(self, n: int = 5000, sell_rate: float = 0.15) -> tuple[list[PropertyBundle], list[int]]:
        """
        Generate n synthetic properties with labels.

        Properties labeled as 'sold' have stronger distress/vacancy/probate signals.
        Returns (bundles, labels) where labels[i] is 1 if property i sold.
        """
        bundles = []
        labels = []

        n_sellers = int(n * sell_rate)
        seller_indices = set(self.rng.sample(range(n), n_sellers))

        for i in range(n):
            will_sell = i in seller_indices
            bundle = self._generate_property(f"SYN{i:06d}", will_sell)
            bundles.append(bundle)
            labels.append(1 if will_sell else 0)

        return bundles, labels

    def export_training_csvs(
        self,
        output_dir: Path,
        n: int = 5000,
        sell_rate: float = 0.15,
    ) -> dict[str, Path]:
        """Generate data and export as CSVs ready for the train CLI command."""
        output_dir.mkdir(parents=True, exist_ok=True)
        bundles, labels = self.generate(n, sell_rate)

        # Build feature vectors
        features_rows = []
        labels_rows = []

        for bundle, label in zip(bundles, labels):
            feats = self.feature_builder.build(
                parcel_id=bundle.property.parcel_id,
                tax=bundle.tax,
                distress=bundle.distress,
                vacancy=bundle.vacancy,
                equity=bundle.equity,
                probate=bundle.probate,
            )
            d = asdict(feats)
            features_rows.append(d)
            labels_rows.append({"parcel_id": bundle.property.parcel_id, "sold": label})

        # Write features CSV
        features_path = output_dir / "features.csv"
        with open(features_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=features_rows[0].keys())
            writer.writeheader()
            for row in features_rows:
                writer.writerow(row)

        # Write labels CSV
        labels_path = output_dir / "labels.csv"
        with open(labels_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["parcel_id", "sold"])
            writer.writeheader()
            writer.writerows(labels_rows)

        # Write raw property CSVs (for the predict CLI)
        props_path = output_dir / "properties.csv"
        with open(props_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["parcel_id", "address", "city", "state", "zip_code", "county"])
            writer.writeheader()
            for b in bundles:
                writer.writerow({
                    "parcel_id": b.property.parcel_id,
                    "address": b.property.address,
                    "city": b.property.city,
                    "state": b.property.state,
                    "zip_code": b.property.zip_code,
                    "county": b.property.county,
                })

        return {
            "features": features_path,
            "labels": labels_path,
            "properties": props_path,
        }

    # ------------------------------------------------------------------
    # Internal generators
    # ------------------------------------------------------------------

    def _generate_property(self, parcel_id: str, will_sell: bool) -> PropertyBundle:
        city, county = self.rng.choice(TEXAS_CITIES)
        street = self.rng.choice(STREETS)
        num = self.rng.randint(100, 9999)

        prop = Property(
            parcel_id=parcel_id,
            address=f"{num} {street}",
            city=city,
            state="TX",
            zip_code=f"7{self.rng.randint(1000, 9999)}",
            county=county,
        )

        tax = self._gen_tax(parcel_id, will_sell)
        distress = self._gen_distress(parcel_id, will_sell)
        vacancy = self._gen_vacancy(parcel_id, will_sell)
        equity = self._gen_equity(parcel_id, tax.market_value, will_sell)
        probate = self._gen_probate(parcel_id, will_sell)

        return PropertyBundle(prop, tax, distress, vacancy, equity, probate)

    def _gen_tax(self, pid: str, will_sell: bool) -> TaxRecord:
        market_value = self.rng.randint(100_000, 800_000)
        assessed_value = int(market_value * self.rng.uniform(0.7, 0.95))

        if will_sell:
            tax_delinquent = self.rng.random() < 0.45
            years_delinquent = self.rng.randint(1, 5) if tax_delinquent else 0
            owner_occupied = self.rng.random() < 0.30
            years_ago = self.rng.randint(8, 25)
        else:
            tax_delinquent = self.rng.random() < 0.05
            years_delinquent = self.rng.randint(0, 1) if tax_delinquent else 0
            owner_occupied = self.rng.random() < 0.85
            years_ago = self.rng.randint(1, 15)

        last_sale = self.reference_date - timedelta(days=years_ago * 365)
        annual_tax = int(market_value * 0.02)
        delinquent_amount = annual_tax * years_delinquent if tax_delinquent else 0

        return TaxRecord(
            parcel_id=pid,
            assessed_value=assessed_value,
            market_value=market_value,
            annual_tax=annual_tax,
            tax_delinquent=tax_delinquent,
            delinquent_amount=delinquent_amount,
            years_delinquent=years_delinquent,
            last_sale_date=last_sale,
            owner_name=f"Owner_{pid}",
            owner_occupied=owner_occupied,
        )

    def _gen_distress(self, pid: str, will_sell: bool) -> list[DistressSignal]:
        signals = []
        if will_sell:
            # Sellers: 60% chance of at least one distress signal
            if self.rng.random() < 0.60:
                n_signals = self.rng.choices([1, 2, 3], weights=[50, 35, 15])[0]
                types = self.rng.sample(list(DistressType), min(n_signals, len(DistressType)))
                for dtype in types:
                    days_ago = self.rng.randint(5, 180)
                    signals.append(DistressSignal(
                        parcel_id=pid,
                        distress_type=dtype,
                        filing_date=self.reference_date - timedelta(days=days_ago),
                        amount=self.rng.randint(5000, 500000) if self.rng.random() < 0.7 else None,
                    ))
        else:
            # Non-sellers: 5% chance
            if self.rng.random() < 0.05:
                dtype = self.rng.choice(list(DistressType))
                signals.append(DistressSignal(
                    parcel_id=pid,
                    distress_type=dtype,
                    filing_date=self.reference_date - timedelta(days=self.rng.randint(30, 365)),
                    resolved=self.rng.random() < 0.6,
                ))
        return signals

    def _gen_vacancy(self, pid: str, will_sell: bool) -> list[VacancyIndicator]:
        signals = []
        if will_sell:
            if self.rng.random() < 0.50:
                n = self.rng.choices([1, 2, 3], weights=[40, 40, 20])[0]
                types = self.rng.sample(list(VacancySignal), min(n, len(VacancySignal)))
                for vtype in types:
                    signals.append(VacancyIndicator(
                        parcel_id=pid,
                        signal=vtype,
                        detected_date=self.reference_date - timedelta(days=self.rng.randint(10, 120)),
                        confidence=self.rng.uniform(0.6, 0.98),
                    ))
        else:
            if self.rng.random() < 0.03:
                signals.append(VacancyIndicator(
                    parcel_id=pid,
                    signal=self.rng.choice(list(VacancySignal)),
                    detected_date=self.reference_date - timedelta(days=self.rng.randint(30, 200)),
                    confidence=self.rng.uniform(0.3, 0.6),
                ))
        return signals

    def _gen_equity(self, pid: str, market_value: float, will_sell: bool) -> EquityProfile:
        if will_sell:
            # Mix of underwater and high-equity sellers
            if self.rng.random() < 0.30:
                # Underwater
                mortgage = market_value * self.rng.uniform(1.0, 1.3)
                liens = self.rng.randint(0, 30000)
            else:
                # High equity (long-time owners)
                mortgage = market_value * self.rng.uniform(0.1, 0.6)
                liens = self.rng.randint(0, 10000)
        else:
            mortgage = market_value * self.rng.uniform(0.4, 0.85)
            liens = self.rng.randint(0, 5000) if self.rng.random() < 0.1 else 0

        return EquityProfile(
            parcel_id=pid,
            estimated_market_value=market_value,
            outstanding_mortgage=round(mortgage, 2),
            other_liens=liens,
        )

    def _gen_probate(self, pid: str, will_sell: bool) -> Optional[ProbateFiling]:
        if will_sell:
            if self.rng.random() < 0.25:
                return ProbateFiling(
                    parcel_id=pid,
                    decedent_name=f"Decedent_{pid}",
                    filing_date=self.reference_date - timedelta(days=self.rng.randint(30, 300)),
                    status=self.rng.choice(list(ProbateStatus)),
                    num_heirs=self.rng.randint(1, 6),
                    contested=self.rng.random() < 0.35,
                )
        else:
            if self.rng.random() < 0.02:
                return ProbateFiling(
                    parcel_id=pid,
                    decedent_name=f"Decedent_{pid}",
                    filing_date=self.reference_date - timedelta(days=self.rng.randint(60, 500)),
                    status=ProbateStatus.SETTLED,
                    num_heirs=self.rng.randint(1, 3),
                    contested=False,
                )
        return None
