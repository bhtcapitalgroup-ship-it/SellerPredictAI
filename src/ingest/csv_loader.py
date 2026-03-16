"""
CSV data ingestion — loads property data from CSV files into domain objects.
"""

from __future__ import annotations

import csv
import logging
from datetime import date
from pathlib import Path
from typing import Optional

from src.data.schema import (
    DistressSignal, DistressType, EquityProfile, ProbateFiling,
    ProbateStatus, Property, TaxRecord, VacancyIndicator, VacancySignal,
)
from src.engine.lead_generator import PropertyBundle

logger = logging.getLogger(__name__)

DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%Y/%m/%d"]


def _parse_date(val: str) -> Optional[date]:
    if not val or val.strip() == "":
        return None
    for fmt in DATE_FORMATS:
        try:
            return date.fromisoformat(val) if fmt == "%Y-%m-%d" else date(*__import__("time").strptime(val, fmt)[:3])
        except (ValueError, TypeError):
            continue
    logger.warning(f"Could not parse date: {val}")
    return None


def _float_or(val: str, default: float = 0.0) -> float:
    try:
        return float(val) if val else default
    except (ValueError, TypeError):
        return default


def _int_or(val: str, default: int = 0) -> int:
    try:
        return int(float(val)) if val else default
    except (ValueError, TypeError):
        return default


def _bool_or(val: str, default: bool = False) -> bool:
    if not val:
        return default
    return val.strip().lower() in ("true", "1", "yes", "y")


class CSVLoader:
    """Load property data from CSV files and assemble into PropertyBundles."""

    def load_properties(self, path: Path) -> list[Property]:
        rows = self._read_csv(path)
        results = []
        for r in rows:
            try:
                results.append(Property(
                    parcel_id=r["parcel_id"],
                    address=r["address"],
                    city=r["city"],
                    state=r["state"],
                    zip_code=r["zip_code"],
                    county=r.get("county", ""),
                    latitude=_float_or(r.get("latitude", ""), None),
                    longitude=_float_or(r.get("longitude", ""), None),
                ))
            except KeyError as e:
                logger.warning(f"Skipping property row, missing field: {e}")
        return results

    def load_tax_records(self, path: Path) -> dict[str, TaxRecord]:
        rows = self._read_csv(path)
        result = {}
        for r in rows:
            pid = r.get("parcel_id", "")
            if not pid:
                continue
            result[pid] = TaxRecord(
                parcel_id=pid,
                assessed_value=_float_or(r.get("assessed_value", "")),
                market_value=_float_or(r.get("market_value", "")),
                annual_tax=_float_or(r.get("annual_tax", "")),
                tax_delinquent=_bool_or(r.get("tax_delinquent", "")),
                delinquent_amount=_float_or(r.get("delinquent_amount", "")),
                years_delinquent=_int_or(r.get("years_delinquent", "")),
                last_sale_date=_parse_date(r.get("last_sale_date", "")),
                last_sale_price=_float_or(r.get("last_sale_price", ""), None),
                owner_name=r.get("owner_name", ""),
                owner_occupied=_bool_or(r.get("owner_occupied", "true"), True),
            )
        return result

    def load_distress_signals(self, path: Path) -> dict[str, list[DistressSignal]]:
        rows = self._read_csv(path)
        result: dict[str, list[DistressSignal]] = {}
        for r in rows:
            pid = r.get("parcel_id", "")
            if not pid:
                continue
            try:
                dtype = DistressType(r["distress_type"])
            except (ValueError, KeyError):
                logger.warning(f"Invalid distress_type for {pid}: {r.get('distress_type')}")
                continue
            sig = DistressSignal(
                parcel_id=pid,
                distress_type=dtype,
                filing_date=_parse_date(r.get("filing_date", "")) or date.today(),
                amount=_float_or(r.get("amount", ""), None),
                resolved=_bool_or(r.get("resolved", "")),
                source=r.get("source", ""),
            )
            result.setdefault(pid, []).append(sig)
        return result

    def load_vacancy_indicators(self, path: Path) -> dict[str, list[VacancyIndicator]]:
        rows = self._read_csv(path)
        result: dict[str, list[VacancyIndicator]] = {}
        for r in rows:
            pid = r.get("parcel_id", "")
            if not pid:
                continue
            try:
                signal = VacancySignal(r["signal"])
            except (ValueError, KeyError):
                logger.warning(f"Invalid vacancy signal for {pid}: {r.get('signal')}")
                continue
            vi = VacancyIndicator(
                parcel_id=pid,
                signal=signal,
                detected_date=_parse_date(r.get("detected_date", "")) or date.today(),
                confidence=_float_or(r.get("confidence", "0.5"), 0.5),
                source=r.get("source", ""),
            )
            result.setdefault(pid, []).append(vi)
        return result

    def load_equity_profiles(self, path: Path) -> dict[str, EquityProfile]:
        rows = self._read_csv(path)
        result = {}
        for r in rows:
            pid = r.get("parcel_id", "")
            if not pid:
                continue
            result[pid] = EquityProfile(
                parcel_id=pid,
                estimated_market_value=_float_or(r.get("estimated_market_value", "")),
                outstanding_mortgage=_float_or(r.get("outstanding_mortgage", "")),
                other_liens=_float_or(r.get("other_liens", "")),
            )
        return result

    def load_probate_filings(self, path: Path) -> dict[str, ProbateFiling]:
        rows = self._read_csv(path)
        result = {}
        for r in rows:
            pid = r.get("parcel_id", "")
            if not pid:
                continue
            try:
                status = ProbateStatus(r.get("status", "filed"))
            except ValueError:
                status = ProbateStatus.FILED
            result[pid] = ProbateFiling(
                parcel_id=pid,
                decedent_name=r.get("decedent_name", ""),
                filing_date=_parse_date(r.get("filing_date", "")) or date.today(),
                status=status,
                executor_name=r.get("executor_name"),
                estimated_estate_value=_float_or(r.get("estimated_estate_value", ""), None),
                num_heirs=_int_or(r.get("num_heirs", ""), None),
                contested=_bool_or(r.get("contested", "")),
            )
        return result

    def build_bundles(
        self,
        properties_csv: Path,
        tax_csv: Optional[Path] = None,
        distress_csv: Optional[Path] = None,
        vacancy_csv: Optional[Path] = None,
        equity_csv: Optional[Path] = None,
        probate_csv: Optional[Path] = None,
    ) -> list[PropertyBundle]:
        """Master method: load all CSVs and join into PropertyBundles."""
        properties = self.load_properties(properties_csv)
        tax_map = self.load_tax_records(tax_csv) if tax_csv else {}
        distress_map = self.load_distress_signals(distress_csv) if distress_csv else {}
        vacancy_map = self.load_vacancy_indicators(vacancy_csv) if vacancy_csv else {}
        equity_map = self.load_equity_profiles(equity_csv) if equity_csv else {}
        probate_map = self.load_probate_filings(probate_csv) if probate_csv else {}

        bundles = []
        for prop in properties:
            pid = prop.parcel_id
            bundles.append(PropertyBundle(
                property=prop,
                tax=tax_map.get(pid),
                distress=distress_map.get(pid, []),
                vacancy=vacancy_map.get(pid, []),
                equity=equity_map.get(pid),
                probate=probate_map.get(pid),
            ))

        logger.info(f"Built {len(bundles)} property bundles from CSV data")
        return bundles

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
