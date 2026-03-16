"""
AI prediction model for seller likelihood scoring.

Supports two modes:
  1. Rule-based weighted scoring (no training data needed, works day one)
  2. Gradient-boosted classifier (XGBoost/LightGBM, trained on historical sales)
"""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

from src.data.schema import LeadTier, PropertyFeatures, SellerPrediction


# ---------------------------------------------------------------------------
# Weight configuration for rule-based mode
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    # Distress signals — strongest predictors
    "has_foreclosure":          0.18,
    "has_bankruptcy":           0.12,
    "has_tax_lien":             0.08,
    "has_divorce":              0.06,
    "num_distress_signals":     0.04,   # per signal
    "distress_recency_bonus":   0.06,   # if < 90 days

    # Probate — high conversion
    "has_probate":              0.15,
    "probate_contested":        0.05,
    "multiple_heirs":           0.04,   # num_heirs > 1

    # Vacancy — strong intent signal
    "has_utility_inactive":     0.08,
    "has_mail_forwarding":      0.06,
    "vacancy_confidence":       0.05,   # scaled by max_vacancy_confidence

    # Equity — motivational context
    "is_underwater":            0.10,
    "high_equity":              0.06,
    "tax_delinquent":           0.07,

    # Ownership duration
    "long_hold":                0.05,   # > 10 years
    "absentee_owner":           0.04,   # not owner-occupied
}


class SellerPredictor:
    """
    Predicts probability that a homeowner will sell in the near term.
    """

    def __init__(
        self,
        mode: str = "rules",
        model_path: Optional[Path] = None,
        weights: Optional[dict[str, float]] = None,
    ):
        self.mode = mode
        self.weights = weights or DEFAULT_WEIGHTS
        self._ml_model = None

        if mode == "ml" and model_path:
            self._load_ml_model(model_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, features: PropertyFeatures) -> SellerPrediction:
        if self.mode == "ml" and self._ml_model is not None:
            score, factors = self._predict_ml(features)
        else:
            score, factors = self._predict_rules(features)

        tier = SellerPrediction.tier_from_score(score)
        return SellerPrediction(
            parcel_id=features.parcel_id,
            sell_probability=round(score, 4),
            tier=tier,
            top_factors=factors[:5],
        )

    def predict_batch(self, feature_list: list[PropertyFeatures]) -> list[SellerPrediction]:
        return [self.predict(f) for f in feature_list]

    def train(self, X: np.ndarray, y: np.ndarray, save_path: Optional[Path] = None):
        """Train a gradient-boosted classifier on historical data."""
        try:
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )
        model.fit(X, y)
        self._ml_model = model
        self.mode = "ml"

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(model, f)

        return model

    # ------------------------------------------------------------------
    # Rule-based scoring
    # ------------------------------------------------------------------

    def _predict_rules(self, f: PropertyFeatures) -> tuple[float, list[str]]:
        contributions: list[tuple[float, str]] = []
        w = self.weights

        if f.has_foreclosure:
            contributions.append((w["has_foreclosure"], "Active foreclosure filing"))
        if f.has_bankruptcy:
            contributions.append((w["has_bankruptcy"], "Bankruptcy filing"))
        if f.has_tax_lien:
            contributions.append((w["has_tax_lien"], "Tax lien recorded"))
        if f.has_divorce:
            contributions.append((w["has_divorce"], "Divorce filing"))
        if f.num_distress_signals > 1:
            bonus = min(f.num_distress_signals - 1, 3) * w["num_distress_signals"]
            contributions.append((bonus, f"Multiple distress signals ({f.num_distress_signals})"))
        if f.days_since_latest_distress < 90:
            contributions.append((w["distress_recency_bonus"], "Recent distress event (<90 days)"))

        if f.has_probate:
            contributions.append((w["has_probate"], "Property in probate"))
        if f.probate_contested:
            contributions.append((w["probate_contested"], "Contested probate"))
        if f.num_heirs > 1:
            contributions.append((w["multiple_heirs"], f"Multiple heirs ({f.num_heirs})"))

        if f.has_utility_inactive:
            contributions.append((w["has_utility_inactive"], "Utilities inactive — likely vacant"))
        if f.has_mail_forwarding:
            contributions.append((w["has_mail_forwarding"], "Mail forwarding detected"))
        if f.num_vacancy_signals > 0:
            scaled = w["vacancy_confidence"] * f.max_vacancy_confidence
            contributions.append((scaled, f"Vacancy confidence {f.max_vacancy_confidence:.0%}"))

        if f.is_underwater:
            contributions.append((w["is_underwater"], "Negative equity — underwater"))
        if f.high_equity:
            contributions.append((w["high_equity"], "High equity (>50%) — motivated by gains"))
        if f.tax_delinquent:
            contributions.append((w["tax_delinquent"], "Delinquent property taxes"))

        if f.years_since_last_sale > 10:
            contributions.append((w["long_hold"], "Owned 10+ years"))
        if not f.owner_occupied:
            contributions.append((w["absentee_owner"], "Absentee owner"))

        raw_score = sum(c[0] for c in contributions)
        # Sigmoid squash to [0, 1]
        score = self._sigmoid(raw_score, midpoint=0.35, steepness=6.0)

        # Sort factors by contribution
        contributions.sort(key=lambda x: x[0], reverse=True)
        factors = [c[1] for c in contributions]

        return score, factors

    # ------------------------------------------------------------------
    # ML-based scoring
    # ------------------------------------------------------------------

    def _predict_ml(self, f: PropertyFeatures) -> tuple[float, list[str]]:
        vec = self._features_to_array(f)
        proba = self._ml_model.predict_proba(vec.reshape(1, -1))[0][1]

        # Use feature importances for explainability
        importances = self._ml_model.feature_importances_
        feature_names = self._feature_names()
        indexed = sorted(
            zip(importances, feature_names, vec.flatten()),
            key=lambda x: x[0] * abs(x[2]),
            reverse=True,
        )
        factors = [name for imp, name, val in indexed if val != 0][:5]

        return float(proba), factors

    def _load_ml_model(self, path: Path) -> None:
        with open(path, "rb") as f:
            self._ml_model = pickle.load(f)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(x: float, midpoint: float = 0.0, steepness: float = 1.0) -> float:
        z = steepness * (x - midpoint)
        return 1.0 / (1.0 + math.exp(-z))

    @staticmethod
    def _features_to_array(f: PropertyFeatures) -> np.ndarray:
        d = asdict(f)
        d.pop("parcel_id")
        return np.array([float(v) for v in d.values()], dtype=np.float64)

    @staticmethod
    def _feature_names() -> list[str]:
        from dataclasses import fields as dc_fields
        return [fld.name for fld in dc_fields(PropertyFeatures) if fld.name != "parcel_id"]
