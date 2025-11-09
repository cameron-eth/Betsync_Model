"""
Inference helpers for the Betsync NBA hybrid model bundle.
"""

from __future__ import annotations

import joblib
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from .metadata import load_metadata, require_artifact


@dataclass
class NBAModelBundle:
    """Container for the NBA hybrid models and preprocessing artifacts."""

    ml_model: any
    spread_model: any
    preprocessor: Optional[any]
    feature_list: list[str]
    metadata: object

    def prepare_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the incoming DataFrame matches the expected feature list."""
        missing = [col for col in self.feature_list if col not in raw_df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        prepared = raw_df[self.feature_list].copy()
        if self.preprocessor is not None:
            prepared = pd.DataFrame(
                self.preprocessor.transform(prepared),
                columns=self.feature_list,
                index=prepared.index,
            )
        return prepared

    def predict(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Run both ML (moneyline) and spread models and return a combined DataFrame."""
        features = self.prepare_features(raw_df)

        ml_preds = self.ml_model.predict_proba(features)[:, 1]
        spread_preds = self.spread_model.predict(features)

        results = pd.DataFrame(
            {
                "ml_home_win_probability": ml_preds,
                "spread_cover_prediction": spread_preds,
            },
            index=features.index,
        )
        return results


def load_bundle(preprocessor_expected: bool = False) -> NBAModelBundle:
    """Load the trained models, preprocessor, and metadata."""
    ml_model = joblib.load(require_artifact("ml_model.joblib"))
    spread_model = joblib.load(require_artifact("spread_model.joblib"))

    preprocessor = None
    if preprocessor_expected:
        try:
            preprocessor = joblib.load(require_artifact("preprocessor.joblib"))
        except FileNotFoundError:
            preprocessor = None

    feature_list: list[str] = []
    try:
        import json

        feature_json_path = require_artifact("feature_list.json")
        with feature_json_path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        feature_list = payload["features"] if isinstance(payload, dict) and "features" in payload else payload
    except FileNotFoundError:
        alt_path = require_artifact("feature_list.joblib")
        feature_list = joblib.load(alt_path)

    metadata = load_metadata()

    return NBAModelBundle(
        ml_model=ml_model,
        spread_model=spread_model,
        preprocessor=preprocessor,
        feature_list=feature_list,
        metadata=metadata,
    )

