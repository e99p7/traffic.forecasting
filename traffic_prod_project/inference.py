from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


COLUMN_ALIASES = {
    "Период": "period",
    "period": "period",
    "date": "period",
    "Прямые заходы": "direct",
    "direct": "direct",
    "Переходы из поисковых систем": "search",
    "Переходы из поисковых систем\t": "search",
    "search": "search",
    "Переходы из социальных сетей": "social",
    "social": "social",
    "Переходы по ссылкам на сайтах": "referral",
    "referral": "referral",
}

FEATURE_COLUMNS = ["direct", "search", "social", "referral"]


@dataclass
class PredictionResult:
    prediction: float
    target: str
    model_version: str


class TrafficForecastService:
    def __init__(self, artifacts_dir: str | Path = "artifacts") -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.model_path = self.artifacts_dir / "model.keras"
        self.x_scaler_path = self.artifacts_dir / "x_scaler.joblib"
        self.y_scaler_path = self.artifacts_dir / "y_scaler.joblib"
        self.metadata_path = self.artifacts_dir / "metadata.json"

        self._validate_artifacts_exist()

        self.model = load_model(self.model_path)
        self.x_scaler = joblib.load(self.x_scaler_path)
        self.y_scaler = joblib.load(self.y_scaler_path)
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

        self.window_size: int = int(self.metadata["window_size"])
        self.feature_columns: list[str] = list(self.metadata["feature_columns"])
        self.target_column: str = str(self.metadata["target_column"])
        self.model_version: str = str(self.metadata.get("model_version", "unknown"))

    def _validate_artifacts_exist(self) -> None:
        missing = [
            str(path.name)
            for path in [self.model_path, self.x_scaler_path, self.y_scaler_path, self.metadata_path]
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Artifacts are missing. Run `python train.py` first. Missing files: "
                + ", ".join(missing)
            )

    @staticmethod
    def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        result = df.rename(columns={col: COLUMN_ALIASES.get(col, col) for col in df.columns}).copy()

        missing = [col for col in FEATURE_COLUMNS if col not in result.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        for col in FEATURE_COLUMNS:
            result[col] = pd.to_numeric(result[col], errors="coerce")
            if result[col].isna().any():
                raise ValueError(f"Column '{col}' contains non-numeric values.")
            if (result[col] < 0).any():
                raise ValueError(f"Column '{col}' contains negative values.")

        if "period" in result.columns:
            result["period"] = pd.to_datetime(result["period"], errors="coerce")
            if result["period"].isna().any():
                raise ValueError("Column 'period' contains invalid dates.")
            result = (
                result
                .sort_values("period")
                .drop_duplicates(subset=["period"], keep="last")
                .reset_index(drop=True)
            )

        return result

    def predict_next(self, history_df: pd.DataFrame) -> PredictionResult:
        history_df = self._canonicalize_columns(history_df)

        if len(history_df) < self.window_size:
            raise ValueError(
                f"Need at least {self.window_size} rows of history, got {len(history_df)}."
            )

        features = history_df[self.feature_columns].tail(self.window_size).to_numpy(dtype=np.float32)
        scaled = self.x_scaler.transform(features)
        x = np.expand_dims(scaled, axis=0)

        pred_scaled = self.model.predict(x, verbose=0)
        pred = float(self.y_scaler.inverse_transform(pred_scaled)[0, 0])

        return PredictionResult(
            prediction=pred,
            target=self.target_column,
            model_version=self.model_version,
        )

    def predict_from_records(self, records: list[dict[str, Any]]) -> PredictionResult:
        history_df = pd.DataFrame(records)
        return self.predict_next(history_df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run traffic forecast inference on a CSV file.")
    parser.add_argument(
        "--input-csv",
        default="data/traffic_year.csv",
        help="CSV with recent history. Default uses bundled traffic_year.csv.",
    )
    parser.add_argument("--artifacts-dir", default="artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = TrafficForecastService(args.artifacts_dir)
    history_df = pd.read_csv(args.input_csv)
    result = service.predict_next(history_df)
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
