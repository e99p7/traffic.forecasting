from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


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
DEFAULT_TARGET_COLUMN = "referral"
DEFAULT_DATA_FILES = ["data/traffic.csv", "data/traffic_year.csv"]


@dataclass
class TrainArtifacts:
    model_path: Path
    x_scaler_path: Path
    y_scaler_path: Path
    metadata_path: Path


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.rename(columns={col: COLUMN_ALIASES.get(col, col) for col in df.columns}).copy()

    missing = [col for col in FEATURE_COLUMNS if col not in result.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Expected columns compatible with: {FEATURE_COLUMNS}."
        )

    if "period" not in result.columns:
        raise ValueError("Input CSV must contain a date column ('Период' or 'period').")

    result["period"] = pd.to_datetime(result["period"], errors="coerce")
    if result["period"].isna().any():
        bad_rows = result[result["period"].isna()].index.tolist()[:5]
        raise ValueError(f"Column 'period' contains invalid dates. Bad rows: {bad_rows}")

    for col in FEATURE_COLUMNS:
        result[col] = pd.to_numeric(result[col], errors="coerce")
        if result[col].isna().any():
            bad_rows = result[result[col].isna()].index.tolist()[:5]
            raise ValueError(f"Column '{col}' contains non-numeric values. Bad rows: {bad_rows}")
        if (result[col] < 0).any():
            bad_rows = result[result[col] < 0].index.tolist()[:5]
            raise ValueError(f"Column '{col}' contains negative values. Bad rows: {bad_rows}")

    return result[["period"] + FEATURE_COLUMNS]


def load_dataset(csv_paths: Iterable[str]) -> tuple[pd.DataFrame, dict]:
    frames = []
    sources = []
    for csv_path in csv_paths:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        df = pd.read_csv(path)
        df = canonicalize_columns(df)
        df["source_file"] = path.name
        frames.append(df)
        sources.append(path.name)

    dataset = pd.concat(frames, ignore_index=True)
    rows_before = len(dataset)

    dataset = (
        dataset
        .sort_values("period")
        .drop_duplicates(subset=["period"], keep="last")
        .reset_index(drop=True)
    )

    duplicate_rows_removed = rows_before - len(dataset)
    stats = {
        "source_files": sources,
        "rows_before_dedup": int(rows_before),
        "rows_after_dedup": int(len(dataset)),
        "duplicate_rows_removed": int(duplicate_rows_removed),
        "date_min": str(dataset["period"].min().date()),
        "date_max": str(dataset["period"].max().date()),
    }
    return dataset.drop(columns=["source_file"], errors="ignore"), stats


def build_model(window_size: int, n_features: int) -> Sequential:
    model = Sequential(
        [
            Conv1D(300, 4, input_shape=(window_size, n_features), activation="linear"),
            Flatten(),
            Dense(300, activation="linear"),
            Dense(1, activation="linear"),
        ]
    )
    model.compile(loss="mse", optimizer=Adam(learning_rate=1e-4))
    return model


def make_generators(
    data: pd.DataFrame,
    target_column: str,
    window_size: int,
    val_len: int,
    batch_size: int,
):
    if len(data) <= window_size + val_len:
        raise ValueError(
            "Dataset is too short for the selected window_size and val_len. "
            f"Need more than {window_size + val_len} rows, got {len(data)}."
        )

    train_len = len(data) - val_len
    train_df = data.iloc[:train_len].copy()
    val_df = data.iloc[train_len - window_size:].copy()

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    x_train_raw = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train_raw = train_df[[target_column]].to_numpy(dtype=np.float32)

    x_scaler.fit(x_train_raw)
    y_scaler.fit(y_train_raw)

    x_train = x_scaler.transform(x_train_raw)
    y_train = y_scaler.transform(y_train_raw)

    x_val_raw = val_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_val_raw = val_df[[target_column]].to_numpy(dtype=np.float32)

    x_val = x_scaler.transform(x_val_raw)
    y_val = y_scaler.transform(y_val_raw)

    train_gen = TimeseriesGenerator(
        x_train,
        y_train,
        length=window_size,
        sampling_rate=1,
        batch_size=batch_size,
    )
    val_gen = TimeseriesGenerator(
        x_val,
        y_val,
        length=window_size,
        sampling_rate=1,
        batch_size=batch_size,
    )

    return train_gen, val_gen, x_scaler, y_scaler, train_len


def evaluate_model(model: Sequential, val_gen: TimeseriesGenerator, y_scaler: MinMaxScaler) -> dict:
    x_val_batches = []
    y_val_batches = []

    for x_batch, y_batch in val_gen:
        x_val_batches.append(x_batch)
        y_val_batches.append(y_batch)

    x_val = np.concatenate(x_val_batches, axis=0)
    y_val = np.concatenate(y_val_batches, axis=0)

    y_pred_scaled = model.predict(x_val, verbose=0)
    y_pred = y_scaler.inverse_transform(y_pred_scaled).reshape(-1)
    y_true = y_scaler.inverse_transform(y_val).reshape(-1)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    return {
        "mae": mae,
        "rmse": rmse,
        "n_validation_points": int(len(y_true)),
        "y_true_mean": float(np.mean(y_true)),
        "y_pred_mean": float(np.mean(y_pred)),
    }


def save_artifacts(
    artifacts_dir: Path,
    model: Sequential,
    x_scaler: MinMaxScaler,
    y_scaler: MinMaxScaler,
    metadata: dict,
) -> TrainArtifacts:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "model.keras"
    x_scaler_path = artifacts_dir / "x_scaler.joblib"
    y_scaler_path = artifacts_dir / "y_scaler.joblib"
    metadata_path = artifacts_dir / "metadata.json"

    model.save(model_path)
    joblib.dump(x_scaler, x_scaler_path)
    joblib.dump(y_scaler, y_scaler_path)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return TrainArtifacts(
        model_path=model_path,
        x_scaler_path=x_scaler_path,
        y_scaler_path=y_scaler_path,
        metadata_path=metadata_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train traffic forecasting model. By default uses data/traffic.csv and data/traffic_year.csv."
    )
    parser.add_argument(
        "--train-csvs",
        nargs="+",
        default=DEFAULT_DATA_FILES,
        help="CSV files for training. Duplicates by date are removed automatically.",
    )
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory for saving model artifacts.")
    parser.add_argument("--window-size", type=int, default=60, help="Length of input history window.")
    parser.add_argument("--val-len", type=int, default=60, help="Number of last rows reserved for validation.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--model-version", default="v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_column = COLUMN_ALIASES.get(args.target_column, args.target_column)

    if target_column not in FEATURE_COLUMNS:
        raise ValueError(f"Unsupported target column: {args.target_column}. Use one of: {FEATURE_COLUMNS}")

    data, data_stats = load_dataset(args.train_csvs)
    train_gen, val_gen, x_scaler, y_scaler, train_len = make_generators(
        data=data,
        target_column=target_column,
        window_size=args.window_size,
        val_len=args.val_len,
        batch_size=args.batch_size,
    )

    model = build_model(window_size=args.window_size, n_features=len(FEATURE_COLUMNS))
    callbacks = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
    )

    metrics = evaluate_model(model, val_gen, y_scaler)
    metadata = {
        "model_version": args.model_version,
        "window_size": args.window_size,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": target_column,
        "train_rows": int(train_len),
        "validation_rows": int(args.val_len),
        "data_stats": data_stats,
        "history": {
            "final_train_loss": float(history.history["loss"][-1]),
            "final_val_loss": float(history.history["val_loss"][-1]),
            "best_val_loss": float(min(history.history["val_loss"])),
            "epochs_ran": len(history.history["loss"]),
        },
        "metrics": metrics,
    }

    artifacts = save_artifacts(
        artifacts_dir=Path(args.artifacts_dir),
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        metadata=metadata,
    )

    print("Training complete.")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Saved model to: {artifacts.model_path}")
    print(f"Saved x_scaler to: {artifacts.x_scaler_path}")
    print(f"Saved y_scaler to: {artifacts.y_scaler_path}")
    print(f"Saved metadata to: {artifacts.metadata_path}")


if __name__ == "__main__":
    main()
