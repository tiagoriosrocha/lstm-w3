from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import math
import random
import sys
import time
from typing import Any

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None

try:
    from sklearn.metrics import (
        explained_variance_score,
        mean_absolute_error,
        mean_squared_error,
        median_absolute_error,
        r2_score,
    )
    from sklearn.preprocessing import StandardScaler
except ImportError:
    explained_variance_score = None
    mean_absolute_error = None
    mean_squared_error = None
    median_absolute_error = None
    r2_score = None
    StandardScaler = None

try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    nn = None
    Dataset = object

try:
    import resource
except ImportError:
    resource = None


BASE_TARGET_COLUMNS = [
    "P-ANULAR",
    "P-JUS-CKGL",
    "P-MON-CKP",
    "P-TPT",
    "T-JUS-CKP",
    "T-TPT",
]

STATE_COLUMNS = [
    "ESTADO-DHSV",
    "ESTADO-M1",
    "ESTADO-M2",
    "ESTADO-PXO",
    "ESTADO-SDV-GL",
    "ESTADO-SDV-P",
    "ESTADO-W1",
    "ESTADO-W2",
    "ESTADO-XO",
]

AUX_ANALOG_COLUMNS = [
    "P-PDG",
    "QGL",
    "T-PDG",
]

CONTINUOUS_COLUMNS = BASE_TARGET_COLUMNS + AUX_ANALOG_COLUMNS


@dataclass
class PreprocessingBundle:
    target_columns: list[str]
    auxiliary_columns: list[str]
    raw_input_columns: list[str]
    input_columns: list[str]
    raw_target_input_columns: list[str]
    target_scaler_mean: list[float]
    target_scaler_scale: list[float]
    aux_scaler_mean: list[float]
    aux_scaler_scale: list[float]
    diff_scaler_mean: list[float]
    diff_scaler_scale: list[float]
    dev_scaler_mean: list[float]
    dev_scaler_scale: list[float]
    std_scaler_mean: list[float]
    std_scaler_scale: list[float]
    clip_bounds: dict[str, dict[str, float]]
    well_to_id: dict[str, int]
    split_counts: dict[str, int]
    selected_files: dict[str, list[str]]
    max_files_per_well: int | None
    rolling_window: int
    sequence_length_recommendation: int


@dataclass
class StreamingPredictionResult:
    processed_batches: int
    processed_windows: int
    total_batches_estimate: int | None
    total_windows_estimate: int | None
    elapsed_seconds: float
    export_files: list[str]
    preview_df: pd.DataFrame | None = None
    global_metrics_scaled_df: pd.DataFrame | None = None
    global_metrics_original_df: pd.DataFrame | None = None
    per_feature_scaled_df: pd.DataFrame | None = None
    per_feature_original_df: pd.DataFrame | None = None
    class_metrics_scaled_df: pd.DataFrame | None = None
    class_metrics_original_df: pd.DataFrame | None = None


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch nao esta instalado neste ambiente. "
            "Instale `torch` para executar os notebooks de treino e teste da versao3."
        )


def require_tabular_stack() -> None:
    if pd is None or StandardScaler is None:
        raise ImportError(
            "Dependencias de dados nao estao instaladas neste ambiente. "
            "Instale `pandas`, `pyarrow` e `scikit-learn` para executar o pipeline_v3."
        )


def require_parquet_engine() -> None:
    if pa is None or pq is None:
        raise ImportError(
            "A exportacao incremental em Parquet exige `pyarrow`. "
            "Instale `pyarrow` para salvar previsoes em disco."
        )


def _parse_series_metadata(file_path: str | Path) -> dict[str, str]:
    path = Path(file_path)
    class_label = path.parent.name
    stem = path.stem
    if "_" in stem:
        well_name, start_token = stem.split("_", 1)
    else:
        well_name, start_token = stem, stem
    source_type = "well"
    if well_name.startswith("SIMULATED"):
        source_type = "simulated"
    elif well_name.startswith("DRAWN"):
        source_type = "drawn"
    return {
        "class_label": class_label,
        "well_name": well_name,
        "start_token": start_token,
        "series_id": f"{class_label}__{stem}",
        "source_type": source_type,
        "file_path": str(path.resolve()),
    }


def discover_all_dataset_files(
    dataset_root: Path,
    allowed_class_labels: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    require_tabular_stack()
    rows: list[dict[str, Any]] = []
    for file_path in sorted(dataset_root.rglob("*.parquet")):
        if file_path.parent.name == "demos":
            continue
        metadata = _parse_series_metadata(file_path)
        if allowed_class_labels is not None and metadata["class_label"] not in allowed_class_labels:
            continue
        rows.append(metadata)

    if not rows:
        raise FileNotFoundError(f"Nenhum arquivo .parquet foi encontrado em {dataset_root}.")

    return (
        pd.DataFrame(rows)
        .sort_values(["class_label", "well_name", "start_token", "file_path"])
        .reset_index(drop=True)
    )


def discover_balanced_normal_files(
    dataset_root: Path,
    class_labels: tuple[str, ...] = ("0",),
    max_files_per_well: int | None = 25,
) -> pd.DataFrame:
    require_tabular_stack()
    manifest = discover_all_dataset_files(dataset_root, allowed_class_labels=class_labels)
    if max_files_per_well is None:
        return manifest

    selected_parts = []
    for _, well_df in manifest.groupby(["class_label", "well_name"], sort=True):
        selected_parts.append(well_df.head(max_files_per_well))
    return pd.concat(selected_parts, ignore_index=True)


def split_manifest_by_well(
    manifest: pd.DataFrame,
    train_frac: float = 0.7,
    validation_frac: float = 0.15,
) -> pd.DataFrame:
    require_tabular_stack()
    split_parts = []
    for _, well_df in manifest.groupby("well_name", sort=True):
        well_df = well_df.sort_values("start_token").reset_index(drop=True)
        n_files = len(well_df)

        if n_files >= 5:
            train_n = max(3, int(math.floor(n_files * train_frac)))
            validation_n = max(1, int(math.floor(n_files * validation_frac)))
            if train_n + validation_n >= n_files:
                train_n = n_files - 2
                validation_n = 1
        elif n_files == 4:
            train_n, validation_n = 2, 1
        elif n_files == 3:
            train_n, validation_n = 2, 0
        else:
            train_n, validation_n = n_files, 0

        test_n = n_files - train_n - validation_n
        split_labels = (
            ["train"] * train_n
            + ["validation"] * validation_n
            + ["test"] * test_n
        )
        assigned = well_df.copy()
        assigned["split"] = split_labels
        split_parts.append(assigned)

    return pd.concat(split_parts, ignore_index=True)


def split_manifest_by_series(
    manifest: pd.DataFrame,
    train_frac: float = 0.7,
    validation_frac: float = 0.15,
) -> pd.DataFrame:
    require_tabular_stack()
    split_parts = []
    grouped = manifest.groupby(["class_label", "well_name"], sort=True)
    for _, group_df in grouped:
        group_df = group_df.sort_values(["start_token", "file_path"]).reset_index(drop=True)
        n_files = len(group_df)

        if n_files >= 5:
            train_n = max(3, int(math.floor(n_files * train_frac)))
            validation_n = max(1, int(math.floor(n_files * validation_frac)))
            if train_n + validation_n >= n_files:
                train_n = n_files - 2
                validation_n = 1
        elif n_files == 4:
            train_n, validation_n = 2, 1
        elif n_files == 3:
            train_n, validation_n = 2, 0
        else:
            train_n, validation_n = n_files, 0

        test_n = n_files - train_n - validation_n
        split_labels = (
            ["train"] * train_n
            + ["validation"] * validation_n
            + ["test"] * test_n
        )
        assigned = group_df.copy()
        assigned["split"] = split_labels
        split_parts.append(assigned)

    return pd.concat(split_parts, ignore_index=True)


def read_parquet_with_timestamp(file_path: str | Path) -> pd.DataFrame:
    require_tabular_stack()
    df = pd.read_parquet(file_path)
    if "timestamp" not in df.columns:
        if df.index.name == "timestamp":
            df = df.reset_index()
        else:
            df = df.reset_index().rename(columns={df.index.name or "index": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def _sanitize_numeric_series(
    series: pd.Series,
    *,
    fill_strategy: str = "interpolate",
) -> pd.Series:
    require_tabular_stack()
    clean_series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)

    if fill_strategy == "ffill_mode":
        clean_series = clean_series.ffill().bfill()
        if clean_series.isna().any():
            mode_value = clean_series.mode(dropna=True)
            if len(mode_value) > 0:
                clean_series = clean_series.fillna(mode_value.iloc[0])
    else:
        clean_series = clean_series.interpolate(limit_direction="both")
        clean_series = clean_series.ffill().bfill()

    if clean_series.isna().all():
        clean_series = clean_series.fillna(0.0)
    elif clean_series.isna().any():
        clean_series = clean_series.fillna(float(clean_series.median(skipna=True)))

    return clean_series.astype(np.float64)


def clean_base_frame(
    file_path: str | Path,
    target_columns: list[str] | None = None,
    candidate_auxiliary_columns: list[str] | None = None,
) -> pd.DataFrame:
    require_tabular_stack()
    target_columns = target_columns or BASE_TARGET_COLUMNS
    candidate_auxiliary_columns = candidate_auxiliary_columns or (STATE_COLUMNS + AUX_ANALOG_COLUMNS)

    raw_df = read_parquet_with_timestamp(file_path)
    available_targets = [col for col in target_columns if col in raw_df.columns]
    available_aux = [col for col in candidate_auxiliary_columns if col in raw_df.columns]
    keep_columns = ["timestamp"] + available_targets + available_aux
    df = raw_df[keep_columns].copy()

    for column in available_targets:
        df[column] = _sanitize_numeric_series(df[column], fill_strategy="interpolate")

    for column in available_aux:
        if column in STATE_COLUMNS:
            df[column] = _sanitize_numeric_series(df[column], fill_strategy="ffill_mode")
        else:
            df[column] = _sanitize_numeric_series(df[column], fill_strategy="interpolate")

    numeric_columns = available_targets + available_aux
    for column in numeric_columns:
        values = df[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            fallback = 0.0 if np.isnan(values).all() else float(np.nanmedian(values))
            repaired = np.nan_to_num(values, nan=fallback, posinf=fallback, neginf=fallback)
            df[column] = repaired

    return df


def collect_training_reference_frame(train_manifest: pd.DataFrame) -> pd.DataFrame:
    require_tabular_stack()
    parts = []
    for file_path in train_manifest["file_path"]:
        frame = clean_base_frame(file_path)
        parts.append(frame[BASE_TARGET_COLUMNS + [col for col in STATE_COLUMNS + AUX_ANALOG_COLUMNS if col in frame.columns]])
    return pd.concat(parts, ignore_index=True)


def select_auxiliary_columns(
    training_reference_df: pd.DataFrame,
    min_unique_values: int = 2,
    min_std: float = 1e-8,
) -> pd.DataFrame:
    require_tabular_stack()
    rows = []
    for column in [col for col in STATE_COLUMNS + AUX_ANALOG_COLUMNS if col in training_reference_df.columns]:
        series = pd.to_numeric(training_reference_df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        nunique = int(series.nunique(dropna=True))
        std_value = float(series.std(skipna=True)) if series.notna().any() else 0.0
        null_pct = float(series.isna().mean() * 100)
        selected = (nunique >= min_unique_values) and (std_value > min_std)
        rows.append(
            {
                "column": column,
                "null_pct": null_pct,
                "nunique": nunique,
                "std": std_value,
                "selected_for_input": selected,
            }
        )
    return pd.DataFrame(rows).sort_values(["selected_for_input", "column"], ascending=[False, True]).reset_index(drop=True)


def _safe_scale(values: np.ndarray) -> list[float]:
    scale = np.asarray(values, dtype=np.float64)
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    return scale.tolist()


def split_single_frame_temporally(
    frame: pd.DataFrame,
    train_frac: float = 0.7,
    validation_frac: float = 0.15,
) -> dict[str, pd.DataFrame]:
    require_tabular_stack()
    n_rows = len(frame)
    if n_rows < 3:
        raise ValueError("O arquivo precisa ter pelo menos 3 linhas para formar treino, validacao e teste.")

    train_end = int(math.floor(n_rows * train_frac))
    validation_end = int(math.floor(n_rows * (train_frac + validation_frac)))

    train_end = min(max(train_end, 1), n_rows - 2)
    validation_end = min(max(validation_end, train_end + 1), n_rows - 1)

    return {
        "train": frame.iloc[:train_end].reset_index(drop=True),
        "validation": frame.iloc[train_end:validation_end].reset_index(drop=True),
        "test": frame.iloc[validation_end:].reset_index(drop=True),
    }


def _build_bundle_from_clean_frames(
    clean_frames: list[pd.DataFrame],
    auxiliary_columns: list[str],
    *,
    well_names: list[str],
    selected_files: dict[str, list[str]],
    split_counts: dict[str, int],
    max_files_per_well: int | None,
    rolling_window: int,
    sequence_length_recommendation: int,
    quantile_low: float,
    quantile_high: float,
) -> PreprocessingBundle:
    require_tabular_stack()
    target_scaler = StandardScaler()
    aux_scaler = StandardScaler()
    diff_scaler = StandardScaler()
    dev_scaler = StandardScaler()
    std_scaler = StandardScaler()

    fit_aux = len(auxiliary_columns) > 0
    train_target_parts = [frame[BASE_TARGET_COLUMNS].copy() for frame in clean_frames]
    train_continuous_parts = [
        frame[[col for col in CONTINUOUS_COLUMNS if col in frame.columns]].copy()
        for frame in clean_frames
    ]

    continuous_train_df = pd.concat(train_continuous_parts, ignore_index=True)
    clip_bounds: dict[str, dict[str, float]] = {}
    for column in [col for col in CONTINUOUS_COLUMNS if col in continuous_train_df.columns]:
        clip_bounds[column] = {
            "low": float(continuous_train_df[column].quantile(quantile_low)),
            "high": float(continuous_train_df[column].quantile(quantile_high)),
        }

    for frame in clean_frames:
        clipped = apply_clip_bounds(frame, clip_bounds)
        target_values = np.nan_to_num(
            clipped[BASE_TARGET_COLUMNS].to_numpy(dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        target_scaler.partial_fit(target_values)

        if fit_aux:
            aux_values = np.nan_to_num(
                clipped[auxiliary_columns].to_numpy(dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            aux_scaler.partial_fit(aux_values)

        diff_features, dev_features, std_features = build_derived_feature_arrays(
            clipped,
            target_columns=BASE_TARGET_COLUMNS,
            rolling_window=rolling_window,
        )
        diff_scaler.partial_fit(np.nan_to_num(diff_features.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0))
        dev_scaler.partial_fit(np.nan_to_num(dev_features.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0))
        std_scaler.partial_fit(np.nan_to_num(std_features.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0))

    raw_target_input_columns = [f"raw__{column}" for column in BASE_TARGET_COLUMNS]
    raw_aux_input_columns = [f"raw__{column}" for column in auxiliary_columns]
    input_columns = (
        raw_target_input_columns
        + raw_aux_input_columns
        + [f"diff1__{column}" for column in BASE_TARGET_COLUMNS]
        + [f"dev_roll{rolling_window}__{column}" for column in BASE_TARGET_COLUMNS]
        + [f"std_roll{rolling_window}__{column}" for column in BASE_TARGET_COLUMNS]
    )

    unique_wells = sorted(set(well_names))
    well_to_id = {well_name: idx for idx, well_name in enumerate(unique_wells)}

    return PreprocessingBundle(
        target_columns=BASE_TARGET_COLUMNS,
        auxiliary_columns=auxiliary_columns,
        raw_input_columns=BASE_TARGET_COLUMNS + auxiliary_columns,
        input_columns=input_columns,
        raw_target_input_columns=raw_target_input_columns,
        target_scaler_mean=target_scaler.mean_.tolist(),
        target_scaler_scale=_safe_scale(target_scaler.scale_),
        aux_scaler_mean=(aux_scaler.mean_.tolist() if fit_aux else []),
        aux_scaler_scale=(_safe_scale(aux_scaler.scale_) if fit_aux else []),
        diff_scaler_mean=diff_scaler.mean_.tolist(),
        diff_scaler_scale=_safe_scale(diff_scaler.scale_),
        dev_scaler_mean=dev_scaler.mean_.tolist(),
        dev_scaler_scale=_safe_scale(dev_scaler.scale_),
        std_scaler_mean=std_scaler.mean_.tolist(),
        std_scaler_scale=_safe_scale(std_scaler.scale_),
        clip_bounds=clip_bounds,
        well_to_id=well_to_id,
        split_counts=split_counts,
        selected_files=selected_files,
        max_files_per_well=max_files_per_well,
        rolling_window=rolling_window,
        sequence_length_recommendation=sequence_length_recommendation,
    )


def fit_preprocessing_bundle(
    train_manifest: pd.DataFrame,
    auxiliary_columns: list[str],
    max_files_per_well: int | None,
    rolling_window: int = 5,
    sequence_length_recommendation: int = 60,
    quantile_low: float = 0.001,
    quantile_high: float = 0.999,
) -> PreprocessingBundle:
    require_tabular_stack()
    clean_frames = [
        clean_base_frame(file_path, candidate_auxiliary_columns=auxiliary_columns)
        for file_path in train_manifest["file_path"]
    ]
    return _build_bundle_from_clean_frames(
        clean_frames=clean_frames,
        auxiliary_columns=auxiliary_columns,
        well_names=train_manifest["well_name"].tolist(),
        selected_files={
            "train": train_manifest["file_path"].tolist(),
            "validation": [],
            "test": [],
        },
        split_counts=train_manifest["well_name"].value_counts().sort_index().to_dict(),
        max_files_per_well=max_files_per_well,
        rolling_window=rolling_window,
        sequence_length_recommendation=sequence_length_recommendation,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
    )


def fit_preprocessing_bundle_from_train_frame(
    train_frame: pd.DataFrame,
    auxiliary_columns: list[str],
    *,
    well_name: str = "single_well",
    source_file: str | None = None,
    rolling_window: int = 5,
    sequence_length_recommendation: int = 60,
    quantile_low: float = 0.001,
    quantile_high: float = 0.999,
) -> PreprocessingBundle:
    require_tabular_stack()
    selected_files = {
        "train": [source_file] if source_file else [],
        "validation": [source_file] if source_file else [],
        "test": [source_file] if source_file else [],
    }
    return _build_bundle_from_clean_frames(
        clean_frames=[train_frame.reset_index(drop=True)],
        auxiliary_columns=auxiliary_columns,
        well_names=[well_name],
        selected_files=selected_files,
        split_counts={"train": 1, "validation": 1, "test": 1},
        max_files_per_well=1,
        rolling_window=rolling_window,
        sequence_length_recommendation=sequence_length_recommendation,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
    )


def apply_clip_bounds(frame: pd.DataFrame, clip_bounds: dict[str, dict[str, float]]) -> pd.DataFrame:
    require_tabular_stack()
    clipped = frame.copy()
    for column, bounds in clip_bounds.items():
        if column in clipped.columns:
            clipped[column] = clipped[column].clip(lower=bounds["low"], upper=bounds["high"])
    return clipped


def build_derived_feature_arrays(
    frame: pd.DataFrame,
    target_columns: list[str],
    rolling_window: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require_tabular_stack()
    target_df = frame[target_columns].copy()
    diff_features = target_df.diff().fillna(0.0)
    rolling_mean = target_df.rolling(window=rolling_window, min_periods=1).mean()
    rolling_std = target_df.rolling(window=rolling_window, min_periods=1).std().fillna(0.0)
    deviation_features = target_df - rolling_mean
    return (
        diff_features.to_numpy(dtype=np.float32),
        deviation_features.to_numpy(dtype=np.float32),
        rolling_std.to_numpy(dtype=np.float32),
    )


def transform_frame_to_engineered_features(
    file_path: str | Path,
    bundle: PreprocessingBundle,
) -> pd.DataFrame:
    require_tabular_stack()
    metadata = _parse_series_metadata(file_path)
    frame = clean_base_frame(file_path, candidate_auxiliary_columns=bundle.auxiliary_columns)
    return transform_clean_frame_to_engineered_features(
        frame=frame,
        bundle=bundle,
        series_id=metadata["series_id"],
        well_name=metadata["well_name"],
        class_label=metadata["class_label"],
        source_type=metadata["source_type"],
        file_path=metadata["file_path"],
    )


def transform_clean_frame_to_engineered_features(
    frame: pd.DataFrame,
    bundle: PreprocessingBundle,
    *,
    series_id: str,
    well_name: str = "single_well",
    class_label: str = "",
    source_type: str = "",
    file_path: str = "",
) -> pd.DataFrame:
    require_tabular_stack()
    clipped = apply_clip_bounds(frame, bundle.clip_bounds)

    target_df = clipped[bundle.target_columns].copy()
    target_scaled = (target_df.to_numpy(dtype=np.float32) - np.asarray(bundle.target_scaler_mean)) / np.asarray(bundle.target_scaler_scale)
    target_scaled = np.nan_to_num(target_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    target_scaled_df = pd.DataFrame(target_scaled, columns=[f"target__{column}" for column in bundle.target_columns])

    raw_target_scaled_df = pd.DataFrame(
        target_scaled,
        columns=[f"raw__{column}" for column in bundle.target_columns],
    )

    input_parts = [raw_target_scaled_df]
    if bundle.auxiliary_columns:
        aux_values = clipped[bundle.auxiliary_columns].to_numpy(dtype=np.float32)
        aux_scaled = (aux_values - np.asarray(bundle.aux_scaler_mean)) / np.asarray(bundle.aux_scaler_scale)
        aux_scaled = np.nan_to_num(aux_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        input_parts.append(pd.DataFrame(aux_scaled, columns=[f"raw__{column}" for column in bundle.auxiliary_columns]))

    diff_features, dev_features, std_features = build_derived_feature_arrays(
        clipped,
        target_columns=bundle.target_columns,
        rolling_window=bundle.rolling_window,
    )
    diff_scaled = (diff_features - np.asarray(bundle.diff_scaler_mean)) / np.asarray(bundle.diff_scaler_scale)
    dev_scaled = (dev_features - np.asarray(bundle.dev_scaler_mean)) / np.asarray(bundle.dev_scaler_scale)
    std_scaled = (std_features - np.asarray(bundle.std_scaler_mean)) / np.asarray(bundle.std_scaler_scale)
    diff_scaled = np.nan_to_num(diff_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    dev_scaled = np.nan_to_num(dev_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    std_scaled = np.nan_to_num(std_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    input_parts.append(pd.DataFrame(diff_scaled, columns=[f"diff1__{column}" for column in bundle.target_columns]))
    input_parts.append(pd.DataFrame(dev_scaled, columns=[f"dev_roll{bundle.rolling_window}__{column}" for column in bundle.target_columns]))
    input_parts.append(pd.DataFrame(std_scaled, columns=[f"std_roll{bundle.rolling_window}__{column}" for column in bundle.target_columns]))

    metadata_df = pd.DataFrame(
        {
            "series_id": [series_id] * len(clipped),
            "well_name": [well_name] * len(clipped),
            "class_label": [class_label] * len(clipped),
            "source_type": [source_type] * len(clipped),
            "well_id": [bundle.well_to_id.get(well_name, -1)] * len(clipped),
            "file_path": [file_path] * len(clipped),
            "timestamp": clipped["timestamp"].astype("datetime64[ns]"),
        }
    )

    engineered_df = pd.concat([metadata_df, target_scaled_df] + input_parts, axis=1)
    return engineered_df


def save_bundle(bundle: PreprocessingBundle, output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(bundle.__dict__, ensure_ascii=False, indent=2))


def load_bundle(bundle_path: str | Path) -> PreprocessingBundle:
    payload = json.loads(Path(bundle_path).read_text())
    return PreprocessingBundle(**payload)


def update_bundle_split_files(
    bundle: PreprocessingBundle,
    split_manifest: pd.DataFrame,
) -> PreprocessingBundle:
    require_tabular_stack()
    updated = bundle.__dict__.copy()
    updated["selected_files"] = {
        split_name: split_df["file_path"].tolist()
        for split_name, split_df in split_manifest.groupby("split", sort=False)
    }
    updated["split_counts"] = split_manifest["split"].value_counts().sort_index().to_dict()
    return PreprocessingBundle(**updated)


def write_manifest_csv(split_manifest: pd.DataFrame, output_path: str | Path) -> None:
    require_tabular_stack()
    split_manifest.to_csv(output_path, index=False)


def load_grouped_sequences(
    parquet_path: str | Path,
    input_columns: list[str],
    target_columns: list[str],
) -> list[dict[str, Any]]:
    require_tabular_stack()
    df = pd.read_parquet(parquet_path)
    groups: list[dict[str, Any]] = []
    target_feature_columns = [f"target__{column}" for column in target_columns]
    for series_id, group_df in df.groupby("series_id", sort=True):
        ordered = group_df.sort_values("timestamp").reset_index(drop=True)
        groups.append(
            {
                "series_id": series_id,
                "well_name": ordered["well_name"].iloc[0],
                "class_label": ordered["class_label"].iloc[0] if "class_label" in ordered.columns else "",
                "source_type": ordered["source_type"].iloc[0] if "source_type" in ordered.columns else "",
                "well_id": int(ordered["well_id"].iloc[0]),
                "timestamps": ordered["timestamp"].to_numpy(),
                "inputs": ordered[input_columns].to_numpy(dtype=np.float32),
                "targets": ordered[target_feature_columns].to_numpy(dtype=np.float32),
            }
        )
    return groups


def load_grouped_sequences_from_directory(
    parquet_dir: str | Path,
    input_columns: list[str],
    target_columns: list[str],
) -> list[dict[str, Any]]:
    require_tabular_stack()
    groups: list[dict[str, Any]] = []
    for parquet_path in sorted(Path(parquet_dir).glob("*.parquet")):
        groups.extend(load_grouped_sequences(parquet_path, input_columns=input_columns, target_columns=target_columns))
    return groups


def limit_groups_for_inference(
    groups: list[dict[str, Any]],
    max_series: int | None = None,
) -> list[dict[str, Any]]:
    if max_series is None:
        return groups
    return groups[: max(int(max_series), 0)]


def count_group_windows(
    groups: list[dict[str, Any]],
    sequence_length: int,
) -> int:
    return int(
        sum(
            max(len(group["targets"]) - sequence_length, 0)
            for group in groups
        )
    )


if torch is not None:
    class GroupedWindowDataset(Dataset):
        def __init__(
            self,
            groups: list[dict[str, Any]],
            sequence_length: int,
            sampled_windows: int | None = None,
            seed: int = 42,
        ) -> None:
            self.groups = groups
            self.sequence_length = sequence_length
            self.sampled_windows = sampled_windows
            self.seed = seed
            self._full_index = self._build_full_index()
            self._active_index = self._full_index.copy()
            self.resample()

        def _build_full_index(self) -> np.ndarray:
            index_rows = []
            for group_idx, group in enumerate(self.groups):
                n_rows = len(group["targets"])
                max_start = n_rows - self.sequence_length - 1
                if max_start < 0:
                    continue
                starts = np.arange(max_start + 1, dtype=np.int32)
                group_column = np.full_like(starts, fill_value=group_idx)
                index_rows.append(np.stack([group_column, starts], axis=1))
            if not index_rows:
                return np.zeros((0, 2), dtype=np.int32)
            return np.concatenate(index_rows, axis=0).astype(np.int32)

        def resample(self, epoch: int = 0) -> None:
            if self.sampled_windows is None or self.sampled_windows >= len(self._full_index):
                self._active_index = self._full_index.copy()
                return
            rng = np.random.default_rng(self.seed + epoch)
            sample_ids = rng.choice(len(self._full_index), size=self.sampled_windows, replace=False)
            self._active_index = self._full_index[np.sort(sample_ids)]

        def __len__(self) -> int:
            return len(self._active_index)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            group_idx, start_idx = self._active_index[idx]
            group = self.groups[int(group_idx)]
            end_idx = int(start_idx) + self.sequence_length
            x = group["inputs"][int(start_idx):end_idx]
            y = group["targets"][end_idx]
            timestamp = group["timestamps"][end_idx]
            return {
                "x": torch.tensor(x, dtype=torch.float32),
                "y": torch.tensor(y, dtype=torch.float32),
                "well_id": torch.tensor(group["well_id"], dtype=torch.long),
                "group_idx": torch.tensor(int(group_idx), dtype=torch.long),
                "step_idx": torch.tensor(end_idx, dtype=torch.long),
                "timestamp_ns": torch.tensor(pd.Timestamp(timestamp).value, dtype=torch.long),
            }


    class ConvResidualBlock(nn.Module):
        def __init__(self, channels: int, dilation: int, dropout: float) -> None:
            super().__init__()
            padding = dilation
            self.block = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
                nn.Dropout(dropout),
            )
            self.norm = nn.BatchNorm1d(channels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            out = self.block(x)
            out = out[..., : residual.size(-1)]
            out = self.norm(out + residual)
            return torch.relu(out)


    class HybridResidualForecaster(nn.Module):
        def __init__(
            self,
            input_size: int,
            target_size: int,
            well_count: int,
            raw_target_positions: list[int],
            model_dim: int = 128,
            gru_hidden_size: int = 128,
            gru_layers: int = 2,
            well_embedding_dim: int = 16,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.raw_target_positions = raw_target_positions
            self.input_norm = nn.LayerNorm(input_size)
            self.input_projection = nn.Linear(input_size, model_dim)
            self.conv_blocks = nn.ModuleList(
                [
                    ConvResidualBlock(model_dim, dilation=1, dropout=dropout),
                    ConvResidualBlock(model_dim, dilation=2, dropout=dropout),
                    ConvResidualBlock(model_dim, dilation=4, dropout=dropout),
                ]
            )
            self.gru = nn.GRU(
                input_size=model_dim,
                hidden_size=gru_hidden_size,
                num_layers=gru_layers,
                dropout=dropout if gru_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=True,
            )
            self.attention_score = nn.Linear(gru_hidden_size * 2, 1)
            self.well_embedding = nn.Embedding(num_embeddings=max(well_count, 1), embedding_dim=well_embedding_dim)
            self.head = nn.Sequential(
                nn.Linear(gru_hidden_size * 2 + model_dim + well_embedding_dim + target_size, 192),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(192, 96),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(96, target_size),
            )

        def forward(
            self,
            x: torch.Tensor,
            well_id: torch.Tensor,
            return_attention: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            last_target = x[:, -1, self.raw_target_positions]
            encoded = self.input_projection(self.input_norm(x))
            conv_features = encoded.transpose(1, 2)
            for block in self.conv_blocks:
                conv_features = block(conv_features)
            conv_last = conv_features[:, :, -1]
            temporal_features = conv_features.transpose(1, 2)
            gru_out, _ = self.gru(temporal_features)
            attention_logits = self.attention_score(gru_out)
            attention_weights = torch.softmax(attention_logits, dim=1)
            attention_context = (gru_out * attention_weights).sum(dim=1)
            well_context = self.well_embedding(well_id)
            fused = torch.cat([attention_context, conv_last, well_context, last_target], dim=1)
            delta_prediction = self.head(fused)
            prediction = last_target + delta_prediction
            if return_attention:
                return prediction, delta_prediction, attention_weights.squeeze(-1)
            return prediction, delta_prediction
else:
    class GroupedWindowDataset:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


    class ConvResidualBlock:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


    class HybridResidualForecaster:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


def composite_forecasting_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    last_target: torch.Tensor,
    delta_weight: float = 0.7,
    absolute_weight: float = 0.3,
) -> tuple[torch.Tensor, dict[str, float]]:
    require_torch()
    true_delta = target - last_target
    pred_delta = prediction - last_target
    delta_loss = torch.nn.functional.smooth_l1_loss(pred_delta, true_delta)
    absolute_loss = torch.nn.functional.mse_loss(prediction, target)
    total_loss = delta_weight * delta_loss + absolute_weight * absolute_loss
    metrics = {
        "delta_loss": float(delta_loss.detach().cpu().item()),
        "absolute_loss": float(absolute_loss.detach().cpu().item()),
        "total_loss": float(total_loss.detach().cpu().item()),
    }
    return total_loss, metrics


def run_epoch(
    model: HybridResidualForecaster,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    raw_target_positions: list[int],
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
) -> dict[str, Any]:
    require_torch()
    is_training = optimizer is not None
    model.train(is_training)

    loss_values = []
    delta_losses = []
    absolute_losses = []
    predictions = []
    targets = []
    persistence_predictions = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        well_id = batch["well_id"].to(device)
        last_target = x[:, -1, raw_target_positions]

        with torch.set_grad_enabled(is_training):
            prediction, _ = model(x, well_id)
            loss, loss_metrics = composite_forecasting_loss(prediction, y, last_target)
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        loss_values.append(loss_metrics["total_loss"])
        delta_losses.append(loss_metrics["delta_loss"])
        absolute_losses.append(loss_metrics["absolute_loss"])
        predictions.append(prediction.detach().cpu().numpy())
        targets.append(y.detach().cpu().numpy())
        persistence_predictions.append(last_target.detach().cpu().numpy())

    y_true = np.concatenate(targets, axis=0)
    y_pred = np.concatenate(predictions, axis=0)
    y_persist = np.concatenate(persistence_predictions, axis=0)
    model_mae = mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1))
    persistence_mae = mean_absolute_error(y_true.reshape(-1), y_persist.reshape(-1))
    improvement_pct = (persistence_mae - model_mae) / persistence_mae * 100 if persistence_mae != 0 else np.nan

    epoch_metrics = {
        "loss": float(np.mean(loss_values)),
        "delta_loss": float(np.mean(delta_losses)),
        "absolute_loss": float(np.mean(absolute_losses)),
        "model_mae": float(model_mae),
        "persistence_mae": float(persistence_mae),
        "mae_improvement_pct": float(improvement_pct),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_persist": y_persist,
    }

    if scheduler is not None and not is_training:
        scheduler.step(epoch_metrics["model_mae"])

    return epoch_metrics


@dataclass
class RegressionMetricAccumulator:
    feature_names: list[str]
    count: int = 0
    sum_abs_error: np.ndarray = field(init=False)
    sum_sq_error: np.ndarray = field(init=False)
    sum_error: np.ndarray = field(init=False)
    sum_true: np.ndarray = field(init=False)
    sum_true_sq: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        feature_count = len(self.feature_names)
        self.sum_abs_error = np.zeros(feature_count, dtype=np.float64)
        self.sum_sq_error = np.zeros(feature_count, dtype=np.float64)
        self.sum_error = np.zeros(feature_count, dtype=np.float64)
        self.sum_true = np.zeros(feature_count, dtype=np.float64)
        self.sum_true_sq = np.zeros(feature_count, dtype=np.float64)

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        if y_true_arr.size == 0:
            return
        if y_true_arr.shape != y_pred_arr.shape:
            raise ValueError("As matrizes de y_true e y_pred precisam ter o mesmo formato.")
        if y_true_arr.ndim != 2:
            raise ValueError("As matrizes de métricas precisam ter duas dimensões: [amostras, features].")

        true64 = y_true_arr.astype(np.float64, copy=False)
        pred64 = y_pred_arr.astype(np.float64, copy=False)
        error64 = pred64 - true64

        self.count += int(true64.shape[0])
        self.sum_abs_error += np.abs(error64).sum(axis=0)
        self.sum_sq_error += np.square(error64).sum(axis=0)
        self.sum_error += error64.sum(axis=0)
        self.sum_true += true64.sum(axis=0)
        self.sum_true_sq += np.square(true64).sum(axis=0)

    def _per_feature_r2(self) -> np.ndarray:
        if self.count == 0:
            return np.full(len(self.feature_names), np.nan, dtype=np.float64)
        sst = self.sum_true_sq - np.square(self.sum_true) / float(self.count)
        r2_values = np.full(len(self.feature_names), np.nan, dtype=np.float64)
        valid_mask = sst > 1e-12
        r2_values[valid_mask] = 1.0 - (self.sum_sq_error[valid_mask] / sst[valid_mask])
        perfect_mask = (~valid_mask) & (self.sum_sq_error <= 1e-12)
        r2_values[perfect_mask] = 1.0
        return r2_values

    def to_global_metrics(self, label: str) -> dict[str, float | str]:
        total_values = self.count * len(self.feature_names)
        if total_values == 0:
            return {
                "modelo": label,
                "MSE": np.nan,
                "RMSE": np.nan,
                "MAE": np.nan,
                "R2_medio": np.nan,
            }

        mse = float(self.sum_sq_error.sum() / total_values)
        mae = float(self.sum_abs_error.sum() / total_values)
        r2_values = self._per_feature_r2()
        r2_mean = float(np.nanmean(r2_values)) if not np.isnan(r2_values).all() else np.nan
        return {
            "modelo": label,
            "MSE": mse,
            "RMSE": float(math.sqrt(mse)),
            "MAE": mae,
            "R2_medio": r2_mean,
        }

    def to_per_feature_metrics(
        self,
        baseline: RegressionMetricAccumulator | None = None,
    ) -> pd.DataFrame:
        require_tabular_stack()
        rows = []
        model_r2 = self._per_feature_r2()
        baseline_r2 = baseline._per_feature_r2() if baseline is not None else None

        for feature_idx, feature_name in enumerate(self.feature_names):
            if self.count == 0:
                model_mse = np.nan
                model_mae = np.nan
                model_bias = np.nan
                model_rmse = np.nan
                residual_std = np.nan
            else:
                model_mse = float(self.sum_sq_error[feature_idx] / self.count)
                model_mae = float(self.sum_abs_error[feature_idx] / self.count)
                model_bias = float(self.sum_error[feature_idx] / self.count)
                model_rmse = float(math.sqrt(model_mse))
                residual_std = float(math.sqrt(max(model_mse - model_bias ** 2, 0.0)))

            baseline_mse = np.nan
            baseline_mae = np.nan
            baseline_rmse = np.nan
            mae_improvement_pct = np.nan
            rmse_improvement_pct = np.nan
            baseline_feature_r2 = np.nan

            if baseline is not None and baseline.count > 0:
                baseline_mse = float(baseline.sum_sq_error[feature_idx] / baseline.count)
                baseline_mae = float(baseline.sum_abs_error[feature_idx] / baseline.count)
                baseline_rmse = float(math.sqrt(baseline_mse))
                baseline_feature_r2 = float(baseline_r2[feature_idx]) if baseline_r2 is not None else np.nan
                if baseline_mae != 0:
                    mae_improvement_pct = float((baseline_mae - model_mae) / baseline_mae * 100.0)
                if baseline_rmse != 0:
                    rmse_improvement_pct = float((baseline_rmse - model_rmse) / baseline_rmse * 100.0)

            rows.append(
                {
                    "feature": feature_name,
                    "model_mae": model_mae,
                    "baseline_mae": baseline_mae,
                    "mae_melhora_pct": mae_improvement_pct,
                    "model_mse": model_mse,
                    "baseline_mse": baseline_mse,
                    "model_rmse": model_rmse,
                    "baseline_rmse": baseline_rmse,
                    "rmse_melhora_pct": rmse_improvement_pct,
                    "model_r2": float(model_r2[feature_idx]),
                    "baseline_r2": baseline_feature_r2,
                    "model_bias": model_bias,
                    "model_residual_std": residual_std,
                }
            )

        return pd.DataFrame(rows).sort_values("model_rmse", ascending=False).reset_index(drop=True)


@dataclass
class PredictionChunkWriter:
    output_dir: Path
    chunk_rows: int
    export_prefix: str = "predictions_part"
    files: list[str] = field(default_factory=list)
    _buffer: dict[str, list[np.ndarray]] = field(default_factory=dict, init=False)
    _buffered_rows: int = 0
    _part_idx: int = 0

    def __post_init__(self) -> None:
        require_parquet_engine()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.chunk_rows <= 0:
            raise ValueError("export_chunk_rows precisa ser maior que zero.")

    def append(self, columns: dict[str, np.ndarray]) -> None:
        if not columns:
            return
        row_count = len(next(iter(columns.values())))
        if row_count == 0:
            return
        if not self._buffer:
            self._buffer = {column_name: [] for column_name in columns}
        for column_name, values in columns.items():
            self._buffer[column_name].append(values)
        self._buffered_rows += row_count
        if self._buffered_rows >= self.chunk_rows:
            self.flush()

    def flush(self) -> None:
        if self._buffered_rows == 0:
            return

        table_columns = {}
        for column_name, parts in self._buffer.items():
            values = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
            table_columns[column_name] = _to_arrow_array(values)

        self._part_idx += 1
        output_path = self.output_dir / f"{self.export_prefix}_{self._part_idx:04d}.parquet"
        pq.write_table(pa.table(table_columns), output_path)
        self.files.append(str(output_path))
        self._buffer = {}
        self._buffered_rows = 0


def _to_arrow_array(values: np.ndarray) -> pa.Array:
    if np.issubdtype(values.dtype, np.datetime64):
        return pa.array(values.astype("datetime64[ns]"))
    if values.dtype == object:
        return pa.array(values.tolist())
    return pa.array(values)


def _build_group_metadata_lookup(groups: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "series_id": np.asarray([group["series_id"] for group in groups], dtype=object),
        "well_name": np.asarray([group["well_name"] for group in groups], dtype=object),
        "class_label": np.asarray([group.get("class_label", "") for group in groups], dtype=object),
        "source_type": np.asarray([group.get("source_type", "") for group in groups], dtype=object),
    }


def _build_prediction_columns(
    *,
    metadata_lookup: dict[str, np.ndarray],
    group_idx: np.ndarray,
    step_idx: np.ndarray,
    timestamp_ns: np.ndarray,
    feature_names: list[str],
    scaled_true: np.ndarray,
    scaled_pred: np.ndarray,
    scaled_baseline: np.ndarray,
    original_true: np.ndarray | None = None,
    original_pred: np.ndarray | None = None,
    original_baseline: np.ndarray | None = None,
    include_scaled: bool = False,
    include_original: bool = True,
) -> dict[str, np.ndarray]:
    columns: dict[str, np.ndarray] = {
        "series_id": metadata_lookup["series_id"][group_idx],
        "well_name": metadata_lookup["well_name"][group_idx],
        "class_label": metadata_lookup["class_label"][group_idx],
        "source_type": metadata_lookup["source_type"][group_idx],
        "step_idx": step_idx.astype(np.int32, copy=False),
        "timestamp": timestamp_ns.astype("datetime64[ns]"),
    }

    if include_original:
        if original_true is None or original_pred is None or original_baseline is None:
            raise ValueError("As matrizes em escala original precisam ser fornecidas para exportar valores originais.")
        for feature_idx, feature_name in enumerate(feature_names):
            columns[f"real__{feature_name}"] = original_true[:, feature_idx].astype(np.float32, copy=False)
            columns[f"pred__{feature_name}"] = original_pred[:, feature_idx].astype(np.float32, copy=False)
            columns[f"persist__{feature_name}"] = original_baseline[:, feature_idx].astype(np.float32, copy=False)

    if include_scaled:
        for feature_idx, feature_name in enumerate(feature_names):
            columns[f"real_scaled__{feature_name}"] = scaled_true[:, feature_idx].astype(np.float32, copy=False)
            columns[f"pred_scaled__{feature_name}"] = scaled_pred[:, feature_idx].astype(np.float32, copy=False)
            columns[f"persist_scaled__{feature_name}"] = scaled_baseline[:, feature_idx].astype(np.float32, copy=False)

    return columns


def _update_group_metric_map(
    metric_map: dict[str, RegressionMetricAccumulator],
    *,
    feature_names: list[str],
    group_labels: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    if y_true.size == 0:
        return
    unique_labels = np.unique(group_labels)
    for group_label in unique_labels:
        label_key = str(group_label)
        label_mask = group_labels == group_label
        if label_key not in metric_map:
            metric_map[label_key] = RegressionMetricAccumulator(feature_names=feature_names)
        metric_map[label_key].update(y_true[label_mask], y_pred[label_mask])


def _build_class_level_metrics_df(
    *,
    model_label: str,
    baseline_label: str,
    model_map: dict[str, RegressionMetricAccumulator],
    baseline_map: dict[str, RegressionMetricAccumulator],
) -> pd.DataFrame:
    require_tabular_stack()
    rows = []
    all_labels = sorted(set(model_map) | set(baseline_map))
    for class_label in all_labels:
        if class_label in model_map:
            rows.append(model_map[class_label].to_global_metrics(f"{model_label}_classe_{class_label}"))
        if class_label in baseline_map:
            rows.append(baseline_map[class_label].to_global_metrics(f"{baseline_label}_classe_{class_label}"))
    return pd.DataFrame(rows)


def _attach_global_improvement_columns(metrics_df: pd.DataFrame) -> pd.DataFrame:
    require_tabular_stack()
    if metrics_df.empty or len(metrics_df) < 2:
        return metrics_df

    enriched = metrics_df.copy()
    for improvement_column in ["mse_melhora_pct", "rmse_melhora_pct", "mae_melhora_pct"]:
        enriched[improvement_column] = np.nan

    model_idx = enriched.index[0]
    baseline_idx = enriched.index[1]
    metric_pairs = [
        ("MSE", "mse_melhora_pct"),
        ("RMSE", "rmse_melhora_pct"),
        ("MAE", "mae_melhora_pct"),
    ]
    for metric_name, improvement_column in metric_pairs:
        baseline_value = float(enriched.loc[baseline_idx, metric_name])
        model_value = float(enriched.loc[model_idx, metric_name])
        if baseline_value != 0:
            enriched.loc[model_idx, improvement_column] = float((baseline_value - model_value) / baseline_value * 100.0)

    return enriched


def _estimate_total_inference_work(
    loader: torch.utils.data.DataLoader,
    groups: list[dict[str, Any]],
    *,
    max_batches: int | None = None,
    max_windows: int | None = None,
    max_series: int | None = None,
) -> tuple[int | None, int | None]:
    dataset = getattr(loader, "dataset", None)
    batch_size = getattr(loader, "batch_size", None) or 1
    total_windows = None

    if dataset is not None and hasattr(dataset, "sequence_length"):
        dataset_groups = groups
        if max_series is not None:
            dataset_groups = limit_groups_for_inference(dataset_groups, max_series=max_series)
        total_windows = count_group_windows(dataset_groups, sequence_length=int(dataset.sequence_length))
    elif dataset is not None:
        total_windows = len(dataset)

    if total_windows is None:
        total_batches = max_batches
        return None, total_batches

    if max_windows is not None:
        total_windows = min(total_windows, int(max_windows))
    if max_batches is not None:
        total_windows = min(total_windows, int(max_batches) * int(batch_size))

    total_batches = int(math.ceil(total_windows / batch_size)) if batch_size else None
    return int(total_windows), total_batches


def _format_seconds(seconds: float | None) -> str:
    if seconds is None or np.isnan(seconds):
        return "?"
    total_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _get_process_memory_mb() -> float | None:
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(usage) / (1024.0 * 1024.0)
    return float(usage) / 1024.0


def _print_streaming_progress(
    *,
    batch_number: int,
    total_batches: int | None,
    processed_windows: int,
    total_windows: int | None,
    elapsed_seconds: float,
    last_batch_seconds: float,
    log_memory: bool,
) -> None:
    batch_text = f"{batch_number}/{total_batches}" if total_batches is not None else str(batch_number)
    window_text = f"{processed_windows:,}"
    percent_text = ""
    eta_text = "?"

    if total_windows is not None and total_windows > 0:
        window_text = f"{processed_windows:,}/{total_windows:,}"
        progress_pct = processed_windows / total_windows * 100.0
        percent_text = f" ({progress_pct:5.1f}%)"
        remaining_windows = max(total_windows - processed_windows, 0)
        eta_seconds = (elapsed_seconds / processed_windows * remaining_windows) if processed_windows > 0 else None
        eta_text = _format_seconds(eta_seconds)

    memory_text = ""
    if log_memory:
        memory_mb = _get_process_memory_mb()
        if memory_mb is not None:
            memory_text = f" | mem {memory_mb:,.0f} MB"

    print(
        "[predict_loader_streaming] "
        f"batch {batch_text} | janelas {window_text}{percent_text} "
        f"| tempo {_format_seconds(elapsed_seconds)} | eta {eta_text} "
        f"| ultimo_batch {last_batch_seconds:.3f}s{memory_text}"
    )


def predict_loader_streaming(
    model: HybridResidualForecaster,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    raw_target_positions: list[int],
    groups: list[dict[str, Any]],
    *,
    bundle: PreprocessingBundle | None = None,
    collect_metrics: bool = True,
    export_predictions: bool = False,
    export_dir: str | Path | None = None,
    export_chunk_rows: int = 100_000,
    export_scale: str = "original",
    preview_rows: int = 2048,
    max_batches: int | None = None,
    max_windows: int | None = None,
    max_series: int | None = None,
    progress_every: int = 25,
    log_memory: bool = False,
    model_label: str = "Modelo_v3",
    baseline_label: str = "Persistencia",
) -> StreamingPredictionResult:
    require_torch()
    require_tabular_stack()

    if export_scale not in {"scaled", "original", "both"}:
        raise ValueError("export_scale precisa ser um dentre: 'scaled', 'original' ou 'both'.")
    if export_predictions and export_dir is None:
        raise ValueError("export_dir precisa ser informado quando export_predictions=True.")
    if export_predictions and export_scale in {"original", "both"} and bundle is None:
        raise ValueError("A exportacao na escala original exige um preprocessing bundle.")

    feature_names = list(bundle.target_columns) if bundle is not None else list(BASE_TARGET_COLUMNS)
    metadata_lookup = _build_group_metadata_lookup(groups)
    total_windows_estimate, total_batches_estimate = _estimate_total_inference_work(
        loader,
        groups,
        max_batches=max_batches,
        max_windows=max_windows,
        max_series=max_series,
    )

    needs_original_arrays = bundle is not None and (
        collect_metrics
        or preview_rows > 0
        or (export_predictions and export_scale in {"original", "both"})
    )

    model_scaled_metrics = RegressionMetricAccumulator(feature_names=feature_names) if collect_metrics else None
    baseline_scaled_metrics = RegressionMetricAccumulator(feature_names=feature_names) if collect_metrics else None
    model_original_metrics = RegressionMetricAccumulator(feature_names=feature_names) if collect_metrics and bundle is not None else None
    baseline_original_metrics = RegressionMetricAccumulator(feature_names=feature_names) if collect_metrics and bundle is not None else None

    class_model_scaled: dict[str, RegressionMetricAccumulator] = {}
    class_baseline_scaled: dict[str, RegressionMetricAccumulator] = {}
    class_model_original: dict[str, RegressionMetricAccumulator] = {}
    class_baseline_original: dict[str, RegressionMetricAccumulator] = {}

    chunk_writer = PredictionChunkWriter(Path(export_dir), export_chunk_rows) if export_predictions else None
    preview_parts: list[pd.DataFrame] = []
    preview_count = 0

    processed_batches = 0
    processed_windows = 0
    start_time = time.perf_counter()

    model.eval()
    with torch.no_grad():
        for loader_batch_idx, batch in enumerate(loader, start=1):
            if max_batches is not None and loader_batch_idx > max_batches:
                break

            batch_row_count = int(batch["x"].shape[0])
            group_idx_np = batch["group_idx"].cpu().numpy()
            selected_indices = np.arange(batch_row_count, dtype=np.int64)

            if max_series is not None:
                selected_indices = selected_indices[group_idx_np[selected_indices] < int(max_series)]
                if selected_indices.size == 0:
                    if batch_row_count > 0 and int(group_idx_np.min()) >= int(max_series):
                        break
                    continue

            if max_windows is not None:
                remaining_windows = int(max_windows) - processed_windows
                if remaining_windows <= 0:
                    break
                selected_indices = selected_indices[:remaining_windows]
                if selected_indices.size == 0:
                    break

            if selected_indices.size == batch_row_count:
                x_cpu = batch["x"]
                y_cpu = batch["y"]
                well_id_cpu = batch["well_id"]
                step_idx_np = batch["step_idx"].cpu().numpy()
                timestamp_ns_np = batch["timestamp_ns"].cpu().numpy()
                selected_group_idx = group_idx_np
            else:
                selected_indices_tensor = torch.from_numpy(selected_indices)
                x_cpu = batch["x"].index_select(0, selected_indices_tensor)
                y_cpu = batch["y"].index_select(0, selected_indices_tensor)
                well_id_cpu = batch["well_id"].index_select(0, selected_indices_tensor)
                step_idx_np = batch["step_idx"].cpu().numpy()[selected_indices]
                timestamp_ns_np = batch["timestamp_ns"].cpu().numpy()[selected_indices]
                selected_group_idx = group_idx_np[selected_indices]

            batch_start = time.perf_counter()
            last_target_cpu = x_cpu[:, -1, raw_target_positions]

            x_device = x_cpu.to(device, non_blocking=True)
            well_id_device = well_id_cpu.to(device, non_blocking=True)
            prediction, _ = model(x_device, well_id_device)

            scaled_pred = prediction.detach().cpu().numpy().astype(np.float32, copy=False)
            scaled_true = y_cpu.cpu().numpy().astype(np.float32, copy=False)
            scaled_baseline = last_target_cpu.cpu().numpy().astype(np.float32, copy=False)

            original_true = None
            original_pred = None
            original_baseline = None
            if needs_original_arrays:
                original_true = inverse_transform_targets(scaled_true, bundle).astype(np.float32, copy=False)
                original_pred = inverse_transform_targets(scaled_pred, bundle).astype(np.float32, copy=False)
                original_baseline = inverse_transform_targets(scaled_baseline, bundle).astype(np.float32, copy=False)

            class_labels = metadata_lookup["class_label"][selected_group_idx]

            if collect_metrics and model_scaled_metrics is not None and baseline_scaled_metrics is not None:
                model_scaled_metrics.update(scaled_true, scaled_pred)
                baseline_scaled_metrics.update(scaled_true, scaled_baseline)
                _update_group_metric_map(
                    class_model_scaled,
                    feature_names=feature_names,
                    group_labels=class_labels,
                    y_true=scaled_true,
                    y_pred=scaled_pred,
                )
                _update_group_metric_map(
                    class_baseline_scaled,
                    feature_names=feature_names,
                    group_labels=class_labels,
                    y_true=scaled_true,
                    y_pred=scaled_baseline,
                )

                if model_original_metrics is not None and baseline_original_metrics is not None:
                    model_original_metrics.update(original_true, original_pred)
                    baseline_original_metrics.update(original_true, original_baseline)
                    _update_group_metric_map(
                        class_model_original,
                        feature_names=feature_names,
                        group_labels=class_labels,
                        y_true=original_true,
                        y_pred=original_pred,
                    )
                    _update_group_metric_map(
                        class_baseline_original,
                        feature_names=feature_names,
                        group_labels=class_labels,
                        y_true=original_true,
                        y_pred=original_baseline,
                    )

            if chunk_writer is not None:
                chunk_writer.append(
                    _build_prediction_columns(
                        metadata_lookup=metadata_lookup,
                        group_idx=selected_group_idx,
                        step_idx=step_idx_np,
                        timestamp_ns=timestamp_ns_np,
                        feature_names=feature_names,
                        scaled_true=scaled_true,
                        scaled_pred=scaled_pred,
                        scaled_baseline=scaled_baseline,
                        original_true=original_true,
                        original_pred=original_pred,
                        original_baseline=original_baseline,
                        include_scaled=export_scale in {"scaled", "both"},
                        include_original=export_scale in {"original", "both"},
                    )
                )

            if preview_rows > 0 and preview_count < preview_rows:
                preview_take = min(int(preview_rows) - preview_count, len(selected_group_idx))
                preview_columns = _build_prediction_columns(
                    metadata_lookup=metadata_lookup,
                    group_idx=selected_group_idx[:preview_take],
                    step_idx=step_idx_np[:preview_take],
                    timestamp_ns=timestamp_ns_np[:preview_take],
                    feature_names=feature_names,
                    scaled_true=scaled_true[:preview_take],
                    scaled_pred=scaled_pred[:preview_take],
                    scaled_baseline=scaled_baseline[:preview_take],
                    original_true=original_true[:preview_take] if original_true is not None else None,
                    original_pred=original_pred[:preview_take] if original_pred is not None else None,
                    original_baseline=original_baseline[:preview_take] if original_baseline is not None else None,
                    include_scaled=True,
                    include_original=original_true is not None,
                )
                preview_parts.append(pd.DataFrame(preview_columns))
                preview_count += preview_take

            processed_batches += 1
            processed_windows += len(selected_group_idx)
            batch_elapsed = time.perf_counter() - batch_start
            elapsed_seconds = time.perf_counter() - start_time

            should_log = progress_every and (
                processed_batches % progress_every == 0
                or (total_windows_estimate is not None and processed_windows >= total_windows_estimate)
                or (max_windows is not None and processed_windows >= max_windows)
            )
            if should_log:
                _print_streaming_progress(
                    batch_number=processed_batches,
                    total_batches=total_batches_estimate,
                    processed_windows=processed_windows,
                    total_windows=total_windows_estimate,
                    elapsed_seconds=elapsed_seconds,
                    last_batch_seconds=batch_elapsed,
                    log_memory=log_memory,
                )

            del x_device, well_id_device, prediction

            if max_windows is not None and processed_windows >= int(max_windows):
                break

    if processed_windows == 0:
        raise ValueError("Nenhuma janela foi processada. Revise max_batches, max_windows e max_series.")

    if chunk_writer is not None:
        chunk_writer.flush()

    elapsed_seconds = time.perf_counter() - start_time
    preview_df = pd.concat(preview_parts, ignore_index=True) if preview_parts else pd.DataFrame()

    global_metrics_scaled_df = None
    global_metrics_original_df = None
    per_feature_scaled_df = None
    per_feature_original_df = None
    class_metrics_scaled_df = None
    class_metrics_original_df = None

    if collect_metrics and model_scaled_metrics is not None and baseline_scaled_metrics is not None:
        global_metrics_scaled_df = _attach_global_improvement_columns(
            pd.DataFrame(
                [
                    model_scaled_metrics.to_global_metrics(model_label),
                    baseline_scaled_metrics.to_global_metrics(baseline_label),
                ]
            )
        )
        per_feature_scaled_df = model_scaled_metrics.to_per_feature_metrics(baseline=baseline_scaled_metrics)
        class_metrics_scaled_df = _build_class_level_metrics_df(
            model_label=model_label,
            baseline_label=baseline_label,
            model_map=class_model_scaled,
            baseline_map=class_baseline_scaled,
        )

    if collect_metrics and model_original_metrics is not None and baseline_original_metrics is not None:
        global_metrics_original_df = _attach_global_improvement_columns(
            pd.DataFrame(
                [
                    model_original_metrics.to_global_metrics(model_label),
                    baseline_original_metrics.to_global_metrics(baseline_label),
                ]
            )
        )
        per_feature_original_df = model_original_metrics.to_per_feature_metrics(baseline=baseline_original_metrics)
        class_metrics_original_df = _build_class_level_metrics_df(
            model_label=model_label,
            baseline_label=baseline_label,
            model_map=class_model_original,
            baseline_map=class_baseline_original,
        )

    return StreamingPredictionResult(
        processed_batches=processed_batches,
        processed_windows=processed_windows,
        total_batches_estimate=total_batches_estimate,
        total_windows_estimate=total_windows_estimate,
        elapsed_seconds=elapsed_seconds,
        export_files=(chunk_writer.files if chunk_writer is not None else []),
        preview_df=preview_df,
        global_metrics_scaled_df=global_metrics_scaled_df,
        global_metrics_original_df=global_metrics_original_df,
        per_feature_scaled_df=per_feature_scaled_df,
        per_feature_original_df=per_feature_original_df,
        class_metrics_scaled_df=class_metrics_scaled_df,
        class_metrics_original_df=class_metrics_original_df,
    )


# Mantido por compatibilidade com notebooks antigos.
# Para conjuntos grandes, prefira `predict_loader_streaming`, que evita montar
# um DataFrame gigante e reduz o custo de CPU/RAM no loop de inferencia.
def predict_loader(
    model: HybridResidualForecaster,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    raw_target_positions: list[int],
    groups: list[dict[str, Any]],
) -> pd.DataFrame:
    require_torch()
    require_tabular_stack()
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            well_id = batch["well_id"].to(device)
            group_idx = batch["group_idx"].cpu().numpy()
            step_idx = batch["step_idx"].cpu().numpy()
            timestamp_ns = batch["timestamp_ns"].cpu().numpy()
            prediction, _ = model(x, well_id)
            last_target = x[:, -1, raw_target_positions]

            prediction_np = prediction.cpu().numpy()
            target_np = y.cpu().numpy()
            persistence_np = last_target.cpu().numpy()

            for row_idx in range(len(group_idx)):
                base_row = {
                    "series_id": groups[int(group_idx[row_idx])]["series_id"],
                    "well_name": groups[int(group_idx[row_idx])]["well_name"],
                    "class_label": groups[int(group_idx[row_idx])].get("class_label", ""),
                    "source_type": groups[int(group_idx[row_idx])].get("source_type", ""),
                    "step_idx": int(step_idx[row_idx]),
                    "timestamp": pd.to_datetime(int(timestamp_ns[row_idx])),
                }
                for feature_idx, feature_name in enumerate(BASE_TARGET_COLUMNS):
                    base_row[f"real__{feature_name}"] = float(target_np[row_idx, feature_idx])
                    base_row[f"pred__{feature_name}"] = float(prediction_np[row_idx, feature_idx])
                    base_row[f"persist__{feature_name}"] = float(persistence_np[row_idx, feature_idx])
                rows.append(base_row)

    return pd.DataFrame(rows)


def inverse_transform_targets(values: np.ndarray, bundle: PreprocessingBundle) -> np.ndarray:
    mean = np.asarray(bundle.target_scaler_mean, dtype=np.float32)
    scale = np.asarray(bundle.target_scaler_scale, dtype=np.float32)
    return values * scale + mean


def compute_global_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str,
) -> dict[str, float | str]:
    require_tabular_stack()
    flat_true = y_true.reshape(-1)
    flat_pred = y_pred.reshape(-1)
    mse = mean_squared_error(flat_true, flat_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(flat_true, flat_pred)
    medae = median_absolute_error(flat_true, flat_pred)
    r2 = r2_score(y_true, y_pred, multioutput="uniform_average")
    evs = explained_variance_score(y_true, y_pred, multioutput="uniform_average")
    return {
        "modelo": label,
        "MSE": float(mse),
        "RMSE": rmse,
        "MAE": float(mae),
        "MedAE": float(medae),
        "R2_medio": float(r2),
        "ExplainedVariance_media": float(evs),
    }


def compute_per_feature_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_baseline: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    require_tabular_stack()
    rows = []
    for feature_idx, feature_name in enumerate(feature_names):
        feature_true = y_true[:, feature_idx]
        feature_pred = y_pred[:, feature_idx]
        feature_baseline = y_baseline[:, feature_idx]
        model_mse = mean_squared_error(feature_true, feature_pred)
        baseline_mse = mean_squared_error(feature_true, feature_baseline)
        model_rmse = float(np.sqrt(model_mse))
        baseline_rmse = float(np.sqrt(baseline_mse))
        model_mae = mean_absolute_error(feature_true, feature_pred)
        baseline_mae = mean_absolute_error(feature_true, feature_baseline)
        rows.append(
            {
                "feature": feature_name,
                "model_mae": float(model_mae),
                "baseline_mae": float(baseline_mae),
                "mae_melhora_pct": float((baseline_mae - model_mae) / baseline_mae * 100) if baseline_mae != 0 else np.nan,
                "model_rmse": model_rmse,
                "baseline_rmse": baseline_rmse,
                "rmse_melhora_pct": float((baseline_rmse - model_rmse) / baseline_rmse * 100) if baseline_rmse != 0 else np.nan,
                "model_r2": float(r2_score(feature_true, feature_pred)),
                "baseline_r2": float(r2_score(feature_true, feature_baseline)),
                "model_bias": float(np.mean(feature_pred - feature_true)),
                "model_residual_std": float(np.std(feature_true - feature_pred)),
            }
        )
    return pd.DataFrame(rows).sort_values("model_rmse", ascending=False).reset_index(drop=True)
