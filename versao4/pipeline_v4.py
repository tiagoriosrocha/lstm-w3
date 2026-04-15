from __future__ import annotations

from collections import defaultdict
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

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from versao3.pipeline_v3 import (  # noqa: E402
    AUX_ANALOG_COLUMNS,
    BASE_TARGET_COLUMNS,
    CONTINUOUS_COLUMNS,
    STATE_COLUMNS,
    _parse_series_metadata,
    clean_base_frame,
    discover_all_dataset_files,
    discover_balanced_normal_files,
    read_parquet_with_timestamp,
    require_parquet_engine,
    require_tabular_stack,
    require_torch,
    set_seed,
    split_manifest_by_series,
    split_manifest_by_well,
    write_manifest_csv,
)


@dataclass
class PreprocessingBundle:
    target_columns: list[str]
    auxiliary_columns: list[str]
    raw_input_columns: list[str]
    input_columns: list[str]
    raw_target_input_columns: list[str]
    target_scaler_center: list[float]
    target_scaler_scale: list[float]
    aux_scaler_center: list[float]
    aux_scaler_scale: list[float]
    diff_scaler_center: list[float]
    diff_scaler_scale: list[float]
    dev_scaler_center: list[float]
    dev_scaler_scale: list[float]
    std_scaler_center: list[float]
    std_scaler_scale: list[float]
    clip_bounds: dict[str, dict[str, float]]
    log_transform_columns: list[str]
    scaler_strategy: str
    well_to_id: dict[str, int]
    split_counts: dict[str, int]
    selected_files: dict[str, list[str]]
    max_files_per_well: int | None
    rolling_window: int
    sequence_length_recommendation: int
    forecast_horizon_recommendation: int


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
    well_metrics_scaled_df: pd.DataFrame | None = None
    well_metrics_original_df: pd.DataFrame | None = None
    series_metrics_scaled_df: pd.DataFrame | None = None
    series_metrics_original_df: pd.DataFrame | None = None
    horizon_metrics_scaled_df: pd.DataFrame | None = None
    horizon_metrics_original_df: pd.DataFrame | None = None


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


def sample_frame_rows(frame: pd.DataFrame, max_rows: int | None = None) -> pd.DataFrame:
    require_tabular_stack()
    if max_rows is None or len(frame) <= int(max_rows):
        return frame.reset_index(drop=True)
    sample_idx = np.linspace(0, len(frame) - 1, num=int(max_rows), dtype=np.int64)
    sample_idx = np.unique(sample_idx)
    return frame.iloc[sample_idx].reset_index(drop=True)


def collect_training_reference_frame(
    train_manifest: pd.DataFrame,
    max_rows_per_series: int | None = 512,
) -> pd.DataFrame:
    require_tabular_stack()
    parts = []
    keep_candidates = BASE_TARGET_COLUMNS + [col for col in STATE_COLUMNS + AUX_ANALOG_COLUMNS]
    for file_path in train_manifest["file_path"]:
        frame = clean_base_frame(file_path)
        sampled = sample_frame_rows(frame, max_rows=max_rows_per_series)
        available_columns = [col for col in keep_candidates if col in sampled.columns]
        parts.append(sampled[available_columns])
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


def profile_continuous_columns(
    reference_df: pd.DataFrame,
    columns: list[str] | None = None,
    high_dynamic_ratio: float = 100.0,
    high_abs_p99: float = 1e4,
) -> pd.DataFrame:
    require_tabular_stack()
    columns = columns or [col for col in CONTINUOUS_COLUMNS if col in reference_df.columns]
    rows = []
    for column in columns:
        series = pd.to_numeric(reference_df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        non_null = series.dropna()
        if non_null.empty:
            rows.append(
                {
                    "column": column,
                    "non_null_rows": 0,
                    "min": np.nan,
                    "p01": np.nan,
                    "p50": np.nan,
                    "p99": np.nan,
                    "max": np.nan,
                    "iqr": np.nan,
                    "positive_pct": np.nan,
                    "zero_pct": np.nan,
                    "dynamic_ratio_p99_p50": np.nan,
                    "recommended_log_transform": False,
                }
            )
            continue

        p01 = float(non_null.quantile(0.01))
        p50 = float(non_null.quantile(0.50))
        p99 = float(non_null.quantile(0.99))
        iqr = float(non_null.quantile(0.75) - non_null.quantile(0.25))
        denom = max(abs(p50), 1e-6)
        dynamic_ratio = abs(p99) / denom
        positive_pct = float((non_null > 0).mean() * 100.0)
        zero_pct = float((non_null == 0).mean() * 100.0)
        recommended_log = bool(positive_pct >= 95.0 and (dynamic_ratio >= high_dynamic_ratio or abs(p99) >= high_abs_p99))
        rows.append(
            {
                "column": column,
                "non_null_rows": int(len(non_null)),
                "min": float(non_null.min()),
                "p01": p01,
                "p50": p50,
                "p99": p99,
                "max": float(non_null.max()),
                "iqr": iqr,
                "positive_pct": positive_pct,
                "zero_pct": zero_pct,
                "dynamic_ratio_p99_p50": float(dynamic_ratio),
                "recommended_log_transform": recommended_log,
            }
        )
    return pd.DataFrame(rows).sort_values("column").reset_index(drop=True)


def recommend_log_transform_columns(
    profile_df: pd.DataFrame,
    candidate_columns: list[str] | None = None,
) -> list[str]:
    require_tabular_stack()
    filtered = profile_df.copy()
    if candidate_columns is not None:
        filtered = filtered[filtered["column"].isin(candidate_columns)].copy()
    return filtered.loc[filtered["recommended_log_transform"], "column"].tolist()


def _safe_scale(values: np.ndarray) -> list[float]:
    scale = np.asarray(values, dtype=np.float64)
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    return scale.tolist()


def _safe_numeric_matrix(values: np.ndarray, clip_abs: float | None = 1e12) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if clip_abs is not None:
        matrix = np.clip(matrix, -clip_abs, clip_abs)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix


def signed_log1p(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return np.sign(arr) * np.log1p(np.abs(arr))


def inverse_signed_log1p(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return np.sign(arr) * np.expm1(np.abs(arr))


def apply_clip_bounds(frame: pd.DataFrame, clip_bounds: dict[str, dict[str, float]]) -> pd.DataFrame:
    require_tabular_stack()
    clipped = frame.copy()
    for column, bounds in clip_bounds.items():
        if column in clipped.columns:
            clipped[column] = clipped[column].clip(lower=bounds["low"], upper=bounds["high"])
    return clipped


def apply_log_transform(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    require_tabular_stack()
    if not columns:
        return frame.copy()
    transformed = frame.copy()
    for column in columns:
        if column in transformed.columns:
            transformed[column] = signed_log1p(transformed[column].to_numpy(dtype=np.float64))
    return transformed


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
        _safe_numeric_matrix(diff_features.to_numpy(dtype=np.float64)),
        _safe_numeric_matrix(deviation_features.to_numpy(dtype=np.float64)),
        _safe_numeric_matrix(rolling_std.to_numpy(dtype=np.float64)),
    )


def _fit_scaler_stats(values: np.ndarray, strategy: str = "robust") -> tuple[list[float], list[float]]:
    matrix = _safe_numeric_matrix(values, clip_abs=None)
    if matrix.size == 0:
        return [], []
    if strategy == "robust":
        center = np.median(matrix, axis=0)
        q75 = np.quantile(matrix, 0.75, axis=0)
        q25 = np.quantile(matrix, 0.25, axis=0)
        scale = q75 - q25
    elif strategy == "standard":
        center = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
    else:
        raise ValueError("scaler_strategy precisa ser 'robust' ou 'standard'.")
    return center.tolist(), _safe_scale(scale)


def _scale_matrix(values: np.ndarray, center: list[float], scale: list[float]) -> np.ndarray:
    matrix = _safe_numeric_matrix(values)
    center_arr = np.asarray(center, dtype=np.float64)
    scale_arr = np.asarray(scale, dtype=np.float64)
    return ((matrix - center_arr) / scale_arr).astype(np.float32, copy=False)


def _inverse_scale_matrix(values: np.ndarray, center: list[float], scale: list[float]) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    center_arr = np.asarray(center, dtype=np.float64)
    scale_arr = np.asarray(scale, dtype=np.float64)
    return matrix * scale_arr + center_arr


def fit_preprocessing_bundle(
    train_manifest: pd.DataFrame,
    auxiliary_columns: list[str],
    max_files_per_well: int | None,
    rolling_window: int = 5,
    sequence_length_recommendation: int = 60,
    forecast_horizon_recommendation: int = 5,
    quantile_low: float = 0.001,
    quantile_high: float = 0.999,
    scaler_strategy: str = "robust",
    reference_rows_per_series: int = 512,
    log_transform_columns: list[str] | None = None,
) -> PreprocessingBundle:
    require_tabular_stack()
    log_transform_columns = sorted(set(log_transform_columns or []))
    clean_frames = [
        clean_base_frame(file_path, candidate_auxiliary_columns=auxiliary_columns)
        for file_path in train_manifest["file_path"]
    ]

    continuous_reference_parts = []
    for frame in clean_frames:
        available = [col for col in CONTINUOUS_COLUMNS if col in frame.columns]
        continuous_reference_parts.append(sample_frame_rows(frame[available], max_rows=reference_rows_per_series))
    continuous_reference_df = pd.concat(continuous_reference_parts, ignore_index=True)

    clip_bounds: dict[str, dict[str, float]] = {}
    for column in [col for col in CONTINUOUS_COLUMNS if col in continuous_reference_df.columns]:
        clip_bounds[column] = {
            "low": float(continuous_reference_df[column].quantile(quantile_low)),
            "high": float(continuous_reference_df[column].quantile(quantile_high)),
        }

    target_parts: list[np.ndarray] = []
    aux_parts: list[np.ndarray] = []
    diff_parts: list[np.ndarray] = []
    dev_parts: list[np.ndarray] = []
    std_parts: list[np.ndarray] = []

    for frame in clean_frames:
        clipped = apply_clip_bounds(frame, clip_bounds)
        transformed = apply_log_transform(clipped, log_transform_columns)
        sampled = sample_frame_rows(transformed, max_rows=reference_rows_per_series)

        target_parts.append(sampled[BASE_TARGET_COLUMNS].to_numpy(dtype=np.float64))
        if auxiliary_columns:
            aux_parts.append(sampled[auxiliary_columns].to_numpy(dtype=np.float64))

        diff_features, dev_features, std_features = build_derived_feature_arrays(
            sampled,
            target_columns=BASE_TARGET_COLUMNS,
            rolling_window=rolling_window,
        )
        diff_parts.append(diff_features)
        dev_parts.append(dev_features)
        std_parts.append(std_features)

    target_center, target_scale = _fit_scaler_stats(np.concatenate(target_parts, axis=0), strategy=scaler_strategy)
    if aux_parts:
        aux_center, aux_scale = _fit_scaler_stats(np.concatenate(aux_parts, axis=0), strategy=scaler_strategy)
    else:
        aux_center, aux_scale = [], []
    diff_center, diff_scale = _fit_scaler_stats(np.concatenate(diff_parts, axis=0), strategy=scaler_strategy)
    dev_center, dev_scale = _fit_scaler_stats(np.concatenate(dev_parts, axis=0), strategy=scaler_strategy)
    std_center, std_scale = _fit_scaler_stats(np.concatenate(std_parts, axis=0), strategy=scaler_strategy)

    raw_target_input_columns = [f"raw__{column}" for column in BASE_TARGET_COLUMNS]
    raw_aux_input_columns = [f"raw__{column}" for column in auxiliary_columns]
    input_columns = (
        raw_target_input_columns
        + raw_aux_input_columns
        + [f"diff1__{column}" for column in BASE_TARGET_COLUMNS]
        + [f"dev_roll{rolling_window}__{column}" for column in BASE_TARGET_COLUMNS]
        + [f"std_roll{rolling_window}__{column}" for column in BASE_TARGET_COLUMNS]
    )

    unique_wells = sorted(set(train_manifest["well_name"].tolist()))
    well_to_id = {well_name: idx for idx, well_name in enumerate(unique_wells)}

    return PreprocessingBundle(
        target_columns=BASE_TARGET_COLUMNS,
        auxiliary_columns=auxiliary_columns,
        raw_input_columns=BASE_TARGET_COLUMNS + auxiliary_columns,
        input_columns=input_columns,
        raw_target_input_columns=raw_target_input_columns,
        target_scaler_center=target_center,
        target_scaler_scale=target_scale,
        aux_scaler_center=aux_center,
        aux_scaler_scale=aux_scale,
        diff_scaler_center=diff_center,
        diff_scaler_scale=diff_scale,
        dev_scaler_center=dev_center,
        dev_scaler_scale=dev_scale,
        std_scaler_center=std_center,
        std_scaler_scale=std_scale,
        clip_bounds=clip_bounds,
        log_transform_columns=log_transform_columns,
        scaler_strategy=scaler_strategy,
        well_to_id=well_to_id,
        split_counts=train_manifest["well_name"].value_counts().sort_index().to_dict(),
        selected_files={
            "train": train_manifest["file_path"].tolist(),
            "validation": [],
            "test": [],
        },
        max_files_per_well=max_files_per_well,
        rolling_window=rolling_window,
        sequence_length_recommendation=sequence_length_recommendation,
        forecast_horizon_recommendation=forecast_horizon_recommendation,
    )


def fit_preprocessing_bundle_from_train_frame(
    train_frame: pd.DataFrame,
    auxiliary_columns: list[str],
    *,
    well_name: str = "single_well",
    source_file: str | None = None,
    rolling_window: int = 5,
    sequence_length_recommendation: int = 60,
    forecast_horizon_recommendation: int = 5,
    scaler_strategy: str = "robust",
    log_transform_columns: list[str] | None = None,
) -> PreprocessingBundle:
    require_tabular_stack()
    temp_manifest = pd.DataFrame(
        {
            "file_path": [source_file or "memory_frame"],
            "well_name": [well_name],
        }
    )
    temp_path = PROJECT_ROOT / "artifacts" / "_temp_v4_single_frame.parquet"
    train_frame.to_parquet(temp_path, index=False)
    temp_manifest.loc[0, "file_path"] = str(temp_path)
    bundle = fit_preprocessing_bundle(
        train_manifest=temp_manifest,
        auxiliary_columns=auxiliary_columns,
        max_files_per_well=1,
        rolling_window=rolling_window,
        sequence_length_recommendation=sequence_length_recommendation,
        forecast_horizon_recommendation=forecast_horizon_recommendation,
        scaler_strategy=scaler_strategy,
        log_transform_columns=log_transform_columns,
    )
    try:
        temp_path.unlink(missing_ok=True)
    except Exception:
        pass
    return bundle


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
    transformed = apply_log_transform(clipped, bundle.log_transform_columns)

    target_values = transformed[bundle.target_columns].to_numpy(dtype=np.float64)
    target_scaled = _scale_matrix(target_values, bundle.target_scaler_center, bundle.target_scaler_scale)
    target_scaled_df = pd.DataFrame(target_scaled, columns=[f"target__{column}" for column in bundle.target_columns])
    raw_target_scaled_df = pd.DataFrame(target_scaled, columns=[f"raw__{column}" for column in bundle.target_columns])

    input_parts = [raw_target_scaled_df]
    if bundle.auxiliary_columns:
        aux_values = transformed[bundle.auxiliary_columns].to_numpy(dtype=np.float64)
        aux_scaled = _scale_matrix(aux_values, bundle.aux_scaler_center, bundle.aux_scaler_scale)
        input_parts.append(pd.DataFrame(aux_scaled, columns=[f"raw__{column}" for column in bundle.auxiliary_columns]))

    diff_features, dev_features, std_features = build_derived_feature_arrays(
        transformed,
        target_columns=bundle.target_columns,
        rolling_window=bundle.rolling_window,
    )
    diff_scaled = _scale_matrix(diff_features, bundle.diff_scaler_center, bundle.diff_scaler_scale)
    dev_scaled = _scale_matrix(dev_features, bundle.dev_scaler_center, bundle.dev_scaler_scale)
    std_scaled = _scale_matrix(std_features, bundle.std_scaler_center, bundle.std_scaler_scale)

    input_parts.append(pd.DataFrame(diff_scaled, columns=[f"diff1__{column}" for column in bundle.target_columns]))
    input_parts.append(pd.DataFrame(dev_scaled, columns=[f"dev_roll{bundle.rolling_window}__{column}" for column in bundle.target_columns]))
    input_parts.append(pd.DataFrame(std_scaled, columns=[f"std_roll{bundle.rolling_window}__{column}" for column in bundle.target_columns]))

    metadata_df = pd.DataFrame(
        {
            "series_id": [series_id] * len(transformed),
            "well_name": [well_name] * len(transformed),
            "class_label": [class_label] * len(transformed),
            "source_type": [source_type] * len(transformed),
            "well_id": [bundle.well_to_id.get(well_name, -1)] * len(transformed),
            "file_path": [file_path] * len(transformed),
            "timestamp": transformed["timestamp"].astype("datetime64[ns]"),
        }
    )
    return pd.concat([metadata_df, target_scaled_df] + input_parts, axis=1)


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


def inverse_transform_targets(values: np.ndarray, bundle: PreprocessingBundle) -> np.ndarray:
    restored = _inverse_scale_matrix(values, bundle.target_scaler_center, bundle.target_scaler_scale)
    output = np.asarray(restored, dtype=np.float64).copy()
    target_log_columns = [col for col in bundle.target_columns if col in bundle.log_transform_columns]
    for column in target_log_columns:
        feature_idx = bundle.target_columns.index(column)
        output[..., feature_idx] = inverse_signed_log1p(output[..., feature_idx])
    return output.astype(np.float32, copy=False)


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
    forecast_horizon: int = 1,
) -> int:
    return int(
        sum(
            max(len(group["targets"]) - sequence_length - forecast_horizon + 1, 0)
            for group in groups
        )
    )


if torch is not None:
    class GroupedWindowDataset(Dataset):
        def __init__(
            self,
            groups: list[dict[str, Any]],
            sequence_length: int,
            forecast_horizon: int = 1,
            sampled_windows: int | None = None,
            seed: int = 42,
            balance_by_class: bool = False,
        ) -> None:
            self.groups = groups
            self.sequence_length = int(sequence_length)
            self.forecast_horizon = int(forecast_horizon)
            self.sampled_windows = sampled_windows
            self.seed = seed
            self.balance_by_class = balance_by_class
            self._full_index = self._build_full_index()
            self._class_index_map = self._build_class_index_map()
            self._active_index = self._full_index.copy()
            self.resample()

        def _build_full_index(self) -> np.ndarray:
            index_rows = []
            for group_idx, group in enumerate(self.groups):
                n_rows = len(group["targets"])
                max_start = n_rows - self.sequence_length - self.forecast_horizon
                if max_start < 0:
                    continue
                starts = np.arange(max_start + 1, dtype=np.int32)
                group_column = np.full_like(starts, fill_value=group_idx)
                index_rows.append(np.stack([group_column, starts], axis=1))
            if not index_rows:
                return np.zeros((0, 2), dtype=np.int32)
            return np.concatenate(index_rows, axis=0).astype(np.int32)

        def _build_class_index_map(self) -> dict[str, np.ndarray]:
            if len(self._full_index) == 0:
                return {}
            label_to_rows: dict[str, list[int]] = defaultdict(list)
            for row_idx, (group_idx, _) in enumerate(self._full_index):
                label = str(self.groups[int(group_idx)].get("class_label", ""))
                label_to_rows[label].append(row_idx)
            return {label: np.asarray(row_ids, dtype=np.int64) for label, row_ids in label_to_rows.items()}

        def resample(self, epoch: int = 0) -> None:
            if self.sampled_windows is None or self.sampled_windows >= len(self._full_index):
                self._active_index = self._full_index.copy()
                return
            rng = np.random.default_rng(self.seed + epoch)
            if self.balance_by_class and self._class_index_map:
                label_keys = sorted(self._class_index_map)
                per_class = max(self.sampled_windows // max(len(label_keys), 1), 1)
                chosen_parts = []
                for label in label_keys:
                    available = self._class_index_map[label]
                    take = min(len(available), per_class)
                    chosen_parts.append(rng.choice(available, size=take, replace=False))
                chosen = np.concatenate(chosen_parts, axis=0) if chosen_parts else np.array([], dtype=np.int64)
                remaining = max(self.sampled_windows - len(chosen), 0)
                if remaining > 0:
                    available_pool = np.setdiff1d(np.arange(len(self._full_index), dtype=np.int64), np.unique(chosen), assume_unique=False)
                    if len(available_pool) > 0:
                        extra_take = min(len(available_pool), remaining)
                        extra = rng.choice(available_pool, size=extra_take, replace=False)
                        chosen = np.concatenate([chosen, extra], axis=0)
                if len(chosen) > self.sampled_windows:
                    chosen = rng.choice(chosen, size=self.sampled_windows, replace=False)
                self._active_index = self._full_index[np.sort(np.unique(chosen))]
            else:
                sample_ids = rng.choice(len(self._full_index), size=self.sampled_windows, replace=False)
                self._active_index = self._full_index[np.sort(sample_ids)]

        def __len__(self) -> int:
            return len(self._active_index)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            group_idx, start_idx = self._active_index[idx]
            group = self.groups[int(group_idx)]
            end_idx = int(start_idx) + self.sequence_length
            target_end = end_idx + self.forecast_horizon
            x = group["inputs"][int(start_idx):end_idx]
            y = group["targets"][end_idx:target_end]
            timestamps = np.asarray(group["timestamps"][end_idx:target_end]).astype("datetime64[ns]").astype(np.int64)
            step_idx = np.arange(end_idx, target_end, dtype=np.int64)
            return {
                "x": torch.tensor(x, dtype=torch.float32),
                "y": torch.tensor(y, dtype=torch.float32),
                "well_id": torch.tensor(group["well_id"], dtype=torch.long),
                "group_idx": torch.tensor(int(group_idx), dtype=torch.long),
                "step_idx": torch.tensor(step_idx, dtype=torch.long),
                "timestamp_ns": torch.tensor(timestamps, dtype=torch.long),
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


    class PureLSTMForecaster(nn.Module):
        def __init__(
            self,
            input_size: int,
            target_size: int,
            forecast_horizon: int,
            well_count: int,
            raw_target_positions: list[int],
            hidden_size: int = 128,
            num_layers: int = 2,
            well_embedding_dim: int = 16,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.target_size = int(target_size)
            self.forecast_horizon = int(forecast_horizon)
            self.raw_target_positions = raw_target_positions
            self.input_norm = nn.LayerNorm(input_size)
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.well_embedding = nn.Embedding(num_embeddings=max(well_count, 1), embedding_dim=well_embedding_dim)
            self.head = nn.Sequential(
                nn.Linear(hidden_size + well_embedding_dim + target_size, 192),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(192, forecast_horizon * target_size),
            )

        def forward(self, x: torch.Tensor, well_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            last_target = x[:, -1, self.raw_target_positions]
            encoded, _ = self.lstm(self.input_norm(x))
            temporal_context = encoded[:, -1, :]
            well_context = self.well_embedding(well_id)
            fused = torch.cat([temporal_context, well_context, last_target], dim=1)
            delta_prediction = self.head(fused).view(-1, self.forecast_horizon, self.target_size)
            baseline = last_target.unsqueeze(1).expand(-1, self.forecast_horizon, -1)
            prediction = baseline + delta_prediction
            return prediction, delta_prediction


    class HybridResidualForecaster(nn.Module):
        def __init__(
            self,
            input_size: int,
            target_size: int,
            forecast_horizon: int,
            well_count: int,
            raw_target_positions: list[int],
            model_dim: int = 128,
            gru_hidden_size: int = 128,
            gru_layers: int = 2,
            well_embedding_dim: int = 16,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.target_size = int(target_size)
            self.forecast_horizon = int(forecast_horizon)
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
                nn.Linear(96, forecast_horizon * target_size),
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
            delta_prediction = self.head(fused).view(-1, self.forecast_horizon, self.target_size)
            baseline = last_target.unsqueeze(1).expand(-1, self.forecast_horizon, -1)
            prediction = baseline + delta_prediction
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


    class PureLSTMForecaster:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


    class HybridResidualForecaster:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


def build_model(
    model_name: str,
    *,
    input_size: int,
    target_size: int,
    forecast_horizon: int,
    well_count: int,
    raw_target_positions: list[int],
    model_dim: int = 128,
    hidden_size: int = 128,
    recurrent_layers: int = 2,
    well_embedding_dim: int = 16,
    dropout: float = 0.2,
) -> nn.Module:
    require_torch()
    if model_name == "pure_lstm_forecaster_v4":
        return PureLSTMForecaster(
            input_size=input_size,
            target_size=target_size,
            forecast_horizon=forecast_horizon,
            well_count=well_count,
            raw_target_positions=raw_target_positions,
            hidden_size=hidden_size,
            num_layers=recurrent_layers,
            well_embedding_dim=well_embedding_dim,
            dropout=dropout,
        )
    if model_name == "hybrid_residual_forecaster_v4":
        return HybridResidualForecaster(
            input_size=input_size,
            target_size=target_size,
            forecast_horizon=forecast_horizon,
            well_count=well_count,
            raw_target_positions=raw_target_positions,
            model_dim=model_dim,
            gru_hidden_size=hidden_size,
            gru_layers=recurrent_layers,
            well_embedding_dim=well_embedding_dim,
            dropout=dropout,
        )
    raise ValueError(f"Modelo desconhecido: {model_name}")


def load_model_from_config(
    model_config: dict[str, Any],
    well_count: int,
    device: torch.device,
) -> nn.Module:
    require_torch()
    model = build_model(
        model_name=model_config["model_name"],
        input_size=model_config["input_size"],
        target_size=model_config["target_size"],
        forecast_horizon=model_config["forecast_horizon"],
        well_count=well_count,
        raw_target_positions=model_config["raw_target_positions"],
        model_dim=model_config.get("model_dim", 128),
        hidden_size=model_config.get("hidden_size", model_config.get("gru_hidden_size", 128)),
        recurrent_layers=model_config.get("recurrent_layers", model_config.get("gru_layers", 2)),
        well_embedding_dim=model_config.get("well_embedding_dim", 16),
        dropout=model_config.get("dropout", 0.2),
    )
    model.load_state_dict(torch.load(model_config["model_path"], map_location=device))
    model.to(device)
    model.eval()
    return model


def multi_horizon_forecasting_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    last_target: torch.Tensor,
    target_weights: list[float] | np.ndarray | torch.Tensor | None = None,
    horizon_weights: list[float] | np.ndarray | torch.Tensor | None = None,
    delta_weight: float = 0.6,
    absolute_weight: float = 0.4,
) -> tuple[torch.Tensor, dict[str, float]]:
    require_torch()
    baseline = last_target.unsqueeze(1).expand_as(target)
    true_delta = target - baseline
    pred_delta = prediction - baseline

    weights = torch.ones_like(target)
    if target_weights is not None:
        target_weight_tensor = torch.as_tensor(target_weights, dtype=prediction.dtype, device=prediction.device).view(1, 1, -1)
        weights = weights * target_weight_tensor
    if horizon_weights is not None:
        horizon_weight_tensor = torch.as_tensor(horizon_weights, dtype=prediction.dtype, device=prediction.device).view(1, -1, 1)
        weights = weights * horizon_weight_tensor

    delta_element = torch.nn.functional.smooth_l1_loss(pred_delta, true_delta, reduction="none")
    absolute_element = torch.nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    delta_loss = (delta_element * weights).mean()
    absolute_loss = (absolute_element * weights).mean()
    total_loss = delta_weight * delta_loss + absolute_weight * absolute_loss
    return total_loss, {
        "delta_loss": float(delta_loss.detach().cpu().item()),
        "absolute_loss": float(absolute_loss.detach().cpu().item()),
        "total_loss": float(total_loss.detach().cpu().item()),
    }


def _flatten_metric_matrix(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr.reshape(-1, arr.shape[-1])
    raise ValueError("As matrizes de métricas precisam ser 2D ou 3D.")


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    raw_target_positions: list[int],
    *,
    target_weights: list[float] | None = None,
    horizon_weights: list[float] | None = None,
    use_amp: bool = False,
    grad_scaler: torch.cuda.amp.GradScaler | None = None,
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
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        well_id = batch["well_id"].to(device, non_blocking=True)
        last_target = x[:, -1, raw_target_positions]

        with torch.set_grad_enabled(is_training):
            autocast_enabled = bool(use_amp and device.type == "cuda")
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
                prediction, _ = model(x, well_id)
                loss, loss_metrics = multi_horizon_forecasting_loss(
                    prediction,
                    y,
                    last_target,
                    target_weights=target_weights,
                    horizon_weights=horizon_weights,
                )

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                if grad_scaler is not None and autocast_enabled:
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

        loss_values.append(loss_metrics["total_loss"])
        delta_losses.append(loss_metrics["delta_loss"])
        absolute_losses.append(loss_metrics["absolute_loss"])
        predictions.append(prediction.detach().cpu().numpy().astype(np.float32, copy=False))
        targets.append(y.detach().cpu().numpy().astype(np.float32, copy=False))
        persistence_predictions.append(last_target.unsqueeze(1).expand_as(y).detach().cpu().numpy().astype(np.float32, copy=False))

    y_true = np.concatenate(targets, axis=0)
    y_pred = np.concatenate(predictions, axis=0)
    y_persist = np.concatenate(persistence_predictions, axis=0)

    flat_true = _flatten_metric_matrix(y_true)
    flat_pred = _flatten_metric_matrix(y_pred)
    flat_persist = _flatten_metric_matrix(y_persist)
    model_mae = float(np.mean(np.abs(flat_true - flat_pred)))
    persistence_mae = float(np.mean(np.abs(flat_true - flat_persist)))
    improvement_pct = float((persistence_mae - model_mae) / persistence_mae * 100.0) if persistence_mae != 0 else np.nan

    return {
        "loss": float(np.mean(loss_values)),
        "delta_loss": float(np.mean(delta_losses)),
        "absolute_loss": float(np.mean(absolute_losses)),
        "model_mae": model_mae,
        "persistence_mae": persistence_mae,
        "mae_improvement_pct": improvement_pct,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_persist": y_persist,
    }


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
        true2d = _flatten_metric_matrix(y_true).astype(np.float64, copy=False)
        pred2d = _flatten_metric_matrix(y_pred).astype(np.float64, copy=False)
        if true2d.size == 0:
            return
        if true2d.shape != pred2d.shape:
            raise ValueError("As matrizes de y_true e y_pred precisam ter o mesmo formato.")
        error64 = pred2d - true2d
        self.count += int(true2d.shape[0])
        self.sum_abs_error += np.abs(error64).sum(axis=0)
        self.sum_sq_error += np.square(error64).sum(axis=0)
        self.sum_error += error64.sum(axis=0)
        self.sum_true += true2d.sum(axis=0)
        self.sum_true_sq += np.square(true2d).sum(axis=0)

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
            return {"modelo": label, "MSE": np.nan, "RMSE": np.nan, "MAE": np.nan, "R2_medio": np.nan}
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

    def to_per_feature_metrics(self, baseline: RegressionMetricAccumulator | None = None) -> pd.DataFrame:
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
    partition_by: str | None = None
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
            table_columns[column_name] = values

        if self.partition_by is None:
            self._write_table(table_columns, self.output_dir)
        else:
            partition_values = np.asarray(table_columns[self.partition_by], dtype=object)
            for partition_value in np.unique(partition_values):
                mask = partition_values == partition_value
                partition_dir = self.output_dir / f"{self.partition_by}={partition_value}"
                partition_subset = {
                    name: values[mask]
                    for name, values in table_columns.items()
                }
                self._write_table(partition_subset, partition_dir)

        self._buffer = {}
        self._buffered_rows = 0

    def _write_table(self, columns: dict[str, np.ndarray], output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        table = pa.table({name: _to_arrow_array(values) for name, values in columns.items()})
        self._part_idx += 1
        output_path = output_dir / f"{self.export_prefix}_{self._part_idx:04d}.parquet"
        pq.write_table(table, output_path)
        self.files.append(str(output_path))


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
        "[predict_loader_streaming_v4] "
        f"batch {batch_text} | janelas {window_text}{percent_text} "
        f"| tempo {_format_seconds(elapsed_seconds)} | eta {eta_text} "
        f"| ultimo_batch {last_batch_seconds:.3f}s{memory_text}"
    )


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
        dataset_groups = groups if max_series is None else limit_groups_for_inference(groups, max_series=max_series)
        total_windows = count_group_windows(
            dataset_groups,
            sequence_length=int(dataset.sequence_length),
            forecast_horizon=int(getattr(dataset, "forecast_horizon", 1)),
        )
    elif dataset is not None:
        total_windows = len(dataset)
    if total_windows is None:
        return None, max_batches
    if max_windows is not None:
        total_windows = min(total_windows, int(max_windows))
    if max_batches is not None:
        total_windows = min(total_windows, int(max_batches) * int(batch_size))
    total_batches = int(math.ceil(total_windows / batch_size)) if batch_size else None
    return int(total_windows), total_batches


def _build_prediction_columns(
    *,
    metadata_lookup: dict[str, np.ndarray],
    group_idx: np.ndarray,
    step_idx: np.ndarray,
    timestamp_ns: np.ndarray,
    horizon_step: np.ndarray,
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
        "horizon_step": horizon_step.astype(np.int16, copy=False),
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


def _build_side_by_side_metrics_df(
    *,
    key_name: str,
    model_map: dict[str, RegressionMetricAccumulator],
    baseline_map: dict[str, RegressionMetricAccumulator],
) -> pd.DataFrame:
    require_tabular_stack()
    rows = []
    all_keys = sorted(set(model_map) | set(baseline_map))
    for key in all_keys:
        model_metrics = model_map[key].to_global_metrics("model") if key in model_map else {"MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "R2_medio": np.nan}
        baseline_metrics = baseline_map[key].to_global_metrics("baseline") if key in baseline_map else {"MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "R2_medio": np.nan}
        model_mae = float(model_metrics["MAE"])
        baseline_mae = float(baseline_metrics["MAE"])
        model_rmse = float(model_metrics["RMSE"])
        baseline_rmse = float(baseline_metrics["RMSE"])
        rows.append(
            {
                key_name: key,
                "model_MAE": model_mae,
                "baseline_MAE": baseline_mae,
                "mae_melhora_pct": float((baseline_mae - model_mae) / baseline_mae * 100.0) if baseline_mae not in [0.0, np.nan] and baseline_mae == baseline_mae else np.nan,
                "model_RMSE": model_rmse,
                "baseline_RMSE": baseline_rmse,
                "rmse_melhora_pct": float((baseline_rmse - model_rmse) / baseline_rmse * 100.0) if baseline_rmse not in [0.0, np.nan] and baseline_rmse == baseline_rmse else np.nan,
                "model_R2": float(model_metrics["R2_medio"]),
                "baseline_R2": float(baseline_metrics["R2_medio"]),
                "linhas_avaliadas": int(model_map[key].count if key in model_map else baseline_map[key].count),
            }
        )
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
    metric_pairs = [("MSE", "mse_melhora_pct"), ("RMSE", "rmse_melhora_pct"), ("MAE", "mae_melhora_pct")]
    for metric_name, improvement_column in metric_pairs:
        baseline_value = float(enriched.loc[baseline_idx, metric_name])
        model_value = float(enriched.loc[model_idx, metric_name])
        if baseline_value != 0:
            enriched.loc[model_idx, improvement_column] = float((baseline_value - model_value) / baseline_value * 100.0)
    return enriched


def predict_loader_streaming(
    model: nn.Module,
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
    export_partition_by: str | None = None,
    preview_rows: int = 2048,
    max_batches: int | None = None,
    max_windows: int | None = None,
    max_series: int | None = None,
    progress_every: int = 25,
    log_memory: bool = False,
    model_label: str = "Modelo_v4",
    baseline_label: str = "Persistencia",
    use_amp: bool = False,
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
        collect_metrics or preview_rows > 0 or (export_predictions and export_scale in {"original", "both"})
    )

    model_scaled_metrics = RegressionMetricAccumulator(feature_names=feature_names) if collect_metrics else None
    baseline_scaled_metrics = RegressionMetricAccumulator(feature_names=feature_names) if collect_metrics else None
    model_original_metrics = RegressionMetricAccumulator(feature_names=feature_names) if collect_metrics and bundle is not None else None
    baseline_original_metrics = RegressionMetricAccumulator(feature_names=feature_names) if collect_metrics and bundle is not None else None

    class_model_scaled: dict[str, RegressionMetricAccumulator] = {}
    class_baseline_scaled: dict[str, RegressionMetricAccumulator] = {}
    class_model_original: dict[str, RegressionMetricAccumulator] = {}
    class_baseline_original: dict[str, RegressionMetricAccumulator] = {}
    well_model_scaled: dict[str, RegressionMetricAccumulator] = {}
    well_baseline_scaled: dict[str, RegressionMetricAccumulator] = {}
    well_model_original: dict[str, RegressionMetricAccumulator] = {}
    well_baseline_original: dict[str, RegressionMetricAccumulator] = {}
    series_model_scaled: dict[str, RegressionMetricAccumulator] = {}
    series_baseline_scaled: dict[str, RegressionMetricAccumulator] = {}
    series_model_original: dict[str, RegressionMetricAccumulator] = {}
    series_baseline_original: dict[str, RegressionMetricAccumulator] = {}
    horizon_model_scaled: dict[str, RegressionMetricAccumulator] = {}
    horizon_baseline_scaled: dict[str, RegressionMetricAccumulator] = {}
    horizon_model_original: dict[str, RegressionMetricAccumulator] = {}
    horizon_baseline_original: dict[str, RegressionMetricAccumulator] = {}

    chunk_writer = PredictionChunkWriter(Path(export_dir), export_chunk_rows, partition_by=export_partition_by) if export_predictions else None
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

            batch_window_count = int(batch["x"].shape[0])
            group_idx_np = batch["group_idx"].cpu().numpy()
            selected_indices = np.arange(batch_window_count, dtype=np.int64)

            if max_series is not None:
                selected_indices = selected_indices[group_idx_np[selected_indices] < int(max_series)]
                if selected_indices.size == 0:
                    if batch_window_count > 0 and int(group_idx_np.min()) >= int(max_series):
                        break
                    continue

            if max_windows is not None:
                remaining_windows = int(max_windows) - processed_windows
                if remaining_windows <= 0:
                    break
                selected_indices = selected_indices[:remaining_windows]
                if selected_indices.size == 0:
                    break

            if selected_indices.size == batch_window_count:
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
            autocast_enabled = bool(use_amp and device.type == "cuda")
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
                prediction, _ = model(x_device, well_id_device)

            scaled_pred_3d = prediction.detach().cpu().numpy().astype(np.float32, copy=False)
            scaled_true_3d = y_cpu.cpu().numpy().astype(np.float32, copy=False)
            scaled_baseline_3d = last_target_cpu.unsqueeze(1).expand_as(y_cpu).cpu().numpy().astype(np.float32, copy=False)

            original_true_3d = None
            original_pred_3d = None
            original_baseline_3d = None
            if needs_original_arrays:
                original_true_3d = inverse_transform_targets(scaled_true_3d, bundle).astype(np.float32, copy=False)
                original_pred_3d = inverse_transform_targets(scaled_pred_3d, bundle).astype(np.float32, copy=False)
                original_baseline_3d = inverse_transform_targets(scaled_baseline_3d, bundle).astype(np.float32, copy=False)

            if collect_metrics and model_scaled_metrics is not None and baseline_scaled_metrics is not None:
                model_scaled_metrics.update(scaled_true_3d, scaled_pred_3d)
                baseline_scaled_metrics.update(scaled_true_3d, scaled_baseline_3d)
                if model_original_metrics is not None and baseline_original_metrics is not None:
                    model_original_metrics.update(original_true_3d, original_pred_3d)
                    baseline_original_metrics.update(original_true_3d, original_baseline_3d)

                horizon_count = scaled_true_3d.shape[1]
                batch_class_labels = metadata_lookup["class_label"][selected_group_idx]
                batch_well_names = metadata_lookup["well_name"][selected_group_idx]
                batch_series_ids = metadata_lookup["series_id"][selected_group_idx]
                repeated_class_labels = np.repeat(batch_class_labels, horizon_count)
                repeated_well_names = np.repeat(batch_well_names, horizon_count)
                repeated_series_ids = np.repeat(batch_series_ids, horizon_count)

                flat_scaled_true = scaled_true_3d.reshape(-1, scaled_true_3d.shape[-1])
                flat_scaled_pred = scaled_pred_3d.reshape(-1, scaled_pred_3d.shape[-1])
                flat_scaled_baseline = scaled_baseline_3d.reshape(-1, scaled_baseline_3d.shape[-1])
                _update_group_metric_map(class_model_scaled, feature_names=feature_names, group_labels=repeated_class_labels, y_true=flat_scaled_true, y_pred=flat_scaled_pred)
                _update_group_metric_map(class_baseline_scaled, feature_names=feature_names, group_labels=repeated_class_labels, y_true=flat_scaled_true, y_pred=flat_scaled_baseline)
                _update_group_metric_map(well_model_scaled, feature_names=feature_names, group_labels=repeated_well_names, y_true=flat_scaled_true, y_pred=flat_scaled_pred)
                _update_group_metric_map(well_baseline_scaled, feature_names=feature_names, group_labels=repeated_well_names, y_true=flat_scaled_true, y_pred=flat_scaled_baseline)
                _update_group_metric_map(series_model_scaled, feature_names=feature_names, group_labels=repeated_series_ids, y_true=flat_scaled_true, y_pred=flat_scaled_pred)
                _update_group_metric_map(series_baseline_scaled, feature_names=feature_names, group_labels=repeated_series_ids, y_true=flat_scaled_true, y_pred=flat_scaled_baseline)
                for horizon_idx in range(horizon_count):
                    horizon_key = str(horizon_idx + 1)
                    if horizon_key not in horizon_model_scaled:
                        horizon_model_scaled[horizon_key] = RegressionMetricAccumulator(feature_names=feature_names)
                        horizon_baseline_scaled[horizon_key] = RegressionMetricAccumulator(feature_names=feature_names)
                    horizon_model_scaled[horizon_key].update(scaled_true_3d[:, horizon_idx, :], scaled_pred_3d[:, horizon_idx, :])
                    horizon_baseline_scaled[horizon_key].update(scaled_true_3d[:, horizon_idx, :], scaled_baseline_3d[:, horizon_idx, :])

                if model_original_metrics is not None and baseline_original_metrics is not None:
                    flat_original_true = original_true_3d.reshape(-1, original_true_3d.shape[-1])
                    flat_original_pred = original_pred_3d.reshape(-1, original_pred_3d.shape[-1])
                    flat_original_baseline = original_baseline_3d.reshape(-1, original_baseline_3d.shape[-1])
                    _update_group_metric_map(class_model_original, feature_names=feature_names, group_labels=repeated_class_labels, y_true=flat_original_true, y_pred=flat_original_pred)
                    _update_group_metric_map(class_baseline_original, feature_names=feature_names, group_labels=repeated_class_labels, y_true=flat_original_true, y_pred=flat_original_baseline)
                    _update_group_metric_map(well_model_original, feature_names=feature_names, group_labels=repeated_well_names, y_true=flat_original_true, y_pred=flat_original_pred)
                    _update_group_metric_map(well_baseline_original, feature_names=feature_names, group_labels=repeated_well_names, y_true=flat_original_true, y_pred=flat_original_baseline)
                    _update_group_metric_map(series_model_original, feature_names=feature_names, group_labels=repeated_series_ids, y_true=flat_original_true, y_pred=flat_original_pred)
                    _update_group_metric_map(series_baseline_original, feature_names=feature_names, group_labels=repeated_series_ids, y_true=flat_original_true, y_pred=flat_original_baseline)
                    for horizon_idx in range(horizon_count):
                        horizon_key = str(horizon_idx + 1)
                        if horizon_key not in horizon_model_original:
                            horizon_model_original[horizon_key] = RegressionMetricAccumulator(feature_names=feature_names)
                            horizon_baseline_original[horizon_key] = RegressionMetricAccumulator(feature_names=feature_names)
                        horizon_model_original[horizon_key].update(original_true_3d[:, horizon_idx, :], original_pred_3d[:, horizon_idx, :])
                        horizon_baseline_original[horizon_key].update(original_true_3d[:, horizon_idx, :], original_baseline_3d[:, horizon_idx, :])

            flat_group_idx = np.repeat(selected_group_idx, scaled_true_3d.shape[1])
            flat_step_idx = step_idx_np.reshape(-1)
            flat_timestamp_ns = timestamp_ns_np.reshape(-1)
            flat_horizon_step = np.tile(np.arange(1, scaled_true_3d.shape[1] + 1, dtype=np.int16), len(selected_group_idx))
            flat_scaled_true = scaled_true_3d.reshape(-1, scaled_true_3d.shape[-1])
            flat_scaled_pred = scaled_pred_3d.reshape(-1, scaled_pred_3d.shape[-1])
            flat_scaled_baseline = scaled_baseline_3d.reshape(-1, scaled_baseline_3d.shape[-1])
            flat_original_true = original_true_3d.reshape(-1, original_true_3d.shape[-1]) if original_true_3d is not None else None
            flat_original_pred = original_pred_3d.reshape(-1, original_pred_3d.shape[-1]) if original_pred_3d is not None else None
            flat_original_baseline = original_baseline_3d.reshape(-1, original_baseline_3d.shape[-1]) if original_baseline_3d is not None else None

            if chunk_writer is not None:
                chunk_writer.append(
                    _build_prediction_columns(
                        metadata_lookup=metadata_lookup,
                        group_idx=flat_group_idx,
                        step_idx=flat_step_idx,
                        timestamp_ns=flat_timestamp_ns,
                        horizon_step=flat_horizon_step,
                        feature_names=feature_names,
                        scaled_true=flat_scaled_true,
                        scaled_pred=flat_scaled_pred,
                        scaled_baseline=flat_scaled_baseline,
                        original_true=flat_original_true,
                        original_pred=flat_original_pred,
                        original_baseline=flat_original_baseline,
                        include_scaled=export_scale in {"scaled", "both"},
                        include_original=export_scale in {"original", "both"},
                    )
                )

            if preview_rows > 0 and preview_count < preview_rows:
                preview_take = min(int(preview_rows) - preview_count, len(flat_group_idx))
                preview_parts.append(
                    pd.DataFrame(
                        _build_prediction_columns(
                            metadata_lookup=metadata_lookup,
                            group_idx=flat_group_idx[:preview_take],
                            step_idx=flat_step_idx[:preview_take],
                            timestamp_ns=flat_timestamp_ns[:preview_take],
                            horizon_step=flat_horizon_step[:preview_take],
                            feature_names=feature_names,
                            scaled_true=flat_scaled_true[:preview_take],
                            scaled_pred=flat_scaled_pred[:preview_take],
                            scaled_baseline=flat_scaled_baseline[:preview_take],
                            original_true=flat_original_true[:preview_take] if flat_original_true is not None else None,
                            original_pred=flat_original_pred[:preview_take] if flat_original_pred is not None else None,
                            original_baseline=flat_original_baseline[:preview_take] if flat_original_baseline is not None else None,
                            include_scaled=True,
                            include_original=flat_original_true is not None,
                        )
                    )
                )
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
    well_metrics_scaled_df = None
    well_metrics_original_df = None
    series_metrics_scaled_df = None
    series_metrics_original_df = None
    horizon_metrics_scaled_df = None
    horizon_metrics_original_df = None

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
        class_metrics_scaled_df = _build_side_by_side_metrics_df(key_name="class_label", model_map=class_model_scaled, baseline_map=class_baseline_scaled)
        well_metrics_scaled_df = _build_side_by_side_metrics_df(key_name="well_name", model_map=well_model_scaled, baseline_map=well_baseline_scaled)
        series_metrics_scaled_df = _build_side_by_side_metrics_df(key_name="series_id", model_map=series_model_scaled, baseline_map=series_baseline_scaled)
        horizon_metrics_scaled_df = _build_side_by_side_metrics_df(key_name="horizon_step", model_map=horizon_model_scaled, baseline_map=horizon_baseline_scaled)

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
        class_metrics_original_df = _build_side_by_side_metrics_df(key_name="class_label", model_map=class_model_original, baseline_map=class_baseline_original)
        well_metrics_original_df = _build_side_by_side_metrics_df(key_name="well_name", model_map=well_model_original, baseline_map=well_baseline_original)
        series_metrics_original_df = _build_side_by_side_metrics_df(key_name="series_id", model_map=series_model_original, baseline_map=series_baseline_original)
        horizon_metrics_original_df = _build_side_by_side_metrics_df(key_name="horizon_step", model_map=horizon_model_original, baseline_map=horizon_baseline_original)

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
        well_metrics_scaled_df=well_metrics_scaled_df,
        well_metrics_original_df=well_metrics_original_df,
        series_metrics_scaled_df=series_metrics_scaled_df,
        series_metrics_original_df=series_metrics_original_df,
        horizon_metrics_scaled_df=horizon_metrics_scaled_df,
        horizon_metrics_original_df=horizon_metrics_original_df,
    )


def compute_global_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict[str, float | str]:
    accumulator = RegressionMetricAccumulator(feature_names=list(BASE_TARGET_COLUMNS))
    accumulator.update(y_true, y_pred)
    return accumulator.to_global_metrics(label)


def compute_per_feature_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_baseline: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    require_tabular_stack()
    model_metrics = RegressionMetricAccumulator(feature_names=feature_names)
    baseline_metrics = RegressionMetricAccumulator(feature_names=feature_names)
    model_metrics.update(y_true, y_pred)
    baseline_metrics.update(y_true, y_baseline)
    return model_metrics.to_per_feature_metrics(baseline=baseline_metrics)


def export_streaming_result_tables(
    result: StreamingPredictionResult,
    output_dir: str | Path,
    prefix: str = "teste_v4",
) -> dict[str, str]:
    require_tabular_stack()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    table_map = {
        "global_scaled": result.global_metrics_scaled_df,
        "global_original": result.global_metrics_original_df,
        "per_feature_scaled": result.per_feature_scaled_df,
        "per_feature_original": result.per_feature_original_df,
        "class_scaled": result.class_metrics_scaled_df,
        "class_original": result.class_metrics_original_df,
        "well_scaled": result.well_metrics_scaled_df,
        "well_original": result.well_metrics_original_df,
        "series_scaled": result.series_metrics_scaled_df,
        "series_original": result.series_metrics_original_df,
        "horizon_scaled": result.horizon_metrics_scaled_df,
        "horizon_original": result.horizon_metrics_original_df,
    }
    saved_files = {}
    for suffix, df in table_map.items():
        if df is None or df.empty:
            continue
        file_path = output_path / f"{prefix}_{suffix}.csv"
        df.to_csv(file_path, index=False)
        saved_files[suffix] = str(file_path)
    return saved_files
