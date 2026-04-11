from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import random
from typing import Any

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

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
            "Instale `torch` para executar os notebooks de treino e teste da versao2."
        )


def require_tabular_stack() -> None:
    if pd is None or StandardScaler is None:
        raise ImportError(
            "Dependencias de dados nao estao instaladas neste ambiente. "
            "Instale `pandas`, `pyarrow` e `scikit-learn` para executar o pipeline_v2."
        )


def discover_balanced_normal_files(
    dataset_root: Path,
    class_labels: tuple[str, ...] = ("0",),
    max_files_per_well: int | None = 25,
) -> pd.DataFrame:
    require_tabular_stack()
    rows: list[dict[str, Any]] = []
    for class_label in class_labels:
        class_dir = dataset_root / class_label
        for file_path in sorted(class_dir.glob("*.parquet")):
            well_name, start_token = file_path.stem.split("_", 1)
            rows.append(
                {
                    "class_label": class_label,
                    "well_name": well_name,
                    "start_token": start_token,
                    "file_path": str(file_path.resolve()),
                }
            )

    manifest = pd.DataFrame(rows).sort_values(["well_name", "start_token"]).reset_index(drop=True)
    if max_files_per_well is None:
        return manifest

    selected_parts = []
    for _, well_df in manifest.groupby("well_name", sort=True):
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
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df[column] = df[column].interpolate(limit_direction="both")
        df[column] = df[column].ffill().bfill()

    for column in available_aux:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if column in STATE_COLUMNS:
            df[column] = df[column].ffill().bfill()
            if df[column].isna().any():
                mode_value = df[column].mode(dropna=True)
                if len(mode_value) > 0:
                    df[column] = df[column].fillna(mode_value.iloc[0])
                else:
                    df[column] = df[column].fillna(0.0)
        else:
            df[column] = df[column].interpolate(limit_direction="both")
            df[column] = df[column].ffill().bfill()

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
        series = pd.to_numeric(training_reference_df[column], errors="coerce")
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
        target_values = clipped[BASE_TARGET_COLUMNS].to_numpy(dtype=np.float32)
        target_scaler.partial_fit(target_values)

        if fit_aux:
            aux_values = clipped[auxiliary_columns].to_numpy(dtype=np.float32)
            aux_scaler.partial_fit(aux_values)

        diff_features, dev_features, std_features = build_derived_feature_arrays(
            clipped,
            target_columns=BASE_TARGET_COLUMNS,
            rolling_window=rolling_window,
        )
        diff_scaler.partial_fit(diff_features.astype(np.float32))
        dev_scaler.partial_fit(dev_features.astype(np.float32))
        std_scaler.partial_fit(std_features.astype(np.float32))

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
    well_name, start_token = Path(file_path).stem.split("_", 1)
    frame = clean_base_frame(file_path, candidate_auxiliary_columns=bundle.auxiliary_columns)
    return transform_clean_frame_to_engineered_features(
        frame=frame,
        bundle=bundle,
        series_id=f"{well_name}_{start_token}",
        well_name=well_name,
        file_path=str(Path(file_path).resolve()),
    )


def transform_clean_frame_to_engineered_features(
    frame: pd.DataFrame,
    bundle: PreprocessingBundle,
    *,
    series_id: str,
    well_name: str = "single_well",
    file_path: str = "",
) -> pd.DataFrame:
    require_tabular_stack()
    clipped = apply_clip_bounds(frame, bundle.clip_bounds)

    target_df = clipped[bundle.target_columns].copy()
    target_scaled = (target_df.to_numpy(dtype=np.float32) - np.asarray(bundle.target_scaler_mean)) / np.asarray(bundle.target_scaler_scale)
    target_scaled_df = pd.DataFrame(target_scaled, columns=[f"target__{column}" for column in bundle.target_columns])

    raw_target_scaled_df = pd.DataFrame(
        target_scaled,
        columns=[f"raw__{column}" for column in bundle.target_columns],
    )

    input_parts = [raw_target_scaled_df]
    if bundle.auxiliary_columns:
        aux_values = clipped[bundle.auxiliary_columns].to_numpy(dtype=np.float32)
        aux_scaled = (aux_values - np.asarray(bundle.aux_scaler_mean)) / np.asarray(bundle.aux_scaler_scale)
        input_parts.append(pd.DataFrame(aux_scaled, columns=[f"raw__{column}" for column in bundle.auxiliary_columns]))

    diff_features, dev_features, std_features = build_derived_feature_arrays(
        clipped,
        target_columns=bundle.target_columns,
        rolling_window=bundle.rolling_window,
    )
    diff_scaled = (diff_features - np.asarray(bundle.diff_scaler_mean)) / np.asarray(bundle.diff_scaler_scale)
    dev_scaled = (dev_features - np.asarray(bundle.dev_scaler_mean)) / np.asarray(bundle.dev_scaler_scale)
    std_scaled = (std_features - np.asarray(bundle.std_scaler_mean)) / np.asarray(bundle.std_scaler_scale)

    input_parts.append(pd.DataFrame(diff_scaled, columns=[f"diff1__{column}" for column in bundle.target_columns]))
    input_parts.append(pd.DataFrame(dev_scaled, columns=[f"dev_roll{bundle.rolling_window}__{column}" for column in bundle.target_columns]))
    input_parts.append(pd.DataFrame(std_scaled, columns=[f"std_roll{bundle.rolling_window}__{column}" for column in bundle.target_columns]))

    metadata_df = pd.DataFrame(
        {
            "series_id": [series_id] * len(clipped),
            "well_name": [well_name] * len(clipped),
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
