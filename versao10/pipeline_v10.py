from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import sys
from typing import Any

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from sklearn.utils.class_weight import compute_class_weight
except ImportError:
    compute_class_weight = None

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    WeightedRandomSampler = None

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from versao3.pipeline_v3 import set_seed  # noqa: E402
from versao4.pipeline_v4 import require_tabular_stack, require_torch  # noqa: E402
from versao9.pipeline_v9 import (  # noqa: E402
    build_metrics_table,
    discover_series_manifest,
    evaluate_predictions,
    export_evaluation_artifacts,
    fit_lgbm_baseline,
    fit_random_forest_baseline,
    fit_xgboost_baseline,
    load_attribute_catalog,
    load_event_catalog,
    plot_confusion_matrix_for_predictions,
    require_classification_stack,
    require_plotting_stack,
    run_baseline_suite,
    save_bundle as _save_bundle_v9,
    stratified_split_manifest,
)


ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "reports_v10"
IGNORE_INDEX = -100
FULL_FEATURE_COLUMNS = [
    "ABER-CKGL",
    "ABER-CKP",
    "ESTADO-DHSV",
    "ESTADO-M1",
    "ESTADO-M2",
    "ESTADO-PXO",
    "ESTADO-SDV-GL",
    "ESTADO-SDV-P",
    "ESTADO-W1",
    "ESTADO-W2",
    "ESTADO-XO",
    "P-ANULAR",
    "P-JUS-BS",
    "P-JUS-CKGL",
    "P-JUS-CKP",
    "P-MON-CKGL",
    "P-MON-CKP",
    "P-MON-SDV-P",
    "P-PDG",
    "PT-P",
    "P-TPT",
    "QBS",
    "QGL",
    "T-JUS-CKP",
    "T-MON-CKP",
    "T-PDG",
    "T-TPT",
]
STATE_SENSOR_COLUMNS = [name for name in FULL_FEATURE_COLUMNS if name.startswith("ESTADO-")]
CONTINUOUS_SENSOR_COLUMNS = [name for name in FULL_FEATURE_COLUMNS if name not in STATE_SENSOR_COLUMNS]
OBSERVATION_CLASS_CODES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 101, 102, 103, 105, 106, 107, 108, 109]
OBSERVATION_STATE_CODES = [0, 1, 2, 3, 4, 5, 6, 7, 8]
SOURCE_TYPE_MAPPING = {"well": 0, "simulated": 1, "drawn": 2}
SERIES_STAT_NAMES = [
    "mean",
    "std",
    "min",
    "max",
    "median",
    "first",
    "last",
    "slope",
    "mean_abs_diff",
]


@dataclass
class ClassificationBundle:
    selected_columns: list[str]
    continuous_columns: list[str]
    state_columns: list[str]
    sequence_length: int
    scaler_mean: list[float]
    scaler_scale: list[float]
    class_labels: list[int]
    class_names: list[str]
    class_descriptions: dict[str, str]
    statistical_feature_names: list[str]
    split_counts: dict[str, int]
    selected_files: dict[str, list[str]]
    observation_class_codes: list[int]
    observation_state_codes: list[int]
    observation_class_mapping: dict[str, int]
    observation_state_mapping: dict[str, int]
    source_mapping: dict[str, int]


@dataclass
class PreparedClassificationArtifacts:
    run_dir: str
    bundle_path: str
    manifest_path: str
    attribute_catalog_path: str
    event_catalog_path: str
    split_npz_paths: dict[str, str]
    split_metadata_paths: dict[str, str]


@dataclass
class MultiTaskTrainingSummary:
    model_name: str
    checkpoint_path: str
    config_path: str
    history_path: str
    best_epoch: int
    best_val_macro_f1: float
    best_val_accuracy: float
    best_val_balanced_accuracy: float


def _write_json(payload: dict[str, Any], output_path: str | Path) -> None:
    Path(output_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_bundle(bundle: ClassificationBundle, output_path: str | Path) -> None:
    _save_bundle_v9(bundle, output_path)


def load_bundle(bundle_path: str | Path) -> ClassificationBundle:
    payload = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    return ClassificationBundle(**payload)


def _safe_numeric_matrix(values: np.ndarray, clip_abs: float = 1e12) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    matrix = np.clip(matrix, -clip_abs, clip_abs)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix


def _safe_scale(values: np.ndarray) -> list[float]:
    scale = np.asarray(values, dtype=np.float64)
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    return scale.tolist()


def build_statistical_feature_names(feature_columns: list[str]) -> list[str]:
    names = []
    for column in feature_columns:
        for stat_name in SERIES_STAT_NAMES:
            names.append(f"{stat_name}__{column}")
    return names


def compute_statistical_feature_vector(
    sequence_array: np.ndarray,
    feature_columns: list[str],
) -> np.ndarray:
    sequence = np.asarray(sequence_array, dtype=np.float64)
    if sequence.ndim != 2:
        raise ValueError("A sequencia precisa ter duas dimensoes: [tempo, features].")

    time_axis = np.arange(sequence.shape[0], dtype=np.float64)
    feature_values = []
    for feature_idx, _ in enumerate(feature_columns):
        values = sequence[:, feature_idx]
        if len(values) > 1:
            slope = float(np.polyfit(time_axis, values, deg=1)[0])
            mean_abs_diff = float(np.abs(np.diff(values)).mean())
        else:
            slope = 0.0
            mean_abs_diff = 0.0
        feature_values.extend(
            [
                float(values.mean()),
                float(values.std()),
                float(values.min()),
                float(values.max()),
                float(np.median(values)),
                float(values[0]),
                float(values[-1]),
                slope,
                mean_abs_diff,
            ]
        )
    return np.asarray(feature_values, dtype=np.float32)


def _prepare_raw_frame(file_path: str | Path) -> pd.DataFrame:
    require_tabular_stack()
    frame = pd.read_parquet(file_path).copy()
    if "timestamp" not in frame.columns:
        index_name = frame.index.name or "timestamp"
        frame = frame.reset_index().rename(columns={index_name: "timestamp"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    for column_name in FULL_FEATURE_COLUMNS + ["class", "state"]:
        if column_name not in frame.columns:
            frame[column_name] = np.nan
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame


def _fill_series(values: np.ndarray, *, discrete: bool) -> np.ndarray:
    series = pd.Series(values, dtype="float64")
    if series.notna().sum() == 0:
        return np.zeros(len(series), dtype=np.float64)
    if not discrete:
        series = series.interpolate(method="linear", limit_direction="both")
    series = series.ffill().bfill().fillna(0.0)
    return series.to_numpy(dtype=np.float64)


def _compute_frozen_mask(values: np.ndarray, *, tolerance: float = 1e-12, min_run: int = 3) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return np.zeros(0, dtype=np.float64)
    frozen = np.zeros(len(arr), dtype=np.float64)
    if min_run <= 1:
        return frozen
    for end in range(min_run - 1, len(arr)):
        window = arr[end - min_run + 1 : end + 1]
        if np.isfinite(window).all() and np.max(window) - np.min(window) <= tolerance:
            frozen[end - min_run + 1 : end + 1] = 1.0
    return frozen


def _resample_numeric(values: np.ndarray, sequence_length: int, *, discrete: bool = False) -> np.ndarray:
    values_arr = np.asarray(values, dtype=np.float64)
    if len(values_arr) == 0:
        raise ValueError("Nao e possivel reamostrar uma sequencia vazia.")
    if len(values_arr) == 1:
        return np.repeat(values_arr, repeats=sequence_length)

    source_pos = np.linspace(0.0, 1.0, len(values_arr), dtype=np.float64)
    target_pos = np.linspace(0.0, 1.0, sequence_length, dtype=np.float64)
    if discrete:
        interpolated_idx = np.interp(target_pos, source_pos, np.arange(len(values_arr), dtype=np.float64))
        nearest_idx = np.rint(interpolated_idx).astype(np.int64)
        nearest_idx = np.clip(nearest_idx, 0, len(values_arr) - 1)
        return values_arr[nearest_idx]
    return np.interp(target_pos, source_pos, values_arr)


def _resample_labels(values: np.ndarray, sequence_length: int, mapping: dict[str, int]) -> np.ndarray:
    values_arr = np.asarray(values, dtype=object)
    if len(values_arr) == 0:
        raise ValueError("Nao e possivel reamostrar labels vazios.")
    if len(values_arr) == 1:
        single = values_arr[0]
        if pd.isna(single):
            return np.full(sequence_length, IGNORE_INDEX, dtype=np.int64)
        return np.full(sequence_length, mapping.get(str(int(single)), IGNORE_INDEX), dtype=np.int64)

    source_pos = np.linspace(0.0, 1.0, len(values_arr), dtype=np.float64)
    target_pos = np.linspace(0.0, 1.0, sequence_length, dtype=np.float64)
    interpolated_idx = np.interp(target_pos, source_pos, np.arange(len(values_arr), dtype=np.float64))
    nearest_idx = np.rint(interpolated_idx).astype(np.int64)
    nearest_idx = np.clip(nearest_idx, 0, len(values_arr) - 1)

    mapped = []
    for idx in nearest_idx:
        value = values_arr[idx]
        if pd.isna(value):
            mapped.append(IGNORE_INDEX)
        else:
            mapped.append(mapping.get(str(int(value)), IGNORE_INDEX))
    return np.asarray(mapped, dtype=np.int64)


def _compute_balanced_class_weights(class_labels: list[int], y: np.ndarray) -> np.ndarray:
    y_arr = np.asarray(y, dtype=np.int64)
    counts = np.asarray([(y_arr == int(label)).sum() for label in class_labels], dtype=np.float64)
    n_present = max(int((counts > 0).sum()), 1)
    total = max(float(len(y_arr)), 1.0)
    weights = np.zeros_like(counts, dtype=np.float64)
    present_mask = counts > 0
    weights[present_mask] = total / (n_present * counts[present_mask])
    weights[~present_mask] = 0.0
    return weights


def _compute_balanced_index_weights(num_classes: int, y_indices: np.ndarray, ignore_index: int = IGNORE_INDEX) -> np.ndarray:
    values = np.asarray(y_indices, dtype=np.int64)
    valid = values != ignore_index
    counts = np.bincount(values[valid], minlength=num_classes).astype(np.float64)
    n_present = max(int((counts > 0).sum()), 1)
    total = max(float(valid.sum()), 1.0)
    weights = np.zeros_like(counts, dtype=np.float64)
    present_mask = counts > 0
    weights[present_mask] = total / (n_present * counts[present_mask])
    weights[~present_mask] = 0.0
    return weights


def fit_classification_bundle(
    train_manifest: pd.DataFrame,
    split_manifest: pd.DataFrame,
    *,
    dataset_root: str | Path,
    sequence_length: int = 180,
) -> ClassificationBundle:
    require_classification_stack()
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    for file_path in train_manifest["file_path"]:
        frame = _prepare_raw_frame(file_path)
        columns = []
        for column_name in FULL_FEATURE_COLUMNS:
            raw_values = pd.to_numeric(frame[column_name], errors="coerce").to_numpy(dtype=np.float64)
            filled_values = _fill_series(raw_values, discrete=column_name in STATE_SENSOR_COLUMNS)
            columns.append(
                _resample_numeric(
                    filled_values,
                    sequence_length,
                    discrete=column_name in STATE_SENSOR_COLUMNS,
                )
            )
        sequence = _safe_numeric_matrix(np.stack(columns, axis=1))
        scaler.partial_fit(sequence)

    event_catalog = load_event_catalog(dataset_root)
    class_labels = event_catalog["class_label"].astype(int).tolist()
    class_names = [str(value) for value in class_labels]
    class_descriptions = {
        str(row["class_label"]): str(row["description"])
        for _, row in event_catalog.iterrows()
    }

    observation_class_mapping = {
        str(code): idx for idx, code in enumerate(OBSERVATION_CLASS_CODES)
    }
    observation_state_mapping = {
        str(code): idx for idx, code in enumerate(OBSERVATION_STATE_CODES)
    }
    statistical_feature_names = build_statistical_feature_names(FULL_FEATURE_COLUMNS)
    selected_files = {
        split_name: split_df["file_path"].tolist()
        for split_name, split_df in split_manifest.groupby("split", sort=False)
    }
    split_counts = split_manifest["split"].value_counts().sort_index().to_dict()

    return ClassificationBundle(
        selected_columns=FULL_FEATURE_COLUMNS.copy(),
        continuous_columns=CONTINUOUS_SENSOR_COLUMNS.copy(),
        state_columns=STATE_SENSOR_COLUMNS.copy(),
        sequence_length=int(sequence_length),
        scaler_mean=scaler.mean_.tolist(),
        scaler_scale=_safe_scale(scaler.scale_),
        class_labels=class_labels,
        class_names=class_names,
        class_descriptions=class_descriptions,
        statistical_feature_names=statistical_feature_names,
        split_counts=split_counts,
        selected_files=selected_files,
        observation_class_codes=OBSERVATION_CLASS_CODES.copy(),
        observation_state_codes=OBSERVATION_STATE_CODES.copy(),
        observation_class_mapping=observation_class_mapping,
        observation_state_mapping=observation_state_mapping,
        source_mapping=SOURCE_TYPE_MAPPING.copy(),
    )


def transform_manifest_to_arrays(
    manifest: pd.DataFrame,
    bundle: ClassificationBundle,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    require_tabular_stack()

    mean_arr = np.asarray(bundle.scaler_mean, dtype=np.float64)
    scale_arr = np.asarray(bundle.scaler_scale, dtype=np.float64)

    sequence_parts = []
    tabular_parts = []
    missing_parts = []
    frozen_parts = []
    step_class_parts = []
    step_state_parts = []
    source_parts = []
    labels = []
    metadata_rows = []

    for _, row in manifest.iterrows():
        frame = _prepare_raw_frame(row["file_path"])
        sequence_columns = []
        missing_columns = []
        frozen_columns = []

        for column_name in bundle.selected_columns:
            raw_values = pd.to_numeric(frame[column_name], errors="coerce").to_numpy(dtype=np.float64)
            missing_mask = (~np.isfinite(raw_values)).astype(np.float64)
            filled_values = _fill_series(raw_values, discrete=column_name in bundle.state_columns)
            frozen_mask = _compute_frozen_mask(filled_values)

            sequence_columns.append(
                _resample_numeric(
                    filled_values,
                    bundle.sequence_length,
                    discrete=column_name in bundle.state_columns,
                )
            )
            missing_columns.append(
                _resample_numeric(
                    missing_mask,
                    bundle.sequence_length,
                    discrete=True,
                )
            )
            frozen_columns.append(
                _resample_numeric(
                    frozen_mask,
                    bundle.sequence_length,
                    discrete=True,
                )
            )

        sequence = _safe_numeric_matrix(np.stack(sequence_columns, axis=1))
        sequence_scaled = _safe_numeric_matrix((sequence - mean_arr) / scale_arr).astype(np.float32, copy=False)
        missing_seq = _safe_numeric_matrix(np.stack(missing_columns, axis=1)).astype(np.float32, copy=False)
        frozen_seq = _safe_numeric_matrix(np.stack(frozen_columns, axis=1)).astype(np.float32, copy=False)
        statistical_vector = compute_statistical_feature_vector(sequence_scaled, bundle.selected_columns)

        step_class = _resample_labels(
            frame["class"].to_numpy(),
            bundle.sequence_length,
            bundle.observation_class_mapping,
        )
        step_state = _resample_labels(
            frame["state"].to_numpy(),
            bundle.sequence_length,
            bundle.observation_state_mapping,
        )

        sequence_parts.append(sequence_scaled)
        tabular_parts.append(statistical_vector)
        missing_parts.append(missing_seq)
        frozen_parts.append(frozen_seq)
        step_class_parts.append(step_class)
        step_state_parts.append(step_state)
        source_parts.append(int(bundle.source_mapping.get(str(row["source_type"]), 0)))
        labels.append(int(row["class_label_int"]))

        metadata_rows.append(
            {
                "series_id": row["series_id"],
                "file_path": row["file_path"],
                "class_label": int(row["class_label_int"]),
                "class_name": str(row["class_label"]),
                "class_description": bundle.class_descriptions.get(str(row["class_label"]), ""),
                "well_name": row["well_name"],
                "source_type": row["source_type"],
                "n_rows_original": int(len(frame)),
            }
        )

    arrays = {
        "X_seq": np.stack(sequence_parts, axis=0).astype(np.float32, copy=False),
        "X_tab": np.stack(tabular_parts, axis=0).astype(np.float32, copy=False),
        "X_missing": np.stack(missing_parts, axis=0).astype(np.float32, copy=False),
        "X_frozen": np.stack(frozen_parts, axis=0).astype(np.float32, copy=False),
        "y": np.asarray(labels, dtype=np.int64),
        "y_step_class": np.stack(step_class_parts, axis=0).astype(np.int64, copy=False),
        "y_step_state": np.stack(step_state_parts, axis=0).astype(np.int64, copy=False),
        "source_id": np.asarray(source_parts, dtype=np.int64),
    }
    metadata_df = pd.DataFrame(metadata_rows)
    return arrays, metadata_df


def prepare_classification_artifacts(
    *,
    dataset_root: str | Path,
    run_name: str = "classificacao_v10_multitarefa",
    train_frac: float = 0.70,
    validation_frac: float = 0.15,
    random_state: int = 42,
    sequence_length: int = 180,
) -> PreparedClassificationArtifacts:
    require_classification_stack()
    dataset_root = Path(dataset_root)
    run_dir = ARTIFACTS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = discover_series_manifest(dataset_root)
    split_manifest = stratified_split_manifest(
        manifest,
        train_frac=train_frac,
        validation_frac=validation_frac,
        random_state=random_state,
    )
    train_manifest = split_manifest.loc[split_manifest["split"] == "train"].reset_index(drop=True)
    bundle = fit_classification_bundle(
        train_manifest=train_manifest,
        split_manifest=split_manifest,
        dataset_root=dataset_root,
        sequence_length=sequence_length,
    )

    bundle_path = run_dir / "bundle_v10.json"
    manifest_path = run_dir / "split_manifest_v10.csv"
    attribute_catalog_path = run_dir / "catalogo_atributos.csv"
    event_catalog_path = run_dir / "catalogo_eventos.csv"

    save_bundle(bundle, bundle_path)
    split_manifest.to_csv(manifest_path, index=False)
    load_attribute_catalog(dataset_root).to_csv(attribute_catalog_path, index=False)
    load_event_catalog(dataset_root).to_csv(event_catalog_path, index=False)

    split_npz_paths: dict[str, str] = {}
    split_metadata_paths: dict[str, str] = {}
    for split_name in ["train", "validation", "test"]:
        split_df = split_manifest.loc[split_manifest["split"] == split_name].reset_index(drop=True)
        arrays, metadata_df = transform_manifest_to_arrays(split_df, bundle)
        npz_path = run_dir / f"{split_name}_arrays.npz"
        metadata_path = run_dir / f"{split_name}_metadata.csv"
        np.savez_compressed(npz_path, **arrays)
        metadata_df.to_csv(metadata_path, index=False)
        split_npz_paths[split_name] = str(npz_path)
        split_metadata_paths[split_name] = str(metadata_path)

    return PreparedClassificationArtifacts(
        run_dir=str(run_dir),
        bundle_path=str(bundle_path),
        manifest_path=str(manifest_path),
        attribute_catalog_path=str(attribute_catalog_path),
        event_catalog_path=str(event_catalog_path),
        split_npz_paths=split_npz_paths,
        split_metadata_paths=split_metadata_paths,
    )


def load_split_arrays(npz_path: str | Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as payload:
        return {key: payload[key] for key in payload.files}


if torch is not None:
    class SourceAwareMultitaskTemporalModel(nn.Module):
        def __init__(
            self,
            input_size: int,
            tabular_size: int,
            num_classes: int,
            num_step_classes: int,
            num_state_classes: int,
            source_vocab_size: int,
            *,
            hidden_size: int = 128,
            num_layers: int = 2,
            source_embedding_dim: int = 8,
            tabular_hidden_size: int = 128,
            dropout: float = 0.25,
        ) -> None:
            super().__init__()
            self.hidden_size = int(hidden_size)
            self.source_embedding_dim = int(source_embedding_dim)
            augmented_input_size = input_size * 3

            self.source_embedding = nn.Embedding(source_vocab_size, self.source_embedding_dim)
            self.input_norm = nn.LayerNorm(augmented_input_size)
            self.input_projection = nn.Linear(augmented_input_size, self.hidden_size)
            self.local_conv = nn.Sequential(
                nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=5, padding=2),
                nn.GELU(),
            )
            self.context_encoder = nn.LSTM(
                input_size=self.hidden_size + self.source_embedding_dim,
                hidden_size=self.hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=True,
            )
            context_dim = self.hidden_size * 2
            self.context_norm = nn.LayerNorm(context_dim)
            self.step_class_head = nn.Sequential(
                nn.Linear(context_dim, context_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(context_dim, num_step_classes),
            )
            self.step_state_head = nn.Sequential(
                nn.Linear(context_dim, context_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(context_dim, num_state_classes),
            )
            self.attention_score = nn.Linear(context_dim, 1)
            self.tabular_branch = nn.Sequential(
                nn.LayerNorm(tabular_size),
                nn.Linear(tabular_size, tabular_hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(tabular_hidden_size, tabular_hidden_size),
                nn.GELU(),
            )
            fusion_dim = context_dim * 3 + tabular_hidden_size + self.source_embedding_dim
            self.classifier = nn.Sequential(
                nn.LayerNorm(fusion_dim),
                nn.Linear(fusion_dim, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )

        def forward(
            self,
            x_seq: torch.Tensor,
            x_tab: torch.Tensor,
            x_missing: torch.Tensor,
            x_frozen: torch.Tensor,
            source_id: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            augmented = torch.cat([x_seq, x_missing, x_frozen], dim=-1)
            projected = self.input_projection(self.input_norm(augmented))
            conv_features = self.local_conv(projected.transpose(1, 2)).transpose(1, 2)
            local_features = projected + conv_features

            source_embedding = self.source_embedding(source_id)
            source_context = source_embedding.unsqueeze(1).expand(-1, local_features.size(1), -1)
            encoder_input = torch.cat([local_features, source_context], dim=-1)
            sequence_output, (hidden_state, _) = self.context_encoder(encoder_input)
            sequence_output = self.context_norm(sequence_output)

            step_class_logits = self.step_class_head(sequence_output)
            step_state_logits = self.step_state_head(sequence_output)

            forward_last = hidden_state[-2]
            backward_last = hidden_state[-1]
            last_hidden = torch.cat([forward_last, backward_last], dim=1)
            mean_pool = sequence_output.mean(dim=1)
            attention_logits = self.attention_score(sequence_output).squeeze(-1)
            attention_weights = torch.softmax(attention_logits, dim=1).unsqueeze(-1)
            attention_pool = torch.sum(sequence_output * attention_weights, dim=1)

            tabular_features = self.tabular_branch(x_tab)
            fused = torch.cat(
                [last_hidden, mean_pool, attention_pool, tabular_features, source_embedding],
                dim=1,
            )
            instance_logits = self.classifier(fused)
            return {
                "instance_logits": instance_logits,
                "step_class_logits": step_class_logits,
                "step_state_logits": step_state_logits,
            }
else:
    class SourceAwareMultitaskTemporalModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


def _default_device(device: str | None = None) -> torch.device:
    require_torch()
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _multitask_loader(
    X_seq: np.ndarray,
    X_tab: np.ndarray,
    X_missing: np.ndarray,
    X_frozen: np.ndarray,
    y: np.ndarray,
    y_step_class: np.ndarray,
    y_step_state: np.ndarray,
    source_id: np.ndarray,
    *,
    batch_size: int = 64,
    shuffle: bool = False,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    require_torch()
    dataset = TensorDataset(
        torch.tensor(X_seq, dtype=torch.float32),
        torch.tensor(X_tab, dtype=torch.float32),
        torch.tensor(X_missing, dtype=torch.float32),
        torch.tensor(X_frozen, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
        torch.tensor(y_step_class, dtype=torch.long),
        torch.tensor(y_step_state, dtype=torch.long),
        torch.tensor(source_id, dtype=torch.long),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        pin_memory=torch.cuda.is_available(),
    )


def _build_weighted_sampler(y: np.ndarray, class_labels: list[int]) -> WeightedRandomSampler:
    require_torch()
    class_weight_values = _compute_balanced_class_weights(class_labels, y)
    class_weight_map = {
        int(label): float(weight)
        for label, weight in zip(class_labels, class_weight_values.tolist(), strict=False)
    }
    sample_weights = np.asarray([class_weight_map[int(label)] for label in y], dtype=np.float64)
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def _predict_multitask_logits(
    model: nn.Module,
    X_seq: np.ndarray,
    X_tab: np.ndarray,
    X_missing: np.ndarray,
    X_frozen: np.ndarray,
    source_id: np.ndarray,
    *,
    batch_size: int = 128,
    device: str | None = None,
) -> np.ndarray:
    require_torch()
    device_obj = _default_device(device)
    loader = _multitask_loader(
        X_seq,
        X_tab,
        X_missing,
        X_frozen,
        np.zeros(len(X_seq), dtype=np.int64),
        np.full((len(X_seq), X_seq.shape[1]), IGNORE_INDEX, dtype=np.int64),
        np.full((len(X_seq), X_seq.shape[1]), IGNORE_INDEX, dtype=np.int64),
        source_id,
        batch_size=batch_size,
        shuffle=False,
    )
    logits_parts = []
    model.eval()
    with torch.no_grad():
        for batch_seq, batch_tab, batch_missing, batch_frozen, _, _, _, batch_source in loader:
            batch_output = model(
                batch_seq.to(device_obj, non_blocking=True),
                batch_tab.to(device_obj, non_blocking=True),
                batch_missing.to(device_obj, non_blocking=True),
                batch_frozen.to(device_obj, non_blocking=True),
                batch_source.to(device_obj, non_blocking=True),
            )
            logits_parts.append(batch_output["instance_logits"].detach().cpu().numpy())
    return np.concatenate(logits_parts, axis=0)


def predict_multitask_model_classes(
    model: nn.Module,
    X_seq: np.ndarray,
    X_tab: np.ndarray,
    X_missing: np.ndarray,
    X_frozen: np.ndarray,
    source_id: np.ndarray,
    *,
    batch_size: int = 128,
    device: str | None = None,
) -> np.ndarray:
    logits = _predict_multitask_logits(
        model,
        X_seq,
        X_tab,
        X_missing,
        X_frozen,
        source_id,
        batch_size=batch_size,
        device=device,
    )
    return logits.argmax(axis=1).astype(np.int64, copy=False)


def train_multitask_temporal_model(
    X_train_seq: np.ndarray,
    X_train_tab: np.ndarray,
    X_train_missing: np.ndarray,
    X_train_frozen: np.ndarray,
    y_train: np.ndarray,
    y_train_step_class: np.ndarray,
    y_train_step_state: np.ndarray,
    train_source_id: np.ndarray,
    X_val_seq: np.ndarray,
    X_val_tab: np.ndarray,
    X_val_missing: np.ndarray,
    X_val_frozen: np.ndarray,
    y_val: np.ndarray,
    y_val_step_class: np.ndarray,
    y_val_step_state: np.ndarray,
    val_source_id: np.ndarray,
    *,
    output_dir: str | Path,
    class_labels: list[int],
    observation_class_codes: list[int],
    observation_state_codes: list[int],
    source_vocab_size: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    source_embedding_dim: int = 8,
    tabular_hidden_size: int = 128,
    dropout: float = 0.25,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    epochs: int = 55,
    patience: int = 10,
    lambda_step_class: float = 0.35,
    lambda_step_state: float = 0.15,
    random_state: int = 42,
    device: str | None = None,
) -> MultiTaskTrainingSummary:
    require_classification_stack()
    require_torch()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(random_state)
    device_obj = _default_device(device)

    model = SourceAwareMultitaskTemporalModel(
        input_size=int(X_train_seq.shape[-1]),
        tabular_size=int(X_train_tab.shape[-1]),
        num_classes=len(class_labels),
        num_step_classes=len(observation_class_codes),
        num_state_classes=len(observation_state_codes),
        source_vocab_size=source_vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        source_embedding_dim=source_embedding_dim,
        tabular_hidden_size=tabular_hidden_size,
        dropout=dropout,
    ).to(device_obj)

    instance_weight_values = _compute_balanced_class_weights(class_labels, y_train)
    step_class_weight_values = _compute_balanced_index_weights(
        len(observation_class_codes),
        y_train_step_class.reshape(-1),
    )
    step_state_weight_values = _compute_balanced_index_weights(
        len(observation_state_codes),
        y_train_step_state.reshape(-1),
    )

    instance_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(instance_weight_values, dtype=torch.float32, device=device_obj)
    )
    step_class_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(step_class_weight_values, dtype=torch.float32, device=device_obj),
        ignore_index=IGNORE_INDEX,
    )
    step_state_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(step_state_weight_values, dtype=torch.float32, device=device_obj),
        ignore_index=IGNORE_INDEX,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(patience // 2, 1),
    )

    train_sampler = _build_weighted_sampler(y_train, class_labels)
    train_loader = _multitask_loader(
        X_train_seq,
        X_train_tab,
        X_train_missing,
        X_train_frozen,
        y_train,
        y_train_step_class,
        y_train_step_state,
        train_source_id,
        batch_size=batch_size,
        sampler=train_sampler,
    )

    checkpoint_path = output_dir / "multitask_temporal_best.pt"
    config_path = output_dir / "multitask_temporal_config.json"
    history_path = output_dir / "multitask_temporal_history.csv"

    history_rows = []
    best_signature: tuple[float, float, float] | None = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        epoch_instance_losses = []
        epoch_step_class_losses = []
        epoch_step_state_losses = []

        for (
            batch_seq,
            batch_tab,
            batch_missing,
            batch_frozen,
            batch_y,
            batch_step_class,
            batch_step_state,
            batch_source,
        ) in train_loader:
            batch_seq = batch_seq.to(device_obj, non_blocking=True)
            batch_tab = batch_tab.to(device_obj, non_blocking=True)
            batch_missing = batch_missing.to(device_obj, non_blocking=True)
            batch_frozen = batch_frozen.to(device_obj, non_blocking=True)
            batch_y = batch_y.to(device_obj, non_blocking=True)
            batch_step_class = batch_step_class.to(device_obj, non_blocking=True)
            batch_step_state = batch_step_state.to(device_obj, non_blocking=True)
            batch_source = batch_source.to(device_obj, non_blocking=True)

            outputs = model(batch_seq, batch_tab, batch_missing, batch_frozen, batch_source)
            instance_loss = instance_criterion(outputs["instance_logits"], batch_y)
            step_class_loss = step_class_criterion(
                outputs["step_class_logits"].reshape(-1, len(observation_class_codes)),
                batch_step_class.reshape(-1),
            )
            step_state_loss = step_state_criterion(
                outputs["step_state_logits"].reshape(-1, len(observation_state_codes)),
                batch_step_state.reshape(-1),
            )
            loss = instance_loss + lambda_step_class * step_class_loss + lambda_step_state * step_state_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu().item()))
            epoch_instance_losses.append(float(instance_loss.detach().cpu().item()))
            epoch_step_class_losses.append(float(step_class_loss.detach().cpu().item()))
            epoch_step_state_losses.append(float(step_state_loss.detach().cpu().item()))

        train_pred = predict_multitask_model_classes(
            model,
            X_train_seq,
            X_train_tab,
            X_train_missing,
            X_train_frozen,
            train_source_id,
            batch_size=batch_size,
            device=str(device_obj),
        )
        val_pred = predict_multitask_model_classes(
            model,
            X_val_seq,
            X_val_tab,
            X_val_missing,
            X_val_frozen,
            val_source_id,
            batch_size=batch_size,
            device=str(device_obj),
        )

        train_eval = evaluate_predictions(y_train, train_pred, class_labels=class_labels)
        val_eval = evaluate_predictions(y_val, val_pred, class_labels=class_labels)

        current_row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(np.mean(epoch_losses)),
            "train_instance_loss": float(np.mean(epoch_instance_losses)),
            "train_step_class_loss": float(np.mean(epoch_step_class_losses)),
            "train_step_state_loss": float(np.mean(epoch_step_state_losses)),
            "train_accuracy": train_eval["accuracy"],
            "train_macro_f1": train_eval["macro_f1"],
            "train_balanced_accuracy": train_eval["balanced_accuracy"],
            "val_accuracy": val_eval["accuracy"],
            "val_macro_f1": val_eval["macro_f1"],
            "val_balanced_accuracy": val_eval["balanced_accuracy"],
        }
        history_rows.append(current_row)

        scheduler.step(val_eval["macro_f1"])
        current_signature = (
            float(val_eval["macro_f1"]),
            float(val_eval["balanced_accuracy"]),
            float(val_eval["accuracy"]),
        )
        if best_signature is None or current_signature > best_signature:
            best_signature = current_signature
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            _write_json(
                {
                    "input_size": int(X_train_seq.shape[-1]),
                    "tabular_size": int(X_train_tab.shape[-1]),
                    "num_classes": len(class_labels),
                    "num_step_classes": len(observation_class_codes),
                    "num_state_classes": len(observation_state_codes),
                    "source_vocab_size": source_vocab_size,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "source_embedding_dim": source_embedding_dim,
                    "tabular_hidden_size": tabular_hidden_size,
                    "dropout": dropout,
                    "checkpoint_path": str(checkpoint_path),
                },
                config_path,
            )
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    best_history_row = pd.DataFrame(history_rows).loc[lambda df: df["epoch"] == best_epoch].iloc[0]
    return MultiTaskTrainingSummary(
        model_name="source_aware_multitask_temporal_model",
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path),
        history_path=str(history_path),
        best_epoch=int(best_epoch),
        best_val_macro_f1=float(best_history_row["val_macro_f1"]),
        best_val_accuracy=float(best_history_row["val_accuracy"]),
        best_val_balanced_accuracy=float(best_history_row["val_balanced_accuracy"]),
    )


def load_multitask_temporal_model(
    config_path: str | Path,
    *,
    device: str | None = None,
) -> nn.Module:
    require_torch()
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    device_obj = _default_device(device)
    model = SourceAwareMultitaskTemporalModel(
        input_size=int(payload["input_size"]),
        tabular_size=int(payload["tabular_size"]),
        num_classes=int(payload["num_classes"]),
        num_step_classes=int(payload["num_step_classes"]),
        num_state_classes=int(payload["num_state_classes"]),
        source_vocab_size=int(payload["source_vocab_size"]),
        hidden_size=int(payload["hidden_size"]),
        num_layers=int(payload["num_layers"]),
        source_embedding_dim=int(payload["source_embedding_dim"]),
        tabular_hidden_size=int(payload["tabular_hidden_size"]),
        dropout=float(payload["dropout"]),
    )
    model.load_state_dict(torch.load(payload["checkpoint_path"], map_location=device_obj))
    model.to(device_obj)
    model.eval()
    return model
