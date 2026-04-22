from __future__ import annotations

from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
import time
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
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_class_weight
except ImportError:
    RandomForestClassifier = None
    ConfusionMatrixDisplay = None
    accuracy_score = None
    balanced_accuracy_score = None
    classification_report = None
    confusion_matrix = None
    f1_score = None
    train_test_split = None
    StandardScaler = None
    compute_class_weight = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
except ImportError:
    torch = None
    nn = None
    F = None
    DataLoader = None
    TensorDataset = None
    WeightedRandomSampler = None

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

from versao3.pipeline_v3 import discover_all_dataset_files, set_seed  # noqa: E402
from versao4.pipeline_v4 import (  # noqa: E402
    AUX_ANALOG_COLUMNS,
    BASE_TARGET_COLUMNS,
    STATE_COLUMNS,
    clean_base_frame,
    require_tabular_stack,
    require_torch,
)


# Nesta versao, mantemos a classificacao por serie do 3W, mas trocamos
# a hipotese arquitetural principal.
# Em vez de uma LSTM monolitica sobre toda a sequencia, propomos um modelo
# hibrido e hierarquico:
# 1) a serie e lida em janelas temporais;
# 2) sinais continuos e sinais de estado seguem por ramos separados;
# 3) uma segunda LSTM agrega a sequencia de janelas;
# 4) o embedding sequencial final e fundido com X_tab.
CANDIDATE_FEATURE_COLUMNS = BASE_TARGET_COLUMNS + STATE_COLUMNS + AUX_ANALOG_COLUMNS
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "reports_v9"
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


def _safe_numeric_matrix(values: np.ndarray, clip_abs: float = 1e12) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    matrix = np.clip(matrix, -clip_abs, clip_abs)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix


def require_classification_stack() -> None:
    require_tabular_stack()
    if (
        RandomForestClassifier is None
        or ConfusionMatrixDisplay is None
        or accuracy_score is None
        or balanced_accuracy_score is None
        or classification_report is None
        or confusion_matrix is None
        or f1_score is None
        or train_test_split is None
        or StandardScaler is None
        or compute_class_weight is None
    ):
        raise ImportError(
            "Dependencias de classificacao nao estao instaladas. "
            "Instale scikit-learn para executar a versao9."
        )


def require_plotting_stack() -> None:
    if plt is None:
        raise ImportError(
            "matplotlib nao esta instalado neste ambiente. "
            "Instale matplotlib para plotar as matrizes de confusao."
        )


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


@dataclass
class PreparedClassificationArtifacts:
    run_dir: str
    bundle_path: str
    manifest_path: str
    attribute_catalog_path: str
    event_catalog_path: str
    feature_selection_report_path: str
    split_npz_paths: dict[str, str]
    split_metadata_paths: dict[str, str]


@dataclass
class HybridTrainingSummary:
    model_name: str
    checkpoint_path: str
    config_path: str
    history_path: str
    best_epoch: int
    best_val_macro_f1: float
    best_val_accuracy: float
    best_val_balanced_accuracy: float


def _json_ready(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return str(value)


def _write_json(payload: dict[str, Any], output_path: str | Path) -> None:
    Path(output_path).write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _dataset_ini_path(dataset_root: str | Path) -> Path:
    return Path(dataset_root) / "dataset.ini"


def load_dataset_config(dataset_root: str | Path) -> ConfigParser:
    parser = ConfigParser()
    parser.optionxform = str
    parser.read(_dataset_ini_path(dataset_root), encoding="utf-8")
    return parser


def load_attribute_catalog(dataset_root: str | Path) -> pd.DataFrame:
    require_tabular_stack()
    parser = load_dataset_config(dataset_root)
    rows = []
    for attribute_name, description in parser["PARQUET_FILE_PROPERTIES"].items():
        role = "metadado"
        if attribute_name in BASE_TARGET_COLUMNS:
            role = "variavel_principal"
        elif attribute_name in STATE_COLUMNS:
            role = "estado_discreto"
        elif attribute_name in AUX_ANALOG_COLUMNS:
            role = "variavel_auxiliar_analogica"
        elif attribute_name == "class":
            role = "rotulo"
        elif attribute_name == "state":
            role = "estado_operacional"
        rows.append(
            {
                "atributo": attribute_name,
                "papel_no_pipeline": role,
                "descricao_oficial": description,
            }
        )
    return pd.DataFrame(rows)


def load_event_catalog(dataset_root: str | Path) -> pd.DataFrame:
    require_tabular_stack()
    parser = load_dataset_config(dataset_root)
    rows = []
    for event_name in [name.strip() for name in parser["EVENTS"]["NAMES"].split(",")]:
        section = parser[event_name]
        rows.append(
            {
                "class_label": int(section["LABEL"]),
                "event_name": event_name,
                "description": section["DESCRIPTION"],
                "transient_event": bool(section.get("TRANSIENT", "False").lower() == "true"),
            }
        )
    return pd.DataFrame(rows).sort_values("class_label").reset_index(drop=True)


def save_bundle(bundle: ClassificationBundle, output_path: str | Path) -> None:
    _write_json(bundle.__dict__, output_path)


def load_bundle(bundle_path: str | Path) -> ClassificationBundle:
    payload = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    return ClassificationBundle(**payload)


def discover_series_manifest(dataset_root: str | Path) -> pd.DataFrame:
    require_tabular_stack()
    manifest = discover_all_dataset_files(Path(dataset_root))
    manifest = manifest.copy()
    manifest["class_label_int"] = manifest["class_label"].astype(int)
    return manifest


def stratified_split_manifest(
    manifest: pd.DataFrame,
    train_frac: float = 0.70,
    validation_frac: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    require_classification_stack()
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac precisa estar entre 0 e 1.")
    if not 0.0 <= validation_frac < 1.0:
        raise ValueError("validation_frac precisa estar entre 0 e 1.")
    if train_frac + validation_frac >= 1.0:
        raise ValueError("train_frac + validation_frac precisa ser menor que 1.")

    train_df, temp_df = train_test_split(
        manifest,
        train_size=train_frac,
        stratify=manifest["class_label_int"],
        random_state=random_state,
    )
    if len(temp_df) == 0:
        raise ValueError("Nao sobraram amostras para validacao e teste.")

    relative_validation_frac = validation_frac / (1.0 - train_frac)
    validation_df, test_df = train_test_split(
        temp_df,
        train_size=relative_validation_frac,
        stratify=temp_df["class_label_int"],
        random_state=random_state,
    )

    train_df = train_df.copy()
    validation_df = validation_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    validation_df["split"] = "validation"
    test_df["split"] = "test"

    return (
        pd.concat([train_df, validation_df, test_df], ignore_index=True)
        .sort_values(["split", "class_label_int", "well_name", "start_token"])
        .reset_index(drop=True)
    )


def _sample_rows(frame: pd.DataFrame, max_rows: int | None = None) -> pd.DataFrame:
    require_tabular_stack()
    if max_rows is None or len(frame) <= int(max_rows):
        return frame.reset_index(drop=True)
    idx = np.linspace(0, len(frame) - 1, num=int(max_rows), dtype=np.int64)
    idx = np.unique(idx)
    return frame.iloc[idx].reset_index(drop=True)


def build_feature_selection_report(
    train_manifest: pd.DataFrame,
    max_rows_per_series: int | None = 256,
    min_unique_values: int = 2,
    min_std: float = 1e-8,
) -> pd.DataFrame:
    require_tabular_stack()
    parts = []
    for file_path in train_manifest["file_path"]:
        frame = clean_base_frame(
            file_path,
            target_columns=BASE_TARGET_COLUMNS,
            candidate_auxiliary_columns=STATE_COLUMNS + AUX_ANALOG_COLUMNS,
        )
        parts.append(_sample_rows(frame[CANDIDATE_FEATURE_COLUMNS], max_rows=max_rows_per_series))
    reference_df = pd.concat(parts, ignore_index=True)

    rows = []
    for column in [col for col in CANDIDATE_FEATURE_COLUMNS if col in reference_df.columns]:
        series = pd.to_numeric(reference_df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        std_value = float(series.std(skipna=True)) if series.notna().any() else 0.0
        nunique = int(series.nunique(dropna=True))
        null_pct = float(series.isna().mean() * 100.0)
        selected = (nunique >= min_unique_values) and (std_value > min_std)
        column_type = "continuous"
        if column in STATE_COLUMNS:
            column_type = "state"
        elif column in BASE_TARGET_COLUMNS:
            column_type = "main_signal"
        elif column in AUX_ANALOG_COLUMNS:
            column_type = "aux_signal"
        rows.append(
            {
                "column": column,
                "column_type": column_type,
                "null_pct": null_pct,
                "nunique": nunique,
                "std": std_value,
                "selected_for_modeling": selected,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["selected_for_modeling", "column"], ascending=[False, True])
        .reset_index(drop=True)
    )


def _safe_scale(values: np.ndarray) -> list[float]:
    scale = np.asarray(values, dtype=np.float64)
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    return scale.tolist()


def resample_frame_to_fixed_length(
    frame: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
    state_columns: list[str] | None = None,
) -> np.ndarray:
    require_tabular_stack()
    state_columns = state_columns or []
    if sequence_length <= 0:
        raise ValueError("sequence_length precisa ser maior que zero.")

    n_rows = len(frame)
    if n_rows == 0:
        raise ValueError("A serie nao pode estar vazia.")

    if n_rows == 1:
        single_row = _safe_numeric_matrix(frame[feature_columns].to_numpy(dtype=np.float64))
        return np.repeat(single_row, repeats=sequence_length, axis=0)

    source_pos = np.linspace(0.0, 1.0, n_rows, dtype=np.float64)
    target_pos = np.linspace(0.0, 1.0, sequence_length, dtype=np.float64)

    columns_resampled = []
    for column in feature_columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        if column in state_columns:
            interpolated_idx = np.interp(target_pos, source_pos, np.arange(n_rows, dtype=np.float64))
            nearest_idx = np.rint(interpolated_idx).astype(np.int64)
            nearest_idx = np.clip(nearest_idx, 0, n_rows - 1)
            resampled = values[nearest_idx]
        else:
            resampled = np.interp(target_pos, source_pos, values)
        columns_resampled.append(resampled)

    matrix = np.stack(columns_resampled, axis=1)
    return _safe_numeric_matrix(matrix)


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


def fit_classification_bundle(
    train_manifest: pd.DataFrame,
    split_manifest: pd.DataFrame,
    *,
    dataset_root: str | Path,
    feature_selection_report: pd.DataFrame,
    sequence_length: int = 120,
) -> ClassificationBundle:
    require_classification_stack()
    selected_columns = (
        feature_selection_report
        .loc[feature_selection_report["selected_for_modeling"], "column"]
        .tolist()
    )
    if not selected_columns:
        raise ValueError("Nenhuma coluna informativa foi selecionada para a classificacao.")

    state_columns = [column for column in selected_columns if column in STATE_COLUMNS]
    continuous_columns = [column for column in selected_columns if column not in state_columns]

    scaler = StandardScaler()
    for file_path in train_manifest["file_path"]:
        frame = clean_base_frame(
            file_path,
            target_columns=BASE_TARGET_COLUMNS,
            candidate_auxiliary_columns=STATE_COLUMNS + AUX_ANALOG_COLUMNS,
        )
        sequence = resample_frame_to_fixed_length(
            frame,
            feature_columns=selected_columns,
            sequence_length=sequence_length,
            state_columns=state_columns,
        )
        scaler.partial_fit(sequence)

    event_catalog = load_event_catalog(dataset_root)
    class_labels = event_catalog["class_label"].astype(int).tolist()
    class_names = [str(value) for value in class_labels]
    class_descriptions = {
        str(row["class_label"]): str(row["description"])
        for _, row in event_catalog.iterrows()
    }

    statistical_feature_names = build_statistical_feature_names(selected_columns)
    selected_files = {
        split_name: split_df["file_path"].tolist()
        for split_name, split_df in split_manifest.groupby("split", sort=False)
    }
    split_counts = split_manifest["split"].value_counts().sort_index().to_dict()

    return ClassificationBundle(
        selected_columns=selected_columns,
        continuous_columns=continuous_columns,
        state_columns=state_columns,
        sequence_length=int(sequence_length),
        scaler_mean=scaler.mean_.tolist(),
        scaler_scale=_safe_scale(scaler.scale_),
        class_labels=class_labels,
        class_names=class_names,
        class_descriptions=class_descriptions,
        statistical_feature_names=statistical_feature_names,
        split_counts=split_counts,
        selected_files=selected_files,
    )


def transform_manifest_to_arrays(
    manifest: pd.DataFrame,
    bundle: ClassificationBundle,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    require_tabular_stack()

    sequence_parts = []
    tabular_parts = []
    labels = []
    metadata_rows = []
    mean_arr = np.asarray(bundle.scaler_mean, dtype=np.float64)
    scale_arr = np.asarray(bundle.scaler_scale, dtype=np.float64)

    for _, row in manifest.iterrows():
        frame = clean_base_frame(
            row["file_path"],
            target_columns=BASE_TARGET_COLUMNS,
            candidate_auxiliary_columns=STATE_COLUMNS + AUX_ANALOG_COLUMNS,
        )
        sequence = resample_frame_to_fixed_length(
            frame,
            feature_columns=bundle.selected_columns,
            sequence_length=bundle.sequence_length,
            state_columns=bundle.state_columns,
        )
        sequence_scaled = _safe_numeric_matrix((sequence.astype(np.float64) - mean_arr) / scale_arr).astype(np.float32, copy=False)
        statistical_vector = compute_statistical_feature_vector(sequence_scaled, bundle.selected_columns)

        sequence_parts.append(sequence_scaled)
        tabular_parts.append(statistical_vector)
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
        "y": np.asarray(labels, dtype=np.int64),
    }
    metadata_df = pd.DataFrame(metadata_rows)
    return arrays, metadata_df


def prepare_classification_artifacts(
    *,
    dataset_root: str | Path,
    run_name: str = "classificacao_v6",
    train_frac: float = 0.70,
    validation_frac: float = 0.15,
    random_state: int = 42,
    sequence_length: int = 120,
    max_rows_per_series_for_selection: int | None = 256,
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

    feature_selection_report = build_feature_selection_report(
        train_manifest,
        max_rows_per_series=max_rows_per_series_for_selection,
    )
    bundle = fit_classification_bundle(
        train_manifest=train_manifest,
        split_manifest=split_manifest,
        dataset_root=dataset_root,
        feature_selection_report=feature_selection_report,
        sequence_length=sequence_length,
    )

    bundle_path = run_dir / "bundle_v9.json"
    manifest_path = run_dir / "split_manifest_v6.csv"
    attribute_catalog_path = run_dir / "catalogo_atributos.csv"
    event_catalog_path = run_dir / "catalogo_eventos.csv"
    feature_selection_report_path = run_dir / "relatorio_selecao_de_features.csv"

    save_bundle(bundle, bundle_path)
    split_manifest.to_csv(manifest_path, index=False)
    load_attribute_catalog(dataset_root).to_csv(attribute_catalog_path, index=False)
    load_event_catalog(dataset_root).to_csv(event_catalog_path, index=False)
    feature_selection_report.to_csv(feature_selection_report_path, index=False)

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
        feature_selection_report_path=str(feature_selection_report_path),
        split_npz_paths=split_npz_paths,
        split_metadata_paths=split_metadata_paths,
    )


def load_split_arrays(npz_path: str | Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as payload:
        return {key: payload[key] for key in payload.files}


def build_metrics_table(metrics_by_model: dict[str, dict[str, Any]]) -> pd.DataFrame:
    require_tabular_stack()
    rows = []
    for model_name, metrics in metrics_by_model.items():
        rows.append(
            {
                "modelo": model_name,
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["macro_f1", "balanced_accuracy"], ascending=False).reset_index(drop=True)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[int] | None = None,
) -> dict[str, Any]:
    require_classification_stack()
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    labels = class_labels or sorted(np.unique(np.concatenate([y_true_arr, y_pred_arr])).tolist())
    report_dict = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index": "label"})
    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr, labels=labels),
        "classification_report_df": report_df,
        "y_true": y_true_arr,
        "y_pred": y_pred_arr,
        "labels": labels,
    }


def export_evaluation_artifacts(
    evaluation: dict[str, Any],
    output_dir: str | Path,
    prefix: str,
) -> dict[str, str]:
    require_tabular_stack()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_json_path = output_path / f"{prefix}_metrics.json"
    report_csv_path = output_path / f"{prefix}_classification_report.csv"
    predictions_npz_path = output_path / f"{prefix}_predictions.npz"

    _write_json(
        {
            "accuracy": evaluation["accuracy"],
            "macro_f1": evaluation["macro_f1"],
            "balanced_accuracy": evaluation["balanced_accuracy"],
            "labels": evaluation["labels"],
            "confusion_matrix": evaluation["confusion_matrix"],
        },
        metrics_json_path,
    )
    evaluation["classification_report_df"].to_csv(report_csv_path, index=False)
    np.savez_compressed(
        predictions_npz_path,
        y_true=evaluation["y_true"],
        y_pred=evaluation["y_pred"],
    )
    return {
        "metrics_json": str(metrics_json_path),
        "report_csv": str(report_csv_path),
        "predictions_npz": str(predictions_npz_path),
    }


def plot_confusion_matrix_for_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[int],
    *,
    normalize: str | None = None,
    title: str | None = None,
    ax: Any | None = None,
) -> Any:
    require_classification_stack()
    require_plotting_stack()
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 8))
    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=class_labels,
        normalize=normalize,
        cmap="Blues",
        ax=ax,
        colorbar=False,
        values_format=".2f" if normalize else "d",
    )
    ax.set_title(title or "Matriz de confusao")
    ax.set_xlabel("Classe predita")
    ax.set_ylabel("Classe real")
    return display


if torch is not None:
    class ClassBalancedFocalLoss(nn.Module):
        def __init__(self, class_weights: torch.Tensor, gamma: float = 1.5) -> None:
            super().__init__()
            self.register_buffer("class_weights", class_weights)
            self.gamma = float(gamma)

        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            log_probs = F.log_softmax(logits, dim=1)
            probs = log_probs.exp()
            target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            target_weights = self.class_weights.gather(0, targets)
            focal_factor = (1.0 - target_probs).pow(self.gamma)
            losses = -target_weights * focal_factor * target_log_probs
            return losses.mean()


    class HybridHierarchicalLSTMClassifier(nn.Module):
        def __init__(
            self,
            input_size: int,
            tabular_size: int,
            num_classes: int,
            continuous_indices: list[int],
            state_indices: list[int],
            *,
            window_size: int = 20,
            continuous_hidden_size: int = 96,
            state_hidden_size: int = 64,
            context_hidden_size: int = 160,
            context_num_layers: int = 2,
            tabular_hidden_size: int = 128,
            dropout: float = 0.25,
            bidirectional: bool = True,
        ) -> None:
            super().__init__()
            self.window_size = int(window_size)
            self.bidirectional = bool(bidirectional)
            self.output_multiplier = 2 if self.bidirectional else 1

            self.register_buffer(
                "continuous_indices_tensor",
                torch.tensor(continuous_indices, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "state_indices_tensor",
                torch.tensor(state_indices, dtype=torch.long),
                persistent=False,
            )

            self.input_norm = nn.LayerNorm(input_size)

            continuous_dim = len(continuous_indices)
            state_dim = len(state_indices)
            if continuous_dim == 0 and state_dim == 0:
                raise ValueError("E necessario informar pelo menos um grupo de features.")

            self.continuous_encoder = None
            self.continuous_input_norm = None
            self.continuous_output_norm = None
            self.continuous_attention = None
            continuous_summary_dim = 0
            if continuous_dim > 0:
                continuous_output_dim = continuous_hidden_size * self.output_multiplier
                self.continuous_input_norm = nn.LayerNorm(continuous_dim)
                self.continuous_encoder = nn.LSTM(
                    input_size=continuous_dim,
                    hidden_size=continuous_hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=self.bidirectional,
                )
                self.continuous_output_norm = nn.LayerNorm(continuous_output_dim)
                self.continuous_attention = nn.Linear(continuous_output_dim, 1)
                continuous_summary_dim = continuous_output_dim * 3

            self.state_encoder = None
            self.state_input_norm = None
            self.state_output_norm = None
            self.state_attention = None
            state_summary_dim = 0
            if state_dim > 0:
                state_output_dim = state_hidden_size * self.output_multiplier
                self.state_input_norm = nn.LayerNorm(state_dim)
                self.state_encoder = nn.LSTM(
                    input_size=state_dim,
                    hidden_size=state_hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=self.bidirectional,
                )
                self.state_output_norm = nn.LayerNorm(state_output_dim)
                self.state_attention = nn.Linear(state_output_dim, 1)
                state_summary_dim = state_output_dim * 3

            window_feature_dim = continuous_summary_dim + state_summary_dim
            self.window_projection = nn.Sequential(
                nn.LayerNorm(window_feature_dim),
                nn.Linear(window_feature_dim, context_hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            context_output_dim = context_hidden_size * self.output_multiplier
            self.context_encoder = nn.LSTM(
                input_size=context_hidden_size,
                hidden_size=context_hidden_size,
                num_layers=context_num_layers,
                dropout=dropout if context_num_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=self.bidirectional,
            )
            self.context_output_norm = nn.LayerNorm(context_output_dim)
            self.context_attention = nn.Linear(context_output_dim, 1)

            self.tabular_branch = nn.Sequential(
                nn.LayerNorm(tabular_size),
                nn.Linear(tabular_size, tabular_hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(tabular_hidden_size, tabular_hidden_size),
                nn.GELU(),
            )

            fusion_input_dim = context_output_dim * 3 + tabular_hidden_size
            self.classifier = nn.Sequential(
                nn.LayerNorm(fusion_input_dim),
                nn.Linear(fusion_input_dim, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )

        def _windowify(self, sequence: torch.Tensor) -> torch.Tensor:
            batch_size, n_steps, n_features = sequence.shape
            remainder = n_steps % self.window_size
            if remainder != 0:
                pad_steps = self.window_size - remainder
                pad_chunk = sequence[:, -1:, :].expand(batch_size, pad_steps, n_features)
                sequence = torch.cat([sequence, pad_chunk], dim=1)
            n_windows = sequence.shape[1] // self.window_size
            return sequence.reshape(batch_size, n_windows, self.window_size, n_features)

        def _pool_lstm_outputs(
            self,
            sequence_output: torch.Tensor,
            hidden_state: torch.Tensor,
            attention_layer: nn.Linear,
        ) -> torch.Tensor:
            if self.bidirectional:
                forward_last = hidden_state[-2]
                backward_last = hidden_state[-1]
                last_hidden = torch.cat([forward_last, backward_last], dim=1)
            else:
                last_hidden = hidden_state[-1]

            mean_pool = sequence_output.mean(dim=1)
            attention_logits = attention_layer(sequence_output).squeeze(-1)
            attention_weights = torch.softmax(attention_logits, dim=1).unsqueeze(-1)
            attention_pool = torch.sum(sequence_output * attention_weights, dim=1)
            return torch.cat([last_hidden, mean_pool, attention_pool], dim=1)

        def _encode_branch(
            self,
            branch_sequence: torch.Tensor,
            *,
            encoder: nn.Module,
            input_norm: nn.Module,
            output_norm: nn.Module,
            attention_layer: nn.Module,
        ) -> torch.Tensor:
            windows = self._windowify(branch_sequence)
            batch_size, n_windows, window_size, n_features = windows.shape
            flat_windows = windows.reshape(batch_size * n_windows, window_size, n_features)
            flat_windows = input_norm(flat_windows)
            sequence_output, (hidden_state, _) = encoder(flat_windows)
            sequence_output = output_norm(sequence_output)
            pooled = self._pool_lstm_outputs(sequence_output, hidden_state, attention_layer)
            return pooled.reshape(batch_size, n_windows, -1)

        def forward(self, x_seq: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
            normalized_sequence = self.input_norm(x_seq)
            branch_windows = []

            if self.continuous_encoder is not None and self.continuous_indices_tensor.numel() > 0:
                continuous_sequence = normalized_sequence.index_select(2, self.continuous_indices_tensor)
                branch_windows.append(
                    self._encode_branch(
                        continuous_sequence,
                        encoder=self.continuous_encoder,
                        input_norm=self.continuous_input_norm,
                        output_norm=self.continuous_output_norm,
                        attention_layer=self.continuous_attention,
                    )
                )

            if self.state_encoder is not None and self.state_indices_tensor.numel() > 0:
                state_sequence = normalized_sequence.index_select(2, self.state_indices_tensor)
                branch_windows.append(
                    self._encode_branch(
                        state_sequence,
                        encoder=self.state_encoder,
                        input_norm=self.state_input_norm,
                        output_norm=self.state_output_norm,
                        attention_layer=self.state_attention,
                    )
                )

            if not branch_windows:
                raise RuntimeError("Nenhum ramo da arquitetura recebeu features validas.")

            window_features = torch.cat(branch_windows, dim=-1)
            window_embeddings = self.window_projection(window_features)

            context_output, (context_hidden, _) = self.context_encoder(window_embeddings)
            context_output = self.context_output_norm(context_output)
            context_features = self._pool_lstm_outputs(
                context_output,
                context_hidden,
                self.context_attention,
            )

            tabular_features = self.tabular_branch(x_tab)
            fused_features = torch.cat([context_features, tabular_features], dim=1)
            return self.classifier(fused_features)
else:
    class ClassBalancedFocalLoss:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


    class HybridHierarchicalLSTMClassifier:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


def _default_device(device: str | None = None) -> torch.device:
    require_torch()
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def _multimodal_loader(
    X_seq: np.ndarray,
    X_tab: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int = 64,
    shuffle: bool = False,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    require_torch()
    dataset = TensorDataset(
        torch.tensor(X_seq, dtype=torch.float32),
        torch.tensor(X_tab, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
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


def _predict_hybrid_logits(
    model: nn.Module,
    X_seq: np.ndarray,
    X_tab: np.ndarray,
    *,
    batch_size: int = 128,
    device: str | None = None,
) -> np.ndarray:
    require_torch()
    device_obj = _default_device(device)
    loader = _multimodal_loader(
        X_seq,
        X_tab,
        np.zeros(len(X_seq), dtype=np.int64),
        batch_size=batch_size,
        shuffle=False,
    )
    logits_parts = []
    model.eval()
    with torch.no_grad():
        for batch_seq, batch_tab, _ in loader:
            batch_logits = model(
                batch_seq.to(device_obj, non_blocking=True),
                batch_tab.to(device_obj, non_blocking=True),
            )
            logits_parts.append(batch_logits.detach().cpu().numpy())
    return np.concatenate(logits_parts, axis=0)


def predict_hybrid_lstm_classes(
    model: nn.Module,
    X_seq: np.ndarray,
    X_tab: np.ndarray,
    *,
    batch_size: int = 128,
    device: str | None = None,
) -> np.ndarray:
    logits = _predict_hybrid_logits(
        model,
        X_seq,
        X_tab,
        batch_size=batch_size,
        device=device,
    )
    return logits.argmax(axis=1).astype(np.int64, copy=False)


def train_hybrid_lstm_classifier(
    X_train_seq: np.ndarray,
    X_train_tab: np.ndarray,
    y_train: np.ndarray,
    X_val_seq: np.ndarray,
    X_val_tab: np.ndarray,
    y_val: np.ndarray,
    *,
    output_dir: str | Path,
    class_labels: list[int],
    continuous_indices: list[int],
    state_indices: list[int],
    window_size: int = 20,
    continuous_hidden_size: int = 96,
    state_hidden_size: int = 64,
    context_hidden_size: int = 160,
    context_num_layers: int = 2,
    tabular_hidden_size: int = 128,
    dropout: float = 0.25,
    bidirectional: bool = True,
    learning_rate: float = 6e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    epochs: int = 50,
    patience: int = 10,
    focal_gamma: float = 1.5,
    random_state: int = 42,
    device: str | None = None,
) -> HybridTrainingSummary:
    require_classification_stack()
    require_torch()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(random_state)
    device_obj = _default_device(device)

    model = HybridHierarchicalLSTMClassifier(
        input_size=int(X_train_seq.shape[-1]),
        tabular_size=int(X_train_tab.shape[-1]),
        num_classes=len(class_labels),
        continuous_indices=continuous_indices,
        state_indices=state_indices,
        window_size=window_size,
        continuous_hidden_size=continuous_hidden_size,
        state_hidden_size=state_hidden_size,
        context_hidden_size=context_hidden_size,
        context_num_layers=context_num_layers,
        tabular_hidden_size=tabular_hidden_size,
        dropout=dropout,
        bidirectional=bidirectional,
    ).to(device_obj)

    class_weight_values = _compute_balanced_class_weights(class_labels, y_train)
    class_weight_tensor = torch.tensor(class_weight_values, dtype=torch.float32, device=device_obj)
    criterion = ClassBalancedFocalLoss(
        class_weights=class_weight_tensor,
        gamma=focal_gamma,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(patience // 2, 1),
    )

    train_sampler = _build_weighted_sampler(y_train, class_labels)
    train_loader = _multimodal_loader(
        X_train_seq,
        X_train_tab,
        y_train,
        batch_size=batch_size,
        sampler=train_sampler,
    )
    checkpoint_path = output_dir / "lstm_hibrida_hierarquica_best.pt"
    config_path = output_dir / "lstm_hibrida_hierarquica_config.json"
    history_path = output_dir / "lstm_hibrida_hierarquica_history.csv"

    history_rows = []
    best_signature: tuple[float, float, float] | None = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        for batch_seq, batch_tab, batch_y in train_loader:
            batch_seq = batch_seq.to(device_obj, non_blocking=True)
            batch_tab = batch_tab.to(device_obj, non_blocking=True)
            batch_y = batch_y.to(device_obj, non_blocking=True)
            logits = model(batch_seq, batch_tab)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

        train_pred = predict_hybrid_lstm_classes(
            model,
            X_train_seq,
            X_train_tab,
            batch_size=batch_size,
            device=str(device_obj),
        )
        val_pred = predict_hybrid_lstm_classes(
            model,
            X_val_seq,
            X_val_tab,
            batch_size=batch_size,
            device=str(device_obj),
        )

        train_eval = evaluate_predictions(y_train, train_pred, class_labels=class_labels)
        val_eval = evaluate_predictions(y_val, val_pred, class_labels=class_labels)

        current_row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(np.mean(epoch_losses)),
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
                    "class_labels": class_labels,
                    "continuous_indices": continuous_indices,
                    "state_indices": state_indices,
                    "window_size": window_size,
                    "continuous_hidden_size": continuous_hidden_size,
                    "state_hidden_size": state_hidden_size,
                    "context_hidden_size": context_hidden_size,
                    "context_num_layers": context_num_layers,
                    "tabular_hidden_size": tabular_hidden_size,
                    "dropout": dropout,
                    "bidirectional": bidirectional,
                    "focal_gamma": focal_gamma,
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
    return HybridTrainingSummary(
        model_name="lstm_hibrida_hierarquica_multiclasse",
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path),
        history_path=str(history_path),
        best_epoch=int(best_epoch),
        best_val_macro_f1=float(best_history_row["val_macro_f1"]),
        best_val_accuracy=float(best_history_row["val_accuracy"]),
        best_val_balanced_accuracy=float(best_history_row["val_balanced_accuracy"]),
    )


def load_hybrid_lstm_classifier(
    config_path: str | Path,
    *,
    device: str | None = None,
) -> nn.Module:
    require_torch()
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    device_obj = _default_device(device)
    model = HybridHierarchicalLSTMClassifier(
        input_size=int(payload["input_size"]),
        tabular_size=int(payload["tabular_size"]),
        num_classes=int(payload["num_classes"]),
        continuous_indices=[int(index) for index in payload["continuous_indices"]],
        state_indices=[int(index) for index in payload["state_indices"]],
        window_size=int(payload["window_size"]),
        continuous_hidden_size=int(payload["continuous_hidden_size"]),
        state_hidden_size=int(payload["state_hidden_size"]),
        context_hidden_size=int(payload["context_hidden_size"]),
        context_num_layers=int(payload["context_num_layers"]),
        tabular_hidden_size=int(payload["tabular_hidden_size"]),
        dropout=float(payload["dropout"]),
        bidirectional=bool(payload["bidirectional"]),
    )
    model.load_state_dict(torch.load(payload["checkpoint_path"], map_location=device_obj))
    model.to(device_obj)
    model.eval()
    return model


predict_lstm_classes = predict_hybrid_lstm_classes
train_lstm_classifier = train_hybrid_lstm_classifier
load_lstm_classifier = load_hybrid_lstm_classifier


def fit_random_forest_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    random_state: int = 42,
    n_estimators: int = 400,
    max_depth: int | None = None,
) -> RandomForestClassifier:
    require_classification_stack()
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)
    return model


def fit_xgboost_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    random_state: int = 42,
    n_estimators: int = 400,
    max_depth: int = 6,
    learning_rate: float = 0.05,
) -> Any | None:
    require_classification_stack()
    if XGBClassifier is None:
        return None
    model = XGBClassifier(
        objective="multi:softmax",
        num_class=int(len(np.unique(y_train))),
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        n_jobs=4,
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train)
    return model


def fit_lgbm_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    random_state: int = 42,
) -> Any | None:
    require_classification_stack()
    if LGBMClassifier is None:
        return None
    model = LGBMClassifier(
        objective="multiclass",
        num_class=int(len(np.unique(y_train))),
        boosting_type="gbdt",
        num_leaves=178,
        max_depth=22,
        learning_rate=0.01222,
        min_child_samples=63,
        subsample=0.6509,
        colsample_bytree=0.8193,
        n_estimators=949,
        feature_fraction=1.0,
        reg_alpha=2.60187,
        reg_lambda=0.03128,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(X_train, y_train)
    return model


def run_baseline_suite(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    class_labels: list[int],
    output_dir: str | Path,
    random_state: int = 42,
) -> dict[str, dict[str, Any]]:
    require_classification_stack()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}

    rf_model = fit_random_forest_baseline(X_train, y_train, random_state=random_state)
    rf_val_pred = rf_model.predict(X_val)
    rf_test_pred = rf_model.predict(X_test)
    rf_val_eval = evaluate_predictions(y_val, rf_val_pred, class_labels=class_labels)
    rf_test_eval = evaluate_predictions(y_test, rf_test_pred, class_labels=class_labels)
    export_evaluation_artifacts(rf_val_eval, output_dir, "random_forest_validation")
    export_evaluation_artifacts(rf_test_eval, output_dir, "random_forest_test")
    with open(output_dir / "random_forest_model.pkl", "wb") as file_pointer:
        pickle.dump(rf_model, file_pointer)
    results["random_forest"] = {
        "available": True,
        "model": rf_model,
        "validation": rf_val_eval,
        "test": rf_test_eval,
    }

    if XGBClassifier is not None:
        xgb_model = fit_xgboost_baseline(X_train, y_train, random_state=random_state)
        xgb_val_pred = xgb_model.predict(X_val)
        xgb_test_pred = xgb_model.predict(X_test)
        xgb_val_eval = evaluate_predictions(y_val, xgb_val_pred, class_labels=class_labels)
        xgb_test_eval = evaluate_predictions(y_test, xgb_test_pred, class_labels=class_labels)
        export_evaluation_artifacts(xgb_val_eval, output_dir, "xgboost_validation")
        export_evaluation_artifacts(xgb_test_eval, output_dir, "xgboost_test")
        with open(output_dir / "xgboost_model.pkl", "wb") as file_pointer:
            pickle.dump(xgb_model, file_pointer)
        results["xgboost"] = {
            "available": True,
            "model": xgb_model,
            "validation": xgb_val_eval,
            "test": xgb_test_eval,
        }
    else:
        results["xgboost"] = {
            "available": False,
            "model": None,
            "validation": None,
            "test": None,
            "message": "xgboost nao esta instalado; a baseline foi executada apenas com RandomForest.",
        }

    if LGBMClassifier is not None:
        lgbm_model = fit_lgbm_baseline(X_train, y_train, random_state=random_state)
        lgbm_val_pred = lgbm_model.predict(X_val)
        lgbm_test_pred = lgbm_model.predict(X_test)
        lgbm_val_eval = evaluate_predictions(y_val, lgbm_val_pred, class_labels=class_labels)
        lgbm_test_eval = evaluate_predictions(y_test, lgbm_test_pred, class_labels=class_labels)
        export_evaluation_artifacts(lgbm_val_eval, output_dir, "lgbm_validation")
        export_evaluation_artifacts(lgbm_test_eval, output_dir, "lgbm_test")
        with open(output_dir / "lgbm_model.pkl", "wb") as file_pointer:
            pickle.dump(lgbm_model, file_pointer)
        results["lgbm"] = {
            "available": True,
            "model": lgbm_model,
            "validation": lgbm_val_eval,
            "test": lgbm_test_eval,
        }
    else:
        results["lgbm"] = {
            "available": False,
            "model": None,
            "validation": None,
            "test": None,
            "message": "lightgbm nao esta instalado; a baseline LGBM nao foi executada.",
        }

    summary_rows = []
    for model_name, payload in results.items():
        if not payload["available"]:
            continue
        summary_rows.append(
            {
                "modelo": model_name,
                "split": "validation",
                "accuracy": payload["validation"]["accuracy"],
                "macro_f1": payload["validation"]["macro_f1"],
                "balanced_accuracy": payload["validation"]["balanced_accuracy"],
            }
        )
        summary_rows.append(
            {
                "modelo": model_name,
                "split": "test",
                "accuracy": payload["test"]["accuracy"],
                "macro_f1": payload["test"]["macro_f1"],
                "balanced_accuracy": payload["test"]["balanced_accuracy"],
            }
        )
    pd.DataFrame(summary_rows).to_csv(output_dir / "baseline_summary.csv", index=False)
    return results
