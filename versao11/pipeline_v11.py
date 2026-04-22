from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

from versao10 import pipeline_v10 as v10


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "reports_v11"

IGNORE_INDEX = v10.IGNORE_INDEX
FULL_FEATURE_COLUMNS = v10.FULL_FEATURE_COLUMNS
STATE_SENSOR_COLUMNS = v10.STATE_SENSOR_COLUMNS
CONTINUOUS_SENSOR_COLUMNS = v10.CONTINUOUS_SENSOR_COLUMNS
OBSERVATION_CLASS_CODES = v10.OBSERVATION_CLASS_CODES
OBSERVATION_STATE_CODES = v10.OBSERVATION_STATE_CODES
SOURCE_TYPE_MAPPING = v10.SOURCE_TYPE_MAPPING
ALL_NULL_FEATURE_COLUMNS = [
    "ABER-CKGL",
    "ABER-CKP",
    "P-JUS-BS",
    "P-JUS-CKP",
    "P-MON-CKGL",
    "P-MON-SDV-P",
    "PT-P",
    "QBS",
    "T-MON-CKP",
]
SELECTED_FEATURE_COLUMNS = [
    column_name
    for column_name in FULL_FEATURE_COLUMNS
    if column_name not in ALL_NULL_FEATURE_COLUMNS
]
SELECTED_STATE_SENSOR_COLUMNS = [name for name in SELECTED_FEATURE_COLUMNS if name in STATE_SENSOR_COLUMNS]
SELECTED_CONTINUOUS_SENSOR_COLUMNS = [name for name in SELECTED_FEATURE_COLUMNS if name not in SELECTED_STATE_SENSOR_COLUMNS]

ClassificationBundle = v10.ClassificationBundle
PreparedClassificationArtifacts = v10.PreparedClassificationArtifacts
MultiTaskTrainingSummary = v10.MultiTaskTrainingSummary

build_metrics_table = v10.build_metrics_table
discover_series_manifest = v10.discover_series_manifest
evaluate_predictions = v10.evaluate_predictions
export_evaluation_artifacts = v10.export_evaluation_artifacts
fit_lgbm_baseline = v10.fit_lgbm_baseline
fit_random_forest_baseline = v10.fit_random_forest_baseline
fit_xgboost_baseline = v10.fit_xgboost_baseline
load_attribute_catalog = v10.load_attribute_catalog
load_bundle = v10.load_bundle
load_event_catalog = v10.load_event_catalog
load_multitask_temporal_model = v10.load_multitask_temporal_model
load_split_arrays = v10.load_split_arrays
plot_confusion_matrix_for_predictions = v10.plot_confusion_matrix_for_predictions
predict_multitask_model_classes = v10.predict_multitask_model_classes
require_classification_stack = v10.require_classification_stack
require_plotting_stack = v10.require_plotting_stack
save_bundle = v10.save_bundle
stratified_split_manifest = v10.stratified_split_manifest
train_multitask_temporal_model = v10.train_multitask_temporal_model


def _require_dataframe_stack() -> None:
    if pd is None:
        raise ImportError(
            "pandas nao esta instalado neste ambiente. "
            "Instale as dependencias do projeto para executar a versao11."
        )


def _training_state_phase_from_value(value: object) -> int:
    if pd is not None and pd.isna(value):
        return IGNORE_INDEX
    try:
        code = int(value)
    except (TypeError, ValueError):
        return IGNORE_INDEX
    if code == 0:
        return 0
    if code == 1:
        return 1
    if code == 2:
        return 2
    return IGNORE_INDEX


def _map_training_state_phases(values: np.ndarray) -> np.ndarray:
    mapped = [_training_state_phase_from_value(value) for value in np.asarray(values, dtype=object)]
    return np.asarray(mapped, dtype=np.int64)


def _compute_feature_nan_ratio(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
) -> float:
    feature_columns = feature_columns or SELECTED_FEATURE_COLUMNS
    matrix = (
        frame[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64, copy=False)
    )
    if matrix.size == 0:
        return 1.0
    return float((~np.isfinite(matrix)).mean())


def build_feature_selection_report() -> pd.DataFrame:
    _require_dataframe_stack()
    rows = []
    for column_name in FULL_FEATURE_COLUMNS:
        dropped = column_name in ALL_NULL_FEATURE_COLUMNS
        rows.append(
            {
                "column": column_name,
                "null_pct": 100.0 if dropped else np.nan,
                "selected_for_modeling": not dropped,
                "selection_reason": "all_null_feature_removed" if dropped else "kept_for_modeling",
                "column_type": "state" if column_name in STATE_SENSOR_COLUMNS else "continuous",
            }
        )
    return pd.DataFrame(rows)


def build_series_quality_report(
    manifest: pd.DataFrame,
    *,
    max_nan_ratio: float = 0.70,
) -> pd.DataFrame:
    require_classification_stack()
    _require_dataframe_stack()

    rows = []
    for _, row in manifest.iterrows():
        frame = v10._prepare_raw_frame(row["file_path"])
        state_phases = _map_training_state_phases(frame["state"].to_numpy())

        nan_ratio = _compute_feature_nan_ratio(frame)
        n_rows_original = int(len(frame))
        n_normal_rows = int((state_phases == 0).sum())
        n_transient_rows = int((state_phases == 1).sum())
        n_failure_rows = int((state_phases == 2).sum())
        n_negative_rows = int(np.isin(state_phases, [1, 2]).sum())

        keep_for_v11 = True
        drop_reason = ""
        if int(row["class_label_int"]) != 0 and n_negative_rows == 0:
            keep_for_v11 = False
            drop_reason = "no_negative_state_segment"

        rows.append(
            {
                "file_path": row["file_path"],
                "series_id": row["series_id"],
                "class_label_int": int(row["class_label_int"]),
                "source_type": row["source_type"],
                "n_rows_original": n_rows_original,
                "nan_ratio_selected_features": nan_ratio,
                "n_state_normal_rows": n_normal_rows,
                "n_state_transient_rows": n_transient_rows,
                "n_state_failure_rows": n_failure_rows,
                "n_negative_state_rows": n_negative_rows,
                "keep_for_v11": bool(keep_for_v11),
                "drop_reason": drop_reason,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["keep_for_v11", "class_label_int", "series_id"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def _minimum_series_per_class_for_stratified_split(
    *,
    train_frac: float,
) -> int:
    temp_frac = 1.0 - float(train_frac)
    if temp_frac <= 0.0:
        raise ValueError("train_frac precisa ser menor que 1 para permitir validacao e teste.")
    return max(2, int(np.ceil(2.0 / temp_frac)))


def _apply_split_feasibility_filter(
    quality_report: pd.DataFrame,
    *,
    train_frac: float,
) -> pd.DataFrame:
    report = quality_report.copy()
    minimum_required = _minimum_series_per_class_for_stratified_split(train_frac=train_frac)

    primary_kept = report.loc[report["keep_for_v11"]].copy()
    class_counts = (
        primary_kept["class_label_int"]
        .value_counts()
        .sort_index()
    )
    report["class_count_after_primary_filters"] = (
        report["class_label_int"].map(class_counts).fillna(0).astype(int)
    )
    report["min_series_per_class_for_split"] = int(minimum_required)

    rare_classes = class_counts.loc[class_counts < minimum_required].index.tolist()
    if rare_classes:
        rare_mask = report["keep_for_v11"] & report["class_label_int"].isin(rare_classes)
        report.loc[rare_mask, "keep_for_v11"] = False
        report.loc[rare_mask, "drop_reason"] = "too_few_series_for_stratified_split"

    return report.sort_values(
        ["keep_for_v11", "class_label_int", "series_id"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def _prepare_frame_for_split(
    *,
    file_path: str | Path,
    global_class_label: int,
    split_name: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = v10._prepare_raw_frame(file_path)
    nan_ratio = _compute_feature_nan_ratio(frame)
    n_rows_original = int(len(frame))

    negative_only_applied = str(split_name) == "train" and int(global_class_label) != 0
    n_rows_removed_for_training_focus = 0

    if negative_only_applied:
        state_phases = _map_training_state_phases(frame["state"].to_numpy())
        keep_mask = np.isin(state_phases, [1, 2])
        filtered_frame = frame.loc[keep_mask].reset_index(drop=True)
        n_rows_removed_for_training_focus = int(n_rows_original - len(filtered_frame))
        if filtered_frame.empty:
            raise ValueError(
                "A serie ficou vazia apos remover estados ausentes/normal no treino. "
                f"Arquivo: {file_path}"
            )
        frame = filtered_frame

    metadata = {
        "nan_ratio_selected_features": float(nan_ratio),
        "n_rows_original": n_rows_original,
        "n_rows_used": int(len(frame)),
        "n_rows_removed_for_training_focus": int(n_rows_removed_for_training_focus),
        "negative_only_applied": bool(negative_only_applied),
        "preprocessing_mode": "state_negative_only_for_fault_train" if negative_only_applied else "full_series",
    }
    return frame, metadata


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
    for _, row in train_manifest.iterrows():
        frame, _ = _prepare_frame_for_split(
            file_path=row["file_path"],
            global_class_label=int(row["class_label_int"]),
            split_name="train",
        )

        columns = []
        for column_name in SELECTED_FEATURE_COLUMNS:
            raw_values = pd.to_numeric(frame[column_name], errors="coerce").to_numpy(dtype=np.float64)
            filled_values = v10._fill_series(raw_values, discrete=column_name in SELECTED_STATE_SENSOR_COLUMNS)
            columns.append(
                v10._resample_numeric(
                    filled_values,
                    sequence_length,
                    discrete=column_name in SELECTED_STATE_SENSOR_COLUMNS,
                )
            )
        sequence = v10._safe_numeric_matrix(np.stack(columns, axis=1))
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
    statistical_feature_names = v10.build_statistical_feature_names(SELECTED_FEATURE_COLUMNS)
    selected_files = {
        split_name: split_df["file_path"].tolist()
        for split_name, split_df in split_manifest.groupby("split", sort=False)
    }
    split_counts = split_manifest["split"].value_counts().sort_index().to_dict()

    return ClassificationBundle(
        selected_columns=SELECTED_FEATURE_COLUMNS.copy(),
        continuous_columns=SELECTED_CONTINUOUS_SENSOR_COLUMNS.copy(),
        state_columns=SELECTED_STATE_SENSOR_COLUMNS.copy(),
        sequence_length=int(sequence_length),
        scaler_mean=scaler.mean_.tolist(),
        scaler_scale=v10._safe_scale(scaler.scale_),
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
    v10.require_tabular_stack()

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
        split_name = str(row["split"]) if "split" in row else "train"
        frame, frame_metadata = _prepare_frame_for_split(
            file_path=row["file_path"],
            global_class_label=int(row["class_label_int"]),
            split_name=split_name,
        )

        sequence_columns = []
        missing_columns = []
        frozen_columns = []

        for column_name in bundle.selected_columns:
            raw_values = pd.to_numeric(frame[column_name], errors="coerce").to_numpy(dtype=np.float64)
            missing_mask = (~np.isfinite(raw_values)).astype(np.float64)
            filled_values = v10._fill_series(raw_values, discrete=column_name in bundle.state_columns)
            frozen_mask = v10._compute_frozen_mask(filled_values)

            sequence_columns.append(
                v10._resample_numeric(
                    filled_values,
                    bundle.sequence_length,
                    discrete=column_name in bundle.state_columns,
                )
            )
            missing_columns.append(
                v10._resample_numeric(
                    missing_mask,
                    bundle.sequence_length,
                    discrete=True,
                )
            )
            frozen_columns.append(
                v10._resample_numeric(
                    frozen_mask,
                    bundle.sequence_length,
                    discrete=True,
                )
            )

        sequence = v10._safe_numeric_matrix(np.stack(sequence_columns, axis=1))
        sequence_scaled = v10._safe_numeric_matrix((sequence - mean_arr) / scale_arr).astype(np.float32, copy=False)
        missing_seq = v10._safe_numeric_matrix(np.stack(missing_columns, axis=1)).astype(np.float32, copy=False)
        frozen_seq = v10._safe_numeric_matrix(np.stack(frozen_columns, axis=1)).astype(np.float32, copy=False)
        statistical_vector = v10.compute_statistical_feature_vector(sequence_scaled, bundle.selected_columns)

        step_class = v10._resample_labels(
            frame["class"].to_numpy(),
            bundle.sequence_length,
            bundle.observation_class_mapping,
        )
        step_state = v10._resample_labels(
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
                "split": split_name,
                **frame_metadata,
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
    run_name: str = "classificacao_v11_segmentos_negativos",
    train_frac: float = 0.70,
    validation_frac: float = 0.15,
    random_state: int = 42,
    sequence_length: int = 180,
    max_nan_ratio: float = 0.70,
) -> PreparedClassificationArtifacts:
    require_classification_stack()
    _require_dataframe_stack()

    dataset_root = Path(dataset_root)
    run_dir = ARTIFACTS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = discover_series_manifest(dataset_root)
    feature_selection_report = build_feature_selection_report()
    quality_report = build_series_quality_report(
        manifest,
        max_nan_ratio=max_nan_ratio,
    )
    quality_report = _apply_split_feasibility_filter(
        quality_report,
        train_frac=train_frac,
    )
    quality_report_path = run_dir / "series_quality_report.csv"
    feature_selection_report_path = run_dir / "feature_selection_report.csv"
    quality_report.to_csv(quality_report_path, index=False)
    feature_selection_report.to_csv(feature_selection_report_path, index=False)

    manifest_with_quality = manifest.merge(
        quality_report[
            [
                "file_path",
                "nan_ratio_selected_features",
                "n_rows_original",
                "n_negative_state_rows",
                "keep_for_v11",
                "drop_reason",
            ]
        ],
        on="file_path",
        how="left",
    )
    filtered_manifest = (
        manifest_with_quality
        .loc[manifest_with_quality["keep_for_v11"]]
        .drop(columns=["keep_for_v11"])
        .reset_index(drop=True)
    )
    if filtered_manifest.empty:
        raise ValueError(
            "Nenhuma serie restou apos os filtros da versao11. "
            "Revise a regra de recorte por `state` ou a disponibilidade das classes."
        )
    surviving_class_counts = filtered_manifest["class_label_int"].value_counts().sort_index()
    if len(surviving_class_counts) < 2:
        raise ValueError(
            "Menos de duas classes restaram apos os filtros da versao11, o que inviabiliza "
            "a classificacao multiclasse. Revise a regra de recorte por `state` ou os criterios de selecao."
        )

    split_manifest = stratified_split_manifest(
        filtered_manifest,
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

    bundle_path = run_dir / "bundle_v11.json"
    manifest_path = run_dir / "split_manifest_v11.csv"
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
