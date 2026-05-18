from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

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

from versao3.pipeline_v3 import set_seed
from versao10 import pipeline_v10 as v10
from versao11 import pipeline_v11 as v11


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "reports_v12"

IGNORE_INDEX = v10.IGNORE_INDEX
FULL_FEATURE_COLUMNS = v10.FULL_FEATURE_COLUMNS
STATE_SENSOR_COLUMNS = v10.STATE_SENSOR_COLUMNS
CONTINUOUS_SENSOR_COLUMNS = v10.CONTINUOUS_SENSOR_COLUMNS
OBSERVATION_CLASS_CODES = v10.OBSERVATION_CLASS_CODES
OBSERVATION_STATE_CODES = v10.OBSERVATION_STATE_CODES
SOURCE_TYPE_MAPPING = v10.SOURCE_TYPE_MAPPING

ALL_NULL_FEATURE_COLUMNS = v11.ALL_NULL_FEATURE_COLUMNS
SELECTED_FEATURE_COLUMNS = v11.SELECTED_FEATURE_COLUMNS
SELECTED_STATE_SENSOR_COLUMNS = v11.SELECTED_STATE_SENSOR_COLUMNS
SELECTED_CONTINUOUS_SENSOR_COLUMNS = v11.SELECTED_CONTINUOUS_SENSOR_COLUMNS

ClassificationBundle = v10.ClassificationBundle
PreparedClassificationArtifacts = v10.PreparedClassificationArtifacts
MultiTaskTrainingSummary = v10.MultiTaskTrainingSummary

build_feature_selection_report = v11.build_feature_selection_report
build_metrics_table = v10.build_metrics_table
discover_series_manifest = v10.discover_series_manifest
evaluate_predictions = v10.evaluate_predictions
export_evaluation_artifacts = v10.export_evaluation_artifacts
fit_lgbm_baseline = v10.fit_lgbm_baseline
fit_random_forest_baseline = v10.fit_random_forest_baseline
fit_xgboost_baseline = v10.fit_xgboost_baseline
load_attribute_catalog = v10.load_attribute_catalog
load_event_catalog = v10.load_event_catalog
plot_confusion_matrix_for_predictions = v10.plot_confusion_matrix_for_predictions
require_classification_stack = v10.require_classification_stack
require_plotting_stack = v10.require_plotting_stack
save_bundle = v10.save_bundle
stratified_split_manifest = v10.stratified_split_manifest


def _write_json(payload: dict[str, Any], output_path: str | Path) -> None:
    Path(output_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_bundle(bundle_path: str | Path) -> ClassificationBundle:
    payload = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    return ClassificationBundle(**payload)


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
        frame = v10._prepare_raw_frame(file_path)
        columns = []
        for column_name in SELECTED_FEATURE_COLUMNS:
            raw_values = pd.to_numeric(frame[column_name], errors="coerce").to_numpy(dtype=np.float64)
            filled_values = v10._fill_series(
                raw_values,
                discrete=column_name in SELECTED_STATE_SENSOR_COLUMNS,
            )
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
        frame = v10._prepare_raw_frame(row["file_path"])
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
        sequence_scaled = v10._safe_numeric_matrix(
            (sequence - mean_arr) / scale_arr
        ).astype(np.float32, copy=False)
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
                "split": str(row["split"]) if "split" in row else "",
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
    run_name: str = "classificacao_v12_profunda_hierarquica_multitarefa",
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

    bundle_path = run_dir / "bundle_v12.json"
    manifest_path = run_dir / "split_manifest_v12.csv"
    attribute_catalog_path = run_dir / "catalogo_atributos.csv"
    event_catalog_path = run_dir / "catalogo_eventos.csv"
    feature_selection_report_path = run_dir / "feature_selection_report.csv"

    save_bundle(bundle, bundle_path)
    split_manifest.to_csv(manifest_path, index=False)
    load_attribute_catalog(dataset_root).to_csv(attribute_catalog_path, index=False)
    load_event_catalog(dataset_root).to_csv(event_catalog_path, index=False)
    build_feature_selection_report().to_csv(feature_selection_report_path, index=False)

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


def _default_feature_columns_for_input_size(
    input_size: int,
    feature_columns: list[str] | None,
) -> list[str]:
    if feature_columns is not None:
        if len(feature_columns) != int(input_size):
            raise ValueError("feature_columns precisa ter o mesmo tamanho da ultima dimensao de X_seq.")
        return [str(column_name) for column_name in feature_columns]
    if int(input_size) == len(SELECTED_FEATURE_COLUMNS):
        return SELECTED_FEATURE_COLUMNS.copy()
    if int(input_size) == len(FULL_FEATURE_COLUMNS):
        return FULL_FEATURE_COLUMNS.copy()
    raise ValueError(
        "Nao foi possivel inferir feature_columns automaticamente. "
        "Informe a lista de colunas usada para gerar X_seq."
    )


def resolve_feature_group_indices(
    feature_columns: list[str],
    state_columns: list[str] | None = None,
) -> tuple[list[int], list[int]]:
    state_names = set(state_columns if state_columns is not None else STATE_SENSOR_COLUMNS)
    state_indices = [
        idx for idx, column_name in enumerate(feature_columns)
        if column_name in state_names
    ]
    continuous_indices = [
        idx for idx, column_name in enumerate(feature_columns)
        if column_name not in state_names
    ]
    if not continuous_indices and not state_indices:
        raise ValueError("E necessario ter ao menos uma feature continua ou de estado.")
    return continuous_indices, state_indices


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


def _compute_balanced_index_weights(
    num_classes: int,
    y_indices: np.ndarray,
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
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


if torch is not None:
    @dataclass
    class _BranchSpec:
        feature_indices: list[int]
        hidden_size: int
        projection_size: int
        name: str


    class DeepHierarchicalMultitaskTemporalModel(nn.Module):
        def __init__(
            self,
            input_size: int,
            tabular_size: int,
            num_classes: int,
            num_step_classes: int,
            num_state_classes: int,
            source_vocab_size: int,
            continuous_indices: list[int],
            state_indices: list[int],
            *,
            window_size: int = 20,
            continuous_hidden_size: int = 128,
            state_hidden_size: int = 96,
            local_num_layers: int = 2,
            branch_projection_size: int = 160,
            context_hidden_size: int = 192,
            context_num_layers: int = 3,
            source_embedding_dim: int = 12,
            tabular_hidden_size: int = 160,
            dropout: float = 0.30,
            bidirectional: bool = True,
        ) -> None:
            super().__init__()
            self.window_size = int(window_size)
            self.bidirectional = bool(bidirectional)
            self.output_multiplier = 2 if self.bidirectional else 1
            self.source_embedding_dim = int(source_embedding_dim)
            self.branch_projection_size = int(branch_projection_size)
            self.local_num_layers = int(local_num_layers)

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

            self.sequence_input_norm = nn.LayerNorm(input_size)
            self.source_embedding = nn.Embedding(source_vocab_size, self.source_embedding_dim)

            self.branch_specs = [
                _BranchSpec(
                    feature_indices=continuous_indices,
                    hidden_size=int(continuous_hidden_size),
                    projection_size=int(branch_projection_size),
                    name="continuous",
                ),
                _BranchSpec(
                    feature_indices=state_indices,
                    hidden_size=int(state_hidden_size),
                    projection_size=int(branch_projection_size),
                    name="state",
                ),
            ]

            self.branch_modules = nn.ModuleDict()
            total_branch_dim = 0
            for spec in self.branch_specs:
                if not spec.feature_indices:
                    continue
                branch_input_dim = len(spec.feature_indices) * 3
                branch_output_dim = spec.hidden_size * self.output_multiplier
                pooled_dim = branch_output_dim * 3
                self.branch_modules[spec.name] = nn.ModuleDict(
                    {
                        "input_norm": nn.LayerNorm(branch_input_dim),
                        "encoder": nn.LSTM(
                            input_size=branch_input_dim,
                            hidden_size=spec.hidden_size,
                            num_layers=self.local_num_layers,
                            dropout=dropout if self.local_num_layers > 1 else 0.0,
                            batch_first=True,
                            bidirectional=self.bidirectional,
                        ),
                        "output_norm": nn.LayerNorm(branch_output_dim),
                        "attention": nn.Linear(branch_output_dim, 1),
                        "projection": nn.Sequential(
                            nn.LayerNorm(pooled_dim),
                            nn.Linear(pooled_dim, spec.projection_size),
                            nn.GELU(),
                            nn.Dropout(dropout),
                        ),
                    }
                )
                total_branch_dim += spec.projection_size

            if total_branch_dim == 0:
                raise ValueError("A arquitetura precisa receber pelo menos um ramo de features.")

            self.context_encoder = nn.LSTM(
                input_size=total_branch_dim + self.source_embedding_dim,
                hidden_size=context_hidden_size,
                num_layers=context_num_layers,
                dropout=dropout if context_num_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=self.bidirectional,
            )
            context_dim = context_hidden_size * self.output_multiplier
            self.context_norm = nn.LayerNorm(context_dim)
            self.context_attention = nn.Linear(context_dim, 1)

            self.step_class_head = nn.Sequential(
                nn.LayerNorm(context_dim),
                nn.Linear(context_dim, context_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(context_dim, num_step_classes),
            )
            self.step_state_head = nn.Sequential(
                nn.LayerNorm(context_dim),
                nn.Linear(context_dim, context_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(context_dim, num_state_classes),
            )

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
                nn.Linear(fusion_dim, 320),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(320, 160),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(160, num_classes),
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

        def _restore_step_resolution(
            self,
            window_logits: torch.Tensor,
            target_steps: int,
        ) -> torch.Tensor:
            expanded_logits = window_logits.repeat_interleave(self.window_size, dim=1)
            return expanded_logits[:, :target_steps, :]

        def _encode_branch(
            self,
            branch_sequence: torch.Tensor,
            *,
            branch_name: str,
        ) -> torch.Tensor:
            branch_modules = self.branch_modules[branch_name]
            windows = self._windowify(branch_sequence)
            batch_size, n_windows, window_size, n_features = windows.shape
            flat_windows = windows.reshape(batch_size * n_windows, window_size, n_features)
            flat_windows = branch_modules["input_norm"](flat_windows)
            sequence_output, (hidden_state, _) = branch_modules["encoder"](flat_windows)
            sequence_output = branch_modules["output_norm"](sequence_output)
            pooled = self._pool_lstm_outputs(sequence_output, hidden_state, branch_modules["attention"])
            projected = branch_modules["projection"](pooled)
            return projected.reshape(batch_size, n_windows, -1)

        def forward(
            self,
            x_seq: torch.Tensor,
            x_tab: torch.Tensor,
            x_missing: torch.Tensor,
            x_frozen: torch.Tensor,
            source_id: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            normalized_sequence = self.sequence_input_norm(x_seq)
            branch_windows = []

            if self.continuous_indices_tensor.numel() > 0:
                continuous_sequence = normalized_sequence.index_select(2, self.continuous_indices_tensor)
                continuous_missing = x_missing.index_select(2, self.continuous_indices_tensor)
                continuous_frozen = x_frozen.index_select(2, self.continuous_indices_tensor)
                branch_windows.append(
                    self._encode_branch(
                        torch.cat([continuous_sequence, continuous_missing, continuous_frozen], dim=-1),
                        branch_name="continuous",
                    )
                )

            if self.state_indices_tensor.numel() > 0:
                state_sequence = normalized_sequence.index_select(2, self.state_indices_tensor)
                state_missing = x_missing.index_select(2, self.state_indices_tensor)
                state_frozen = x_frozen.index_select(2, self.state_indices_tensor)
                branch_windows.append(
                    self._encode_branch(
                        torch.cat([state_sequence, state_missing, state_frozen], dim=-1),
                        branch_name="state",
                    )
                )

            if not branch_windows:
                raise RuntimeError("Nenhum ramo recebeu features validas.")

            window_features = torch.cat(branch_windows, dim=-1)
            source_embedding = self.source_embedding(source_id)
            source_context = source_embedding.unsqueeze(1).expand(-1, window_features.size(1), -1)
            context_input = torch.cat([window_features, source_context], dim=-1)

            context_output, (context_hidden, _) = self.context_encoder(context_input)
            context_output = self.context_norm(context_output)
            step_class_logits = self._restore_step_resolution(
                self.step_class_head(context_output),
                target_steps=x_seq.size(1),
            )
            step_state_logits = self._restore_step_resolution(
                self.step_state_head(context_output),
                target_steps=x_seq.size(1),
            )
            context_features = self._pool_lstm_outputs(
                context_output,
                context_hidden,
                self.context_attention,
            )

            tabular_features = self.tabular_branch(x_tab)
            fused_features = torch.cat([context_features, tabular_features, source_embedding], dim=1)
            instance_logits = self.classifier(fused_features)
            return {
                "instance_logits": instance_logits,
                "step_class_logits": step_class_logits,
                "step_state_logits": step_state_logits,
            }
else:
    class DeepHierarchicalMultitaskTemporalModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            v10.require_torch()


def _default_device(device: str | None = None) -> torch.device:
    v10.require_torch()
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
    v10.require_torch()
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
    v10.require_torch()
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
    v10.require_torch()
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
    feature_columns: list[str] | None = None,
    state_columns: list[str] | None = None,
    window_size: int = 20,
    continuous_hidden_size: int = 128,
    state_hidden_size: int = 96,
    local_num_layers: int = 2,
    branch_projection_size: int = 160,
    context_hidden_size: int = 192,
    context_num_layers: int = 3,
    source_embedding_dim: int = 12,
    tabular_hidden_size: int = 160,
    dropout: float = 0.30,
    bidirectional: bool = True,
    learning_rate: float = 4e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 48,
    epochs: int = 65,
    patience: int = 12,
    lambda_step_class: float = 0.35,
    lambda_step_state: float = 0.15,
    random_state: int = 42,
    device: str | None = None,
) -> MultiTaskTrainingSummary:
    require_classification_stack()
    v10.require_torch()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(random_state)
    device_obj = _default_device(device)

    resolved_feature_columns = _default_feature_columns_for_input_size(
        int(X_train_seq.shape[-1]),
        feature_columns,
    )
    resolved_state_columns = [
        str(column_name)
        for column_name in (
            state_columns
            if state_columns is not None
            else [
                column_name
                for column_name in resolved_feature_columns
                if column_name in STATE_SENSOR_COLUMNS
            ]
        )
    ]
    continuous_indices, state_indices = resolve_feature_group_indices(
        resolved_feature_columns,
        resolved_state_columns,
    )

    model = DeepHierarchicalMultitaskTemporalModel(
        input_size=int(X_train_seq.shape[-1]),
        tabular_size=int(X_train_tab.shape[-1]),
        num_classes=len(class_labels),
        num_step_classes=len(observation_class_codes),
        num_state_classes=len(observation_state_codes),
        source_vocab_size=source_vocab_size,
        continuous_indices=continuous_indices,
        state_indices=state_indices,
        window_size=window_size,
        continuous_hidden_size=continuous_hidden_size,
        state_hidden_size=state_hidden_size,
        local_num_layers=local_num_layers,
        branch_projection_size=branch_projection_size,
        context_hidden_size=context_hidden_size,
        context_num_layers=context_num_layers,
        source_embedding_dim=source_embedding_dim,
        tabular_hidden_size=tabular_hidden_size,
        dropout=dropout,
        bidirectional=bidirectional,
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

    checkpoint_path = output_dir / "hierarchical_multitask_temporal_best.pt"
    config_path = output_dir / "hierarchical_multitask_temporal_config.json"
    history_path = output_dir / "hierarchical_multitask_temporal_history.csv"

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
                    "feature_columns": resolved_feature_columns,
                    "state_columns": resolved_state_columns,
                    "window_size": window_size,
                    "continuous_hidden_size": continuous_hidden_size,
                    "state_hidden_size": state_hidden_size,
                    "local_num_layers": local_num_layers,
                    "branch_projection_size": branch_projection_size,
                    "context_hidden_size": context_hidden_size,
                    "context_num_layers": context_num_layers,
                    "source_embedding_dim": source_embedding_dim,
                    "tabular_hidden_size": tabular_hidden_size,
                    "dropout": dropout,
                    "bidirectional": bidirectional,
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
        model_name="deep_hierarchical_multitask_temporal_model",
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
    v10.require_torch()
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    device_obj = _default_device(device)
    feature_columns = [
        str(column_name)
        for column_name in payload.get("feature_columns", SELECTED_FEATURE_COLUMNS)
    ]
    state_columns = [
        str(column_name)
        for column_name in payload.get(
            "state_columns",
            [column_name for column_name in feature_columns if column_name in STATE_SENSOR_COLUMNS],
        )
    ]
    continuous_indices, state_indices = resolve_feature_group_indices(feature_columns, state_columns)

    model = DeepHierarchicalMultitaskTemporalModel(
        input_size=int(payload["input_size"]),
        tabular_size=int(payload["tabular_size"]),
        num_classes=int(payload["num_classes"]),
        num_step_classes=int(payload["num_step_classes"]),
        num_state_classes=int(payload["num_state_classes"]),
        source_vocab_size=int(payload["source_vocab_size"]),
        continuous_indices=continuous_indices,
        state_indices=state_indices,
        window_size=int(payload["window_size"]),
        continuous_hidden_size=int(payload["continuous_hidden_size"]),
        state_hidden_size=int(payload["state_hidden_size"]),
        local_num_layers=int(payload["local_num_layers"]),
        branch_projection_size=int(payload["branch_projection_size"]),
        context_hidden_size=int(payload["context_hidden_size"]),
        context_num_layers=int(payload["context_num_layers"]),
        source_embedding_dim=int(payload["source_embedding_dim"]),
        tabular_hidden_size=int(payload["tabular_hidden_size"]),
        dropout=float(payload["dropout"]),
        bidirectional=bool(payload["bidirectional"]),
    )
    model.load_state_dict(torch.load(payload["checkpoint_path"], map_location=device_obj))
    model.to(device_obj)
    model.eval()
    return model
