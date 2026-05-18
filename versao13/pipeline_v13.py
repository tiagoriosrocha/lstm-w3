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
from versao12 import pipeline_v12 as v12


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "reports_v13"

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


@dataclass
class TrainingSummary:
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
    return v12.fit_classification_bundle(
        train_manifest=train_manifest,
        split_manifest=split_manifest,
        dataset_root=dataset_root,
        sequence_length=sequence_length,
    )


def transform_manifest_to_arrays(
    manifest: pd.DataFrame,
    bundle: ClassificationBundle,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    arrays, metadata_df = v12.transform_manifest_to_arrays(manifest, bundle)
    return arrays, metadata_df


def fit_univariate_sequence_projection(
    X_seq: np.ndarray,
) -> dict[str, Any]:
    sequence = np.asarray(X_seq, dtype=np.float64)
    if sequence.ndim != 3:
        raise ValueError("X_seq precisa ter formato [amostras, tempo, features].")

    n_samples, n_steps, n_features = sequence.shape
    flat_matrix = sequence.reshape(n_samples * n_steps, n_features)
    projection_mean = flat_matrix.mean(axis=0)
    centered = flat_matrix - projection_mean
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    component = vh[0]
    explained_variance = singular_values ** 2
    explained_variance_ratio = float(
        explained_variance[0] / max(explained_variance.sum(), 1e-12)
    )
    return {
        "projection_type": "first_principal_component_per_timestep",
        "input_feature_count": int(n_features),
        "projection_mean": projection_mean.tolist(),
        "projection_component": component.tolist(),
        "explained_variance_ratio": explained_variance_ratio,
    }


def apply_univariate_sequence_projection(
    X_seq: np.ndarray,
    projection_payload: dict[str, Any],
) -> np.ndarray:
    sequence = np.asarray(X_seq, dtype=np.float64)
    if sequence.ndim != 3:
        raise ValueError("X_seq precisa ter formato [amostras, tempo, features].")

    mean_arr = np.asarray(projection_payload["projection_mean"], dtype=np.float64)
    component_arr = np.asarray(projection_payload["projection_component"], dtype=np.float64)
    if sequence.shape[-1] != len(mean_arr) or sequence.shape[-1] != len(component_arr):
        raise ValueError("A dimensao de features de X_seq nao bate com a projecao univariada salva.")

    centered = sequence - mean_arr.reshape(1, 1, -1)
    projected = np.tensordot(centered, component_arr, axes=([-1], [0]))
    return projected[..., np.newaxis].astype(np.float32, copy=False)


def load_univariate_sequence_projection(projection_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(projection_path).read_text(encoding="utf-8"))


def prepare_classification_artifacts(
    *,
    dataset_root: str | Path,
    run_name: str = "classificacao_v13_bigru_mha",
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

    bundle_path = run_dir / "bundle_v13.json"
    manifest_path = run_dir / "split_manifest_v13.csv"
    attribute_catalog_path = run_dir / "catalogo_atributos.csv"
    event_catalog_path = run_dir / "catalogo_eventos.csv"
    feature_selection_report_path = run_dir / "feature_selection_report.csv"
    projection_path = run_dir / "univariate_projection_v13.json"

    save_bundle(bundle, bundle_path)
    split_manifest.to_csv(manifest_path, index=False)
    load_attribute_catalog(dataset_root).to_csv(attribute_catalog_path, index=False)
    load_event_catalog(dataset_root).to_csv(event_catalog_path, index=False)
    build_feature_selection_report().to_csv(feature_selection_report_path, index=False)

    raw_arrays_by_split: dict[str, dict[str, np.ndarray]] = {}
    metadata_by_split: dict[str, pd.DataFrame] = {}
    for split_name in ["train", "validation", "test"]:
        split_df = split_manifest.loc[split_manifest["split"] == split_name].reset_index(drop=True)
        raw_arrays, metadata_df = transform_manifest_to_arrays(split_df, bundle)
        raw_arrays_by_split[split_name] = raw_arrays
        metadata_by_split[split_name] = metadata_df

    projection_payload = fit_univariate_sequence_projection(raw_arrays_by_split["train"]["X_seq"])
    _write_json(projection_payload, projection_path)

    split_npz_paths: dict[str, str] = {}
    split_metadata_paths: dict[str, str] = {}
    for split_name in ["train", "validation", "test"]:
        arrays = dict(raw_arrays_by_split[split_name])
        arrays["X_seq_bigru"] = apply_univariate_sequence_projection(
            arrays["X_seq"],
            projection_payload,
        )
        npz_path = run_dir / f"{split_name}_arrays.npz"
        metadata_path = run_dir / f"{split_name}_metadata.csv"
        np.savez_compressed(npz_path, **arrays)
        metadata_by_split[split_name].to_csv(metadata_path, index=False)
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
    return v12.load_split_arrays(npz_path)


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


if torch is not None:
    class BiGRUMultiHeadAttentionModel(nn.Module):
        def __init__(
            self,
            *,
            input_size: int = 1,
            num_classes: int,
            hidden_size: int = 48,
            num_layers: int = 1,
            attention_heads: int = 8,
            dropout: float = 0.30,
            fc_hidden_size: int | None = None,
        ) -> None:
            super().__init__()
            self.hidden_size = int(hidden_size)
            self.num_layers = int(num_layers)
            self.model_dim = self.hidden_size * 2
            self.fc_hidden_size = int(fc_hidden_size or self.model_dim)

            if self.model_dim % int(attention_heads) != 0:
                raise ValueError("2 * hidden_size precisa ser divisivel por attention_heads.")

            self.bigru = nn.GRU(
                input_size=int(input_size),
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if self.num_layers > 1 else 0.0,
            )
            self.post_gru_dropout = nn.Dropout(dropout)
            self.post_gru_norm = nn.LayerNorm(self.model_dim)
            self.mha = nn.MultiheadAttention(
                embed_dim=self.model_dim,
                num_heads=int(attention_heads),
                dropout=dropout,
                batch_first=True,
            )
            self.attention_dropout = nn.Dropout(dropout)
            self.residual_norm = nn.LayerNorm(self.model_dim)
            self.classifier = nn.Sequential(
                nn.Linear(self.model_dim, self.fc_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.fc_hidden_size, int(num_classes)),
            )

        def forward(self, x_seq: torch.Tensor) -> dict[str, torch.Tensor]:
            hidden_sequence, _ = self.bigru(x_seq)
            hidden_norm = self.post_gru_norm(self.post_gru_dropout(hidden_sequence))
            attention_output, _ = self.mha(hidden_norm, hidden_norm, hidden_norm, need_weights=False)
            residual_output = self.residual_norm(
                hidden_norm + torch.relu(self.attention_dropout(attention_output))
            )
            pooled_features = residual_output.mean(dim=1)
            logits = self.classifier(pooled_features)
            return {
                "logits": logits,
                "pooled_features": pooled_features,
            }
else:
    class BiGRUMultiHeadAttentionModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            v10.require_torch()


def _default_device(device: str | None = None) -> torch.device:
    v10.require_torch()
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _validate_univariate_sequence_batch(
    X_seq: np.ndarray,
) -> np.ndarray:
    sequence = np.asarray(X_seq, dtype=np.float32)
    if sequence.ndim != 3:
        raise ValueError("X_seq precisa ter formato [amostras, tempo, features].")
    if sequence.shape[-1] != 1:
        raise ValueError(
            "Este modelo espera sequencias univariadas em X_seq. "
            "Use o array X_seq_bigru gerado no pre-processamento da versao13."
        )
    return sequence


def _sequence_loader(
    X_seq: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int = 32,
    shuffle: bool = False,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    v10.require_torch()
    dataset = TensorDataset(
        torch.tensor(X_seq, dtype=torch.float32),
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


def _predict_bigru_attention_logits(
    model: nn.Module,
    X_seq: np.ndarray,
    *,
    batch_size: int = 128,
    device: str | None = None,
) -> np.ndarray:
    v10.require_torch()
    device_obj = _default_device(device)
    sequence = _validate_univariate_sequence_batch(X_seq)
    loader = _sequence_loader(
        sequence,
        np.zeros(len(sequence), dtype=np.int64),
        batch_size=batch_size,
        shuffle=False,
    )
    logits_parts = []
    model.eval()
    with torch.no_grad():
        for batch_seq, _ in loader:
            batch_output = model(batch_seq.to(device_obj, non_blocking=True))
            logits_parts.append(batch_output["logits"].detach().cpu().numpy())
    return np.concatenate(logits_parts, axis=0)


def predict_bigru_attention_model_classes(
    model: nn.Module,
    X_seq: np.ndarray,
    *,
    batch_size: int = 128,
    device: str | None = None,
) -> np.ndarray:
    logits = _predict_bigru_attention_logits(
        model,
        X_seq,
        batch_size=batch_size,
        device=device,
    )
    return logits.argmax(axis=1).astype(np.int64, copy=False)


def train_bigru_attention_model(
    X_train_seq: np.ndarray,
    y_train: np.ndarray,
    X_val_seq: np.ndarray,
    y_val: np.ndarray,
    *,
    output_dir: str | Path,
    class_labels: list[int],
    hidden_size: int = 48,
    num_layers: int = 1,
    attention_heads: int = 8,
    dropout: float = 0.30,
    fc_hidden_size: int | None = None,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    epochs: int = 30,
    patience: int = 8,
    random_state: int = 42,
    device: str | None = None,
) -> TrainingSummary:
    require_classification_stack()
    v10.require_torch()

    train_sequence = _validate_univariate_sequence_batch(X_train_seq)
    val_sequence = _validate_univariate_sequence_batch(X_val_seq)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(random_state)
    device_obj = _default_device(device)

    model = BiGRUMultiHeadAttentionModel(
        input_size=int(train_sequence.shape[-1]),
        num_classes=len(class_labels),
        hidden_size=hidden_size,
        num_layers=num_layers,
        attention_heads=attention_heads,
        dropout=dropout,
        fc_hidden_size=fc_hidden_size,
    ).to(device_obj)

    class_weight_values = _compute_balanced_class_weights(class_labels, y_train)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weight_values, dtype=torch.float32, device=device_obj)
    )

    optimizer = torch.optim.Adam(
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
    train_loader = _sequence_loader(
        train_sequence,
        y_train,
        batch_size=batch_size,
        sampler=train_sampler,
    )

    checkpoint_path = output_dir / "bigru_mha_best.pt"
    config_path = output_dir / "bigru_mha_config.json"
    history_path = output_dir / "bigru_mha_history.csv"

    history_rows = []
    best_signature: tuple[float, float, float] | None = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        for batch_seq, batch_y in train_loader:
            batch_seq = batch_seq.to(device_obj, non_blocking=True)
            batch_y = batch_y.to(device_obj, non_blocking=True)

            outputs = model(batch_seq)
            loss = criterion(outputs["logits"], batch_y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu().item()))

        train_pred = predict_bigru_attention_model_classes(
            model,
            train_sequence,
            batch_size=batch_size,
            device=str(device_obj),
        )
        val_pred = predict_bigru_attention_model_classes(
            model,
            val_sequence,
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
                    "input_size": int(train_sequence.shape[-1]),
                    "sequence_length": int(train_sequence.shape[1]),
                    "num_classes": len(class_labels),
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "attention_heads": attention_heads,
                    "dropout": dropout,
                    "fc_hidden_size": int(fc_hidden_size or (hidden_size * 2)),
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
    return TrainingSummary(
        model_name="bigru_multihead_attention_model",
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path),
        history_path=str(history_path),
        best_epoch=int(best_epoch),
        best_val_macro_f1=float(best_history_row["val_macro_f1"]),
        best_val_accuracy=float(best_history_row["val_accuracy"]),
        best_val_balanced_accuracy=float(best_history_row["val_balanced_accuracy"]),
    )


def load_bigru_attention_model(
    config_path: str | Path,
    *,
    device: str | None = None,
) -> nn.Module:
    v10.require_torch()
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    device_obj = _default_device(device)

    model = BiGRUMultiHeadAttentionModel(
        input_size=int(payload["input_size"]),
        num_classes=int(payload["num_classes"]),
        hidden_size=int(payload["hidden_size"]),
        num_layers=int(payload["num_layers"]),
        attention_heads=int(payload["attention_heads"]),
        dropout=float(payload["dropout"]),
        fc_hidden_size=int(payload["fc_hidden_size"]),
    )
    model.load_state_dict(torch.load(payload["checkpoint_path"], map_location=device_obj))
    model.to(device_obj)
    model.eval()
    return model
