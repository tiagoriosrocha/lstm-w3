from __future__ import annotations

from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
import json
import time
from typing import Any

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:
    torch = None
    DataLoader = None

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "reports_v5"

from versao4.pipeline_v4 import (
    AUX_ANALOG_COLUMNS,
    BASE_TARGET_COLUMNS,
    CONTINUOUS_COLUMNS,
    STATE_COLUMNS,
    GroupedWindowDataset,
    PreprocessingBundle,
    StreamingPredictionResult,
    build_model,
    collect_training_reference_frame,
    compute_per_feature_metrics,
    count_group_windows,
    discover_all_dataset_files,
    export_streaming_result_tables,
    fit_preprocessing_bundle,
    load_bundle,
    load_grouped_sequences_from_directory,
    load_model_from_config,
    predict_loader_streaming,
    profile_continuous_columns,
    recommend_log_transform_columns,
    require_tabular_stack,
    require_torch,
    run_epoch,
    save_bundle,
    select_auxiliary_columns,
    set_seed,
    split_manifest_by_series,
    split_manifest_by_well,
    transform_frame_to_engineered_features,
    update_bundle_split_files,
    write_manifest_csv,
)


DEFAULT_MODEL_SPECS: dict[str, dict[str, Any]] = {
    "pure_lstm_forecaster_v4": {
        "alias": "lstm_pura",
        "hidden_size": 128,
        "recurrent_layers": 2,
        "well_embedding_dim": 16,
        "dropout": 0.20,
    },
    "hybrid_residual_forecaster_v4": {
        "alias": "hibrido_residual",
        "model_dim": 128,
        "hidden_size": 128,
        "recurrent_layers": 2,
        "well_embedding_dim": 16,
        "dropout": 0.20,
    },
}


@dataclass
class PreparedArtifacts:
    run_dir: str
    bundle_path: str
    manifest_path: str
    attribute_catalog_path: str
    event_catalog_path: str
    synthetic_catalog_path: str
    auxiliary_report_path: str
    continuous_profile_path: str
    split_directories: dict[str, str]
    selected_auxiliary_columns: list[str]
    recommended_log_transform_columns: list[str]


@dataclass
class TrainingRunSummary:
    model_name: str
    model_alias: str
    best_epoch: int
    best_val_mae: float
    best_val_persistence_mae: float
    best_ratio_to_persistence: float
    best_val_loss: float
    features_where_persistence_wins: int
    epochs_ran: int
    model_path: str
    model_config_path: str
    history_path: str
    summary_path: str


@dataclass
class EvaluationRunSummary:
    model_name: str
    model_alias: str
    model_config_path: str
    exported_table_paths: dict[str, str]
    result: StreamingPredictionResult


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


def _dataset_ini_path(dataset_root: str | Path | None = None) -> Path:
    root = Path(dataset_root) if dataset_root is not None else PROJECT_ROOT / "3W" / "dataset"
    return root / "dataset.ini"


def load_dataset_config(dataset_root: str | Path | None = None) -> ConfigParser:
    parser = ConfigParser()
    parser.optionxform = str
    parser.read(_dataset_ini_path(dataset_root), encoding="utf-8")
    return parser


def load_attribute_catalog(dataset_root: str | Path | None = None) -> pd.DataFrame:
    require_tabular_stack()
    parser = load_dataset_config(dataset_root)
    rows = []
    for attribute_name, description in parser["PARQUET_FILE_PROPERTIES"].items():
        role = "metadado"
        if attribute_name in BASE_TARGET_COLUMNS:
            role = "variavel_alvo"
        elif attribute_name in STATE_COLUMNS:
            role = "estado_discreto"
        elif attribute_name in AUX_ANALOG_COLUMNS:
            role = "variavel_auxiliar_analogica"
        elif attribute_name == "class":
            role = "rotulo_de_evento"
        elif attribute_name == "state":
            role = "estado_operacional"
        rows.append(
            {
                "atributo": attribute_name,
                "papel_no_projeto": role,
                "descricao_oficial": description,
            }
        )
    return pd.DataFrame(rows)


def load_event_catalog(dataset_root: str | Path | None = None) -> pd.DataFrame:
    require_tabular_stack()
    parser = load_dataset_config(dataset_root)
    rows = []
    for event_name in [name.strip() for name in parser["EVENTS"]["NAMES"].split(",")]:
        section = parser[event_name]
        rows.append(
            {
                "codigo_classe": int(section["LABEL"]),
                "nome_interno": event_name,
                "descricao": section["DESCRIPTION"],
                "transiente": bool(section.get("TRANSIENT", "False").lower() == "true"),
            }
        )
    return pd.DataFrame(rows).sort_values("codigo_classe").reset_index(drop=True)


def build_synthetic_attribute_catalog(rolling_window: int = 5) -> pd.DataFrame:
    require_tabular_stack()
    rows = [
        {
            "atributo_sintetico": "target__<variavel>",
            "origem": "BASE_TARGET_COLUMNS",
            "definicao": "Representacao escalonada da variavel-alvo utilizada como supervisao do modelo.",
            "justificativa_metodologica": "Padroniza a escala das saidas e reduz instabilidade numerica na otimizacao.",
        },
        {
            "atributo_sintetico": "raw__<variavel-alvo>",
            "origem": "BASE_TARGET_COLUMNS",
            "definicao": "Versao escalonada do valor bruto mais recente de cada variavel-alvo.",
            "justificativa_metodologica": "Permite que a rede aprenda correcoes residuais sobre a dinamica imediatamente observada.",
        },
        {
            "atributo_sintetico": "raw__<variavel_auxiliar>",
            "origem": "STATE_COLUMNS + AUX_ANALOG_COLUMNS selecionadas",
            "definicao": "Versao escalonada das covariaveis auxiliares admitidas no bundle.",
            "justificativa_metodologica": "Insere contexto operacional adicional sem romper a coerencia temporal da serie.",
        },
        {
            "atributo_sintetico": "diff1__<variavel>",
            "origem": "BASE_TARGET_COLUMNS",
            "definicao": "Primeira diferenca temporal entre a observacao corrente e a observacao imediatamente anterior.",
            "justificativa_metodologica": "Realca variacoes instantaneas e facilita a identificacao de mudancas de regime.",
        },
        {
            "atributo_sintetico": f"dev_roll{rolling_window}__<variavel>",
            "origem": "BASE_TARGET_COLUMNS",
            "definicao": f"Desvio da observacao em relacao a media movel de ordem {rolling_window}.",
            "justificativa_metodologica": "Captura afastamentos locais em relacao ao comportamento recente da serie.",
        },
        {
            "atributo_sintetico": f"std_roll{rolling_window}__<variavel>",
            "origem": "BASE_TARGET_COLUMNS",
            "definicao": f"Desvio-padrao movel em uma janela de {rolling_window} observacoes.",
            "justificativa_metodologica": "Resume a volatilidade local e oferece um indutor simples de heterocedasticidade.",
        },
        {
            "atributo_sintetico": "well_id",
            "origem": "well_name",
            "definicao": "Codificacao inteira do identificador do poco.",
            "justificativa_metodologica": "Viabiliza o uso de embedding de contexto por poco sem expandir o espaco dimensional por one-hot encoding.",
        },
        {
            "atributo_sintetico": "series_id",
            "origem": "class_label + nome do arquivo",
            "definicao": "Identificador unico da serie temporal de origem.",
            "justificativa_metodologica": "Permite rastreabilidade dos resultados por serie durante a avaliacao streaming.",
        },
    ]
    return pd.DataFrame(rows)


def build_glossary_dataframe() -> pd.DataFrame:
    require_tabular_stack()
    entries = [
        ("MAE", "metrica", "Erro absoluto medio entre valores previstos e observados.", "Mede erro medio sem amplificar discrepancias extremas."),
        ("MSE", "metrica", "Erro quadratico medio entre previsao e referencia.", "Penaliza fortemente erros de grande magnitude."),
        ("RMSE", "metrica", "Raiz quadrada do MSE.", "Retorna o erro em escala comparavel a unidade original da variavel."),
        ("R2", "metrica", "Coeficiente de determinacao.", "Indica a fracao da variabilidade explicada pelo modelo."),
        ("baseline", "avaliacao", "Metodo de referencia usado como padrao minimo de comparacao.", "Neste projeto, a baseline principal e a persistencia."),
        ("persistencia", "avaliacao", "Estrategia que projeta o ultimo valor observado para o horizonte futuro.", "Serve como referencia forte em series de curta memoria."),
        ("janela deslizante", "series_temporais", "Subsequencia temporal usada como entrada do modelo.", "A janela resume o contexto recente para prever o futuro imediato."),
        ("horizonte de previsao", "series_temporais", "Numero de passos futuros previstos por amostra.", "Na configuracao recomendada da versao5, o horizonte e multi-step."),
        ("serie temporal multivariada", "series_temporais", "Sequencia cronologica composta por varias variaveis observadas simultaneamente.", "O 3W fornece sinais de pressao, temperatura, vazao e estados operacionais."),
        ("split temporal", "series_temporais", "Particao treino-validacao-teste que preserva a ordem cronologica.", "Evita vazamento de informacao futura."),
        ("data leakage", "series_temporais", "Uso inadvertido de informacao futura ou externa durante o treinamento.", "Compromete a validade da avaliacao experimental."),
        ("outlier", "pre_processamento", "Observacao extrema em relacao ao comportamento dominante dos dados.", "Pode distorcer escalonamento e treinamento."),
        ("interpolacao", "pre_processamento", "Estimativa de valores faltantes com base em observacoes vizinhas.", "Foi usada nas variaveis analogicas continuas."),
        ("forward fill", "pre_processamento", "Preenchimento de ausencias com o ultimo valor valido observado.", "Apropriado para sinais de estado quase discretos."),
        ("robust scaler", "pre_processamento", "Escalonador baseado em mediana e intervalo interquartil.", "Reduz sensibilidade a amplitudes extremas."),
        ("feature engineering", "pre_processamento", "Construcao de atributos derivados para ampliar a capacidade descritiva dos dados.", "As features diff1, dev_roll e std_roll sao exemplos."),
        ("diff1", "pre_processamento", "Primeira diferenca temporal.", "Aproxima a derivada discreta do sinal."),
        ("rolling mean", "pre_processamento", "Media movel calculada em uma janela finita.", "Resume tendencia local."),
        ("rolling standard deviation", "pre_processamento", "Desvio-padrao movel.", "Resume variabilidade local."),
        ("desvio local", "pre_processamento", "Diferenca entre a observacao atual e sua media movel.", "Quantifica afastamento do comportamento recente."),
        ("batch", "otimizacao", "Conjunto de amostras processadas conjuntamente em cada iteracao.", "Equilibra custo computacional e estabilidade do gradiente."),
        ("epoch", "otimizacao", "Passagem completa do algoritmo sobre o conjunto amostrado de treinamento.", "Usada para monitorar convergencia."),
        ("learning rate", "otimizacao", "Passo de atualizacao dos parametros durante a descida de gradiente.", "Valores inadequados podem causar divergencia ou lentidao."),
        ("weight decay", "otimizacao", "Penalizacao L2 aplicada aos parametros.", "Auxilia no controle de sobreajuste."),
        ("early stopping", "otimizacao", "Interrupcao do treino quando a validacao deixa de melhorar.", "Evita overfitting e reduz custo computacional."),
        ("gradient clipping", "otimizacao", "Limitacao da norma do gradiente.", "Importante para estabilidade em redes recorrentes."),
        ("dropout", "regularizacao", "Desativacao aleatoria de unidades durante o treino.", "Reduz coadaptacao excessiva entre neuronios."),
        ("overfitting", "regularizacao", "Ajuste excessivo aos dados de treino com perda de generalizacao.", "Sinaliza baixa robustez fora da amostra."),
        ("generalizacao", "regularizacao", "Capacidade do modelo de manter desempenho em dados nao vistos.", "E o objetivo central da avaliacao em teste."),
        ("loss", "otimizacao", "Funcao objetivo minimizada durante o treinamento.", "Resume o erro de previsao em uma quantidade escalar."),
        ("Smooth L1", "otimizacao", "Funcao de perda robusta que combina propriedades de L1 e L2.", "Mitiga a influencia de outliers sem ignorar a magnitude dos erros."),
        ("RNN", "redes_recorrentes", "Rede neural recorrente para modelagem sequencial.", "Familia arquitetural base para LSTM e GRU."),
        ("LSTM", "redes_recorrentes", "Long Short-Term Memory, arquitetura recorrente com portas de memoria.", "Adequada para dependencias temporais de medio e longo alcance."),
        ("GRU", "redes_recorrentes", "Gated Recurrent Unit, variante recorrente mais compacta que a LSTM.", "Empregada no modelo hibrido residual."),
        ("hidden state", "redes_recorrentes", "Estado latente que resume informacao processada ate um dado passo temporal.", "Transporta memoria dinamica ao longo da sequencia."),
        ("embedding", "redes_recorrentes", "Representacao vetorial densa de uma categoria discreta.", "Na versao5, o poco e representado por embedding."),
        ("attention", "redes_recorrentes", "Mecanismo que pondera a relevancia relativa dos passos da sequencia.", "Permite concentrar o modelo em trechos mais informativos."),
        ("bidirectional", "redes_recorrentes", "Processamento recorrente nos sentidos temporal direto e reverso.", "Expande a capacidade de sumarizacao contextual dentro da janela."),
        ("aprendizado residual", "redes_recorrentes", "Estrategia em que a rede aprende um delta sobre uma referencia simples.", "A previsao final corrige a baseline de persistencia."),
        ("convolucao temporal", "redes_recorrentes", "Operacao convolucional aplicada ao eixo temporal.", "Capta padroes locais e regularidades de curta duracao."),
        ("streaming inference", "avaliacao", "Inferencia feita em fluxo, sem materializar toda a matriz de previsoes simultaneamente.", "Viabiliza avaliacao em larga escala com menor uso de memoria."),
        ("token", "pln", "Unidade elementar de texto apos segmentacao.", "Termo central em PLN, embora nao seja usado diretamente neste projeto de series temporais."),
        ("tokenizacao", "pln", "Processo de segmentar texto em tokens.", "Etapa fundamental em pipelines de linguagem natural."),
        ("vocabulario", "pln", "Conjunto de tokens conhecidos por um modelo.", "Define o espaco discreto de simbolos manipulados em PLN."),
        ("OOV", "pln", "Out-of-vocabulary; token ausente do vocabulario conhecido.", "Pode degradar a cobertura lexical de um modelo de linguagem."),
        ("padding", "pln", "Insercao de simbolos de preenchimento para igualar comprimentos.", "Analogo ao alinhamento de sequencias de tamanhos distintos."),
        ("truncation", "pln", "Corte de parte da sequencia para respeitar comprimento maximo.", "Afeta a quantidade de contexto fornecida ao modelo."),
        ("self-attention", "pln", "Mecanismo em que cada posicao da sequencia pondera as demais posicoes.", "Base dos Transformers modernos."),
        ("Transformer", "pln", "Arquitetura sequencial baseada predominantemente em atencao.", "Popular em PLN e cada vez mais adotada em series temporais."),
        ("seq2seq", "pln", "Arquitetura de sequencia-para-sequencia.", "Relaciona uma sequencia de entrada a uma sequencia de saida."),
        ("corpus", "pln", "Colecao estruturada de textos usada em treinamento ou avaliacao.", "Equivale, por analogia, a um conjunto de series em estudos temporais."),
        ("modelo de linguagem", "pln", "Modelo probabilistico ou neural que estima distribuicoes sobre sequencias de texto.", "Nao e o foco deste projeto, mas compartilha principios de modelagem sequencial."),
        ("perplexidade", "pln", "Metrica de incerteza comum em modelos de linguagem.", "Quanto menor, maior a capacidade do modelo de antecipar a sequencia textual."),
    ]
    return pd.DataFrame(
        entries,
        columns=["termo", "categoria", "definicao_academica", "relacao_com_o_projeto"],
    )


def summarize_dataset_inventory(
    dataset_root: str | Path,
    include_row_counts: bool = True,
) -> dict[str, pd.DataFrame]:
    require_tabular_stack()
    manifest = discover_all_dataset_files(Path(dataset_root))

    overview_rows = [
        {"estatistica": "arquivos", "valor": int(len(manifest))},
        {"estatistica": "classes_distintas", "valor": int(manifest["class_label"].nunique())},
        {"estatistica": "pocos_distintos", "valor": int(manifest["well_name"].nunique())},
        {"estatistica": "tipos_de_origem", "valor": int(manifest["source_type"].nunique())},
    ]

    class_df = (
        manifest["class_label"]
        .value_counts()
        .rename_axis("class_label")
        .reset_index(name="arquivos")
        .sort_values("class_label")
        .reset_index(drop=True)
    )
    source_df = (
        manifest["source_type"]
        .value_counts()
        .rename_axis("source_type")
        .reset_index(name="arquivos")
        .sort_values("source_type")
        .reset_index(drop=True)
    )
    well_df = (
        manifest["well_name"]
        .value_counts()
        .rename_axis("well_name")
        .reset_index(name="arquivos")
        .sort_values(["arquivos", "well_name"], ascending=[False, True])
        .reset_index(drop=True)
    )

    length_df = pd.DataFrame()
    if include_row_counts and pq is not None:
        length_rows = []
        for _, row in manifest.iterrows():
            metadata = pq.ParquetFile(row["file_path"]).metadata
            length_rows.append(
                {
                    "series_id": row["series_id"],
                    "class_label": row["class_label"],
                    "source_type": row["source_type"],
                    "well_name": row["well_name"],
                    "n_linhas": int(metadata.num_rows),
                }
            )
        length_df = pd.DataFrame(length_rows)
        if not length_df.empty:
            overview_rows.extend(
                [
                    {"estatistica": "linhas_totais", "valor": int(length_df["n_linhas"].sum())},
                    {"estatistica": "mediana_linhas_por_serie", "valor": float(length_df["n_linhas"].median())},
                    {"estatistica": "media_linhas_por_serie", "valor": float(length_df["n_linhas"].mean())},
                ]
            )

    return {
        "manifest": manifest,
        "overview": pd.DataFrame(overview_rows),
        "class_distribution": class_df,
        "source_distribution": source_df,
        "well_distribution": well_df,
        "length_distribution": length_df,
    }


def _write_chunk(parts: list[pd.DataFrame], output_path: Path) -> None:
    require_tabular_stack()
    if not parts:
        return
    chunk_df = pd.concat(parts, ignore_index=True)
    chunk_df.to_parquet(output_path, index=False)


def export_engineered_split(
    split_manifest: pd.DataFrame,
    bundle: PreprocessingBundle,
    output_dir: str | Path,
    *,
    split_name: str,
    files_per_chunk: int = 16,
) -> list[str]:
    require_tabular_stack()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    current_parts: list[pd.DataFrame] = []
    current_files = 0
    part_idx = 0
    saved_files: list[str] = []

    for file_path in split_manifest["file_path"]:
        current_parts.append(transform_frame_to_engineered_features(file_path, bundle))
        current_files += 1
        if current_files >= files_per_chunk:
            part_idx += 1
            chunk_path = output_path / f"{split_name}_part_{part_idx:04d}.parquet"
            _write_chunk(current_parts, chunk_path)
            saved_files.append(str(chunk_path))
            current_parts = []
            current_files = 0

    if current_parts:
        part_idx += 1
        chunk_path = output_path / f"{split_name}_part_{part_idx:04d}.parquet"
        _write_chunk(current_parts, chunk_path)
        saved_files.append(str(chunk_path))

    return saved_files


def prepare_comparative_artifacts(
    *,
    dataset_root: str | Path,
    run_name: str = "comparativo_v5",
    split_strategy: str = "by_well",
    train_frac: float = 0.70,
    validation_frac: float = 0.15,
    rolling_window: int = 5,
    sequence_length: int = 60,
    forecast_horizon: int = 5,
    scaler_strategy: str = "robust",
    max_rows_per_series_for_reference: int = 512,
    files_per_chunk: int = 16,
) -> PreparedArtifacts:
    require_tabular_stack()
    dataset_root = Path(dataset_root)
    run_dir = ARTIFACTS_ROOT / run_name
    preprocessed_root = run_dir / "preprocessed"
    reports_root = run_dir / "reports"
    preprocessed_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    manifest = discover_all_dataset_files(dataset_root)
    if split_strategy == "by_well":
        split_manifest = split_manifest_by_well(
            manifest,
            train_frac=train_frac,
            validation_frac=validation_frac,
        )
    elif split_strategy == "by_series":
        split_manifest = split_manifest_by_series(
            manifest,
            train_frac=train_frac,
            validation_frac=validation_frac,
        )
    else:
        raise ValueError("split_strategy precisa ser 'by_well' ou 'by_series'.")

    train_manifest = split_manifest.loc[split_manifest["split"] == "train"].reset_index(drop=True)
    reference_df = collect_training_reference_frame(
        train_manifest=train_manifest,
        max_rows_per_series=max_rows_per_series_for_reference,
    )
    auxiliary_report = select_auxiliary_columns(reference_df)
    selected_auxiliary_columns = (
        auxiliary_report.loc[auxiliary_report["selected_for_input"], "column"].tolist()
    )

    continuous_profile = profile_continuous_columns(reference_df)
    recommended_log_columns = recommend_log_transform_columns(
        continuous_profile,
        candidate_columns=BASE_TARGET_COLUMNS + selected_auxiliary_columns,
    )

    bundle = fit_preprocessing_bundle(
        train_manifest=train_manifest,
        auxiliary_columns=selected_auxiliary_columns,
        max_files_per_well=None,
        rolling_window=rolling_window,
        sequence_length_recommendation=sequence_length,
        forecast_horizon_recommendation=forecast_horizon,
        scaler_strategy=scaler_strategy,
        log_transform_columns=recommended_log_columns,
    )
    bundle = update_bundle_split_files(bundle, split_manifest)

    bundle_path = run_dir / "bundle_v5.json"
    manifest_path = run_dir / "split_manifest_v5.csv"
    attribute_catalog_path = reports_root / "catalogo_atributos_originais.csv"
    event_catalog_path = reports_root / "catalogo_eventos.csv"
    synthetic_catalog_path = reports_root / "catalogo_atributos_sinteticos.csv"
    auxiliary_report_path = reports_root / "relatorio_variaveis_auxiliares.csv"
    continuous_profile_path = reports_root / "perfil_colunas_continuas.csv"

    save_bundle(bundle, bundle_path)
    write_manifest_csv(split_manifest, manifest_path)
    load_attribute_catalog(dataset_root).to_csv(attribute_catalog_path, index=False)
    load_event_catalog(dataset_root).to_csv(event_catalog_path, index=False)
    build_synthetic_attribute_catalog(rolling_window=rolling_window).to_csv(synthetic_catalog_path, index=False)
    auxiliary_report.to_csv(auxiliary_report_path, index=False)
    continuous_profile.to_csv(continuous_profile_path, index=False)

    split_directories: dict[str, str] = {}
    for split_name in ["train", "validation", "test"]:
        split_dir = preprocessed_root / split_name
        split_subset = split_manifest.loc[split_manifest["split"] == split_name].reset_index(drop=True)
        export_engineered_split(
            split_subset,
            bundle,
            split_dir,
            split_name=split_name,
            files_per_chunk=files_per_chunk,
        )
        split_directories[split_name] = str(split_dir)

    return PreparedArtifacts(
        run_dir=str(run_dir),
        bundle_path=str(bundle_path),
        manifest_path=str(manifest_path),
        attribute_catalog_path=str(attribute_catalog_path),
        event_catalog_path=str(event_catalog_path),
        synthetic_catalog_path=str(synthetic_catalog_path),
        auxiliary_report_path=str(auxiliary_report_path),
        continuous_profile_path=str(continuous_profile_path),
        split_directories=split_directories,
        selected_auxiliary_columns=selected_auxiliary_columns,
        recommended_log_transform_columns=recommended_log_columns,
    )


def load_prepared_groups(
    bundle_path: str | Path,
    split_directories: dict[str, str | Path],
) -> tuple[PreprocessingBundle, dict[str, list[dict[str, Any]]]]:
    bundle = load_bundle(bundle_path)
    groups = {
        split_name: load_grouped_sequences_from_directory(
            parquet_dir=split_dir,
            input_columns=bundle.input_columns,
            target_columns=bundle.target_columns,
        )
        for split_name, split_dir in split_directories.items()
    }
    return bundle, groups


def _default_device(device: str | torch.device | None = None) -> torch.device:
    require_torch()
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_dataloader(
    groups: list[dict[str, Any]],
    *,
    sequence_length: int,
    forecast_horizon: int,
    batch_size: int,
    sampled_windows: int | None,
    seed: int,
    balance_by_class: bool,
    num_workers: int,
) -> tuple[GroupedWindowDataset, DataLoader]:
    require_torch()
    dataset = GroupedWindowDataset(
        groups=groups,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        sampled_windows=sampled_windows,
        seed=seed,
        balance_by_class=balance_by_class,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def _build_model_config(
    *,
    model_name: str,
    bundle: PreprocessingBundle,
    forecast_horizon: int,
    model_path: str | Path,
    spec: dict[str, Any],
) -> dict[str, Any]:
    config = {
        "model_name": model_name,
        "input_size": len(bundle.input_columns),
        "target_size": len(bundle.target_columns),
        "forecast_horizon": forecast_horizon,
        "raw_target_positions": [
            bundle.input_columns.index(column_name)
            for column_name in bundle.raw_target_input_columns
        ],
        "model_path": str(model_path),
    }
    for field_name in ["model_dim", "hidden_size", "recurrent_layers", "well_embedding_dim", "dropout"]:
        if field_name in spec:
            config[field_name] = spec[field_name]
    return config


def train_single_model(
    *,
    model_name: str,
    bundle: PreprocessingBundle,
    train_groups: list[dict[str, Any]],
    validation_groups: list[dict[str, Any]],
    output_dir: str | Path,
    forecast_horizon: int | None = None,
    sequence_length: int | None = None,
    epochs: int = 20,
    patience: int = 5,
    batch_size: int = 128,
    sampled_windows_train: int | None = 140_000,
    sampled_windows_validation: int | None = 50_000,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    balance_by_class: bool = True,
    num_workers: int = 0,
    seed: int = 42,
    device: str | torch.device | None = None,
    use_amp: bool | None = None,
) -> TrainingRunSummary:
    require_torch()
    require_tabular_stack()

    if model_name not in DEFAULT_MODEL_SPECS:
        raise ValueError(f"Modelo desconhecido para a versao5: {model_name}")

    device_obj = _default_device(device)
    if use_amp is None:
        use_amp = device_obj.type == "cuda"

    spec = DEFAULT_MODEL_SPECS[model_name]
    model_alias = spec["alias"]
    forecast_horizon = int(forecast_horizon or bundle.forecast_horizon_recommendation)
    sequence_length = int(sequence_length or bundle.sequence_length_recommendation)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    train_dataset, train_loader = _build_dataloader(
        train_groups,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        batch_size=batch_size,
        sampled_windows=sampled_windows_train,
        seed=seed,
        balance_by_class=balance_by_class,
        num_workers=num_workers,
    )
    validation_dataset, validation_loader = _build_dataloader(
        validation_groups,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        batch_size=batch_size,
        sampled_windows=sampled_windows_validation,
        seed=seed + 1_000,
        balance_by_class=False,
        num_workers=num_workers,
    )

    model = build_model(
        model_name=model_name,
        input_size=len(bundle.input_columns),
        target_size=len(bundle.target_columns),
        forecast_horizon=forecast_horizon,
        well_count=max(len(bundle.well_to_id), 1),
        raw_target_positions=[
            bundle.input_columns.index(column_name)
            for column_name in bundle.raw_target_input_columns
        ],
        model_dim=spec.get("model_dim", 128),
        hidden_size=spec.get("hidden_size", 128),
        recurrent_layers=spec.get("recurrent_layers", 2),
        well_embedding_dim=spec.get("well_embedding_dim", 16),
        dropout=spec.get("dropout", 0.20),
    )
    model.to(device_obj)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(patience // 2, 1),
    )
    grad_scaler = torch.cuda.amp.GradScaler(enabled=bool(use_amp and device_obj.type == "cuda"))

    history_rows: list[dict[str, Any]] = []
    best_signature: tuple[float, float, float] | None = None
    best_summary_payload: dict[str, Any] | None = None
    best_features_where_persistence_wins = 0
    best_epoch = 0
    patience_counter = 0

    model_path = output_dir / f"{model_alias}_best.pt"
    model_config_path = output_dir / f"{model_alias}_model_config.json"
    history_path = output_dir / f"{model_alias}_history.csv"
    summary_path = output_dir / f"{model_alias}_summary.json"

    training_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_dataset.resample(epoch=epoch)

        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device_obj,
            [
                bundle.input_columns.index(column_name)
                for column_name in bundle.raw_target_input_columns
            ],
            use_amp=bool(use_amp),
            grad_scaler=grad_scaler,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            None,
            device_obj,
            [
                bundle.input_columns.index(column_name)
                for column_name in bundle.raw_target_input_columns
            ],
            use_amp=bool(use_amp),
            grad_scaler=None,
        )
        scheduler.step(validation_metrics["model_mae"])

        val_ratio = float(validation_metrics["model_mae"] / validation_metrics["persistence_mae"]) if validation_metrics["persistence_mae"] != 0 else np.nan
        per_feature_val = compute_per_feature_metrics(
            validation_metrics["y_true"],
            validation_metrics["y_pred"],
            validation_metrics["y_persist"],
            feature_names=bundle.target_columns,
        )
        features_where_persistence_wins = int(
            (per_feature_val["mae_melhora_pct"].fillna(-np.inf) <= 0.0).sum()
        )

        current_row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_metrics["loss"]),
            "train_mae": float(train_metrics["model_mae"]),
            "train_persistence_mae": float(train_metrics["persistence_mae"]),
            "train_improvement_pct": float(train_metrics["mae_improvement_pct"]),
            "val_loss": float(validation_metrics["loss"]),
            "val_mae": float(validation_metrics["model_mae"]),
            "val_persistence_mae": float(validation_metrics["persistence_mae"]),
            "val_improvement_pct": float(validation_metrics["mae_improvement_pct"]),
            "val_ratio_to_persistence": val_ratio,
            "features_where_persistence_wins": features_where_persistence_wins,
        }
        history_rows.append(current_row)

        current_signature = (
            float(val_ratio if not np.isnan(val_ratio) else np.inf),
            float(validation_metrics["model_mae"]),
            float(validation_metrics["loss"]),
        )
        if best_signature is None or current_signature < best_signature:
            best_signature = current_signature
            best_epoch = epoch
            best_features_where_persistence_wins = features_where_persistence_wins
            patience_counter = 0
            torch.save(model.state_dict(), model_path)

            model_config = _build_model_config(
                model_name=model_name,
                bundle=bundle,
                forecast_horizon=forecast_horizon,
                model_path=model_path,
                spec=spec,
            )
            _write_json(model_config, model_config_path)

            best_summary_payload = {
                "model_name": model_name,
                "model_alias": model_alias,
                "best_epoch": epoch,
                "best_val_mae": float(validation_metrics["model_mae"]),
                "best_val_persistence_mae": float(validation_metrics["persistence_mae"]),
                "best_ratio_to_persistence": val_ratio,
                "best_val_loss": float(validation_metrics["loss"]),
                "features_where_persistence_wins": features_where_persistence_wins,
                "sequence_length": sequence_length,
                "forecast_horizon": forecast_horizon,
                "sampled_windows_train": sampled_windows_train,
                "sampled_windows_validation": sampled_windows_validation,
                "train_possible_windows": count_group_windows(
                    train_groups,
                    sequence_length=sequence_length,
                    forecast_horizon=forecast_horizon,
                ),
                "validation_possible_windows": count_group_windows(
                    validation_groups,
                    sequence_length=sequence_length,
                    forecast_horizon=forecast_horizon,
                ),
                "training_seconds": float(time.perf_counter() - training_start),
            }
            _write_json(best_summary_payload, summary_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(history_path, index=False)

    if best_summary_payload is None:
        raise RuntimeError("Nenhum checkpoint foi salvo durante o treinamento.")

    return TrainingRunSummary(
        model_name=model_name,
        model_alias=model_alias,
        best_epoch=int(best_summary_payload["best_epoch"]),
        best_val_mae=float(best_summary_payload["best_val_mae"]),
        best_val_persistence_mae=float(best_summary_payload["best_val_persistence_mae"]),
        best_ratio_to_persistence=float(best_summary_payload["best_ratio_to_persistence"]),
        best_val_loss=float(best_summary_payload["best_val_loss"]),
        features_where_persistence_wins=best_features_where_persistence_wins,
        epochs_ran=int(len(history_df)),
        model_path=str(model_path),
        model_config_path=str(model_config_path),
        history_path=str(history_path),
        summary_path=str(summary_path),
    )


def train_comparative_models(
    *,
    bundle: PreprocessingBundle,
    train_groups: list[dict[str, Any]],
    validation_groups: list[dict[str, Any]],
    output_dir: str | Path,
    model_names: list[str] | None = None,
    **train_kwargs: Any,
) -> tuple[dict[str, TrainingRunSummary], pd.DataFrame]:
    require_tabular_stack()
    model_names = model_names or list(DEFAULT_MODEL_SPECS.keys())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, TrainingRunSummary] = {}
    rows = []
    for model_name in model_names:
        summary = train_single_model(
            model_name=model_name,
            bundle=bundle,
            train_groups=train_groups,
            validation_groups=validation_groups,
            output_dir=output_dir / DEFAULT_MODEL_SPECS[model_name]["alias"],
            **train_kwargs,
        )
        summaries[model_name] = summary
        rows.append(
            {
                "model_name": summary.model_name,
                "model_alias": summary.model_alias,
                "best_epoch": summary.best_epoch,
                "best_val_mae": summary.best_val_mae,
                "best_val_persistence_mae": summary.best_val_persistence_mae,
                "best_ratio_to_persistence": summary.best_ratio_to_persistence,
                "best_val_loss": summary.best_val_loss,
                "features_where_persistence_wins": summary.features_where_persistence_wins,
                "epochs_ran": summary.epochs_ran,
                "model_config_path": summary.model_config_path,
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("best_ratio_to_persistence").reset_index(drop=True)
    summary_df.to_csv(output_dir / "comparativo_validacao.csv", index=False)
    return summaries, summary_df


def evaluate_saved_model(
    *,
    model_config_path: str | Path,
    bundle: PreprocessingBundle,
    test_groups: list[dict[str, Any]],
    output_dir: str | Path,
    sequence_length: int | None = None,
    forecast_horizon: int | None = None,
    batch_size: int = 256,
    num_workers: int = 0,
    preview_rows: int = 2_048,
    progress_every: int = 25,
    log_memory: bool = False,
    device: str | torch.device | None = None,
    use_amp: bool | None = None,
) -> EvaluationRunSummary:
    require_torch()
    require_tabular_stack()

    device_obj = _default_device(device)
    if use_amp is None:
        use_amp = device_obj.type == "cuda"

    model_config_path = Path(model_config_path)
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    model_name = str(model_config["model_name"])
    model_alias = DEFAULT_MODEL_SPECS[model_name]["alias"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequence_length = int(sequence_length or bundle.sequence_length_recommendation)
    forecast_horizon = int(forecast_horizon or model_config.get("forecast_horizon", bundle.forecast_horizon_recommendation))

    _, test_loader = _build_dataloader(
        test_groups,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        batch_size=batch_size,
        sampled_windows=None,
        seed=42,
        balance_by_class=False,
        num_workers=num_workers,
    )

    model = load_model_from_config(
        model_config=model_config,
        well_count=max(len(bundle.well_to_id), 1),
        device=device_obj,
    )
    result = predict_loader_streaming(
        model=model,
        loader=test_loader,
        device=device_obj,
        raw_target_positions=model_config["raw_target_positions"],
        groups=test_groups,
        bundle=bundle,
        collect_metrics=True,
        export_predictions=False,
        preview_rows=preview_rows,
        progress_every=progress_every,
        log_memory=log_memory,
        model_label=model_alias,
        baseline_label="persistencia",
        use_amp=bool(use_amp),
    )
    exported_tables = export_streaming_result_tables(
        result,
        output_dir=output_dir,
        prefix=f"teste_{model_alias}",
    )
    return EvaluationRunSummary(
        model_name=model_name,
        model_alias=model_alias,
        model_config_path=str(model_config_path),
        exported_table_paths=exported_tables,
        result=result,
    )


def _merge_model_metric_tables(
    evaluations: dict[str, EvaluationRunSummary],
    *,
    attr_name: str,
    key_column: str,
    model_value_map: dict[str, str],
    baseline_value_map: dict[str, str],
) -> pd.DataFrame:
    require_tabular_stack()
    comparison_df: pd.DataFrame | None = None
    baseline_df: pd.DataFrame | None = None

    for evaluation in evaluations.values():
        source_df = getattr(evaluation.result, attr_name)
        if source_df is None or source_df.empty:
            continue

        selected_model = source_df[[key_column] + list(model_value_map.keys())].copy()
        selected_model = selected_model.rename(
            columns={
                metric_name: alias_template.format(alias=evaluation.model_alias)
                for metric_name, alias_template in model_value_map.items()
            }
        )
        comparison_df = selected_model if comparison_df is None else comparison_df.merge(selected_model, on=key_column, how="outer")

        if baseline_df is None:
            baseline_df = source_df[[key_column] + list(baseline_value_map.keys())].copy()
            baseline_df = baseline_df.rename(columns=baseline_value_map)

    if comparison_df is None:
        return pd.DataFrame()
    if baseline_df is not None:
        comparison_df = comparison_df.merge(baseline_df, on=key_column, how="left")
    return comparison_df.sort_values(key_column).reset_index(drop=True)


def build_global_comparison_tables(
    evaluations: dict[str, EvaluationRunSummary],
) -> dict[str, pd.DataFrame]:
    require_tabular_stack()
    tables: dict[str, pd.DataFrame] = {}
    for scale_name, attr_name in [
        ("scaled", "global_metrics_scaled_df"),
        ("original", "global_metrics_original_df"),
    ]:
        rows = []
        baseline_row = None
        for evaluation in evaluations.values():
            source_df = getattr(evaluation.result, attr_name)
            if source_df is None or source_df.empty:
                continue
            model_row = source_df.iloc[0].to_dict()
            model_row["modelo"] = evaluation.model_alias
            rows.append(model_row)
            if baseline_row is None and len(source_df) > 1:
                baseline_row = source_df.iloc[1].to_dict()
                baseline_row["modelo"] = "persistencia"
        if baseline_row is not None:
            rows.append(baseline_row)
        if rows:
            tables[f"global_{scale_name}"] = (
                pd.DataFrame(rows)
                .sort_values(["modelo"])
                .reset_index(drop=True)
            )
        else:
            tables[f"global_{scale_name}"] = pd.DataFrame()
    return tables


def build_feature_comparison_tables(
    evaluations: dict[str, EvaluationRunSummary],
) -> dict[str, pd.DataFrame]:
    require_tabular_stack()
    return {
        "per_feature_scaled": _merge_model_metric_tables(
            evaluations,
            attr_name="per_feature_scaled_df",
            key_column="feature",
            model_value_map={
                "model_mae": "mae__{alias}",
                "model_rmse": "rmse__{alias}",
                "model_r2": "r2__{alias}",
            },
            baseline_value_map={
                "baseline_mae": "mae__persistencia",
                "baseline_rmse": "rmse__persistencia",
                "baseline_r2": "r2__persistencia",
            },
        ),
        "per_feature_original": _merge_model_metric_tables(
            evaluations,
            attr_name="per_feature_original_df",
            key_column="feature",
            model_value_map={
                "model_mae": "mae__{alias}",
                "model_rmse": "rmse__{alias}",
                "model_r2": "r2__{alias}",
            },
            baseline_value_map={
                "baseline_mae": "mae__persistencia",
                "baseline_rmse": "rmse__persistencia",
                "baseline_r2": "r2__persistencia",
            },
        ),
    }


def build_slice_comparison_tables(
    evaluations: dict[str, EvaluationRunSummary],
) -> dict[str, pd.DataFrame]:
    require_tabular_stack()
    slices = {
        "class_scaled": ("class_metrics_scaled_df", "class_label"),
        "class_original": ("class_metrics_original_df", "class_label"),
        "well_scaled": ("well_metrics_scaled_df", "well_name"),
        "well_original": ("well_metrics_original_df", "well_name"),
        "horizon_scaled": ("horizon_metrics_scaled_df", "horizon_step"),
        "horizon_original": ("horizon_metrics_original_df", "horizon_step"),
    }
    tables = {}
    for table_name, (attr_name, key_column) in slices.items():
        tables[table_name] = _merge_model_metric_tables(
            evaluations,
            attr_name=attr_name,
            key_column=key_column,
            model_value_map={
                "model_MAE": "mae__{alias}",
                "model_RMSE": "rmse__{alias}",
                "model_R2": "r2__{alias}",
            },
            baseline_value_map={
                "baseline_MAE": "mae__persistencia",
                "baseline_RMSE": "rmse__persistencia",
                "baseline_R2": "r2__persistencia",
            },
        )
    return tables


def evaluate_comparative_models(
    *,
    bundle: PreprocessingBundle,
    test_groups: list[dict[str, Any]],
    model_config_paths: list[str | Path],
    output_dir: str | Path,
    **evaluation_kwargs: Any,
) -> tuple[dict[str, EvaluationRunSummary], dict[str, pd.DataFrame]]:
    require_tabular_stack()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_config_paths:
        raise ValueError(
            "Nenhum arquivo de configuracao de modelo foi encontrado para avaliacao. "
            "Execute primeiro o notebook de modelagem comparativa para gerar os checkpoints."
        )

    evaluations: dict[str, EvaluationRunSummary] = {}
    for config_path in model_config_paths:
        evaluation = evaluate_saved_model(
            model_config_path=config_path,
            bundle=bundle,
            test_groups=test_groups,
            output_dir=output_dir / Path(config_path).stem,
            **evaluation_kwargs,
        )
        evaluations[evaluation.model_alias] = evaluation

    comparison_tables = {}
    comparison_tables.update(build_global_comparison_tables(evaluations))
    comparison_tables.update(build_feature_comparison_tables(evaluations))
    comparison_tables.update(build_slice_comparison_tables(evaluations))

    for table_name, df in comparison_tables.items():
        if df is None or df.empty:
            continue
        df.to_csv(output_dir / f"{table_name}.csv", index=False)

    return evaluations, comparison_tables
