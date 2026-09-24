"""Settings of a run: the model, the training schedule, the cohort and the data locations.

Model, training and cohort settings are read from ``config.yaml`` next to this
file. Data locations come from the command line, so the same settings run on a
workstation and on a hosted notebook.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml

CONFIG_FILE = Path(__file__).with_name("config.yaml")
REPOSITORY = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPOSITORY / "data" / "processed"
DEFAULT_RESULTS_DIR = REPOSITORY / "results"

@dataclass(frozen=True)
class DataPaths:
    """Locations of the raw datasets, the preprocessed cohorts and the run outputs."""

    processed_root: Path
    results_root: Path
    mimic3_raw: Path | None = None
    mimic4_raw: Path | None = None
    drug_reference: Path | None = None

    def cohort_dir(self, cohort_directory: str) -> Path:
        """Return the directory holding one cohort's preprocessed files."""
        return self.processed_root / cohort_directory


@dataclass(frozen=True)
class CohortConfig:
    """Identity of one patient cohort and the folder its files live in."""

    name: str
    description: str
    directory: str
    source: str
    diagnosis_revisions: tuple[str, ...] = ("icd9",)
    admission_order: str = "time"
    minimum_admissions: int = 2


@dataclass(frozen=True)
class ModelConfig:
    """Sizes and switches of the network.

    The four ``use_*`` switches select the input channels; turning them off
    reproduces the ablations reported in the paper.
    """

    hidden_dim: int = 128
    code_embedding_dim: int = 768
    note_embedding_dim: int = 768
    encoder_layers: int = 2
    attention_heads: int = 4
    note_projection_dim: int = 64
    lab_projection_dim: int | None = None
    graph_layers: int = 2
    morgan_bits: int = 256
    morgan_radius: int = 2
    relation_count: int = 4
    max_visits: int = 30
    dropout: float = 0.3
    gradient_clip: float = 1.0
    selection_temperature: float = 0.6
    attention_temperature: float = 20.0
    use_notes: bool = True
    use_labs: bool = True
    use_copy_head: bool = True
    use_visit_selection: bool = True
    use_graph_relations: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    """Optimiser, schedule, loss weights and the decision threshold."""

    epochs: int = 120
    batch_size: int = 64
    learning_rate: float = 5.0e-4
    min_learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-5
    lr_patience: int = 8
    lr_factor: float = 0.5
    early_stopping_patience: int = 20
    bce_weight: float = 0.3
    jaccard_weight: float = 1.5
    margin_weight: float = 0.05
    interaction_weight: float = 0.2
    positive_weight_cap: float = 5.0
    positive_weight_floor: float = 0.1
    decision_threshold: float = 0.5
    dataloader_workers: int = 0
    seeds: tuple[int, ...] = (42, 123, 456, 789, 1024)


@dataclass(frozen=True)
class RunConfig:
    """Everything one training run needs."""

    cohort: CohortConfig
    model: ModelConfig
    training: TrainingConfig
    paths: DataPaths
    seed: int = 42
    device: str = "cuda"
    run_name: str = "full"
    lab_count: int = 200

    @property
    def lab_vector_dim(self) -> int:
        """Width of the laboratory vector: one z-score and one flag per test."""
        return 2 * self.lab_count

def _read_config() -> dict[str, Any]:
    """Read ``config.yaml``."""
    with open(CONFIG_FILE, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _build(cls: type, values: dict[str, Any]) -> Any:
    """Instantiate a dataclass from a dictionary, rejecting unknown keys."""
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown settings {sorted(unknown)}")
    return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in values.items()})


def make_paths(
    data_dir: Path | None = None,
    results_dir: Path | None = None,
    mimic_dir: Path | None = None,
    drug_reference_dir: Path | None = None,
    source: str = "mimic3",
) -> DataPaths:
    """Resolve the data locations given on the command line, with the repository defaults."""

    def resolve(path: Path | None) -> Path | None:
        return None if path is None else Path(path).expanduser().resolve()

    raw = resolve(mimic_dir)
    return DataPaths(
        processed_root=resolve(data_dir) or DEFAULT_DATA_DIR,
        results_root=resolve(results_dir) or DEFAULT_RESULTS_DIR,
        mimic3_raw=raw if source == "mimic3" else None,
        mimic4_raw=raw if source == "mimic4" else None,
        drug_reference=resolve(drug_reference_dir),
    )


def load_cohort(cohort_name: str) -> CohortConfig:
    """Read one cohort definition from the ``cohorts`` section of ``config.yaml``."""
    cohorts = _read_config().get("cohorts", {})
    if cohort_name not in cohorts:
        raise ValueError(f"Unknown cohort '{cohort_name}'. Available: {sorted(cohorts)}")
    return _build(CohortConfig, {"name": cohort_name, **cohorts[cohort_name]})


def load_run_config(
    cohort_name: str,
    paths: DataPaths,
    model_overrides: dict[str, Any] | None = None,
    training_overrides: dict[str, Any] | None = None,
    **run_overrides: Any,
) -> RunConfig:
    """Assemble a :class:`RunConfig` from ``config.yaml`` plus command-line overrides.

    Overrides given as ``None`` are ignored, so a flag that was not passed never
    replaces a configured value.
    """
    settings = _read_config()
    model_values = dict(settings.get("model", {}))
    model_values.update({k: v for k, v in (model_overrides or {}).items() if v is not None})
    training_values = dict(settings.get("training", {}))
    training_values.update({k: v for k, v in (training_overrides or {}).items() if v is not None})
    config = RunConfig(
        cohort=load_cohort(cohort_name),
        model=_build(ModelConfig, model_values),
        training=_build(TrainingConfig, training_values),
        paths=paths,
    )
    clean = {k: v for k, v in run_overrides.items() if v is not None}
    return replace(config, **clean) if clean else config


def config_to_dict(config: RunConfig) -> dict[str, Any]:
    """Flatten a run configuration into JSON-serialisable form for the result file."""

    def unpack(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return {f.name: unpack(getattr(obj, f.name)) for f in fields(obj)}
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, tuple):
            return list(obj)
        return obj

    return unpack(config)
