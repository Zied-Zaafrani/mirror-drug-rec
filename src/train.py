"""Training and evaluation of one configuration, and the result file it writes.

    python src/train.py --cohort mimic3 --seed 42
    python src/train.py --cohort mimic3 --all_seeds
    python src/train.py --cohort mimic3 --no_notes --no_labs --no_copy_head   # an ablation

Each run trains on the training partition, keeps the epoch with the best
validation Jaccard, evaluates it on the test partition and writes one JSON
result file under ``results/<cohort>/<configuration>/``.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from config import (
    ModelConfig,
    RunConfig,
    TrainingConfig,
    config_to_dict,
    load_run_config,
    make_paths,
)
from dataset import (
    DRUG_POSITION,
    CohortArtifacts,
    DrugGraph,
    PatientSplit,
    PrescriptionDataset,
    build_drug_graph,
    collate_instances,
    coprescription_probabilities,
    load_cohort_artifacts,
    select,
    split_patients,
)
from metrics import (
    count_matched_threshold,
    evaluate_baseline,
    evaluate_recommendations,
    frequency_tiers,
    overlap_by_tier,
    prescription_frequency,
    recommend,
    sweep_thresholds,
)
from model import Mirror

EPSILON = 1e-8


def positive_class_weights(
    records: list,
    drug_count: int,
    floor: float = 0.1,
    cap: float = 5.0,
) -> torch.Tensor:
    """Weight each drug class by how rarely it is prescribed.

    The weight is the ratio of admissions without the class to admissions with
    it, clipped so that a class prescribed a handful of times cannot dominate
    the gradient.

    Args:
        records: the training patients only.
        drug_count: size of the drug vocabulary.
        floor: smallest weight allowed.
        cap: largest weight allowed.

    Returns:
        ``(drug_count,)`` weights for the cross-entropy term.
    """
    prescribed = np.zeros(drug_count, dtype=np.float64)
    admissions = 0
    for patient in records:
        for admission in patient:
            for drug in admission[DRUG_POSITION]:
                if drug < drug_count:
                    prescribed[drug] += 1
            admissions += 1
    weights = (admissions - prescribed) / np.maximum(prescribed, 1.0)
    return torch.tensor(np.clip(weights, floor, cap), dtype=torch.float32)


class RecommendationLoss(nn.Module):
    """Weighted sum of the four training terms.

    Args:
        config: supplies the four weights and the weight clipping range.
        interaction_matrix: ``(drugs, drugs)``, non-zero for interacting pairs.
        class_weights: per-class weights from :func:`positive_class_weights`.
    """

    def __init__(
        self,
        config: TrainingConfig,
        interaction_matrix: torch.Tensor,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.bce_weight = config.bce_weight
        self.jaccard_weight = config.jaccard_weight
        self.margin_weight = config.margin_weight
        self.interaction_weight = config.interaction_weight
        self.register_buffer("interaction_matrix", interaction_matrix)
        if class_weights is None:
            self.class_weights = None
        else:
            self.register_buffer("class_weights", class_weights)

    def forward(self, scores: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Score one batch.

        Args:
            scores: ``(batch, drugs)`` model output before the sigmoid.
            target: ``(batch, drugs)`` binary prescriptions of the predicted
                admission.

        Returns:
            The scalar loss and a dictionary of the individual terms, for logging.
        """
        terms: dict[str, torch.Tensor] = {}
        terms["cross_entropy"] = F.binary_cross_entropy_with_logits(
            scores, target, pos_weight=self.class_weights
        )
        terms["overlap"] = self._overlap(scores, target)
        terms["margin"] = F.multilabel_soft_margin_loss(scores, target)
        terms["interaction"] = self._interaction(scores)

        total = (
            self.bce_weight * terms["cross_entropy"]
            + self.jaccard_weight * terms["overlap"]
            + self.margin_weight * terms["margin"]
            + self.interaction_weight * terms["interaction"]
        )
        reported = {name: float(value.item()) for name, value in terms.items()}
        reported["total"] = float(total.item())
        return total, reported

    @staticmethod
    def _overlap(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """One minus the overlap between the predicted and the recorded set.

        Probabilities stand in for set membership, which makes the measure
        differentiable.
        """
        probabilities = torch.sigmoid(scores)
        intersection = (probabilities * target).sum(dim=1)
        union = probabilities.sum(dim=1) + target.sum(dim=1) - intersection
        return (1.0 - intersection / (union + EPSILON)).mean()

    def _interaction(self, scores: torch.Tensor) -> torch.Tensor:
        """Expected number of interacting pairs among the recommended classes.

        Both classes of a pair are counted through their probabilities, and the
        sum is divided by the vocabulary size so the term keeps the same scale
        across cohorts.
        """
        probabilities = torch.sigmoid(scores)
        paired = (probabilities @ self.interaction_matrix) * probabilities
        return (paired.sum(dim=1) / probabilities.size(1)).mean()


SCHEMA_VERSION = 1
RESULT_PREFIX = "result_"


@dataclass
class RunResult:
    """Everything one training run reports.

    Attributes:
        cohort: name of the cohort the run used.
        variant: which input channels were active, for example ``full`` or
            ``codes_only``.
        seed: seed of the run.
        metrics: test-set metrics.
        validation_jaccard: best overlap reached on the validation set.
        baseline: metrics of the copy-forward baseline on the same test set.
        threshold_sweep: metrics at several decision thresholds.
        tier_metrics: overlap per drug-frequency tier.
        partition_sizes: patients and prediction instances per partition.
        parameter_counts: parameters per part of the model.
        timings: seconds spent training and predicting.
        config: the full run configuration.
        environment: interpreter, library and device description.
    """

    cohort: str
    variant: str
    seed: int
    metrics: dict[str, float]
    validation_jaccard: float = 0.0
    baseline: dict[str, float] = field(default_factory=dict)
    threshold_sweep: dict[str, dict[str, float]] = field(default_factory=dict)
    tier_metrics: dict[str, float] = field(default_factory=dict)
    partition_sizes: dict[str, dict[str, int]] = field(default_factory=dict)
    parameter_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    written_at: str = ""

    def file_name(self) -> str:
        """Name of the file this result is written to."""
        return f"{RESULT_PREFIX}{self.cohort}_{self.variant}_seed{self.seed}.json"


def describe_environment(device: str) -> dict[str, str]:
    """Record the interpreter, the tensor library and the device of a run."""
    import torch

    description = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": device,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        description["gpu"] = torch.cuda.get_device_name(0)
    return description


def write_result(result: RunResult, directory: Path) -> Path:
    """Write one result to ``directory`` and return the path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result.written_at = datetime.now(UTC).isoformat(timespec="seconds")
    path = directory / result.file_name()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(asdict(result), handle, indent=2)
    return path


def read_result(path: Path) -> RunResult:
    """Read one result file.

    Raises:
        ValueError: the file was written by a different schema version.
    """
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{Path(path).name} declares schema version {version}, this code reads "
            f"version {SCHEMA_VERSION}."
        )
    known = {f for f in RunResult.__dataclass_fields__}
    return RunResult(**{key: value for key, value in payload.items() if key in known})


TENSOR_KEYS = (
    "history_length",
    "drugs_per_visit",
    "drug_history",
    "previous_drugs",
    "target",
    "note_vector",
    "has_note",
    "lab_vector",
    "has_labs",
)
LIST_KEYS = ("diagnosis_codes", "procedure_codes", "diagnosis_mask", "procedure_mask")


@dataclass
class Predictions:
    """Model output for one partition, in the shape the metrics expect."""

    recorded: np.ndarray
    probabilities: np.ndarray
    previous: np.ndarray
    admission_ids: np.ndarray


def move_to_device(batch: dict, device: torch.device) -> dict:
    """Copy one batch onto ``device``, keeping the per-admission lists intact."""
    moved = dict(batch)
    for key in TENSOR_KEYS:
        moved[key] = batch[key].to(device)
    for key in LIST_KEYS:
        moved[key] = [tensor.to(device) for tensor in batch[key]]
    return moved


def train_one_epoch(
    model: Mirror,
    loader: DataLoader,
    loss_function: RecommendationLoss,
    optimiser: torch.optim.Optimizer,
    graph: DrugGraph,
    device: torch.device,
    gradient_clip: float,
) -> dict[str, float]:
    """Run one pass over the training set and return the mean of each loss term."""
    model.train()
    totals: dict[str, float] = {}
    batches = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        optimiser.zero_grad()
        scores, _ = model(batch, graph)
        loss, terms = loss_function(scores, batch["target"])
        loss.backward()
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimiser.step()
        for name, value in terms.items():
            totals[name] = totals.get(name, 0.0) + value
        batches += 1
    return {name: value / max(batches, 1) for name, value in totals.items()}


@torch.no_grad()
def predict(
    model: Mirror,
    loader: DataLoader,
    graph: DrugGraph,
    device: torch.device,
) -> Predictions:
    """Score every prediction instance in a partition."""
    model.eval()
    recorded, probabilities, previous, admission_ids = [], [], [], []
    for batch in loader:
        moved = move_to_device(batch, device)
        scores, _ = model(moved, graph)
        probabilities.append(torch.sigmoid(scores).cpu().numpy())
        recorded.append(batch["target"].numpy())
        previous.append(batch["previous_drugs"].numpy())
        admission_ids.append(batch["admission_id"].numpy())
    return Predictions(
        recorded=np.concatenate(recorded),
        probabilities=np.concatenate(probabilities),
        previous=np.concatenate(previous),
        admission_ids=np.concatenate(admission_ids),
    )


def score(
    predictions: Predictions,
    interaction_matrix: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute the reported metrics from a partition's predictions."""
    return evaluate_recommendations(
        recorded=predictions.recorded,
        recommended=recommend(predictions.probabilities, threshold),
        probabilities=predictions.probabilities,
        interaction_matrix=interaction_matrix,
        previous=predictions.previous,
    )


def _loaders(
    artifacts: CohortArtifacts,
    split: PatientSplit,
    config: RunConfig,
) -> tuple[dict[str, DataLoader], dict[str, dict[str, int]]]:
    """Build one loader per partition and report how large each one is."""
    loaders: dict[str, DataLoader] = {}
    sizes: dict[str, dict[str, int]] = {}
    for name, indices in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        dataset = PrescriptionDataset(
            records=select(artifacts.records, indices),
            artifacts=artifacts,
            use_notes=config.model.use_notes,
            use_labs=config.model.use_labs,
        )
        loaders[name] = DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=name == "train",
            collate_fn=collate_instances,
            num_workers=config.training.dataloader_workers,
        )
        sizes[name] = {"patients": len(indices), "instances": len(dataset)}
    return loaders, sizes


def run(
    config: RunConfig,
    report: Callable[[str], None] = print,
    save_model_to: Path | None = None,
) -> RunResult:
    """Train and evaluate one configuration, and return its result.

    Args:
        config: the full run configuration.
        report: where progress lines go; pass a no-op to stay silent.
        save_model_to: write the best epoch's weights to this path when given.

    Returns:
        The :class:`RunResult` for this run, already holding every reported
        number. Writing it to disk is the caller's decision.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device)

    cohort_dir = config.paths.cohort_dir(config.cohort.directory)
    report(f"Cohort {config.cohort.name} from {cohort_dir}")
    artifacts = load_cohort_artifacts(
        cohort_dir,
        use_notes=config.model.use_notes,
        use_labs=config.model.use_labs,
        expected_lab_dim=config.lab_vector_dim if config.model.use_labs else None,
    )
    report(
        f"{len(artifacts.records)} patients, {artifacts.admission_count()} admissions, "
        f"{artifacts.drug_count} drug classes"
    )

    split = split_patients(len(artifacts.records))
    train_records = select(artifacts.records, split.train)
    loaders, partition_sizes = _loaders(artifacts, split, config)
    report(
        "Partitions: "
        + ", ".join(
            f"{name} {size['patients']} patients / {size['instances']} instances"
            for name, size in partition_sizes.items()
        )
    )

    graph = build_drug_graph(
        interaction_matrix=artifacts.interaction_matrix,
        coprescription_matrix=artifacts.coprescription_matrix,
        drug_codes=artifacts.drug_codes,
        coprescription_weights=coprescription_probabilities(train_records, artifacts.drug_count),
        use_relations=config.model.use_graph_relations,
    ).to(device)
    report(f"Drug graph: {graph.edge_count} edges ({graph.counts_per_relation()})")

    model = Mirror(artifacts, config.model).to(device)
    counts = model.parameter_counts()
    report(
        f"Parameters: {counts['total']['trainable']:,} trainable "
        f"of {counts['total']['total']:,}"
    )

    loss_function = RecommendationLoss(
        config=config.training,
        interaction_matrix=torch.from_numpy(artifacts.interaction_matrix).float(),
        class_weights=positive_class_weights(
            train_records,
            artifacts.drug_count,
            floor=config.training.positive_weight_floor,
            cap=config.training.positive_weight_cap,
        ),
    ).to(device)
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode="max",
        factor=config.training.lr_factor,
        patience=config.training.lr_patience,
        min_lr=config.training.min_learning_rate,
    )

    best_validation = -1.0
    best_weights = copy.deepcopy(model.state_dict())
    epochs_without_gain = 0
    epochs_run = 0
    training_seconds = 0.0

    for epoch in range(1, config.training.epochs + 1):
        started = time.perf_counter()
        losses = train_one_epoch(
            model,
            loaders["train"],
            loss_function,
            optimiser,
            graph,
            device,
            config.model.gradient_clip,
        )
        validation = score(
            predict(model, loaders["validation"], graph, device),
            artifacts.interaction_matrix,
            config.training.decision_threshold,
        )
        scheduler.step(validation["jaccard"])
        training_seconds += time.perf_counter() - started
        epochs_run = epoch
        report(
            f"epoch {epoch:3d}  loss {losses['total']:.4f}  "
            f"validation jaccard {validation['jaccard']:.4f}  "
            f"f1 {validation['f1']:.4f}  interactions {validation['interaction_rate']:.4f}"
        )

        if validation["jaccard"] > best_validation:
            best_validation = validation["jaccard"]
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= config.training.early_stopping_patience:
                report(f"No validation gain for {epochs_without_gain} epochs, stopping")
                break

    model.load_state_dict(best_weights)
    if save_model_to is not None:
        Path(save_model_to).parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_weights, save_model_to)
        report(f"Model weights written to {save_model_to}")

    started = time.perf_counter()
    test_predictions = predict(model, loaders["test"], graph, device)
    prediction_seconds = time.perf_counter() - started
    metrics = score(
        test_predictions, artifacts.interaction_matrix, config.training.decision_threshold
    )
    recommended = recommend(test_predictions.probabilities, config.training.decision_threshold)

    tiers = frequency_tiers(prescription_frequency(train_records, artifacts.drug_count))
    instances = max(len(test_predictions.recorded), 1)
    result = RunResult(
        cohort=config.cohort.name,
        variant=config.run_name,
        seed=config.seed,
        metrics=metrics,
        validation_jaccard=best_validation,
        baseline=evaluate_baseline(
            select(artifacts.records, split.test),
            artifacts.drug_count,
            artifacts.interaction_matrix,
        ),
        threshold_sweep={
            f"{threshold:.2f}": values
            for threshold, values in sweep_thresholds(
                test_predictions.recorded,
                test_predictions.probabilities,
                artifacts.interaction_matrix,
            ).items()
        },
        tier_metrics=overlap_by_tier(test_predictions.recorded, recommended, tiers),
        partition_sizes=partition_sizes,
        parameter_counts=counts,
        timings={
            "training_seconds": round(training_seconds, 2),
            "seconds_per_epoch": round(training_seconds / max(epochs_run, 1), 2),
            "epochs_run": epochs_run,
            "prediction_seconds": round(prediction_seconds, 3),
            "milliseconds_per_instance": round(prediction_seconds / instances * 1000, 3),
        },
        config=config_to_dict(config),
        environment=describe_environment(config.device),
    )
    result.metrics["count_matched_threshold"] = count_matched_threshold(
        test_predictions.recorded, test_predictions.probabilities
    )
    report(
        f"Test jaccard {metrics['jaccard']:.4f}  f1 {metrics['f1']:.4f}  "
        f"average precision {metrics['average_precision']:.4f}  "
        f"interactions {metrics['interaction_rate']:.4f}  "
        f"started {metrics['change_started']:.4f}  stopped {metrics['change_stopped']:.4f}"
    )
    return result


def variant_name(model: ModelConfig) -> str:
    """Name a configuration after the input channels it uses, for the result file."""
    active = [
        name
        for name, enabled in (
            ("notes", model.use_notes),
            ("labs", model.use_labs),
            ("copy", model.use_copy_head),
        )
        if enabled
    ]
    base = "full" if len(active) == 3 else ("_".join(active) if active else "codes_only")
    if not model.use_graph_relations:
        base += "_no_graph"
    if not model.use_visit_selection:
        base += "_no_selection"
    return base


def main(argv: list[str] | None = None) -> int:
    """Train one configuration and write its result file."""
    parser = argparse.ArgumentParser(description="Train MIRROR on one cohort.")
    parser.add_argument("--cohort", default="mimic3", help="a cohort listed in config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all_seeds", action="store_true",
                        help="run the five seeds listed in config.yaml")
    parser.add_argument("--device", default="cuda", help="cuda, cpu or cuda:N")
    parser.add_argument("--data_dir", type=Path, default=None,
                        help="folder holding one sub-folder per cohort (default: data/processed)")
    parser.add_argument("--results_dir", type=Path, default=None,
                        help="where the result files go (default: results)")
    parser.add_argument("--variant", default=None, help="name of the configuration")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--interaction_weight", type=float, default=None,
                        help="weight of the drug-interaction penalty (0.2 in the paper)")
    parser.add_argument("--lab_count", type=int, default=200)
    parser.add_argument("--save_model", type=Path, default=None,
                        help="write the weights of the best epoch to this file")
    parser.add_argument("--no_notes", action="store_true", help="ablation: drop the discharge note")
    parser.add_argument("--no_labs", action="store_true",
                        help="ablation: drop the laboratory values")
    parser.add_argument("--no_copy_head", action="store_true", help="ablation: drop the copy head")
    parser.add_argument("--no_visit_selection", action="store_true",
                        help="ablation: use the most recent admission instead of a selection")
    parser.add_argument("--self_loops_only", action="store_true",
                        help="ablation: drug graph without neighbour edges")
    arguments = parser.parse_args(argv)

    config = load_run_config(
        arguments.cohort,
        paths=make_paths(arguments.data_dir, arguments.results_dir),
        model_overrides={
            "use_notes": False if arguments.no_notes else None,
            "use_labs": False if arguments.no_labs else None,
            "use_copy_head": False if arguments.no_copy_head else None,
            "use_visit_selection": False if arguments.no_visit_selection else None,
            "use_graph_relations": False if arguments.self_loops_only else None,
        },
        training_overrides={
            "epochs": arguments.epochs,
            "batch_size": arguments.batch_size,
            "learning_rate": arguments.learning_rate,
            "interaction_weight": arguments.interaction_weight,
        },
        seed=arguments.seed,
        device=arguments.device,
        lab_count=arguments.lab_count,
    )
    config = replace(config, run_name=arguments.variant or variant_name(config.model))
    seeds = list(config.training.seeds) if arguments.all_seeds else [arguments.seed]
    for seed in seeds:
        seeded = replace(config, seed=seed)
        print(f"=== {seeded.cohort.name} / {seeded.run_name} / seed {seed} ===")
        result = run(seeded, save_model_to=arguments.save_model)
        destination = seeded.paths.results_root / seeded.cohort.name / seeded.run_name
        path = write_result(result, destination)
        print(f"Result written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
