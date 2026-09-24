"""Evaluation: overlap, F1, PRAUC, DDI rate, the change-aware measures and the baseline.

The change-aware measures compare the drugs that were started or stopped since
the previous admission with the ones the model started or stopped. The
copy-forward baseline repeats the previous admission's prescription.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

from dataset import DRUG_POSITION

EPSILON = 1e-8


def overlap(recorded: np.ndarray, recommended: np.ndarray) -> float:
    """Mean intersection over union between the recommended and recorded sets."""
    intersection = np.sum(recorded * recommended, axis=1)
    union = np.sum(np.clip(recorded + recommended, 0, 1), axis=1)
    return float(np.mean(intersection / np.maximum(union, EPSILON)))


def precision_recall(recorded: np.ndarray, recommended: np.ndarray) -> tuple[float, float]:
    """Mean precision and mean recall over admissions."""
    true_positives = np.sum(recorded * recommended, axis=1)
    precision = true_positives / np.maximum(recommended.sum(axis=1), EPSILON)
    recall = true_positives / np.maximum(recorded.sum(axis=1), EPSILON)
    return float(np.mean(precision)), float(np.mean(recall))


def f1(recorded: np.ndarray, recommended: np.ndarray) -> float:
    """Mean F1 over admissions, counting prescribed classes only.

    Classes correctly left out are not counted. Including them would raise the
    number without saying anything about prescribing.
    """
    precision, recall = (
        np.sum(recorded * recommended, axis=1) / np.maximum(recommended.sum(axis=1), EPSILON),
        np.sum(recorded * recommended, axis=1) / np.maximum(recorded.sum(axis=1), EPSILON),
    )
    return float(np.mean(2 * precision * recall / np.maximum(precision + recall, EPSILON)))


def average_precision(recorded: np.ndarray, probabilities: np.ndarray) -> float:
    """Mean area under the precision-recall curve over admissions.

    Admissions with no recorded prescription are skipped, the curve being
    undefined for them.
    """
    scores = [
        average_precision_score(row_recorded, row_probabilities)
        for row_recorded, row_probabilities in zip(recorded, probabilities, strict=True)
        if row_recorded.sum() > 0
    ]
    return float(np.mean(scores)) if scores else 0.0


def interaction_rate(recommended: np.ndarray, interaction_matrix: np.ndarray) -> float:
    """Share of recommended drug pairs that are known to interact.

    Pairs are counted over the whole set of admissions rather than per
    admission, so admissions recommending more drugs weigh more, as they should:
    they carry more pairs. An admission recommending one class carries no pair
    and so contributes nothing either way.

    Both counts are quadratic forms of each row, which is why this reads as
    matrix arithmetic rather than a loop over pairs: for a binary row the number
    of interacting pairs it holds is half of ``row @ matrix @ row``.
    """
    pairs = np.asarray(interaction_matrix, dtype=np.float64).copy()
    np.fill_diagonal(pairs, 0.0)
    rows = np.asarray(recommended, dtype=np.float64)
    interacting = float(((rows @ pairs) * rows).sum()) / 2.0
    sizes = rows.sum(axis=1)
    total = float((sizes * (sizes - 1.0) / 2.0).sum())
    return interacting / max(total, EPSILON)


def change_overlap(
    recorded: np.ndarray,
    recommended: np.ndarray,
    previous: np.ndarray,
) -> tuple[float, float]:
    """Overlap restricted to the drugs that changed since the previous admission.

    Args:
        recorded: ``(admissions, drugs)`` what was prescribed.
        recommended: ``(admissions, drugs)`` what the model recommended.
        previous: ``(admissions, drugs)`` what the previous admission prescribed.

    Returns:
        The score on started drugs and the score on stopped drugs. Each average
        covers only the admissions where that kind of change is present in the
        record or in the recommendation; an admission that changed nothing and
        was predicted to change nothing is not evidence either way.
    """
    started_recorded = np.clip(recorded - previous, 0, 1)
    started_recommended = np.clip(recommended - previous, 0, 1)
    stopped_recorded = np.clip(previous - recorded, 0, 1)
    stopped_recommended = np.clip(previous - recommended, 0, 1)

    def restricted(target: np.ndarray, prediction: np.ndarray) -> float:
        union = np.sum(np.clip(target + prediction, 0, 1), axis=1)
        changed = union > 0
        if not np.any(changed):
            return 0.0
        intersection = np.sum(target * prediction, axis=1)[changed]
        return float(np.mean(intersection / union[changed]))

    return (
        restricted(started_recorded, started_recommended),
        restricted(stopped_recorded, stopped_recommended),
    )


def evaluate_recommendations(
    recorded: np.ndarray,
    recommended: np.ndarray,
    probabilities: np.ndarray,
    interaction_matrix: np.ndarray,
    previous: np.ndarray | None = None,
    known_average_precision: float | None = None,
) -> dict[str, float]:
    """Compute every reported metric for one set of recommendations.

    Args:
        recorded: ``(admissions, drugs)`` binary record.
        recommended: ``(admissions, drugs)`` binary recommendation.
        probabilities: ``(admissions, drugs)`` model probabilities.
        interaction_matrix: ``(drugs, drugs)`` interacting pairs.
        previous: ``(admissions, drugs)`` previous admission's prescriptions;
            without it the two change metrics are left out.
        known_average_precision: the value to report instead of computing it.
            Average precision ranks the probabilities and so is the one metric
            here that does not move with the decision threshold; a caller
            sweeping the threshold computes it once and passes it back.

    Returns:
        A dictionary of metric name to value.
    """
    precision, recall = precision_recall(recorded, recommended)
    metrics = {
        "jaccard": overlap(recorded, recommended),
        "f1": f1(recorded, recommended),
        "average_precision": (
            average_precision(recorded, probabilities)
            if known_average_precision is None
            else known_average_precision
        ),
        "interaction_rate": interaction_rate(recommended, interaction_matrix),
        "precision": precision,
        "recall": recall,
        "drugs_recommended": float(np.mean(recommended.sum(axis=1))),
        "drugs_recorded": float(np.mean(recorded.sum(axis=1))),
    }
    if previous is not None:
        started, stopped = change_overlap(recorded, recommended, previous)
        metrics["change_started"] = started
        metrics["change_stopped"] = stopped
    return metrics


DEFAULT_THRESHOLD = 0.5
SWEEP_THRESHOLDS = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)


def recommend(probabilities: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Return the binary recommendation matrix at one threshold."""
    return (probabilities >= threshold).astype(np.float32)


def sweep_thresholds(
    recorded: np.ndarray,
    probabilities: np.ndarray,
    interaction_matrix: np.ndarray,
    thresholds: tuple[float, ...] = SWEEP_THRESHOLDS,
) -> dict[float, dict[str, float]]:
    """Report overlap, set size and interaction rate at several thresholds.

    Args:
        recorded: ``(admissions, drugs)`` binary record.
        probabilities: ``(admissions, drugs)`` model probabilities.
        interaction_matrix: ``(drugs, drugs)`` interacting pairs.
        thresholds: thresholds to try.

    Returns:
        A dictionary from threshold to its metrics.
    """
    results = {}
    for threshold in thresholds:
        recommended = recommend(probabilities, threshold)
        results[threshold] = {
            "jaccard": overlap(recorded, recommended),
            "interaction_rate": interaction_rate(recommended, interaction_matrix),
            "drugs_recommended": float(np.mean(recommended.sum(axis=1))),
        }
    return results


def count_matched_threshold(
    recorded: np.ndarray,
    probabilities: np.ndarray,
    thresholds: tuple[float, ...] = SWEEP_THRESHOLDS,
) -> float:
    """Return the threshold whose recommendation size is closest to the record.

    Reported alongside the main results to show how much of the gap to the
    record is a matter of recommending too many or too few classes.
    """
    target = float(np.mean(recorded.sum(axis=1)))
    sizes = {
        threshold: abs(float(np.mean(recommend(probabilities, threshold).sum(axis=1))) - target)
        for threshold in thresholds
    }
    return min(sizes, key=sizes.get)


RARE_BELOW = 0.10
UNIVERSAL_FROM = 0.40
TIER_NAMES = ("rare", "moderate", "universal")


def prescription_frequency(records: list, drug_count: int) -> np.ndarray:
    """Share of training admissions in which each drug class appears."""
    counts = np.zeros(drug_count, dtype=np.float64)
    admissions = 0
    for patient in records:
        for admission in patient:
            for drug in admission[DRUG_POSITION]:
                if drug < drug_count:
                    counts[drug] += 1
            admissions += 1
    return (counts / max(admissions, 1)).astype(np.float32)


def frequency_tiers(frequency: np.ndarray) -> dict[str, np.ndarray]:
    """Group drug indices into the three frequency tiers."""
    rare = frequency < RARE_BELOW
    universal = frequency >= UNIVERSAL_FROM
    return {
        "rare": np.flatnonzero(rare),
        "moderate": np.flatnonzero(~rare & ~universal),
        "universal": np.flatnonzero(universal),
    }


def overlap_by_tier(
    recorded: np.ndarray,
    recommended: np.ndarray,
    tiers: dict[str, np.ndarray],
) -> dict[str, float]:
    """Overlap inside each tier, plus the number of classes in it.

    An admission with no recorded and no recommended class in a tier is left out
    of that tier's average, since its overlap is undefined.
    """
    results: dict[str, float] = {}
    for name in TIER_NAMES:
        drugs = tiers[name]
        results[f"drugs_{name}"] = float(len(drugs))
        if len(drugs) == 0:
            results[f"jaccard_{name}"] = 0.0
            continue
        tier_recorded = recorded[:, drugs]
        tier_recommended = recommended[:, drugs]
        union = np.sum(np.clip(tier_recorded + tier_recommended, 0, 1), axis=1)
        present = union > 0
        if not np.any(present):
            results[f"jaccard_{name}"] = 0.0
            continue
        intersection = np.sum(tier_recorded * tier_recommended, axis=1)[present]
        results[f"jaccard_{name}"] = float(
            np.mean(intersection / np.maximum(union[present], EPSILON))
        )
    return results


def repeat_previous_admission(records: list, drug_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the baseline's recommendations for one partition.

    Args:
        records: patient records of the partition.
        drug_count: size of the drug vocabulary.

    Returns:
        The ``(admissions, drugs)`` record and the baseline's identical-shaped
        recommendation, one row per admission after a patient's first.
    """
    recorded_rows: list[np.ndarray] = []
    repeated_rows: list[np.ndarray] = []
    for patient in records:
        for position in range(1, len(patient)):
            recorded = np.zeros(drug_count, dtype=np.float32)
            repeated = np.zeros(drug_count, dtype=np.float32)
            for drug in patient[position][DRUG_POSITION]:
                if drug < drug_count:
                    recorded[drug] = 1.0
            for drug in patient[position - 1][DRUG_POSITION]:
                if drug < drug_count:
                    repeated[drug] = 1.0
            recorded_rows.append(recorded)
            repeated_rows.append(repeated)
    return np.stack(recorded_rows), np.stack(repeated_rows)


def evaluate_baseline(
    records: list,
    drug_count: int,
    interaction_matrix: np.ndarray,
) -> dict[str, float]:
    """Score the repeat-previous-admission baseline on one partition."""
    recorded, repeated = repeat_previous_admission(records, drug_count)
    return evaluate_recommendations(
        recorded=recorded,
        recommended=repeated,
        probabilities=repeated,
        interaction_matrix=interaction_matrix,
        previous=repeated,
    )


def recorded_interaction_rate(
    records: list,
    drug_count: int,
    interaction_matrix: np.ndarray,
) -> float:
    """Interaction rate of the prescriptions as recorded by the clinicians."""
    recorded, _ = repeat_previous_admission(records, drug_count)
    return interaction_rate(recorded, interaction_matrix)
