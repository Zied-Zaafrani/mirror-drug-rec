"""What a trained model recommends on held-out patients, by drug name.

A Jaccard of 0.57 says little to a clinician. This script shows the same
predictions in terms a reader can check: the name of every drug class, and
whether the model recommended it rightly, recommended it wrongly or missed it.

    python src/showcase.py --cohort mimic3 --weights model.pt
        one row per drug class over the test patients: how often it was
        prescribed, found, missed and wrongly recommended, and how many new
        starts the model found. Counts only, so it can go wherever the
        aggregate results go.

    python src/showcase.py --cohort mimic3 --weights model.pt --patients 6
        one card per admission for a spread of test patients, each drug class
        marked right [+], wrong [x] or missed [-]. These are individual records
        from a credentialed database, so keep them where those records may be kept.

In the class view, a count between 1 and 10 is printed as ``<11`` and classes
prescribed fewer than 11 times are pooled into one row, the usual rule for
releasing counts drawn from patient records. The weights come from
``src/train.py --save_model``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import load_run_config, make_paths
from dataset import (
    DEFAULT_LANGUAGE,
    DIAGNOSIS,
    DRUG,
    CodeLabels,
    CohortArtifacts,
    PrescriptionDataset,
    build_drug_graph,
    collate_instances,
    coprescription_probabilities,
    load_cohort_artifacts,
    load_labels,
    select,
    split_patients,
)
from metrics import evaluate_recommendations
from model import Mirror
from train import move_to_device

SMALL_CELL = 11
NAME_WIDTH = 44
CONDITION_LIMIT = 6
PREVIOUS_LIMIT = 8

AGREED_MARK = "[+]"
EXTRA_MARK = "[x]"
MISSED_MARK = "[-]"
PARTITIONS = ("train", "validation", "test")


@dataclass
class Predictions:
    """The model's output over one partition, one row per prediction instance."""

    probabilities: np.ndarray
    recorded: np.ndarray
    previous: np.ndarray
    admission_ids: np.ndarray
    history_lengths: np.ndarray
    has_note: np.ndarray
    has_labs: np.ndarray
    instances: list[tuple[int, int]] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.probabilities.shape[0])


@dataclass
class Session:
    """One cohort, one trained model and its predictions over one partition."""

    artifacts: CohortArtifacts
    records: list
    predictions: Predictions
    labels: dict[str, CodeLabels]
    partition: str

    @property
    def drug_codes(self) -> list[str]:
        """The class code of each drug index."""
        return self.artifacts.drug_codes

    def language(self, code: str) -> CodeLabels:
        """The names in one language, falling back to English."""
        return self.labels.get(code, self.labels[DEFAULT_LANGUAGE])

    def recommended(self, threshold: float) -> np.ndarray:
        """The binary recommendation matrix at one threshold."""
        return (self.predictions.probabilities >= threshold).astype(np.float32)

    def summary(self, threshold: float) -> dict:
        """The reported metrics over the partition at one threshold."""
        predictions = self.predictions
        return evaluate_recommendations(
            recorded=predictions.recorded,
            recommended=self.recommended(threshold),
            probabilities=predictions.probabilities,
            interaction_matrix=self.artifacts.interaction_matrix,
            previous=predictions.previous,
        )

    def case_rows(self, threshold: float) -> list[dict]:
        """The overlap of every admission with its recorded prescription."""
        recorded, recommended = self.predictions.recorded, self.recommended(threshold)
        intersection = np.sum(recorded * recommended, axis=1)
        union = np.sum(np.clip(recorded + recommended, 0, 1), axis=1)
        agreement = intersection / np.maximum(union, 1e-8)
        return [{"index": i, "agreement": float(agreement[i])} for i in range(len(recorded))]

    def case(self, index: int, threshold: float, language: str = DEFAULT_LANGUAGE) -> dict:
        """What one admission was, what was prescribed and what the model recommended."""
        predictions = self.predictions
        labels = self.language(language)
        patient_index, admission_index = predictions.instances[index]
        admission = self.records[patient_index][admission_index]
        probabilities = predictions.probabilities[index]
        diagnoses = self.artifacts.diagnosis_codes
        context = labels.context.get(int(predictions.admission_ids[index]))
        return {
            "context": context.as_values() if context else {},
            "earlierVisits": int(predictions.history_lengths[index]),
            "hasNote": bool(predictions.has_note[index] > 0.5),
            "hasLabs": bool(predictions.has_labs[index] > 0.5),
            "conditions": [
                {"code": diagnoses[i], "name": labels.label(DIAGNOSIS, diagnoses[i])}
                for i in admission[0]
            ],
            "classes": [
                {
                    "code": code,
                    "name": labels.label(DRUG, code),
                    "probability": float(probabilities[drug]),
                    "recommended": bool(probabilities[drug] >= threshold),
                    "prescribed": bool(predictions.recorded[index, drug] > 0.5),
                    "onBefore": bool(predictions.previous[index, drug] > 0.5),
                }
                for drug, code in enumerate(self.drug_codes)
            ],
        }


def load_session(
    cohort_dir: Path,
    weights: Path,
    model_config,
    partition: str = "test",
    device: str = "cpu",
    language: str = DEFAULT_LANGUAGE,
) -> Session:
    """Load the cohort and the trained model, and score one partition once."""
    artifacts = load_cohort_artifacts(
        cohort_dir, use_notes=model_config.use_notes, use_labs=model_config.use_labs
    )
    split = split_patients(len(artifacts.records))
    parts = {"train": split.train, "validation": split.validation, "test": split.test}
    records = select(artifacts.records, parts[partition])
    training_records = select(artifacts.records, split.train)

    torch_device = torch.device(device)
    model = Mirror(artifacts, model_config).to(torch_device)
    model.load_state_dict(torch.load(weights, map_location=torch_device, weights_only=True))
    model.eval()
    graph = build_drug_graph(
        artifacts.interaction_matrix,
        artifacts.coprescription_matrix,
        artifacts.drug_codes,
        coprescription_probabilities(training_records, artifacts.drug_count),
        use_relations=model_config.use_graph_relations,
    ).to(torch_device)

    dataset = PrescriptionDataset(
        records, artifacts, use_notes=model.uses_notes, use_labs=model.uses_labs
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_instances)
    columns = {
        "probabilities": [], "recorded": [], "previous": [], "admission_ids": [],
        "history_lengths": [], "has_note": [], "has_labs": [],
    }
    with torch.no_grad():
        for batch in loader:
            scores, _ = model(move_to_device(batch, torch_device), graph)
            columns["probabilities"].append(torch.sigmoid(scores).cpu().numpy())
            columns["recorded"].append(batch["target"].numpy())
            columns["previous"].append(batch["previous_drugs"].numpy())
            columns["admission_ids"].append(batch["admission_id"].numpy())
            columns["history_lengths"].append(batch["history_length"].numpy())
            columns["has_note"].append(batch["has_note"].numpy())
            columns["has_labs"].append(batch["has_labs"].numpy())
    predictions = Predictions(
        **{key: np.concatenate(parts) for key, parts in columns.items()},
        instances=list(dataset.instances),
    )
    labels = {DEFAULT_LANGUAGE: load_labels(cohort_dir, DEFAULT_LANGUAGE)}
    if language != DEFAULT_LANGUAGE:
        labels[language] = load_labels(cohort_dir, language)
    return Session(artifacts=artifacts, records=records, predictions=predictions,
                   labels=labels, partition=partition)


def _count(value: int) -> str:
    """A count as printed, hiding the small ones."""
    return f"<{SMALL_CELL}" if 0 < value < SMALL_CELL else f"{value:,}"


def _share(part: int, whole: int) -> str:
    """A percentage, or a dash when there is nothing to divide."""
    return f"{100 * part / whole:.0f}%" if whole else "-"


def _name(text: str, width: int = NAME_WIDTH) -> str:
    """Fit a name into a column."""
    text = text or ""
    return text if len(text) <= width else text[: width - 3] + "..."


def class_counts(session: Session, threshold: float) -> list[dict]:
    """Per drug class: prescribed, found, missed, wrongly recommended, starts found."""
    predictions = session.predictions
    recorded = predictions.recorded > 0.5
    previous = predictions.previous > 0.5
    recommended = session.recommended(threshold).astype(bool)
    started = recorded & ~previous
    return [
        {
            "code": code,
            "prescribed": int(recorded[:, drug].sum()),
            "found": int((recorded[:, drug] & recommended[:, drug]).sum()),
            "missed": int((recorded[:, drug] & ~recommended[:, drug]).sum()),
            "extra": int((~recorded[:, drug] & recommended[:, drug]).sum()),
            "started": int(started[:, drug].sum()),
            "started_found": int((started[:, drug] & recommended[:, drug]).sum()),
        }
        for drug, code in enumerate(session.drug_codes)
    ]


def by_class(session: Session, threshold: float = 0.5, language: str = DEFAULT_LANGUAGE) -> str:
    """The drug-class table as text."""
    labels = session.language(language)
    rows = class_counts(session, threshold)
    shown = sorted((r for r in rows if r["prescribed"] >= SMALL_CELL),
                   key=lambda r: -r["prescribed"])
    pooled = [r for r in rows if r["prescribed"] < SMALL_CELL]
    summary = session.summary(threshold)

    lines = [
        f"What the model recommends on the {session.partition} patients "
        f"({len(session.predictions):,} admissions the model never trained on)",
        f"Jaccard {summary['jaccard']:.4f}, F1 {summary['f1']:.4f}, "
        f"DDI rate {summary['interaction_rate']:.4f}, threshold {threshold:g}",
        "",
        "  Found:  share of prescriptions of the class that the model recommended",
        "  Right:  share of the model's recommendations of the class that were prescribed",
        "  Wrong:  recommended although not prescribed",
        "  Starts: prescriptions new since the previous admission, and how many the model found",
        "",
        f"  {'Drug class':<{NAME_WIDTH}} {'ATC':<5} {'Prescribed':>10} {'Found':>6} "
        f"{'Right':>6} {'Missed':>7} {'Wrong':>7} {'Starts found':>14}",
    ]
    for row in shown:
        recommended = row["found"] + row["extra"]
        starts = (f"{_count(row['started_found'])} of {_count(row['started'])}"
                  if row["started"] else "-")
        lines.append(
            f"  {_name(labels.label(DRUG, row['code']) or row['code']):<{NAME_WIDTH}} "
            f"{row['code']:<5} {_count(row['prescribed']):>10} "
            f"{_share(row['found'], row['prescribed']):>6} "
            f"{_share(row['found'], recommended):>6} "
            f"{_count(row['missed']):>7} {_count(row['extra']):>7} {starts:>14}"
        )
    if pooled:
        prescribed = sum(r["prescribed"] for r in pooled)
        found = sum(r["found"] for r in pooled)
        extra = sum(r["extra"] for r in pooled)
        title = _name(f"{len(pooled)} classes prescribed fewer than {SMALL_CELL} times")
        lines.append(
            f"  {title:<{NAME_WIDTH}} "
            f"{'':<5} {_count(prescribed):>10} {_share(found, prescribed):>6} "
            f"{_share(found, found + extra):>6} {_count(prescribed - found):>7} "
            f"{_count(extra):>7} {'':>14}"
        )
    lines.append("")
    lines.append(f"Counts from 1 to {SMALL_CELL - 1} are shown as <{SMALL_CELL}.")
    return "\n".join(lines)


def pick_admissions(session: Session, threshold: float, count: int) -> list[int]:
    """Choose admissions spread evenly from the hardest to the easiest for the model.

    Only admissions whose previous admission carries a prescription are used, so
    every card has a previous regimen to compare against. The choice depends only
    on the predictions, so it is the same on every run of the same model.
    """
    rows = session.case_rows(threshold)
    candidates = [r for r in rows if session.predictions.previous[r["index"]].sum() > 0]
    if not candidates:
        candidates = rows
    candidates.sort(key=lambda r: (r["agreement"], r["index"]))
    if count >= len(candidates):
        return [r["index"] for r in candidates]
    positions = np.linspace(0.1, 0.9, count) * (len(candidates) - 1)
    return [candidates[int(round(p))]["index"] for p in positions]


def _first(items: list[str], limit: int) -> list[str]:
    """The first few items, and how many were left out."""
    if len(items) <= limit:
        return items
    return items[:limit] + [f"and {len(items) - limit} more"]


def _card(case: dict, number: int, total: int) -> str:
    """One admission as text."""
    context = case.get("context", {})
    who = ", ".join(
        part for part in (
            context.get("sex", ""),
            f"age {context['age']}" if context.get("age") else "",
            f"{context['admissionType']} admission" if context.get("admissionType") else "",
            f"{context['stayDays']} days in hospital" if context.get("stayDays") else "",
        ) if part
    )
    classes = case["classes"]
    agreed = [c for c in classes if c["prescribed"] and c["recommended"]]
    extra = [c for c in classes if c["recommended"] and not c["prescribed"]]
    missed = [c for c in classes if c["prescribed"] and not c["recommended"]]
    stopped = [c for c in classes if c["onBefore"] and not c["prescribed"]]
    union = len(agreed) + len(extra) + len(missed)
    overlap = len(agreed) / union if union else 0.0

    def name(entry: dict) -> str:
        return _name(entry["name"] or entry["code"])

    shown = _first([c["name"] or c["code"] for c in case["conditions"]], CONDITION_LIMIT)
    previous = _first([name(c) for c in classes if c["onBefore"]], PREVIOUS_LIMIT)

    lines = [
        f"Hidden patient {number} of {total}: {who}",
        f"  {case['earlierVisits']} earlier admission(s) read by the model; "
        f"discharge note {'present' if case['hasNote'] else 'absent'}, "
        f"laboratory values {'present' if case['hasLabs'] else 'absent'}",
        f"  Conditions: {'; '.join(shown) if shown else 'none recorded'}",
        f"  Previous prescription: {'; '.join(previous) if previous else 'none'}",
        f"  Overlap with the prescription on record: {overlap:.2f} "
        f"({len(agreed)} right, {len(extra)} wrong, {len(missed)} missed)",
        "",
    ]
    for title, mark, entries in (
        ("Recommended and prescribed", AGREED_MARK, agreed),
        ("Recommended but not prescribed", EXTRA_MARK, extra),
        ("Prescribed but not recommended", MISSED_MARK, missed),
    ):
        if not entries:
            continue
        lines.append(f"  {title}")
        for entry in sorted(entries, key=lambda c: -c["probability"]):
            change = "continued" if entry["onBefore"] else "new this admission"
            if not entry["prescribed"]:
                change = "was on the previous prescription" if entry["onBefore"] else ""
            lines.append(
                f"    {mark} {name(entry):<{NAME_WIDTH}} {entry['code']:<5} "
                f"p={entry['probability']:.2f}  {change}".rstrip()
            )
    if stopped:
        kept = [c for c in stopped if c["recommended"]]
        lines.append(
            f"  Stopped since the previous admission: {len(stopped)}, "
            f"of which the model also left out {len(stopped) - len(kept)}"
        )
    return "\n".join(lines)


def card(session: Session, index: int, threshold: float = 0.5,
         language: str = DEFAULT_LANGUAGE) -> str:
    """The card of one prediction instance."""
    return _card(session.case(index, threshold, language), 1, 1)


def patients(session: Session, threshold: float = 0.5, count: int = 6,
             language: str = DEFAULT_LANGUAGE) -> str:
    """Cards for a spread of held-out admissions."""
    chosen = pick_admissions(session, threshold, count)
    header = [
        f"{len(chosen)} {session.partition} admissions, "
        "from the hardest to the easiest for the model",
        f"  {AGREED_MARK} recommended and prescribed   {EXTRA_MARK} recommended, not prescribed"
        f"   {MISSED_MARK} prescribed, not recommended",
    ]
    cards = [_card(session.case(index, threshold, language), number, len(chosen))
             for number, index in enumerate(chosen, start=1)]
    return "\n\n".join(["\n".join(header), *cards])


def main(argv: list[str] | None = None) -> int:
    """Print the drug-class table, or cards for held-out patients."""
    parser = argparse.ArgumentParser(description="Show what the model recommends, by drug name.")
    parser.add_argument("--cohort", default="mimic3", help="a cohort listed in config.yaml")
    parser.add_argument("--weights", type=Path, required=True,
                        help="weights written by src/train.py --save_model")
    parser.add_argument("--data_dir", type=Path, default=None,
                        help="folder holding one sub-folder per cohort (default: data/processed)")
    parser.add_argument("--patients", type=int, default=0,
                        help="print cards for this many test admissions instead of the table")
    parser.add_argument("--instance", type=int, default=None,
                        help="print the card of this one prediction instance")
    parser.add_argument("--partition", default="test", choices=PARTITIONS)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--no_notes", action="store_true",
                        help="for a model trained with --no_notes")
    parser.add_argument("--no_labs", action="store_true", help="for a model trained with --no_labs")
    parser.add_argument("--no_copy_head", action="store_true",
                        help="for a model trained with --no_copy_head")
    parser.add_argument("--output", type=Path, default=None,
                        help="write to a file instead of the terminal")
    arguments = parser.parse_args(argv)

    config = load_run_config(
        arguments.cohort,
        paths=make_paths(arguments.data_dir),
        model_overrides={
            "use_notes": False if arguments.no_notes else None,
            "use_labs": False if arguments.no_labs else None,
            "use_copy_head": False if arguments.no_copy_head else None,
        },
        device=arguments.device,
    )
    session = load_session(
        config.paths.cohort_dir(config.cohort.directory),
        arguments.weights,
        config.model,
        partition=arguments.partition,
        device=arguments.device,
        language=arguments.language,
    )
    if arguments.instance is not None:
        text = card(session, arguments.instance, arguments.threshold, arguments.language)
    elif arguments.patients:
        text = patients(session, arguments.threshold, arguments.patients, arguments.language)
    else:
        text = by_class(session, arguments.threshold, arguments.language)

    if arguments.output is None:
        print(text)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text + "\n", encoding="utf-8")
        print(f"Written to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
