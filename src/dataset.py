"""The processed cohort on disk, the patient split, the model inputs and the drug graph.

A cohort folder holds the records, the vocabulary, the drug matrices, the code
embeddings, the note vectors and the laboratory vectors written by
``preprocess.py`` and ``features.py``. This file loads them, splits the patients
into training, validation and test partitions, turns every admission after a
patient's first into one prediction instance, and builds the drug graph.
"""

from __future__ import annotations

import csv
import dataclasses
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

RECORDS_FILE = "records.pkl"
VOCABULARY_FILE = "vocabulary.pkl"
INTERACTION_FILE = "interaction_matrix.pkl"
COPRESCRIPTION_FILE = "coprescription_matrix.pkl"
CODE_EMBEDDING_FILE = "code_embeddings.pt"
NOTE_EMBEDDING_FILE = "note_embeddings.pkl"
NOTE_MEAN_FILE = "note_mean.npy"
LAB_FEATURE_FILE = "lab_features.pkl"

DIAGNOSIS_POSITION = 0
PROCEDURE_POSITION = 1
DRUG_POSITION = 2
ADMISSION_ID_POSITION = 3


@dataclass
class CohortArtifacts:
    """Every preprocessed input for one cohort.

    Attributes:
        records: one entry per patient, each a list of admissions ordered in
            time; an admission is ``[diagnosis_ids, procedure_ids, drug_ids,
            admission_id]``.
        diagnosis_count, procedure_count, drug_count: vocabulary sizes.
        drug_codes: drug index to drug class code, used for the drug-class
            relation in the graph and for readable output.
        diagnosis_codes, procedure_codes: the code behind each index, kept so
            output can name a condition rather than number it.
        interaction_matrix: 1 where two drug classes are a known interacting
            pair.
        coprescription_matrix: how often two drug classes appear in the same
            admission, computed during preprocessing.
        diagnosis_embeddings, procedure_embeddings, drug_embeddings: frozen
            language-model embeddings of the code descriptions.
        molecular_fingerprints: binary structural fingerprint per drug class.
        note_embeddings, note_available: discharge-note vector per admission and
            a flag for admissions without a usable note.
        note_mean: mean note vector of the training admissions, subtracted
            before the note vector enters the model.
        lab_vectors, lab_available: laboratory vector per admission and a flag
            for admissions without measurements.
        lab_test_names: name of each laboratory test, in vector order.
        lab_training_mean, lab_training_spread: the mean and spread each test
            was standardised with, kept so a standardised value can be shown
            back in the unit it was measured in.
    """

    records: list[list[list[Any]]]
    diagnosis_count: int
    procedure_count: int
    drug_count: int
    drug_codes: list[str]
    interaction_matrix: np.ndarray
    coprescription_matrix: np.ndarray
    diagnosis_embeddings: torch.Tensor
    procedure_embeddings: torch.Tensor
    drug_embeddings: torch.Tensor
    molecular_fingerprints: torch.Tensor
    note_embeddings: dict[int, np.ndarray] | None = None
    note_available: dict[int, bool] | None = None
    note_mean: np.ndarray | None = None
    lab_vectors: dict[int, np.ndarray] | None = None
    lab_available: dict[int, bool] | None = None
    lab_test_names: list[str] | None = None
    lab_training_mean: np.ndarray | None = None
    lab_training_spread: np.ndarray | None = None
    diagnosis_codes: list[str] = dataclasses.field(default_factory=list)
    procedure_codes: list[str] = dataclasses.field(default_factory=list)

    @property
    def note_dim(self) -> int:
        """Width of one discharge-note vector."""
        if self.note_mean is not None:
            return int(self.note_mean.shape[0])
        if self.note_embeddings:
            return int(next(iter(self.note_embeddings.values())).shape[0])
        return 0

    @property
    def lab_dim(self) -> int:
        """Width of one laboratory vector."""
        if self.lab_vectors:
            return int(next(iter(self.lab_vectors.values())).shape[0])
        return 0

    def admission_count(self) -> int:
        """Total number of admissions across all patients."""
        return sum(len(patient) for patient in self.records)

    def prediction_count(self) -> int:
        """Number of admissions the model can be asked about, one per non-first admission."""
        return sum(max(len(patient) - 1, 0) for patient in self.records)


def _read_pickle(path: Path) -> Any:
    """Read a pickle file, naming the file in the error when it is absent."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} is missing from {path.parent}. "
            "Run the preprocessing and feature steps for this cohort first "
            "(see data/README.md)."
        )
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _require_keys(mapping: dict[str, Any], keys: tuple[str, ...], source: Path) -> None:
    """Raise if any expected key is absent from a loaded file."""
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise KeyError(f"{source.name} is missing the entries {missing}")


def _as_lookup(
    identifiers: np.ndarray, values: np.ndarray, available: np.ndarray
) -> tuple[dict[int, np.ndarray], dict[int, bool]]:
    """Turn parallel arrays keyed by admission identifier into two dictionaries."""
    value_by_admission = {int(key): np.asarray(values[i]) for i, key in enumerate(identifiers)}
    flag_by_admission = {int(key): bool(available[i]) for i, key in enumerate(identifiers)}
    return value_by_admission, flag_by_admission


def load_cohort_artifacts(
    cohort_dir: Path,
    use_notes: bool = True,
    use_labs: bool = True,
    expected_lab_dim: int | None = None,
) -> CohortArtifacts:
    """Load one cohort's records, vocabulary, matrices, embeddings, notes and labs.

    Args:
        cohort_dir: directory holding the cohort's preprocessed files.
        use_notes: load the discharge-note vectors.
        use_labs: load the laboratory vectors.
        expected_lab_dim: if given, the laboratory vectors must have this width.

    Returns:
        A populated :class:`CohortArtifacts`.

    Raises:
        FileNotFoundError: a required file is missing.
        ValueError: the laboratory width does not match ``expected_lab_dim``.
    """
    cohort_dir = Path(cohort_dir)
    records = _read_pickle(cohort_dir / RECORDS_FILE)
    vocabulary = _read_pickle(cohort_dir / VOCABULARY_FILE)
    _require_keys(
        vocabulary,
        ("diagnosis_count", "procedure_count", "drug_codes"),
        cohort_dir / VOCABULARY_FILE,
    )
    interaction_matrix = np.asarray(_read_pickle(cohort_dir / INTERACTION_FILE), dtype=np.float32)
    coprescription_matrix = np.asarray(
        _read_pickle(cohort_dir / COPRESCRIPTION_FILE), dtype=np.float32
    )

    embedding_path = cohort_dir / CODE_EMBEDDING_FILE
    if not embedding_path.exists():
        raise FileNotFoundError(
            f"{CODE_EMBEDDING_FILE} is missing from {cohort_dir}. "
            "Build it with src/features.py (see data/README.md)."
        )
    embeddings = torch.load(embedding_path, map_location="cpu", weights_only=True)
    _require_keys(
        embeddings,
        ("diagnosis", "procedure", "drug", "molecular_fingerprint"),
        embedding_path,
    )

    drug_codes = [str(code) for code in vocabulary["drug_codes"]]
    artifacts = CohortArtifacts(
        records=records,
        diagnosis_count=int(vocabulary["diagnosis_count"]),
        procedure_count=int(vocabulary["procedure_count"]),
        drug_count=len(drug_codes),
        drug_codes=drug_codes,
        diagnosis_codes=[str(code) for code in vocabulary.get("diagnosis_codes", [])],
        procedure_codes=[str(code) for code in vocabulary.get("procedure_codes", [])],
        interaction_matrix=interaction_matrix,
        coprescription_matrix=coprescription_matrix,
        diagnosis_embeddings=embeddings["diagnosis"].float(),
        procedure_embeddings=embeddings["procedure"].float(),
        drug_embeddings=embeddings["drug"].float(),
        molecular_fingerprints=embeddings["molecular_fingerprint"].float(),
    )

    if use_notes:
        note_path = cohort_dir / NOTE_EMBEDDING_FILE
        note_file = _read_pickle(note_path)
        _require_keys(note_file, ("admission_ids", "embeddings", "has_note"), note_path)
        artifacts.note_embeddings, artifacts.note_available = _as_lookup(
            np.asarray(note_file["admission_ids"]),
            np.asarray(note_file["embeddings"]),
            np.asarray(note_file["has_note"]),
        )
        mean_path = cohort_dir / NOTE_MEAN_FILE
        if not mean_path.exists():
            raise FileNotFoundError(
                f"{NOTE_MEAN_FILE} is missing from {cohort_dir}. The note channel needs "
                "the training-set mean that the note step writes beside the embeddings."
            )
        artifacts.note_mean = np.load(mean_path).astype(np.float32)

    if use_labs:
        lab_path = cohort_dir / LAB_FEATURE_FILE
        lab_file = _read_pickle(lab_path)
        _require_keys(lab_file, ("admission_ids", "vectors", "has_labs"), lab_path)
        artifacts.lab_vectors, artifacts.lab_available = _as_lookup(
            np.asarray(lab_file["admission_ids"]),
            np.asarray(lab_file["vectors"]),
            np.asarray(lab_file["has_labs"]),
        )
        artifacts.lab_test_names = [str(name) for name in lab_file.get("test_names", [])]
        if "training_mean" in lab_file and "training_spread" in lab_file:
            mean, spread = lab_file["training_mean"], lab_file["training_spread"]
            artifacts.lab_training_mean = np.asarray(mean, dtype=np.float32)
            artifacts.lab_training_spread = np.asarray(spread, dtype=np.float32)
        if expected_lab_dim is not None and artifacts.lab_dim != expected_lab_dim:
            raise ValueError(
                f"{LAB_FEATURE_FILE} holds vectors of width {artifacts.lab_dim}, but the "
                f"configuration expects {expected_lab_dim} (two values per test). "
                "Rebuild the laboratory features or set lab_count to match."
            )

    return artifacts


def write_cohort_artifacts(
    cohort_dir: Path,
    records: list,
    vocabulary: dict[str, Any],
    interaction_matrix: np.ndarray,
    coprescription_matrix: np.ndarray,
) -> None:
    """Write the records, vocabulary and the two drug matrices for one cohort.

    The embedding, note and laboratory files are written by the feature builders.
    """
    cohort_dir = Path(cohort_dir)
    cohort_dir.mkdir(parents=True, exist_ok=True)
    with open(cohort_dir / RECORDS_FILE, "wb") as handle:
        pickle.dump(records, handle)
    with open(cohort_dir / VOCABULARY_FILE, "wb") as handle:
        pickle.dump(vocabulary, handle)
    with open(cohort_dir / INTERACTION_FILE, "wb") as handle:
        pickle.dump(np.asarray(interaction_matrix, dtype=np.float32), handle)
    with open(cohort_dir / COPRESCRIPTION_FILE, "wb") as handle:
        pickle.dump(np.asarray(coprescription_matrix, dtype=np.float32), handle)


TRAIN_FRACTION = 2 / 3


@dataclass(frozen=True)
class PatientSplit:
    """Record indices of the three partitions."""

    train: list[int]
    test: list[int]
    validation: list[int]

    def sizes(self) -> dict[str, int]:
        """Patient count per partition."""
        return {
            "train": len(self.train),
            "test": len(self.test),
            "validation": len(self.validation),
        }


def split_patients(record_count: int) -> PatientSplit:
    """Split ``record_count`` patient records into two thirds, one sixth, one sixth.

    Args:
        record_count: number of patient records in the cohort file.

    Returns:
        A :class:`PatientSplit` whose three index lists are disjoint and cover
        every record.

    Raises:
        ValueError: the cohort is too small to fill all three partitions.
    """
    train_end = int(record_count * TRAIN_FRACTION)
    test_end = train_end + int((record_count - train_end) / 2)
    split = PatientSplit(
        train=list(range(0, train_end)),
        test=list(range(train_end, test_end)),
        validation=list(range(test_end, record_count)),
    )
    empty = [name for name, size in split.sizes().items() if size == 0]
    if empty:
        raise ValueError(
            f"A cohort of {record_count} records leaves these partitions empty: {empty}."
        )
    return split


def select(records: list, indices: list[int]) -> list:
    """Return the records at ``indices``, keeping their order."""
    return [records[i] for i in indices]


LABEL_FILE = "code_labels.csv"
CONTEXT_FILE = "patient_context.csv"
DEFAULT_LANGUAGE = "en"


def label_file(language: str = DEFAULT_LANGUAGE) -> str:
    """Name of the label file for one language."""
    return LABEL_FILE if language == DEFAULT_LANGUAGE else f"code_labels.{language}.csv"

DIAGNOSIS = "diagnosis"
PROCEDURE = "procedure"
DRUG = "drug"
KINDS = (DIAGNOSIS, PROCEDURE, DRUG)

# MIMIC-III shifts the birth date of anyone over 89, so their computed age comes
# out around three hundred. The real age is not in the data and must not be
# guessed, so it is reported as the band the de-identification leaves.
OLDEST_REPORTABLE_AGE = 89
AGE_BAND_LABEL = "90+"


@dataclass
class AdmissionContext:
    """What is known about one admission beyond its codes.

    None of this reaches the model. It is here so that a reader can see who the
    recommendation is for.
    """

    age: str = ""
    sex: str = ""
    admission_type: str = ""
    stay_days: float | None = None

    def as_dict(self) -> dict:
        """Render for the printed form, leaving out what is unknown.

        The keys are the words the terminal prints, which is why they read as
        English rather than as identifiers.
        """
        fields = {
            "age": self.age,
            "sex": self.sex,
            "admission type": self.admission_type.lower() if self.admission_type else "",
            "days in hospital": f"{self.stay_days:.0f}" if self.stay_days is not None else "",
        }
        return {name: value for name, value in fields.items() if value}

    def as_values(self) -> dict:
        """The same facts under keys that do not change with the language.

        A caller that names these in its own language needs stable keys
        rather than English words.
        """
        values = {
            "age": self.age,
            "sex": self.sex,
            "admissionType": self.admission_type.lower() if self.admission_type else "",
            "stayDays": f"{self.stay_days:.0f}" if self.stay_days is not None else "",
        }
        return {name: value for name, value in values.items() if value}


@dataclass
class CodeLabels:
    """Names for the codes of one cohort, by kind."""

    names: dict[str, dict[str, str]] = field(
        default_factory=lambda: {kind: {} for kind in KINDS}
    )
    context: dict[int, AdmissionContext] = field(default_factory=dict)

    def label(self, kind: str, code: str) -> str:
        """Return the name of one code, or an empty string when unknown."""
        return self.names.get(kind, {}).get(code, "")

    def describe(self, kind: str, code: str) -> str:
        """Return ``code`` followed by its name, or just the code."""
        name = self.label(kind, code)
        return f"{code} {name}" if name else code

    @property
    def has_labels(self) -> bool:
        """Whether any names were loaded."""
        return any(self.names[kind] for kind in KINDS)


def format_age(years: float) -> str:
    """Report an age, banding the ages the de-identification has shifted."""
    if years > OLDEST_REPORTABLE_AGE:
        return AGE_BAND_LABEL
    return f"{years:.0f}"


def _read_into(labels: CodeLabels, path: Path) -> None:
    """Read one label file into ``labels``, replacing the names it covers."""
    if not path.exists():
        return
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            kind = (row.get("kind") or "").strip()
            code = (row.get("code") or "").strip()
            name = (row.get("label") or "").strip()
            if kind in labels.names and code and name:
                labels.names[kind][code] = name


def load_labels(cohort_dir: Path, language: str = DEFAULT_LANGUAGE) -> CodeLabels:
    """Read the names and the admission details, in one language.

    The English names are read first and the chosen language is laid over them,
    so a partial translation shows what it covers and leaves the rest readable
    rather than blank.
    """
    cohort_dir = Path(cohort_dir)
    labels = CodeLabels()

    _read_into(labels, cohort_dir / LABEL_FILE)
    if language != DEFAULT_LANGUAGE:
        _read_into(labels, cohort_dir / label_file(language))

    context_path = cohort_dir / CONTEXT_FILE
    if context_path.exists():
        with open(context_path, encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    admission = int(row["admission_id"])
                except (KeyError, TypeError, ValueError):
                    continue
                stay = row.get("stay_days")
                labels.context[admission] = AdmissionContext(
                    age=(row.get("age") or "").strip(),
                    sex=(row.get("sex") or "").strip(),
                    admission_type=(row.get("admission_type") or "").strip(),
                    stay_days=float(stay) if stay not in (None, "") else None,
                )

    return labels


def write_labels(
    cohort_dir: Path,
    rows: list[tuple[str, str, str]],
    language: str = DEFAULT_LANGUAGE,
) -> Path:
    """Write the code name file for one cohort and return its path."""
    path = Path(cohort_dir) / label_file(language)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kind", "code", "label"])
        writer.writerows(rows)
    return path


Z_SCORE_LIMIT = 5.0


class PrescriptionDataset(Dataset):
    """Prediction instances for one partition of patients.

    Args:
        records: patient records of this partition.
        artifacts: the cohort's vocabulary sizes, notes and laboratory values.
        use_notes: feed the discharge-note vector; when false the note slot is
            zeros and its availability flag is zero.
        use_labs: feed the laboratory vector, under the same rule.
    """

    def __init__(
        self,
        records: list,
        artifacts: CohortArtifacts,
        use_notes: bool = True,
        use_labs: bool = True,
    ):
        self.records = records
        self.drug_count = artifacts.drug_count
        self.use_notes = use_notes and artifacts.note_embeddings is not None
        self.use_labs = use_labs and artifacts.lab_vectors is not None
        self.note_embeddings = artifacts.note_embeddings or {}
        self.note_available = artifacts.note_available or {}
        self.lab_vectors = artifacts.lab_vectors or {}
        self.lab_available = artifacts.lab_available or {}
        self.note_dim = artifacts.note_dim if self.use_notes else 0
        self.lab_dim = artifacts.lab_dim if self.use_labs else 0
        self.instances = [
            (patient_index, admission_index)
            for patient_index, patient in enumerate(records)
            for admission_index in range(1, len(patient))
        ]

    def __len__(self) -> int:
        return len(self.instances)

    def _multi_hot(self, drug_ids: list[int]) -> np.ndarray:
        """Turn a list of drug indices into a binary vector over the drug vocabulary."""
        vector = np.zeros(self.drug_count, dtype=np.float32)
        for drug_id in drug_ids:
            if drug_id >= self.drug_count:
                raise ValueError(
                    f"Drug index {drug_id} is outside the vocabulary of {self.drug_count} "
                    "classes. The records and the vocabulary come from different "
                    "preprocessing runs."
                )
            vector[drug_id] = 1.0
        return vector

    def _note_for(self, admission_id: int) -> tuple[np.ndarray, float]:
        """Return the note vector of one admission and whether a note was found."""
        if not self.use_notes:
            return np.zeros(0, dtype=np.float32), 0.0
        vector = self.note_embeddings.get(admission_id)
        if vector is None or not self.note_available.get(admission_id, False):
            return np.zeros(self.note_dim, dtype=np.float32), 0.0
        return np.asarray(vector, dtype=np.float32), 1.0

    def _labs_for(self, admission_id: int) -> tuple[np.ndarray, float]:
        """Return the laboratory vector of one admission and whether measurements exist.

        The first half of the vector holds z-scored values, the second half holds
        one flag per test which is 1 when the test was not measured. Values are
        clipped so that a single extreme reading cannot dominate the projection,
        and an admission whose every test is flagged missing counts as having no
        laboratory data at all.
        """
        if not self.use_labs:
            return np.zeros(0, dtype=np.float32), 0.0
        vector = self.lab_vectors.get(admission_id)
        if vector is None or not self.lab_available.get(admission_id, False):
            return np.zeros(self.lab_dim, dtype=np.float32), 0.0
        vector = np.asarray(vector, dtype=np.float32).copy()
        test_count = vector.shape[0] // 2
        vector[:test_count] = np.clip(vector[:test_count], -Z_SCORE_LIMIT, Z_SCORE_LIMIT)
        if np.all(vector[test_count : 2 * test_count] > 0.5):
            return np.zeros(self.lab_dim, dtype=np.float32), 0.0
        return vector, 1.0

    def __getitem__(self, index: int) -> dict:
        """Build one prediction instance."""
        patient_index, admission_index = self.instances[index]
        patient = self.records[patient_index]
        history = patient[:admission_index]
        predicted = patient[admission_index]

        diagnosis_codes = [admission[DIAGNOSIS_POSITION] for admission in history]
        procedure_codes = [admission[PROCEDURE_POSITION] for admission in history]
        drugs_per_visit = np.stack(
            [self._multi_hot(admission[DRUG_POSITION]) for admission in history], axis=0
        )
        admission_id = int(predicted[ADMISSION_ID_POSITION])
        note_vector, has_note = self._note_for(admission_id)
        lab_vector, has_labs = self._labs_for(admission_id)

        return {
            "diagnosis_codes": diagnosis_codes,
            "procedure_codes": procedure_codes,
            "history_length": len(history),
            "drugs_per_visit": drugs_per_visit,
            "drug_history": np.clip(drugs_per_visit.sum(axis=0), 0.0, 1.0),
            "previous_drugs": drugs_per_visit[-1],
            "target": self._multi_hot(predicted[DRUG_POSITION]),
            "note_vector": note_vector,
            "has_note": has_note,
            "lab_vector": lab_vector,
            "has_labs": has_labs,
            "admission_id": admission_id,
        }


def _pad_code_lists(
    batch: list[dict], key: str, timestep: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad one timestep's code lists into a tensor plus a validity mask."""
    lists = [
        item[key][timestep] if timestep < item["history_length"] else []
        for item in batch
    ]
    width = max(max((len(codes) for codes in lists), default=0), 1)
    codes_tensor = torch.zeros(len(batch), width, dtype=torch.long)
    mask_tensor = torch.zeros(len(batch), width, dtype=torch.bool)
    for row, codes in enumerate(lists):
        if codes:
            codes_tensor[row, : len(codes)] = torch.tensor(codes, dtype=torch.long)
            mask_tensor[row, : len(codes)] = True
    return codes_tensor, mask_tensor


def collate_instances(batch: list[dict]) -> dict:
    """Collate prediction instances into padded batch tensors.

    Patients contribute different numbers of admissions and each admission a
    different number of codes, so the code tensors are built one timestep at a
    time and carry a mask marking the real entries.
    """
    longest_history = max(item["history_length"] for item in batch)
    diagnosis_codes, diagnosis_mask = [], []
    procedure_codes, procedure_mask = [], []
    for timestep in range(longest_history):
        codes, mask = _pad_code_lists(batch, "diagnosis_codes", timestep)
        diagnosis_codes.append(codes)
        diagnosis_mask.append(mask)
        codes, mask = _pad_code_lists(batch, "procedure_codes", timestep)
        procedure_codes.append(codes)
        procedure_mask.append(mask)

    drug_count = batch[0]["target"].shape[0]
    drugs_per_visit = torch.zeros(len(batch), longest_history, drug_count)
    for row, item in enumerate(batch):
        length = item["history_length"]
        drugs_per_visit[row, :length] = torch.from_numpy(item["drugs_per_visit"][:length])

    def stack(key: str) -> torch.Tensor:
        return torch.from_numpy(np.stack([item[key] for item in batch])).float()

    def scalars(key: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.tensor([item[key] for item in batch], dtype=dtype)

    return {
        "diagnosis_codes": diagnosis_codes,
        "diagnosis_mask": diagnosis_mask,
        "procedure_codes": procedure_codes,
        "procedure_mask": procedure_mask,
        "history_length": scalars("history_length", torch.long),
        "drugs_per_visit": drugs_per_visit,
        "drug_history": stack("drug_history"),
        "previous_drugs": stack("previous_drugs"),
        "target": stack("target"),
        "note_vector": stack("note_vector"),
        "has_note": scalars("has_note"),
        "lab_vector": stack("lab_vector"),
        "has_labs": scalars("has_labs"),
        "admission_id": scalars("admission_id", torch.long),
    }


INTERACTION_RELATION = 0
COPRESCRIPTION_RELATION = 1
SELF_LOOP_RELATION = 2
DRUG_CLASS_RELATION = 3

RELATION_NAMES = {
    INTERACTION_RELATION: "interaction",
    COPRESCRIPTION_RELATION: "co-prescription",
    SELF_LOOP_RELATION: "self-loop",
    DRUG_CLASS_RELATION: "drug class",
}

COPRESCRIPTION_THRESHOLD = 0.05
CLASS_PREFIX_LENGTH = 3
BROADER_CLASS_PREFIX_LENGTH = 2
MINIMUM_EDGE_WEIGHT = 1e-8


@dataclass
class DrugGraph:
    """Edges of the drug graph in the form the encoder consumes.

    Attributes:
        edge_index: ``(2, edge_count)`` source and target node of each edge.
        edge_relation: ``(edge_count,)`` relation identifier of each edge.
        edge_weight: ``(edge_count,)`` weight of each edge; one for every
            relation except co-prescription, which carries a conditional
            probability.
    """

    edge_index: torch.Tensor
    edge_relation: torch.Tensor
    edge_weight: torch.Tensor

    @property
    def edge_count(self) -> int:
        """Total number of directed edges."""
        return int(self.edge_index.shape[1])

    def counts_per_relation(self) -> dict[str, int]:
        """Number of edges of each relation, keyed by readable name."""
        relations, counts = torch.unique(self.edge_relation, return_counts=True)
        return {
            RELATION_NAMES.get(int(relation), str(int(relation))): int(count)
            for relation, count in zip(relations.tolist(), counts.tolist(), strict=True)
        }

    def to(self, device: torch.device) -> DrugGraph:
        """Move the three tensors onto ``device``."""
        return DrugGraph(
            edge_index=self.edge_index.to(device),
            edge_relation=self.edge_relation.to(device),
            edge_weight=self.edge_weight.to(device),
        )


def coprescription_probabilities(records: list, drug_count: int) -> np.ndarray:
    """Estimate the chance of prescribing one drug class given another.

    Args:
        records: the training patients only.
        drug_count: size of the drug vocabulary.

    Returns:
        A ``(drug_count, drug_count)`` matrix whose entry ``[i, j]`` is the
        fraction of admissions containing class ``i`` that also contain class
        ``j``. The matrix is asymmetric because the two classes differ in how
        often they are prescribed.
    """
    together = np.zeros((drug_count, drug_count), dtype=np.float64)
    occurrences = np.zeros(drug_count, dtype=np.float64)
    for patient in records:
        for admission in patient:
            drugs = [drug for drug in admission[DRUG_POSITION] if drug < drug_count]
            occurrences[drugs] += 1
            for first in drugs:
                for second in drugs:
                    if first != second:
                        together[first, second] += 1

    probabilities = np.zeros((drug_count, drug_count), dtype=np.float32)
    prescribed = occurrences > 0
    probabilities[prescribed] = (
        together[prescribed] / occurrences[prescribed, None]
    ).astype(np.float32)
    return probabilities


def _class_groups(drug_codes: list[str], prefix_length: int) -> dict[str, list[int]]:
    """Group drug indices by the leading characters of their class code."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, code in enumerate(drug_codes):
        if len(code) >= prefix_length:
            groups[code[:prefix_length]].append(index)
    return groups


def _class_edges(drug_codes: list[str]) -> list[tuple[int, int]]:
    """Connect drug classes that share a therapeutic class.

    Classes are grouped on the first three characters of the class code. A class
    alone in its group is then grouped on the first two characters instead, which
    connects it to a broader family rather than leaving it isolated.
    """
    groups = _class_groups(drug_codes, CLASS_PREFIX_LENGTH)
    alone = {members[0] for members in groups.values() if len(members) == 1}
    broader = {
        prefix: [index for index in members if index in alone]
        for prefix, members in _class_groups(drug_codes, BROADER_CLASS_PREFIX_LENGTH).items()
    }

    edges: list[tuple[int, int]] = []
    for members in list(groups.values()) + list(broader.values()):
        if len(members) < 2:
            continue
        edges.extend(
            (source, target) for source in members for target in members if source != target
        )
    return edges


def build_drug_graph(
    interaction_matrix: np.ndarray,
    coprescription_matrix: np.ndarray,
    drug_codes: list[str],
    coprescription_weights: np.ndarray | None = None,
    use_relations: bool = True,
) -> DrugGraph:
    """Assemble the drug graph from the two matrices and the class codes.

    Args:
        interaction_matrix: ``(drug_count, drug_count)``, non-zero where a pair
            of classes is a known interaction.
        coprescription_matrix: ``(drug_count, drug_count)`` co-prescription
            counts; a pair becomes an edge when its value exceeds
            ``COPRESCRIPTION_THRESHOLD``.
        drug_codes: class code of each drug index, used for the class relation.
        coprescription_weights: conditional probabilities from
            :func:`coprescription_probabilities`; when omitted every
            co-prescription edge weighs one.
        use_relations: when false only self-loops are kept, which is the
            ablation that removes all neighbour information while leaving the
            model unchanged.

    Returns:
        A :class:`DrugGraph` with symmetric interaction and class edges and
        directed co-prescription weights.
    """
    drug_count = int(interaction_matrix.shape[0])
    sources: list[int] = []
    targets: list[int] = []
    relations: list[int] = []

    def add(source: int, target: int, relation: int) -> None:
        sources.append(source)
        targets.append(target)
        relations.append(relation)

    if use_relations:
        interacting = np.argwhere(np.triu(interaction_matrix, k=1) > 0)
        for first, second in interacting:
            add(int(first), int(second), INTERACTION_RELATION)
            add(int(second), int(first), INTERACTION_RELATION)

        co_prescribed = np.argwhere(np.triu(coprescription_matrix, k=1) > COPRESCRIPTION_THRESHOLD)
        for first, second in co_prescribed:
            add(int(first), int(second), COPRESCRIPTION_RELATION)
            add(int(second), int(first), COPRESCRIPTION_RELATION)

    for node in range(drug_count):
        add(node, node, SELF_LOOP_RELATION)

    if use_relations:
        for source, target in _class_edges(drug_codes):
            add(source, target, DRUG_CLASS_RELATION)

    weights = np.ones(len(sources), dtype=np.float32)
    if coprescription_weights is not None:
        for position, relation in enumerate(relations):
            if relation == COPRESCRIPTION_RELATION:
                weight = coprescription_weights[sources[position], targets[position]]
                weights[position] = max(float(weight), MINIMUM_EDGE_WEIGHT)

    return DrugGraph(
        edge_index=torch.tensor([sources, targets], dtype=torch.long),
        edge_relation=torch.tensor(relations, dtype=torch.long),
        edge_weight=torch.tensor(weights, dtype=torch.float32),
    )
