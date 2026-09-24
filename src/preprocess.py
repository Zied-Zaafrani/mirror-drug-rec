"""Step 1 of the data preparation: raw MIMIC tables to one cohort folder.

    python src/preprocess.py --cohort mimic3 --mimic_dir <MIMIC-III> --drug_reference_dir <files>

Maps prescriptions to ATC-3 drug classes, keeps the admissions that have
diagnoses, procedures and prescriptions, orders each patient's admissions,
and writes the records, the vocabulary, the drug-interaction matrix, the
co-prescription matrix and the admission details (age, sex, stay).
"""

from __future__ import annotations

import argparse
import ast
import csv
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_cohort, make_paths
from dataset import (
    ADMISSION_ID_POSITION,
    CONTEXT_FILE,
    TRAIN_FRACTION,
    format_age,
    write_cohort_artifacts,
)

PRODUCT_TO_CONCEPT_FILE = "ndc_to_rxnorm.txt"
CONCEPT_TO_CLASS_FILE = "rxnorm_to_atc.csv"
STRUCTURE_FILE = "atc3_structures.pkl"
INTERACTION_FILE = "atc3_interactions.csv"

# A drug class is named by the first four characters of its therapeutic code,
# for example A02B. Longer codes name a single substance, which is finer than
# the level this model recommends at.
CLASS_CODE_LENGTH = 4
REQUIRED_FILES = (
    PRODUCT_TO_CONCEPT_FILE,
    CONCEPT_TO_CLASS_FILE,
    STRUCTURE_FILE,
    INTERACTION_FILE,
)


@dataclass
class DrugReference:
    """The reference data needed to turn product codes into drug classes.

    Attributes:
        product_to_class: product code to drug class code.
        structures: class code to the molecular structure of its representative
            drug, which the fingerprint is computed from.
        interacting_pairs: class-code pairs known to interact.
    """

    product_to_class: dict[str, str]
    structures: dict[str, str]
    interacting_pairs: set[tuple[str, str]]

    @property
    def classes_with_structure(self) -> set[str]:
        """Class codes that have a molecular structure, the ones kept in the vocabulary."""
        return set(self.structures)


def missing_reference_files(directory: Path) -> list[str]:
    """Return the names of the reference files that are absent from ``directory``."""
    directory = Path(directory)
    return [name for name in REQUIRED_FILES if not (directory / name).exists()]


def _read_product_to_concept(path: Path) -> dict[str, str]:
    """Read the product-code to drug-concept map.

    Two layouts are accepted, because the public sources ship both: one pair per
    line separated by a tab, or a single dictionary literal.
    """
    content = path.read_text(encoding="utf-8", errors="replace").strip()
    if content.startswith("{") or "u'" in content[:200]:
        literal = content if content.startswith("{") else "{" + content + "}"
        parsed = ast.literal_eval(literal)
        return {str(key): str(value) for key, value in parsed.items() if str(value)}

    mapping: dict[str, str] = {}
    for line in content.splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 2 and parts[0] and parts[1]:
            mapping[parts[0]] = parts[1]
    return mapping


def _read_concept_to_class(path: Path) -> dict[str, str]:
    """Read the drug-concept to therapeutic-class map.

    The first mapping listed for a concept is the one kept. The file order is
    part of the reference data, so it is not sorted first: sorting would change
    which class a concept with several entries ends up in, and with it the size
    of the drug vocabulary.
    """
    table = pd.read_csv(path, dtype=str)
    table.columns = [column.upper().strip() for column in table.columns]
    concept_column = next((c for c in table.columns if "RXCUI" in c), None)
    class_column = next((c for c in table.columns if "ATC" in c), None)
    if concept_column is None or class_column is None:
        raise ValueError(
            f"{path.name} needs a drug-concept column and a therapeutic-class column; "
            f"found {list(table.columns)}."
        )
    table = table.drop_duplicates(subset=[concept_column], keep="first")
    pairs = table[[concept_column, class_column]].dropna()
    return dict(zip(pairs[concept_column], pairs[class_column], strict=True))


def _read_structures(path: Path) -> dict[str, str]:
    """Read the class-code to molecular-structure dictionary."""
    with open(path, "rb") as handle:
        structures = pickle.load(handle)
    if not isinstance(structures, dict):
        raise ValueError(f"{path.name} must hold a dictionary of class code to structure.")
    return {str(code): str(structure) for code, structure in structures.items()}


def _read_interactions(path: Path) -> set[tuple[str, str]]:
    """Read the interacting class pairs, ordered within each pair."""
    table = pd.read_csv(path, dtype=str)
    columns = {column.lower().strip(): column for column in table.columns}
    first, second = columns.get("atc3_a"), columns.get("atc3_b")
    if first is None or second is None:
        raise ValueError(f"{path.name} needs the columns 'atc3_a' and 'atc3_b'.")
    pairs = table[[first, second]].dropna()
    return {
        (min(str(a), str(b)), max(str(a), str(b)))
        for a, b in zip(pairs[first], pairs[second], strict=True)
        if str(a) != str(b)
    }


def load_reference(directory: Path) -> DrugReference:
    """Load all four reference files.

    Raises:
        FileNotFoundError: one of the files is absent.
    """
    directory = Path(directory)
    missing = missing_reference_files(directory)
    if missing:
        raise FileNotFoundError(
            f"The drug reference directory {directory} is missing {missing}. "
            "See data/README.md for what each file contains."
        )

    concept_of_product = _read_product_to_concept(directory / PRODUCT_TO_CONCEPT_FILE)
    class_of_concept = _read_concept_to_class(directory / CONCEPT_TO_CLASS_FILE)
    product_to_class = {
        product: class_of_concept[concept][:CLASS_CODE_LENGTH]
        for product, concept in concept_of_product.items()
        if concept in class_of_concept
        and len(class_of_concept[concept]) >= CLASS_CODE_LENGTH
    }
    return DrugReference(
        product_to_class=product_to_class,
        structures=_read_structures(directory / STRUCTURE_FILE),
        interacting_pairs=_read_interactions(directory / INTERACTION_FILE),
    )


def interaction_matrix(
    reference: DrugReference,
    drug_codes: list[str],
) -> np.ndarray:
    """Build the interaction matrix for one drug vocabulary.

    Args:
        reference: the loaded reference data.
        drug_codes: class code of each drug index.

    Returns:
        A symmetric ``(drugs, drugs)`` matrix with one at interacting pairs.
    """
    index_of = {code: index for index, code in enumerate(drug_codes)}
    matrix = np.zeros((len(drug_codes), len(drug_codes)), dtype=np.float32)
    for first, second in reference.interacting_pairs:
        if first in index_of and second in index_of:
            matrix[index_of[first], index_of[second]] = 1.0
            matrix[index_of[second], index_of[first]] = 1.0
    return matrix


SUBJECT = "subject_id"
ADMISSION = "hadm_id"
ADMIT_TIME = "admit_time"
CODE = "code"
DRUG_CLASS = "drug_class"

DEFAULT_DRUG_VOCABULARY = 300
DEFAULT_DIAGNOSIS_VOCABULARY = 2000
MINIMUM_ADMISSIONS = 2
ADMISSION_ORDERS = ("time", "identifier")


@dataclass
class Cohort:
    """A built cohort, ready to be written to disk.

    Attributes:
        records: one entry per patient, each a list of admissions in the
            cohort's admission order, each admission ``[diagnosis_ids, procedure_ids, drug_ids,
            admission_id]``.
        diagnosis_codes, procedure_codes, drug_codes: the code behind each
            index, in index order.
        coprescription_counts: how often two drug classes share an admission,
            counted on the training patients only.
    """

    records: list[list[list]]
    diagnosis_codes: list[str]
    procedure_codes: list[str]
    drug_codes: list[str]
    coprescription_counts: np.ndarray

    @property
    def vocabulary(self) -> dict:
        """The vocabulary entry written beside the records."""
        return {
            "diagnosis_count": len(self.diagnosis_codes),
            "procedure_count": len(self.procedure_codes),
            "drug_codes": self.drug_codes,
            "diagnosis_codes": self.diagnosis_codes,
            "procedure_codes": self.procedure_codes,
        }

    def summary(self) -> dict[str, int | float]:
        """Counts worth printing after a build."""
        admissions = sum(len(patient) for patient in self.records)
        return {
            "patients": len(self.records),
            "admissions": admissions,
            "prediction_instances": admissions - len(self.records),
            "drug_classes": len(self.drug_codes),
            "diagnosis_codes": len(self.diagnosis_codes),
            "procedure_codes": len(self.procedure_codes),
            "drugs_per_admission": round(
                float(np.mean([len(a[2]) for p in self.records for a in p])), 2
            ),
        }


def map_products_to_classes(prescriptions: pd.DataFrame, reference: DrugReference) -> pd.DataFrame:
    """Replace product codes with drug classes and keep one row per class and admission."""
    mapped = prescriptions.copy()
    mapped[DRUG_CLASS] = mapped[CODE].map(reference.product_to_class)
    mapped = mapped.dropna(subset=[DRUG_CLASS])
    return mapped.drop_duplicates(subset=[ADMISSION, DRUG_CLASS])


def keep_patients_with_repeat_admissions(
    prescriptions: pd.DataFrame, minimum: int = MINIMUM_ADMISSIONS
) -> pd.DataFrame:
    """Drop patients with fewer than ``minimum`` admissions.

    A patient seen once has no history to read and no admission to predict, so
    they cannot form a prediction instance.
    """
    counts = (
        prescriptions[[SUBJECT, ADMISSION]]
        .drop_duplicates()
        .groupby(SUBJECT)[ADMISSION]
        .nunique()
    )
    keep = set(counts[counts >= minimum].index)
    return prescriptions[prescriptions[SUBJECT].isin(keep)].copy()


def select_drug_classes(
    prescriptions: pd.DataFrame,
    reference: DrugReference,
    vocabulary_size: int = DEFAULT_DRUG_VOCABULARY,
) -> pd.DataFrame:
    """Keep classes that have a molecular structure, then the most frequent ones.

    The structure filter comes first because the graph needs a fingerprint for
    every class it holds. Applying it after the frequency cut would leave the
    vocabulary short of the requested size for no stated reason.
    """
    with_structure = prescriptions[prescriptions[DRUG_CLASS].isin(reference.classes_with_structure)]
    frequent = set(with_structure[DRUG_CLASS].value_counts().head(vocabulary_size).index)
    return with_structure[with_structure[DRUG_CLASS].isin(frequent)].copy()


def select_frequent_codes(
    table: pd.DataFrame, vocabulary_size: int | None = DEFAULT_DIAGNOSIS_VOCABULARY
) -> pd.DataFrame:
    """Keep the ``vocabulary_size`` most frequent codes, or all of them when None."""
    if vocabulary_size is None:
        return table.copy()
    frequent = set(table[CODE].value_counts().head(vocabulary_size).index)
    return table[table[CODE].isin(frequent)].copy()


def _codes_per_admission(table: pd.DataFrame) -> dict[int, list[str]]:
    """Group a code table into one sorted list of codes per admission."""
    grouped = table.groupby(ADMISSION)[CODE].apply(lambda codes: sorted(set(codes)))
    return {int(admission): codes for admission, codes in grouped.items()}


def build_cohort(
    prescriptions: pd.DataFrame,
    diagnoses: pd.DataFrame,
    procedures: pd.DataFrame,
    admission_times: pd.DataFrame,
    train_fraction: float,
    admission_order: str = "time",
    minimum_admissions: int = MINIMUM_ADMISSIONS,
) -> Cohort:
    """Assemble the records, the vocabularies and the co-prescription counts.

    Args:
        prescriptions: rows of subject, admission and drug class.
        diagnoses: rows of subject, admission and diagnosis code.
        procedures: rows of subject, admission and procedure code.
        admission_times: admission identifier and its admission time.
        train_fraction: share of patients whose admissions the co-prescription
            counts are taken from; the same share the training partition uses,
            so the counts never see a test admission.
        admission_order: ``time`` or ``identifier``, how each patient's
            admissions are ordered.
        minimum_admissions: patients with fewer admissions are left out.

    Returns:
        The built :class:`Cohort`.

    Raises:
        ValueError: no admission has all three kinds of code, or
            ``admission_order`` is not one of ``ADMISSION_ORDERS``.
    """
    if admission_order not in ADMISSION_ORDERS:
        raise ValueError(f"admission_order must be one of {ADMISSION_ORDERS}.")

    drugs_by_admission = {
        int(admission): sorted(set(classes))
        for admission, classes in prescriptions.groupby(ADMISSION)[DRUG_CLASS]
        .apply(list)
        .items()
    }
    diagnoses_by_admission = _codes_per_admission(diagnoses)
    procedures_by_admission = _codes_per_admission(procedures)

    complete = (
        set(drugs_by_admission) & set(diagnoses_by_admission) & set(procedures_by_admission)
    )
    if not complete:
        raise ValueError(
            "No admission carries diagnoses, procedures and prescriptions at once. "
            "Check that the three tables come from the same release."
        )

    subject_of_admission = dict(
        zip(
            prescriptions[ADMISSION].astype(int),
            prescriptions[SUBJECT].astype(int),
            strict=True,
        )
    )
    time_of_admission = dict(
        zip(
            admission_times[ADMISSION].astype(int),
            pd.to_datetime(admission_times[ADMIT_TIME]),
            strict=True,
        )
    )

    admissions_by_patient: dict[int, list[int]] = defaultdict(list)
    for admission in complete:
        subject = subject_of_admission.get(admission)
        if subject is not None and admission in time_of_admission:
            admissions_by_patient[subject].append(admission)

    diagnosis_codes: set[str] = set()
    procedure_codes: set[str] = set()
    drug_codes: set[str] = set()
    ordered_patients: list[tuple[int, list[int]]] = []
    for subject in sorted(admissions_by_patient):
        if admission_order == "time":
            admissions = sorted(admissions_by_patient[subject], key=time_of_admission.get)
        else:
            admissions = sorted(admissions_by_patient[subject])
        if len(admissions) < minimum_admissions:
            continue
        ordered_patients.append((subject, admissions))
        for admission in admissions:
            diagnosis_codes.update(diagnoses_by_admission[admission])
            procedure_codes.update(procedures_by_admission[admission])
            drug_codes.update(drugs_by_admission[admission])

    diagnosis_index = {code: index for index, code in enumerate(sorted(diagnosis_codes))}
    procedure_index = {code: index for index, code in enumerate(sorted(procedure_codes))}
    drug_index = {code: index for index, code in enumerate(sorted(drug_codes))}

    records: list[list[list]] = []
    for _, admissions in ordered_patients:
        patient: list[list] = []
        for admission in admissions:
            patient.append(
                [
                    sorted(diagnosis_index[c] for c in diagnoses_by_admission[admission]),
                    sorted(procedure_index[c] for c in procedures_by_admission[admission]),
                    sorted(drug_index[c] for c in drugs_by_admission[admission]),
                    admission,
                ]
            )
        records.append(patient)

    training_patients = records[: int(len(records) * train_fraction)]
    return Cohort(
        records=records,
        diagnosis_codes=sorted(diagnosis_codes),
        procedure_codes=sorted(procedure_codes),
        drug_codes=sorted(drug_codes),
        coprescription_counts=coprescription_counts(training_patients, len(drug_index)),
    )


def coprescription_counts(records: list, drug_count: int) -> np.ndarray:
    """Count how often each pair of drug classes shares an admission.

    Args:
        records: the training patients only.
        drug_count: size of the drug vocabulary.

    Returns:
        A symmetric ``(drug_count, drug_count)`` matrix scaled so its largest
        entry is one, which is the scale the graph's threshold assumes.
    """
    counts = np.zeros((drug_count, drug_count), dtype=np.float32)
    for patient in records:
        for admission in patient:
            drugs = admission[2]
            for position, first in enumerate(drugs):
                for second in drugs[position + 1 :]:
                    counts[first, second] += 1.0
                    counts[second, first] += 1.0
    largest = counts.max() if counts.size else 0.0
    return counts / largest if largest > 0 else counts


COLUMNS = ("admission_id", "age", "sex", "admission_type", "stay_days")
SEX = {"M": "male", "F": "female"}
SECONDS_PER_DAY = 86400
DAYS_PER_YEAR = 365.25


def _mimic3(mimic_dir: Path) -> pd.DataFrame:
    """Admission table joined to the patient table, with the age in years."""
    admissions = pd.read_csv(
        mimic_dir / "ADMISSIONS.csv.gz",
        usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME", "ADMISSION_TYPE"],
        compression="gzip",
    )
    patients = pd.read_csv(
        mimic_dir / "PATIENTS.csv.gz", usecols=["SUBJECT_ID", "GENDER", "DOB"], compression="gzip"
    )
    table = admissions.merge(patients, on="SUBJECT_ID", how="left")
    for column in ("ADMITTIME", "DISCHTIME", "DOB"):
        table[column] = pd.to_datetime(table[column], errors="coerce")
    table["years"] = (table["ADMITTIME"] - table["DOB"]).dt.days / DAYS_PER_YEAR
    return table.rename(
        columns={
            "HADM_ID": "admission",
            "ADMITTIME": "admitted",
            "DISCHTIME": "discharged",
            "ADMISSION_TYPE": "admission_type",
            "GENDER": "gender",
        }
    )


def _mimic4(mimic_dir: Path) -> pd.DataFrame:
    """Admission table joined to the patient table, with the age in years.

    The newer source gives each patient an age at one reference year, so the age
    at an admission is that age plus the years between the two.
    """
    admissions = pd.read_csv(
        mimic_dir / "hosp" / "admissions.csv.gz",
        usecols=["subject_id", "hadm_id", "admittime", "dischtime", "admission_type"],
        compression="gzip",
    )
    patients = pd.read_csv(
        mimic_dir / "hosp" / "patients.csv.gz",
        usecols=["subject_id", "gender", "anchor_age", "anchor_year"],
        compression="gzip",
    )
    table = admissions.merge(patients, on="subject_id", how="left")
    for column in ("admittime", "dischtime"):
        table[column] = pd.to_datetime(table[column], errors="coerce")
    table["years"] = table["anchor_age"] + (table["admittime"].dt.year - table["anchor_year"])
    return table.rename(
        columns={"hadm_id": "admission", "admittime": "admitted", "dischtime": "discharged"}
    )


def admission_context(mimic_dir: Path, source: str, admission_ids: set[int]) -> list[dict]:
    """Describe each requested admission.

    Args:
        mimic_dir: directory holding the raw tables.
        source: ``mimic3`` or ``mimic4``.
        admission_ids: the admissions to describe.

    Returns:
        One dictionary per admission with the keys in ``COLUMNS``; a value the
        tables do not hold is an empty string.
    """
    table = _mimic3(Path(mimic_dir)) if source == "mimic3" else _mimic4(Path(mimic_dir))
    table = table[table["admission"].astype(int).isin(admission_ids)]

    rows = []
    for row in table.itertuples():
        stay = (
            (row.discharged - row.admitted).total_seconds() / SECONDS_PER_DAY
            if pd.notna(row.discharged) and pd.notna(row.admitted)
            else None
        )
        rows.append(
            {
                "admission_id": int(row.admission),
                "age": format_age(float(row.years)) if pd.notna(row.years) else "",
                "sex": SEX.get(str(row.gender), ""),
                "admission_type": str(row.admission_type or "").capitalize(),
                "stay_days": f"{stay:.1f}" if stay is not None else "",
            }
        )
    return rows


def write_admission_context(cohort_dir: Path, rows: list[dict]) -> Path:
    """Write the admission details for one cohort and return the path."""
    path = Path(cohort_dir) / CONTEXT_FILE
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return path


MIMIC3_PRESCRIPTION_TABLE = "PRESCRIPTIONS.csv.gz"
MIMIC3_DIAGNOSIS_TABLE = "DIAGNOSES_ICD.csv.gz"
MIMIC3_PROCEDURE_TABLE = "PROCEDURES_ICD.csv.gz"
MIMIC3_ADMISSION_TABLE = "ADMISSIONS.csv.gz"
UNKNOWN_PRODUCT_CODE = "0"


def read_mimic3_prescriptions(mimic_dir: Path) -> pd.DataFrame:
    """Read the prescription table as rows of subject, admission and product code."""
    table = pd.read_csv(
        mimic_dir / MIMIC3_PRESCRIPTION_TABLE,
        usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "STARTDATE", "NDC"],
        compression="gzip",
        dtype={"NDC": str},
    ).dropna(subset=["SUBJECT_ID", "HADM_ID"])

    table["SUBJECT_ID"] = table["SUBJECT_ID"].astype(int)
    table["HADM_ID"] = table["HADM_ID"].astype(int)
    table["ICUSTAY_ID"] = pd.to_numeric(table["ICUSTAY_ID"], errors="coerce")
    table["STARTDATE"] = pd.to_datetime(table["STARTDATE"], errors="coerce")
    table = table.sort_values(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "STARTDATE"])

    table["NDC"] = table["NDC"].astype(str).str.strip()
    table = table[table["NDC"] != UNKNOWN_PRODUCT_CODE]
    table["NDC"] = table["NDC"].replace({"": np.nan, "nan": np.nan})
    table["NDC"] = table.groupby("SUBJECT_ID")["NDC"].ffill()
    table = table.dropna(subset=["NDC"])

    return (
        table[["SUBJECT_ID", "HADM_ID", "NDC"]]
        .rename(columns={"SUBJECT_ID": SUBJECT, "HADM_ID": ADMISSION, "NDC": CODE})
        .drop_duplicates()
    )


def _read_mimic3_code_table(path: Path) -> pd.DataFrame:
    """Read a diagnosis or procedure table as rows of subject, admission and code."""
    table = pd.read_csv(
        path,
        usecols=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"],
        compression="gzip",
        dtype={"ICD9_CODE": str},
    ).dropna(subset=["HADM_ID", "ICD9_CODE"])
    table["HADM_ID"] = table["HADM_ID"].astype(int)
    return table.rename(
        columns={"SUBJECT_ID": SUBJECT, "HADM_ID": ADMISSION, "ICD9_CODE": CODE}
    )


def read_diagnoses(mimic_dir: Path) -> pd.DataFrame:
    """Read the diagnosis table."""
    return _read_mimic3_code_table(mimic_dir / MIMIC3_DIAGNOSIS_TABLE)


def read_procedures(mimic_dir: Path) -> pd.DataFrame:
    """Read the procedure table."""
    return _read_mimic3_code_table(mimic_dir / MIMIC3_PROCEDURE_TABLE)


def read_mimic3_admission_times(mimic_dir: Path) -> pd.DataFrame:
    """Read each admission's start time, which orders a patient's admissions."""
    table = pd.read_csv(
        mimic_dir / MIMIC3_ADMISSION_TABLE,
        usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME"],
        compression="gzip",
    ).dropna(subset=["HADM_ID", "ADMITTIME"])
    table["HADM_ID"] = table["HADM_ID"].astype(int)
    table["ADMITTIME"] = pd.to_datetime(table["ADMITTIME"], errors="coerce")
    table = table.dropna(subset=["ADMITTIME"]).drop_duplicates(subset=["HADM_ID"])
    return table.rename(columns={"HADM_ID": ADMISSION, "ADMITTIME": ADMIT_TIME})[
        [ADMISSION, ADMIT_TIME]
    ]


def build_mimic3(
    mimic_dir: Path,
    reference: DrugReference,
    train_fraction: float,
    drug_vocabulary_size: int = 300,
    diagnosis_vocabulary_size: int = 2000,
    admission_order: str = "time",
    minimum_admissions: int = 2,
    report=print,
) -> Cohort:
    """Build the MIMIC-III cohort.

    Args:
        mimic_dir: directory holding the raw tables.
        reference: the loaded drug reference data.
        train_fraction: share of patients the co-prescription counts come from.
        drug_vocabulary_size: how many drug classes to keep before the structure
            filter narrows them further.
        diagnosis_vocabulary_size: how many diagnosis codes to keep.
        admission_order: ``time`` or ``identifier``, how each patient's
            admissions are ordered.
        minimum_admissions: patients with fewer admissions are left out of the
            records.
        report: where progress lines go.

    Returns:
        The built cohort.
    """
    mimic_dir = Path(mimic_dir)
    report(f"Reading tables from {mimic_dir}")
    prescriptions = read_mimic3_prescriptions(mimic_dir)
    diagnoses = read_diagnoses(mimic_dir)
    procedures = read_procedures(mimic_dir)
    admission_times = read_mimic3_admission_times(mimic_dir)
    report(
        f"  {len(prescriptions):,} prescription rows, {len(diagnoses):,} diagnosis rows, "
        f"{len(procedures):,} procedure rows"
    )

    prescriptions = keep_patients_with_repeat_admissions(prescriptions)
    prescriptions = map_products_to_classes(prescriptions, reference)
    prescriptions = select_drug_classes(prescriptions, reference, drug_vocabulary_size)
    report(f"  {prescriptions['drug_class'].nunique()} drug classes kept")

    diagnoses = select_frequent_codes(diagnoses, diagnosis_vocabulary_size)
    procedures = select_frequent_codes(procedures, None)

    cohort = build_cohort(
        prescriptions=prescriptions,
        diagnoses=diagnoses,
        procedures=procedures,
        admission_times=admission_times,
        train_fraction=train_fraction,
        admission_order=admission_order,
        minimum_admissions=minimum_admissions,
    )
    for name, value in cohort.summary().items():
        report(f"  {name}: {value}")
    return cohort


MIMIC4_PRESCRIPTION_TABLE = "hosp/prescriptions.csv.gz"
MIMIC4_DIAGNOSIS_TABLE = "hosp/diagnoses_icd.csv.gz"
MIMIC4_PROCEDURE_TABLE = "hosp/procedures_icd.csv.gz"
MIMIC4_ADMISSION_TABLE = "hosp/admissions.csv.gz"
INTENSIVE_CARE_TABLE = "icu/icustays.csv.gz"

OLDER_REVISION = 9
COHORTS = ("mimic4_icu", "mimic4_hospital", "mimic4_mixed")


def read_mimic4_prescriptions(mimic_dir: Path) -> pd.DataFrame:
    """Read the prescription table as rows of subject, admission and product code."""
    table = pd.read_csv(
        mimic_dir / MIMIC4_PRESCRIPTION_TABLE,
        usecols=["subject_id", "hadm_id", "ndc"],
        compression="gzip",
        dtype={"ndc": str},
    ).dropna(subset=["subject_id", "hadm_id", "ndc"])
    table["subject_id"] = table["subject_id"].astype(int)
    table["hadm_id"] = table["hadm_id"].astype(int)
    table["ndc"] = table["ndc"].astype(str).str.strip().str.zfill(11)
    return table.rename(columns={"ndc": CODE}).drop_duplicates()


def _read_mimic4_code_table(path: Path, revisions: tuple[int, ...]) -> pd.DataFrame:
    """Read a diagnosis or procedure table, keeping the requested code revisions.

    The revision is kept in the code itself, because the same digits mean
    different things in the two revisions.
    """
    table = pd.read_csv(
        path,
        usecols=["subject_id", "hadm_id", "icd_code", "icd_version"],
        compression="gzip",
        dtype={"icd_code": str, "icd_version": int},
    ).dropna(subset=["hadm_id", "icd_code"])
    table["hadm_id"] = table["hadm_id"].astype(int)
    table = table[table["icd_version"].isin(revisions)]
    table[CODE] = (
        "icd" + table["icd_version"].astype(str) + "_" + table["icd_code"].str.strip()
    )
    return table[[SUBJECT, ADMISSION, CODE]]


def read_mimic4_admission_times(mimic_dir: Path) -> pd.DataFrame:
    """Read each admission's start time, which orders a patient's admissions."""
    table = pd.read_csv(
        mimic_dir / MIMIC4_ADMISSION_TABLE,
        usecols=["hadm_id", "admittime"],
        compression="gzip",
    ).dropna(subset=["hadm_id", "admittime"])
    table["hadm_id"] = table["hadm_id"].astype(int)
    table["admittime"] = pd.to_datetime(table["admittime"], errors="coerce")
    table = table.dropna(subset=["admittime"]).drop_duplicates(subset=["hadm_id"])
    return table.rename(columns={"admittime": ADMIT_TIME})[[ADMISSION, ADMIT_TIME]]


def read_intensive_care_admissions(mimic_dir: Path) -> set[int]:
    """Return the admissions that include a stay on an intensive-care unit."""
    path = mimic_dir / INTENSIVE_CARE_TABLE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is required for the intensive-care cohort. Download the unit "
            "tables of the release, or build one of the other two cohorts."
        )
    stays = pd.read_csv(path, usecols=["hadm_id"], compression="gzip").dropna()
    return set(stays["hadm_id"].astype(int))


def build_mimic4(
    cohort_name: str,
    mimic_dir: Path,
    reference: DrugReference,
    train_fraction: float,
    drug_vocabulary_size: int = 300,
    diagnosis_vocabulary_size: int = 2000,
    admission_order: str = "time",
    minimum_admissions: int = 2,
    report=print,
) -> Cohort:
    """Build one of the three MIMIC-IV cohorts.

    Args:
        cohort_name: one of ``mimic4_icu``, ``mimic4_hospital``, ``mimic4_mixed``.
        mimic_dir: directory holding the raw tables.
        reference: the loaded drug reference data.
        train_fraction: share of patients the co-prescription counts come from.
        drug_vocabulary_size: how many drug classes to keep before the structure
            filter narrows them further.
        diagnosis_vocabulary_size: how many diagnosis codes to keep.
        admission_order: ``time`` or ``identifier``, how each patient's
            admissions are ordered.
        minimum_admissions: patients with fewer admissions are left out of the
            records.
        report: where progress lines go.

    Returns:
        The built cohort.

    Raises:
        ValueError: the cohort name is not one of the three.
    """
    if cohort_name not in COHORTS:
        raise ValueError(f"Unknown MIMIC-IV cohort '{cohort_name}'. Choose from {COHORTS}.")

    mimic_dir = Path(mimic_dir)
    revisions = (OLDER_REVISION,) if cohort_name == "mimic4_hospital" else (9, 10)
    report(f"Reading tables from {mimic_dir} for {cohort_name}")
    prescriptions = read_mimic4_prescriptions(mimic_dir)
    diagnoses = _read_mimic4_code_table(mimic_dir / MIMIC4_DIAGNOSIS_TABLE, revisions)
    procedures = _read_mimic4_code_table(mimic_dir / MIMIC4_PROCEDURE_TABLE, revisions)
    admission_times = read_mimic4_admission_times(mimic_dir)

    if cohort_name == "mimic4_icu":
        intensive_care = read_intensive_care_admissions(mimic_dir)
        report(f"  {len(intensive_care):,} admissions with an intensive-care stay")
        prescriptions = prescriptions[prescriptions[ADMISSION].isin(intensive_care)]
        diagnoses = diagnoses[diagnoses[ADMISSION].isin(intensive_care)]
        procedures = procedures[procedures[ADMISSION].isin(intensive_care)]

    prescriptions = keep_patients_with_repeat_admissions(prescriptions)
    prescriptions = map_products_to_classes(prescriptions, reference)
    prescriptions = select_drug_classes(prescriptions, reference, drug_vocabulary_size)
    report(f"  {prescriptions['drug_class'].nunique()} drug classes kept")

    diagnoses = select_frequent_codes(diagnoses, diagnosis_vocabulary_size)
    procedures = select_frequent_codes(procedures, None)

    cohort = build_cohort(
        prescriptions=prescriptions,
        diagnoses=diagnoses,
        procedures=procedures,
        admission_times=admission_times,
        train_fraction=train_fraction,
        admission_order=admission_order,
        minimum_admissions=minimum_admissions,
    )
    for name, value in cohort.summary().items():
        report(f"  {name}: {value}")
    return cohort


def main(argv: list[str] | None = None) -> int:
    """Build one cohort's records, vocabulary, drug matrices and admission details."""
    parser = argparse.ArgumentParser(description="Build one cohort from the raw MIMIC tables.")
    parser.add_argument("--cohort", required=True, help="a cohort listed in config.yaml")
    parser.add_argument("--mimic_dir", type=Path, required=True,
                        help="MIMIC-III folder (ADMISSIONS.csv.gz, ...) "
                             "or MIMIC-IV folder (hosp/, icu/)")
    parser.add_argument("--drug_reference_dir", type=Path, required=True,
                        help="folder with the drug mapping, structure and interaction files")
    parser.add_argument("--data_dir", type=Path, default=None,
                        help="where the cohort folder is written (default: data/processed)")
    parser.add_argument("--drug_vocabulary", type=int, default=300,
                        help="drug classes to keep before the structure filter")
    parser.add_argument("--diagnosis_vocabulary", type=int, default=2000,
                        help="diagnosis codes to keep")
    arguments = parser.parse_args(argv)

    cohort = load_cohort(arguments.cohort)
    paths = make_paths(arguments.data_dir, None, arguments.mimic_dir,
                       arguments.drug_reference_dir, cohort.source)
    missing = missing_reference_files(paths.drug_reference)
    if missing:
        print(f"Missing reference files in {paths.drug_reference}: {missing}. See data/README.md.")
        return 1
    reference = load_reference(paths.drug_reference)
    print(f"Reference: {len(reference.product_to_class):,} product codes, "
          f"{len(reference.structures):,} classes with a structure, "
          f"{len(reference.interacting_pairs):,} interacting pairs")

    build = build_mimic3 if cohort.source == "mimic3" else build_mimic4
    options = {"cohort_name": cohort.name} if cohort.source == "mimic4" else {}
    built = build(
        mimic_dir=arguments.mimic_dir,
        reference=reference,
        train_fraction=TRAIN_FRACTION,
        drug_vocabulary_size=arguments.drug_vocabulary,
        diagnosis_vocabulary_size=arguments.diagnosis_vocabulary,
        admission_order=cohort.admission_order,
        minimum_admissions=cohort.minimum_admissions,
        **options,
    )
    cohort_dir = paths.cohort_dir(cohort.directory)
    write_cohort_artifacts(
        cohort_dir=cohort_dir,
        records=built.records,
        vocabulary=built.vocabulary,
        interaction_matrix=interaction_matrix(reference, built.drug_codes),
        coprescription_matrix=built.coprescription_counts,
    )
    admissions = {int(a[ADMISSION_ID_POSITION]) for patient in built.records for a in patient}
    context = admission_context(arguments.mimic_dir, cohort.source, admissions)
    write_admission_context(cohort_dir, context)
    print(f"Cohort written to {cohort_dir}")
    print(f"Next: python src/features.py --cohort {cohort.name} "
          "--mimic_dir ... --drug_reference_dir ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
