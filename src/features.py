"""Step 2 of the data preparation: the three input channels of one cohort.

    python src/features.py --cohort mimic3 --mimic_dir <MIMIC-III> --drug_reference_dir <files>

Embeds the diagnosis, procedure and drug descriptions with PubMedBERT and the
drug structures as Morgan fingerprints, embeds each discharge summary with
Bio_ClinicalBERT (medication sections removed), and builds the laboratory
vectors (last value of each test, z-scored with training statistics, plus a
missing flag). The two language-model steps are the slow ones; run them on a
graphics card.
"""

from __future__ import annotations

import argparse
import pickle
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import load_cohort, make_paths
from dataset import (
    ADMISSION_ID_POSITION,
    CODE_EMBEDDING_FILE,
    DIAGNOSIS,
    DRUG,
    LAB_FEATURE_FILE,
    NOTE_EMBEDDING_FILE,
    NOTE_MEAN_FILE,
    PROCEDURE,
    RECORDS_FILE,
    TRAIN_FRACTION,
    VOCABULARY_FILE,
    write_labels,
)
from preprocess import DrugReference, load_reference


@dataclass(frozen=True)
class ReferenceRange:
    """Bounds for one laboratory test.

    Attributes:
        name: the test's name.
        error_low, error_high: outside these, a value is a recording error.
        valid_low, valid_high: outside these but inside the error bounds, a
            value is real and clipped to the bound.
        unit: the unit the bounds are given in.
    """

    name: str
    error_low: float
    valid_low: float
    valid_high: float
    error_high: float
    unit: str


REFERENCE_RANGES: dict[int, ReferenceRange] = {
    50912: ReferenceRange("Creatinine", 0, 0.1, 60, 66, "mg/dL"),
    51006: ReferenceRange("Urea nitrogen", 0, 0, 250, 275, "mg/dL"),
    50861: ReferenceRange("Alanine aminotransferase", 0, 2, 10000, 11000, "IU/L"),
    50878: ReferenceRange("Aspartate aminotransferase", 0, 6, 20000, 22000, "IU/L"),
    50885: ReferenceRange("Bilirubin", 0, 0.1, 60, 66, "mg/dL"),
    50863: ReferenceRange("Alkaline phosphatase", 0, 20, 3625, 4000, "IU/L"),
    51237: ReferenceRange("International normalised ratio", 0, 0.5, 20, 50, "ratio"),
    51274: ReferenceRange("Prothrombin time", 0, 9.9, 97.1, 150, "seconds"),
    51275: ReferenceRange("Partial thromboplastin time", 0, 18.8, 150, 150, "seconds"),
    50983: ReferenceRange("Sodium", 0, 50, 225, 250, "mEq/L"),
    50971: ReferenceRange("Potassium", 0, 0, 12, 15, "mEq/L"),
    50960: ReferenceRange("Magnesium", 0, 0, 20, 22, "mg/dL"),
    50893: ReferenceRange("Calcium", 0, 4.0, 20, 40, "mg/dL"),
    50931: ReferenceRange("Glucose", 0, 33, 2000, 2200, "mg/dL"),
    50862: ReferenceRange("Albumin", 0, 0.6, 6, 60, "g/dL"),
    50813: ReferenceRange("Lactate", 0, 0.4, 30, 33, "mmol/L"),
    51301: ReferenceRange("White blood cells", 0, 0, 1000, 1100, "K/uL"),
    51222: ReferenceRange("Haemoglobin", 0, 0, 25, 30, "g/dL"),
}


def bound_value(test_id: int, value: float) -> float | None:
    """Apply the bounds of one test to one value.

    Returns:
        The value, clipped to the valid range when it falls outside it, or None
        when it is outside the error bounds and should count as not measured.
        A test with no published bounds returns the value unchanged.
    """
    limits = REFERENCE_RANGES.get(test_id)
    if limits is None:
        return value
    if value < limits.error_low or value > limits.error_high:
        return None
    return min(max(value, limits.valid_low), limits.valid_high)


TEXT_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
MAX_DESCRIPTION_TOKENS = 64
CODE_BATCH_SIZE = 64
CODE_EMBEDDING_DIM = 768


def read_code_descriptions(mimic_dir: Path, source: str) -> dict[str, str]:
    """Read the written description of every diagnosis and procedure code.

    Args:
        mimic_dir: directory holding the raw tables.
        source: ``mimic3`` or ``mimic4``; the two releases name these files and
            their columns differently.

    Returns:
        Code to description, keyed by the code without a revision marker.
    """
    descriptions: dict[str, str] = {}
    if source == "mimic3":
        for name in ("D_ICD_DIAGNOSES.csv.gz", "D_ICD_PROCEDURES.csv.gz"):
            path = mimic_dir / name
            if not path.exists():
                continue
            table = pd.read_csv(
                path, usecols=["ICD9_CODE", "LONG_TITLE", "SHORT_TITLE"],
                compression="gzip", dtype=str,
            )
            for code, long_title, short_title in zip(
                table["ICD9_CODE"], table["LONG_TITLE"], table["SHORT_TITLE"], strict=True
            ):
                if not isinstance(code, str):
                    continue
                title = long_title if isinstance(long_title, str) else short_title
                if isinstance(title, str):
                    descriptions.setdefault(code.strip(), title.strip())
    else:
        for name in ("d_icd_diagnoses.csv.gz", "d_icd_procedures.csv.gz"):
            path = mimic_dir / "hosp" / name
            if not path.exists():
                continue
            table = pd.read_csv(
                path, usecols=["icd_code", "long_title"], compression="gzip", dtype=str
            ).dropna()
            for code, title in zip(table["icd_code"], table["long_title"], strict=True):
                descriptions.setdefault(code.strip(), title.strip())
    return descriptions


def describe_code(code: str, descriptions: dict[str, str]) -> str:
    """Return the description of one code, trying the ways codes are written.

    Codes reach this function with a revision marker and in several spellings:
    with or without the decimal point, and with or without leading zeros. The
    code itself is returned when no description matches, which still gives the
    language model something to read.
    """
    bare = code.split("_", 1)[1] if code.startswith(("icd9_", "icd10_")) else code
    bare = bare.strip()
    for candidate in (bare, bare.replace(".", ""), bare.zfill(5), bare.lstrip("0") or "0"):
        if candidate in descriptions:
            return descriptions[candidate]
    return f"clinical code {bare}"


def describe_drug_class(code: str, class_names: dict[str, str]) -> str:
    """Return the description of one drug class, or its code when unnamed."""
    return class_names.get(code, f"drug class {code}")


def embed_descriptions(
    texts: list[str],
    device: str = "cpu",
    model_name: str = TEXT_MODEL,
    batch_size: int = CODE_BATCH_SIZE,
    report=print,
) -> np.ndarray:
    """Embed a list of descriptions with a biomedical language model.

    The first output position of each description is taken as its vector, which
    is the position the model is trained to summarise a sequence in.

    Args:
        texts: one description per code.
        device: where the language model runs.
        model_name: the published model to load.
        batch_size: descriptions per forward pass.
        report: where progress lines go.

    Returns:
        ``(len(texts), 768)``.
    """
    from transformers import AutoModel, AutoTokenizer

    report(f"Embedding {len(texts):,} descriptions with {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    vectors = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=MAX_DESCRIPTION_TOKENS,
                return_tensors="pt",
            ).to(device)
            output = model(**batch).last_hidden_state[:, 0, :]
            vectors.append(output.cpu().float().numpy())
    return np.concatenate(vectors, axis=0).astype(np.float32)


def molecular_fingerprints(
    drug_codes: list[str],
    reference: DrugReference,
    bits: int = 256,
    radius: int = 2,
    report=print,
) -> np.ndarray:
    """Compute a binary structural fingerprint per drug class.

    A class stands for several substances; the first structure the reference
    lists for it is used. A class with no structure gets a row of zeros, which
    the graph reads as no structural information rather than as a wrong one.

    Args:
        drug_codes: class code of each drug index.
        reference: the loaded drug reference data.
        bits: length of the fingerprint.
        radius: how far around each atom the fingerprint looks.
        report: where progress lines go.

    Returns:
        ``(len(drug_codes), bits)``.
    """
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    fingerprints = np.zeros((len(drug_codes), bits), dtype=np.float32)
    described = 0
    for index, code in enumerate(drug_codes):
        structures = reference.structures.get(code)
        if structures is None:
            continue
        if isinstance(structures, str):
            structures = [structures]
        for structure in structures:
            molecule = Chem.MolFromSmiles(structure)
            if molecule is not None:
                fingerprints[index] = np.asarray(
                    generator.GetFingerprint(molecule), dtype=np.float32
                )
                described += 1
                break
    report(f"  {described}/{len(drug_codes)} drug classes have a molecular structure")
    return fingerprints


def build_code_embeddings(
    cohort_dir: Path,
    mimic_dir: Path,
    source: str,
    diagnosis_codes: list[str],
    procedure_codes: list[str],
    drug_codes: list[str],
    reference: DrugReference,
    class_names: dict[str, str] | None = None,
    device: str = "cpu",
    bits: int = 256,
    radius: int = 2,
    report=print,
) -> Path:
    """Write the embedding file for one cohort and return its path.

    The descriptions read here are also written beside it as a name per code, so
    that output can say what a code means instead of only naming it.
    """
    descriptions = read_code_descriptions(Path(mimic_dir), source)
    report(f"  {len(descriptions):,} code descriptions available")

    diagnosis_texts = [describe_code(code, descriptions) for code in diagnosis_codes]
    procedure_texts = [describe_code(code, descriptions) for code in procedure_codes]
    drug_texts = [describe_drug_class(code, class_names or {}) for code in drug_codes]

    label_path = write_labels(
        cohort_dir,
        [
            (kind, code, text)
            for kind, codes, texts in (
                (DIAGNOSIS, diagnosis_codes, diagnosis_texts),
                (PROCEDURE, procedure_codes, procedure_texts),
                (DRUG, drug_codes, drug_texts),
            )
            for code, text in zip(codes, texts, strict=True)
        ],
    )
    report(f"  names for {len(diagnosis_codes) + len(procedure_codes) + len(drug_codes):,} "
           f"codes written to {label_path.name}")

    embedded = embed_descriptions(
        diagnosis_texts + procedure_texts + drug_texts, device=device, report=report
    )
    first = len(diagnosis_texts)
    second = first + len(procedure_texts)

    path = Path(cohort_dir) / CODE_EMBEDDING_FILE
    torch.save(
        {
            "diagnosis": torch.from_numpy(embedded[:first]),
            "procedure": torch.from_numpy(embedded[first:second]),
            "drug": torch.from_numpy(embedded[second:]),
            "molecular_fingerprint": torch.from_numpy(
                molecular_fingerprints(drug_codes, reference, bits, radius, report)
            ),
            "model_name": TEXT_MODEL,
        },
        path,
    )
    report(f"  written to {path}")
    return path


NOTE_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
WINDOW_TOKENS = 512
WINDOW_OVERLAP = 128
NOTE_BATCH_SIZE = 8
NOTE_EMBEDDING_DIM = 768
REMOVED_MARKER = "[medication section removed]"

_NEXT_HEADING = r"(?=\n\s*(?:[A-Z][A-Za-z][A-Za-z\-/ ]{2,}:|[A-Z][A-Z][A-Z\-/ ]{2,})|\Z)"
MEDICATION_SECTIONS = [
    rf"(?is)discharge\s+medications?\s*:.*?{_NEXT_HEADING}",
    rf"(?is)medications?\s+(?:on|at)\s+discharge\s*:.*?{_NEXT_HEADING}",
    rf"(?is)discharge\s+meds?\s*:.*?{_NEXT_HEADING}",
    rf"(?is)patient'?s?\s+medications?\s+(?:on|at)\s+discharge\s*:.*?{_NEXT_HEADING}",
    rf"(?is)medications?\s+on\s+admission\s*:.*?{_NEXT_HEADING}",
    rf"(?is)admission\s+medications?\s*:.*?{_NEXT_HEADING}",
    rf"(?is)home\s+medications?\s*:.*?{_NEXT_HEADING}",
    rf"(?is)outpatient\s+medications?\s*:.*?{_NEXT_HEADING}",
    rf"(?is)pre-?(?:admission|hospital)\s+medications?\s*:.*?{_NEXT_HEADING}",
    rf"(?is)current\s+medications?\s*:.*?{_NEXT_HEADING}",
    rf"(?is)medications?\s+prior\s+to\s+admission\s*:.*?{_NEXT_HEADING}",
]
REMAINING_MEDICATION_HEADINGS = [
    r"(?i)discharge\s+medications?\s*:",
    r"(?i)medications?\s+(?:on|at)\s+discharge\s*:",
    r"(?i)discharge\s+meds?\s*:",
    r"(?i)medications?\s+on\s+admission\s*:",
    r"(?i)admission\s+medications?\s*:",
    r"(?i)home\s+medications?\s*:",
]


def remove_medication_sections(text: str) -> str:
    """Cut every medication list out of one summary."""
    for pattern in MEDICATION_SECTIONS:
        text = re.sub(pattern, REMOVED_MARKER, text, flags=re.DOTALL)
    return re.sub(r"\s+", " ", text).strip()


def count_remaining_medication_headings(texts: list[str]) -> int:
    """Count summaries that still contain a medication heading after the removal.

    Reported after the removal step so that a change in the note format shows up
    as a number rather than as a silent leak.
    """
    return sum(
        any(re.search(pattern, text) for pattern in REMAINING_MEDICATION_HEADINGS)
        for text in texts
    )


def read_discharge_summaries(
    mimic_dir: Path,
    source: str,
    admission_ids: set[int],
    chunk_size: int = 50_000,
    report=print,
) -> dict[int, str]:
    """Read the discharge summary of each requested admission.

    The note table is several gigabytes, so it is read in chunks and filtered to
    the cohort as it goes.

    Args:
        mimic_dir: directory holding the note table.
        source: ``mimic3`` or ``mimic4``.
        admission_ids: the admissions to keep.
        chunk_size: rows per chunk.
        report: where progress lines go.

    Returns:
        Admission identifier to the text of its summary.

    Raises:
        FileNotFoundError: the note table is not where it is expected.
    """
    import pandas as pd

    mimic_dir = Path(mimic_dir)
    summaries: dict[int, str] = {}

    if source == "mimic3":
        path = mimic_dir / "NOTEEVENTS.csv.gz"
        if not path.exists():
            raise FileNotFoundError(f"{path} is required for the note channel.")
        columns = ["HADM_ID", "CATEGORY", "ISERROR", "TEXT"]
        for chunk in pd.read_csv(
            path, usecols=columns, compression="gzip", chunksize=chunk_size
        ):
            chunk = chunk[chunk["CATEGORY"] == "Discharge summary"]
            chunk = chunk[chunk["ISERROR"] != 1].dropna(subset=["HADM_ID", "TEXT"])
            for admission, text in zip(chunk["HADM_ID"].astype(int), chunk["TEXT"], strict=True):
                if admission in admission_ids and admission not in summaries:
                    summaries[admission] = text
    else:
        path = mimic_dir / "discharge.csv.gz"
        if not path.exists():
            path = mimic_dir / "note" / "discharge.csv.gz"
        if not path.exists():
            raise FileNotFoundError(
                f"discharge.csv.gz was not found under {mimic_dir}. The note release is a "
                "separate download from the main tables."
            )
        for chunk in pd.read_csv(
            path, usecols=["hadm_id", "text"], compression="gzip", chunksize=chunk_size
        ):
            chunk = chunk.dropna(subset=["hadm_id", "text"])
            for admission, text in zip(chunk["hadm_id"].astype(int), chunk["text"], strict=True):
                if admission in admission_ids and admission not in summaries:
                    summaries[admission] = text

    report(f"  {len(summaries):,} of {len(admission_ids):,} admissions have a summary")
    return summaries


def _windows(tokens: list[int], size: int, overlap: int) -> list[list[int]]:
    """Split a token list into overlapping windows."""
    if len(tokens) <= size:
        return [tokens]
    step = size - overlap
    return [
        tokens[start : start + size]
        for start in range(0, len(tokens), step)
        if tokens[start : start + size]
    ]


def embed_summaries(
    summaries: dict[int, str],
    admission_ids: list[int],
    device: str = "cpu",
    model_name: str = NOTE_MODEL,
    batch_size: int = NOTE_BATCH_SIZE,
    report=print,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed one summary per admission.

    Args:
        summaries: admission identifier to cleaned text.
        admission_ids: every admission of the cohort, in the order the output
            rows follow.
        device: where the language model runs.
        model_name: the published clinical language model to load.
        batch_size: windows per forward pass.
        report: where progress lines go.

    Returns:
        The ``(admissions, 768)`` vectors and an ``(admissions,)`` flag that is
        true where a summary was found.
    """
    from transformers import AutoModel, AutoTokenizer

    report(f"Embedding {len(summaries):,} summaries with {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    vectors = np.zeros((len(admission_ids), NOTE_EMBEDDING_DIM), dtype=np.float32)
    found = np.zeros(len(admission_ids), dtype=bool)

    with torch.no_grad():
        for row, admission in enumerate(admission_ids):
            text = summaries.get(admission)
            if not text:
                continue
            tokens = tokenizer.encode(text, add_special_tokens=True)
            windows = _windows(tokens, WINDOW_TOKENS, WINDOW_OVERLAP)

            window_vectors = []
            for start in range(0, len(windows), batch_size):
                batch = windows[start : start + batch_size]
                width = max(len(window) for window in batch)
                ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long)
                mask = torch.zeros((len(batch), width), dtype=torch.long)
                for index, window in enumerate(batch):
                    ids[index, : len(window)] = torch.tensor(window, dtype=torch.long)
                    mask[index, : len(window)] = 1
                hidden = model(
                    input_ids=ids.to(device), attention_mask=mask.to(device)
                ).last_hidden_state
                weights = mask.to(device).unsqueeze(-1).float()
                pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
                window_vectors.append(pooled.cpu().numpy())

            stacked = np.concatenate(window_vectors, axis=0)
            if stacked.shape[0] == 1:
                vectors[row] = stacked[0]
            else:
                lengths = np.linalg.norm(stacked, axis=1)
                share = lengths / (lengths.sum() + 1e-8)
                vectors[row] = (share[:, None] * stacked).sum(axis=0)
            found[row] = True

            if (row + 1) % 2000 == 0:
                report(f"    {row + 1:,}/{len(admission_ids):,} admissions")

    return vectors, found


def build_note_vectors(
    cohort_dir: Path,
    mimic_dir: Path,
    source: str,
    records: list,
    train_fraction: float,
    device: str = "cpu",
    report=print,
) -> Path:
    """Write the note files for one cohort and return the embedding path.

    The mean vector is taken over the training admissions only, so no statistic
    of the test admissions reaches the model.
    """
    admission_ids = [
        int(admission[ADMISSION_ID_POSITION]) for patient in records for admission in patient
    ]
    training_admissions = {
        int(admission[ADMISSION_ID_POSITION])
        for patient in records[: int(len(records) * train_fraction)]
        for admission in patient
    }

    raw = read_discharge_summaries(mimic_dir, source, set(admission_ids), report=report)
    cleaned = {admission: remove_medication_sections(text) for admission, text in raw.items()}
    remaining = count_remaining_medication_headings(list(cleaned.values()))
    report(f"  {remaining} summaries still show a medication heading after removal")

    vectors, found = embed_summaries(cleaned, admission_ids, device=device, report=report)

    cohort_dir = Path(cohort_dir)
    path = cohort_dir / NOTE_EMBEDDING_FILE
    with open(path, "wb") as handle:
        pickle.dump(
            {
                "admission_ids": np.asarray(admission_ids, dtype=np.int64),
                "embeddings": vectors,
                "has_note": found,
                "model_name": NOTE_MODEL,
            },
            handle,
        )

    in_training = np.array(
        [admission in training_admissions for admission in admission_ids]
    ) & found
    if not in_training.any():
        raise ValueError("No training admission has a summary, so no mean can be computed.")
    np.save(cohort_dir / NOTE_MEAN_FILE, vectors[in_training].mean(axis=0))
    report(f"  written to {path} and {cohort_dir / NOTE_MEAN_FILE}")
    return path


MEASUREMENT_TABLE = {"mimic3": "LABEVENTS.csv.gz", "mimic4": "hosp/labevents.csv.gz"}
TEST_DICTIONARY = {"mimic3": "D_LABITEMS.csv.gz", "mimic4": "hosp/d_labitems.csv.gz"}
COLUMNS = {
    "mimic3": ("HADM_ID", "ITEMID", "VALUENUM", "CHARTTIME"),
    "mimic4": ("hadm_id", "itemid", "valuenum", "charttime"),
}
DEFAULT_TEST_COUNT = 200
CHUNK_SIZE = 2_000_000
MINIMUM_SPREAD = 1e-6


def read_measurements(
    mimic_dir: Path,
    source: str,
    admission_ids: set[int],
    report=print,
) -> pd.DataFrame:
    """Read the measurements of the cohort's admissions.

    The measurement table is the largest table in either release, so it is read
    in chunks and filtered to the cohort as it goes.

    Returns:
        A frame with one row per measurement: admission, test, value and time.
    """
    admission, test, value, time = COLUMNS[source]
    path = Path(mimic_dir) / MEASUREMENT_TABLE[source]
    if not path.exists():
        raise FileNotFoundError(f"{path} is required for the laboratory channel.")

    kept = []
    for chunk in pd.read_csv(
        path,
        usecols=[admission, test, value, time],
        compression="gzip",
        chunksize=CHUNK_SIZE,
    ):
        chunk = chunk.dropna(subset=[admission, value])
        chunk = chunk[chunk[admission].astype(int).isin(admission_ids)]
        if not chunk.empty:
            kept.append(chunk)
    if not kept:
        raise ValueError("No measurement belongs to this cohort's admissions.")

    frame = pd.concat(kept, ignore_index=True)
    frame.columns = ["admission", "test", "value", "time"]
    frame["admission"] = frame["admission"].astype(int)
    frame["test"] = frame["test"].astype(int)
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    report(f"  {len(frame):,} measurements for {frame['admission'].nunique():,} admissions")
    return frame


def read_test_names(mimic_dir: Path, source: str) -> dict[int, str]:
    """Read the name of each test identifier."""
    path = Path(mimic_dir) / TEST_DICTIONARY[source]
    if not path.exists():
        return {}
    if source == "mimic3":
        table = pd.read_csv(path, usecols=["ITEMID", "LABEL"], compression="gzip")
        return dict(zip(table["ITEMID"].astype(int), table["LABEL"].astype(str), strict=True))
    table = pd.read_csv(path, usecols=["itemid", "label"], compression="gzip")
    return dict(zip(table["itemid"].astype(int), table["label"].astype(str), strict=True))


def most_frequent_tests(frame: pd.DataFrame, count: int = DEFAULT_TEST_COUNT) -> list[int]:
    """Return the identifiers of the tests ordered for the most admissions."""
    per_admission = frame[["admission", "test"]].drop_duplicates()
    counts = Counter(per_admission["test"].tolist())
    return [test for test, _ in counts.most_common(count)]


def bound_measurements(frame: pd.DataFrame, report=print) -> pd.DataFrame:
    """Drop impossible values and clip extreme ones.

    A value outside the error bounds of its test is dropped rather than clipped:
    it is a recording error, and clipping it would invent a measurement.
    """
    bounded = frame.copy()
    bounded["value"] = [
        bound_value(int(test), float(value))
        for test, value in zip(bounded["test"], bounded["value"], strict=True)
    ]
    dropped = bounded["value"].isna().sum()
    known = sum(1 for test in frame["test"].unique() if int(test) in REFERENCE_RANGES)
    report(
        f"  {dropped:,} values outside the published bounds dropped "
        f"({known} of {frame['test'].nunique()} tests have bounds)"
    )
    return bounded.dropna(subset=["value"])


def last_value_per_admission(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the last recorded value of each test in each admission."""
    ordered = frame.sort_values(["admission", "test", "time"])
    return ordered.drop_duplicates(subset=["admission", "test"], keep="last")


def to_matrix(
    frame: pd.DataFrame,
    admission_ids: list[int],
    tests: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Lay the measurements out as a value matrix and a missingness matrix.

    Returns:
        The ``(admissions, tests)`` values, with zero where a test is missing,
        and the ``(admissions, tests)`` flags, one where a test is missing.
    """
    row_of = {admission: row for row, admission in enumerate(admission_ids)}
    column_of = {test: column for column, test in enumerate(tests)}
    values = np.full((len(admission_ids), len(tests)), np.nan, dtype=np.float32)
    for admission, test, value in zip(
        frame["admission"], frame["test"], frame["value"], strict=True
    ):
        row = row_of.get(int(admission))
        column = column_of.get(int(test))
        if row is not None and column is not None:
            values[row, column] = value
    missing = np.isnan(values).astype(np.float32)
    return np.nan_to_num(values, nan=0.0), missing


def standardise(
    values: np.ndarray,
    missing: np.ndarray,
    training_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardise each test with the training admissions' own mean and spread.

    Args:
        values: ``(admissions, tests)`` measured values.
        missing: ``(admissions, tests)`` flags, one where a test is missing.
        training_rows: boolean mask of the training admissions.

    Returns:
        The standardised values with zero where a test is missing, plus the mean
        and the spread of each test, which are kept for reference.
    """
    test_count = values.shape[1]
    means = np.zeros(test_count, dtype=np.float32)
    spreads = np.ones(test_count, dtype=np.float32)
    for column in range(test_count):
        present = training_rows & (missing[:, column] == 0)
        if present.sum() > 1:
            column_values = values[present, column]
            means[column] = column_values.mean()
            spreads[column] = max(float(column_values.std()), MINIMUM_SPREAD)

    standardised = (values - means) / spreads
    standardised[missing == 1] = 0.0
    return standardised.astype(np.float32), means, spreads


def build_lab_vectors(
    cohort_dir: Path,
    mimic_dir: Path,
    source: str,
    records: list,
    train_fraction: float,
    test_count: int = DEFAULT_TEST_COUNT,
    report=print,
) -> Path:
    """Write the laboratory file for one cohort and return its path."""
    admission_ids = [
        int(admission[ADMISSION_ID_POSITION]) for patient in records for admission in patient
    ]
    training_admissions = {
        int(admission[ADMISSION_ID_POSITION])
        for patient in records[: int(len(records) * train_fraction)]
        for admission in patient
    }

    frame = read_measurements(mimic_dir, source, set(admission_ids), report=report)
    tests = most_frequent_tests(frame, test_count)
    report(f"  keeping the {len(tests)} most frequently ordered tests")
    frame = frame[frame["test"].isin(tests)]
    frame = bound_measurements(frame, report=report)
    frame = last_value_per_admission(frame)

    values, missing = to_matrix(frame, admission_ids, tests)
    training_rows = np.array([admission in training_admissions for admission in admission_ids])
    standardised, means, spreads = standardise(values, missing, training_rows)

    measured = (missing == 0).any(axis=1)
    report(f"  {int(measured.sum()):,} of {len(admission_ids):,} admissions have measurements")

    names = read_test_names(mimic_dir, source)
    path = Path(cohort_dir) / LAB_FEATURE_FILE
    with open(path, "wb") as handle:
        pickle.dump(
            {
                "admission_ids": np.asarray(admission_ids, dtype=np.int64),
                "vectors": np.concatenate([standardised, missing], axis=1),
                "has_labs": measured,
                "test_ids": np.asarray(tests, dtype=np.int64),
                "test_names": [names.get(test, str(test)) for test in tests],
                "training_mean": means,
                "training_spread": spreads,
            },
            handle,
        )
    report(f"  written to {path}")
    return path

CHANNELS = ("codes", "notes", "labs")


def _read_class_names(path: Path | None) -> dict[str, str]:
    """Read drug class names, which give the language model more than a code to read."""
    if path is None:
        return {}
    table = pd.read_csv(path, dtype=str).dropna()
    code_column, name_column = table.columns[0], table.columns[1]
    return dict(zip(table[code_column].str.strip(), table[name_column].str.strip(), strict=True))


def main(argv: list[str] | None = None) -> int:
    """Build the code embeddings, note vectors and laboratory vectors of one cohort."""
    parser = argparse.ArgumentParser(description="Build the three input channels of one cohort.")
    parser.add_argument("--cohort", required=True, help="a cohort listed in config.yaml")
    parser.add_argument("--mimic_dir", type=Path, required=True)
    parser.add_argument("--drug_reference_dir", type=Path, default=None,
                        help="needed for the code embeddings")
    parser.add_argument("--data_dir", type=Path, default=None,
                        help="folder holding the cohort folder (default: data/processed)")
    parser.add_argument("--device", default="cpu", help="cuda or cpu for the language models")
    parser.add_argument("--only", choices=CHANNELS, default=None,
                        help="build one channel instead of all three")
    parser.add_argument("--lab_count", type=int, default=200,
                        help="how many of the most frequent tests to keep")
    parser.add_argument("--class_names", type=Path, default=None,
                        help="optional two-column file of drug class code and name")
    arguments = parser.parse_args(argv)

    cohort = load_cohort(arguments.cohort)
    cohort_dir = make_paths(arguments.data_dir).cohort_dir(cohort.directory)
    if not (cohort_dir / RECORDS_FILE).exists():
        print(f"{cohort_dir / RECORDS_FILE} is missing. Run src/preprocess.py first.")
        return 1
    with open(cohort_dir / RECORDS_FILE, "rb") as handle:
        records = pickle.load(handle)
    with open(cohort_dir / VOCABULARY_FILE, "rb") as handle:
        vocabulary = pickle.load(handle)

    wanted = [arguments.only] if arguments.only else list(CHANNELS)
    if "codes" in wanted:
        if arguments.drug_reference_dir is None:
            print("--drug_reference_dir is needed for the code embeddings.")
            return 1
        print("=== code and drug embeddings ===")
        build_code_embeddings(
            cohort_dir=cohort_dir,
            mimic_dir=arguments.mimic_dir,
            source=cohort.source,
            diagnosis_codes=vocabulary["diagnosis_codes"],
            procedure_codes=vocabulary["procedure_codes"],
            drug_codes=vocabulary["drug_codes"],
            reference=load_reference(arguments.drug_reference_dir),
            class_names=_read_class_names(arguments.class_names),
            device=arguments.device,
        )
    if "notes" in wanted:
        print("=== discharge summaries ===")
        build_note_vectors(
            cohort_dir=cohort_dir,
            mimic_dir=arguments.mimic_dir,
            source=cohort.source,
            records=records,
            train_fraction=TRAIN_FRACTION,
            device=arguments.device,
        )
    if "labs" in wanted:
        print("=== laboratory measurements ===")
        build_lab_vectors(
            cohort_dir=cohort_dir,
            mimic_dir=arguments.mimic_dir,
            source=cohort.source,
            records=records,
            train_fraction=TRAIN_FRACTION,
            test_count=arguments.lab_count,
        )
    print(f"Next: python src/train.py --cohort {cohort.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
