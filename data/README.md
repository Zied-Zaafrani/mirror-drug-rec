# Data

MIMIC-III and MIMIC-IV are released by PhysioNet under a credentialed data use
agreement, which does not allow redistribution. This folder therefore holds no
data. The two preparation scripts write each cohort to `data/processed/<cohort>/`.

## What to download

| Dataset | Needed for | Files the scripts read |
|---|---|---|
| [MIMIC-III v1.4](https://physionet.org/content/mimiciii/1.4/) | `mimic3` | `ADMISSIONS`, `PATIENTS`, `DIAGNOSES_ICD`, `PROCEDURES_ICD`, `PRESCRIPTIONS`, `LABEVENTS`, `NOTEEVENTS`, `D_ICD_*` (`.csv.gz`) |
| [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/) | `mimic4_*` | `hosp/` tables, and `icu/icustays.csv.gz` for the intensive-care cohort |
| [MIMIC-IV-Note v2.2](https://physionet.org/content/mimic-iv-note/2.2/) | `mimic4_*` | `discharge.csv.gz`, placed in the MIMIC-IV folder or in its `note/` sub-folder |

Four drug reference files map prescriptions to drug classes and give their
structures and interactions. Three are the public mapping files of the
[SafeDrug repository](https://github.com/ycq091044/SafeDrug), used unchanged
under the names below; the fourth is included here. Put all four in one folder
(this `data/` folder works):

| File | Holds | Source |
|---|---|---|
| `ndc_to_rxnorm.txt` | NDC product code to RxNorm concept | SafeDrug `data/input/ndc2RXCUI.txt` |
| `rxnorm_to_atc.csv` | RxNorm concept to ATC class | SafeDrug `data/input/RXCUI2atc4.csv` |
| `atc3_structures.pkl` | ATC-3 class to the SMILES string of its drugs | SafeDrug `data/output/atc3toSMILES.pkl` |
| `atc3_interactions.csv` | interacting ATC-3 pairs from TWOSIDES | included in this folder |

## Preparing a cohort

```bash
python src/preprocess.py --cohort mimic3 --mimic_dir /data/mimic-iii-1.4 --drug_reference_dir data
python src/features.py   --cohort mimic3 --mimic_dir /data/mimic-iii-1.4 --drug_reference_dir data --device cuda
```

The cohorts are `mimic3`, `mimic4_icu`, `mimic4_hospital` and `mimic4_mixed`,
defined in [src/config.yaml](../src/config.yaml).

`preprocess.py` maps prescriptions to ATC-3 classes and keeps the classes with a
known structure. It keeps the 2,000 most frequent diagnosis codes and the
admissions that have diagnoses, procedures and prescriptions, then orders each
patient's admissions. MIMIC-III admissions are ordered by admission time; the
MIMIC-IV cohorts follow the published MIMIC-IV preprocessing and order by
admission identifier. It writes:

| File | Holds |
|---|---|
| `records.pkl` | per patient, a list of admissions, each as diagnosis codes, procedure codes, drug classes and admission identifier |
| `vocabulary.pkl` | the code behind every index |
| `interaction_matrix.pkl` | 1 where two drug classes are a known interacting pair |
| `coprescription_matrix.pkl` | how often two classes share an admission, counted on the training patients |
| `patient_context.csv` | age, sex, admission type and length of stay, used only by `showcase.py` |

`features.py` builds the three input channels. It is the slow step, so run it on
a graphics card:

| File | Holds |
|---|---|
| `code_embeddings.pt` | PubMedBERT embeddings of every diagnosis, procedure and drug description, and a Morgan fingerprint per drug class |
| `code_labels.csv` | the name of every code, used only by `showcase.py` |
| `note_embeddings.pkl` | a Bio_ClinicalBERT vector of each discharge summary with its medication sections removed, and a flag where none exists |
| `note_mean.npy` | the mean note vector of the training admissions |
| `lab_features.pkl` | per admission, the z-scored last value of each of the 200 most frequent tests and a flag per test marking it missing |

The note mean and the laboratory standardisation use training admissions only.
