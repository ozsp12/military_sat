# military_sat

Archive and utilities for entrance-exam material used in the project.

## ITA exam archive

Official ITA entrance-exam PDFs are mirrored from:

https://www.vestibular.ita.br/provas.htm

The archive contains the exam papers and the official first-phase answer keys. Second-phase answer keys are excluded.

```text
ita_provas/
├── manifest.xlsx
├── 2026/
│   ├── 2026_fase1.pdf
│   ├── gabarito_2026.pdf
│   ├── matematica_2026_2f.pdf
│   ├── fisica_2026_2f.pdf
│   └── ...
├── 2025/
├── ...
└── 2008/
```

`manifest.xlsx` contains one row per PDF with year, label, repository path, file size, SHA-256 digest, and original source URL.

### Download

```bash
python -m pip install -r requirements.txt
python scripts/download_ita_provas.py
```

To force verification against the current upstream files:

```bash
python scripts/download_ita_provas.py --refresh --min-files 95
```

## ITA datasets

`scripts/build_ita_datasets.py` processes the archived PDFs and separates numbered questions by type rather than by exam phase.

```text
ita_data/
├── ita_multiple_choice.parquet
├── ita_multiple_choice.xlsx
├── ita_multiple_choice_images.parquet
├── ita_multiple_choice_images/
├── ita_essay_questions.parquet
├── ita_essay_questions.xlsx
├── ita_essay_questions_images.parquet
└── ita_essay_questions_images/
```

`ita_multiple_choice.parquet` is long format: one row per answer alternative. `ita_essay_questions.parquet` contains one row per open-ended question. Both preserve `exam_phase`, subject, year, source PDF, page provenance, OCR/review flags, and one-to-many image relationships through separate image Parquets.

For the 2008–2018 single-phase exams, Physics, Mathematics and Chemistry questions 1–20 are classified as multiple choice and 21–30 as essay questions; English and Portuguese numbered questions are multiple choice. From 2019 onward, first-phase questions are multiple choice and second-phase Physics, Mathematics and Chemistry questions are essay questions. Writing prompts are excluded.

Build locally with:

```bash
python scripts/build_ita_datasets.py
```

## Automation

`.github/workflows/update-ita-provas.yml` updates the archive automatically on the first day of each month and can also be executed manually.

`.github/workflows/build-ita-datasets.yml` validates the unified builder in pull requests and regenerates the versioned multiple-choice and essay datasets after relevant changes reach `main`.
