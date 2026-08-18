# military_sat

Archive and utilities for entrance-exam material used in the project.

## ITA exam archive

Official ITA entrance-exam PDFs are mirrored from:

https://www.vestibular.ita.br/provas.htm

The archive contains the exam papers only; answer keys are excluded by default.

```text
ita_provas/
├── manifest.csv
├── 2026/
│   ├── 2026_fase1.pdf
│   ├── matematica_2026_2f.pdf
│   ├── fisica_2026_2f.pdf
│   └── ...
├── 2025/
├── ...
└── 2008/
```

`manifest.csv` records the year, label, repository path, file size, SHA-256 digest, and original source URL for each PDF.

### Download

```bash
python -m pip install -r requirements.txt
python scripts/download_ita_provas.py
```

To force verification against the current upstream files:

```bash
python scripts/download_ita_provas.py --refresh --min-files 95
```

To include answer keys as well:

```bash
python scripts/download_ita_provas.py --include-gabaritos
```

## Automation

`.github/workflows/update-ita-provas.yml` updates the archive automatically on the first day of each month and can also be executed manually. Changes to the downloader or workflow trigger an update after they reach `main`.
