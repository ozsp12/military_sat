#!/usr/bin/env python3
"""Export the ITA phase 1 Parquet datasets to a formatted Excel workbook."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_QUESTIONS = Path("data/ita_phase_1.parquet")
DEFAULT_IMAGES = Path("data/ita_phase_1_images.parquet")
DEFAULT_OUTPUT = Path("data/ita_phase_1.xlsx")

COLUMN_WIDTHS = {
    "alternative_id": 27,
    "question_id": 23,
    "year": 8,
    "question_number": 16,
    "subject": 14,
    "question_text": 55,
    "alternative": 12,
    "alternative_text": 45,
    "has_image": 12,
    "ocr_used": 12,
    "needs_review": 14,
    "source_pdf": 36,
    "page_start": 12,
    "page_end": 12,
    "image_id": 31,
    "image_path": 58,
    "page": 10,
    "width_px": 12,
    "height_px": 12,
}

WRAPPED_COLUMNS = {
    "question_text",
    "alternative_text",
    "source_pdf",
    "image_path",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def format_sheet(writer: pd.ExcelWriter, sheet_name: str, frame: pd.DataFrame) -> None:
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]

    header_format = workbook.add_format(
        {
            "bold": True,
            "bg_color": "#D9EAF7",
            "border": 1,
            "text_wrap": True,
            "valign": "top",
        }
    )
    wrap_format = workbook.add_format(
        {
            "text_wrap": True,
            "valign": "top",
        }
    )

    worksheet.freeze_panes(1, 0)
    worksheet.autofilter(0, 0, len(frame), len(frame.columns) - 1)
    worksheet.set_row(0, 24, header_format)

    for column_index, column_name in enumerate(frame.columns):
        width = COLUMN_WIDTHS.get(
            column_name,
            max(12, min(24, len(column_name) + 2)),
        )
        cell_format = wrap_format if column_name in WRAPPED_COLUMNS else None
        worksheet.set_column(column_index, column_index, width, cell_format)


def export_workbook(
    questions_path: Path,
    images_path: Path,
    output_path: Path,
) -> None:
    questions = pd.read_parquet(questions_path)
    images = pd.read_parquet(images_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        questions.to_excel(writer, sheet_name="questions", index=False)
        images.to_excel(writer, sheet_name="images", index=False)
        format_sheet(writer, "questions", questions)
        format_sheet(writer, "images", images)

    print(
        f"[done ] {output_path}: {len(questions)} question-alternative rows, "
        f"{len(images)} image rows"
    )


def main() -> int:
    args = parse_args()
    for path in (args.questions, args.images):
        if not path.is_file():
            raise SystemExit(f"Required dataset not found: {path}")

    export_workbook(args.questions, args.images, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
