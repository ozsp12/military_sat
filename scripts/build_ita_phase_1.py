#!/usr/bin/env python3
"""Build a long-format dataset from ITA first-phase entrance exams.

Each dataset row represents one answer alternative. Questions with visual
content are rendered as PNG crops and related through ``question_id`` in a
second Parquet table.
"""

from __future__ import annotations

import argparse
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pymupdf as fitz
import pandas as pd

DEFAULT_INPUT = Path("ita_provas")
DEFAULT_DATASET = Path("data/ita_phase_1.parquet")
DEFAULT_IMAGE_DATASET = Path("data/ita_phase_1_images.parquet")
DEFAULT_IMAGE_ROOT = Path("data/ita_phase_1_images")
OCR_LANGUAGE = "por+eng"
OCR_DPI = 300
OCR_MIN_CHARS = 80

QUESTION_RE = re.compile(
    r"\bQuest.{0,12}?\s([0-9SOIl]{1,3})\s*\.",
    re.IGNORECASE,
)
QUESTION_PREFIX_RE = re.compile(r"\bQuest", re.IGNORECASE)
ALTERNATIVE_RE = re.compile(r"(?<![A-Za-zÀ-ÿ])([A-E])\s*\(\s*\)", re.IGNORECASE)
YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
QUESTION_NUMBER_TRANSLATION = str.maketrans(
    {"S": "5", "s": "5", "O": "0", "o": "0", "I": "1", "l": "1"}
)

SUBJECT_ALIASES = {
    "FISICA": "fisica",
    "MATEMATICA": "matematica",
    "QUIMICA": "quimica",
    "INGLES": "ingles",
    "PORTUGUES": "portugues",
    "LINGUAPORTUGUESA": "portugues",
}

QUESTION_COLUMNS = [
    "alternative_id",
    "question_id",
    "year",
    "question_number",
    "subject",
    "question_text",
    "alternative",
    "alternative_text",
    "has_image",
    "ocr_used",
    "source_pdf",
    "page_start",
    "page_end",
]

IMAGE_COLUMNS = [
    "image_id",
    "question_id",
    "image_path",
    "page",
    "width_px",
    "height_px",
]


@dataclass(frozen=True)
class TextLine:
    page_index: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


@dataclass(frozen=True)
class QuestionStart:
    year: int
    number: int
    subject: str
    page_index: int
    y0: float


@dataclass(frozen=True)
class PageSegment:
    page_index: int
    y0: float
    y1: float


@dataclass(frozen=True)
class Boundary:
    page_index: int
    y0: float


def normalize_label(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^A-Za-z ]+", " ", value).upper()
    return re.sub(r"\s+", " ", value).strip()


def canonical_subject(line: str) -> str | None:
    normalized = normalize_label(line).replace(" ", "")
    return SUBJECT_ALIASES.get(normalized)


def parse_question_number(token: str) -> int:
    normalized = token.translate(QUESTION_NUMBER_TRANSLATION)
    if not normalized.isdigit():
        raise ValueError(f"Invalid question-number token: {token!r}")
    return int(normalized)


def payload_to_lines(page: fitz.Page, payload: dict[str, object]) -> list[TextLine]:
    result: list[TextLine] = []
    for block in payload.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(span.get("text", "") for span in spans).strip()
            if not text:
                continue
            x0, y0, x1, y1 = line["bbox"]
            result.append(TextLine(page.number, x0, y0, x1, y1, text))
    return sorted(result, key=lambda item: (item.y0, item.x0))


def alphanumeric_count(lines: Iterable[TextLine]) -> int:
    return sum(sum(character.isalnum() for character in line.text) for line in lines)


def extract_lines(page: fitz.Page) -> tuple[list[TextLine], bool]:
    native_payload = page.get_text("dict", sort=True)
    native_lines = payload_to_lines(page, native_payload)
    if alphanumeric_count(native_lines) >= OCR_MIN_CHARS:
        return native_lines, False

    try:
        textpage = page.get_textpage_ocr(
            language=OCR_LANGUAGE,
            dpi=OCR_DPI,
            full=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Page {page.number + 1} has insufficient embedded text and requires "
            "OCR. Install Tesseract language data for Portuguese and English."
        ) from exc

    ocr_payload = page.get_text("dict", textpage=textpage, sort=True)
    ocr_lines = payload_to_lines(page, ocr_payload)
    if alphanumeric_count(ocr_lines) <= alphanumeric_count(native_lines):
        return native_lines, False
    return ocr_lines, True


def match_question_header(lines: list[TextLine], index: int) -> re.Match[str] | None:
    """Match a question header, allowing OCR to split `Questão` from its number."""
    line = lines[index]
    match = QUESTION_RE.search(line.text)
    if match or not QUESTION_PREFIX_RE.search(line.text) or index + 1 >= len(lines):
        return match

    next_line = lines[index + 1]
    # Only join nearby lines on the same page. This avoids accidentally attaching
    # a distant numbered paragraph to a heading that merely contains "Quest".
    if next_line.page_index != line.page_index or next_line.y0 - line.y1 > 25:
        return None
    return QUESTION_RE.search(f"{line.text} {next_line.text}")


def discover_questions(
    doc: fitz.Document, year: int
) -> tuple[list[QuestionStart], list[Boundary], list[list[TextLine]], list[bool]]:
    all_lines: list[list[TextLine]] = []
    page_ocr: list[bool] = []
    starts: list[QuestionStart] = []
    subject_boundaries: list[Boundary] = []
    current_subject: str | None = None

    for page in doc:
        lines, used_ocr = extract_lines(page)
        all_lines.append(lines)
        page_ocr.append(used_ocr)
        for index, line in enumerate(lines):
            subject = canonical_subject(line.text)
            if subject:
                current_subject = subject
                subject_boundaries.append(Boundary(page.number, line.y0))
                continue

            match = match_question_header(lines, index)
            if not match:
                continue
            number = parse_question_number(match.group(1))
            if current_subject is None:
                raise ValueError(
                    f"Could not infer subject before question {number} "
                    f"on page {page.number + 1}."
                )
            starts.append(
                QuestionStart(
                    year=year,
                    number=number,
                    subject=current_subject,
                    page_index=page.number,
                    y0=line.y0,
                )
            )

    starts.sort(key=lambda item: (item.page_index, item.y0, item.number))
    subject_boundaries.sort(key=lambda item: (item.page_index, item.y0))
    return starts, subject_boundaries, all_lines, page_ocr


def question_segments(
    doc: fitz.Document,
    start: QuestionStart,
    end: Boundary | None,
) -> list[PageSegment]:
    last_page = end.page_index if end else len(doc) - 1
    segments: list[PageSegment] = []

    for page_index in range(start.page_index, last_page + 1):
        page = doc[page_index]
        top = start.y0 if page_index == start.page_index else 0.0
        if end and page_index == end.page_index:
            bottom = end.y0
        else:
            bottom = page.rect.height
        if bottom > top + 1:
            segments.append(PageSegment(page_index, top, bottom))
    return segments


def segment_text(
    all_lines: list[list[TextLine]],
    segments: Iterable[PageSegment],
) -> str:
    chunks: list[str] = []
    for segment in segments:
        page_lines = all_lines[segment.page_index]
        lines = [
            line.text
            for line in page_lines
            if line.y0 >= segment.y0 - 1 and line.y0 < segment.y1 - 1
            and not (
                line.y0 > 0.94 * max(item.y1 for item in page_lines)
                and line.text.isdigit()
            )
        ]
        chunks.extend(lines)
    return re.sub(r"\s+", " ", " ".join(chunks)).strip()


def split_question_text(raw: str, expected_number: int) -> tuple[str, dict[str, str]]:
    header = QUESTION_RE.search(raw)
    if not header or parse_question_number(header.group(1)) != expected_number:
        raise ValueError(f"Question marker {expected_number} not found in extracted text.")

    body = raw[header.end():].strip()
    matches = list(ALTERNATIVE_RE.finditer(body))
    by_label: dict[str, tuple[int, int]] = {}
    for match in matches:
        label = match.group(1).upper()
        by_label.setdefault(label, (match.start(), match.end()))

    expected = {"A", "B", "C", "D", "E"}
    if set(by_label) != expected:
        raise ValueError(
            f"Question {expected_number}: expected alternatives A-E; "
            f"found {sorted(by_label)}."
        )

    ordered = sorted(
        ((label, start, end) for label, (start, end) in by_label.items()),
        key=lambda item: item[1],
    )
    question_text = body[: ordered[0][1]].strip()
    if not question_text:
        raise ValueError(f"Question {expected_number}: empty statement.")

    alternatives: dict[str, str] = {}
    for index, (label, _start, end) in enumerate(ordered):
        next_start = ordered[index + 1][1] if index + 1 < len(ordered) else len(body)
        alternatives[label] = body[end:next_start].strip()

    if any(not value for value in alternatives.values()):
        empty = sorted(label for label, value in alternatives.items() if not value)
        raise ValueError(f"Question {expected_number}: empty alternatives {empty}.")

    return question_text, alternatives


def rect_overlap(a: fitz.Rect, b: fitz.Rect) -> float:
    intersection = a & b
    return max(0.0, intersection.width) * max(0.0, intersection.height)


def segment_has_graphics(page: fitz.Page, clip: fitz.Rect) -> bool:
    for image in page.get_images(full=True):
        xref = image[0]
        for rect in page.get_image_rects(xref):
            if rect_overlap(rect, clip) >= 100.0:
                return True

    drawing_hits = 0
    for drawing in page.get_drawings():
        rect = fitz.Rect(drawing["rect"])
        if rect_overlap(rect, clip) <= 0:
            continue
        if rect.width >= 20 and rect.height >= 20:
            return True
        if rect.width >= 5 or rect.height >= 5:
            drawing_hits += 1
    return drawing_hits >= 4


def render_question_images(
    doc: fitz.Document,
    question_id: str,
    segments: list[PageSegment],
    image_root: Path,
    scale: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    counter = 0

    for segment in segments:
        page = doc[segment.page_index]
        clip = fitz.Rect(0, segment.y0, page.rect.width, segment.y1)
        if not segment_has_graphics(page, clip):
            continue

        counter += 1
        image_id = f"{question_id}-IMG{counter:02d}"
        year = question_id.split("-")[2]
        output_dir = image_root / year
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{image_id}.png"

        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
        pix.save(output_path)
        rows.append(
            {
                "image_id": image_id,
                "question_id": question_id,
                "image_path": output_path.as_posix(),
                "page": segment.page_index + 1,
                "width_px": pix.width,
                "height_px": pix.height,
            }
        )
    return rows


def infer_year(pdf: Path) -> int:
    match = YEAR_RE.search(pdf.stem)
    if not match:
        raise ValueError(f"Cannot infer year from {pdf}.")
    return int(match.group(1))


def build_exam(
    pdf: Path,
    image_root: Path,
    scale: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    year = infer_year(pdf)
    question_rows: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []

    with fitz.open(pdf) as doc:
        starts, subject_boundaries, all_lines, page_ocr = discover_questions(doc, year)
        if not starts:
            raise ValueError(f"No questions detected in {pdf}.")

        numbers = [item.number for item in starts]
        if len(numbers) != len(set(numbers)):
            raise ValueError(f"Duplicate question numbers in {pdf}: {numbers}")
        if sorted(numbers) != list(range(1, max(numbers) + 1)):
            raise ValueError(f"Non-contiguous question numbering in {pdf}: {numbers}")

        source_pdf = pdf.as_posix()
        for index, start in enumerate(starts):
            next_start = starts[index + 1] if index + 1 < len(starts) else None
            candidates: list[Boundary] = []
            if next_start:
                candidates.append(Boundary(next_start.page_index, next_start.y0))
            candidates.extend(
                boundary
                for boundary in subject_boundaries
                if (boundary.page_index, boundary.y0) > (start.page_index, start.y0)
            )
            end = (
                min(candidates, key=lambda item: (item.page_index, item.y0))
                if candidates
                else None
            )
            segments = question_segments(doc, start, end)
            raw = segment_text(all_lines, segments)
            statement, alternatives = split_question_text(raw, start.number)

            question_id = f"ITA-1F-{year}-Q{start.number:03d}"
            images = render_question_images(doc, question_id, segments, image_root, scale)
            image_rows.extend(images)
            has_image = bool(images)
            ocr_used = any(page_ocr[item.page_index] for item in segments)

            page_start = min(item.page_index for item in segments) + 1
            page_end = max(item.page_index for item in segments) + 1
            for label in "ABCDE":
                question_rows.append(
                    {
                        "alternative_id": f"{question_id}-{label}",
                        "question_id": question_id,
                        "year": year,
                        "question_number": start.number,
                        "subject": start.subject,
                        "question_text": statement,
                        "alternative": label,
                        "alternative_text": alternatives[label],
                        "has_image": has_image,
                        "ocr_used": ocr_used,
                        "source_pdf": source_pdf,
                        "page_start": page_start,
                        "page_end": page_end,
                    }
                )

    print(f"        OCR pages: {sum(page_ocr)}")
    return question_rows, image_rows


def validate_dataset(questions: pd.DataFrame, images: pd.DataFrame) -> None:
    if questions.empty:
        raise ValueError("Question dataset is empty.")
    if questions["alternative_id"].duplicated().any():
        duplicates = questions.loc[
            questions["alternative_id"].duplicated(), "alternative_id"
        ].tolist()
        raise ValueError(f"Duplicate alternative_id values: {duplicates[:10]}")

    counts = questions.groupby("question_id")["alternative"].nunique()
    bad = counts[counts != 5]
    if not bad.empty:
        raise ValueError(f"Questions without exactly five alternatives: {bad.to_dict()}")

    if questions[["subject", "question_text", "alternative_text"]].isna().any().any():
        raise ValueError("Null required values detected in question dataset.")

    if not images.empty:
        if images["image_id"].duplicated().any():
            raise ValueError("Duplicate image_id values detected.")
        known_questions = set(questions["question_id"])
        orphaned = sorted(set(images["question_id"]) - known_questions)
        if orphaned:
            raise ValueError(f"Orphaned image rows: {orphaned[:10]}")
        missing_paths = [path for path in images["image_path"] if not Path(path).is_file()]
        if missing_paths:
            raise ValueError(f"Missing rendered images: {missing_paths[:10]}")

    image_questions = set(images["question_id"]) if not images.empty else set()
    flag_questions = set(questions.loc[questions["has_image"], "question_id"])
    if image_questions != flag_questions:
        raise ValueError("has_image flags are inconsistent with the image relationship table.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--image-dataset", type=Path, default=DEFAULT_IMAGE_DATASET)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="Rendering scale for question crops containing graphics (default: 2.0).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdfs = sorted(args.input.glob("*/[0-9][0-9][0-9][0-9]_fase1.pdf"))
    if not pdfs:
        raise SystemExit(f"No first-phase PDFs found under {args.input}.")
    if args.scale <= 0:
        raise SystemExit("--scale must be positive.")

    if args.image_root.exists():
        shutil.rmtree(args.image_root)
    args.image_root.mkdir(parents=True, exist_ok=True)

    question_rows: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []
    for pdf in pdfs:
        print(f"[parse] {pdf}")
        exam_questions, exam_images = build_exam(pdf, args.image_root, args.scale)
        question_rows.extend(exam_questions)
        image_rows.extend(exam_images)
        print(
            f"        {len(exam_questions) // 5} questions, "
            f"{len(exam_images)} image crop(s)"
        )

    questions = pd.DataFrame(question_rows, columns=QUESTION_COLUMNS)
    images = pd.DataFrame(image_rows, columns=IMAGE_COLUMNS)
    validate_dataset(questions, images)

    args.dataset.parent.mkdir(parents=True, exist_ok=True)
    questions.sort_values(["year", "question_number", "alternative"]).to_parquet(
        args.dataset, index=False, engine="pyarrow"
    )
    images.sort_values(["question_id", "image_id"]).to_parquet(
        args.image_dataset, index=False, engine="pyarrow"
    )

    print(
        f"[done ] {questions['question_id'].nunique()} questions, "
        f"{len(questions)} alternatives, {len(images)} images"
    )
    print(f"        {args.dataset}")
    print(f"        {args.image_dataset}")
    print(f"        {args.image_root}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
