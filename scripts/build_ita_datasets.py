#!/usr/bin/env python3
"""Build ITA multiple-choice and essay-question datasets from archived PDFs.

Outputs are organized by question type rather than exam phase:

* ``ita_multiple_choice``: one row per answer alternative;
* ``ita_essay_questions``: one row per open-ended question.

Both datasets have a separate one-to-many image table keyed by ``question_id``
and an XLSX inspection workbook. Redacao/writing-prompt PDFs are deliberately
excluded from these two datasets.
"""

from __future__ import annotations

import argparse
import re
import shutil
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd
import pymupdf as fitz

DEFAULT_INPUT = Path("ita_provas")
DATA_ROOT = Path("data")

MC_DATASET = DATA_ROOT / "ita_multiple_choice.parquet"
MC_IMAGE_DATASET = DATA_ROOT / "ita_multiple_choice_images.parquet"
MC_EXCEL = DATA_ROOT / "ita_multiple_choice.xlsx"
MC_IMAGE_ROOT = DATA_ROOT / "ita_multiple_choice_images"

ESSAY_DATASET = DATA_ROOT / "ita_essay_questions.parquet"
ESSAY_IMAGE_DATASET = DATA_ROOT / "ita_essay_questions_images.parquet"
ESSAY_EXCEL = DATA_ROOT / "ita_essay_questions.xlsx"
ESSAY_IMAGE_ROOT = DATA_ROOT / "ita_essay_questions_images"

OCR_LANGUAGE = "por+eng"
OCR_DPI = 300
OCR_MIN_CHARS = 80

QUESTION_RE = re.compile(
    r"\bQuest.{0,12}?\s*([0-9SOIl]{1,3})\s*[\.:]",
    re.IGNORECASE,
)
QUESTION_PREFIX_RE = re.compile(r"\bQuest", re.IGNORECASE)
ALTERNATIVE_RE = re.compile(r"(?<![A-Za-zÀ-ÿ])([A-E])\s*[\(\[]\s*[\)\]]", re.IGNORECASE)
ALTERNATIVE_PAREN_RE = re.compile(r"(?<![A-Za-zÀ-ÿ])([A-E])\s*\(\s*\)", re.IGNORECASE)
STANDALONE_ALTERNATIVE_RE = re.compile(r"(?<![A-Za-zÀ-ÿ0-9])([A-E])(?=[\s\)\.\-:])")
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
SUBJECT_CODES = {
    "fisica": "FIS",
    "matematica": "MAT",
    "quimica": "QUI",
    "ingles": "ING",
    "portugues": "POR",
}
PHASE_CODES = {"single_phase": "SP", "phase_1": "P1", "phase_2": "P2"}
SCIENCE_SUBJECTS = {"fisica", "matematica", "quimica"}
LANGUAGE_SUBJECTS = {"ingles", "portugues"}
LEGACY_FIRST_YEAR = 2008
LEGACY_LAST_YEAR = 2018

MC_COLUMNS = [
    "alternative_id",
    "question_id",
    "year",
    "exam_phase",
    "question_number",
    "subject",
    "question_text",
    "alternative",
    "alternative_text",
    "has_image",
    "ocr_used",
    "needs_review",
    "source_pdf",
    "page_start",
    "page_end",
]
ESSAY_COLUMNS = [
    "question_id",
    "year",
    "exam_phase",
    "question_number",
    "subject",
    "question_text",
    "has_image",
    "ocr_used",
    "needs_review",
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

QuestionKind = Literal["multiple_choice", "essay"]


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
class AlternativeMarker:
    label: str
    start: int
    end: int


@dataclass(frozen=True)
class Boundary:
    page_index: int
    y0: float


def normalize_label(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^A-Za-z0-9 ]+", " ", value).upper()
    return re.sub(r"\s+", " ", value).strip()


def canonical_subject(value: str) -> str | None:
    normalized = normalize_label(value).replace(" ", "")
    return SUBJECT_ALIASES.get(normalized)


def infer_subject_from_filename(pdf: Path) -> str | None:
    normalized = normalize_label(pdf.stem).replace(" ", "")
    for token, subject in (
        ("FISICA", "fisica"),
        ("MATEMATICA", "matematica"),
        ("QUIMICA", "quimica"),
        ("INGLES", "ingles"),
        ("PORTUGUES", "portugues"),
    ):
        if token in normalized:
            return subject
    return None


def infer_year(pdf: Path) -> int:
    for candidate in (pdf.parent.name, pdf.stem):
        match = YEAR_RE.search(candidate)
        if match:
            return int(match.group(1))
    raise ValueError(f"Cannot infer year from {pdf}.")


def infer_phase(pdf: Path) -> str:
    normalized = normalize_label(pdf.stem).replace(" ", "")
    raw = pdf.stem.lower()
    if "FASE1" in normalized or "fase1" in raw:
        return "phase_1"
    if "FASE2" in normalized or re.search(r"(?:^|_)2f(?:_|$)", raw):
        return "phase_2"
    return "single_phase"


def should_skip_pdf(pdf: Path) -> bool:
    normalized = normalize_label(pdf.stem).replace(" ", "")
    return any(token in normalized for token in ("GABARITO", "REDACAO"))


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
            f"Page {page.number + 1} has insufficient embedded text and requires OCR. "
            "Install Tesseract language data for Portuguese and English."
        ) from exc

    ocr_payload = page.get_text("dict", textpage=textpage, sort=True)
    ocr_lines = payload_to_lines(page, ocr_payload)
    if alphanumeric_count(ocr_lines) <= alphanumeric_count(native_lines):
        return native_lines, False
    return ocr_lines, True


def match_question_header(lines: list[TextLine], index: int) -> re.Match[str] | None:
    line = lines[index]
    match = QUESTION_RE.search(line.text)
    if match or not QUESTION_PREFIX_RE.search(line.text) or index + 1 >= len(lines):
        return match
    next_line = lines[index + 1]
    if next_line.page_index != line.page_index or next_line.y0 - line.y1 > 28:
        return None
    return QUESTION_RE.search(f"{line.text} {next_line.text}")


def discover_questions(
    doc: fitz.Document,
    year: int,
    fixed_subject: str | None,
) -> tuple[list[QuestionStart], list[Boundary], list[list[TextLine]], list[bool]]:
    all_lines: list[list[TextLine]] = []
    page_ocr: list[bool] = []
    candidates: list[QuestionStart] = []
    stop_boundaries: list[Boundary] = []
    current_subject = fixed_subject

    for page in doc:
        lines, used_ocr = extract_lines(page)
        all_lines.append(lines)
        page_ocr.append(used_ocr)

        for index, line in enumerate(lines):
            normalized = normalize_label(line.text).replace(" ", "")
            if normalized == "REDACAO" or normalized.startswith("PROPOSTADEREDACAO"):
                stop_boundaries.append(Boundary(page.number, line.y0))
            if fixed_subject is None:
                subject = canonical_subject(line.text)
                if subject:
                    current_subject = subject
                    continue

            match = match_question_header(lines, index)
            if not match:
                continue
            number = parse_question_number(match.group(1))
            if current_subject is None:
                continue
            candidates.append(QuestionStart(year, number, current_subject, page.number, line.y0))

    ordered = sorted(candidates, key=lambda item: (item.page_index, item.y0, item.number))
    if not ordered:
        return [], stop_boundaries, all_lines, page_ocr

    starts: list[QuestionStart] = []
    expected = 1
    for item in ordered:
        if item.number == expected:
            starts.append(item)
            expected += 1

    stop_boundaries.sort(key=lambda item: (item.page_index, item.y0))
    return starts, stop_boundaries, all_lines, page_ocr


def question_segments(
    doc: fitz.Document,
    start: QuestionStart,
    end: QuestionStart | Boundary | None,
) -> list[PageSegment]:
    last_page = end.page_index if end else len(doc) - 1
    result: list[PageSegment] = []
    for page_index in range(start.page_index, last_page + 1):
        page = doc[page_index]
        top = start.y0 if page_index == start.page_index else 0.0
        bottom = end.y0 if end and page_index == end.page_index else page.rect.height
        if bottom > top + 1:
            result.append(PageSegment(page_index, top, bottom))
    return result


def next_boundary(
    start: QuestionStart,
    next_question: QuestionStart | None,
    stop_boundaries: list[Boundary],
) -> QuestionStart | Boundary | None:
    candidates: list[QuestionStart | Boundary] = []
    if next_question is not None:
        candidates.append(next_question)
    for boundary in stop_boundaries:
        if (boundary.page_index, boundary.y0) > (start.page_index, start.y0):
            candidates.append(boundary)
            break
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item.page_index, item.y0))


def segment_text(all_lines: list[list[TextLine]], segments: Iterable[PageSegment]) -> str:
    chunks: list[str] = []
    for segment in segments:
        page_lines = all_lines[segment.page_index]
        if not page_lines:
            continue
        page_bottom = max(line.y1 for line in page_lines)
        for line in page_lines:
            if line.y0 < segment.y0 - 1 or line.y0 >= segment.y1 - 1:
                continue
            if line.y0 > 0.94 * page_bottom and line.text.isdigit():
                continue
            chunks.append(line.text)
    return re.sub(r"\s+", " ", " ".join(chunks)).strip()


def strip_question_header(raw: str, expected_number: int) -> tuple[str, bool]:
    match = QUESTION_RE.search(raw)
    if not match:
        return raw.strip(), True
    try:
        number = parse_question_number(match.group(1))
    except ValueError:
        return raw.strip(), True
    if number != expected_number:
        return raw.strip(), True
    return raw[match.end():].strip(" .:-"), False


def smallest_complete_alternative_window(body: str) -> list[AlternativeMarker]:
    candidates = [
        AlternativeMarker(match.group(1).upper(), match.start(), match.end())
        for match in STANDALONE_ALTERNATIVE_RE.finditer(body)
    ]
    required = set("ABCDE")
    best: tuple[int, int, int] | None = None
    counts: dict[str, int] = {}
    left = 0
    for right, marker in enumerate(candidates):
        counts[marker.label] = counts.get(marker.label, 0) + 1
        while required.issubset(counts):
            span = candidates[right].end - candidates[left].start
            if best is None or span < best[0]:
                best = (span, left, right)
            label = candidates[left].label
            counts[label] -= 1
            if counts[label] == 0:
                del counts[label]
            left += 1
    if best is None:
        return []
    window = candidates[best[1] : best[2] + 1]
    selected: dict[str, AlternativeMarker] = {}
    for marker in window:
        selected.setdefault(marker.label, marker)
    if set(selected) != required:
        return []
    return sorted(selected.values(), key=lambda item: item.start)


def locate_alternative_markers(body: str) -> tuple[list[AlternativeMarker], bool]:
    primary: dict[str, AlternativeMarker] = {}
    for pattern in (ALTERNATIVE_PAREN_RE, ALTERNATIVE_RE):
        for match in pattern.finditer(body):
            label = match.group(1).upper()
            primary.setdefault(label, AlternativeMarker(label, match.start(), match.end()))
        if set(primary) == set("ABCDE"):
            return sorted(primary.values(), key=lambda item: item.start), False
    fallback = smallest_complete_alternative_window(body)
    return fallback, bool(fallback)


def split_multiple_choice(body: str) -> tuple[str, dict[str, str], bool]:
    markers, used_fallback = locate_alternative_markers(body)
    complete = {marker.label for marker in markers} == set("ABCDE")
    if not complete:
        return body.strip(), {label: "" for label in "ABCDE"}, True

    statement = body[: markers[0].start].strip()
    alternatives: dict[str, str] = {}
    for index, marker in enumerate(markers):
        next_start = markers[index + 1].start if index + 1 < len(markers) else len(body)
        value = body[marker.end:next_start].strip()
        if used_fallback:
            value = re.sub(r"^\s*(?:\(\s*[O0]?\s*\)|[O0](?=\s))\s*", "", value)
        alternatives[marker.label] = value

    incomplete = not statement or any(not alternatives[label] for label in "ABCDE")
    if not statement:
        statement = body.strip()
    return statement, alternatives, used_fallback or incomplete


def classify_question(
    *,
    year: int,
    exam_phase: str,
    subject: str,
    number: int,
    body: str,
) -> QuestionKind:
    if LEGACY_FIRST_YEAR <= year <= LEGACY_LAST_YEAR and exam_phase == "single_phase":
        if subject in SCIENCE_SUBJECTS:
            return "multiple_choice" if number <= 20 else "essay"
        if subject in LANGUAGE_SUBJECTS:
            return "multiple_choice"

    if exam_phase == "phase_1":
        return "multiple_choice"

    if exam_phase == "phase_2" and subject in SCIENCE_SUBJECTS:
        return "essay"

    markers, _ = locate_alternative_markers(body)
    return "multiple_choice" if {m.label for m in markers} == set("ABCDE") else "essay"


def question_id(kind: QuestionKind, year: int, phase: str, subject: str, number: int) -> str:
    kind_code = "MC" if kind == "multiple_choice" else "ESSAY"
    return f"ITA-{kind_code}-{year}-{PHASE_CODES[phase]}-{SUBJECT_CODES[subject]}-Q{number:03d}"


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
    qid: str,
    segments: list[PageSegment],
    image_root: Path,
    scale: float,
    force: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    counter = 0
    year = str(next(part for part in qid.split("-") if part.isdigit() and len(part) == 4))
    output_dir = image_root / year

    for segment in segments:
        page = doc[segment.page_index]
        clip = fitz.Rect(0, segment.y0, page.rect.width, segment.y1)
        if not force and not segment_has_graphics(page, clip):
            continue
        counter += 1
        image_id = f"{qid}-IMG{counter:02d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{image_id}.png"
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
        pix.save(output_path)
        rows.append(
            {
                "image_id": image_id,
                "question_id": qid,
                "image_path": output_path.as_posix(),
                "page": segment.page_index + 1,
                "width_px": pix.width,
                "height_px": pix.height,
            }
        )
    return rows


def parse_pdf(
    pdf: Path,
    *,
    mc_image_root: Path,
    essay_image_root: Path,
    scale: float,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    year = infer_year(pdf)
    phase = infer_phase(pdf)
    fixed_subject = infer_subject_from_filename(pdf)

    mc_rows: list[dict[str, object]] = []
    essay_rows: list[dict[str, object]] = []
    mc_images: list[dict[str, object]] = []
    essay_images: list[dict[str, object]] = []

    with fitz.open(pdf) as doc:
        starts, stop_boundaries, all_lines, page_ocr = discover_questions(doc, year, fixed_subject)
        if not starts:
            print(f"[skip ] {pdf}: no numbered questions found")
            return mc_rows, essay_rows, mc_images, essay_images

        for index, start in enumerate(starts):
            next_question = starts[index + 1] if index + 1 < len(starts) else None
            end = next_boundary(start, next_question, stop_boundaries)
            segments = question_segments(doc, start, end)
            if not segments:
                continue
            raw = segment_text(all_lines, segments)
            body, header_review = strip_question_header(raw, start.number)
            ocr_used = any(page_ocr[segment.page_index] for segment in segments)
            kind = classify_question(
                year=year,
                exam_phase=phase,
                subject=start.subject,
                number=start.number,
                body=body,
            )
            qid = question_id(kind, year, phase, start.subject, start.number)
            source_pdf = pdf.as_posix()
            page_start = segments[0].page_index + 1
            page_end = segments[-1].page_index + 1

            if kind == "multiple_choice":
                statement, alternatives, split_review = split_multiple_choice(body)
                needs_review = header_review or split_review
                images = render_question_images(
                    doc, qid, segments, mc_image_root, scale, force=needs_review
                )
                has_image = bool(images)
                for label in "ABCDE":
                    mc_rows.append(
                        {
                            "alternative_id": f"{qid}-{label}",
                            "question_id": qid,
                            "year": year,
                            "exam_phase": phase,
                            "question_number": start.number,
                            "subject": start.subject,
                            "question_text": statement,
                            "alternative": label,
                            "alternative_text": alternatives.get(label, ""),
                            "has_image": has_image,
                            "ocr_used": ocr_used,
                            "needs_review": needs_review,
                            "source_pdf": source_pdf,
                            "page_start": page_start,
                            "page_end": page_end,
                        }
                    )
                mc_images.extend(images)
            else:
                question_text = body.strip()
                needs_review = header_review or not question_text
                images = render_question_images(
                    doc, qid, segments, essay_image_root, scale, force=needs_review
                )
                has_image = bool(images)
                essay_rows.append(
                    {
                        "question_id": qid,
                        "year": year,
                        "exam_phase": phase,
                        "question_number": start.number,
                        "subject": start.subject,
                        "question_text": question_text,
                        "has_image": has_image,
                        "ocr_used": ocr_used,
                        "needs_review": needs_review,
                        "source_pdf": source_pdf,
                        "page_start": page_start,
                        "page_end": page_end,
                    }
                )
                essay_images.extend(images)

    print(
        f"[parse] {pdf}: "
        f"{len({row['question_id'] for row in mc_rows})} MC, "
        f"{len(essay_rows)} essay"
    )
    return mc_rows, essay_rows, mc_images, essay_images


def write_parquet(rows: list[dict[str, object]], columns: list[str], path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_parquet(path, index=False)
    return frame


def write_excel(
    questions: pd.DataFrame,
    images: pd.DataFrame,
    path: Path,
    *,
    text_columns: set[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        questions.to_excel(writer, sheet_name="questions", index=False)
        images.to_excel(writer, sheet_name="images", index=False)
        workbook = writer.book
        header_format = workbook.add_format({"bold": True, "text_wrap": False})
        wrap_format = workbook.add_format({"text_wrap": True, "valign": "top"})

        for sheet_name, frame in (("questions", questions), ("images", images)):
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes(1, 0)
            if len(frame.columns):
                worksheet.autofilter(0, 0, max(len(frame), 1), len(frame.columns) - 1)
            for col_index, column in enumerate(frame.columns):
                worksheet.write(0, col_index, column, header_format)
                if column in text_columns:
                    width = 55 if column in {"question_text", "alternative_text"} else 36
                    worksheet.set_column(col_index, col_index, width, wrap_format)
                elif column in {"source_pdf", "image_path", "question_id", "alternative_id", "image_id"}:
                    worksheet.set_column(col_index, col_index, 42)
                else:
                    worksheet.set_column(col_index, col_index, max(12, min(22, len(column) + 3)))


def validate_xlsx(path: Path) -> None:
    if not path.is_file() or not zipfile.is_zipfile(path):
        raise ValueError(f"Missing or invalid XLSX: {path}")
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        required = {"xl/workbook.xml", "xl/worksheets/sheet1.xml", "xl/worksheets/sheet2.xml"}
        missing = required - names
        if missing:
            raise ValueError(f"{path}: missing XLSX members {sorted(missing)}")


def validate_outputs(
    mc: pd.DataFrame,
    essay: pd.DataFrame,
    mc_images: pd.DataFrame,
    essay_images: pd.DataFrame,
) -> None:
    if list(mc.columns) != MC_COLUMNS:
        raise ValueError(f"Unexpected MC schema: {list(mc.columns)}")
    if list(essay.columns) != ESSAY_COLUMNS:
        raise ValueError(f"Unexpected essay schema: {list(essay.columns)}")
    if list(mc_images.columns) != IMAGE_COLUMNS or list(essay_images.columns) != IMAGE_COLUMNS:
        raise ValueError("Unexpected image schema")

    if mc["alternative_id"].duplicated().any():
        raise ValueError("alternative_id is not unique")
    if essay["question_id"].duplicated().any():
        raise ValueError("essay question_id is not unique")
    if mc_images["image_id"].duplicated().any() or essay_images["image_id"].duplicated().any():
        raise ValueError("image_id is not unique")

    if not mc.empty:
        counts = mc.groupby("question_id")["alternative"].nunique()
        if not (counts == 5).all():
            raise ValueError("At least one multiple-choice question does not have A-E rows")
        expected_labels = mc.groupby("question_id")["alternative"].apply(lambda s: set(s))
        if not expected_labels.apply(lambda labels: labels == set("ABCDE")).all():
            raise ValueError("At least one multiple-choice question does not contain A-E")

    for frame in (mc_images, essay_images):
        if any(not Path(path).is_file() for path in frame["image_path"].tolist()):
            raise ValueError("At least one image_path is missing")

    mc_review_without_image = mc["needs_review"] & ~mc["has_image"]
    essay_review_without_image = essay["needs_review"] & ~essay["has_image"]
    if mc_review_without_image.any() or essay_review_without_image.any():
        raise ValueError("A question needing review has no rendered source image")

    validate_xlsx(MC_EXCEL)
    validate_xlsx(ESSAY_EXCEL)


def clean_output_roots() -> None:
    for root in (MC_IMAGE_ROOT, ESSAY_IMAGE_ROOT):
        if root.exists():
            shutil.rmtree(root)
    legacy_paths = [
        DATA_ROOT / "ita_phase_1.parquet",
        DATA_ROOT / "ita_phase_1_images.parquet",
        DATA_ROOT / "ita_phase_1.xlsx",
        DATA_ROOT / "ita_phase_1_images",
    ]
    for path in legacy_paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def build(input_root: Path, scale: float) -> None:
    clean_output_roots()
    pdfs = sorted(pdf for pdf in input_root.glob("*/*.pdf") if not should_skip_pdf(pdf))
    if not pdfs:
        raise SystemExit(f"No ITA PDFs found under {input_root}")

    all_mc: list[dict[str, object]] = []
    all_essay: list[dict[str, object]] = []
    all_mc_images: list[dict[str, object]] = []
    all_essay_images: list[dict[str, object]] = []

    for pdf in pdfs:
        mc, essay, mc_images, essay_images = parse_pdf(
            pdf,
            mc_image_root=MC_IMAGE_ROOT,
            essay_image_root=ESSAY_IMAGE_ROOT,
            scale=scale,
        )
        all_mc.extend(mc)
        all_essay.extend(essay)
        all_mc_images.extend(mc_images)
        all_essay_images.extend(essay_images)

    all_mc.sort(key=lambda r: (r["year"], r["exam_phase"], r["subject"], r["question_number"], r["alternative"]))
    all_essay.sort(key=lambda r: (r["year"], r["exam_phase"], r["subject"], r["question_number"]))
    all_mc_images.sort(key=lambda r: r["image_id"])
    all_essay_images.sort(key=lambda r: r["image_id"])

    mc_df = write_parquet(all_mc, MC_COLUMNS, MC_DATASET)
    essay_df = write_parquet(all_essay, ESSAY_COLUMNS, ESSAY_DATASET)
    mc_images_df = write_parquet(all_mc_images, IMAGE_COLUMNS, MC_IMAGE_DATASET)
    essay_images_df = write_parquet(all_essay_images, IMAGE_COLUMNS, ESSAY_IMAGE_DATASET)

    write_excel(mc_df, mc_images_df, MC_EXCEL, text_columns={"question_text", "alternative_text"})
    write_excel(essay_df, essay_images_df, ESSAY_EXCEL, text_columns={"question_text"})
    validate_outputs(mc_df, essay_df, mc_images_df, essay_images_df)

    mc_questions = mc_df["question_id"].nunique()
    print(f"[done ] multiple choice: {mc_questions} questions, {len(mc_df)} alternatives, {len(mc_images_df)} images")
    print(f"[done ] essay: {len(essay_df)} questions, {len(essay_images_df)} images")
    print(
        "        review: "
        f"{mc_df.loc[mc_df['needs_review'], 'question_id'].nunique()} MC, "
        f"{int(essay_df['needs_review'].sum())} essay"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--scale", type=float, default=2.0, help="PNG render scale")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build(args.input_root, args.scale)


if __name__ == "__main__":
    main()
