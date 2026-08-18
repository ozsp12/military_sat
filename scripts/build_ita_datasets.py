#!/usr/bin/env python3
"""Build ITA multiple-choice and essay-question datasets from archived PDFs.

The archive changes format over time. The builder therefore classifies numbered
questions by type, keeps phase as metadata, and retries an incomplete known exam
with full-page OCR before accepting a partial sequence.
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

INPUT_ROOT = Path("ita_provas")
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
LEGACY_FIRST_YEAR = 2008
LEGACY_LAST_YEAR = 2018

QUESTION_RE = re.compile(
    r"\bQuest.{0,12}?\s*([0-9SOIl]{1,3})\s*[\.:]", re.IGNORECASE
)
QUESTION_PREFIX_RE = re.compile(r"\bQuest", re.IGNORECASE)
BARE_QUESTION_RE = re.compile(r"^\s*([0-9]{1,3})\.\s+\S")
ALTERNATIVE_RE = re.compile(
    r"(?<![A-Za-zÀ-ÿ])([A-E])\s*[\(\[]\s*[\)\]]", re.IGNORECASE
)
ALTERNATIVE_PAREN_RE = re.compile(
    r"(?<![A-Za-zÀ-ÿ])([A-E])\s*\(\s*\)", re.IGNORECASE
)
STANDALONE_ALTERNATIVE_RE = re.compile(
    r"(?<![A-Za-zÀ-ÿ0-9])([A-E])(?=[\s\)\.\-:])"
)
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
PHASE1_COUNTS = {
    2019: 60,
    2020: 70,
    2021: 70,
    2022: 70,
    2023: 60,
    2024: 60,
    2025: 48,
    2026: 48,
}

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
class Boundary:
    page_index: int
    y0: float


@dataclass(frozen=True)
class AlternativeMarker:
    label: str
    start: int
    end: int


def normalize_label(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^A-Za-z0-9 ]+", " ", value).upper()
    return re.sub(r"\s+", " ", value).strip()


def canonical_subject(value: str) -> str | None:
    return SUBJECT_ALIASES.get(normalize_label(value).replace(" ", ""))


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
    raise ValueError(f"Cannot infer year from {pdf}")


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


def expected_question_count(year: int, phase: str, subject: str | None) -> int | None:
    if LEGACY_FIRST_YEAR <= year <= LEGACY_LAST_YEAR and phase == "single_phase":
        return 30 if subject in SCIENCE_SUBJECTS else 20 if subject in LANGUAGE_SUBJECTS else None
    if phase == "phase_1":
        return PHASE1_COUNTS.get(year)
    if phase == "phase_2" and subject in SCIENCE_SUBJECTS:
        return 10
    if phase == "phase_2" and subject == "portugues" and year >= 2025:
        return 15
    return None


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
    return sum(sum(ch.isalnum() for ch in line.text) for line in lines)


def native_text_is_suspicious(lines: list[TextLine]) -> bool:
    text = " ".join(line.text for line in lines)
    if not text:
        return True
    corrupted_tokens = ("4XHVW", "&RQVWDQWH", "SURSRVLo", "TXHVWmR")
    if any(token in text for token in corrupted_tokens):
        return True
    controls = sum(ord(ch) < 32 and ch not in "\n\r\t" for ch in text)
    dollar_signs = text.count("$")
    return controls >= 3 or dollar_signs >= max(20, len(text) // 100)


def ocr_lines(page: fitz.Page) -> list[TextLine]:
    try:
        textpage = page.get_textpage_ocr(
            language=OCR_LANGUAGE,
            dpi=OCR_DPI,
            full=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Page {page.number + 1} requires OCR. Install Tesseract por+eng data."
        ) from exc
    return payload_to_lines(page, page.get_text("dict", textpage=textpage, sort=True))


def extract_lines(page: fitz.Page, force_ocr: bool = False) -> tuple[list[TextLine], bool]:
    native = payload_to_lines(page, page.get_text("dict", sort=True))
    if force_ocr:
        ocr = ocr_lines(page)
        return (ocr, True) if ocr else (native, False)

    native_count = alphanumeric_count(native)
    suspicious = native_text_is_suspicious(native)
    if native_count >= OCR_MIN_CHARS and not suspicious:
        return native, False

    ocr = ocr_lines(page)
    ocr_count = alphanumeric_count(ocr)
    if suspicious and ocr_count >= max(20, OCR_MIN_CHARS // 2):
        return ocr, True
    if ocr_count > native_count:
        return ocr, True
    return native, False


def match_question_header(lines: list[TextLine], index: int) -> re.Match[str] | None:
    line = lines[index]
    match = QUESTION_RE.search(line.text)
    if match or not QUESTION_PREFIX_RE.search(line.text) or index + 1 >= len(lines):
        return match
    nxt = lines[index + 1]
    if nxt.page_index != line.page_index or nxt.y0 - line.y1 > 28:
        return None
    return QUESTION_RE.search(f"{line.text} {nxt.text}")


def bare_question_candidates(
    all_lines: list[list[TextLine]], year: int, subject: str | None
) -> list[QuestionStart]:
    if subject is None:
        return []
    result: list[QuestionStart] = []
    for page_index, lines in enumerate(all_lines):
        for line in lines:
            match = BARE_QUESTION_RE.match(line.text)
            if not match:
                continue
            number = int(match.group(1))
            if 1 <= number <= 40:
                result.append(QuestionStart(year, number, subject, page_index, line.y0))
    return result


def choose_sequence(candidates: list[QuestionStart], expected_count: int | None) -> list[QuestionStart]:
    ordered = sorted(candidates, key=lambda q: (q.page_index, q.y0, q.number))
    if not ordered:
        return []

    # Prefer a sequence beginning at 1. Legacy Portuguese is numbered 21--40,
    # so if no question 1 exists the first physical marker is the seed.
    seed_index = next((i for i, q in enumerate(ordered) if q.number == 1), 0)
    seed = ordered[seed_index]
    result = [seed]
    expected = seed.number + 1
    for item in ordered[seed_index + 1 :]:
        if item.number == expected:
            result.append(item)
            expected += 1
            if expected_count and len(result) >= expected_count:
                break
    return result


def discover_from_lines(
    all_lines: list[list[TextLine]],
    year: int,
    fixed_subject: str | None,
    expected_count: int | None,
) -> tuple[list[QuestionStart], list[Boundary]]:
    candidates: list[QuestionStart] = []
    stop_boundaries: list[Boundary] = []
    current_subject = fixed_subject

    for page_index, lines in enumerate(all_lines):
        for index, line in enumerate(lines):
            normalized = normalize_label(line.text).replace(" ", "")
            if normalized == "REDACAO" or normalized.startswith("PROPOSTADEREDACAO"):
                stop_boundaries.append(Boundary(page_index, line.y0))

            if fixed_subject is None:
                subject = canonical_subject(line.text)
                if subject:
                    current_subject = subject
                    continue

            match = match_question_header(lines, index)
            if not match or current_subject is None:
                continue
            candidates.append(
                QuestionStart(
                    year,
                    parse_question_number(match.group(1)),
                    current_subject,
                    page_index,
                    line.y0,
                )
            )

    # Bare numeric headers are only considered for single-subject PDFs. They are
    # useful for occasional layout anomalies but cannot displace a normal marker:
    # physical order + strict sequential selection still determines acceptance.
    candidates.extend(bare_question_candidates(all_lines, year, fixed_subject))
    starts = choose_sequence(candidates, expected_count)
    stop_boundaries.sort(key=lambda b: (b.page_index, b.y0))
    return starts, stop_boundaries


def extract_document(
    doc: fitz.Document, force_ocr: bool = False
) -> tuple[list[list[TextLine]], list[bool]]:
    all_lines: list[list[TextLine]] = []
    page_ocr: list[bool] = []
    for page in doc:
        lines, used_ocr = extract_lines(page, force_ocr=force_ocr)
        all_lines.append(lines)
        page_ocr.append(used_ocr)
    return all_lines, page_ocr


def discover_questions(
    doc: fitz.Document,
    year: int,
    phase: str,
    fixed_subject: str | None,
) -> tuple[list[QuestionStart], list[Boundary], list[list[TextLine]], list[bool]]:
    expected_count = expected_question_count(year, phase, fixed_subject)
    all_lines, page_ocr = extract_document(doc, force_ocr=False)
    starts, stops = discover_from_lines(all_lines, year, fixed_subject, expected_count)

    if expected_count is not None and len(starts) < expected_count:
        print(
            f"[ocr  ] retry {year} {fixed_subject or phase}: "
            f"{len(starts)}/{expected_count} question markers"
        )
        ocr_all_lines, ocr_page_flags = extract_document(doc, force_ocr=True)
        ocr_starts, ocr_stops = discover_from_lines(
            ocr_all_lines, year, fixed_subject, expected_count
        )
        if len(ocr_starts) > len(starts):
            starts, stops = ocr_starts, ocr_stops
            all_lines, page_ocr = ocr_all_lines, ocr_page_flags

    return starts, stops, all_lines, page_ocr


def next_boundary(
    start: QuestionStart,
    next_question: QuestionStart | None,
    stop_boundaries: list[Boundary],
) -> QuestionStart | Boundary | None:
    choices: list[QuestionStart | Boundary] = []
    if next_question is not None:
        choices.append(next_question)
    for boundary in stop_boundaries:
        if (boundary.page_index, boundary.y0) > (start.page_index, start.y0):
            choices.append(boundary)
            break
    return min(choices, key=lambda x: (x.page_index, x.y0)) if choices else None


def question_segments(
    doc: fitz.Document,
    start: QuestionStart,
    end: QuestionStart | Boundary | None,
) -> list[PageSegment]:
    last_page = end.page_index if end else len(doc) - 1
    segments: list[PageSegment] = []
    for page_index in range(start.page_index, last_page + 1):
        page = doc[page_index]
        y0 = start.y0 if page_index == start.page_index else 0.0
        y1 = end.y0 if end and page_index == end.page_index else page.rect.height
        if y1 > y0 + 1:
            segments.append(PageSegment(page_index, y0, y1))
    return segments


def segment_text(all_lines: list[list[TextLine]], segments: Iterable[PageSegment]) -> str:
    chunks: list[str] = []
    for segment in segments:
        for line in all_lines[segment.page_index]:
            if segment.y0 - 1 <= line.y0 < segment.y1 - 1:
                chunks.append(line.text)
    return re.sub(r"\s+", " ", " ".join(chunks)).strip()


def strip_question_header(raw: str, expected_number: int) -> tuple[str, bool]:
    match = QUESTION_RE.search(raw)
    if match:
        try:
            number = parse_question_number(match.group(1))
        except ValueError:
            return raw.strip(), True
        if number == expected_number:
            return raw[match.end() :].strip(" .:-"), False
        return raw.strip(), True

    bare = BARE_QUESTION_RE.match(raw)
    if bare and int(bare.group(1)) == expected_number:
        return raw[bare.end() :].strip(" .:-"), False
    return raw.strip(), True


def smallest_complete_alternative_window(body: str) -> list[AlternativeMarker]:
    candidates = [
        AlternativeMarker(m.group(1).upper(), m.start(), m.end())
        for m in STANDALONE_ALTERNATIVE_RE.finditer(body)
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
    return sorted(selected.values(), key=lambda m: m.start)


def locate_alternative_markers(body: str) -> tuple[list[AlternativeMarker], bool]:
    for pattern in (ALTERNATIVE_PAREN_RE, ALTERNATIVE_RE):
        found: dict[str, AlternativeMarker] = {}
        for match in pattern.finditer(body):
            label = match.group(1).upper()
            found.setdefault(label, AlternativeMarker(label, match.start(), match.end()))
        if set(found) == set("ABCDE"):
            return sorted(found.values(), key=lambda m: m.start), False
    fallback = smallest_complete_alternative_window(body)
    return fallback, bool(fallback)


def split_multiple_choice(body: str) -> tuple[str, dict[str, str], bool]:
    markers, used_fallback = locate_alternative_markers(body)
    if {m.label for m in markers} != set("ABCDE"):
        return body.strip(), {label: "" for label in "ABCDE"}, True

    statement = body[: markers[0].start].strip()
    alternatives: dict[str, str] = {}
    for index, marker in enumerate(markers):
        nxt = markers[index + 1].start if index + 1 < len(markers) else len(body)
        value = body[marker.end : nxt].strip()
        if used_fallback:
            value = re.sub(r"^\s*(?:\(\s*[O0]?\s*\)|[O0](?=\s))\s*", "", value)
        alternatives[marker.label] = value

    incomplete = not statement or any(not alternatives.get(label) for label in "ABCDE")
    if not statement:
        statement = body.strip()
    return statement, alternatives, used_fallback or incomplete


def classify_question(
    year: int, phase: str, subject: str, number: int, body: str
) -> QuestionKind:
    if LEGACY_FIRST_YEAR <= year <= LEGACY_LAST_YEAR and phase == "single_phase":
        if subject in SCIENCE_SUBJECTS:
            return "multiple_choice" if number <= 20 else "essay"
        if subject in LANGUAGE_SUBJECTS:
            return "multiple_choice"

    if phase == "phase_1":
        return "multiple_choice"
    if phase == "phase_2" and subject in SCIENCE_SUBJECTS:
        return "essay"
    if phase == "phase_2" and subject == "portugues" and year >= 2025:
        return "multiple_choice"

    markers, _ = locate_alternative_markers(body)
    return "multiple_choice" if {m.label for m in markers} == set("ABCDE") else "essay"


def make_question_id(
    kind: QuestionKind, year: int, phase: str, subject: str, number: int
) -> str:
    kind_code = "MC" if kind == "multiple_choice" else "ESSAY"
    return (
        f"ITA-{kind_code}-{year}-{PHASE_CODES[phase]}-"
        f"{SUBJECT_CODES[subject]}-Q{number:03d}"
    )


def rect_overlap(a: fitz.Rect, b: fitz.Rect) -> float:
    intersection = a & b
    return max(0.0, intersection.width) * max(0.0, intersection.height)


def segment_has_graphics(page: fitz.Page, clip: fitz.Rect) -> bool:
    for image in page.get_images(full=True):
        for rect in page.get_image_rects(image[0]):
            if rect_overlap(rect, clip) >= 100:
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
    year: int,
    segments: list[PageSegment],
    image_root: Path,
    scale: float,
    force: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    output_dir = image_root / str(year)
    counter = 0

    for segment in segments:
        page = doc[segment.page_index]
        clip = fitz.Rect(0, segment.y0, page.rect.width, segment.y1)
        if not force and not segment_has_graphics(page, clip):
            continue
        counter += 1
        image_id = f"{qid}-IMG{counter:02d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{image_id}.png"
        pix = page.get_pixmap(
            matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False
        )
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
    pdf: Path, scale: float
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
        starts, stops, all_lines, page_ocr = discover_questions(
            doc, year, phase, fixed_subject
        )
        if not starts:
            print(f"[skip ] {pdf}: no numbered questions found")
            return mc_rows, essay_rows, mc_images, essay_images

        for index, start in enumerate(starts):
            next_question = starts[index + 1] if index + 1 < len(starts) else None
            end = next_boundary(start, next_question, stops)
            segments = question_segments(doc, start, end)
            if not segments:
                continue

            raw = segment_text(all_lines, segments)
            body, header_review = strip_question_header(raw, start.number)
            ocr_used = any(page_ocr[s.page_index] for s in segments)
            kind = classify_question(year, phase, start.subject, start.number, body)
            qid = make_question_id(kind, year, phase, start.subject, start.number)
            source_pdf = pdf.as_posix()
            page_start = segments[0].page_index + 1
            page_end = segments[-1].page_index + 1

            if kind == "multiple_choice":
                statement, alternatives, split_review = split_multiple_choice(body)
                needs_review = header_review or split_review
                images = render_question_images(
                    doc,
                    qid,
                    year,
                    segments,
                    MC_IMAGE_ROOT,
                    scale,
                    needs_review,
                )
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
                            "has_image": bool(images),
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
                    doc,
                    qid,
                    year,
                    segments,
                    ESSAY_IMAGE_ROOT,
                    scale,
                    needs_review,
                )
                essay_rows.append(
                    {
                        "question_id": qid,
                        "year": year,
                        "exam_phase": phase,
                        "question_number": start.number,
                        "subject": start.subject,
                        "question_text": question_text,
                        "has_image": bool(images),
                        "ocr_used": ocr_used,
                        "needs_review": needs_review,
                        "source_pdf": source_pdf,
                        "page_start": page_start,
                        "page_end": page_end,
                    }
                )
                essay_images.extend(images)

    print(
        f"[parse] {pdf}: {len({r['question_id'] for r in mc_rows})} MC, "
        f"{len(essay_rows)} essay"
    )
    return mc_rows, essay_rows, mc_images, essay_images


def write_parquet(
    rows: list[dict[str, object]], columns: list[str], path: Path
) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_parquet(path, index=False)
    return frame


def write_excel(questions: pd.DataFrame, images: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        questions.to_excel(writer, sheet_name="questions", index=False)
        images.to_excel(writer, sheet_name="images", index=False)
        workbook = writer.book
        header = workbook.add_format({"bold": True, "valign": "top"})
        wrap = workbook.add_format({"text_wrap": True, "valign": "top"})

        for sheet_name, frame in (("questions", questions), ("images", images)):
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes(1, 0)
            if len(frame.columns):
                worksheet.autofilter(
                    0, 0, max(len(frame), 1), len(frame.columns) - 1
                )
            for column_index, column in enumerate(frame.columns):
                worksheet.write(0, column_index, column, header)
                if column in {"question_text", "alternative_text"}:
                    worksheet.set_column(column_index, column_index, 55, wrap)
                elif column in {
                    "source_pdf",
                    "image_path",
                    "question_id",
                    "alternative_id",
                    "image_id",
                }:
                    worksheet.set_column(column_index, column_index, 42, wrap)
                else:
                    worksheet.set_column(
                        column_index,
                        column_index,
                        max(12, min(22, len(column) + 3)),
                    )


def validate_xlsx(path: Path) -> None:
    if not path.is_file() or not zipfile.is_zipfile(path):
        raise ValueError(f"Invalid XLSX: {path}")
    with zipfile.ZipFile(path) as archive:
        required = {
            "xl/workbook.xml",
            "xl/worksheets/sheet1.xml",
            "xl/worksheets/sheet2.xml",
        }
        missing = required - set(archive.namelist())
        if missing:
            raise ValueError(f"{path}: missing XLSX members: {sorted(missing)}")


def question_frame(mc: pd.DataFrame) -> pd.DataFrame:
    return mc.drop_duplicates("question_id")


def validate_historical_counts(mc: pd.DataFrame, essay: pd.DataFrame) -> None:
    mcq = question_frame(mc)

    for year in range(LEGACY_FIRST_YEAR, LEGACY_LAST_YEAR + 1):
        year_mc = mcq[
            (mcq["year"] == year) & (mcq["exam_phase"] == "single_phase")
        ]
        year_essay = essay[
            (essay["year"] == year) & (essay["exam_phase"] == "single_phase")
        ]
        mc_counts = year_mc.groupby("subject").size().to_dict()
        essay_counts = year_essay.groupby("subject").size().to_dict()
        expected_mc = {
            "fisica": 20,
            "ingles": 20,
            "matematica": 20,
            "portugues": 20,
            "quimica": 20,
        }
        expected_essay = {"fisica": 10, "matematica": 10, "quimica": 10}
        if mc_counts != expected_mc:
            raise ValueError(
                f"{year}: legacy MC counts {mc_counts}, expected {expected_mc}"
            )
        if essay_counts != expected_essay:
            raise ValueError(
                f"{year}: legacy essay counts {essay_counts}, expected {expected_essay}"
            )

    for year, expected in PHASE1_COUNTS.items():
        count = len(
            mcq[(mcq["year"] == year) & (mcq["exam_phase"] == "phase_1")]
        )
        if count != expected:
            raise ValueError(
                f"{year}: phase-1 MC count {count}, expected {expected}"
            )

        science_essay = essay[
            (essay["year"] == year)
            & (essay["exam_phase"] == "phase_2")
            & (essay["subject"].isin(SCIENCE_SUBJECTS))
        ]
        subject_counts = science_essay.groupby("subject").size().to_dict()
        expected_science = {"fisica": 10, "matematica": 10, "quimica": 10}
        if subject_counts != expected_science:
            raise ValueError(
                f"{year}: phase-2 science essay counts {subject_counts}, "
                f"expected {expected_science}"
            )

    for year in (2025, 2026):
        port_mc = mcq[
            (mcq["year"] == year)
            & (mcq["exam_phase"] == "phase_2")
            & (mcq["subject"] == "portugues")
        ]
        port_essay = essay[
            (essay["year"] == year)
            & (essay["exam_phase"] == "phase_2")
            & (essay["subject"] == "portugues")
        ]
        if len(port_mc) != 15 or len(port_essay) != 0:
            raise ValueError(
                f"{year}: expected 15 objective Portuguese questions and 0 essay; "
                f"found {len(port_mc)} MC and {len(port_essay)} essay"
            )

    if len(mcq) != 1616:
        raise ValueError(f"Expected 1616 multiple-choice questions, found {len(mcq)}")
    if len(essay) != 570:
        raise ValueError(f"Expected 570 essay questions, found {len(essay)}")


def validate_outputs(
    mc: pd.DataFrame,
    essay: pd.DataFrame,
    mc_images: pd.DataFrame,
    essay_images: pd.DataFrame,
) -> None:
    if list(mc.columns) != MC_COLUMNS or list(essay.columns) != ESSAY_COLUMNS:
        raise ValueError("Unexpected question schema")
    if (
        list(mc_images.columns) != IMAGE_COLUMNS
        or list(essay_images.columns) != IMAGE_COLUMNS
    ):
        raise ValueError("Unexpected image schema")
    if mc["alternative_id"].duplicated().any():
        raise ValueError("alternative_id is not unique")
    if essay["question_id"].duplicated().any():
        raise ValueError("essay question_id is not unique")
    if mc_images["image_id"].duplicated().any() or essay_images[
        "image_id"
    ].duplicated().any():
        raise ValueError("image_id is not unique")

    if not (mc.groupby("question_id")["alternative"].nunique() == 5).all():
        raise ValueError(
            "At least one multiple-choice question does not have five alternatives"
        )
    labels = mc.groupby("question_id")["alternative"].apply(set)
    if not labels.apply(lambda value: value == set("ABCDE")).all():
        raise ValueError("At least one multiple-choice question does not contain A-E")

    for frame in (mc_images, essay_images):
        if any(not Path(path).is_file() for path in frame["image_path"].tolist()):
            raise ValueError("At least one image_path is missing")

    if (mc["needs_review"] & ~mc["has_image"]).any():
        raise ValueError("An MC question needing review has no source image")
    if (essay["needs_review"] & ~essay["has_image"]).any():
        raise ValueError("An essay question needing review has no source image")
    if mc["source_pdf"].str.contains("redacao", case=False, na=False).any():
        raise ValueError("Writing prompt leaked into multiple-choice dataset")
    if essay["source_pdf"].str.contains("redacao", case=False, na=False).any():
        raise ValueError("Writing prompt leaked into essay dataset")

    validate_historical_counts(mc, essay)
    validate_xlsx(MC_EXCEL)
    validate_xlsx(ESSAY_EXCEL)


def clean_outputs() -> None:
    for path in (
        MC_IMAGE_ROOT,
        ESSAY_IMAGE_ROOT,
        DATA_ROOT / "ita_phase_1_images",
    ):
        if path.is_dir():
            shutil.rmtree(path)

    for path in (
        MC_DATASET,
        MC_IMAGE_DATASET,
        MC_EXCEL,
        ESSAY_DATASET,
        ESSAY_IMAGE_DATASET,
        ESSAY_EXCEL,
        DATA_ROOT / "ita_phase_1.parquet",
        DATA_ROOT / "ita_phase_1_images.parquet",
        DATA_ROOT / "ita_phase_1.xlsx",
    ):
        if path.exists():
            path.unlink()


def build(input_root: Path, scale: float) -> None:
    clean_outputs()
    pdfs = sorted(
        pdf for pdf in input_root.glob("*/*.pdf") if not should_skip_pdf(pdf)
    )
    if not pdfs:
        raise SystemExit(f"No ITA PDFs found under {input_root}")

    mc_rows: list[dict[str, object]] = []
    essay_rows: list[dict[str, object]] = []
    mc_images: list[dict[str, object]] = []
    essay_images: list[dict[str, object]] = []

    for pdf in pdfs:
        a, b, c, d = parse_pdf(pdf, scale)
        mc_rows.extend(a)
        essay_rows.extend(b)
        mc_images.extend(c)
        essay_images.extend(d)

    mc_rows.sort(
        key=lambda r: (
            r["year"],
            r["exam_phase"],
            r["subject"],
            r["question_number"],
            r["alternative"],
        )
    )
    essay_rows.sort(
        key=lambda r: (
            r["year"], r["exam_phase"], r["subject"], r["question_number"]
        )
    )
    mc_images.sort(key=lambda r: r["image_id"])
    essay_images.sort(key=lambda r: r["image_id"])

    mc = write_parquet(mc_rows, MC_COLUMNS, MC_DATASET)
    essay = write_parquet(essay_rows, ESSAY_COLUMNS, ESSAY_DATASET)
    mc_image_df = write_parquet(mc_images, IMAGE_COLUMNS, MC_IMAGE_DATASET)
    essay_image_df = write_parquet(
        essay_images, IMAGE_COLUMNS, ESSAY_IMAGE_DATASET
    )
    write_excel(mc, mc_image_df, MC_EXCEL)
    write_excel(essay, essay_image_df, ESSAY_EXCEL)
    validate_outputs(mc, essay, mc_image_df, essay_image_df)

    print(
        f"[done ] multiple choice: {mc['question_id'].nunique()} questions, "
        f"{len(mc)} alternatives, {len(mc_image_df)} images"
    )
    print(f"[done ] essay: {len(essay)} questions, {len(essay_image_df)} images")
    print(
        "        review: "
        f"{mc.loc[mc['needs_review'], 'question_id'].nunique()} MC, "
        f"{int(essay['needs_review'].sum())} essay"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--scale", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build(args.input_root, args.scale)


if __name__ == "__main__":
    main()
