#!/usr/bin/env python3
"""Download official ITA entrance-exam PDFs and organize them by year."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse
from xml.sax.saxutils import escape

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

INDEX_URL = "https://www.vestibular.ita.br/provas.htm"
DEFAULT_OUTPUT = Path("ita_provas")
USER_AGENT = "military_sat/1.0 (+https://github.com/ozsp12/military_sat)"


@dataclass(frozen=True)
class ExamFile:
    year: int
    label: str
    url: str

    @property
    def filename(self) -> str:
        return Path(unquote(urlparse(self.url).path)).name


def build_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
    )
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def discover_files(session: requests.Session, include_gabaritos: bool = False) -> list[ExamFile]:
    response = session.get(INDEX_URL, timeout=(15, 60))
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")

    discovered: dict[str, ExamFile] = {}

    for row in soup.find_all("tr"):
        row_text = row.get_text(" ", strip=True)
        year_match = re.search(r"\b(20\d{2})\b", row_text)
        if not year_match:
            continue

        year = int(year_match.group(1))
        for link in row.find_all("a", href=True):
            label = link.get_text(" ", strip=True)
            href = link["href"].strip()
            url = urljoin(INDEX_URL, href)
            parsed = urlparse(url)

            if parsed.netloc != "www.vestibular.ita.br":
                continue
            if not parsed.path.lower().endswith(".pdf"):
                continue
            if not include_gabaritos and (
                "gabarito" in label.casefold() or "gabarito" in href.casefold()
            ):
                continue

            discovered[url] = ExamFile(year=year, label=label, url=url)

    return sorted(
        discovered.values(),
        key=lambda item: (-item.year, item.label.casefold(), item.filename.casefold()),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid_pdf(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    with path.open("rb") as handle:
        return handle.read(5) == b"%PDF-"


def download_file(
    session: requests.Session,
    item: ExamFile,
    output_root: Path,
    refresh: bool,
) -> Path:
    destination = output_root / str(item.year) / item.filename
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not refresh and is_valid_pdf(destination):
        print(f"[skip] {destination}")
        return destination

    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()

    print(f"[get ] {item.year} | {item.label:<10} | {item.filename}")
    try:
        with session.get(item.url, stream=True, timeout=(15, 180)) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)

        if not is_valid_pdf(temporary):
            raise RuntimeError(f"Downloaded file is not a valid PDF: {item.url}")

        if destination.exists() and is_valid_pdf(destination):
            if sha256_file(destination) == sha256_file(temporary):
                temporary.unlink()
                return destination

        temporary.replace(destination)
        return destination
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _xlsx_cell(ref: str, value: object, style: int | None = None) -> str:
    style_attr = f' s="{style}"' if style is not None else ""
    if isinstance(value, int):
        return f'<c r="{ref}"{style_attr}><v>{value}</v></c>'
    text = escape(str(value))
    return f'<c r="{ref}" t="inlineStr"{style_attr}><is><t>{text}</t></is></c>'


def _write_zip_text(archive: zipfile.ZipFile, name: str, content: str) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    archive.writestr(info, content.encode("utf-8"))


def write_manifest(rows: list[tuple[ExamFile, Path]], output_root: Path) -> None:
    """Write a deterministic, formatted Excel manifest without extra dependencies."""

    legacy_csv = output_root / "manifest.csv"
    legacy_csv.unlink(missing_ok=True)

    manifest = output_root / "manifest.xlsx"
    headers = ["year", "label", "path", "bytes", "sha256", "source_url"]
    data = [
        [
            item.year,
            item.label,
            path.as_posix(),
            path.stat().st_size,
            sha256_file(path),
            item.url,
        ]
        for item, path in rows
    ]

    last_row = len(data) + 1
    sheet_rows = [
        '<row r="1" ht="22" customHeight="1">'
        + "".join(
            _xlsx_cell(f"{column}1", header, style=1)
            for column, header in zip("ABCDEF", headers)
        )
        + "</row>"
    ]

    for row_number, values in enumerate(data, start=2):
        cells = []
        for column, value in zip("ABCDEF", values):
            style = 2 if column == "D" else None
            cells.append(_xlsx_cell(f"{column}{row_number}", value, style=style))
        sheet_rows.append(f'<row r="{row_number}">' + "".join(cells) + "</row>")

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  <Override PartName="/xl/tables/table1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.table+xml"/>
</Types>"""

    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""

    workbook = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <bookViews><workbookView xWindow="0" yWindow="0" windowWidth="24000" windowHeight="12000"/></bookViews>
  <sheets><sheet name="Manifest" sheetId="1" r:id="rId1"/></sheets>
</workbook>"""

    workbook_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

    styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2">
    <font><sz val="11"/><name val="Calibri"/><family val="2"/><scheme val="minor"/></font>
    <font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/><family val="2"/><scheme val="minor"/></font>
  </fonts>
  <fills count="3">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="3">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>
    <xf numFmtId="3" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
  <tableStyles count="0" defaultTableStyle="TableStyleMedium2" defaultPivotStyle="PivotStyleLight16"/>
</styleSheet>"""

    sheet = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <dimension ref="A1:F{last_row}"/>
  <sheetViews>
    <sheetView workbookViewId="0">
      <pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>
      <selection pane="bottomLeft" activeCell="A2" sqref="A2"/>
    </sheetView>
  </sheetViews>
  <sheetFormatPr defaultRowHeight="15"/>
  <cols>
    <col min="1" max="1" width="10" customWidth="1"/>
    <col min="2" max="2" width="18" customWidth="1"/>
    <col min="3" max="3" width="45" customWidth="1"/>
    <col min="4" max="4" width="14" customWidth="1"/>
    <col min="5" max="5" width="68" customWidth="1"/>
    <col min="6" max="6" width="65" customWidth="1"/>
  </cols>
  <sheetData>{''.join(sheet_rows)}</sheetData>
  <autoFilter ref="A1:F{last_row}"/>
  <tableParts count="1"><tablePart r:id="rId1"/></tableParts>
</worksheet>"""

    sheet_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/table" Target="../tables/table1.xml"/>
</Relationships>"""

    table_columns = "".join(
        f'<tableColumn id="{index}" name="{escape(header)}"/>'
        for index, header in enumerate(headers, start=1)
    )
    table = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<table xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" id="1" name="ManifestTable" displayName="ManifestTable" ref="A1:F{last_row}" totalsRowShown="0">
  <autoFilter ref="A1:F{last_row}"/>
  <tableColumns count="6">{table_columns}</tableColumns>
  <tableStyleInfo name="TableStyleMedium2" showFirstColumn="0" showLastColumn="0" showRowStripes="1" showColumnStripes="0"/>
</table>"""

    with zipfile.ZipFile(manifest, "w") as archive:
        _write_zip_text(archive, "[Content_Types].xml", content_types)
        _write_zip_text(archive, "_rels/.rels", root_rels)
        _write_zip_text(archive, "xl/workbook.xml", workbook)
        _write_zip_text(archive, "xl/_rels/workbook.xml.rels", workbook_rels)
        _write_zip_text(archive, "xl/styles.xml", styles)
        _write_zip_text(archive, "xl/worksheets/sheet1.xml", sheet)
        _write_zip_text(archive, "xl/worksheets/_rels/sheet1.xml.rels", sheet_rels)
        _write_zip_text(archive, "xl/tables/table1.xml", table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download official ITA entrance-exam PDFs by year."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--include-gabaritos",
        action="store_true",
        help="Also download answer-key PDFs.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download existing files and replace them only if content changed.",
    )
    parser.add_argument(
        "--min-files",
        type=int,
        default=1,
        help="Fail if fewer than this number of PDFs are discovered.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with build_session() as session:
        files = discover_files(session, include_gabaritos=args.include_gabaritos)
        print(f"Discovered {len(files)} PDF(s).")

        if len(files) < args.min_files:
            print(
                f"Expected at least {args.min_files} PDF(s), found {len(files)}.",
                file=sys.stderr,
            )
            return 2

        rows: list[tuple[ExamFile, Path]] = []
        for item in files:
            path = download_file(session, item, args.output, refresh=args.refresh)
            rows.append((item, path))

    write_manifest(rows, args.output)
    print(f"Archive ready at {args.output}/ with {len(rows)} PDF(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
