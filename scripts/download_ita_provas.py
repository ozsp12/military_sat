#!/usr/bin/env python3
"""Download official ITA entrance-exam PDFs and organize them by year."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

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


def write_manifest(rows: list[tuple[ExamFile, Path]], output_root: Path) -> None:
    manifest = output_root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["year", "label", "path", "bytes", "sha256", "source_url"])
        for item, path in rows:
            writer.writerow(
                [
                    item.year,
                    item.label,
                    path.as_posix(),
                    path.stat().st_size,
                    sha256_file(path),
                    item.url,
                ]
            )


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
