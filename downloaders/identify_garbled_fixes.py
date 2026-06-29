#!/usr/bin/env python3
"""
Identify which dataset .txt files had a garbled-text fix applied.

Background
----------
`run_antarctic_db_go_pipeline.py::fix_garbled_texts` finds .txt files whose
bytes look like a broken encoding, re-OCRs the matching PDF, and OVERWRITES the
.txt file on disk. It does *not* update the pipeline sqlite DB. As a result the
DB still holds the original (garbled) text while the on-disk file holds the
fixed text.

This script reconstructs, for every dataset .txt file, exactly the bytes that
`extract-documents` would have written from the DB (see
Submodules/antarctic-database-go/papers.go: SaveFullTextFiles / SaveOcrTextFiles)
and flags a file as "fixed" when:

  1. the DB-reconstructed text is garbled (same heuristic the pipeline uses), and
  2. the on-disk .txt now differs from that DB text (i.e. it was overwritten).

Both conditions together avoid false positives from files that were garbled in
the DB but never successfully re-OCR'd (no PDF / no API key), as well as from
incidental byte differences in files that were never garbled.

Usage
-----
    python downloaders/identify_garbled_fixes.py [--dataset-dir DIR] [--db FILE]
                                                 [--output FILE] [--verbose]

With no arguments it auto-selects the newest dataset-* directory and the
document-pipeline.sqlite3 in data/antarctic-db/processed/.
"""

import argparse
import sqlite3
import sys
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "antarctic-db" / "processed"


def is_garbled(raw: bytes) -> bool:
    """Mirror of run_antarctic_db_go_pipeline._is_garbled, operating on bytes.

    True when the first 500 bytes are >10% non-printable (broken encoding).
    Empty content is not considered garbled.
    """
    raw = raw[:500]
    if not raw.strip():
        return False
    non_printable = sum(1 for b in raw if b < 32 and b not in (9, 10, 13))
    return non_printable / len(raw) > 0.1


def url_to_txt_relpath(url: str) -> str | None:
    """Replicate papers.go: outputDir + TrimPrefix(parsed.Path, "/"), ext -> .txt.

    Returns the dataset-relative .txt path, or None for an unknown extension.
    """
    path = urlparse(url).path.lstrip("/")
    lower = path.lower()
    if lower.endswith(".pdf") or lower.endswith(".doc"):
        return path[:-4] + ".txt"
    if lower.endswith(".docx"):
        return path[:-5] + ".txt"
    return None


def collect_fulltext(conn: sqlite3.Connection) -> dict[str, bytes]:
    """relpath -> DB text bytes for modern (non-scanned) documents.

    Mirrors SaveFullTextFiles: latest full_text row per document id, written
    verbatim as the document text.
    """
    rows = conn.execute(
        """
        SELECT d.url, ft.document_text
        FROM documents d
        JOIN (
            SELECT id, MAX(timestamp) AS latest_ts
            FROM full_text
            GROUP BY id
        ) latest ON d.id = latest.id
        JOIN full_text ft ON ft.id = latest.id AND ft.timestamp = latest.latest_ts
        """
    ).fetchall()

    out: dict[str, bytes] = {}
    for url, text in rows:
        relpath = url_to_txt_relpath(_as_str(url))
        if relpath is None:
            continue
        out[relpath] = text if isinstance(text, bytes) else (text or "").encode("utf-8")
    return out


def collect_ocr(conn: sqlite3.Connection) -> dict[str, bytes]:
    """relpath -> DB text bytes for scanned documents.

    Mirrors SaveOcrTextFiles: for each document, concatenate the latest OCR
    page_text for pages 1..N, each followed by a newline.
    """
    # Latest OCR text per (id, page_nr).
    rows = conn.execute(
        """
        SELECT p.id, p.url, p.page_nr, o.page_text
        FROM pages p
        JOIN (
            SELECT id, page_nr, MAX(timestamp) AS latest_ts
            FROM ocr
            GROUP BY id, page_nr
        ) latest ON p.id = latest.id AND p.page_nr = latest.page_nr
        JOIN ocr o ON o.id = latest.id AND o.page_nr = latest.page_nr
                  AND o.timestamp = latest.latest_ts
        """
    ).fetchall()

    pages: dict[str, dict[int, bytes]] = {}
    urls: dict[str, str] = {}
    for doc_id, url, page_nr, page_text in rows:
        doc_id = _as_str(doc_id)
        urls[doc_id] = _as_str(url)
        text = page_text if isinstance(page_text, bytes) else (page_text or "").encode("utf-8")
        pages.setdefault(doc_id, {})[page_nr] = text

    # Authoritative page count per document (matches the second query in papers.go).
    counts = {
        _as_str(doc_id): n
        for doc_id, n in conn.execute(
            "SELECT id, COUNT(DISTINCT page_nr) FROM pages GROUP BY id"
        ).fetchall()
    }

    out: dict[str, bytes] = {}
    for doc_id, page_map in pages.items():
        relpath = url_to_txt_relpath(urls[doc_id])
        if relpath is None:
            continue
        n = counts.get(doc_id, max(page_map))
        text = b"".join(page_map.get(i, b"") + b"\n" for i in range(1, n + 1))
        out[relpath] = text
    return out


def _as_str(v) -> str:
    return v.decode("utf-8", "replace") if isinstance(v, bytes) else v


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path,
                        help="Dataset directory (default: newest dataset-* in processed/)")
    parser.add_argument("--db", type=Path,
                        default=PROCESSED_DIR / "document-pipeline.sqlite3",
                        help="Pipeline sqlite DB (default: processed/document-pipeline.sqlite3)")
    parser.add_argument("--output", type=Path,
                        help="Write the list of fixed files to this file (one path per line)")
    parser.add_argument("--verbose", action="store_true",
                        help="Also report garbled DB files that were NOT fixed")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    if dataset_dir is None:
        candidates = sorted(PROCESSED_DIR.glob("dataset-*"))
        if not candidates:
            sys.exit(f"No dataset-* directory found in {PROCESSED_DIR}")
        dataset_dir = candidates[-1]
    if not dataset_dir.is_dir():
        sys.exit(f"Dataset directory not found: {dataset_dir}")
    if not args.db.exists():
        sys.exit(f"DB not found: {args.db}")

    print(f"Dataset: {dataset_dir}", file=sys.stderr)
    print(f"DB:      {args.db}", file=sys.stderr)

    # text_factory=bytes => raw bytes, byte-for-byte comparable to the on-disk files.
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.text_factory = bytes
    try:
        db_text = collect_fulltext(conn)
        db_text.update(collect_ocr(conn))
    finally:
        conn.close()

    fixed: list[str] = []
    garbled_not_fixed: list[str] = []
    missing_on_disk = 0

    for relpath, db_bytes in db_text.items():
        if not is_garbled(db_bytes):
            continue
        disk_path = dataset_dir / relpath
        if not disk_path.exists():
            missing_on_disk += 1
            continue
        disk_bytes = disk_path.read_bytes()
        if disk_bytes != db_bytes:
            fixed.append(relpath)
        else:
            garbled_not_fixed.append(relpath)

    fixed.sort()
    garbled_not_fixed.sort()

    for relpath in fixed:
        print(relpath)

    print("", file=sys.stderr)
    print(f"DB documents inspected:        {len(db_text)}", file=sys.stderr)
    print(f"Garbled in DB AND fixed:       {len(fixed)}", file=sys.stderr)
    print(f"Garbled in DB, still garbled:  {len(garbled_not_fixed)}", file=sys.stderr)
    if missing_on_disk:
        print(f"Garbled in DB, no disk file:   {missing_on_disk}", file=sys.stderr)

    if args.verbose and garbled_not_fixed:
        print("\nGarbled in DB but NOT fixed (no successful re-OCR):", file=sys.stderr)
        for relpath in garbled_not_fixed:
            print(f"  {relpath}", file=sys.stderr)

    if args.output:
        args.output.write_text("\n".join(fixed) + ("\n" if fixed else ""), encoding="utf-8")
        print(f"\nWrote {len(fixed)} path(s) to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
