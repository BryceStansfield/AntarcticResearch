#!/usr/bin/env python3
"""Check OCR pipeline progress in document-pipeline.sqlite3."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "data/antarctic-db/processed/document-pipeline.sqlite3"


def main():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    total_docs = cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    scanned_docs = cur.execute("SELECT COUNT(*) FROM documents WHERE is_scanned = 1").fetchone()[0]
    pages_extracted_docs = cur.execute(
        "SELECT COUNT(*) FROM documents WHERE status = 'pages_extracted'"
    ).fetchone()[0]

    total_pages = cur.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
    ocr_pages = cur.execute("SELECT COUNT(DISTINCT id || '|' || page_nr) FROM ocr").fetchone()[0]

    docs_with_ocr = cur.execute("SELECT COUNT(DISTINCT id) FROM ocr").fetchone()[0]
    full_text_docs = cur.execute("SELECT COUNT(DISTINCT id) FROM full_text").fetchone()[0]

    con.close()

    print("=== OCR Pipeline Progress ===\n")

    print("Documents")
    print(f"  Total:           {total_docs:>6,}")
    print(f"  Scanned (need OCR): {scanned_docs:>4,}  ({scanned_docs/total_docs*100:.1f}% of all docs)")
    print(f"  Pages extracted: {pages_extracted_docs:>6,}")
    print()

    print("Pages")
    print(f"  Total extracted: {total_pages:>6,}")
    print(f"  OCR'd:           {ocr_pages:>6,}  ({ocr_pages/total_pages*100:.1f}% of extracted pages)")
    print()

    print("Documents with any OCR")
    print(f"  {docs_with_ocr:,} / {scanned_docs:,} scanned docs  ({docs_with_ocr/scanned_docs*100:.1f}%)")
    print()

    print("Full text assembled")
    print(f"  {full_text_docs:,} documents")


if __name__ == "__main__":
    main()
