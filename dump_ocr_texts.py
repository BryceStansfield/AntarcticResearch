#!/usr/bin/env python3
"""Dump all available OCR'd documents to data/test/ as plain text files."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "data/antarctic-db/processed/document-pipeline.sqlite3"
OUT_DIR = Path(__file__).parent / "data/test"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(DB_PATH)
    rows = con.execute("""
        SELECT o.id, GROUP_CONCAT(o.page_text, char(10))
        FROM (
            SELECT o.id, o.page_nr, o.page_text
            FROM ocr o
            JOIN (
                SELECT id, page_nr, MAX(timestamp) AS latest_ts
                FROM ocr
                GROUP BY id, page_nr
            ) latest ON latest.id = o.id
                     AND latest.page_nr = o.page_nr
                     AND o.timestamp = latest.latest_ts
            ORDER BY o.id, o.page_nr
        ) o
        GROUP BY o.id
    """).fetchall()
    con.close()

    for doc_id, text in rows:
        (OUT_DIR / f"{doc_id}.txt").write_text(text or "", encoding="utf-8")

    print(f"Wrote {len(rows)} files to {OUT_DIR}/")


if __name__ == "__main__":
    main()
