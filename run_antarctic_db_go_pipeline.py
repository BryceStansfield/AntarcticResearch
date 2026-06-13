#!/usr/bin/env python3
"""
Run the antarctic-database-go pipeline from the AntarcticResearch top-level directory.

Pipeline steps:
  1. Build Go binaries
  2. Start the PyMuPDF microservice (port 11000) — required by steps 3 and 5
  3. prepare-document-pipeline  — download & analyse documents, split scanned PDFs
  4. run-ocr                    — OCR scanned pages (needs NVIDIA_API_KEY or ANTHROPIC_API_KEY)
  5. run-fulltext               — extract text from non-scanned documents
  6. assemble-ocr-fulltext      — concatenate per-page OCR results into full_text for scanned docs
  7. extract-documents          — assemble the final dataset

API keys are read from secrets.json at the repo root (gitignored).
They can also be set as environment variables, which take precedence.

Outputs go to:  data/antarctic-db/
  processed/    SQLite databases, Parquet files, logs, dataset directories
  external/     UTAS PDFs (place manually downloaded PDFs in external/utas/)
  raw/          source CSVs (wps_missing.csv lives in the submodule)

A .pipeline_complete sentinel is written after a successful run.
Pass --force to rerun even if the sentinel exists.
"""

import argparse
import csv
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

import pymupdf
import requests

REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_ROOT / "Submodules" / "antarctic-database-go"

# All generated data lands here instead of inside the submodule
DATA_DIR = REPO_ROOT / "data" / "antarctic-db"
PROCESSED_DIR = DATA_DIR / "processed"
UTAS_DIR = DATA_DIR / "external" / "utas"

# Source files that live in the submodule (not generated, not moved)
WPS_CSV = PROJECT_ROOT / "data" / "raw" / "wps_missing.csv"

SENTINEL = PROCESSED_DIR / ".pipeline_complete"
UTAS_SENTINEL = UTAS_DIR / ".downloads_complete"
SECRETS_FILE = REPO_ROOT / "secrets.json"
MICROSERVICE_URL = "http://localhost:11000"

# This file was corrupted in the original UTAS archive (pdfseparate failed at
# page 28). We re-save it with pymupdf after downloading, which replicates what
# macOS Preview's "Create PDF/A" export does: rewrites the file structure from
# scratch, fixing cross-reference and stream corruption.
CORRUPTED_FILE = "b5f7aa95-1d44-490a-a073-2fb7cd564ba0-AU-ATADD-3-BB-AQ-311.pdf"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_secrets() -> None:
    """Load API keys from secrets.json into the environment (env vars win)."""
    if not SECRETS_FILE.exists():
        return
    with open(SECRETS_FILE) as f:
        secrets = json.load(f)
    for key, value in secrets.items():
        if value and not os.environ.get(key):
            os.environ[key] = value


def get_utas_urls() -> list[str]:
    """Return deduplicated list of UTAS PDF URLs from wps_missing.csv."""
    urls: set[str] = set()
    with open(WPS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            url = row.get("url", "").strip()
            if url.startswith("http"):
                urls.add(url)
    return sorted(urls)


def download_with_agreement(session: requests.Session, url: str, dest: Path) -> None:
    """
    Download a UTAS PDF that sits behind a research-use terms page.

    The site returns an HTML page with a hidden token on first visit.
    Resubmitting the same URL with ?token=<value> (using the same session
    cookie) delivers the actual file.
    """
    r1 = session.get(url, timeout=30)
    r1.raise_for_status()

    m = re.search(r'name="token"\s+value="([^"]+)"', r1.text)
    if not m:
        raise RuntimeError(f"No agreement token found on landing page for {url}")

    r2 = session.get(url, params={"token": m.group(1)}, timeout=60)
    r2.raise_for_status()

    if "application/pdf" not in r2.headers.get("Content-Type", ""):
        raise RuntimeError(
            f"Expected PDF after agreeing to terms, got {r2.headers.get('Content-Type')} for {url}"
        )

    dest.write_bytes(r2.content)


def download_utas_pdfs() -> None:
    """Download UTAS PDFs listed in wps_missing.csv, then re-save the known-corrupted file."""
    if UTAS_SENTINEL.exists():
        print(f"UTAS PDFs already downloaded ({UTAS_SENTINEL.relative_to(REPO_ROOT)}).")
        return

    urls = get_utas_urls()
    print(f"==> Downloading {len(urls)} UTAS PDFs ...")

    session = requests.Session()
    for url in urls:
        filename = url.split("/")[-1]
        dest = UTAS_DIR / filename
        if dest.exists():
            print(f"    skip (exists): {filename}")
            continue
        print(f"    {filename}")
        download_with_agreement(session, url, dest)

    # Re-save the known-corrupted file by importing its pages into a fresh
    # pymupdf document. Working page-by-page bypasses the broken cross-reference
    # table and object dictionary that block a direct save(), replicating what
    # macOS Preview's "Create PDF/A" export does.
    corrupted = UTAS_DIR / CORRUPTED_FILE
    if corrupted.exists():
        print(f"==> Re-saving corrupted file via pymupdf: {CORRUPTED_FILE}")
        tmp = corrupted.with_suffix(".tmp.pdf")
        src = pymupdf.open(str(corrupted))
        out = pymupdf.open()
        out.insert_pdf(src)
        out.save(str(tmp), deflate=True, garbage=4)
        out.close()
        src.close()
        tmp.replace(corrupted)
        print("    Done.")

    UTAS_SENTINEL.write_text(
        f"Downloaded: {datetime.datetime.now(datetime.timezone.utc).isoformat()}Z\n"
        f"Files: {len(urls)}\n"
    )
    print("    UTAS downloads complete.")


def datestamp_hash() -> str:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d_%H%M%SZ")
    return f"{stamp}-{uuid.uuid4()}"


def patch_utas_metadata(parquet_file: Path) -> None:
    """
    Fill in the empty metadata fields for UTAS records in document-summary.parquet.

    ingestManualDocuments() in the upstream Go code only sets PaperUrl on these
    records. The remaining fields (meeting year, paper type, parties, etc.) are
    all present in wps_missing.csv and patched in here after the fact.

    Note: the Go struct tag for PaperType is 'party_type' (upstream typo) — we
    match that name exactly so the column is consistent with the rest of the file.
    """
    import pandas as pd

    lookup: dict[str, dict] = {}
    with open(WPS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            url = row.get("url", "").strip()
            if not url.startswith("http"):
                continue
            lookup[url] = {
                "meeting_year":   int(row["meeting_year"]) if row["meeting_year"].strip() else 0,
                "meeting_type":   row["meeting_type"].strip(),
                "meeting_number": int(row["meeting_number"]) if row["meeting_number"].strip() else 0,
                "meeting_name":   row["meeting_name"].strip(),
                "party_type":     row["paper_type"].strip(),
                "paper_name":     row["paper_name"].strip(),
                "paper_number":   int(row["paper_number"]) if row["paper_number"].strip() else 0,
                "paper_revision": int(row["paper_revision"]) if row["paper_revision"].strip() else 0,
                "agendas":        [a.strip() for a in row["agendas"].split(",") if a.strip()],
                "parties":        [p.strip() for p in row["parties"].split(",") if p.strip()],
            }

    df = pd.read_parquet(parquet_file)
    patched = 0
    for i, row in df.iterrows():
        meta = lookup.get(row.get("paper_url", ""))
        if meta is None:
            continue
        for col, val in meta.items():
            df.at[i, col] = val
        patched += 1

    df.to_parquet(parquet_file, index=False)
    print(f"    Patched {patched} UTAS records in {parquet_file.name}.")


def clean_previous_run() -> None:
    print("==> Cleaning previous run artifacts ...")
    targets = [
        PROCESSED_DIR / "document-pipeline.sqlite3",
        PROCESSED_DIR / "document-pipeline.sqlite3-shm",
        PROCESSED_DIR / "document-pipeline.sqlite3-wal",
        PROCESSED_DIR / "document-summary.parquet",
        SENTINEL,
    ]
    for path in targets:
        if path.exists():
            path.unlink()
            print(f"    Removed {path.relative_to(REPO_ROOT)}")
    for log in PROCESSED_DIR.glob("*.log"):
        log.unlink()
        print(f"    Removed {log.relative_to(REPO_ROOT)}")
    for dataset_dir in PROCESSED_DIR.glob("dataset-*"):
        shutil.rmtree(dataset_dir)
        print(f"    Removed {dataset_dir.relative_to(REPO_ROOT)}/")
    print("    Done. (http-cache.sqlite3 preserved)")


def build_binaries() -> None:
    print("==> Building Go binaries ...")
    binaries = [
        ("prepare-document-pipeline", "./cmd/prepare-document-pipeline"),
        ("run-ocr",                   "./cmd/run-ocr"),
        ("run-fulltext",              "./cmd/run-fulltext"),
        ("extract-documents",         "./cmd/extract-documents"),
    ]
    for name, pkg in binaries:
        print(f"    go build -o {name} {pkg}")
        subprocess.run(["go", "build", "-o", name, pkg], cwd=PROJECT_ROOT, check=True)
    print("    Done.")


def wait_for_microservice(timeout_s: int = 30) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(MICROSERVICE_URL + "/analyze", timeout=1)
            return
        except urllib.error.HTTPError:
            return  # Flask is up (endpoint is POST-only, so GET → 405)
        except Exception:
            time.sleep(1)
    print(
        f"WARNING: microservice did not respond within {timeout_s}s; continuing anyway.",
        file=sys.stderr,
    )


def start_microservice() -> subprocess.Popen:
    print("==> Starting PyMuPDF microservice on port 11000 ...")
    proc = subprocess.Popen(
        ["uv", "run", "main.py"],
        cwd=PROJECT_ROOT / "pymupdf-microservice",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    wait_for_microservice()
    print("    Microservice ready.")
    return proc


def assemble_ocr_fulltext() -> None:
    """Check all scanned pages are ocr-done, then write full_text rows from the ocr table."""
    import sqlite3

    db_path = PROCESSED_DIR / "document-pipeline.sqlite3"
    print("==> assemble-ocr-fulltext")

    con = sqlite3.connect(db_path)
    try:
        not_done = con.execute("""
            SELECT COUNT(*) FROM pages p
            JOIN documents d ON d.id = p.id
            WHERE d.is_scanned = 1 AND p.status != 'ocr-done'
        """).fetchone()[0]

        if not_done > 0:
            print(
                f"    FAILED: {not_done} scanned page(s) are not yet ocr-done. "
                "Run run-ocr to completion first.",
                file=sys.stderr,
            )
            sys.exit(1)

        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S.000+00:00"
        )

        # GROUP_CONCAT processes rows in subquery order, so ORDER BY page_nr gives
        # correct page ordering even though GROUP_CONCAT has no ORDER BY clause.
        rows = con.execute("""
            SELECT o.id, o.method, GROUP_CONCAT(o.page_text, char(10))
            FROM (
                SELECT o.id, o.method, o.page_nr, o.page_text
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
            GROUP BY o.id, o.method
        """).fetchall()

        for doc_id, method, doc_text in rows:
            con.execute(
                "INSERT INTO full_text (id, document_text, method, timestamp) VALUES (?, ?, ?, ?)",
                (doc_id, doc_text, method, timestamp),
            )

        con.commit()
        print(f"    Assembled full_text for {len(rows)} scanned document(s).")
    finally:
        con.close()


def run_step(name: str, cmd: list[str], log_file: Path) -> None:
    print(f"==> {name}")
    print(f"    Log: {log_file.relative_to(REPO_ROOT)}")
    with open(log_file, "w") as log:
        result = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=os.environ,
            cwd=PROJECT_ROOT,
        )
    if result.returncode != 0:
        print(
            f"    FAILED (exit {result.returncode}). See {log_file}",
            file=sys.stderr,
        )
        sys.exit(result.returncode)
    print("    Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the antarctic-database-go pipeline."
    )
    parser.add_argument(
        "--ocr-service",
        default="anthropic",
        choices=["nvidia", "anthropic"],
        help="OCR service to use for scanned documents (default: anthropic)",
    )
    parser.add_argument(
        "--ocr-batch-size",
        type=int,
        default=10,
        help="Number of pages per OCR batch (default: 10)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Process only a small subset of documents (for testing)",
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Skip the run-ocr step (e.g. if no scanned documents exist)",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip the prepare-document-pipeline step and UTAS metadata patch (resume after a mid-OCR crash)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun even if a previous successful run's sentinel exists",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove pipeline DB, parquet, logs, and dataset dirs before running (preserves http-cache.sqlite3 and UTAS PDFs)",
    )
    args = parser.parse_args()

    # Load API keys from secrets.json before any checks
    load_secrets()

    if args.clean:
        clean_previous_run()

    # Ensure all output directories exist
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    UTAS_DIR.mkdir(parents=True, exist_ok=True)

    # Download UTAS PDFs before anything else — the pipeline fails immediately
    # without them, and downloads are independent of the pipeline sentinel.
    download_utas_pdfs()

    # Skip if already completed
    if SENTINEL.exists() and not args.force:
        print(
            f"Pipeline already completed (sentinel: {SENTINEL.relative_to(REPO_ROOT)}).\n"
            f"Pass --force to rerun."
        )
        return

    # Warn early if the required API key is missing
    if not args.skip_ocr:
        key_name = "NVIDIA_API_KEY" if args.ocr_service == "nvidia" else "ANTHROPIC_API_KEY"
        if not os.environ.get(key_name):
            print(
                f"WARNING: {key_name} is not set in the environment or secrets.json.\n"
                f"         The OCR step will fail. Pass --skip-ocr to skip it.",
                file=sys.stderr,
            )

    # Step 1: build
    build_binaries()

    # Step 2: microservice (kept alive until the pipeline finishes)
    microservice = start_microservice()

    try:
        tag = datestamp_hash()

        # Step 3: prepare-document-pipeline
        if args.skip_prepare:
            print("==> Skipping prepare-document-pipeline and UTAS metadata patch (--skip-prepare).")
        else:
            prepare_cmd = [
                str(PROJECT_ROOT / "prepare-document-pipeline"),
                "--http-cache",           str(PROCESSED_DIR / "http-cache.sqlite3"),
                "--new-pipeline-db-file", str(PROCESSED_DIR / "document-pipeline.sqlite3"),
                "--document-summary",     str(PROCESSED_DIR / "document-summary.parquet"),
                "--utas-raw-pdfs",        str(UTAS_DIR),
                "--wps-csv",              str(WPS_CSV),
            ]
            if args.quick:
                prepare_cmd.append("--quick")

            run_step(
                "prepare-document-pipeline",
                prepare_cmd,
                PROCESSED_DIR / f"prepare-document-pipeline-{tag}.log",
            )

            print("==> Patching UTAS metadata in document-summary.parquet ...")
            patch_utas_metadata(PROCESSED_DIR / "document-summary.parquet")

        # Step 4: run-ocr
        if not args.skip_ocr:
            run_step(
                f"run-ocr  (service={args.ocr_service})",
                [
                    str(PROJECT_ROOT / "run-ocr"),
                    "--pipeline-db-file", str(PROCESSED_DIR / "document-pipeline.sqlite3"),
                    "--use-asset-upload=false",
                    "--service", args.ocr_service,
                    "--batch-size", str(args.ocr_batch_size),
                ],
                PROCESSED_DIR / f"ocr-pipeline-{tag}.log",
            )

        # Step 5: run-fulltext
        run_step(
            "run-fulltext",
            [
                str(PROJECT_ROOT / "run-fulltext"),
                "--pipeline-db-file", str(PROCESSED_DIR / "document-pipeline.sqlite3"),
            ],
            PROCESSED_DIR / f"fulltext-pipeline-{tag}.log",
        )

        # Step 6: assemble per-page OCR into full_text for scanned documents
        assemble_ocr_fulltext()

        # Step 7: extract-documents
        dataset_dir = PROCESSED_DIR / f"dataset-{tag}"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        run_step(
            "extract-documents",
            [
                str(PROJECT_ROOT / "extract-documents"),
                "--http-cache",          str(PROCESSED_DIR / "http-cache.sqlite3"),
                "--pipeline-db-file",    str(PROCESSED_DIR / "document-pipeline.sqlite3"),
                "--output-dir",          str(dataset_dir),
                "--output-parquet-file", str(dataset_dir / "summary.parquet"),
                "--utas-raw-pdfs",       str(UTAS_DIR),
                "--wps-csv",             str(WPS_CSV),
            ],
            PROCESSED_DIR / f"extract-documents-{tag}.log",
        )

        SENTINEL.write_text(
            f"Completed: {datetime.datetime.now(datetime.timezone.utc).isoformat()}Z\n"
            f"Dataset:   {dataset_dir.relative_to(REPO_ROOT)}\n"
        )

        print(f"\nPipeline complete.")
        print(f"Dataset:  {dataset_dir.relative_to(REPO_ROOT)}")
        print(f"Sentinel: {SENTINEL.relative_to(REPO_ROOT)}")

    finally:
        print("==> Stopping PyMuPDF microservice ...")
        microservice.terminate()
        microservice.wait()


if __name__ == "__main__":
    main()
