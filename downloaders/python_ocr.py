#!/usr/bin/env python3
"""
OCR a PDF using Anthropic's Claude vision API, with NVIDIA as fallback.
Results are cached by PDF content hash in data/python_ocr_cache.sqlite3.

Usage:
    python downloaders/python_ocr.py path/to/document.pdf
"""

import argparse
import base64
import datetime
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import pymupdf
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DB = REPO_ROOT / "data" / "python_ocr_cache.sqlite3"

ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
NVIDIA_MODEL = "meta/llama-4-maverick-17b-128e-instruct"
MAX_TOKENS = 8 * 4096
MAX_B64_BYTES = 5 * 1024 * 1024

ANTHROPIC_PROMPT = (
    "Extract all text from this image (which is a PDF page). "
    "Return only the extracted text, with no additional comments or explanations. "
    "Preserve the exact formatting, paragraph structure, and layout as much as possible."
)

NVIDIA_PROMPT = (
    "You are a precise OCR system. Your only task is to extract text from this image with exact fidelity.\n"
    "Instructions:\n"
    "- Extract ALL text from the image with perfect accuracy\n"
    "- Maintain exact spacing and line breaks as they appear\n"
    "- If you can't read a character with certainty, represent it with [?]\n"
    "- If text is arranged in columns, preserve the column structure\n"
    "- Preserve any bullets, numbering, or indentation\n"
    "- For tables, use plain text formatting with spaces to align columns\n"
    "- Do not add ANY explanatory text, headers, or comments\n"
    "- Do not describe the image or its content\n"
    "- Return ONLY the extracted text"
)


def _open_cache() -> sqlite3.Connection:
    CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(CACHE_DB)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS ocr_cache (
            pdf_hash   TEXT PRIMARY KEY,
            text       TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    con.commit()
    return con


def pdf_hash(pdf_path: Path) -> str:
    return hashlib.sha256(pdf_path.read_bytes()).hexdigest()


def _render_page_png(page: pymupdf.Page, dpi: int) -> bytes:
    mat = pymupdf.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=pymupdf.csRGB)
    return pix.tobytes("png")


def _page_to_base64_png(page: pymupdf.Page) -> str:
    """Render page to PNG, downsampling if needed to stay under the 5 MB base64 limit."""
    for dpi in (300, 150, 100):
        png = _render_page_png(page, dpi)
        b64 = base64.standard_b64encode(png).decode()
        if len(b64.encode()) < MAX_B64_BYTES:
            return b64
    return base64.standard_b64encode(_render_page_png(page, 100)).decode()


def _ocr_page_anthropic(api_key: str, b64_png: str) -> str:
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        json={
            "model": ANTHROPIC_MODEL,
            "max_tokens": MAX_TOKENS,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64_png,
                            },
                        },
                        {"type": "text", "text": ANTHROPIC_PROMPT},
                    ],
                }
            ],
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return "".join(c["text"] for c in data["content"] if c["type"] == "text")


def _ocr_page_nvidia(api_key: str, b64_png: str) -> str:
    resp = requests.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept": "text/event-stream",
        },
        json={
            "model": NVIDIA_MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_png}"}},
                {"type": "text", "text": NVIDIA_PROMPT},
            ]}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": True,
        },
        timeout=120,
        stream=True,
    )
    resp.raise_for_status()

    text = ""
    for line in resp.iter_lines(decode_unicode=True):
        if not line.startswith("data: {"):
            continue
        chunk = json.loads(line[len("data: "):])
        if chunk.get("choices"):
            text += chunk["choices"][0].get("delta", {}).get("content", "")
    return text


def ocr_pdf(pdf_path: Path, anthropic_key: str, nvidia_key: str = "") -> str:
    """OCR all pages of a PDF via Anthropic (NVIDIA as fallback). Cached by content hash."""
    h = pdf_hash(pdf_path)
    con = _open_cache()

    row = con.execute("SELECT text FROM ocr_cache WHERE pdf_hash = ?", (h,)).fetchone()
    if row:
        print(f"  Cache hit: {h[:12]}...", file=sys.stderr)
        con.close()
        return row[0]

    doc = pymupdf.open(str(pdf_path))
    pages_text = []
    for i, page in enumerate(doc):
        print(f"  OCR page {i + 1}/{len(doc)} ...", file=sys.stderr)
        b64 = _page_to_base64_png(page)
        print(f"  Page {i + 1} image size: {len(b64) / 1024:.0f} KB (base64)", file=sys.stderr)
        text = None
        if anthropic_key:
            try:
                text = _ocr_page_anthropic(anthropic_key, b64)
            except Exception as e:
                body = getattr(getattr(e, "response", None), "text", "")
                detail = f"\n    API response: {body}" if body else ""
                print(f"  Anthropic failed (page {i + 1}): {e}{detail}", file=sys.stderr)
        if text is None and nvidia_key:
            print(f"  Falling back to NVIDIA (page {i + 1}) ...", file=sys.stderr)
            try:
                text = _ocr_page_nvidia(nvidia_key, b64)
            except Exception as e:
                body = getattr(getattr(e, "response", None), "text", "")
                detail = f"\n    API response: {body}" if body else ""
                print(f"  NVIDIA failed (page {i + 1}): {e}{detail}", file=sys.stderr)
        if text is None:
            raise RuntimeError(f"No OCR service available for page {i + 1} of {pdf_path}")
        pages_text.append(text)
        if i < len(doc) - 1:
            time.sleep(0.5)
    doc.close()

    result = "\n\n".join(pages_text)
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    con.execute(
        "INSERT INTO ocr_cache (pdf_hash, text, created_at) VALUES (?, ?, ?)",
        (h, result, created_at),
    )
    con.commit()
    con.close()
    return result


def load_api_key(env_var: str) -> str:
    key = os.environ.get(env_var, "")
    if not key:
        secrets_file = REPO_ROOT / "secrets.json"
        if secrets_file.exists():
            secrets = json.loads(secrets_file.read_text())
            key = secrets.get(env_var, "")
    return key


def load_api_keys() -> tuple[str, str]:
    return load_api_key("ANTHROPIC_API_KEY"), load_api_key("NVIDIA_API_KEY")


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR a PDF via Anthropic Claude vision (NVIDIA fallback).")
    parser.add_argument("pdf_path", type=Path, help="Path to PDF file")
    args = parser.parse_args()

    if not args.pdf_path.exists():
        print(f"Error: {args.pdf_path} does not exist", file=sys.stderr)
        sys.exit(1)

    anthropic_key, nvidia_key = load_api_keys()
    if not anthropic_key and not nvidia_key:
        print("Error: neither ANTHROPIC_API_KEY nor NVIDIA_API_KEY is set", file=sys.stderr)
        sys.exit(1)

    print(ocr_pdf(args.pdf_path, anthropic_key, nvidia_key))


if __name__ == "__main__":
    main()
