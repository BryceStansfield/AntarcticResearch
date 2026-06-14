import sqlite3
import sys

DB_PATH = "data/antarctic-db/processed/document-pipeline.sqlite3"

db = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)

rows = db.execute("""
    SELECT
        d.id,
        d.url,
        ft.timestamp
    FROM documents d
    JOIN (
        SELECT id, MAX(timestamp) AS latest_ts
        FROM full_text
        GROUP BY id
    ) latest ON d.id = latest.id
    JOIN full_text ft ON ft.id = latest.id AND ft.timestamp = latest.latest_ts
    ORDER BY d.id, ft.timestamp DESC
""").fetchall()

seen = {}
repeated = {}

for doc_id, url, ts in rows:
    if doc_id in seen:
        repeated.setdefault(doc_id, [seen[doc_id]]).append((url, ts))
    else:
        seen[doc_id] = (url, ts)

if not repeated:
    print("No repeated IDs found.")
    sys.exit(0)

print(f"Found {len(repeated)} repeated ID(s):\n")
for doc_id, occurrences in repeated.items():
    print(f"ID: {doc_id}")
    for url, ts in occurrences:
        print(f"  url={url}  ts={ts}")
    print()
