# Here we generate and cache document embeddings.
import json
import pathlib
import sqlite3

import openai
import secret_management

EMBEDDINGS_DB_PATH = pathlib.Path("data/document_embeddings.sqlite3")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(EMBEDDINGS_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            document_uuid   TEXT    NOT NULL,
            segment_number  INTEGER NOT NULL,
            model_uuid      TEXT    NOT NULL,
            document_type   TEXT    NOT NULL,
            embedding       TEXT    NOT NULL,
            PRIMARY KEY (document_uuid, segment_number, model_uuid)
        )
    """)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.commit()
    return conn


def has_embedding(document_uuid: str, segment_number: int, model_uuid: str) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM embeddings WHERE document_uuid=? AND segment_number=? AND model_uuid=?",
            (document_uuid, segment_number, model_uuid),
        ).fetchone()
    return row is not None


def get_embedding(document_uuid: str, segment_number: int, model_uuid: str) -> list[float] | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT embedding FROM embeddings WHERE document_uuid=? AND segment_number=? AND model_uuid=?",
            (document_uuid, segment_number, model_uuid),
        ).fetchone()
    return json.loads(row[0]) if row else None


def generate_embedding(document_uuid: str, segment_number: int, model_uuid: str, document_type: str, text: str) -> list[float]:
    client = openai.OpenAI(
        api_key=secret_management.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    response = client.embeddings.create(input=text, model=model_uuid)
    vector = response.data[0].embedding

    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO embeddings (document_uuid, segment_number, model_uuid, document_type, embedding) VALUES (?, ?, ?, ?, ?)",
            (document_uuid, segment_number, model_uuid, document_type, json.dumps(vector)),
        )

    return vector
