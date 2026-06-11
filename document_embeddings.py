# Here we generate and cache document embeddings.
import array
import pathlib
import sqlite3

import openai
import secret_management
import pandas
import multiprocessing

EMBEDDINGS_DB_PATH = pathlib.Path("data/document_embeddings.sqlite3")
DEFAULT_EMBEDDING_MODEL = "qwen/qwen3-embedding-4b"

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(EMBEDDINGS_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            document_uuid   TEXT    NOT NULL,
            segment_number  INTEGER NOT NULL,
            model_uuid      TEXT    NOT NULL,
            document_type   TEXT    NOT NULL,
            embedding       BLOB    NOT NULL,
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
    if not row:
        return None
    data = row[0]
    return array.array('f', data).tolist()


def get_or_generate_embedding(document_uuid: str, segment_number: int, document_type: str, text: str, model_uuid: str = DEFAULT_EMBEDDING_MODEL) -> list[float]:
    cached = get_embedding(document_uuid, segment_number, model_uuid)
    if cached is not None:
        return cached
    return generate_embedding(document_uuid, segment_number, document_type, text, model_uuid)


def generate_embedding(document_uuid: str, segment_number: int, document_type: str, text: str, model_uuid: str = DEFAULT_EMBEDDING_MODEL) -> list[float]:
    client = openai.OpenAI(
        api_key=secret_management.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    response = client.embeddings.create(input=text, model=model_uuid)
    vector = response.data[0].embedding

    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO embeddings (document_uuid, segment_number, model_uuid, document_type, embedding) VALUES (?, ?, ?, ?, ?)",
            (document_uuid, segment_number, model_uuid, document_type, array.array('f', vector).tobytes()),
        )

    return vector

def embed_all_measures():
    pd = pandas.read_csv("data/MeasureCorpusEnriched.csv")

    to_embed = []

    for row in pd.itertuples():
        if pandas.isna(row.Content):
            continue
        
        doc_num = row.Document_Number
        text_rep = f"Subject: {row.Subject}\n{row.Content}"

        to_embed.append((f"MEASURE__{doc_num}", 1, "measure", text_rep,))

    with multiprocessing.Pool(processes=20) as pool:
        # map blocks execution until all workers finish returning data
        pool.starmap(generate_embedding, to_embed)

if __name__ == "__main__":
    embed_all_measures()