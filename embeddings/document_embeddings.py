# Here we generate and cache document embeddings.
import array
import json
import pathlib
import sqlite3

import openai
import secret_management
import pandas
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer
import hashlib
import math
import downloaders.map_all_wp_ip_locations

EMBEDDINGS_DB_PATH = pathlib.Path("data/document_embeddings.sqlite3")
DEFAULT_EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
HUGGINGFACE_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
CONTEXT_WINDOW_LIMIT = 32000

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(EMBEDDINGS_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            document_uuid   TEXT    NOT NULL,
            model_uuid      TEXT    NOT NULL,
            document_type   TEXT    NOT NULL,
            embedding       BLOB    NOT NULL,
            PRIMARY KEY (document_uuid, model_uuid)
        )
    """)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.commit()
    return conn


# --------------------------------------------------------------------------- document types
#
# ``document_type`` holds a JSON array of type strings, not one string. A document's uuid is the
# sha256 of its text, so two pipelines that produce byte-identical text share a single row -- a
# working paper whose censored form is unchanged is legitimately both "WorkingPaper" and
# "CensoredWorkingPaperV1". Storing one type let whichever pipeline embedded the text first claim
# the row, and every other type's enumeration silently missed it. Types therefore accumulate.

# A stored cell is either the JSON array or, in a not-yet-migrated database, a bare scalar string.
# This normalises both so json_each can read either, which keeps the module working mid-migration.
_TYPES_AS_JSON = (
    "CASE WHEN document_type LIKE '[%' THEN document_type ELSE json_array(document_type) END"
)

# Add one type to a row's array, in a single statement so concurrent writers (the embedding pool
# runs 200 processes) can't lose a type to a read-modify-write race. UNION de-duplicates and
# ORDER BY value keeps the encoding canonical, so equal type sets always compare equal.
_MERGE_TYPE_SQL = f"""
UPDATE embeddings
   SET document_type = (SELECT json_group_array(value)
                          FROM (SELECT value FROM json_each({_TYPES_AS_JSON})
                                UNION SELECT ?
                                ORDER BY value))
 WHERE document_uuid=? AND model_uuid=?
"""


def parse_document_types(cell) -> list[str]:
    """The list of types stored in a ``document_type`` cell, tolerating the pre-migration scalar."""
    if not cell:
        return []
    if isinstance(cell, str) and cell.startswith("["):
        try:
            return [str(t) for t in json.loads(cell)]
        except json.JSONDecodeError:
            return [cell]
    return [str(cell)]


def get_document_types(document_uuid: str, model_uuid: str = DEFAULT_EMBEDDING_MODEL) -> list[str]:
    """Every type recorded against this document, or [] if it has no embedding."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT document_type FROM embeddings WHERE document_uuid=? AND model_uuid=?",
            (document_uuid, model_uuid),
        ).fetchone()
    return parse_document_types(row[0]) if row else []


def add_document_type(document_uuid: str, document_type: str,
                      model_uuid: str = DEFAULT_EMBEDDING_MODEL) -> bool:
    """Record that ``document_type`` also applies to an already-embedded document.

    Returns True if the row exists (whether or not the type was new), False if there is nothing to
    tag. Safe to call repeatedly."""
    with get_connection() as conn:
        cur = conn.execute(_MERGE_TYPE_SQL, (document_type, document_uuid, model_uuid))
    return cur.rowcount > 0


def has_embedding(document_uuid: str, model_uuid: str = DEFAULT_EMBEDDING_MODEL) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM embeddings WHERE document_uuid=? AND model_uuid=?",
            (document_uuid, model_uuid),
        ).fetchone()
    return row is not None


def get_embedding(document_uuid: str, model_uuid: str = DEFAULT_EMBEDDING_MODEL) -> list[float] | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT embedding FROM embeddings WHERE document_uuid=? AND model_uuid=?",
            (document_uuid, model_uuid),
        ).fetchone()
    if not row:
        return None
    data = row[0]
    return array.array('f', data).tolist()


def get_or_generate_embedding(document_uuid: str, document_type: str, text: str, model_uuid: str = DEFAULT_EMBEDDING_MODEL) -> list[float]:
    cached = get_embedding(document_uuid, model_uuid)
    if cached is not None:
        # Record the type even on a cache hit. This call asserts that `document_type` applies to
        # this text; if another pipeline embedded the identical text first, that assertion is the
        # only place the second type is ever learned, and dropping it is what made whole corpora
        # unenumerable (only 469 of 2420 censored papers were reachable by their own type).
        add_document_type(document_uuid, document_type, model_uuid)
        return cached
    return generate_embedding(document_uuid, document_type, text, model_uuid)


def generate_embedding(document_uuid: str, document_type: str, text: str, model_uuid: str = DEFAULT_EMBEDDING_MODEL) -> list[float]:
    client = openai.OpenAI(
        api_key=secret_management.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    response = client.embeddings.create(input=text, model=model_uuid)
    vector = response.data[0].embedding

    with get_connection() as conn:
        # OR IGNORE then merge, rather than OR REPLACE: if a concurrent worker already stored this
        # text we keep its vector and only contribute our type, so a race can add a type but never
        # drop one and never overwrites an embedding.
        conn.execute(
            "INSERT OR IGNORE INTO embeddings (document_uuid, model_uuid, document_type, embedding) "
            "VALUES (?, ?, json_array(?), ?)",
            (document_uuid, model_uuid, document_type, array.array('f', vector).tobytes()),
        )
        conn.execute(_MERGE_TYPE_SQL, (document_type, document_uuid, model_uuid))

    return vector

def get_embeddings_by_type(document_type: str, model_uuid: str = DEFAULT_EMBEDDING_MODEL) -> list[tuple[str, list[float]]]:
    """Every embedding whose type array contains ``document_type``, ordered by uuid.

    The ORDER BY is load-bearing: without it SQLite returns rows in scan order, which is insert
    history and is free to change after a VACUUM or a rebuild. That order propagates into
    UMAP/HDBSCAN input ordering and silently shifts topic ids, with no seed to blame."""
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT document_uuid, embedding FROM embeddings "
            f"WHERE model_uuid=? AND EXISTS "
            f"  (SELECT 1 FROM json_each({_TYPES_AS_JSON}) WHERE value=?) "
            f"ORDER BY document_uuid",
            (model_uuid, document_type),
        ).fetchall()
    return [
        (document_uuid, array.array('f', embedding).tolist())
        for document_uuid, embedding in rows
    ]

def get_embeddings_by_uuid(document_uuids, model_uuid: str = DEFAULT_EMBEDDING_MODEL) -> dict[str, list[float]]:
    """Bulk lookup of many documents at once: {document_uuid: embedding} for the cached subset.

    Uncached uuids are simply absent from the result. Ids are queried in batches to stay under
    SQLite's bound-variable limit; this beats calling get_embedding in a loop, which reopens the
    database (and re-runs the CREATE TABLE) for every single id."""
    uuids = list(dict.fromkeys(document_uuids))
    found: dict[str, list[float]] = {}
    with get_connection() as conn:
        for start in range(0, len(uuids), 900):
            batch = uuids[start:start + 900]
            placeholders = ",".join("?" * len(batch))
            rows = conn.execute(
                f"SELECT document_uuid, embedding FROM embeddings "
                f"WHERE model_uuid=? AND document_uuid IN ({placeholders})",
                (model_uuid, *batch),
            ).fetchall()
            found.update({uuid: array.array('f', embedding).tolist() for uuid, embedding in rows})
    return found

def get_all_embeddings(model_uuid: str = DEFAULT_EMBEDDING_MODEL) -> list[tuple[str, list[float]]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT document_uuid, embedding FROM embeddings WHERE model_uuid=? ORDER BY document_uuid",
            (model_uuid,),
        ).fetchall()
    return [
        (document_uuid, array.array('f', embedding).tolist())
        for document_uuid, embedding in rows
    ]

_tokenizer = None

def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_NAME, trust_remote_code=True)
    return _tokenizer

def get_tokenized_string(text: str) -> list:
    tokenizer = _get_tokenizer()
    return tokenizer.encode(text)

def token_sequence_to_string(tokens) -> str:
    tokenizer = _get_tokenizer()

    return tokenizer.decode(tokens)

def split_long_document(document_text):
    # This split is based off the rule of thumb that one token \approx 3-4 chars. It works in this context but
    # technically it isn't correct. A document consisting of 100k 𰻝's would almost certainly be more than 32k tokens (the Qwen limit).
    # We ignore this for speed, since tokenizing everything would be a masssive pain.
    if len(document_text) < CONTEXT_WINDOW_LIMIT * 3:
        return [document_text]
    
    tokens = get_tokenized_string(document_text)
    segments = math.ceil(len(tokens)/CONTEXT_WINDOW_LIMIT)
    tokens_per_segment = math.ceil(len(tokens)/segments)

    return [token_sequence_to_string(tokens[i*tokens_per_segment: (i+1)*tokens_per_segment]) for i in range(segments)]

def get_wp_ip_embedding_args(document_text: str, t):
    document_segments = split_long_document(document_text)

    ret_arr = []
    for _, segment in enumerate(document_segments):
        ret_arr.append((hashlib.sha256(segment.encode()).hexdigest(), t, segment,))
    
    return ret_arr

class EmbeddingLookerUpper():
    def __init__(self, document_type: str | None, model_uuid: str = DEFAULT_EMBEDDING_MODEL):
        if isinstance(document_type, str):
            self.embeddings = get_embeddings_by_type(document_type, model_uuid)
        else:
            self.embeddings = get_all_embeddings(model_uuid)
        
        self.nn = NearestNeighbors(metric='cosine').fit([e[1] for e in self.embeddings])
    
    def get_nearest_neighbours(self, document_uuid, n_neighbours=5, model_uuid: str = DEFAULT_EMBEDDING_MODEL):
        document_embedding = get_embedding(document_uuid, model_uuid)

        nearest_neighbours = self.nn.kneighbors([document_embedding], n_neighbors=n_neighbours)
        
        return list(zip(map(lambda i: self.embeddings[i][0],  nearest_neighbours[1][0]), nearest_neighbours[0][0]))

def get_representation_of_measure(row):
    return f"Subject: {row.Subject}\n{row.Content}"

def measure_id_to_uuid(measure_id):
    return f"MEASURE__{measure_id}"

class DocumentTextGetter():
    def __init__(self) -> None:
        self.measures_pd = pandas.read_csv("data/MeasureCorpusEnriched.csv")
        self.wp_ip_map = downloaders.map_all_wp_ip_locations.map_all_wp_ip_file_locations()
        self.wp_ip_info = pandas.read_parquet("data/antarctic-db/processed/document-summary.parquet")
    
    def get_measure_representation(self, measure_id):
        measures_pd_row = self.measures_pd[self.measures_pd["Document_Number"] == measure_id].iloc[0]
        text_rep = get_representation_of_measure(measures_pd_row)
        return {"measure_id": measure_id, "text": text_rep, "year": measures_pd_row["Adoption_Year"]}

    def get_wp_ip_representation(self, document_uuid):
        document_file = pathlib.Path(self.wp_ip_map[document_uuid])
        wp_info_row = self.wp_ip_info[self.wp_ip_info["paper_url"].str.contains(document_file.stem)]

        with open(document_file, "r") as f:
            text = f.read()

        if len(wp_info_row) >= 1: # Multiple rows if multiple attachements.
            wp_info_row = wp_info_row.iloc[0]
            return {"text": text, "year": wp_info_row['meeting_year'], "sort_string": f"YEAR_{wp_info_row['meeting_year']}_DOCNUM_{wp_info_row['paper_number']}_TYPE_{wp_info_row['party_type']}", "parties": wp_info_row['parties'], "paper_language": wp_info_row["paper_language"]}
        return {"text": text}

    def get_document_representation(self, document_uuid: str) -> dict:
        if "MEASURE__" in document_uuid:
            measure_id = int(document_uuid.removeprefix("MEASURE__"))
            return self.get_measure_representation(measure_id)
        else:
            return self.get_wp_ip_representation(document_uuid)

    def get_all_of_type(self, type: str, with_embeddings: bool = False):
        pairs = get_embeddings_by_type(type)
        if with_embeddings:
            return [{**self.get_document_representation(uuid), "uuid": uuid, "embedding": embedding} for uuid, embedding in pairs]
        return [{**self.get_document_representation(uuid), "uuid": uuid} for uuid, _ in pairs]

if __name__ == "__main__":
    print(DocumentTextGetter().get_all_of_type("WorkingPaper"))