"""Migrate ``embeddings.document_type`` from one string to a JSON array of strings.

A document's uuid is the sha256 of its text, so pipelines that produce byte-identical text share a
single row. The old single-valued column let whichever pipeline embedded the text first claim the
row, and every other type's enumeration silently missed it -- only 469 of 2420 censored working
papers were reachable under ``CensoredWorkingPaperV1``, and 42 attachment files that are neither
working nor information papers were filed as ``WorkingPaper`` by a fall-through bug.

Three phases, each idempotent, all inside one transaction:

  1. Wrap every not-yet-migrated scalar cell in a JSON array.
  2. Recompute corpus membership from the source files and merge the corpus types back in, so a
     hash that is legitimately several things ends up carrying all of them.
  3. Strip ``WorkingPaper`` / ``InformationPaper`` from hashes no such file actually produces.

Only the corpus-level types are recomputed here (``WorkingPaper``, ``InformationPaper``,
``CensoredWorkingPaperV1``). The per-experiment ``WPAuthorClf::*`` types repair themselves on the
next benchmark run: ``get_or_generate_embedding`` now records its type on a cache hit, so any
pipeline that asks for a hash it did not embed still gets its type recorded.

Defaults to a dry run. Pass --apply to write.
"""
import argparse
import collections
import pathlib
import sqlite3

from downloaders.map_all_wp_ip_locations import map_all_wp_ip_file_locations
from embeddings.document_embeddings import (
    EMBEDDINGS_DB_PATH, DEFAULT_EMBEDDING_MODEL, _TYPES_AS_JSON,
    get_wp_ip_embedding_args, parse_document_types,
)
from embeddings.working_paper_censorship import ENGLISH_SUFFIX, censor_text

WORKING_PAPER_TYPE = "WorkingPaper"
INFORMATION_PAPER_TYPE = "InformationPaper"
CENSORED_WORKING_PAPER_TYPE = "CensoredWorkingPaperV1"
# The types this migration is authoritative about: it can both add and remove these, because it
# recomputes them from the corpus. Every other type is left exactly as found.
CORPUS_TYPES = (WORKING_PAPER_TYPE, INFORMATION_PAPER_TYPE, CENSORED_WORKING_PAPER_TYPE)

_MERGE = f"""
UPDATE embeddings
   SET document_type = (SELECT json_group_array(value)
                          FROM (SELECT value FROM json_each({_TYPES_AS_JSON})
                                UNION SELECT ?
                                ORDER BY value))
 WHERE document_uuid=? AND model_uuid=?
"""

_STRIP = f"""
UPDATE embeddings
   SET document_type = (SELECT json_group_array(value)
                          FROM (SELECT value FROM json_each({_TYPES_AS_JSON})
                                WHERE value <> ?
                                ORDER BY value))
 WHERE document_uuid=? AND model_uuid=?
"""


def expected_corpus_types() -> dict[str, set[str]]:
    """Recompute {segment hash -> corpus types it belongs to} from the source files.

    Mirrors the writers exactly: ``embed_all`` hashes each /wp/ and /ip/ file's raw text, and
    ``embed_all_censored_working_papers`` hashes ``censor_text`` of each English working paper.
    Files under neither directory are attachments and belong to no corpus type -- the fall-through
    that used to file them under whichever type came before them is what phase 3 cleans up."""
    expected: dict[str, set[str]] = collections.defaultdict(set)
    locations = map_all_wp_ip_file_locations()
    counts = collections.Counter()

    for raw_path in locations.values():
        path = pathlib.Path(raw_path)
        if "/wp/" in raw_path:
            corpus_type = WORKING_PAPER_TYPE
        elif "/ip/" in raw_path:
            corpus_type = INFORMATION_PAPER_TYPE
        else:
            counts["attachment (no corpus type)"] += 1
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            counts["unreadable"] += 1
            continue
        for h, _t, _seg in get_wp_ip_embedding_args(text, corpus_type):
            expected[h].add(corpus_type)
        counts[corpus_type] += 1

        if corpus_type == WORKING_PAPER_TYPE and path.stem.endswith(ENGLISH_SUFFIX):
            for h, _t, _seg in get_wp_ip_embedding_args(censor_text(text), CENSORED_WORKING_PAPER_TYPE):
                expected[h].add(CENSORED_WORKING_PAPER_TYPE)
            counts[CENSORED_WORKING_PAPER_TYPE] += 1

    print("  source files walked:")
    for label, n in sorted(counts.items()):
        print(f"    {label:34} {n:6}")
    print(f"    {'distinct expected hashes':34} {len(expected):6}")
    return expected


def migrate(apply: bool, expected: dict[str, set[str]] | None = None,
            model_uuid: str = DEFAULT_EMBEDDING_MODEL) -> None:
    """Run the three phases. ``expected`` is the {hash -> corpus types} mapping phases 2 and 3 are
    judged against; it defaults to recomputing from the corpus, and is injectable so the phases can
    be tested against a synthetic corpus rather than the real 6.6k files."""
    conn = sqlite3.connect(EMBEDDINGS_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")

    print("Phase 0 — current state")
    total, arrays = conn.execute(
        "SELECT COUNT(*), SUM(document_type LIKE '[%') FROM embeddings"
    ).fetchone()
    print(f"  rows: {total}   already arrays: {arrays or 0}")

    conn.execute("BEGIN IMMEDIATE")
    try:
        print("\nPhase 1 — wrap scalar cells in a JSON array")
        cur = conn.execute(
            "UPDATE embeddings SET document_type = json_array(document_type) "
            "WHERE document_type NOT LIKE '[%'"
        )
        print(f"  wrapped: {cur.rowcount}")

        print("\nPhase 2 — recompute corpus membership from source and merge")
        if expected is None:
            expected = expected_corpus_types()
        present = {
            uuid: set(parse_document_types(cell))
            for uuid, cell in conn.execute(
                "SELECT document_uuid, document_type FROM embeddings WHERE model_uuid=?",
                (model_uuid,),
            )
        }
        added = collections.Counter()
        absent = collections.Counter()
        for h, types in expected.items():
            if h not in present:
                # Text whose hash is not in the database at all: nothing to tag. Expected for the
                # censored types right now, since the censorship fix changed what censor_text emits.
                for t in types:
                    absent[t] += 1
                continue
            for t in sorted(types - present[h]):
                conn.execute(_MERGE, (t, h, model_uuid))
                added[t] += 1
        for t in CORPUS_TYPES:
            print(f"    +{added[t]:6} {t:26} ({absent[t]} expected hashes not yet embedded)")

        print("\nPhase 3 — strip corpus types from hashes no such file produces")
        removed = collections.Counter()
        orphaned = 0
        for uuid, types in present.items():
            legitimate = expected.get(uuid, set())
            for t in sorted((types & set(CORPUS_TYPES)) - legitimate):
                conn.execute(_STRIP, (t, uuid, model_uuid))
                removed[t] += 1
                if not (types - {t}):
                    orphaned += 1
        for t in CORPUS_TYPES:
            print(f"    -{removed[t]:6} {t}")
        print(f"    rows left with an empty type array: {orphaned}")

        if apply:
            conn.commit()
            print("\nCOMMITTED.")
        else:
            conn.rollback()
            print("\nDRY RUN — rolled back. Re-run with --apply to write.")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def report_types(model_uuid: str = DEFAULT_EMBEDDING_MODEL) -> None:
    """Per-type row counts read through the array, so it reflects the post-migration meaning."""
    conn = sqlite3.connect(EMBEDDINGS_DB_PATH)
    rows = conn.execute(
        f"SELECT value, COUNT(*) FROM embeddings, json_each({_TYPES_AS_JSON}) "
        f"WHERE model_uuid=? GROUP BY value ORDER BY COUNT(*) DESC",
        (model_uuid,),
    ).fetchall()
    width = max((len(t) for t, _ in rows), default=10)
    for t, n in rows:
        print(f"  {t:{width}}  {n}")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write the changes (default: dry run)")
    parser.add_argument("--report", action="store_true", help="just print per-type row counts")
    args = parser.parse_args()
    if args.report:
        report_types()
    else:
        migrate(apply=args.apply)
