"""Tests for the embedding store: fetching, persisting, and the multi-valued `document_type`.

A document's uuid is the sha256 of its text, so pipelines that emit byte-identical text share one
row. `document_type` is therefore a JSON array and types accumulate; the single-valued column it
replaced let whichever pipeline embedded the text first claim the row, and every other type's
enumeration silently missed it. These tests pin that accumulation, the ordering guarantee that
keeps UMAP input stable, and the tolerance for pre-array scalar cells (the one-off migration
script that produced them has been deleted, but old databases can still hold them).

Every test runs against a throwaway sqlite file, and the one test that touches `generate_embedding`
stubs the OpenRouter client, so nothing here makes a paid call.
"""
import array
import hashlib
import json
import sqlite3
from types import SimpleNamespace

import pytest

import embeddings.document_embeddings as de

MODEL = de.DEFAULT_EMBEDDING_MODEL
OTHER_MODEL = "some/other-embedder"


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Point the store at a throwaway database."""
    path = tmp_path / "embeddings.sqlite3"
    monkeypatch.setattr(de, "EMBEDDINGS_DB_PATH", path)
    de.get_connection().close()  # create the schema
    return path


def _insert(db, uuid, type_cell, vector=(1.0, 0.0), model=MODEL):
    """Insert a row with `type_cell` written verbatim, so tests can plant either a JSON array or a
    pre-migration scalar."""
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO embeddings (document_uuid, model_uuid, document_type, embedding) "
            "VALUES (?, ?, ?, ?)",
            (uuid, model, type_cell, array.array("f", vector).tobytes()),
        )


def _stored_types(db, uuid, model=MODEL):
    with sqlite3.connect(db) as conn:
        cell = conn.execute(
            "SELECT document_type FROM embeddings WHERE document_uuid=? AND model_uuid=?",
            (uuid, model),
        ).fetchone()[0]
    return cell


# ------------------------------------------------------------------- parse_document_types

@pytest.mark.parametrize("cell, expected", [
    ('["WorkingPaper"]', ["WorkingPaper"]),
    ('["CensoredWorkingPaperV1","WorkingPaper"]', ["CensoredWorkingPaperV1", "WorkingPaper"]),
    ("[]", []),
    ("", []),
    (None, []),
])
def test_parse_document_types_reads_the_array_form(cell, expected):
    assert de.parse_document_types(cell) == expected


def test_parse_document_types_accepts_a_pre_migration_scalar():
    """A database mid-migration still has bare strings; reading one must not raise."""
    assert de.parse_document_types("WorkingPaper") == ["WorkingPaper"]


def test_parse_document_types_treats_malformed_json_as_one_scalar():
    assert de.parse_document_types("[not json") == ["[not json"]


# ----------------------------------------------------------------------- add_document_type

def test_add_document_type_accumulates_rather_than_overwrites(db):
    _insert(db, "h1", '["WorkingPaper"]')
    de.add_document_type("h1", "CensoredWorkingPaperV1")
    assert de.get_document_types("h1") == ["CensoredWorkingPaperV1", "WorkingPaper"]


def test_add_document_type_is_idempotent(db):
    _insert(db, "h1", '["WorkingPaper"]')
    for _ in range(3):
        de.add_document_type("h1", "CensoredWorkingPaperV1")
    assert de.get_document_types("h1") == ["CensoredWorkingPaperV1", "WorkingPaper"]


def test_add_document_type_stores_a_canonical_sorted_encoding(db):
    """Two rows that end up with the same set must compare equal as stored text, so the column can
    be grouped and diffed without re-parsing."""
    _insert(db, "h1", '["WorkingPaper"]')
    _insert(db, "h2", '["CensoredWorkingPaperV1"]')
    de.add_document_type("h1", "CensoredWorkingPaperV1")
    de.add_document_type("h2", "WorkingPaper")
    assert _stored_types(db, "h1") == _stored_types(db, "h2")


def test_add_document_type_upgrades_a_pre_migration_scalar_cell(db):
    _insert(db, "h1", "WorkingPaper")
    de.add_document_type("h1", "CensoredWorkingPaperV1")
    assert json.loads(_stored_types(db, "h1")) == ["CensoredWorkingPaperV1", "WorkingPaper"]


def test_add_document_type_reports_a_missing_row(db):
    assert de.add_document_type("nope", "WorkingPaper") is False


def test_add_document_type_does_not_touch_other_models(db):
    _insert(db, "h1", '["WorkingPaper"]', model=MODEL)
    _insert(db, "h1", '["WorkingPaper"]', model=OTHER_MODEL)
    de.add_document_type("h1", "CensoredWorkingPaperV1", model_uuid=MODEL)
    assert de.get_document_types("h1", OTHER_MODEL) == ["WorkingPaper"]


# -------------------------------------------------------------------- get_embeddings_by_type

def test_a_multi_type_row_is_found_under_every_one_of_its_types(db):
    """The whole point of the array: one row, reachable by each type that produced its text."""
    _insert(db, "h1", '["CensoredWorkingPaperV1","WorkingPaper"]')
    assert [u for u, _ in de.get_embeddings_by_type("WorkingPaper")] == ["h1"]
    assert [u for u, _ in de.get_embeddings_by_type("CensoredWorkingPaperV1")] == ["h1"]


def test_get_embeddings_by_type_does_not_match_an_unrelated_type(db):
    _insert(db, "h1", '["WorkingPaper"]')
    assert de.get_embeddings_by_type("InformationPaper") == []


def test_get_embeddings_by_type_does_not_match_a_type_by_prefix(db):
    """`WPAuthorClf::raw::full` must not be returned for `WPAuthorClf::raw`."""
    _insert(db, "h1", '["WPAuthorClf::raw::full"]')
    assert de.get_embeddings_by_type("WPAuthorClf::raw") == []


def test_get_embeddings_by_type_reads_a_pre_migration_scalar_cell(db):
    _insert(db, "h1", "WorkingPaper")
    assert [u for u, _ in de.get_embeddings_by_type("WorkingPaper")] == ["h1"]


def test_get_embeddings_by_type_is_ordered_by_uuid(db):
    """Unordered rows would reach UMAP/HDBSCAN in insert order and shift topic ids after a VACUUM."""
    for uuid in ["cc", "aa", "bb"]:
        _insert(db, uuid, '["WorkingPaper"]')
    assert [u for u, _ in de.get_embeddings_by_type("WorkingPaper")] == ["aa", "bb", "cc"]


def test_get_embeddings_by_type_filters_by_model(db):
    _insert(db, "h1", '["WorkingPaper"]', model=MODEL)
    _insert(db, "h2", '["WorkingPaper"]', model=OTHER_MODEL)
    assert [u for u, _ in de.get_embeddings_by_type("WorkingPaper", MODEL)] == ["h1"]


def test_get_embeddings_by_type_returns_the_stored_vector(db):
    _insert(db, "h1", '["WorkingPaper"]', vector=(0.5, -0.25))
    assert de.get_embeddings_by_type("WorkingPaper")[0][1] == [0.5, -0.25]


# ------------------------------------------------------------------------- fetch / persist

def test_get_embedding_round_trips_a_vector(db):
    _insert(db, "h1", '["WorkingPaper"]', vector=(0.5, -0.25, 0.125))
    assert de.get_embedding("h1") == [0.5, -0.25, 0.125]


def test_get_embedding_returns_none_when_absent(db):
    assert de.get_embedding("nope") is None


def test_has_embedding(db):
    _insert(db, "h1", '["WorkingPaper"]')
    assert de.has_embedding("h1") is True
    assert de.has_embedding("nope") is False


def test_get_embeddings_by_uuid_returns_only_the_cached_subset(db):
    _insert(db, "h1", '["WorkingPaper"]', vector=(0.5,))
    found = de.get_embeddings_by_uuid(["h1", "missing"])
    assert found == {"h1": [0.5]}


def test_get_all_embeddings_is_ordered_by_uuid(db):
    for uuid in ["cc", "aa", "bb"]:
        _insert(db, uuid, '["WorkingPaper"]')
    assert [u for u, _ in de.get_all_embeddings()] == ["aa", "bb", "cc"]


# ------------------------------------------------- type registration on the cache-hit path

def test_get_or_generate_records_the_type_on_a_cache_hit_without_calling_the_api(db, monkeypatch):
    """The root-cause fix. A second pipeline asking for text someone else already embedded is the
    only moment its type can be learned; the old code returned the cached vector and dropped it."""
    monkeypatch.setattr(de.openai, "OpenAI", lambda **kw: pytest.fail("must not call the API"))
    _insert(db, "h1", '["WorkingPaper"]', vector=(0.5,))

    assert de.get_or_generate_embedding("h1", "CensoredWorkingPaperV1", "text") == [0.5]
    assert de.get_document_types("h1") == ["CensoredWorkingPaperV1", "WorkingPaper"]


def _stub_openai(monkeypatch, vector):
    created = []

    class _Client:
        def __init__(self, **kwargs):
            self.embeddings = SimpleNamespace(
                create=lambda input, model: created.append(input) or SimpleNamespace(
                    data=[SimpleNamespace(embedding=list(vector))]
                )
            )

    monkeypatch.setattr(de.openai, "OpenAI", _Client)
    monkeypatch.setattr(de.secret_management, "get", lambda key: "test-key")
    return created


def test_generate_embedding_persists_the_type_as_an_array(db, monkeypatch):
    _stub_openai(monkeypatch, (0.5, 0.25))
    de.generate_embedding("h1", "WorkingPaper", "some text")
    assert json.loads(_stored_types(db, "h1")) == ["WorkingPaper"]
    assert de.get_embedding("h1") == [0.5, 0.25]


def test_generate_embedding_merges_into_an_existing_row_and_keeps_its_vector(db, monkeypatch):
    """Two workers racing on the same text: the loser must contribute its type without clobbering
    the stored vector."""
    _insert(db, "h1", '["WorkingPaper"]', vector=(0.5, 0.25))
    _stub_openai(monkeypatch, (9.0, 9.0))

    de.generate_embedding("h1", "CensoredWorkingPaperV1", "some text")

    assert de.get_document_types("h1") == ["CensoredWorkingPaperV1", "WorkingPaper"]
    assert de.get_embedding("h1") == [0.5, 0.25], "existing vector must not be overwritten"


# ------------------------------------------------------- building the database from scratch

def test_from_scratch_build_gives_a_no_op_censored_paper_both_types(db, monkeypatch):
    """`embed_all` embeds raw /wp/ text first, then the censored text in a later pass. When
    censoring is a no-op the two passes produce the SAME uuid, and the row must end up carrying
    both types -- this is the case that a single-valued column got wrong for 633 papers."""
    calls = _stub_openai(monkeypatch, (0.5, 0.25))
    raw = "A working paper with no target-country mention."
    censored = raw  # censor_text is a no-op here

    de.get_or_generate_embedding(*de.get_wp_ip_embedding_args(raw, "WorkingPaper")[0][:2],
                                 raw)
    de.get_or_generate_embedding(
        *de.get_wp_ip_embedding_args(censored, "CensoredWorkingPaperV1")[0][:2], censored)

    (uuid,) = {u for u, _ in de.get_embeddings_by_type("WorkingPaper")}
    assert de.get_document_types(uuid) == ["CensoredWorkingPaperV1", "WorkingPaper"]
    assert len(calls) == 1, "the second pass must be a cache hit, not a second paid call"


def test_from_scratch_build_keeps_a_really_censored_paper_as_two_rows(db, monkeypatch):
    """The contrasting case: censoring changed the text, so there are two distinct hashes and each
    carries exactly one type."""
    _stub_openai(monkeypatch, (0.5, 0.25))
    raw = "Submitted by the United Kingdom."
    censored = "Submitted by CountryName."

    de.get_or_generate_embedding(*de.get_wp_ip_embedding_args(raw, "WorkingPaper")[0][:2], raw)
    de.get_or_generate_embedding(
        *de.get_wp_ip_embedding_args(censored, "CensoredWorkingPaperV1")[0][:2], censored)

    assert [de.get_document_types(u) for u, _ in de.get_embeddings_by_type("WorkingPaper")] \
        == [["WorkingPaper"]]
    assert [de.get_document_types(u) for u, _ in de.get_embeddings_by_type("CensoredWorkingPaperV1")] \
        == [["CensoredWorkingPaperV1"]]


def test_a_later_experiment_type_attaches_to_an_existing_corpus_row(db, monkeypatch):
    """Why WPAuthorClf::* is deliberately left out of the migration: running the benchmark after a
    from-scratch corpus build attaches its types to the rows the corpus pass already embedded."""
    _stub_openai(monkeypatch, (0.5,))
    text = "An uncensored working paper."
    de.get_or_generate_embedding(*de.get_wp_ip_embedding_args(text, "WorkingPaper")[0][:2], text)
    de.get_or_generate_embedding(
        *de.get_wp_ip_embedding_args(text, "WPAuthorClf::raw::full")[0][:2], text)

    (uuid,) = {u for u, _ in de.get_embeddings_by_type("WorkingPaper")}
    assert de.get_document_types(uuid) == ["WPAuthorClf::raw::full", "WorkingPaper"]


# --------------------------------------------------------------- empty documents are not embedded

def test_empty_documents_produce_no_embedding_args():
    """The embedding endpoint 400s on an empty input, and a 400 is the same on every retry, so one
    zero-byte file (ATCM2_wp010_e.txt is one) used to take down the entire embedding run. Empty
    text has no embedding worth caching either way, so it is dropped before the pool ever sees it."""
    assert de.get_wp_ip_embedding_args("", "WorkingPaper") == []
    assert de.get_wp_ip_embedding_args("   \n\t  ", "WorkingPaper") == []


def test_non_empty_documents_are_unaffected():
    """The empty-document filter must not perturb the hashes of real documents: those uuids are the
    primary key of the cache, so a change would orphan every row already stored."""
    text = "A working paper with content."
    (uuid, doc_type, segment), = de.get_wp_ip_embedding_args(text, "WorkingPaper")
    assert uuid == hashlib.sha256(text.encode()).hexdigest()
    assert (doc_type, segment) == ("WorkingPaper", text)


# ------------------------------------------------- splitting respects BOTH limits, not just tokens

class _FakeTokenizer:
    """A tokenizer with a fixed, tunable character-per-token density.

    The real Qwen tokenizer is a large download and its density varies by content, which is exactly
    the variable under test. Here one token is `density` copies of a character, so a document's
    token count and character count can be dialled independently."""

    def __init__(self, density):
        self.density = density

    def encode(self, text):
        return list(range(len(text) // self.density))

    def decode(self, tokens):
        return "x" * (len(tokens) * self.density)


@pytest.fixture
def dense_tokenizer(monkeypatch):
    """4.16 chars/token -- the density of ATCM25_ip102_e, the document that triggered the 422."""
    monkeypatch.setattr(de, "_tokenizer", _FakeTokenizer(density=4))


def test_a_token_sized_segment_can_still_exceed_the_character_cap(dense_tokenizer):
    """The regression. 32000 tokens at 4 chars/token is 128000 characters, inside the cap; but the
    old splitter sized purely by tokens, so a dense document produced a segment over the cap and
    the endpoint 422'd. Every segment must now satisfy the character bound independently."""
    text = "x" * 258763  # ATCM25_ip102_e's length
    segments = de.split_long_document(text)

    assert len(segments) > 1
    for segment in segments:
        assert len(segment) < de.MAX_INPUT_CHARACTERS
        assert len(de.get_tokenized_string(segment)) <= de.CONTEXT_WINDOW_LIMIT


def test_character_bound_alone_can_force_a_split(monkeypatch):
    """A document light on tokens but heavy on characters is inside the token window entirely, so
    the token arithmetic alone returns it whole. The character cap still has to break it up."""
    monkeypatch.setattr(de, "_tokenizer", _FakeTokenizer(density=40))
    text = "x" * 400000  # only 10000 tokens, but way over the character cap

    segments = de.split_long_document(text)

    assert len(de.get_tokenized_string(text)) <= de.CONTEXT_WINDOW_LIMIT  # token budget alone: fine
    assert len(segments) > 1
    assert all(len(segment) < de.MAX_INPUT_CHARACTERS for segment in segments)


def test_short_documents_are_returned_whole(dense_tokenizer):
    """The early return must stay a pure passthrough: these uuids are cache keys, so re-splitting
    documents that already embed fine would orphan every row stored against them."""
    text = "x" * (de.CONTEXT_WINDOW_LIMIT * 3 - 1)
    assert de.split_long_document(text) == [text]


def test_the_early_return_can_never_emit_an_oversize_segment():
    """The early return skips tokenization entirely, so it is only safe while its character
    threshold sits below the cap. If CONTEXT_WINDOW_LIMIT is ever raised this fails, which is the
    intent -- the passthrough would start emitting segments the endpoint rejects."""
    assert de.CONTEXT_WINDOW_LIMIT * 3 < de.MAX_INPUT_CHARACTERS


# ------------------------------------------------------ enumeration is per document, not per row

def _getter(wp_ip_map):
    """A `DocumentTextGetter` with only `wp_ip_map` populated.

    `__init__` reads the measure CSV, the wp/ip location map and the document-summary parquet off
    disk; none of that is what `get_all_of_type` is doing, so it is bypassed and the one attribute
    the grouping consults is planted directly. `get_wp_ip_representation` is replaced with a stub
    that names the segment it was called with, so a test can see which segment built the entry.
    """
    getter = object.__new__(de.DocumentTextGetter)
    getter.wp_ip_map = wp_ip_map
    getter.get_wp_ip_representation = lambda uuid: {
        "text": f"text of {wp_ip_map[uuid]}", "built_from": uuid,
    }
    return getter


def test_segments_of_one_document_collapse_to_a_single_entry(db):
    """The bug this replaced: a paper too long for the context window is stored as several rows,
    every one of which resolves to the same file and so came back as a full duplicate document.
    A topic model then clustered the copies, and every per-document tally counted the paper twice."""
    for uuid in ["s1", "s2", "s3"]:
        _insert(db, uuid, '["WorkingPaper"]')
    getter = _getter({"s1": "/wp/long.txt", "s2": "/wp/long.txt", "s3": "/wp/long.txt"})

    documents = getter.get_all_of_type("WorkingPaper")

    assert len(documents) == 1
    assert documents[0]["n_segments"] == 3
    assert documents[0]["source"] == "/wp/long.txt"


def test_distinct_documents_are_not_merged(db):
    _insert(db, "a1", '["WorkingPaper"]')
    _insert(db, "b1", '["WorkingPaper"]')
    getter = _getter({"a1": "/wp/a.txt", "b1": "/wp/b.txt"})

    assert sorted(d["source"] for d in getter.get_all_of_type("WorkingPaper")) \
        == ["/wp/a.txt", "/wp/b.txt"]


def test_a_documents_embedding_is_its_segments_pooled(db):
    """The vector must be what OpenRouterBackend would produce for the whole document -- the
    mean of its segments, re-normalised -- not whichever segment happened to be enumerated first."""
    _insert(db, "s1", '["WorkingPaper"]', vector=(1.0, 0.0))
    _insert(db, "s2", '["WorkingPaper"]', vector=(0.0, 1.0))
    getter = _getter({"s1": "/wp/long.txt", "s2": "/wp/long.txt"})

    (document,) = getter.get_all_of_type("WorkingPaper", with_embeddings=True)

    assert document["embedding"] == pytest.approx([2 ** -0.5, 2 ** -0.5])


def test_segments_with_no_source_file_are_skipped_and_reported(db, capsys):
    """`wp_ip_map` is written once and never invalidated, so after the segmentation changes the
    store holds segment hashes the map has never seen. They belong to no document and cannot be
    pooled -- but dropping them quietly is how a stale map goes unnoticed, so the count is printed."""
    _insert(db, "known", '["WorkingPaper"]')
    _insert(db, "orphan", '["WorkingPaper"]')
    getter = _getter({"known": "/wp/a.txt"})

    documents = getter.get_all_of_type("WorkingPaper")

    assert [d["source"] for d in documents] == ["/wp/a.txt"]
    assert "1 'WorkingPaper' embedding(s) map to no source file" in capsys.readouterr().out


def test_measures_are_their_own_documents(db):
    """Measures are embedded whole under a synthetic uuid rather than hashed per segment, so each
    is already one row -- grouping must leave them exactly as they were."""
    _insert(db, "MEASURE__1", '["measure"]')
    _insert(db, "MEASURE__2", '["measure"]')
    getter = _getter({})
    getter.get_measure_representation = lambda measure_id: {"measure_id": measure_id}

    documents = getter.get_all_of_type("measure")

    assert [d["uuid"] for d in documents] == ["MEASURE__1", "MEASURE__2"]
    assert all(d["n_segments"] == 1 for d in documents)


def test_document_order_follows_uuid_order(db):
    """Grouping keeps first-seen order and the underlying query is ordered by uuid, so the document
    list is stable across rebuilds -- UMAP's input ordering depends on it."""
    for uuid in ["cc", "aa", "bb"]:
        _insert(db, uuid, '["WorkingPaper"]')
    getter = _getter({"aa": "/wp/a.txt", "bb": "/wp/b.txt", "cc": "/wp/c.txt"})

    assert [d["source"] for d in getter.get_all_of_type("WorkingPaper")] \
        == ["/wp/a.txt", "/wp/b.txt", "/wp/c.txt"]


def test_mean_pool_renormalises():
    """Segment vectors are unit-norm; a plain mean of disagreeing segments is not, which would make
    a long document's cosine similarities incomparable with a short one's."""
    import numpy as np

    pooled = de.mean_pool([[1.0, 0.0], [0.0, 1.0]])
    assert np.linalg.norm(pooled) == pytest.approx(1.0)


def test_mean_pool_leaves_a_single_segment_alone():
    assert de.mean_pool([[1.0, 0.0]]) == pytest.approx([1.0, 0.0])


# ---------------------------------------------- content hashing: measures are not content-addressed

def test_generate_embedding_records_the_content_hash(db, monkeypatch):
    _stub_openai(monkeypatch, (0.5,))
    de.generate_embedding("MEASURE__1", "measure", "Subject: A\nOriginal content")
    assert de.get_content_hash("MEASURE__1") == de.content_hash("Subject: A\nOriginal content")


def test_is_stale_detects_changed_text_behind_a_synthetic_uuid(db, monkeypatch):
    """Working papers self-correct: their uuid *is* the hash of their text, so editing the text
    yields a new uuid and a cache miss. Measures are keyed on a synthetic MEASURE__{id}, so an
    edited Subject/Content keeps the same key and the stale vector is returned forever."""
    _stub_openai(monkeypatch, (0.5,))
    de.generate_embedding("MEASURE__1", "measure", "Subject: A\nOriginal content")

    assert de.is_stale("MEASURE__1", "Subject: A\nOriginal content") is False
    assert de.is_stale("MEASURE__1", "Subject: A\nEDITED content") is True


def test_is_stale_is_false_for_an_absent_row(db):
    assert de.is_stale("MEASURE__404", "anything") is False


def test_is_stale_is_false_when_the_hash_was_never_recorded(db):
    """Rows written before the column existed have content_hash NULL. That is unknown, not
    mismatched -- treating it as stale would re-embed the whole corpus to learn nothing."""
    _insert(db, "MEASURE__1", '["measure"]')
    assert de.get_content_hash("MEASURE__1") is None
    assert de.is_stale("MEASURE__1", "anything at all") is False


def test_delete_embedding_forces_regeneration(db, monkeypatch):
    calls = _stub_openai(monkeypatch, (0.5,))
    de.generate_embedding("MEASURE__1", "measure", "first")
    assert de.delete_embedding("MEASURE__1") is True
    assert de.get_embedding("MEASURE__1") is None

    de.generate_embedding("MEASURE__1", "measure", "second")
    assert de.get_content_hash("MEASURE__1") == de.content_hash("second")
    assert len(calls) == 2


def test_delete_embedding_reports_an_absent_row(db):
    assert de.delete_embedding("nope") is False


def test_the_content_hash_column_is_added_to_an_existing_database(tmp_path, monkeypatch):
    """The column arrives by ALTER on a database that predates it, so an existing store picks it
    up in place rather than needing a rebuild."""
    path = tmp_path / "old.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute("""
            CREATE TABLE embeddings (
                document_uuid TEXT NOT NULL, model_uuid TEXT NOT NULL,
                document_type TEXT NOT NULL, embedding BLOB NOT NULL,
                PRIMARY KEY (document_uuid, model_uuid))
        """)
        conn.execute(
            "INSERT INTO embeddings VALUES (?, ?, ?, ?)",
            ("h1", MODEL, '["WorkingPaper"]', array.array("f", (1.0,)).tobytes()),
        )
    monkeypatch.setattr(de, "EMBEDDINGS_DB_PATH", path)

    assert de.get_embedding("h1") == [1.0], "the pre-existing row must survive the migration"
    assert de.get_content_hash("h1") is None


# ------------------------------------------------------------------- EmbeddingLookerUpper (kNN)

def _looker_upper(db, vectors: dict, document_type="WorkingPaper"):
    for uuid, vec in vectors.items():
        _insert(db, uuid, f'["{document_type}"]', vector=vec)
    return de.EmbeddingLookerUpper(document_type)


def test_nearest_neighbour_of_a_document_is_itself(db):
    """Cosine distance to itself is 0, so a document must always be its own first neighbour."""
    lookup = _looker_upper(db, {"a": (1.0, 0.0), "b": (0.0, 1.0), "c": (-1.0, 0.0)})
    (uuid, distance), *_ = lookup.get_nearest_neighbours("a", n_neighbours=1)
    assert uuid == "a"
    assert distance == pytest.approx(0.0, abs=1e-6)


def test_neighbours_come_back_in_increasing_distance(db):
    lookup = _looker_upper(db, {"a": (1.0, 0.0), "b": (0.0, 1.0), "c": (-1.0, 0.0)})
    results = lookup.get_nearest_neighbours("a", n_neighbours=3)

    assert [uuid for uuid, _ in results] == ["a", "b", "c"], "self, orthogonal, then opposite"
    distances = [d for _, d in results]
    assert distances == sorted(distances)


def test_distances_are_cosine_not_euclidean(db):
    """A vector pointing the same way but ten times longer is at cosine distance 0, and would be
    far away under any length-sensitive metric."""
    lookup = _looker_upper(db, {"a": (1.0, 0.0), "long": (10.0, 0.0), "b": (0.0, 1.0)})
    by_uuid = dict(lookup.get_nearest_neighbours("a", n_neighbours=3))

    assert by_uuid["long"] == pytest.approx(0.0, abs=1e-6)
    assert by_uuid["b"] == pytest.approx(1.0, abs=1e-6)


def test_lookup_is_scoped_to_its_document_type(db):
    """The index is built from one type, so a document of another type can never be returned."""
    _insert(db, "other", '["InformationPaper"]', vector=(1.0, 0.0))
    lookup = _looker_upper(db, {"a": (1.0, 0.0), "b": (0.0, 1.0)})

    returned = {uuid for uuid, _ in lookup.get_nearest_neighbours("a", n_neighbours=2)}
    assert "other" not in returned
    assert returned == {"a", "b"}


# ------------------------------- deduped dataset types are registered against the embedded row

def test_register_dropped_types_makes_every_dataset_enumerable(db, monkeypatch):
    """End-to-end for the dedup fix: one shared chunk, three datasets, all three types findable.

    A working paper naming none of the target countries censors to itself, so raw/naive/llm all
    produce one hash. Deduping embeds it once (correctly -- three calls would be three identical
    paid requests), but the two losing type strings then have to be attached to the row that was
    written, or `get_embeddings_by_type` reports those datasets as nearly empty.
    """
    from working_paper_authorship import country_authorship_classifier as cc

    calls = _stub_openai(monkeypatch, (0.5,))
    text = "A working paper naming no target country."
    types = ["WPAuthorClf::raw::full", "WPAuthorClf::naive::full", "WPAuthorClf::llm_censorship::full"]
    units = [de.get_wp_ip_embedding_args(text, t)[0] for t in types]

    unique, seen = cc.collect_unique_units(units)
    for digest, unit_type, unit_text in unique.values():
        de.get_or_generate_embedding(digest, unit_type, unit_text)
    added = cc.register_dropped_types(unique, seen)

    assert len(calls) == 1, "the shared chunk must be embedded exactly once"
    assert added == 2
    for t in types:
        assert len(de.get_embeddings_by_type(t)) == 1, f"{t} must be enumerable"


def test_register_dropped_types_is_idempotent(db, monkeypatch):
    from working_paper_authorship import country_authorship_classifier as cc

    _stub_openai(monkeypatch, (0.5,))
    text = "Shared text."
    types = ["WPAuthorClf::raw::full", "WPAuthorClf::naive::full"]
    units = [de.get_wp_ip_embedding_args(text, t)[0] for t in types]
    unique, seen = cc.collect_unique_units(units)
    for digest, unit_type, unit_text in unique.values():
        de.get_or_generate_embedding(digest, unit_type, unit_text)

    # The count is (hash, type) pairs tagged, so a rerun reports the same number; what must be
    # idempotent is the stored type set, which is a set and cannot accumulate duplicates.
    assert cc.register_dropped_types(unique, seen) == 1
    assert cc.register_dropped_types(unique, seen) == 1
    assert de.get_document_types(units[0][0]) == sorted(types)


def test_register_dropped_types_does_nothing_without_an_embedded_row(db):
    """It tags existing rows, so running it before the embedding pass is a no-op rather than a
    silent write of a typed row with no vector."""
    from working_paper_authorship import country_authorship_classifier as cc

    unique = {"never_embedded": ("never_embedded", "WPAuthorClf::raw::full", "text")}
    seen = {"never_embedded": {"WPAuthorClf::raw::full", "WPAuthorClf::naive::full"}}
    assert cc.register_dropped_types(unique, seen) == 0
    assert de.get_embedding("never_embedded") is None
