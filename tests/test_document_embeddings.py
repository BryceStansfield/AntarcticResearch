"""Tests for the embedding store: fetching, persisting, and the multi-valued `document_type`.

A document's uuid is the sha256 of its text, so pipelines that emit byte-identical text share one
row. `document_type` is therefore a JSON array and types accumulate; the single-valued column it
replaced let whichever pipeline embedded the text first claim the row, and every other type's
enumeration silently missed it. These tests pin that accumulation, the ordering guarantee that
keeps UMAP input stable, and the tolerance for not-yet-migrated scalar cells.

Every test runs against a throwaway sqlite file, and the one test that touches `generate_embedding`
stubs the OpenRouter client, so nothing here makes a paid call.
"""
import array
import json
import sqlite3
from types import SimpleNamespace

import pytest

import embeddings.document_embeddings as de
import embeddings.migrate_document_type_array as mig

MODEL = de.DEFAULT_EMBEDDING_MODEL
OTHER_MODEL = "some/other-embedder"


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Point the store (and the migration, which imported the path by value) at a temp database."""
    path = tmp_path / "embeddings.sqlite3"
    monkeypatch.setattr(de, "EMBEDDINGS_DB_PATH", path)
    monkeypatch.setattr(mig, "EMBEDDINGS_DB_PATH", path)
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


# --------------------------------------------------------------------------------- migration

def test_migration_wraps_scalar_cells_in_an_array(db, capsys):
    _insert(db, "h1", "WorkingPaper")
    mig.migrate(apply=True, expected={"h1": {"WorkingPaper"}})
    assert json.loads(_stored_types(db, "h1")) == ["WorkingPaper"]


def test_migration_merges_a_corpus_type_the_row_was_missing(db):
    """A no-op-censored paper: one row, but it is both the raw and the censored working paper."""
    _insert(db, "h1", "WorkingPaper")
    mig.migrate(apply=True, expected={"h1": {"WorkingPaper", "CensoredWorkingPaperV1"}})
    assert de.get_document_types("h1") == ["CensoredWorkingPaperV1", "WorkingPaper"]


def test_migration_strips_a_corpus_type_no_source_file_produces(db):
    """The attachment case: files under neither /wp/ nor /ip/ were filed as WorkingPaper."""
    _insert(db, "attachment", '["WorkingPaper"]')
    mig.migrate(apply=True, expected={})
    assert de.get_document_types("attachment") == []


def test_migration_leaves_non_corpus_types_alone(db):
    """It only recomputes the corpus types, so experiment types must survive untouched."""
    _insert(db, "h1", '["WPAuthorClf::raw::full","WorkingPaper"]')
    mig.migrate(apply=True, expected={})
    assert de.get_document_types("h1") == ["WPAuthorClf::raw::full"]


def test_migration_dry_run_writes_nothing(db):
    _insert(db, "h1", "WorkingPaper")
    mig.migrate(apply=False, expected={"h1": {"WorkingPaper", "CensoredWorkingPaperV1"}})
    assert _stored_types(db, "h1") == "WorkingPaper"


def test_migration_is_idempotent(db):
    _insert(db, "h1", "WorkingPaper")
    expected = {"h1": {"WorkingPaper", "CensoredWorkingPaperV1"}}
    mig.migrate(apply=True, expected=expected)
    first = _stored_types(db, "h1")
    mig.migrate(apply=True, expected=expected)
    assert _stored_types(db, "h1") == first


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


def test_migration_ignores_expected_hashes_that_are_not_embedded_yet(db):
    """The censorship fix changed what censor_text emits, so many expected hashes have no row yet.
    Those must be skipped, not invented."""
    mig.migrate(apply=True, expected={"never_embedded": {"CensoredWorkingPaperV1"}})
    assert de.get_embedding("never_embedded") is None
