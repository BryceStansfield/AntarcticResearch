"""Tests for the segment-hash -> source-file cache.

This map is the join between the embedding store (keyed on sha256 of a *segment* of a document)
and the files those segments were cut from. Everything that asks "which paper is this embedding"
goes through it, so two properties matter:

* **It has to agree with the embedder byte for byte.** The hashes are the join key, so any
  difference in how the text is decoded produces keys the store has never seen.
* **It has to notice when it is out of date.** It was previously returned on existence alone, so
  after the segmentation changed it kept resolving hashes that no longer existed while the store
  filled with hashes it had never heard of -- surfacing as a KeyError in every consumer.
"""
import hashlib
import json

import pytest

import downloaders.map_all_wp_ip_locations as mapper


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """A processed/ tree with one dataset dir, pointed at by the module's module-level paths."""
    processed = tmp_path / "processed"
    dataset = processed / "dataset-abc"
    dataset.mkdir(parents=True)
    monkeypatch.setattr(mapper, "PROCESSED_PATH", processed)
    monkeypatch.setattr(mapper, "MAP_PATH", processed / "wp_ip_file_locations.json")
    return dataset


def _write(dataset, name, text):
    path = dataset / name
    path.write_text(text, encoding="utf-8")
    return path


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def test_maps_each_documents_hash_to_its_file(corpus):
    _write(corpus, "a.txt", "paper one")
    _write(corpus, "b.txt", "paper two")

    mapping = mapper.map_all_wp_ip_file_locations()

    assert mapping[_sha("paper one")].endswith("a.txt")
    assert mapping[_sha("paper two")].endswith("b.txt")


def test_the_cache_is_reused_when_the_corpus_is_unchanged(corpus, monkeypatch):
    _write(corpus, "a.txt", "paper one")
    mapper.map_all_wp_ip_file_locations()

    monkeypatch.setattr(mapper, "_build", lambda *a: pytest.fail("must not rebuild"))
    assert mapper.map_all_wp_ip_file_locations()[_sha("paper one")].endswith("a.txt")


def test_a_new_file_invalidates_the_cache(corpus):
    _write(corpus, "a.txt", "paper one")
    mapper.map_all_wp_ip_file_locations()

    _write(corpus, "b.txt", "paper two")
    assert _sha("paper two") in mapper.map_all_wp_ip_file_locations()


def test_rewritten_text_invalidates_the_cache(corpus):
    """A re-OCR'd transcript changes the file's content and therefore its hash; a cache trusted on
    existence alone would keep resolving the superseded one."""
    path = _write(corpus, "a.txt", "original text")
    mapper.map_all_wp_ip_file_locations()

    path.write_text("re-ocr'd text", encoding="utf-8")
    import os
    os.utime(path, (0, 0))  # force a differing mtime regardless of filesystem resolution

    mapping = mapper.map_all_wp_ip_file_locations()
    assert _sha("re-ocr'd text") in mapping
    assert _sha("original text") not in mapping


def test_a_segmentation_change_invalidates_the_cache(corpus, monkeypatch):
    """The concrete failure: adding the character cap re-cut every long document, so the cached
    hashes described boundaries that no longer existed. Bumping SEGMENTATION_VERSION rebuilds."""
    _write(corpus, "a.txt", "paper one")
    mapper.map_all_wp_ip_file_locations()

    monkeypatch.setattr(mapper, "SEGMENTATION_VERSION", mapper.SEGMENTATION_VERSION + 1)
    rebuilt = []
    real_build = mapper._build
    monkeypatch.setattr(mapper, "_build", lambda d: rebuilt.append(d) or real_build(d))

    mapper.map_all_wp_ip_file_locations()
    assert rebuilt, "a segmentation change must force a rebuild"


def test_a_pre_fingerprint_cache_is_rebuilt_rather_than_trusted(corpus):
    """Caches written before the fingerprint existed are a bare {hash: path} dict. They carry no
    way to be validated, so they must be rebuilt once into the current shape."""
    _write(corpus, "a.txt", "paper one")
    mapper.MAP_PATH.write_text(json.dumps({"stalehash": "/gone/missing.txt"}))

    mapping = mapper.map_all_wp_ip_file_locations()

    assert "stalehash" not in mapping
    assert _sha("paper one") in mapping
    assert "locations" in json.loads(mapper.MAP_PATH.read_text())


def test_identical_text_in_two_files_keeps_the_first_and_does_not_overwrite(corpus, capsys):
    """Byte-identical papers hash to one key -- the same paper tabled at two meetings, or a
    re-extracted revision. The second used to silently replace the first, so every embedding of
    that text was attributed to whichever file happened to be walked last."""
    _write(corpus, "a.txt", "identical text")
    _write(corpus, "b.txt", "identical text")

    mapping = mapper.map_all_wp_ip_file_locations()

    assert mapping[_sha("identical text")].endswith("a.txt"), "sorted walk order, so first wins"
    assert "collision" in capsys.readouterr().out


def test_force_rebuild_bypasses_a_valid_cache(corpus, monkeypatch):
    _write(corpus, "a.txt", "paper one")
    mapper.map_all_wp_ip_file_locations()

    rebuilt = []
    real_build = mapper._build
    monkeypatch.setattr(mapper, "_build", lambda d: rebuilt.append(d) or real_build(d))
    mapper.map_all_wp_ip_file_locations(force_rebuild=True)
    assert rebuilt


def test_more_than_one_dataset_dir_is_refused(corpus):
    """Two dataset dirs means extract-documents ran twice and there is no single corpus to map."""
    (corpus.parent / "dataset-def").mkdir()
    with pytest.raises(Exception, match="More than one dataset"):
        mapper.map_all_wp_ip_file_locations()


def test_a_missing_dataset_dir_is_reported_clearly(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    processed.mkdir()
    monkeypatch.setattr(mapper, "PROCESSED_PATH", processed)
    monkeypatch.setattr(mapper, "MAP_PATH", processed / "wp_ip_file_locations.json")
    with pytest.raises(FileNotFoundError, match="No dataset"):
        mapper.map_all_wp_ip_file_locations()


def test_decoding_matches_the_embedding_pipeline(corpus):
    """The hashes are a join key with embed_all_documents, which reads utf-8 with errors="ignore".
    A file with an invalid byte must produce the same hash here as the text the embedder hashed --
    errors="replace" or a locale-dependent open() would silently orphan every one of its rows."""
    path = corpus / "bad.txt"
    path.write_bytes(b"caf\xe9 text")           # \xe9 is not valid utf-8

    expected = path.read_text(encoding="utf-8", errors="ignore")
    assert _sha(expected) in mapper.map_all_wp_ip_file_locations()
