"""Map every embedding segment hash back to the working/information paper file it came from.

The embedding store is keyed on ``sha256`` of a *segment* of a document's text, so recovering
"which paper is this row" means rebuilding the same segmentation over the extracted corpus and
hashing it the same way. That is expensive enough to cache, and the cache is what this module owns.
"""
import hashlib
import json
import pathlib

import embeddings.document_embeddings as document_embeddings

PROCESSED_PATH = pathlib.Path("data/antarctic-db/processed")
MAP_PATH = PROCESSED_PATH / "wp_ip_file_locations.json"

# Bumped whenever the segmentation changes shape. The cache stores the hashes `split_long_document`
# produced at the time it was written, so a change to splitting silently invalidates every multi-
# segment entry -- which is exactly what happened when the character cap was added: the map kept
# pointing at boundaries that no longer exist, and consumers hit KeyError on hashes the store had
# but the map had never seen. Bumping this forces a rebuild rather than leaving that to be noticed
# downstream.
SEGMENTATION_VERSION = 2


def _dataset_dir(processed_path: pathlib.Path) -> pathlib.Path:
    dataset_dirs = sorted(p for p in processed_path.iterdir() if "dataset" in str(p))
    if not dataset_dirs:
        raise FileNotFoundError(f"No dataset-* directory under {processed_path}. Run extract-documents.")
    if len(dataset_dirs) > 1:
        raise Exception(
            f"More than one dataset dir under {processed_path}: {[str(p) for p in dataset_dirs]}. "
            "extract-documents was probably run more than once. Delete them and try again."
        )
    return dataset_dirs[0]


def _fingerprint(dataset_dir: pathlib.Path) -> dict:
    """Cheap description of the corpus, to tell a stale cache from a current one.

    Name, size and mtime of every ``.txt``, without reading any of them -- re-reading the corpus is
    the cost the cache exists to avoid, so the check has to be cheaper than the rebuild. This
    catches a re-extract, an added or removed paper, and a rewritten OCR transcript; it will not
    catch an edit that preserves both size and mtime, which no step in this pipeline performs.
    """
    files = sorted(dataset_dir.rglob("*.txt"))
    return {
        "segmentation_version": SEGMENTATION_VERSION,
        "dataset_dir": str(dataset_dir),
        "n_files": len(files),
        "files": [[str(p.relative_to(dataset_dir)), p.stat().st_size, int(p.stat().st_mtime)]
                  for p in files],
    }


def _build(dataset_dir: pathlib.Path) -> tuple[dict, list]:
    """Hash every segment of every paper. Returns (map, collisions)."""
    location_map = {}
    collisions = []
    for path in sorted(dataset_dir.rglob("*.txt")):
        # utf-8 with errors="ignore", byte for byte what embed_all_documents uses to produce the
        # text it hashes. These hashes are the join key between the two, so any difference in
        # decoding -- a locale-dependent open(), or errors="replace" -- yields different text for
        # any file with invalid utf-8 and orphans every one of its rows.
        text = path.read_text(encoding="utf-8", errors="ignore")
        for segment in document_embeddings.split_long_document(text):
            digest = hashlib.sha256(segment.encode()).hexdigest()
            # Two files whose text is byte-identical hash to one key, so the second silently
            # replaced the first and every embedding of that text was attributed to whichever file
            # happened to be walked last. Duplicates are real here -- the same paper is tabled at
            # more than one meeting, and revisions are re-extracted -- so this keeps the first path
            # (walk order is sorted, hence stable) and reports the rest rather than overwriting
            # blind.
            if digest in location_map:
                if location_map[digest] != str(path):
                    collisions.append((digest, location_map[digest], str(path)))
                continue
            location_map[digest] = str(path)
    return location_map, collisions


def map_all_wp_ip_file_locations(force_rebuild: bool = False) -> dict:
    """``{segment sha256: source file path}`` for the extracted corpus, cached on disk.

    The cache is validated against a fingerprint of the corpus rather than trusted on existence
    alone, so re-extracting the dataset or changing the segmentation rebuilds it instead of
    returning a map that silently no longer describes the store.
    """
    dataset_dir = _dataset_dir(PROCESSED_PATH)

    if MAP_PATH.exists() and not force_rebuild:
        try:
            cached = json.loads(MAP_PATH.read_text())
        except json.JSONDecodeError:
            cached = None
        # Pre-fingerprint caches are a bare {hash: path} dict with no "locations" key. They cannot
        # be validated, so they are rebuilt once and written in the current shape.
        if isinstance(cached, dict) and "locations" in cached:
            if cached.get("fingerprint") == _fingerprint(dataset_dir):
                return cached["locations"]
            print(f"  {MAP_PATH.name} does not match the current corpus — rebuilding.")

    location_map, collisions = _build(dataset_dir)
    if collisions:
        print(f"  warning: {len(collisions)} segment hash collision(s) — byte-identical text in "
              f"more than one file; keeping the first path in sorted order. "
              f"First: {collisions[0][1]} == {collisions[0][2]}")

    MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    MAP_PATH.write_text(json.dumps({
        "fingerprint": _fingerprint(dataset_dir),
        "locations": location_map,
    }))
    return location_map


if __name__ == "__main__":
    mapping = map_all_wp_ip_file_locations()
    print(f"{len(mapping)} segment hashes over {len(set(mapping.values()))} files")
