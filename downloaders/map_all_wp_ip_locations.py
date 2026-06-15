import pathlib
import json
import hashlib
import embeddings.document_embeddings as document_embeddings

def map_all_wp_ip_file_locations():
    processed_path = pathlib.Path("data/antarctic-db/processed")
    map_path = processed_path / "wp_ip_file_locations.json"

    if map_path.exists():
        with open(map_path, "r") as f:
            return json.load(f)
    
    dataset_dirs = list(p for p in processed_path.iterdir() if "dataset" in str(p))

    if len(dataset_dirs) > 1:
        raise Exception("More than one dataset_dir. extract-documents was probably run more than once. Delete them and try again.")

    location_map = {}

    dataset_dir = dataset_dirs[0]
    for p in dataset_dir.rglob("*.txt"):
        with open(p, "r") as f:
            text = f.read()
            for segment in document_embeddings.split_long_document(text):
                location_map[hashlib.sha256(segment.encode()).hexdigest()] = str(p)
    
    with open(map_path, "w") as f:
        json.dump(location_map, f)

if __name__ == "__main__":
    map_all_wp_ip_file_locations()