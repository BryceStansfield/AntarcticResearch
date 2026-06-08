import json
import pathlib

_secrets: dict = {}

def _load() -> dict:
    global _secrets
    if not _secrets:
        path = pathlib.Path(__file__).parent / "secrets.json"
        with open(path) as f:
            _secrets = json.load(f)
    return _secrets

def get(key: str) -> str:
    return _load()[key]
