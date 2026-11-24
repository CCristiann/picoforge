"""manifest.py — provenance and resume record of a quantization run.

The identity (pinned sources, recipe, library hash) is fixed when the run
starts; reopening the run with a different identity is refused, so one output
directory never mixes two recipes. Each finished file is recorded with its
sha256: on resume a file counts as done only if it still hashes the same.
"""

import json
from pathlib import Path

from imatrix import file_sha256


class Manifest:
    def __init__(self, path: Path, data: dict):
        self.path, self.data = Path(path), data

    @classmethod
    def load_or_create(cls, path: Path, identity: dict) -> "Manifest":
        path = Path(path)
        if path.exists():
            data = json.loads(path.read_text())
            if data["identity"] != identity:
                raise ValueError(f"{path}: provenance mismatch: {data['identity']} != {identity}")
            return cls(path, data)
        m = cls(path, {"identity": identity, "files": {}})
        m._save()
        return m

    def done(self, name: str, path: Path) -> bool:
        entry = self.data["files"].get(name)
        return bool(entry) and Path(path).is_file() and file_sha256(path) == entry["sha256"]

    def record(self, name: str, path: Path, **info) -> None:
        self.data["files"][name] = {"sha256": file_sha256(path), "bytes": Path(path).stat().st_size, **info}
        self._save()

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2) + "\n")
        tmp.replace(self.path)
