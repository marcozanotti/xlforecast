"""Object storage (TS §1: inputs, results, artifacts as Parquet).

A Protocol with a filesystem implementation and an in-memory one. S3/Azure implement the same
four methods; nothing above this layer knows which is in use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from xlforecast.errors import XLForecastError

__all__ = ["LocalObjectStore", "MemoryObjectStore", "ObjectNotFoundError", "ObjectStore"]


class ObjectNotFoundError(XLForecastError):
    """Requested key is absent from the store."""


class ObjectStore(Protocol):
    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def delete(self, key: str) -> None: ...
    def list_prefix(self, prefix: str) -> list[str]: ...


class MemoryObjectStore:
    """For tests. Deliberately not thread-safe -- if a test needs that, it is testing the
    wrong thing."""

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}

    def put(self, key: str, data: bytes) -> None:
        self._data[key] = data

    def get(self, key: str) -> bytes:
        try:
            return self._data[key]
        except KeyError as exc:
            raise ObjectNotFoundError(f"no object at '{key}'", fix="Check the key.") from exc

    def exists(self, key: str) -> bool:
        return key in self._data

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def list_prefix(self, prefix: str) -> list[str]:
        return sorted(k for k in self._data if k.startswith(prefix))


class LocalObjectStore:
    """Filesystem-backed, for local development and the self-hosted deployment mode (TS §12)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Keys are slash-separated and must not escape the root: a job id arrives from a
        # request, and '../' in one would otherwise write outside the store.
        parts = [p for p in key.split("/") if p not in ("", ".", "..")]
        if not parts:
            raise ObjectNotFoundError(f"invalid key '{key}'", fix="Use a non-empty key.")
        return self.root.joinpath(*parts)

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename, so a reader never observes a half-written checkpoint. FR-801
        # depends on this: a torn checkpoint is worse than a missing one, because a missing
        # one simply re-runs the fold.
        temp = path.with_suffix(path.suffix + ".partial")
        temp.write_bytes(data)
        temp.replace(path)

    def get(self, key: str) -> bytes:
        path = self._path(key)
        if not path.exists():
            raise ObjectNotFoundError(f"no object at '{key}'", fix="Check the key.")
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            path.unlink()

    def list_prefix(self, prefix: str) -> list[str]:
        root = self.root
        return sorted(
            str(p.relative_to(root)).replace("\\", "/")
            for p in root.rglob("*")
            if p.is_file() and str(p.relative_to(root)).replace("\\", "/").startswith(prefix)
        )
