"""The object store, both backends (TS §3)."""

from __future__ import annotations

import pytest

from xlforecast.storage.objects import LocalObjectStore, MemoryObjectStore, ObjectNotFoundError


@pytest.fixture(params=["memory", "local"])
def store(request, tmp_path):
    return MemoryObjectStore() if request.param == "memory" else LocalObjectStore(tmp_path)


class TestBothBackendsBehaveIdentically:
    """The API and worker are written against the Protocol, so a difference here would show
    up as an environment-dependent bug rather than a test failure."""

    def test_put_then_get(self, store):
        store.put("a/b.bin", b"payload")
        assert store.get("a/b.bin") == b"payload"

    def test_exists(self, store):
        assert not store.exists("a/b.bin")
        store.put("a/b.bin", b"x")
        assert store.exists("a/b.bin")

    def test_overwrite(self, store):
        store.put("k", b"one")
        store.put("k", b"two")
        assert store.get("k") == b"two"

    def test_delete(self, store):
        store.put("k", b"x")
        store.delete("k")
        assert not store.exists("k")

    def test_deleting_an_absent_key_is_a_no_op(self, store):
        store.delete("never-existed")

    def test_missing_key_raises_rather_than_returning_empty(self, store):
        """Returning b"" would turn a lost checkpoint into a corrupt one."""
        with pytest.raises(ObjectNotFoundError) as exc:
            store.get("absent")
        assert exc.value.fix

    def test_list_prefix_is_sorted_and_scoped(self, store):
        for key in ("jobs/a/1.json", "jobs/a/2.json", "jobs/b/1.json", "other/x"):
            store.put(key, b"x")
        assert store.list_prefix("jobs/a/") == ["jobs/a/1.json", "jobs/a/2.json"]

    def test_list_prefix_of_nothing_is_empty(self, store):
        assert store.list_prefix("nothing/") == []

    def test_binary_payloads_round_trip_unchanged(self, store):
        payload = bytes(range(256))
        store.put("bin", payload)
        assert store.get("bin") == payload


class TestLocalStoreSpecifics:
    def test_nested_keys_create_directories(self, tmp_path):
        LocalObjectStore(tmp_path).put("a/b/c/d.json", b"x")
        assert (tmp_path / "a" / "b" / "c" / "d.json").exists()

    def test_writes_leave_no_partial_files_behind(self, tmp_path):
        """Write-then-rename: a reader must never observe a half-written checkpoint, because
        a torn one deserialises into nonsense where a missing one simply re-runs the fold."""
        store = LocalObjectStore(tmp_path)
        store.put("a/b.json", b"[]")
        assert list(tmp_path.rglob("*.partial")) == []

    def test_traversal_in_a_key_cannot_escape_the_root(self, tmp_path):
        """Job ids arrive from requests."""
        root = tmp_path / "root"
        LocalObjectStore(root).put("../../../etc/whatever", b"x")
        assert (root / "etc" / "whatever").exists()
        assert not (tmp_path / "etc").exists()

    def test_an_empty_key_is_rejected(self, tmp_path):
        with pytest.raises(ObjectNotFoundError):
            LocalObjectStore(tmp_path).put("///", b"x")

    def test_the_root_is_created_if_absent(self, tmp_path):
        target = tmp_path / "does" / "not" / "exist"
        LocalObjectStore(target)
        assert target.exists()
