import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("hub", Path(__file__).parents[1] / "scripts" / "hub.py")
hub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hub)


class FakeApi:
    """Stores uploads in memory; can be told to 'lose' a file to test the check."""

    def __init__(self, lose=None):
        self.store, self.lose = {}, lose

    def upload_folder(self, folder_path, path_in_repo, **_):
        for p in Path(folder_path).rglob("*"):
            if p.is_file():
                self._put(f"{path_in_repo}/{p.relative_to(folder_path).as_posix()}", p.stat().st_size)

    def upload_file(self, path_or_fileobj, path_in_repo, **_):
        self._put(path_in_repo, Path(path_or_fileobj).stat().st_size)

    def _put(self, path, size):
        if path != self.lose:
            self.store[path] = size


def fake_remote(api, repo, rel):
    return {p: s for p, s in api.store.items() if p == rel or p.startswith(rel + "/")}


def test_push_folder_and_file_verified(tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "remote_files", fake_remote)
    (tmp_path / "outputs" / "adapters" / "a8k" / "care").mkdir(parents=True)
    (tmp_path / "outputs" / "adapters" / "a8k" / "care" / "w.bin").write_bytes(b"x" * 10)
    (tmp_path / "outputs" / "r.jsonl").write_text("1\n2\n")
    api = FakeApi()
    assert hub.push(api, "u/r", tmp_path, "outputs/adapters/a8k")
    assert hub.push(api, "u/r", tmp_path, "outputs/r.jsonl")
    assert api.store == {"outputs/adapters/a8k/care/w.bin": 10, "outputs/r.jsonl": 4}


def test_push_reports_problem_when_hub_is_missing_a_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(hub, "remote_files", fake_remote)
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "a").write_text("a")
    (tmp_path / "d" / "b").write_text("bb")
    assert not hub.push(FakeApi(lose="d/b"), "u/r", tmp_path, "d")
    assert "PROBLEM" in capsys.readouterr().out


def test_check_detects_size_mismatch():
    assert hub.check({"a": 1, "b": 2}, {"a": 1, "b": 3}) == ["b"]
    assert hub.check({"a": 1}, {"a": 1, "extra": 5}) == []
