import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("rs", Path(__file__).parents[1] / "scripts" / "rescore_candidates.py")
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)


def test_rescore_replaces_scores_and_keeps_candidates(tmp_path):
    path = tmp_path / "fairness.jsonl"
    recs = [{"key": u, "uid": u, "question": "q", "value": "fairness", "candidates": ["good a", "bad b", ""],
             "scores": [0.56, 0.56, 0.56]} for u in ["b", "a"]]
    path.write_text("\n".join(json.dumps(r) for r in recs) + '\n{"key": "c", "trunc')
    n = rs.rescore_file(path, lambda texts: [0.9 if t.startswith("good") else 0.1 for t in texts], batch=2)
    assert n == 2
    out = [json.loads(l) for l in path.read_text().splitlines()]
    assert [r["uid"] for r in out] == ["a", "b"]
    assert out[0]["scores"] == [0.9, 0.1, 0.1] and out[0]["candidates"] == ["good a", "bad b", ""]
    assert not path.with_suffix(".jsonl.tmp").exists()
