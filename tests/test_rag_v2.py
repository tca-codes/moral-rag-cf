import importlib.util
from pathlib import Path
from types import SimpleNamespace

from mvrag.aggregator import build_aggregator_messages

spec = importlib.util.spec_from_file_location("add_rots", Path(__file__).parents[1] / "scripts" / "add_rots.py")
ar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ar)


def row(uid, r, rot, morals, ans="Be kind."):
    return SimpleNamespace(uid=uid, row=r, rot=rot, morals=morals, worker_answer=ans)


def test_rot_is_added_and_matches_profile():
    index = ar.rot_index([row("q1", 0, "It's good to obey rules", ["authority"]),
                          row("q1", 1, "You should care for others", ["care", "liberty"])])
    rec = {"key": "x|10000", "profile": [1, 0, 0, 0, 0],
           "retrieved": [{"uid": "q1", "question": "Q?", "answer": "Be kind."},
                         {"uid": "q9", "question": "Other?", "answer": "No."}]}
    out, missing = ar.add_rots(rec, index)
    assert out["retrieved"][0]["rot"] == "You should care for others"
    assert out["retrieved"][0]["rot_morals"] == ["care"]
    assert "rot" not in out["retrieved"][1] and missing == 1
    assert out["retrieved"][0]["question"] == "Q?"  # retrieval itself unchanged


def test_tie_keeps_first_row():
    index = ar.rot_index([row("q1", 1, "second", ["care"]), row("q1", 0, "first", ["care"])])
    assert ar.pick_rot(index[("q1", "Be kind.")], [0, 1, 0, 0, 0])[0] == "first"


def test_prompt_shows_rule_only_when_present():
    answers = {"care": "a", "fairness": "b", "loyalty": "c", "authority": "d", "sanctity": "e"}
    v1 = build_aggregator_messages("Q", answers, (1, 0, 0, 0, 0), [{"question": "P", "answer": "R"}])
    v2 = build_aggregator_messages("Q", answers, (1, 0, 0, 0, 0),
                                   [{"question": "P", "answer": "R", "rot": "Be honest", "rot_morals": ["care"]}])
    assert "Rule of thumb" not in v1[1]["content"]
    assert "(1) Q: P A: R Rule of thumb: Be honest (care)" in v2[1]["content"]
