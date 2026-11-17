import json

from mvrag.agent_runner import run_agents_per_value
from mvrag.agents import VALUES
from mvrag.reward import (build_agent_training_examples, load_selections, reward_summary,
                          sample_and_score, select_best)


def test_select_best_skips_empty_and_breaks_ties_first():
    assert select_best(["a", "b", "c"], [0.2, 0.9, 0.9]) == ("b", 0.9)
    assert select_best(["", "b"], [0.99, 0.1]) == ("b", 0.1)
    assert select_best(["", " "], [0.5, 0.5]) == ("", 0.0)


def test_sample_and_score_resumes_and_selects(tmp_path):
    out = tmp_path / "care.jsonl"
    calls = []

    def sample_fn(chats):
        calls.append(len(chats))
        return [[f"bad {i}", f"good {i}", ""] for i in range(len(chats))]

    def score_fn(texts):
        return [0.9 if t.startswith("good") else 0.1 for t in texts]

    qs = [("a", "q1"), ("b", "q2"), ("c", "q3")]
    assert sample_and_score(qs[:2], "care", sample_fn, score_fn, out, batch_size=2) == 2
    assert sample_and_score(qs, "care", sample_fn, score_fn, out, batch_size=2) == 1
    assert calls == [2, 1]
    sel = load_selections(out)
    assert [s["uid"] for s in sel] == ["a", "b", "c"]
    assert all(s["best"].startswith("good") and s["best_score"] == 0.9 for s in sel)
    summary = reward_summary(sel)
    assert summary["n"] == 3 and summary["mean_best_reward"] > summary["mean_candidate_reward"]
    ex = build_agent_training_examples(sel, "care")
    assert ex[0]["target"] == sel[0]["best"] and "care" in ex[0]["messages"][0]["content"]
    assert ex[0]["messages"][1]["content"] == "q1"


def test_run_agents_per_value_format_and_resume(tmp_path):
    out = tmp_path / "agents.jsonl"
    seen = []

    def gen(value, chats):
        seen.append((value, len(chats)))
        return [f"{value}:{c[1]['content']}" for c in chats]

    qs = [("a", "q1"), ("b", "q2"), ("c", "q3")]
    assert run_agents_per_value(qs[:1], gen, out, questions_per_chunk=2) == 1
    assert run_agents_per_value(qs, gen, out, questions_per_chunk=2) == 2
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["uid"] for r in recs] == ["a", "b", "c"]
    assert recs[1]["answers"] == {v: f"{v}:q2" for v in VALUES}
    assert seen[:5] == [(v, 1) for v in VALUES] and seen[5:] == [(v, 2) for v in VALUES]


def test_select_first_questions_matches_sampling_order():
    from mvrag.reward import select_first_questions

    sel = [{"uid": u} for u in ["c", "a", "d", "b"]]
    assert [s["uid"] for s in select_first_questions(sel, 2)] == ["a", "b"]
