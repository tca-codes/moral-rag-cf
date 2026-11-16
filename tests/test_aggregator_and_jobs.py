import json

from mvrag import data
from mvrag.aggregator import build_aggregator_messages, profile_from_morals, profile_key, profile_text
from mvrag.jobs import read_outputs, run_jobs

ANSWERS = {"care": "c", "fairness": "f", "loyalty": "l", "authority": "a", "sanctity": "s"}


def test_profile_from_morals_ignores_liberty():
    assert profile_from_morals(("loyalty", "care")) == (1, 0, 1, 0, 0)
    assert profile_from_morals(("liberty",)) == (0, 0, 0, 0, 0)
    assert profile_text((1, 0, 1, 0, 0)).startswith("care = 1, fairness = 0, loyalty = 1")
    assert profile_key("abc", (1, 0, 1, 0, 0)) == "abc|10100"


def test_aggregator_prompt_contains_all_inputs():
    user = build_aggregator_messages("Q?", ANSWERS, (0, 1, 0, 0, 0))[1]["content"]
    assert "Question: Q?" in user and "fairness = 1" in user
    assert all(f"[{v}] {a}" in user for v, a in ANSWERS.items())
    assert "Similar past" not in user


def test_aggregator_prompt_leave_one_out_and_retrieval():
    partial = {k: v for k, v in ANSWERS.items() if k != "loyalty"}
    user = build_aggregator_messages("Q?", partial, (0, 0, 1, 0, 0), [{"question": "q2", "answer": "a2"}])[1]["content"]
    assert "[loyalty]" not in user and "[care] c" in user
    assert "(1) Q: q2 A: a2" in user
    assert "(none)" in build_aggregator_messages("Q?", {}, (0, 0, 0, 0, 0))[1]["content"]


def test_run_jobs_resumes_dedupes_and_tolerates_truncation(tmp_path):
    out = tmp_path / "o.jsonl"
    calls = []

    def gen(chats):
        calls.append(len(chats))
        return [c[0]["content"].upper() for c in chats]

    jobs = [(k, [{"role": "user", "content": k}], {"uid": k}) for k in ["a", "b", "a", "c"]]
    assert run_jobs(jobs[:2], gen, out, batch_size=8) == 2
    with out.open("a") as f:
        f.write('{"key": "zz", "outp')  # simulated crash mid-write
    assert run_jobs(jobs, gen, out, batch_size=8) == 1  # only "c" is new
    recs = read_outputs(out)
    assert set(recs) == {"a", "b", "c"} and recs["c"] == {"key": "c", "uid": "c", "output": "C"}
    assert calls == [2, 1]


def test_worker_answer_loaded_and_nan_cleaned(tmp_path):
    root = tmp_path / "MIC"
    root.mkdir()
    rows = [{"Q": "Q1?", "A": "A1", "worker_answer": "W1", "moral": "care"},
            {"Q": "Q2?", "A": "A2", "worker_answer": "nan", "moral": "care"}]
    (root / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    ex = data.load_split("train", root)
    assert ex[0].worker_answer == "W1" and ex[1].worker_answer == ""
