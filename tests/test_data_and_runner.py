import json

import pytest

from mvrag import data
from mvrag.agent_runner import load_done, run_agents
from mvrag.agents import VALUES, build_agent_messages

ROWS = [
    {"Q": "Is it ok to lie?", "A": "Sometimes.", "rot": "It is bad to lie.",
     "moral": "fairness-cheating|care-harm", "moral-vector": "[0, 1, 0]", "chatbot": "gpt"},
    {"Q": "Is it ok  to lie?", "A": "No.", "rot": "Be honest.",
     "moral": "fairness-cheating", "chatbot": "dialoGPT"},
    {"Q": "Should I skip church?", "A": "Up to you.", "rot": "Respect faith.",
     "moral": ["sanctity-degradation"], "chatbot": "gpt"},
]


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


@pytest.fixture
def mic_dir(tmp_path):
    root = tmp_path / "MIC" / "MIC"
    root.mkdir(parents=True)
    write_jsonl(root / "train.jsonl", ROWS)
    write_jsonl(root / "dev.jsonl", ROWS[:1])
    write_jsonl(root / "test.jsonl", ROWS[:2])
    return tmp_path / "MIC"


def test_load_split_parses_fields(mic_dir):
    ex = data.load_split("train", mic_dir)
    assert len(ex) == 3
    assert ex[0].question == "Is it ok to lie?"
    assert ex[0].morals == ("fairness", "care")
    assert ex[0].moral_vector == (0, 1, 0)
    assert ex[2].morals == ("sanctity",)  # list-valued moral field works too


def test_question_id_ignores_whitespace_and_case():
    assert data.question_id("Is it ok  to lie?") == data.question_id("is it ok to lie?")


def test_unique_questions_dedupes_and_is_sorted(mic_dir):
    uq = data.unique_questions(data.load_split("train", mic_dir))
    assert len(uq) == 2  # rows 0 and 1 are the same question
    assert [u for u, _ in uq] == sorted(u for u, _ in uq)


def test_group_by_question_collects_all_replies(mic_dir):
    groups = data.group_by_question(data.load_split("train", mic_dir))
    sizes = sorted(len(v) for v in groups.values())
    assert sizes == [1, 2]


def test_dict_keyed_json_format(tmp_path):
    root = tmp_path / "MIC"
    root.mkdir()
    keyed = {str(i): r for i, r in enumerate(ROWS)}
    (root / "train.jsonl").write_text(json.dumps(keyed), encoding="utf-8")
    assert len(data.load_split("train", root)) == 3


def test_agent_messages_cover_all_values():
    for v in VALUES:
        msgs = build_agent_messages(v, "Q?")
        assert msgs[0]["role"] == "system" and msgs[1]["content"] == "Q?"
    with pytest.raises(ValueError):
        build_agent_messages("liberty", "Q?")


def test_run_agents_resumes_and_skips_done(tmp_path):
    out = tmp_path / "agents.jsonl"
    calls = []

    def fake_gen(chats):
        calls.append(len(chats))
        return [f"reply-{i}" for i in range(len(chats))]

    qs = [("a", "q1"), ("b", "q2"), ("c", "q3")]
    assert run_agents(qs[:2], fake_gen, out, questions_per_chunk=1) == 2
    assert load_done(out) == {"a", "b"}
    # second run over all three only does the missing one
    assert run_agents(qs, fake_gen, out, questions_per_chunk=2) == 1
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["uid"] for r in lines] == ["a", "b", "c"]
    assert set(lines[0]["answers"]) == set(VALUES)


def test_run_agents_tolerates_truncated_last_line(tmp_path):
    out = tmp_path / "agents.jsonl"
    out.write_text('{"uid": "a", "question": "q", "answers": {}}\n{"uid": "b", "ques')
    assert load_done(out) == {"a"}


def test_nan_fields_are_treated_as_missing(tmp_path):
    root = tmp_path / "MIC"
    root.mkdir()
    (root / "train.jsonl").write_text(
        '{"Q": "Q1?", "A": "A1", "rot": NaN, "moral": NaN, "moral-vector": NaN}', encoding="utf-8"
    )
    ex = data.load_split("train", root)[0]
    assert ex.morals == () and ex.rot == "" and ex.moral_vector is None


def test_run_agents_with_custom_builders(tmp_path):
    from mvrag.agents import build_plain_messages

    out = tmp_path / "plain.jsonl"
    seen = []

    def fake_gen(chats):
        seen.extend(chats)
        return ["r"] * len(chats)

    run_agents([("a", "q1"), ("b", "q2")], fake_gen, out, 5, None, {"plain": build_plain_messages})
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["answers"] for r in lines] == [{"plain": "r"}, {"plain": "r"}]
    assert len(seen) == 2 and "conviction" not in seen[0][0]["content"]
