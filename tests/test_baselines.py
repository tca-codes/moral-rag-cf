from mvrag.baselines import build_fewshot_messages, example_pool, pick_examples
from mvrag.data import Example


def ex(uid, row, q, a, morals):
    return Example(uid=uid, row=row, question=q, answer="chat", rot="", morals=morals, moral_vector=None,
                   a_agrees=None, rot_agree=None, severity=None, chatbot="", worker_answer=a)


ROWS = [ex(f"u{i}", i, f"question {i}?", f"answer {i}", ("care",) if i % 2 else ("care", "loyalty")) for i in range(20)]
ROWS.append(ex("u99", 99, "no answer?", "", ("care",)))


def test_pool_and_examples_exclude_own_question_and_are_deterministic():
    pool = example_pool(ROWS)
    assert len(pool["care"]) == 20 and len(pool["loyalty"]) == 10 and pool["fairness"] == []
    picks = pick_examples(pool, "care", "u3", 5, "u3")
    assert len(picks) == 5 and ("question 3?", "answer 3") not in picks
    assert picks == pick_examples(pool, "care", "u3", 5, "u3")
    assert pick_examples(pool, "fairness", "u3", 5, "u3") == []


def test_fewshot_prompt_lists_only_desired_values():
    pool = example_pool(ROWS)
    msgs = build_fewshot_messages("Is it ok?", (1, 0, 1, 0, 0), pool, "u0")
    system, user = msgs[0]["content"], msgs[1]["content"]
    assert "- care:" in system and "- loyalty:" in system and "fairness" not in system
    assert user.count("Q: ") == 5 + 5 + 1 and user.endswith("Q: Is it ok?\nA:")
    assert "question 0?" not in user
