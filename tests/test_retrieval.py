import numpy as np

from mvrag.agg_train import build_training_examples
from mvrag.aggregator import profile_key
from mvrag.data import Example
from mvrag.retrieval import Retriever

VOCAB = ["lie", "friend", "church", "money", "boss", "family"]


def bag_of_words(texts):
    return np.array([[t.lower().count(w) + 0.01 for w in VOCAB] for t in texts], dtype=np.float32)


def ex(uid, row, q, a, morals):
    return Example(uid=uid, row=row, question=q, answer="chat", rot="", morals=morals, moral_vector=None,
                   a_agrees=None, rot_agree=None, severity=None, chatbot="", worker_answer=a)


POOL = [
    ex("t1", 0, "Should I lie to my friend?", "Be honest with friends.", ("fairness",)),
    ex("t1", 1, "Should I lie to my friend?", "Stand by your friend.", ("loyalty",)),
    ex("t2", 2, "Is it ok to lie to family?", "Family deserves the truth.", ("care",)),
    ex("t3", 3, "Should I skip church?", "Respect your faith.", ("sanctity",)),
    ex("t4", 4, "Can I take money from my boss?", "No, that is stealing.", ("fairness",)),
    ex("t5", 5, "Empty reply question about a friend lie", "", ("care",)),
]


def test_retriever_ranks_by_similarity_one_reply_per_question():
    r = Retriever(POOL, bag_of_words)
    assert len(r.entries) == 4  # t5 has no reply, t1's two rows are one question
    res = r.search_many([("q", "Is it wrong to lie to a friend?", (0, 0, 0, 0, 0))], k=2)[0]
    assert [x["uid"] for x in res] == ["t1", "t2"]
    assert res[0]["score"] >= res[1]["score"]


def test_retriever_prefers_profile_reply_and_excludes_own_question():
    r = Retriever(POOL, bag_of_words)
    loyal = r.search_many([("q", "lie to my friend", (0, 0, 1, 0, 0))], k=1)[0][0]
    fair = r.search_many([("q", "lie to my friend", (0, 1, 0, 0, 0))], k=1)[0][0]
    assert loyal["answer"] == "Stand by your friend." and fair["answer"] == "Be honest with friends."
    own = r.search_many([("t1", "Should I lie to my friend?", (0, 1, 0, 0, 0))], k=3)[0]
    assert "t1" not in [x["uid"] for x in own] and len(own) == 3


def test_training_examples_include_retrieved_context():
    rows = [ex("u1", 0, "Q1?", "target one", ("care",))]
    agents = {"u1": {"uid": "u1", "answers": {"care": "c", "fairness": "f", "loyalty": "l", "authority": "a", "sanctity": "s"}}}
    retrieved = {profile_key("u1", (1, 0, 0, 0, 0)): {"retrieved": [{"question": "past q", "answer": "past a"}]}}
    plain = build_training_examples(rows, agents)[0]["messages"][1]["content"]
    rag = build_training_examples(rows, agents, retrieved=retrieved)[0]["messages"][1]["content"]
    assert "Similar past" not in plain and "(1) Q: past q A: past a" in rag
    try:
        build_training_examples(rows, agents, retrieved={})
        assert False, "missing retrieval should raise"
    except ValueError:
        pass
