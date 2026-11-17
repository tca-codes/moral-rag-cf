from mvrag.counterfactual import flips, make_variants, variant_family

ANSWERS = {"care": "c", "fairness": "f", "loyalty": "l", "authority": "a", "sanctity": "s"}
RET = [{"question": "q1", "answer": "a1"}, {"question": "q2", "answer": "a2"}]
POOL = [{"question": "q1", "answer": "a1"}, {"question": "other", "answer": "x"}]


def test_variants_without_retrieval():
    v = make_variants(ANSWERS, None)
    assert set(v) == {"full"} | {f"drop:{x}" for x in ANSWERS}
    assert "loyalty" not in v["drop:loyalty"][0] and len(v["drop:loyalty"][0]) == 4


def test_variants_with_retrieval_and_swaps_are_deterministic():
    v = make_variants(ANSWERS, RET, POOL, "key1")
    assert {"no_ret", "drop_ret:1", "drop_ret:2", "swap_ret:1", "swap_ret:2"} <= set(v)
    assert v["no_ret"][1] == [] and v["drop_ret:1"][1] == [RET[1]]
    assert v["swap_ret:2"][1] == [RET[0], {"question": "other", "answer": "x"}]  # never swaps in one of its own
    assert make_variants(ANSWERS, RET, POOL, "key1") == v
    assert v["full"][0] == ANSWERS and v["full"][1] == RET


def test_family_and_flip():
    assert variant_family("drop_ret:2") == "drop_ret" and variant_family("full") == "full"
    assert flips(0.7, 0.3) and not flips(0.7, 0.6) and flips(0.49, 0.5)
