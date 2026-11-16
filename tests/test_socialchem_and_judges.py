import pytest

from mvrag.judges import binary_metrics
from mvrag.socialchem import build_value_examples, load_rows, parse_judgment, subsample

HEADER = ["area", "split", "rot-bad", "rot-moral-foundations", "action", "action-moral-judgment", "rot"]
ROWS = [
    ["x", "train", "0", "care-harm|fairness-cheating", "helping a friend", "2", "r"],
    ["x", "train", "0", "care-harm", "hurting  a friend", "-2", "r"],
    ["x", "train", "0", "care-harm", "hurting a friend", "-1", "r"],  # duplicate action
    ["x", "train", "0", "care-harm", 'saying "no" to a favor', "0", "r"],  # quotes in text
    ["x", "train", "1", "care-harm", "low quality row", "2", "r"],  # rot-bad dropped
    ["x", "train", "0", "care-harm", "no judgment", "", "r"],
    ["x", "train", "0", "loyalty-betrayal", "betraying your team", "-2", "r"],
    ["x", "dev", "0", "care-harm", "comforting someone", "1", "r"],
]


@pytest.fixture
def tsv(tmp_path):
    path = tmp_path / "social-chem-101.v1.0.tsv"
    path.write_text("\n".join("\t".join(r) for r in [HEADER] + ROWS), encoding="utf-8")
    return path


def test_build_value_examples_filters_and_binarizes(tsv):
    rows = load_rows(tsv)
    care = dict(build_value_examples(rows, "care", "train"))
    assert care == {"helping a friend": 1, "hurting a friend": 0, 'saying "no" to a favor': 1}
    assert dict(build_value_examples(rows, "fairness", "train")) == {"helping a friend": 1}
    assert dict(build_value_examples(rows, "loyalty", "train")) == {"betraying your team": 0}
    assert dict(build_value_examples(rows, "care", "dev")) == {"comforting someone": 1}


def test_load_rows_rejects_wrong_columns(tmp_path):
    bad = tmp_path / "bad.tsv"
    bad.write_text("a\tb\n1\t2", encoding="utf-8")
    with pytest.raises(ValueError):
        load_rows(bad)


def test_parse_judgment():
    assert parse_judgment("-2") == -2 and parse_judgment("0.0") == 0
    assert parse_judgment("") is None and parse_judgment("nan") is None and parse_judgment(None) is None


def test_subsample_is_seeded_and_capped():
    items = list(range(100))
    assert subsample(items, 10, seed=1) == subsample(items, 10, seed=1)
    assert len(subsample(items, 10)) == 10 and len(subsample(items, None)) == 100


def test_binary_metrics():
    m = binary_metrics([1, 1, 1, 0], [1, 1, 0, 0])
    assert m["accuracy"] == 0.75 and m["majority_baseline"] == 0.75 and m["positive_rate"] == 0.75
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
