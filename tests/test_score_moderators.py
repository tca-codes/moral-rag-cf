import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("score_moderators", Path(__file__).parents[1] / "scripts" / "score_moderators.py")
sm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sm)


def test_on_off_profile():
    keys = ["q1|10000", "q2|01100"]
    probs = {"care": [0.9, 0.1], "fairness": [0.5, 0.8], "loyalty": [0.5, 0.6],
             "authority": [0.5, 0.2], "sanctity": [0.5, 0.2]}
    r, on = sm.summarize(probs, keys)
    assert on == [0.9, 0.7]
    assert abs(r["on_profile"] - 0.8) < 1e-9
    assert abs(r["off_profile"] - (0.5 * 4 + 0.1 + 0.2 + 0.2) / 7) < 1e-9
    assert r["per_value"]["care"] == 0.9 and r["per_value"]["fairness"] == 0.8


def test_clean_output_and_profile():
    assert sm.clean_output("A: Be kind.") == "Be kind."
    assert sm.profile_of("abc|00101") == (0, 0, 1, 0, 1)
