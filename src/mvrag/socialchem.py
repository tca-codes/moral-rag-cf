"""Social Chemistry 101 (Forbes et al., 2020): training data for the value judges.

The CMVA-GS paper builds one classifier per moral value from SocialChem: each
annotated action has a moral judgment from -2 (very bad) to 2 (very good), which
is binarized to 0 for negative and 1 otherwise. Those classifiers are the
paper's reward models.

File: social-chem-101.v1.0.tsv (tab-separated). Columns used here:
  split                   train / dev / test / ...
  rot-bad                 1 = low-quality annotation (dropped)
  rot-moral-foundations   "|"-separated, e.g. "care-harm|fairness-cheating"
  action                  short text, e.g. "taking candy from a baby"
  action-moral-judgment   -2..2, or empty
Column names come from the dataset's README; the file itself was not available
when this was written, so `load_rows` checks them and fails loudly if they differ.
"""

from __future__ import annotations

import csv
import random
import urllib.request
import zipfile
from pathlib import Path

from mvrag.data import PAPER_VALUES

SC_URL = "https://storage.googleapis.com/ai2-mosaic-public/projects/social-chemistry/data/social-chem-101.zip"
REQUIRED_COLUMNS = ("split", "rot-bad", "rot-moral-foundations", "action", "action-moral-judgment")


def ensure_socialchem(data_dir: str | Path = "data/socialchem") -> Path:
    """Return the path of the SocialChem .tsv, downloading (~30MB zip) if needed."""
    data_dir = Path(data_dir)
    found = next(data_dir.rglob("social-chem-101*.tsv"), None) if data_dir.exists() else None
    if found is not None:
        return found
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "social-chem-101.zip"
    urllib.request.urlretrieve(SC_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        dest = data_dir.resolve()
        for member in zf.namelist():
            target = (dest / member).resolve()
            if dest != target and dest not in target.parents:
                raise ValueError(f"Unsafe path in zip: {member}")
        zf.extractall(dest)
    found = next(data_dir.rglob("social-chem-101*.tsv"), None)
    if found is None:
        raise FileNotFoundError(f"No social-chem-101*.tsv under {data_dir} after download")
    return found


def load_rows(tsv_path: str | Path) -> list[dict]:
    with open(tsv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"SocialChem file is missing columns {missing}; found {reader.fieldnames}")
        return list(reader)


def parse_judgment(raw) -> float | None:
    """'-2'..'2' -> float; empty or unparseable -> None."""
    if raw is None:
        return None
    raw = str(raw).strip()
    if not raw or raw.lower() == "nan":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def build_value_examples(rows: list[dict], value: str, split: str) -> list[tuple[str, int]]:
    """(action text, label) pairs for one moral value and one split.

    Label follows the paper: 0 if the action is judged negative, 1 otherwise.
    The same action can be annotated several times; annotations are averaged
    so each distinct action appears once.
    """
    if value not in PAPER_VALUES:
        raise ValueError(f"unknown value {value!r}")
    sums: dict[str, list[float]] = {}
    for row in rows:
        if row.get("split") != split or str(row.get("rot-bad", "")).strip() == "1":
            continue
        foundations = [f.strip().lower() for f in (row.get("rot-moral-foundations") or "").split("|")]
        if not any(f.startswith(value) for f in foundations):
            continue
        action = " ".join((row.get("action") or "").split())
        judgment = parse_judgment(row.get("action-moral-judgment"))
        if not action or judgment is None:
            continue
        sums.setdefault(action, []).append(judgment)
    return [(text, 0 if sum(js) / len(js) < 0 else 1) for text, js in sorted(sums.items())]


def subsample(examples: list, n: int | None, seed: int = 0) -> list:
    """Seeded random subset of at most n examples (all of them if n is None)."""
    examples = list(examples)
    random.Random(seed).shuffle(examples)
    return examples if n is None else examples[:n]
