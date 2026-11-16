"""Loading the Moral Integrity Corpus (MIC, Ziems et al. 2022).

MIC is hosted at https://huggingface.co/datasets/SALT-NLP/MIC as MIC.zip with
MIC/train.jsonl, MIC/dev.jsonl and MIC/test.jsonl. Each row is one annotation
(a prompt-reply pair plus one Rule of Thumb), so the same question appears in
several rows.

The exact on-disk schema was NOT verified when this file was written. Field
names below come from the dataset's loading script (Q, A, rot, moral,
moral-vector, A_agrees, rot-agree, violation-severity, chatbot, ...). Run
`python scripts/inspect_mic.py` first and adjust FIELD_ALIASES if needed.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
import zipfile
from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MIC_ZIP_URL = "https://huggingface.co/datasets/SALT-NLP/MIC/resolve/main/MIC.zip"
SPLIT_FILES = {"train": "train.jsonl", "val": "dev.jsonl", "test": "test.jsonl"}

# Moral foundations. The CMVA-GS paper uses the first five; MIC also has liberty.
PAPER_VALUES = ("care", "fairness", "loyalty", "authority", "sanctity")
ALL_FOUNDATIONS = PAPER_VALUES + ("liberty",)

# Canonical name -> candidate keys in a raw record (first match wins).
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "question": ("Q", "question", "prompt"),
    "answer": ("A", "answer", "reply"),
    "rot": ("rot", "RoT"),
    "moral": ("moral", "morals", "moral_foundations"),
    "moral_vector": ("moral-vector", "moral_vector"),
    "a_agrees": ("A_agrees", "a_agrees"),
    "rot_agree": ("rot-agree", "rot_agree"),
    "severity": ("violation-severity", "violation_severity"),
    "chatbot": ("chatbot",),
}


@dataclass(frozen=True)
class Example:
    uid: str  # stable id of the question text (shared by all rows of a question)
    row: int  # row index within its split file
    question: str
    answer: str
    rot: str
    morals: tuple[str, ...]  # base foundation names, e.g. ("care", "fairness")
    moral_vector: tuple | None
    a_agrees: Any
    rot_agree: Any
    severity: Any
    chatbot: str


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def question_id(question: str) -> str:
    """Stable 12-char id for a question, independent of file order."""
    return hashlib.sha1(normalize_text(question).lower().encode("utf-8")).hexdigest()[:12]


def parse_morals(raw: Any) -> tuple[str, ...]:
    """Map raw moral labels (e.g. 'care-harm|Fairness') to base foundation names."""
    if raw is None or isinstance(raw, (int, float)):
        return ()
    if isinstance(raw, str):
        tokens = re.split(r"[|,;]", raw)
    else:
        tokens = [str(t) for t in raw]
    found: list[str] = []
    for tok in tokens:
        tok = tok.strip().lower()
        for name in ALL_FOUNDATIONS:
            if tok.startswith(name) and name not in found:
                found.append(name)
    return tuple(found)


def _parse_vector(raw: Any) -> tuple | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = literal_eval(raw)
        except (ValueError, SyntaxError):
            return None
    try:
        return tuple(raw)
    except TypeError:
        return None


def _is_missing(value: Any) -> bool:
    """None or NaN (MIC stores missing values as NaN)."""
    return value is None or (isinstance(value, float) and value != value)


def _get(rec: dict, field: str, default: Any = None) -> Any:
    for key in FIELD_ALIASES[field]:
        if key in rec and not _is_missing(rec[key]):
            return rec[key]
    return default


def read_records(path: Path) -> list[dict]:
    """Read a .jsonl file, or a whole-file JSON list / dict keyed by row id."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    if isinstance(obj, dict):
        nested = [v for v in obj.values() if isinstance(v, dict)]
        return nested if nested else [obj]  # dict of records, or a single record
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def to_example(rec: dict, row: int) -> Example:
    question = normalize_text(_get(rec, "question", ""))
    return Example(
        uid=question_id(question),
        row=row,
        question=question,
        answer=normalize_text(_get(rec, "answer", "")),
        rot=normalize_text(_get(rec, "rot", "")),
        morals=parse_morals(_get(rec, "moral")),
        moral_vector=_parse_vector(_get(rec, "moral_vector")),
        a_agrees=_get(rec, "a_agrees"),
        rot_agree=_get(rec, "rot_agree"),
        severity=_get(rec, "severity"),
        chatbot=str(_get(rec, "chatbot", "")),
    )


def _safe_extract(zip_path: Path, dest: Path) -> None:
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            target = (dest / member).resolve()
            if dest != target and dest not in target.parents:
                raise ValueError(f"Unsafe path in zip: {member}")
        zf.extractall(dest)


def _download_zip(dest_zip: Path) -> Path:
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            "SALT-NLP/MIC", "MIC.zip", repo_type="dataset", local_dir=str(dest_zip.parent)
        )
        return Path(path)
    except Exception:  # fall back to a plain HTTP download
        urllib.request.urlretrieve(MIC_ZIP_URL, dest_zip)
        return dest_zip


def ensure_mic(data_dir: str | Path = "data/MIC") -> Path:
    """Return the directory holding train/dev/test .jsonl, downloading if needed.

    If you already have the files, put them anywhere under `data_dir`.
    """
    data_dir = Path(data_dir)
    found = next(data_dir.rglob(SPLIT_FILES["train"]), None) if data_dir.exists() else None
    if found is not None:
        return found.parent
    zip_path = _download_zip(data_dir / "MIC.zip")
    _safe_extract(zip_path, data_dir)
    found = next(data_dir.rglob(SPLIT_FILES["train"]), None)
    if found is None:
        raise FileNotFoundError(f"train.jsonl not found under {data_dir} after download")
    return found.parent


def load_split(split: str, data_dir: str | Path = "data/MIC") -> list[Example]:
    if split not in SPLIT_FILES:
        raise ValueError(f"split must be one of {sorted(SPLIT_FILES)}")
    root = ensure_mic(data_dir)
    records = read_records(root / SPLIT_FILES[split])
    return [to_example(rec, i) for i, rec in enumerate(records)]


def group_by_question(examples: list[Example]) -> dict[str, list[Example]]:
    groups: dict[str, list[Example]] = {}
    for ex in examples:
        groups.setdefault(ex.uid, []).append(ex)
    return groups


def unique_questions(examples: list[Example]) -> list[tuple[str, str]]:
    """(uid, question) pairs, one per distinct question, sorted by uid.

    Sorting by uid makes any "first n" subset reproducible regardless of file order.
    """
    seen: dict[str, str] = {}
    for ex in examples:
        seen.setdefault(ex.uid, ex.question)
    return sorted(seen.items())
