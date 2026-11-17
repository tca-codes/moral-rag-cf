"""Retrieval for the RAG extension (course item G11).

For each (question, moral profile) input, retrieve k similar past questions
from the MIC *training* split, with their human-written replies, and give them
to the aggregator as extra context.

Design:
- Similarity is computed between question texts with a sentence embedding
  model (default all-MiniLM-L6-v2); embeddings are L2-normalised, so the dot
  product is the cosine similarity.
- At most one reply is used per retrieved question, so the k examples are k
  different situations.
- Profile-aware: among a retrieved question's replies, one whose MIC moral
  labels overlap the input's profile is preferred.
- Leakage control: the input's own question is never retrieved (this matters
  for training inputs, which come from the same split as the pool).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from mvrag.data import PAPER_VALUES

EmbedFn = Callable[[list[str]], np.ndarray]


class Retriever:
    def __init__(self, examples, embed_fn: EmbedFn):
        """examples: MIC rows to retrieve from (normally the training split)."""
        by_uid: dict[str, dict] = {}
        for ex in examples:
            if not ex.worker_answer or not ex.question:
                continue
            entry = by_uid.setdefault(ex.uid, {"uid": ex.uid, "question": ex.question, "replies": {}})
            morals = entry["replies"].setdefault(ex.worker_answer, set())
            morals.update(m for m in ex.morals if m in PAPER_VALUES)
        self.entries = sorted(by_uid.values(), key=lambda e: e["uid"])
        self.uids = [e["uid"] for e in self.entries]
        self.embed_fn = embed_fn
        self.matrix = _normalise(embed_fn([e["question"] for e in self.entries]))

    def search_many(self, queries: list[tuple[str, str, tuple]], k: int = 3,
                    candidates: int = 50) -> list[list[dict]]:
        """queries: (uid, question, profile). Returns k retrieved examples per query."""
        if not queries:
            return []
        q = _normalise(self.embed_fn([question for _, question, _ in queries]))
        sims = q @ self.matrix.T
        top_n = min(candidates, sims.shape[1])
        results = []
        for row, (uid, _, profile) in enumerate(queries):
            idx = np.argpartition(-sims[row], top_n - 1)[:top_n]
            idx = idx[np.argsort(-sims[row][idx])]
            desired = {v for v, on in zip(PAPER_VALUES, profile) if on}
            picked = []
            for i in idx:
                entry = self.entries[i]
                if entry["uid"] == uid:
                    continue  # never retrieve the input's own question
                picked.append(_pick_reply(entry, desired, float(sims[row][i])))
                if len(picked) == k:
                    break
            results.append(picked)
        return results


def _pick_reply(entry: dict, desired: set, score: float) -> dict:
    """The reply whose moral labels overlap the profile most; ties -> alphabetical."""
    replies = sorted(entry["replies"].items(), key=lambda kv: (-len(kv[1] & desired), kv[0]))
    answer, morals = replies[0]
    return {"uid": entry["uid"], "question": entry["question"], "answer": answer,
            "morals": sorted(morals), "score": round(score, 4)}


def _normalise(matrix) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def sentence_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 256) -> EmbedFn:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def embed(texts: list[str]) -> np.ndarray:
        return model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=len(texts) > 5000)

    return embed
