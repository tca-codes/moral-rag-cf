# Moral-value RAG + counterfactual explanations

Course final project (NLP): a reduced-scale re-implementation of the CMVA-GS
baseline ([Dognin et al., 2024](https://arxiv.org/abs/2403.12805)) on the Moral
Integrity Corpus, extended with **retrieval-augmented generation (G11)** and
**counterfactual explanations (G15)**.

> Status: Week 1 scaffold. Data loading and the five prompt-based moral agents
> work; the aggregator, retrieval, and counterfactual modules are not built yet.

## Idea

The baseline answers a question with five "moral agents" (care, fairness,
loyalty, authority, sanctity) and an aggregator that blends their answers
according to a user's moral profile. This project asks:

1. Does giving the aggregator retrieved, similar MIC examples improve alignment?
2. Which inputs actually drive the output? Remove or swap each agent answer and
   each retrieved passage, then measure how the response and its value-alignment
   scores change.

## Layout

```
src/mvrag/
  data.py          MIC download + schema-tolerant loading
  agents.py        value-specific prompts for the five agents
  generate.py      greedy HF generation (4-bit, fp16 for Colab T4)
  agent_runner.py  resumable agent runs (JSONL, skips finished questions)
  socialchem.py    Social Chemistry 101 loading for the value judges
  judges.py        train/load/score the five value judges (reward models)
  aggregator.py    the contextual aggregator (moderator) prompt and moral profile
  jobs.py          resumable batched generation
  agg_train.py     LoRA training of the aggregator (the paper's CMVA-GS)
scripts/
  inspect_mic.py   print MIC's real schema (run first)
  run_agents.py    generate five agent answers for N questions
  eval_rouge.py    first sanity ROUGE numbers
  train_judges.py  train the five judges on Social Chemistry
  score_agents.py  5x5 table: each advisor scored by each judge
  table1.py        plain model vs moral agents (paper's Table 1, small scale)
  run_aggregator.py   run the prompt-only aggregator over a MIC split
  train_aggregator.py train the aggregator with LoRA (resumable)
  eval_aggregator.py  ROUGE for plain / best agent / aggregator (paper's Figure 2)
tests/             pytest, no GPU or network needed
docs/DECISIONS.md  every deviation from the paper + open questions
```

## Colab quickstart

```python
from google.colab import drive
drive.mount("/content/drive")

!git clone <your-repo-url> /content/moral-rag-cf
%cd /content/moral-rag-cf
!pip install -q -r requirements.txt
!pip install -q -e .

# 1. confirm MIC downloads and the column names match src/mvrag/data.py
!python scripts/inspect_mic.py --data-dir /content/drive/MyDrive/moral-rag-cf/data/MIC

# 2. five agent answers for 100 test questions (resumable)
!python scripts/run_agents.py --split test --n 100 \
    --data-dir /content/drive/MyDrive/moral-rag-cf/data/MIC \
    --out /content/drive/MyDrive/moral-rag-cf/outputs/agents_test.jsonl

# 3. first ROUGE numbers
!python scripts/eval_rouge.py --split test \
    --data-dir /content/drive/MyDrive/moral-rag-cf/data/MIC \
    --agents /content/drive/MyDrive/moral-rag-cf/outputs/agents_test.jsonl
```

Keep `data/` and `outputs/` on Drive so a runtime reset costs nothing.

## Run the tests locally

```
pip install pytest
pytest
```

## Roadmap

- [x] Week 1: repo, MIC loader, prompt-based agents, resumable generation
- [x] Value judges (reward models) and Table 1 at small scale
- [x] Prompt-only aggregator and ROUGE evaluation
- [ ] LoRA-trained aggregator (the paper's CMVA-GS)
- [ ] Retrieval: MiniLM embeddings + FAISS over the MIC train split
- [ ] Counterfactuals: leave-one-out and swap tests on agent answers and passages
- [ ] Report and results

See `docs/DECISIONS.md` for how this differs from the paper.
