# Decisions and deviations from the paper

Paper: Dognin et al., "Contextual Moral Value Alignment Through Context-Based
Aggregation" (arXiv 2403.12805). No code was found for it; this is a
re-implementation from the paper's description, at reduced scale.

## Deviations (copy into the report)

| Paper | This project | Why |
|---|---|---|
| Open Assistant 12B base model | 1-3B instruct model (default Qwen2.5-1.5B-Instruct), 4-bit, fp16 | Colab T4 (16GB) |
| 5 moral agents, each PPO-tuned (TRL) against a value reward model | Same model with 5 value-specific system prompts | RL fine-tuning needs far more compute |
| Aggregator: LoRA SFT, 8x A100-80GB, 21.6K steps | Prompt-only first; small LoRA on a subset if time allows | Compute |
| Full 91.0K/11.4K/11.4K samples | Subset of test questions (n=100 to ~1000) | Compute and time |
| Reward models: one classifier per value | Small classifiers trained on MIC labels (planned) | Needed for value-alignment scores |
| Llama-2 7B/13B and Agg13Llama baselines | Not replicated (or replaced by smaller prompted baselines) | Compute |

## Things the paper does not specify (our choices)

- Agent prompts. Descriptions in `agents.py` are our paraphrases of Moral
  Foundations Theory.
- Format of the aggregator input and how the "moral profile vector" is encoded
  or derived from MIC. Placeholder idea: multi-hot over the five values from a
  row's `morals`.
- Train/val/test split procedure. Note that 91.0K + 11.4K + 11.4K is about
  113.8K, close to MIC's ~114K annotation rows, which suggests the paper's
  "samples" are annotation rows rather than unique prompt-reply pairs. Unconfirmed.
- Which reply is the "ground truth" when a question has several. We currently
  take the best ROUGE over all replies for a question.

## Inconsistencies noticed in the paper

- Text says classifiers are trained for four values (care, fairness, authority,
  sanctity), then uses five (adds loyalty).
- Eq. (1) writes argmin where the intent is clearly to maximize reward.
- Figure 2 reports ROUGE only as a chart, so exact numbers cannot be matched,
  only the ordering (CMVA-GS highest, Agg13Llama lowest).

## Methodology rules

- Greedy decoding everywhere, so counterfactual differences are not sampling noise.
- Retrieval must exclude rows with the same question as the query, since MIC has
  several rows per question and the split may not be by question (check this).
- Select evaluation subsets by sorted question id (`unique_questions`) so they
  are reproducible.

## Open questions for the professor

1. Is combining G11 and G15 allowed?
2. Is any code, split file, or checkpoint available?
3. Is a reduced-scale re-implementation acceptable for the replication grade?
4. Does the paper's split correspond to MIC's official train/dev/test files?

## Confirmed from real MIC data (2026-11-13, Colab)

- Official MIC files have 91,014 / 11,427 / 11,376 rows (train/dev/test), matching
  the paper's 91.0K/11.4K/11.4K. So the paper's "samples" are MIC annotation rows
  and its split is MIC's official split.
- Unique questions: 28,704 / 3,761 / 3,753.
- Columns: A, A_agrees, Q, QA, chatbot, comments, moral, moral-vector, rot,
  rot-agree, split, violation-severity, worker_answer.
- `moral-vector` has 6 entries. One row (moral = loyalty -> index 3) is consistent
  with the order [care, fairness, liberty, loyalty, authority, sanctity]. Verify on
  more rows before relying on it.
- Some rows have no moral label (NaN). Missing text is sometimes the string "nan".
- `worker_answer` is a human-written reply. Open question: is the paper's
  "ground-truth answer" the chatbot reply `A` or `worker_answer`?
- Foundation counts in train: care 46,729; fairness 18,948; liberty 17,499;
  loyalty 17,296; authority 16,171; sanctity 10,140.

## Value judges / reward models (2026-11-13)

- Paper: one classifier per moral value, built from SocialChem action judgments
  binarized to 0 (negative) vs 1 (neutral/positive); P(class 1) is the reward.
- Ours: same data and labeling. `distilroberta-base`, 1 epoch, up to 20K distinct
  actions per value (seeded subsample), max length 64. The paper does not name its
  classifier architecture or training settings, so these are our choices.
- Repeated annotations of the same action are averaged before binarizing.
- Rows flagged `rot-bad` are dropped.
- Known limitation (inherited from the paper's design): each judge is trained on
  good-vs-bad actions *within* one value, so it measures "is this morally
  acceptable" on that value's kind of situation. It is not trained to tell values
  apart. Judges are also trained on short action phrases but applied to full
  replies (domain shift).
- Agent prompts v2 (first person, 1-2 sentences, no AI disclaimers) raised mean
  ROUGE-L from ~0.12 to ~0.15 on 20 test questions, but the five agents still give
  near-identical answers. Motivates trained agents.

## Results log

### 2026-11-13: judges trained (Colab T4, distilroberta-base, 1 epoch)

SocialChem has 355,922 rows (train 233,501 / dev 29,234 / test 29,239).

| judge | dev accuracy | macro-F1 | majority baseline | n dev |
|---|---|---|---|---|
| care | 0.901 | 0.890 | 0.676 | 2000 |
| fairness | 0.866 | 0.864 | 0.545 | 2000 |
| loyalty | 0.884 | 0.869 | 0.643 | 2000 |
| authority | 0.859 | 0.853 | 0.637 | 2000 |
| sanctity | 0.851 | 0.851 | 0.516 | 1743 |

### 2026-11-13: prompted advisors (v2) scored by judges, 20 test questions

Mean P(follows value); rows = advisors, columns = judges.

| advisor | care | fairness | loyalty | authority | sanctity |
|---|---|---|---|---|---|
| care | 0.934 | 0.952 | 0.971 | 0.909 | 0.906 |
| fairness | 0.920 | 0.941 | 0.951 | 0.897 | 0.908 |
| loyalty | 0.857 | 0.914 | 0.946 | 0.846 | 0.876 |
| authority | 0.913 | 0.924 | 0.939 | 0.869 | 0.890 |
| sanctity | 0.905 | 0.946 | 0.961 | 0.815 | 0.916 |

Own-judge mean 0.921 vs other-judge mean 0.910: the advisors are not
differentiated by the judges. n=20 only, so treat as preliminary.

## Model and hardware change (2026-11-16)

- Generation model switched from Qwen2.5-1.5B-Instruct to **Qwen2.5-7B-Instruct**.
  On the same 20 test questions, mean pairwise ROUGE-L between the five agents'
  answers (lower = more distinct) was 0.388 (1.5B), 0.283 (3B), 0.246 (7B).
- Runs moved to a Colab L4 GPU with the model in fp16 (`--no-4bit`), batch of 16
  questions. Measured speed: 40 questions (200 agent answers) in 33 s, about 6x
  the T4 with 4-bit. All "7B_full" outputs use this mode.
- Earlier 1.5B results (Table 1 on 300 questions): plain -> agent was care
  84.0 -> 87.8, fairness 88.8 -> 91.0, loyalty 91.0 -> 91.7, authority 83.1 -> 83.4,
  sanctity 82.8 -> 85.4 (%). Own-judge mean 0.879 vs other-judge mean 0.873.

## Aggregator (2026-11-16)

- Input format is ours (the paper does not publish it): question, moral profile,
  the five agent answers, optionally retrieved examples.
- Moral profile: five 0/1 entries (care, fairness, loyalty, authority, sanctity)
  from the row's `moral` labels. MIC's sixth foundation, liberty, is ignored as in
  the paper. Rows with no label among the five have an all-zero profile and are
  excluded from the ROUGE comparison (reported as "rows scored").
- Prompt-only aggregator = analogue of the paper's "Agg13Llama" baseline (an LLM
  asked to aggregate the agents' answers). The paper's CMVA-GS is the LoRA-trained
  aggregator, still to do.
- Identical (question, profile) inputs are generated once and reused for every
  MIC row that shares them.
- Reference answer: the paper does not say whether the ground truth is MIC's
  `worker_answer` (human-written) or `A` (chatbot reply). Both are reported until
  the professor confirms.
- "best_agent" baseline (ours, not in the paper): the single agent whose value is
  the first one set in the row's profile.

### 2026-11-16: Table 1 on the full test split (3,753 questions, Qwen2.5-7B, L4 fp16)

P(answer follows the value) according to that value's judge.

| value | plain model | moral agent | change | paper PT model | paper agent |
|---|---|---|---|---|---|
| care | 79.53% | 87.18% | +7.65 | 74.70% | 96.74% |
| fairness | 85.71% | 89.18% | +3.48 | 85.20% | 92.40% |
| loyalty | 88.40% | 93.50% | +5.10 | 74.38% | 98.20% |
| authority | 78.72% | 86.64% | +7.93 | 91.58% | 98.83% |
| sanctity | 77.59% | 80.36% | +2.77 | 78.37% | 93.05% |

Agents' own-judge mean 0.874 vs other-judge mean 0.868: every agent beats the
plain model on its own value, but the judges still do not separate the agents
from each other. Paper used 5K MIC questions and PPO-tuned agents.

### 2026-11-16: judges retrained on all available data (`judges/full`)

distilroberta-base, 1 epoch, every distinct action per value, evaluated on the
whole dev set.

| judge | accuracy | macro-F1 | majority | n train | n dev |
|---|---|---|---|---|---|
| care | 0.915 | 0.904 | 0.655 | 81,117 | 11,092 |
| fairness | 0.874 | 0.873 | 0.555 | 32,059 | 4,316 |
| loyalty | 0.890 | 0.881 | 0.647 | 34,254 | 4,555 |
| authority | 0.862 | 0.857 | 0.624 | 19,963 | 2,643 |
| sanctity | 0.852 | 0.852 | 0.516 | 13,196 | 1,743 |

Table 1 rescored with these judges (3,753 test questions, same answers):

| value | plain model | moral agent | change |
|---|---|---|---|
| care | 92.64% | 97.20% | +4.56 |
| fairness | 82.06% | 87.01% | +4.95 |
| loyalty | 83.34% | 89.59% | +6.25 |
| authority | 78.51% | 87.17% | +8.67 |
| sanctity | 81.50% | 88.79% | +7.29 |

Own-judge mean 0.900 vs other-judge mean 0.889. The absolute numbers move with
the judge (care plain 79.5% -> 92.6% between judge versions, on identical text),
so only comparisons made with the same judges are meaningful. `judges/full` is
the set used from here on.

### 2026-11-16: prompt-only aggregator, first 200 inputs (214 rows)

ROUGE-L vs worker_answer: plain 0.132, best_agent 0.120, aggregator 0.123.
ROUGE-L vs chatbot reply A: plain 0.171, best_agent 0.159, aggregator 0.162.
Speed on L4 fp16: 200 inputs in 97 s. Qualitatively the prompt-only aggregator
follows the profile but tends to copy one matching agent's answer verbatim
rather than blend several (e.g. profile care+fairness+loyalty -> the loyalty
answer unchanged).

## Aggregator training settings (ours vs the paper)

| | paper | ours |
|---|---|---|
| base model | Open Assistant 12B | Qwen2.5-7B-Instruct (4-bit while training) |
| method | LoRA, supervised | LoRA, supervised (loss on the target reply only) |
| LoRA rank | 128 | 64 (alpha 128, dropout 0.05, q/k/v/o projections) |
| learning rate | 5e-7 | 1e-4, 3% warmup then linear decay |
| batch | 256 (8 x A100-80GB) | 16 (8 x 2 accumulation, one L4) |
| passes over the data | ~60 epochs (21.6K steps) | 1 epoch |
| data | 91.0K MIC train samples | all MIC train rows with a target reply |
| target | "ground-truth answer" (unspecified) | `worker_answer` by default; `--target answer` for the chatbot reply |

## Reward-trained moral agents (decided 2026-11-16)

Project deadline: 2026-12-03. Decision: train the agents with a reward signal,
as in the paper, instead of only prompting them.

| | paper | ours |
|---|---|---|
| reward | P(answer follows value) from that value's classifier | same, using `judges/full` |
| algorithm | PPO (TRL), batch 256 episodes, 4 epochs per batch, lr 2e-9 | best-of-N selection + supervised LoRA fine-tuning |
| start policy | the pre-trained model | the same model with the value-specific prompt (kept at inference) |
| regularization | KL penalty towards the initial policy | targets are the start policy's own samples |
| data | MIC training questions | first 4,000 MIC training questions (by question id), 4 samples each, temperature 1.0, top-p 0.95 |
| output | 5 fine-tuned LLMs | 5 LoRA adapters (rank 16, 2 epochs, lr 1e-4) on one base model |

Why not PPO: unstable and slow for a 7B model on one GPU. The paper cites TRL's PPO
implementation; TRL 0.12 removed that original PPO trainer and renamed a different
one (PPOv2) to PPO, which needs a policy, a reference policy, a value model and a
reward model that shares the policy's tokenizer, i.e. several copies of the LLM
in memory. Our judges are separate DistilRoBERTa classifiers.

Budget note (2026-11-16): compute units are no longer a constraint; the limits
are the 2026-12-03 deadline and GPU availability on Colab (A100 often unavailable).
Open question for the professor: which exact Open Assistant 12B checkpoint is the
paper's PT model? `OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5` is public, but
its model card describes supervised fine-tuning only, while the paper says the PT
model was also tuned with RL on human preferences.
Why keep the value prompt: without it the five agents would differ only through
five highly correlated judges, and the prompted agents were already barely
distinct (own-judge 0.900 vs other-judge 0.889).
Why 4,000 questions: 4,000 x 5 values x 4 samples = 80K generations (~4 h on an
L4); all 28,704 questions would be ~26 h.
Greedy decoding is still used for every evaluated answer; sampling is only used
to create training candidates.

## 2026-11-17: final judges = RoBERTa-large (`judges/best`)

Trained on an A100 with every available SocialChem example per value, 1 epoch.
First pass (lr 2e-5, no warmup): fairness collapsed to the majority class
(0.555 = majority baseline); the other four improved. Fairness was retrained
with lr 1e-5 and 6% linear warmup + linear decay. `scripts/pick_judges.py` then
took the more accurate judge per value; all five picks are RoBERTa-large.

| judge | DistilRoBERTa (`judges/full`) | RoBERTa-large (`judges/best`) |
|---|---|---|
| care | 0.915 | 0.941 |
| fairness | 0.874 | 0.913 (derived from the reported mean; confirm in judges/best/metrics.json) |
| loyalty | 0.890 | 0.911 |
| authority | 0.862 | 0.877 |
| sanctity | 0.852 | 0.888 |
| mean | 0.879 | 0.906 |

Caveat: these are SocialChem dev accuracies, also used to pick between the two
sets, so they are slightly optimistic. The SocialChem test split is untouched and
can be used to report final judge accuracy.

`judges/best` is used for reward training and for every Table 1 from here on.
All earlier Table 1 numbers (DistilRoBERTa judges) are superseded and are not
comparable with the new ones.

## 2026-11-17: hardware and reward-training scale

- Colab Pro+, NVIDIA A100-SXM4-40GB. Agent LoRA training runs without gradient
  checkpointing on the A100 (0.95 s/step at 16 examples/step, peak 27.1 GB);
  on the L4 it needs gradient checkpointing (3.0 s/step, peak 16 GB).
- Colab's preinstalled torchao 0.10 breaks PEFT; every session runs
  `pip uninstall -y torchao` first.
- Reward training uses 8 candidates per question on the first 8,000 MIC training
  questions (by question id), sampled in two passes (4,000 then +4,000) so that
  every value has a complete 4,000-question set even if the run is stopped early.
