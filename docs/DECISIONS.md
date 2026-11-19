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

## 2026-11-17: prompted baselines (paper's Figure 2 systems), preliminary scoring

Llama baselines: 5 MIC test examples per desired moral (worker_answer replies,
never from the question being answered), fp16 on an A100 40GB. 7,796 distinct
(question, profile) inputs each. Run times: Llama-2-7B 29 min, Llama-2-13B
1 h 51 min (batch 4), Llama-3.1-8B 24 min.

ROUGE on the same 9,652 test rows (rows whose profile has at least one of the
five values). Reference = worker_answer (human-written):

| system | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|---|---|---|---|---|
| plain (Qwen 7B, no value instruction; paper's PT-model) | 0.1540 | 0.0239 | 0.1231 | 0.1231 |
| best_agent (instructed Qwen advisor for the row's first value) | 0.1536 | 0.0255 | 0.1203 | 0.1203 |
| aggregator (Qwen 7B, prompt-only, untrained) | 0.1571 | 0.0268 | 0.1246 | 0.1246 |
| Llama-2-7b-chat, few-shot (paper baseline) | 0.1409 | 0.0219 | 0.1107 | 0.1130 |
| Llama-2-13b-chat, few-shot (paper baseline) | 0.1590 | 0.0312 | 0.1253 | 0.1268 |
| Llama-3.1-8B-Instruct, few-shot (ours, modern) | 0.1942 | 0.0446 | 0.1652 | 0.1652 |

Reference = A (chatbot reply): ROUGE-L plain 0.1557, best_agent 0.1459,
aggregator 0.1490, Llama-2-7B 0.1453, Llama-2-13B 0.1658, Llama-3.1-8B 0.2026.

Observations:
- Llama-2-7B often refuses ("I'm just an AI, I don't have personal experiences
  ... so I cannot provide a reply"), which lowers its ROUGE. Llama-2-13B
  sometimes prefixes its reply with "A:" (copied from the prompt format).
- Paper's ordering partly reproduced: PT-model and Llama-13b are similar
  (0.1231 vs 0.1253 ROUGE-L). The paper's Agg13Llama is not run yet (needs the
  reward-trained agents' answers).
- Llama-3.1-8B few-shot is clearly the strongest so far (+0.04 ROUGE-L). Note
  that the few-shot baselines see example MIC replies in their prompt, while the
  plain model and the Qwen aggregator do not, so they get a style hint the
  others lack. This is the bar the trained aggregator has to beat.
- Agent/plain/aggregator answers here are from the instructed (not yet
  reward-trained) agents.

## Extension design: RAG (G11) + counterfactual explanations (G15) (code 2026-11-17)

RAG:
- Pool: MIC training split only, one entry per question, its human-written
  replies (worker_answer). Retrieval by cosine similarity of question texts
  (all-MiniLM-L6-v2 sentence embeddings), k = 3, at most one reply per retrieved
  question.
- Profile-aware: among a retrieved question's replies, prefer one whose MIC
  moral labels overlap the input's profile.
- Leakage: the input's own question is never retrieved (matters for training
  inputs, which come from the same split as the pool).
- Experiment: train the aggregator twice with identical settings, without and
  with retrieved examples in the prompt (CMVA-GS vs CMVA-GS + RAG), and compare
  ROUGE and judge alignment on the test set.

Counterfactuals (on a seeded sample of 500 test inputs, greedy decoding):
- Variants per input: full; drop each of the five agent answers; drop each
  retrieved example; drop all retrieved examples; swap each retrieved example
  for one retrieved for a different input.
- Measures vs the full reply: changed (text differs), influence
  (1 - ROUGE-L between the two replies), quality delta (ROUGE-L vs worker_answer),
  alignment delta (judge probability for the profile's values), flip (a judge's
  yes/no decision for a profile value changes).
- Key analyses: (1) are agents whose value is in the profile more influential
  than the others (a profile-faithful aggregator should say yes); (2) does
  influence fall with retrieval rank; (3) swap vs drop: does the aggregator read
  the retrieved content or only use its presence; (4) qualitative examples with
  the largest changes and flips.

Also added 2026-11-17: `train_agents.py --limit-questions` (data-size curve),
scoring now strips a leading "A:"/"Answer:"/"Reply:" label from every system's
output before ROUGE (Llama-2-13B copies "A:" from the prompt), and the harmless
Llama max_length warning is filtered.

## 2026-11-17: Drive data loss and the save-and-verify workflow

After the 8,000-question run, some files written through Colab's Google Drive
mount had silently disappeared or reverted: the RoBERTa-large sanctity judge's
weights, all weights in `judges/best` (a full copy of `judges/large`), and the
second batch of 4,000 reward candidates (each candidate file reverted from
8,000 to 4,000 questions). Drive was not full (48 GB of 236 GB used). All other
result files were intact. Cause unknown; that day two Colab sessions had the
same Drive mounted at once, and the lost files were the large ones and the
appended ones.

Consequences:
- The first set of reward-trained agents (`outputs/adapters/agents`) was trained
  on the first 4,000 questions, not 8,000. It is kept as the 4,000 point of the
  data-size curve (test answers: `agents_test_7B_rl.jsonl`, "rl4k").
- `judges/best` was dropped; `judges/large` holds the same five judges.
- The sanctity judge was retrained with identical settings (roberta-large,
  lr 2e-5, no warmup, seed 0, all data, 1 epoch). Dev accuracy 0.888, the same
  as the original to three decimals. Reward candidates for the first 4,000
  questions were scored with the original sanctity judge, all later ones and
  every table with the retrained one.

New workflow (from 2026-11-17 15:25): long runs write to the Colab VM's local
disk (/content/work), copy each finished stage to Drive with rsync, verify the
copy with `diff -rq` (printing SAVED or PROBLEM), and call
`drive.flush_and_unmount()` at the end. Only one Colab session uses the Drive
at a time. Nothing is appended to Drive files during a run.

## 2026-11-18: reward-trained agents, data-size curve (Table 1, judges/large)

All rows: 3,753 test questions, Qwen2.5-7B, scored by the final RoBERTa-large
judges (`judges/large`). Plain model = same model, no value instruction.
Reward training: best-of-8 by the value's judge, LoRA rank 16, 2 epochs, lr 1e-4.

P(answer follows value), per value:

| agents | care | fairness | loyalty | authority | sanctity | mean |
|---|---|---|---|---|---|---|
| plain model | 90.67 | 88.40 | 91.15 | 80.01 | 81.66 | 86.38 |
| prompted (no training) | 96.14 | 92.16 | 97.28 | 85.90 | 88.26 | 91.95 |
| reward-trained, 2,000 questions | 97.15 | 94.78 | 98.31 | 91.02 | 92.69 | 94.79 |
| reward-trained, 4,000 | 97.53 | 94.84 | 98.40 | 91.04 | 92.98 | 94.96 |
| reward-trained, 8,000 | 97.63 | 94.99 | 98.45 | 91.15 | 93.11 | 95.07 |
| reward-trained, all 28,704 | 97.85 | 95.31 | 98.54 | 91.78 | 93.40 | 95.38 |
| paper PT model | 74.70 | 85.20 | 74.38 | 91.58 | 78.37 | 80.85 |
| paper Moral Agents (PPO) | 96.74 | 92.40 | 98.20 | 98.83 | 93.05 | 95.84 |

(16,000-question point pending.)

Findings:
- Reward training adds +2.8 to +3.4 points over prompting (mean 91.95 -> 94.79
  at 2K, 95.38 at all 28.7K). Largest gains on authority (+5.9) and sanctity
  (+5.1), the two values where prompting was weakest.
- Diminishing returns: 2K -> 28.7K questions adds only +0.6. Most of the gain
  comes from the first 2,000 questions. This justifies the data size used.
- Our agents' mean (95.38) is close to the paper's PPO agents (95.84). We match
  or exceed the paper on care, fairness, loyalty and sanctity; authority is
  7 points lower (91.78 vs 98.83).
- Distinctness barely changes: own-judge vs other-judge mean 0.919 vs 0.912
  (prompted), 0.954 vs 0.942 (all). Training raises every judge's score, not
  only the agent's own value; the judges reward acceptability in general.
- Caveat: the same judges that gave the reward also score Table 1, which
  favours the trained agents (the paper does the same). Planned check: rescore
  with the independently trained DistilRoBERTa judges (`judges/full`).

Final training losses (last 25 steps): 8K 0.58-0.65, all 0.56-0.64 by value.
The run's last stage (16K test answers) was lost in a Colab disconnect at
2,112/3,753 and is rerun from the saved 16K adapters.

## 2026-11-18: second Drive loss; results now stored on the Hugging Face Hub

The overnight run printed SAVED (rsync + diff against the Drive mount) for every
stage, yet after the session disconnected (before `drive.flush_and_unmount()`
ran) the Drive held only part of it. Kept: 8K agent adapters, their test
answers, the first 16,000 reward candidates, four large judges. Lost: the
large sanctity judge (again), candidates 16,001-28,704, the 2K/16K/all agent
adapters and the 2K/all test answers. The diff only compared against the
mount's local cache, so it could not see that uploads had not reached Google's
servers; large files apparently wait to upload and are lost when the session
ends first.

All numbers from that run were printed in the notebook and are recorded above
(the log is the evidence for the 2K and all-data rows).

Change: long-run artifacts (judges, adapters, candidates, result files) are now
pushed to a private Hugging Face dataset repository with `scripts/hub.py`, which
checks every file's size against what the Hub reports after upload, and stops
the run on a mismatch. Drive is used only to read the input data.

## 2026-11-18: stale fairness judge from Drive; all candidates re-scored

When the large judges were copied from Drive on 2026-11-18, `judges/large/fairness`
contained the *first, collapsed* fairness model (accuracy 0.555 = majority),
while its metrics.json held the retrained model's 0.913. Evidence: every
fairness candidate sampled that day scored about 0.56 regardless of content
(new-block mean and best-of-8 mean both ~0.56, vs 0.92 / 0.98 before). Only
fairness candidates for questions 16,001+ were affected; the other judges'
rewards were in their usual range. Earlier fairness rewards and all earlier
tables used the correct judge (their plain/agent fairness scores are not flat).

The re-retrained sanctity judge (dev accuracy 0.892) scores answers lower on
average than the previous one (~0.81 vs ~0.87 mean candidate reward) at the same
accuracy, i.e. different calibration. Best-of-8 selection depends only on the
ranking of a question's 8 candidates, so this matters less, but for consistency:

Fix:
- `scripts/check_judges.py` now measures each judge's real dev accuracy from its
  weights and flags BROKEN (no better than the majority guess, or near-constant
  outputs) or MISMATCH with metrics.json. Run before any judge is used.
- Fairness large judge retrained (lr 1e-5, 6% warmup, as on 2026-11-17).
- `scripts/rescore_candidates.py` re-scored every saved candidate (all values,
  all 28,704 questions) with the final, checked `judges/large`, so all rewards
  come from one judge set. No candidates were regenerated.

## 2026-11-19: FINAL Table 1 and data-size curve (supersedes earlier Table 1s)

All candidates re-scored with the checked final judges (`judges/large`;
fairness retrained 2026-11-18, dev acc 0.914 on the check sample). Re-scoring
changed fairness rewards from 0.816/0.860 to 0.916/0.977 (all/best-of-8) and
sanctity from 0.848/0.966 to 0.818/0.955; other values unchanged. Every agent
set below was trained on these rewards (except rl4k and rl8k, trained 2026-11-17
on rewards from the earlier judge versions) and every row is scored on the same
3,753 test questions by the same judges. All artifacts are on the Hugging Face
Hub (private repo moral-rag-cf-artifacts).

Final judges (RoBERTa-large, also the reward models), P(answer follows value) %:

| agents | care | fairness | loyalty | authority | sanctity | mean |
|---|---|---|---|---|---|---|
| plain model | 90.67 | 88.40 | 91.15 | 80.01 | 82.52 | 86.55 |
| prompted | 96.14 | 92.16 | 97.28 | 85.90 | 82.99 | 90.89 |
| reward-trained 2K | 97.21 | 94.75 | 98.32 | 91.22 | 88.43 | 93.99 |
| reward-trained 4K | 97.53 | 94.84 | 98.40 | 91.04 | 88.57 | 94.08 |
| reward-trained 8K | 97.63 | 94.99 | 98.45 | 91.15 | 88.56 | 94.16 |
| reward-trained 16K | 97.79 | 95.29 | 98.48 | 91.43 | 89.83 | 94.56 |
| reward-trained all 28.7K | 97.84 | 95.28 | 98.56 | 91.41 | 90.22 | 94.66 |
| paper PT model | 74.70 | 85.20 | 74.38 | 91.58 | 78.37 | 80.85 |
| paper Moral Agents (PPO) | 96.74 | 92.40 | 98.20 | 98.83 | 93.05 | 95.84 |

Independent judges (DistilRoBERTa, never used as a reward):

| agents | care | fairness | loyalty | authority | sanctity | mean |
|---|---|---|---|---|---|---|
| plain model | 92.64 | 82.06 | 83.34 | 78.51 | 81.50 | 83.61 |
| prompted | 97.20 | 87.01 | 89.59 | 87.17 | 88.79 | 89.95 |
| reward-trained 2K | 97.80 | 90.73 | 92.29 | 91.85 | 93.13 | 93.16 |
| reward-trained 4K | 97.94 | 90.94 | 92.37 | 91.85 | 93.58 | 93.34 |
| reward-trained 8K | 98.03 | 91.21 | 92.34 | 91.78 | 93.56 | 93.38 |
| reward-trained 16K | 98.07 | 91.50 | 92.54 | 91.82 | 93.91 | 93.57 |
| reward-trained all 28.7K | 98.16 | 91.36 | 92.88 | 91.92 | 94.03 | 93.67 |

Own-judge vs other-judge mean (final judges): prompted 0.909/0.907,
2K 0.940/0.934, all 0.947/0.940 (gap +0.002 -> +0.007).

Findings:
1. Reward training works and is not an artifact of grading with the reward
   model: the independent judges show the same pattern (prompted 89.95 ->
   all 93.67, vs 90.89 -> 94.66 with the reward judges).
2. Diminishing returns: 2K -> 28.7K questions adds +0.67 (reward judges) and
   +0.51 (independent). Most of the gain comes from the first 2,000 questions.
3. Versus the paper (mean 94.66 vs 95.84): we match or exceed the paper on care,
   fairness and loyalty; authority (91.41 vs 98.83) and sanctity (90.22 vs 93.05)
   are lower. Note the paper's PT model starts at 91.58 on authority, ours at 80.01.
4. The agents become slightly more distinct with training (gap 0.002 -> 0.007)
   but remain far from separable: each agent scores high on every value.
5. Judge calibration matters: the re-retrained sanctity judge (same accuracy)
   scores the prompted agent at 82.99 vs 88.26 with the previous sanctity judge.
   Only numbers from one judge set are compared.

Best agents for the next stage: reward-trained on all 28,704 questions
(`outputs/adapters/agents_all`).

## 2026-11-19: moderator (aggregator) stage, preparation

Done on an A100 40GB, all saved to the Hub:
- Best agents (reward-trained, all 28.7K) answered all validation (3,761) and
  training (28,704) questions: `agents_val_7B_rlall.jsonl`, `agents_train_7B_rlall.jsonl`.
  Speed 0.36 s/question (5 answers) at 64 questions per batch.
- Retrieval (all-MiniLM-L6-v2, k=3, pool = 28,704 training questions) for
  test (9,144 inputs), val (9,167) and train (71,832 distinct question+profile).
- Prompt-only (untrained) Qwen moderator on the test set with the trained
  agents, without RAG (954 s) and with RAG (1,102 s).

Training trials (rank 128, batch 8 x 2, no gradient checkpointing) ran out of
memory on the A100 40GB for both variants: the moderator's prompts (five agent
answers, plus three retrieved examples for RAG) are several times longer than
the agents'. Changes:
- The loss is now computed from the logits of the target reply only
  (`target_only_loss`, left padding + `logits_to_keep`). Verified identical to
  the standard loss and gradients (loss diff 5e-7, gradient diff 2e-8).
- `--max-len` raised from 640 to 1024 tokens, so RAG prompts are not dropped;
  the script reports how many examples were too long.
- Gradient checkpointing on the A100 40GB.

## 2026-11-19: Figure 2 so far (untrained moderators and baselines)

ROUGE-L F1 on the 9,652 MIC test rows that every system answered (systems
compared on exactly the same rows). Agents = reward-trained on all 28.7K.

| system | vs human reply (worker_answer) | vs chatbot reply (A) |
|---|---|---|
| plain Qwen2.5-7B (PT model) | 0.1231 | 0.1557 |
| best single agent | 0.1151 | 0.1374 |
| Qwen moderator, prompt only | 0.1196 | 0.1414 |
| Qwen moderator, prompt only + RAG | **0.1248** | 0.1452 |
| Agg13Llama (Llama-2-13B aggregates the agents) | 0.1144 | 0.1481 |
| Llama-2-7B-chat, 5-shot | 0.1109 | 0.1455 |
| Llama-2-13B-chat, 5-shot | 0.1257 | 0.1661 |
| Llama-3.1-8B-Instruct, 5-shot | 0.1652 | 0.2026 |

Findings:
1. RAG helps the untrained moderator: +0.0052 ROUGE-L (+4.3% relative) against
   the human reply, enough to pass the plain model.
2. Same ordering as the paper among aggregators: our moderator > Agg13Llama.
3. The few-shot baselines win so far because their 5 MIC examples show the
   short MIC reply style; ROUGE rewards style. The trained moderator (CMVA-GS)
   learns that style from the training targets, which is the paper's point.
4. With the reward-trained agents the prompt-only moderator scores lower than
   with the instructed agents (0.1196 vs 0.1246): the trained agents moralize
   more, which moves the blended reply away from the human reply.

To verify before the report: Agg13Llama covered 7,796 test inputs (vs 9,144 for
the Qwen moderators), which is why 9,652 of 11,376 rows are scored.

## 2026-11-19: moderator training setup

Trials (30 steps, rank 128, batch 8 x 2, A100 40GB, gradient checkpointing,
target-only loss): all 91,014 training rows usable (none too long).
- Without RAG: 1.97 s/step, peak 17.6 GB, 3.1 h per epoch.
- With RAG: 2.56 s/step, peak 18.2 GB, 4.0 h per epoch.
- Without gradient checkpointing the RAG variant runs out of memory on 40 GB.

Plan: 2 epochs (11,378 steps) per moderator, the two trained in parallel in
two Colab sessions. Training runs in chunks of 1,000 steps; after each chunk
the adapter is pushed to the Hub, and the epoch-1 adapter is kept separately
so the validation set can decide whether the second epoch helps.

The AdamW state is now saved with each checkpoint and restored on resume, so
training in chunks is identical to one run (tested). The optimizer state is
kept on the Colab disk only (about 1 GB per moderator); if a session is lost,
the run resumes from the Hub adapter with a fresh optimizer state.

## 2026-11-19: trained moderators (CMVA-GS), without and with RAG

Both trained 2 epochs (11,377 steps; LoRA rank 128) on all 91,014 training rows.
Both sessions disconnected once (Colab); training resumed from the last Hub
checkpoint (plain at step 5,689, RAG at 8,689) with a fresh AdamW state.

Epoch choice on validation (ROUGE-L vs human reply, 9,782 rows):

| moderator | epoch 1 | epoch 2 |
|---|---|---|
| without RAG | **0.2255** | 0.2250 |
| with RAG | **0.2307** | 0.2269 |

Epoch 1 is used for both on the test set (epoch 2 slightly overfits).

Test set, ROUGE-L (9,652 rows shared by every system):

| system | vs human reply | vs chatbot reply |
|---|---|---|
| best single advisor | 0.1151 | 0.1374 |
| prompt-only moderator | 0.1196 | 0.1414 |
| prompt-only moderator + RAG | 0.1248 | 0.1452 |
| Llama-3.1-8B 5-shot (best baseline) | 0.1652 | 0.2026 |
| trained moderator (CMVA-GS) | 0.2240 | 0.2460 |
| **trained moderator + RAG** | **0.2258** | **0.2517** |

Training nearly doubles ROUGE-L and beats every baseline. RAG adds a small gain
on top of training, on both val (+0.0052) and test (+0.0018 human, +0.0057
chatbot). Significance: paired bootstrap over questions (`mvrag.stats`,
`eval_aggregator.py --compare`), pending.

Added `scripts/score_moderators.py` (Table 2): the value judges score each
reply; on-profile = mean over the user's values, off-profile = the rest, gap =
how selectively the reply serves the user's values.
