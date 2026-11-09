# PROTOCOL v1.1 — unified study (binding). Part A: v1.0 base text; Part B: v1.1 amendments supersede Part A where they conflict.

## 1. Title, thesis, causal story

**Title.** *Rate Is Not Displacement: Displacement Accounting and TV-Capped Attention Deletion in Data-Limited Biomedical Transformers.*

**Method name.** CapDrop: TV-capped top-k attention deletion. It is renamed to avoid confusion with arXiv 2504.12088 ("AttentionDrop", five authors, a different method). That preprint and the change in author list are disclosed.

**Thesis.** Attention regularizers are set by nominal rates (p, γ). The forward pass feels only the realized per-row displacement d = ½‖ã−a‖₁. The mean D̄, second moment S2, train/eval mean shift B = ½‖E[ã]−a‖₁ and tail of that displacement all depend on how concentrated attention is. The paper:
1. States exact displacement laws for five families.
2. Calibrates every family to the same realized mean-displacement levels, using a census taken before any outcome is seen.
3. Tests two primary data-limited regimes on PubMed text: a multi-epoch LM trained from scratch, and BiomedBERT fine-tuned on 450 PQA-L examples. In each it asks:
   - does capping the tail at matched mean displacement change held-out performance?
   - does CapDrop beat its fixed-γ special case (tuned Relaxed Attention)?
   - does it beat the strongest tuned control?

A secondary regime and a recipe-failure arm mark the limits of the claims. The secondary regime is compression by distillation from a teacher that sees the conclusion (a "privileged" teacher). The recipe-failure arm is L0.

**Causal story (one paper, one measured variable).**
- **Mechanism.** Scarce labels or tokens push the model to rely on a few high-mass links (sinks, shortcuts). A perturbation reaches the model only through its realized displacement (rate × concentration), and that displacement drives generalization.
- **What CapDrop changes.** For top-k deletion, d equals the deleted mass m, which is heavy-tailed on concentrated rows. CapDrop sets ã = (1−α)a + αb with α = min(1, τ/m), so d = min(m, τ).
  - Deletion frequency is unchanged and the tail is removed.
  - Exactly, TV(b, ã) = (m−τ)₊.
- **Track 1 null.** At p = 0.1 and k = 4, E[m] was 0.017–0.055 and the budget bound in only 4–20% of rows. CapDrop was therefore nearly identical to unbudgeted deletion, so the null result is what this account predicts.
- **Track 2 collapse.** Train CE ended at the label entropy (0.937) after 125 LoRA steps. That is an optimization failure outside the reach of displacement. The stabilized recipe fixes it, and L0 quantifies it.
- **Shared measurements.** Every study measures the same budget activity (E[m], P(m>τ), E[(m−τ)₊], realized TV, B, S2) and runs the same contrasts.

## 2. Research questions, hypotheses, decision rules

**Common rules.**
- n = 10 paired units: fold-models for the encoder and the student, seeds for the LM.
- The test is an exact two-sided sign-flip test on the mean paired difference. It has 1,024 patterns and a minimum p of 0.00195, which is below 0.05/6.
- Holm correction within each family.
- Endpoints (lower is better):
  - Encoder: per-run Brier on the official test, Σ_c(p_c − y_c)², range 0–2.
  - LM: sealed-test NLL in nats/token at the final step.

**Verdicts.**
- **Better / Worse:** Holm-adjusted p < 0.05. For encoder test contrasts, the 95% two-level (runs × test examples) bootstrap CI must also exclude 0.
- **Equivalent:** the 90% CI lies inside the margin.
  - Encoder: two-level bootstrap CI, margin ±0.015 Brier. That is 2.6% of the Brier score of always predicting the class priors (0.569).
  - LM: paired-t CI, margin ±0.02 nats/token (about 2% perplexity).
- **Inconclusive:** anything else.
- n and the margins are fixed. Null claims are made only through TOST.
- Minimum detectable effect (MDE) at 80% power: about 1.34 σ_d at the first Holm step and about 0.99 σ_d at the last. The observed σ_d is reported.

**Primary confirmatory family F1 (6 tests).**
- HB@0.10 is CapDrop (p_hi, k = 4, τ₀.₁₀). HU@0.10 is unbudgeted top-4 deletion (p_lo). Both are calibrated to a realized mean displacement of 0.10.
- They share random numbers, so the keys HU deletes are a subset of the keys HB proposes to delete.
- X\* denotes the level of family X selected on the tuning data.

| ID | Contrast | Regime | Registered expectation |
|---|---|---|---|
| H1-E | HB@0.10 − HU@0.10 | Encoder | Recorded in `predictions.json`: "HB better" if the census clipped excess E[(m−τ)₊] at ℓ = 0.10 is ≥ 0.05, because large displacements should harm fine-tuned pretrained attention. The test is still two-sided. |
| H1-L | HB@0.10 − HU@0.10 | LM | No direction, because in an overfitting regime the tail may regularize. For both H1 tests, "equivalent" is predicted if P(m>τ) < 0.05. |
| H2-E / H2-L | HB\* − RA\* | Encoder / LM | None. This is the novelty test against the special case. |
| H3-E / H3-L | HB\* − C\* | Encoder / LM | None. The literature predicts a null. |

If C\* = RA\*, H3 duplicates H2, but the family still counts as 6 tests.

**Secondary family F2 (4 tests; Holm; sign-flip).**
- S-a: S-HB@0.10 − S-HU@0.10 in the student (test Brier).
- S-b: S-HB@0.10 − S-N.
- S-c: evidence-use gap, EUG = acc(full) − acc(question-only), HB\* vs N (encoder).
- S-d: flip rate under mismatched evidence, HB\* vs N.

**Exploratory (no confirmatory wording).**
- Dose–response over all matched pairs at the 3 levels, on dev and validation metrics only, against the census clipped excess (Spearman ρ, permutation p from 10⁴ permutations).
- A descriptive displacement-accounting model (DAM): Y ~ D̄ + B + S2 + family + (1|block). It is fitted on N/AD/DK/RA and predicts the HU/HB arms. It is heuristic and never presented as causal mediation.
- The census premise κ_E ≥ 2κ_L, where κ = E[m²]/E[m] at p = 0.1, k = 4.
- Failure counts (test accuracy ≤ 55.20), collapse counts (≥ 90% of predictions in one class) and Brown–Forsythe tests. At n = 10 these are descriptive.
- L0 and Q0 compared with the literature.

**Outcome → claim → venue.** This is fixed now. The headline never rests on an exploratory result.

| Outcome | Claim | Venue |
|---|---|---|
| H1 Better in ≥ 1 regime and H3 Better | Tail capping is the active ingredient and is useful in practice | Neural Networks |
| H1 Better; H3 Equivalent or Inconclusive | The tail of the displacement distribution matters at a fixed mean. The cap makes aggressive rates safe but does not beat tuned controls. | Neural Networks (Learning Systems) or Neurocomputing |
| H1 Equivalent | "How much, not how": mean realized displacement is enough, and nominal rates mislead. Framed as a preregistered, displacement-calibrated benchmark with exact laws. | Neurocomputing, or Machine Learning (SJR Q1) |
| H1 Worse | Heavy-tailed displacement regularizes, so the cap removes the useful part | Neural Networks or Neurocomputing |

In every case the paper contributes:
- the exact laws;
- the closed form;
- the census;
- the first fine-tuned question-only and mismatched-evidence baselines on PQA-L (none found in the literature we searched);
- the reconciliation of the Track-2 recipe.

## 3. Studies

### 3.0 Common protocol

**Held-out ledger.**
- **Official PQA-L test (500).**
  - Use 1 was the Track-2 pilot: 13 SmolLM2-family models, with no selection. This protocol is use 2, and the last.
  - Runs read label-free test inputs in-process and write their logits once into sealed files.
  - Test logits of tuning runs that are not selected are deleted unopened after selection; only their hashes are kept.
  - So the test is scored only for the prespecified final configurations.
  - `unseal.py` reads the labels once. It runs only after two conditions hold. First, every declared run (the core plus whichever extensions were run) is complete or has been cut on budget grounds. Second, the hash of the CV/validation report has been recorded.
- **PQA-L CV pool** (500 examples, `data/pqal_pool.jsonl`).
  - Used only for selection and replication.
  - The old dev-100 is retired as a selection set. Its examples sit inside the folds, which is disclosed, and a sensitivity out-of-fold (OOF) analysis excludes them.
- **LM test** (`data/lm/test.bin`, 736,882 tokens): fresh and sealed. Only `sealed_eval.py` reads it, and it returns hashes, never values.
- **LM validation** (725,568 tokens): tuning and monitoring.
- **SciFact dev** (340 pairs, E4 only): used once, as a test.

**Folds: option (b).**
- A seeded, stratified 10-fold split of the 500 non-test examples, with no test access.
- Per label, the PMIDs are sorted and then shuffled with `random.Random(20260925)`. The three lists are concatenated and position j goes to fold j mod 10, which gives 50 examples per fold.
- Why (b):
  - Comparability comes from the official test and the 450/50 training protocol, and BLURB does not say which fold it used.
  - Reproducing `split_dataset.py` would mean reading the test labels, for no gain.

**Decontamination is done** (`data/decontamination.json`).
- PQA-A: 0 flagged.
- PQA-U: 2 near-duplicates of test records, excluded.
- LM validation and test near-duplicates of LM-train: 6 and 2, removed.

**Before the freeze.**
- Add `data/raw/pqa_labeled.parquet`, which holds all 1,000 labelled records, to `env.SEALED`.
- **Incident 1.** While writing this protocol, a shell `cat` printed `data/lm/test_docs.json` and bypassed the Python-level seal.
  - It shows PMIDs and offsets only, with no text.
  - No model or metric was involved, and the PMIDs can be derived from the public hash rule anyway.
  - It is disclosed.
- New rule: no shell access to sealed paths.

**Arms and calibration.**
- Every attention arm *replaces* attention-probability dropout.
- Hidden and residual dropout stay at 0.1 in all arms, so N means "no attention-specific regularization".
- All arms run through the same eager, materialized `attnreg.AttnReg` path.

| Arm | `attnreg` kind | Calibrated to realized mean displacement ℓ |
|---|---|---|
| N | none | — |
| AD | dropout (post-softmax, 1/(1−p)) | p = ℓ. Rows are not normalized, so d is the half-L1 distance, not a TV; E[d] = p and B = 0 exactly. |
| DK | dropkey with the published linear depth schedule `depth_schedule(p_max, l, L)` | p_max such that the mean over layers of E[m] = ℓ, with E[m] = p − pⁿa_max |
| RA | relaxed (fixed γ, uniform over visible keys) | γ = ℓ / mean TV(a, U) |
| HU | hard (top-4, τ = 1) | p_lo such that E[m] = ℓ, with E[m] = pA₄ − 1{n≤4}pⁿa_max |
| HB | hard_budget (CapDrop) | p_hi = 0.5, raised if needed to the smallest p in {0.6, …, 1.0} with E[m](p) ≥ 1.5ℓ; τ such that E[min(m, τ)] = ℓ |
| MX (encoder only) | none + `apply_mixout` on every encoder `nn.Linear`, anchored to the fp32 pretrained weights | p ∈ {0.5, 0.7, 0.9} |

- **Levels.** ℓ ∈ {0.05, 0.10, 0.20} in both regimes. Every family gets exactly 3 configurations, so the tuning budget is equal.
- **Infeasible levels.** If a hard arm cannot reach ℓ even at p = 1, it runs at p = 1 and is flagged.
- **k = 4, fixed.**
  - Throughput was measured at this setting.
  - It allows exact enumeration of all 16 patterns and targets sink mass.
  - The census reports A₄ against A₈ per layer. If median A₄ < 0.2 in a regime, that regime's H1 is registered as "equivalent".
- **Operating point.** p_hi = 0.5 was chosen after the Track-1 null. This is disclosed as motivated by that null, not confirmed by it.
- **Calibration procedure.**
  - `calibrate.py` bisects on the exact expectations from `census.py`: `hard_from_top` in fp64, the dropout/dropkey closed forms, and `relaxed_from_stats`.
  - Averages run over all non-padding rows, heads and layers.
  - It writes `results/predictions.json`, which is hashed before any outcome run outside the N arm. For each level the file records:
    - the parameters;
    - P(m>τ), E[(m−τ)₊], κ, B and S2;
    - the H1 direction.

**Seeds and pairing.**
- Encoder and student fold i uses seed 1000+i. It seeds `torch.manual_seed` (head initialization), the CPU data generator (seed+10000) and `attnreg.NoiseSource(seed)`.
- LM seed s ∈ {1..10} uses 2000+s.
- Noise is counter-based, keyed by (seed, step, layer, pass, stream), and this is tested for two properties:
  - it is safe to recompute under gradient checkpointing;
  - arms get common random numbers.
- Initial-weight and first-batch hashes must match across arms.
- The order of arms within each fold or seed rotates as a Latin square.

**Run hygiene (every runner).**
- **Setup.** Run `env.setup_model_process()` first. It blocks Arrow, applies the seal and the torchvision patch, and prepares data before CUDA starts.
- **Memory.**
  - Cap VRAM at 3.27 GB with `env.cap_vram()`.
  - Assert that free memory from `mem_get_info` is at least cap + 150 MB, at start and every 200 steps.
  - On OOM, retry once with half the token budget and the same effective batch, and log it.
- **Throughput and logging.**
  - `runlib.ThroughputGuard` aborts and re-queues any run below 60% of the S0 rate.
  - `runlib.GpuLogger` logs at 1 Hz, and `runlib.Ledger` records GPU-seconds.
  - A persistent worker per study loads the weights once.
- **Registration and failures.**
  - `runlib.Registry` declares every run before it executes, and results are written once.
  - No seed is replaced. A NaN or divergence counts as a failed run and is kept. A crash without NaN gets one logged rerun with the same seed.
- **Kernels.** Training uses fast non-deterministic kernels. Evaluation uses `use_deterministic_algorithms(True)` with a fixed batch order, so evaluation logits replay bit-exactly.
- **Pins.** torch 2.13.0+cu126, transformers 4.57.6, Python 3.12.10, BiomedBERT `e1354b7a` .bin SHA `ad7bbb66…`.
- **Timing and cost reporting.**
  - Timings come only from runs that had the GPU to themselves.
  - Arms are matched on steps and tokens, and GPU-hours are reported per arm.

### 3.1 S0: engineering, census, freeze (priority 0)

1. **Gate G0.** All of the following must pass:
   - the 48 existing tests plus the new tests in §5;
   - one-step training equivalence of `attnreg.hard_budget` against the oracle `strengthened/attentiondrop.py`. It uses injected random draws on one LM batch and one BiomedBERT microbatch, and loss and gradients must agree within 1e-5 in fp32;
   - rows with m ≥ 0.99 and α = 1, in fp32 and bf16;
   - reproduction of the Track-1 `budget_table.tex` values E[m] = 0.0224 / 0.0531 / 0.0172 / 0.0552 from the archived checkpoints (read-only).

   If G0 fails, use the oracle with `topk` and no host sync, and apply cut C1.
2. **Re-benchmark** the paths that have not been measured, under the cap and with real lengths:
   - LM dropout, dropkey and relaxed;
   - the student (eager and hard_budget);
   - Mixout, R-Drop and dropkey_budget;
   - peak memory at 512 tokens.

   Then re-project the budget.
3. **Freeze.** `freeze.py` hashes PROTOCOL.md, `unified/*.py`, all manifests and `pqal_folds.json` into `FREEZE.json`. The hash is registered on OSF (free) before any outcome run.
4. **Census.** It runs after the freeze, in eval mode, before any outcome is seen.
   - PQA-U records, minus the 2 flagged, are ranked by sha256("unified-u-v1:"+pmid). The first 256 are the census inputs and the next 10,000 are the KD subset.
   - The census runs on:
     - the pretrained BiomedBERT, on the 256 census inputs (`census.RowSampler`, 2,048 rows per layer);
     - the winning N run from LM base tuning (seed 1), at checkpoints at 10/25/50/100% of training held in RAM, on 64 training windows, averaged;
     - the fold-0 student after phase K. The sequence is T₀–T₄ → G3 → T₅–T₉ → all phase-K runs → student census → phase L.
   - `calibrate.py` then writes `predictions.json`, which is hashed.

### 3.2 S1: encoder fine-tuning (priority 1, primary)

**Model and input.**
- The local BiomedBERT-base-uncased-abstract-fulltext `e1354b7a`, loaded in eager mode with `bert_adapter.install`.
- Head: [CLS] → Dropout(0.1) → fp32 Linear(768, 3), initialized N(0, 0.02). No pooler.
- Encoding via `pubmedqa_inputs.encode`:
  - single segment, max 512 tokens;
  - the question is never truncated, and evidence is truncated at head and tail;
  - mean length is 292 tokens, and 7 of 500 inputs are truncated.

**Stabilized recipe (all arms).**
- Optimizer: bias-corrected torch AdamW, lr 2e-5, β (0.9, 0.999), ε 1e-6, weight decay 0.01 through `parameter_groups`.
- Data order: 20 seeded permutations of the 450 examples form a stream of 9,000, cut into 563 batches of 16 (the last batch has 8).
- Schedule: `linear_warmup_decay` with 10% warmup (56 steps); gradient clip 1.0.
- Numerics: fp32 master weights, bf16 autocast, gradient checkpointing, and `microbatches(max_tokens=2048)` with exact accumulation.
- The final checkpoint is used, with no early stopping.

**Gate G_E.**
- N and AD@0.10 run first, on folds 0–1.
- Suppose at least 3 of these 4 runs either fail (dev accuracy ≤ the fold majority) or collapse. Then the base recipe switches once, for every arm, to re-initializing the top 2 layers with lr 3e-5. The switch is disclosed.

**Tuning and finals.**
- **Tuning.** AD, DK, RA, HU, HB and MX each run their 3 levels on folds 0–1, for 36 runs.
  - For each family, ℓ\* is the level with the lowest mean dev Brier over those 100 examples.
  - If the top two are within 0.005, the lower dose is chosen (for MX, the lower p).
  - `select.py` prints only the choice.
- **Finals.**
  - N runs on folds 0–9.
  - Each X\* runs on folds 2–9; its tuning runs supply folds 0–1.
  - HU@0.10 and HB@0.10 run on folds 2–9 if they were not selected.
- **C\*_E** is the arm in {N, AD\*, DK\*, RA\*, HU\*, MX\*} with the lowest mean dev Brier on folds 2–9 only.
- **Q0** is the first fine-tuned question-only baseline: the N recipe with question-only inputs for both training and evaluation, on folds 0–9.
- **L0** is the legacy Track-2 recipe, reported descriptively:
  - LoRA r8/α16 on query and value;
  - lr 1e-4, 5 epochs;
  - two segment IDs, attention dropout 0.1, frozen bf16 backbone;
  - folds 0–9.

**Evaluation.** Every run is evaluated on its dev fold and on the test, in one pass inside the run.
- Input conditions:
  - original;
  - question-only;
  - 3 derangements (another example's evidence), stratified by label and matched on evidence-length quintile (`stratified_derangement`, seeds 1–3; test donors from `prepare_eval_inputs.py`);
  - RESULTS section deleted.
- Metrics:
  - Brier, accuracy, macro-F1, NLL, ECE (15 bins), failure and collapse;
  - EUG, flip rate, the rate of following the donor's label, and KL(full‖mismatch);
  - per-layer budget and concentration statistics each epoch;
  - attention mass from [CLS] to evidence, question and special tokens (eager).

**Analysis.**
- **Confirmatory.** H1-E, H2-E and H3-E on test Brier, conditional on the test set, with the two-level bootstrap as a co-criterion (10⁴ resamples).
- **Replication.** Per-fold CV OOF Brier over folds 2–9 (folds 0–9 for untuned contrasts), with Nadeau–Bengio corrected CIs (variance factor 1/10 + 50/450).
- **Sensitivity.**
  - Test results excluding fold-models 0–1, because those reuse tuning runs.
  - OOF results excluding the dev-100 examples.
- **Manipulation check.** The time-averaged ratio of realized D̄, HB over HU at 0.10, must fall in [0.8, 1.25].
  - If it does not, the as-designed H1 stays confirmatory.
  - A sensitivity regression Δy_i = β₀ + β₁ΔD̄_i + ε_i then reports β₀.
- **Secondary.**
  - McNemar on seed-majority-vote predictions.
  - Accuracy compared with the literature (BLURB 61.02, Tinn 63.92, majority 55.20), labelled as a cross-protocol comparison.

### 3.3 S2: data-limited from-scratch LM (priority 2, primary)

**Data.**
- LM-train is the shortest prefix of `data/lm/train.bin`, in its stored hash order, that reaches at least 1,000,000 tokens. That is about 2.85k abstracts.
- 8 epochs = 8.0M token-steps = 977 steps of 32×256.
- Each epoch starts at a random offset in [0, 256) and shuffles windows with the CPU generator (seed+10000+epoch). Windows are identical across arms.

**Model and optimization.**
- `lm_model.LM`: 6 layers, width 384, 6 heads, context 256, vocab 16,384, untied embeddings. That is 23.3M parameters in total, 10.6M excluding embeddings. Residual dropout 0.1.
- Optimizer: AdamW (0.9, 0.95), weight decay 0.1 on matrices.
- Schedule: 49 warmup steps, then cosine decay to 0.1× peak; gradient clip 1.0.
- Numerics: bf16 with TF32, chunked loss.
- Microbatch: the largest of {32, 16, 8} that passes S0, with an effective batch of 32.

**Base tuning and gate G1.**
- Learning rate for N ∈ {5e-4, 1e-3, 2e-3}, seed 1, chosen on final validation NLL.
- G1 checks that the regime overfits: the winner's final validation NLL must be at least 0.02 above its minimum over the 10 evaluations.
- If it is not, the corpus is halved to ≥ 500k tokens for 16 epochs, and base tuning is repeated once.

**Arms, tuning and finals.**
- Arms: N, AD, DK, RA, HU, HB. There is no Mixout, because there is no pretrained anchor.
- **Tuning.** 3 levels at seed 1, chosen on final validation NLL. Ties within 0.005 nats go to the lower dose.
- **Finals.** Seeds 1–10.
  - Seed 1 reuses the tuning run of the selected level.
  - HU@0.10 and HB@0.10 run on every seed.
- **C\*_L** is the arm in {N, AD\*, DK\*, RA\*, HU\*} with the lowest mean validation NLL over seeds 2–10.

**Evaluation.**
- Validation NLL every 98 steps, over disjoint blocks, scoring every token.
- Test NLL through `sealed_eval.py`, at the final step (primary) and at the best-validation weights kept in CPU RAM (secondary).
- Per-layer diagnostics every 98 steps:
  - realized TV, S2, activation rate, entropy, top-4 mass, BOS/sink mass;
  - the gap between training loss with the perturbation on and in eval mode.

**Statistics.** As in §2, plus:
- a document-level paired bootstrap over the 2,098 test documents;
- the manipulation check;
- a sensitivity analysis that excludes seed 1.

### 3.4 S3: compression (priority 3, secondary)

**Teachers.**
- T_i is BiomedBERT with the privileged input: question + evidence + conclusion. The conclusion is never truncated (mean 339.5 tokens).
- Training: the stabilized recipe on SDPA with attention dropout 0.1, on the fold-i training set, seed 3200+i, for i = 0..9.
- Teachers are never evaluated on the test.
- **Gate G3:** the pooled privileged OOF accuracy of T₀–T₄ (250 examples) must be at least 65%. If not, the students are cut.
- **Cache (v2):** raw fp32 logits, T = 2, and the input SHA-256, for the KD subset and the fold-i training set.

**Students.**
- Architecture: BiomedBERT embeddings plus layers {0, 2, 4, 6, 8, 10} (66.4M parameters), with a fresh head.
- **Phase K:** KD on soft labels only (`kd_loss`, T = 2), 1 epoch over the KD subset, lr 5e-5, batch 32, 6% warmup, SDPA, seed 3000+i. All arms of a fold share this phase.
- **Phase L:** the fold-i training set, with loss CE + 0.5·`kd_loss`, the stabilized recipe for 20 epochs, seed 3100+i.

**Arms (phase L).**
- S-N.
- S-AD at p = 0.10, the BERT default. Its realized displacement is also 0.10.
- S-HU@0.10 and S-HB@0.10, calibrated on the student census.
- There is no re-tuning, which is disclosed.

**Endpoints.**
- Quality: test Brier and accuracy, plus CV dev.
- Retention relative to S1 N on the same folds.
- Cost:
  - parameters (total and non-embedding) and fp16 size in MB;
  - GPU latency at batch 1 and throughput at batch 32 (384 tokens);
  - CPU latency (4 threads).
- Agreement with the teacher.

The pilot's 46.93% is quoted as history only. KD effects depend on each fold's single teacher, so the fold is the unit of analysis.

### 3.5 Extensions

These are declared now, with fixed run counts. They run in this order, only if the budget allows, and all before the single unseal.

| ID | Content | Runs |
|---|---|---|
| E1 | R-Drop in S1: α ∈ {0.1, 0.5, 1}, noise from AD p = 0.1 plus hidden dropout, 2 passes. Tuned like the other families and reported as HB\* − RD\*. C\*_E is not redefined. | 14 |
| E2 | Boundary ladder under the fragile recipe: `BertAdam` without bias correction, lr 2e-5, 3 epochs, two segment IDs. Arms: F-AD (p = 0.1), F-HB@0.10, F-MX (p = 0.9, plus AD 0.1). Each arm has 20 runs (folds 0–9 × seeds 4000+i and 4100+i). | 60 |
| E3 | Targeting and budget-form ablation at 0.10: `dropkey_budget` (uniform candidate, p = 0.5) and `relaxed_budget` (row-adaptive RA), each against HB@0.10, in S1 (10 folds) and S2 (10 seeds) | 40 |
| E4 | SciFact oracle claim–abstract pairs (ETag `cb7da4d8…`, CC BY 4.0 / ODC-By). Train on 919 pairs for 10 epochs. Arms N, RA@ℓ_RA\*, HU@0.10, HB@0.10 × seeds 5001–5010. Dev-340 is used once. Diagnostics: claim-only input, and deletion of gold rationales (shift toward NEI). | 40 |
| E5 | BioLinkBERT-base (`b71f5d70`, Apache-2.0): N, AD@ℓ_AD\*, HU@0.10, HB@0.10 × 10 folds. New census; levels transferred. | 40 |
| E6 | LM on an 8.0M-token prefix for 1 epoch, a regime that does not overfit: N, AD\*, RA\*, HU@0.10, HB@0.10 × 10 seeds | 50 |
| E7 | S-noKD student (phase L without phase K) × 10 folds | 10 |

**E2 inference.**
- A failure is test accuracy ≤ 55.20.
- The comparison is inconclusive if F-AD fails in fewer than 30% of its runs.
- F-HB and F-MX are each compared with F-AD using an exact permutation test stratified by fold (10⁵ permutations), one-sided, with Holm over the 2 tests.
- "Does not reduce failures" is claimed only if the upper 90% bound of the reduction is below 25 percentage points.
- Power is about 0.85 for a drop from 50% to 10%.

## 4. Compute and disk budget

**Per-run minutes.**
- **Encoder:** 20 × the measured time per 450-example epoch, plus 1.0 min for evaluation (6 conditions × 550 examples) and overhead.
  - Measured: N 9.4, AD 10.8, DK 13.4, RA 12.6, HU/HB 17.2, dropkey_budget 17.4, relaxed_budget 13.8.
  - Estimated: MX 13.0, R-Drop 20.6, Q0 1.5, L0 2.5.
- **LM:** 8.0M tokens, plus 1.8 min.
  - From measured rates: N 5.9 (0.86 h per 100M tokens), HU/HB 7.5 (1.19 h per 100M tokens).
  - Estimated: AD 6.4, DK 7.8, RA 6.2.
- **Student:** about 1.9× the encoder rate (estimate).
- **Teacher:** 6.5.

The S0 re-benchmark replaces every estimate.

| Item | Runs | GPU-h, worst case | Expected |
|---|---|---|---|
| S0 | — | 0.8 | 0.8 |
| S1: N, folds 0–9 | 10 | 1.6 | 1.6 |
| S1: tuning, 6 families × 3 levels × folds 0–1 | 36 | 8.4 | 8.4 |
| S1: finals, 6 families × folds 2–9 | 48 | 11.2 | 11.2 |
| S1: completing HU@0.10 and HB@0.10 | ≤16 | 4.6 | 3.1 |
| S1: Q0 and L0 | 20 | 0.7 | 0.7 |
| S2: base learning rate (3) + tuning (15) | 18 | 2.1 | 2.1 |
| S2: finals, N + 5 families × seeds 2–10 | 54 | 6.2 | 6.2 |
| S2: completing HU@0.10 and HB@0.10 | ≤18 | 2.25 | 1.5 |
| S3: 10 teachers + caches + 10 phase-K + 40 phase-L runs | 60 | 6.9 | 6.9 |
| Audits: 2 evaluation replays + 2 retrain replays per study | 6 | 1.2 | 1.2 |
| **Core subtotal** | | **46.0** | **43.7** |
| Contingency, 10% (gate fallbacks: G_E ≈ 0.7, G1 ≈ 0.3; OOM and crash reruns) | | 4.6 | 4.6 |
| **Core total** | | **50.6** | **48.3** |
| Extensions E1 / E2 / E3 / E4 / E5 / E6 / E7 | | 4.8 / 2.9 / 7.6 / 9.6 / 9.1 / 5.6 / 0.9 | |
| **Hard cap** | | **90 h**, or 12 days after the freeze | |

**Disk (new files only).**
- Model-only fp32 audit checkpoints, 2 per study: 1.6 GB.
- One transient phase-K checkpoint: 0.27 GB.
- Sealed logits and logs: ≤ 0.1 GB.
- Extensions: BioLinkBERT 0.43 GB, plus a 0.44 GB audit checkpoint.
- Peak new disk: ≤ 2.9 GB.
- Runners refuse to start with less than 2 GB free. Downloads use `hf_hub_download(local_dir=…, revision=sha)`.

**Stop and cut rules.**
1. **Order.** S0 → S1 → S2 → S3 → audits → E1…E7 → `unseal.py`.
2. **Re-projection.** Re-project after S0, and check the ledger after every study.
   - Trigger: the projected core, including contingency, exceeds 51 h, or actual cost exceeds 1.15× the projection.
   - Response: apply the next cut, before the affected study starts and equally to all arms:
     - C1: S3 drops S-AD.
     - C2: S3 runs folds 0–4 only and becomes descriptive.
     - C3: Q0 and L0 are dropped.
     - C4: S2 is reduced to 6.0M token-steps.
3. **Never cut:**
   - in S1 and S2: N, RA\*, HU@0.10, HB@0.10, HB\*, and the C\* pools;
   - the number of folds or seeds, which stays at 10.
4. **Throughput checks.**
   - A run that takes more than 1.5× its estimate triggers a re-benchmark.
   - Three throughput-guard trips pause the campaign.
5. **Extensions and outcome-blindness.**
   - An extension starts only if its full cost fits the remaining budget.
   - Every rule is driven by budget, never by outcomes, and there are no interim looks.

## 5. Implementation plan

**Existing, used as-is:**
- `env.py`, `attnreg.py`, `census.py`, `bert_adapter.py`, `training_utils.py`, `pubmedqa_inputs.py`, `lm_model.py`, `runlib.py`;
- `data/` and its manifests;
- `strengthened/attentiondrop.py` and `analyze_budget.py`, imported read-only as oracles.

No legacy file that has been hashed is edited.

| File | Purpose | Tests |
|---|---|---|
| `env.py` (pre-freeze edit) | Seal the parquet. A `sealed_access()` context that only `sealed_eval.py`, `prepare_eval_inputs.py` and `unseal.py` may use. Checks for VRAM headroom and a disk floor. | Subprocess test: a sealed read raises outside the context and succeeds inside it |
| `attnreg.py` (pre-freeze edit) | Accumulate realized d² for S2 | Rows with m ≥ 0.99, α = 1: outputs and gradients match the oracle in fp32 and bf16; S2 equals the enumeration |
| `folds.py` | Writes `data/pqal_folds.json` | 10 folds × 50; per-label counts differ by ≤ 1; folds are disjoint and together equal the pool; runs with the seal active |
| `prepare_eval_inputs.py` | The only logged exception to the seal after `decontaminate.py`. Writes label-free test inputs with section labels, 3 derangement donor lists, and section labels for the pool. | Output has no label or `long_answer`; no donor equals its own record; donor label ≠ record label, asserted internally |
| `calibrate.py` | Census → `predictions.json` | Bisection within 1e-4; AD p = ℓ; monotonicity; reproduces the Track-1 table |
| `sealed_eval.py` | Writes test logits and NLL once; returns hashes | A second write is refused; no values are returned |
| `worker.py` | Persistent worker | Two runs in one worker give the same result as two fresh processes (tiny model, CPU) |
| `train_encoder.py` | S1, Q0, L0, extension arms, all conditions | 20-step smoke test per arm; identical initialization and first-batch hashes; accumulation equals a fixed microbatch (1e-5); HU deletions ⊆ HB proposals |
| `train_lm.py` | S2, E6 | Identical batches across arms; 977 steps; every token scored; causality test (existing) |
| `distill.py` | Teachers, cache v2, students | Student weights equal the source layers; KD = 0 when student equals teacher; the validator rejects a hash or T mismatch; the conclusion appears only in teacher inputs |
| `legacy_lora.py` | L0 | B = 0 gives identical logits; survives the class swap |
| `select.py` | Level and C\* selection rules | Synthetic tables with ties |
| `stats_unified.py` | Sign-flip, Holm, TOST, two-level and document bootstrap, Nadeau–Bengio, stratified permutation, Spearman permutation, Brown–Forsythe, McNemar | Agrees with scipy or brute force |
| `unseal.py` | Reads labels once and writes metrics once | Refuses an incomplete registry or a missing CV-report hash |
| `audit.py`, `tables.py` (no torch) | Bit-exact evaluation replay of 2 runs per study, chosen in advance from the FREEZE hash. An independent agent draws retrain replays after the core (tolerance Brier ±0.01, NLL ±0.01). Tables are recomputed from the sealed logits. | Hash chain; tables match |
| `freeze.py`, `benchmark_s0.py`, `prepare_scifact.py` | Freeze, re-benchmark, E4 data | Hash coverage; pinned ETag |

## 6. Paper outline and claim boundaries

1. **Introduction.** Nominal rate versus realized displacement; contributions.
2. **Related work.**
   - DropAttention, DropKey, DropHead, R-Drop, UniDrop, TLM, HiddenKey.
   - Relaxed Attention, the special case b = U with fixed γ; the budget makes γ adapt per row.
   - Mixout, the parameter-space analogue.
   - Fine-tuning instability: Dodge, Mosbach, Zhang.
   - PubMedQA, BLURB, Tinn; knowledge distillation with privileged information.
   - Disclosure of arXiv 2504.12088.
3. **Displacement accounting.** Table 1 lists the exact laws. The propositions are elementary and local to a single head, with no end-to-end guarantee:
   - TV = m;
   - TV(a, ã) = min(m, τ) and TV(b, ã) = (m−τ)₊;
   - ‖Δz‖ ≤ D·min(m, τ);
   - d/dτ E[min(m, τ)] = P(m>τ);
   - E[min²] ≤ τE[min];
   - P(m>τ) ≤ 1 − (1−p)ᵏ;
   - the closed form and its cost.

   The appendix covers three further points:
   - diffusion entropy;
   - that a triangular doubly-stochastic matrix is the identity;
   - why second-order noise expansions do not cover deletion's mean shift.
4. **Preregistered design.** Held-out ledger, calibration, statistics.
5. **Results.**
   - Figures:
     - F1, census: m against min(m, τ), and P(m>τ) by layer and regime;
     - F2, forest plot of families F1 and F2 with their margins;
     - F3, loss against realized D̄ across families;
     - F4, LM curves;
     - F5, PubMedQA per-run strip plot with the 55.20 line;
     - F6, evidence use;
     - F7, compression frontier.
   - Tables:
     - arms and GPU-hours;
     - LM results;
     - PubMedQA results (Brier, accuracy, F1, failures, SD);
     - compression;
     - budget activity;
     - L0 and Q0 against the literature.
6. **Discussion.** Report realized displacement, not rates; limitations.

**Appendices:** proofs, the protocol and its deviations, all runs, the Track-1 and Track-2 history (labelled pre-protocol and exploratory), and the compute ledger.

**Front matter:** abstract ≤ 250 words, ≤ 6 keywords, highlights, CRediT, a corrected generative-AI declaration naming the assistant actually used, and data and code availability.

**Claim boundaries:**
- ≤ 110M parameters, English biomedical abstracts, one 4 GB GPU, non-deterministic bf16 training.
- This is the PubMedQA test's second use, and inference on it is conditional on the test set.
- Results hold only at the prespecified levels.
- CapDrop matches the reference in distribution, not stream for stream.
- Mediation is model-based.
- No state-of-the-art claim (BioLinkBERT reaches 70.2) and no clinical claim.
- Novelty means "not found in the literature we searched". A targeted search (trust-region, clipped or bounded attention noise) is repeated before submission.

## 7. Top risks and mitigations

1. **Null everywhere** (the most likely outcome). The TOST margins, the mechanism test and the venue table make every outcome publishable.
2. **The encoder's minimum detectable effect is too large.** Mitigations: per-run test Brier on 500 examples and n = 10. The MDE is stated, and "inconclusive" is a legitimate verdict.
3. **Calibration drifts during training.** The manipulation check applies, with a prespecified sensitivity analysis. The as-designed analysis stays confirmatory.
4. **Compute overrun or silent VRAM spill.** S0 re-projection, the ledger, cuts C1–C4, protected cells, the VRAM headroom assertion and the throughput guard.
5. **Bug in the closed form.** The log-space form, gate G0 (including m → 1), and replays.
6. **Leakage.** Decontamination, the sealed parquet and accessor, a single unseal, and the ban on shell access.
7. **Logistics.**
   - Disclose 2504.12088 and the change in author list.
   - Verify the Heidelberg corresponding-author/DEAL route and the JCR quartiles.
   - No paid APIs; report every run; replace no seeds.

**Resolution of review items.**
- **Accepted:**
  - the matched-mean causal test, with same-p τ = 1 contrasts dropped from the confirmatory family;
  - a small primary family;
  - n = 10 everywhere, with failure-rate claims only in E2 (20 runs per arm);
  - Brier as the primary endpoint (a proper score), with fixed margins;
  - the multi-epoch LM in the core, with gate G1;
  - equally budgeted tuned baselines, with C\* selected on folds not used for tuning;
  - a privileged teacher gated on 250 OOF examples;
  - common random numbers, the log-space closed form, and a single unseal.
- **Modified or rejected, with reasons:**
  - R-Drop moves to E1, declared before the unseal. It costs 2×, and its noise source is the already-tuned AD family.
  - Confirmation uses the test, with CV as replication, rather than CV alone. The test has 10× more examples per run.
  - k = 16 is rejected: it was never benchmarked and cannot be enumerated exactly. k = 4 targets the few high-mass keys, and A₄ vs A₈ is reported.
  - The official folds are replaced by option (b).
  - No single-epoch LM in the core; E6 covers that regime.
  - n is fixed with a stated MDE, instead of an internal pilot.

---

# Part B: v1.1 amendments (binding)

**Status.** Written 2026-09-25, before any outcome run of this protocol. Part B supersedes Part A wherever the two conflict.

**Sources.**
- Reviewer-2 red-team report (`design/result_7.md`), items R1–R17. Every item was accepted unless marked otherwise below.
- A self-audit that replaces the compute red-team and the integrity red-team. Both agents failed on the account's session limit and returned nothing.

**Before this freeze, the only things executed were:**
- unit tests;
- outcome-blind smoke and throughput runs (seed 999, results kept in the scratch folder, no metrics inspected);
- data preparation and decontamination;
- the incident in §B12.

## B1. The method, the arms, and the 2×2 core

**Method name.**
- The method is called **CapDrop** ("TV-capped attention deletion").
- The earlier arXiv preprint 2504.12088 ("AttentionDrop") is cited as the source of the top-k masking candidate. It has no displacement budget. Its blur and KL components are omitted here; KL consistency is covered by E1.

**New arm HF (fixed-α deletion).**
- Definition: ã = (1−γ)a + γ·b.
  - b is the same p_hi top-4 deletion candidate as HB.
  - γ = ℓ / E[m](p_hi), taken from the census.
  - HF uses the same NoiseSource as HB.
- HB and HF therefore share p, the eligible keys, the uniform draws and the calibrated mean displacement. They differ **only** in the mixing rule: α = min(1, τ/m) for HB versus α = γ·1{m>0} for HF.
- Implementation: attnreg kind `hard_mix` with parameter `gamma`, and a `census.hard_from_top(..., alpha_fixed=γ)` option.
- Tests:
  - oracle equality with `budget_mix` at fixed α;
  - common random numbers: HF and HB produce identical deletion sets;
  - E[d] from census equals the Monte Carlo estimate.

**Core design at ℓ = 0.10 in S1 and S2.** Candidate {uniform U, deletion} × mixing rule {fixed, capped}:

| | fixed α | capped α |
|---|---|---|
| **uniform** | RA@0.10 | RB@0.10 (`relaxed_budget`, τ = 0.10) |
| **deletion** | HF@0.10 | HB@0.10 |

- All four cells run on every unit.
- RB moves from E3 into the core. E3 keeps only `dropkey_budget`.

**Revised confirmatory family F1.**
- 6 tests, Holm-adjusted at α = 0.05. The test is the exact two-sided sign-flip test; the unit is the fold in the encoder and the seed in the LM.
- H1-E / H1-L: **HB@0.10 − HF@0.10.** This isolates the cap.
- H2-E / H2-L: **HB@0.10 − RA\***, the tuned special case.
- H3-E / H3-L: **HB@0.10 − C\***, the strongest tuned control.

**No tuning of CapDrop (R7a).**
- HB\* := HB@0.10 at its prespecified operating point. This is conservative against the method.
- HB still runs at 0.05 and 0.20 on the tuning units, but descriptively only (dose–response).

**Secondary family F2.**
- Separate Holm correction.
- Contrasts:
  - HB@0.10 − HU@0.10;
  - RB@0.10 − RA@0.10;
  - the 2×2 interaction (HB − HF) − (RB − RA);
  - evidence-use gap (EUG) HB vs N;
  - flip rate HB vs N;
  - S-a and S-b, if E-S3 runs.
- Recommended wording for HB − HU: common random numbers on the uniform stream (R10). It must not be described as "subset deletions".

## B2. Registered predictions (replaces the Part A H1 rules; R2)

**Convex-order proposition.** This is stated in the paper and in `predictions.json`.
- The class: every rule with 0 ≤ d ≤ m pathwise and E[d] = ℓ.
- Within that class, min(m, τ_ℓ) ≤_cx d.
- So CapDrop has the smallest second moment S2 among all such rules at matched mean.
- HF belongs to the class. HU@p_lo belongs to it under common random numbers wherever the top-k sets coincide.

**Predicted separation.**
- δ = (S2(HF) − S2(HB)) / ℓ², computed from the census.
- The census premise is κ_E ≥ 2κ_L (Part A exploratory).

**Registered expectations.** The tests themselves stay two-sided.

| Regime | Expectation |
|---|---|
| Encoder (H1-E) | "HB better" if δ ≥ 0.5, otherwise "equivalent" |
| LM (H1-L) | "HB not better" (equivalent or worse) if δ ≥ 0.5, otherwise "equivalent" |

- In the LM, an overfitting regime, the second-order term is expected to do the regularizing, and the cap minimizes it.
- The threshold c = 0.5 is fixed now.

**Second-order justification (Prop.).**
- For distribution-preserving perturbations: E[ℓ(z+Δz)] − ℓ(z) = ∇ℓᵀE[Δz] + ½E[ΔzᵀHΔz] + o.
- The two terms are bounded by ‖E[Δz]‖ ≤ D·B and E‖Δz‖² ≤ D²·S2.
- This is a heuristic motivation for the displacement-accounting model (DAM), not a guarantee. It does not cover the unnormalized dropout rows.

## B3. Wording of the thesis (R3)

**Title.** *Capping realized attention displacement: a preregistered, displacement-calibrated test of TV-capped attention deletion in data-limited biomedical transformers.*

**Wording rules.**
- Say "d bounds the forward effect" (‖Δz‖ ≤ D·d), never "the model feels only d".
- "Rate ≠ displacement" is claimed only for top-k deletion and Relaxed Attention. For dropout, E[d] = p exactly, and for DropKey E[d] ≈ p.
- AD rows are unnormalized: ‖Δz‖ ≤ 2d·max‖v‖. Cross-family matching at ℓ is labelled "matched half-L1 displacement".

**Value-weighted census** (`census_output.py`, eval mode, pretrained BiomedBERT, the 256 census inputs, and the LM tuning-winner checkpoints).
- For each family at ℓ = 0.10, record:
  - E‖Δz‖ / ‖z‖ per head output: exact for the hard/HF/HB kinds by 16-pattern enumeration, Monte Carlo (64 draws) for AD and DK;
  - Spearman ρ between d and ‖Δz‖;
  - the share of deleted mass on [CLS], [SEP] and BOS/EOT.
- Registered rule: if the pooled ρ < 0.5, the premise of displacement accounting is reported as refuted.
- Registered secondary prediction: HB lowers the share of special-token deletion relative to HF. It is reported descriptively.

## B4. Endpoints and calibration effects (R4, R6)

**Encoder: temperature scaling.**
- Per run, fit a temperature T ∈ [0.25, 4] on that run's 50-example dev fold by golden-section search on NLL.
- `unseal.py` applies it to the test logits.
- A "Better" claim (F1, and outcome-table rows 1–2) requires both the raw-Brier contrast and the TS-Brier contrast to be Better (Holm p < 0.05). If only raw Brier is Better, the result is worded "calibration only".
- Discrimination checks: macro one-vs-rest AUROC and macro-F1.

**Label smoothing.** A label-smoothing family LS ε ∈ {0.05, 0.10, 0.20} joins the encoder C\*_E pool. It uses the N attention setting and is tuned like the other families.

**LM regime.**
- A residual-dropout family RD {0.2, 0.3, 0.4} (no attention regularizer) joins the C\*_L pool.
- The early-stopped test NLL (at the best-validation step) is co-primary with the final-step test NLL. A Better or Worse verdict requires the two to agree.
- The LM equivalence margin is ±0.010 nats/token. The σ_d assumption (0.004–0.008) is stated, and the observed value is reported.
- Test NLL is also reported by LM-train frequency bucket (descriptive).
- The regime is motivated by data-constrained scaling (Muennighoff et al., 2023).

## B5. Power, units, top-up and outcome rows (R5)

**Power.**
- Scenarios for σ_d ∈ {0.01, 0.02, 0.03} Brier are reported with the verdict probabilities from R5.
- "Inconclusive" is a legitimate verdict.

**Units.**
- The unit is the fold (encoder) or the seed (LM).
- Top-up P1 is the first extension. It is budget-driven and never decided on outcomes: a second seed (4200+i) for HB@0.10 and HF@0.10 on all 10 folds, 20 runs.
- If P1 completes, H1-E uses the fold means over the two seeds. H2 and H3 always use seed 1 only.

**Added rows to the outcome→claim table.**
- **All encoder F1 verdicts inconclusive.** Claims are restricted to the LM, the census, the theory and the evidence-use baselines; the target is Neurocomputing.
- **H1 signs disagree across regimes.** A claim is made only where the observed sign matches the B2 registered direction. Otherwise the result is descriptive.
- **H3 Worse.** Reported as a cost of the method.

**Venue.**
- Neural Networks only if E4 or E5 reproduces the H1-E verdict (same direction, or equivalence).
- Otherwise Neurocomputing.
- Machine Learning is SJR Q1 but probably JCR Q2. It is a fallback only if the authors accept an SJR definition of Q1.

## B6. Selection honesty and replication (R7)

- Level selection is described as "selected among 3 calibrated levels by a fixed low-information rule (2 folds × 50 dev)".
- All levels are reported on the tuning folds.
- **C\*_E** is selected on the **tuning** folds 0–1, using each family's selected level. The out-of-fold (OOF) replication then uses folds 2–9. This removes the circularity R7 identified.
- **C\*_L** is selected on seed 1 validation.

## B7. Compression moves to an extension (R8)

**E-S3.** Part A §3.4 becomes extension E-S3, with E7 folded in (S-noKD × 10 folds).

**Gate G4.**
- Pooled S-N OOF accuracy on folds 0–4 must be ≥ 57%.
- Otherwise E-S3 is descriptive only, and F7 is titled "retention at one size".

**Arms.** S-AD is dropped (cut C1 applied up front). S-HF@0.10 is added so that the student also has the H1 contrast.

## B8. Extension order (R9; budget-driven, declared now)

**Order:** P1 → E2 (collapse ladder) → E-S3 → E5 (BioLinkBERT-base) → E4 (SciFact) → E1 (R-Drop) → E3 (dropkey_budget only) → E6.

**Rules.**
- Each extension starts only if its full cost fits under the 90-h cap.
- All extensions finish before the single unseal.

## B9. Claims removed or restricted (R10, R11)

**Removed or restricted.**
- "Tail capping is the active ingredient" requires H1 (HB − HF).
- "The cap makes aggressive rates safe" is dropped. The same-p contrast is not run.
- "Mean displacement is enough" is restricted to the tested families at ℓ = 0.10, k = 4.
- "Exact laws for five families" becomes: exact laws for the mean for all families, and exact second moments except for the DropKey mean shift, which is estimated by Monte Carlo.
- The Track-1 null is labelled a retrodiction.
- The "first fine-tuned question-only / mismatched-evidence PQA-L baselines" claim keeps the hedge "in the literature we searched", including in the abstract.

**Required related work.**
- **Clipping analogues:**
  - DP-SGD per-example clipping (Abadi et al., 2016);
  - trust-region and PPO step clipping;
  - conservative policy iteration.
- **Attention sinks:** Xiao et al. (ICLR 2024), Darcet et al. (ICLR 2024), Sun et al. (2024), Clark et al. (2019), Gu et al. (ICLR 2025). The "sinks and shortcuts" mechanism is presented as motivation only, and is tested through the special-token share in B3.
- **DropHead** is excluded from the arms, with the reason given: it perturbs head outputs, not attention rows.

## B10. Statistics details (R13, R15, R16)

- **Run-level CI.** Obtained by inverting the sign-flip test (shift grid in steps of 1e-4).
- **Multiplicity.** Equivalence and co-criterion CIs are reported per comparison and labelled as such. Holm is applied to the F1 p-values.
- **DAM.** Fitted on per-run realized (D̄, B, S2, layer), with no family term. It is descriptive.
- **Gradient path.** The gradient flows through α in HB but not in HF. This is declared as part of the difference between HB and HF. An α-detached HB variant in the LM (10 seeds) is extension E8, after E6.
- **Per-layer reporting.** D̄, S2 and P(m>τ) are reported per layer.

## B11. Manipulation check and recalibration (R12; outcome-blind)

**The check.**
- The HB/HF and HB/HU ratios of time-averaged realized D̄ are evaluated on the tuning runs (folds 0–1; LM seed 1) before any finals start.
- These are manipulation measures, recorded on device, not outcomes.

**If a ratio falls outside [0.8, 1.25]:**
- Calibration is recomputed **once**, on the average of two censuses:
  - the pretrained census;
  - the census of the fold-0 N tuning model at its final step (checkpoint kept for that purpose).
- The finals then use the new parameters.
- The switch and both parameter sets are disclosed.
- No online recalibration is implemented.

## B12. Integrity self-audit (replaces the failed integrity red-team)

1. **Incident 1 (Part A §3.0).** A shell command printed `data/lm/test_docs.json`: PMIDs and offsets only, no text, no model output, no metric.
   - Disclosed.
   - `data/raw/pqa_labeled.parquet` is added to `env.SEALED`.
   - No sealed path is ever opened from a shell.
2. **Sealed access in code.**
   - Test *labels* live only in:
     - `domain_distill/data/test.json`
     - `test_ground_truth.json`
     - `ori_pqal.json`
     - `unified/data/raw/pqa_labeled.parquet`
   - The only readers are:
     - `prepare_eval_inputs.py`, which writes label-free test inputs and derangement donor lists. It asserts internally that each donor's label differs from the record's label and writes no label.
     - `unseal.py`.
   - `unified/data/lm/test.bin` is read only by `sealed_eval.py` inside the runner process, which writes per-document NLL write-once into `results/<study>/sealed/` and returns only a hash.
   - Encoder runners write test logits write-once into the same sealed folder.
   - No runner prints, logs or returns a test quantity.
   - The author never opens `sealed/` before `unseal.py`.
3. **Unselected tuning levels.** Their test logits are deleted unopened after selection, and only their hashes are kept in the registry.
4. **Outcome-blind console.** Runners print only loss health. `select.py` prints only its decisions. Dev metrics are read only by `select.py` and by the G_E and G1 gates.
5. **Pre-freeze runs are outcome-blind.** The seed-999 smoke runs and benchmarks are excluded from every analysis and are not reported as results.
6. **Deletion of unopened files.** This affects only files this study creates. No user file is deleted.

## B13. Compute re-projection (replaces the failed compute red-team)

**Per-run minutes.** These are from the measured real-length rates (20 epochs + evaluation):
- Encoder:
  - N: 9.4 on the eager path, 6.0 on SDPA.
  - AD 10.8, DK 13.4, RA 12.6, RB 13.8, HU/HF/HB 17.2, MX 13.0, LS 9.4 eager / 6.0 SDPA, Q0 1.5, L0 2.5.
- LM (8.0M token-steps + eval):
  - N 5.9, AD 6.4, DK 7.8, RA 6.2, RB 6.5, RD 5.9, HU/HF/HB 7.5.

**Speed-up.** Arms whose attention kind is `none` (N, LS, MX, Q0; LM N and RD) may use the SDPA path.
- This is mathematically identical: stock SDPA matches the eager adapter to 1e-5 in eval.
- Wall-clock is reported from the matched eager benchmark, not from the runs.

| Item | GPU-h |
|---|---|
| S0 (G0, benchmark gaps, census, value census) | 1.5 |
| S1: N ×10 | 1.0 |
| S1: tuning (AD, DK, RA, HU, MX, LS × 3 levels × 2 folds) plus HB@{0.05, 0.20} × 2 folds | 7.5 |
| S1: finals of 6 families × 8 folds | 9.7 |
| S1: HB/HF/RB@0.10 × 10 each, plus RA@0.10 and HU@0.10 completion (expected) | 11.7 |
| S1: Q0, L0 | 0.7 |
| S2: base LR (3) + tuning (AD, DK, RA, HU, RD × 3) + HB@{0.05, 0.20} | 2.1 |
| S2: finals (N, AD, DK, RA, HU, RD × 9 seeds) | 5.7 |
| S2: HB/HF/RB@0.10 × 10, plus RA@0.10 and HU@0.10 completion | 5.3 |
| Audits | 1.2 |
| **Core** | **46.4** |
| **Core with 10% contingency** | **51.0** |

**Extensions** (expected GPU-h): P1 5.7, E2 2.9, E-S3 7.8, E5 9.1, E4 9.6, E1 4.8, E3 3.8, E6 5.6, E8 1.25. The hard cap stays at 90 h.

**Cut order,** applied before a study starts, equally to all arms, and decided by budget only:
1. Q0/L0 → descriptive, folds 0–4.
2. DK drops to 2 levels (0.05, 0.10) in both regimes.
3. S2 reduced to 6.0M token-steps.
4. LS drops to 2 levels.

**Never cut:**
- the 2×2 core cells;
- RA\*, C\* pools, N;
- 10 units.

## B14. Minor items

- **R14.** The two pre-chosen audit runs per study are trained in deterministic mode, both the original and the replay (MATH SDPA, deterministic algorithms), and must replay within 1e-6. Non-deterministic retrain checks use the N arm's between-fold SD as tolerance.
- **R17.** The full protocol text (Parts A and B) is to be registered on OSF under embargo by the authors. OSF registration requires the authors' account, so the local `FREEZE.json` hash and its timestamp serve as the interim record. It is disclosed as such.
- **Encoder data order, clarified.** 20 seeded permutations are concatenated into a stream of 9,000 examples, cut into 563 batches of 16 (the last holds 8). The LR schedule has 563 steps with 56 warmup steps.
- **LM data order, clarified.**
  - For each epoch e, take an offset o_e ∈ [0, 256) and non-overlapping 257-token windows from o_e, shuffled with the CPU generator (seed+10000+e).
  - Concatenate epochs 1–8 and truncate to 977 × 32 windows.
  - Validation runs every 98 steps and at the final step. The best-validation weights are kept in CPU RAM for the co-primary early-stopped test NLL.

## B15. Pre-freeze engineering dispositions

**Gate G0** (`results/g0.json`; confirmatory rerun `results/g0_rerun_threads2.json`).

| Check | Result |
|---|---|
| One-step training equivalence against the archived oracle | Passed. LM: loss difference 0, max gradient difference 7.5e-8. BiomedBERT: loss difference 0, max gradient difference 1.3e-8. |
| Extreme rows (m >= 0.996, alpha = 1) | Passed in fp32 and bf16. |
| Track-1 reproduction | All 180 layer-level values across the 20 archived checkpoints match within 6.9e-9 absolute. |

The Track-1 reproduction reads as failed in `g0.json`, but only against an internal 1e-9 threshold that I had set more strictly than the protocol requires.
- The protocol criterion is to reproduce the published `budget_table.tex` values, which are printed to 4 decimals and so need an error below 5e-5. That criterion is met with a margin of about 7,000.
- The residual is unchanged at 2 and 4 CPU threads. It is a float32-level difference in the forward pass, about 3e-7 relative.
- Disposition: G0 is passed. The 1e-9 threshold is reported as stricter than the requirement.

**Identical arms.**
- If calibration makes two arms identical at some level, for example because an infeasible level forces HB and HF both to p = 1 with no effective cap, the run is executed once and shared.
- The contrast is then reported as identical by construction, with verdict "equivalent", and marked with that reason.

**RB calibration.** RB's τ is calibrated so that E[min(TV(a, U), τ)] = ℓ, like every other family. The "τ = 0.10" in B1 is the label for level ℓ = 0.10.

**Feasibility flags.**
- Near-uniform attention can make a level unreachable. The census on the untrained LM showed mean TV(a, U) = 0.023.
- In that case the arm runs at its maximum strength (γ = 1 or p = 1) and is flagged in `predictions.json`. It is never retuned.

**Analysis scripts written after the freeze** (`select.py`, `stats_unified.py`, `unseal.py`, `audit.py`, `tables.py`).
- They implement only the rules stated in Parts A and B.
- Each file's SHA-256 is appended to `FREEZE_ADDENDUM.json` before its first use, and before any unseal.

**Environment.** Every model process blocks pyarrow and pandas (`env.block_arrow`). Parquet is read only in torch-free processes, because pyarrow access-violates when it is loaded alongside torch. See the G0 run notes.
