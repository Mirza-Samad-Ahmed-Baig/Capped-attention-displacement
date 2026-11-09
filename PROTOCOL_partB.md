
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
