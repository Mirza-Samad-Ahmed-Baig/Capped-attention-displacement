# Capped attention displacement (CapDrop)

Code, frozen protocol, run records and results for the paper

> **CapDrop: capping realized attention displacement in data-limited biomedical transformers**
> Syeda Anshrah Gillani\*, Mirza Samad Ahmed Baig\*, Shahid Munir Shah, Abdul Akbar Khan, Muhammad Omer Khan,
> Muhammad Hamzah Siddiqui (\* equal contribution). Submitted to *Neurocomputing*.

## The idea in one paragraph

Attention regularizers (attention dropout, DropKey, Relaxed Attention, key deletion) are set by a nominal rate. What
actually changes a head's output is the **realized displacement** of each attention row, `d = ½‖ã − a‖₁`. For targeted
perturbations it depends on how concentrated attention is, so methods compared at equal rates receive unequal doses.
**CapDrop** is top-k key deletion with a closed-form mixing coefficient `α = min(1, τ/m)` (`m` = deleted attention
mass). It mixes the original and the perturbed row as `ã = (1 − α)a + αb`, so every row moves by exactly `min(m, τ)`.
We calibrate every method to the same realized mean displacement and test CapDrop in a preregistered study.

## Main results (after the single unseal)

| Regime | CapDrop vs. | Difference | Verdict |
|---|---|---|---|
| Language model (test NLL, nats/token) | fixed-coefficient deletion | −0.045 | Better |
| | tuned Relaxed Attention | −0.032 | Better |
| | tuned DropKey (ℓ = 0.20) | +0.058 | Worse |
| PubMedQA, BiomedBERT (test Brier) | fixed-coefficient deletion | −0.005 | Inconclusive |
| | tuned Relaxed Attention | −0.006 | Inconclusive |
| | label smoothing | +0.132 | Calibration only |

Exact two-sided sign-flip tests over 10 seeds/folds with Holm correction. Full results: `results/analysis.json`,
`paper/main.pdf`.

## Repository layout

| Path | Contents |
|---|---|
| `PROTOCOL.md`, `PROTOCOL_partA.md`, `PROTOCOL_partB.md` | The binding preregistered protocol |
| `FREEZE.json`, `FREEZE_ADDENDUM.jsonl` | Freeze record (protocol, code and data hashes) and every later addendum |
| `DEVIATIONS.jsonl` | Every deviation from the protocol, with timestamp and code hash |
| `attnreg.py` | All attention regularizers, including CapDrop (`hard_budget`) |
| `census.py`, `calibrate.py`, `s0_census.py` | Exact displacement laws and the displacement calibration |
| `encoder_runner.py`, `lm_runner.py`, `lm_model.py` | Training runners (BiomedBERT fine-tuning; the from-scratch LM) |
| `campaign.py`, `drive*.sh`, `select.py`, `budget.py` | Declares every study and runs it in the preregistered order |
| `sealed_eval.py`, `cv_report.py`, `unseal.py` | Sealed test evaluation and the single unseal |
| `analyze.py`, `analyze_ext.py`, `stats_unified.py` | Confirmatory, secondary and extension analysis |
| `figures_*.py`, `tables_results.py`, `graphical_abstract.py`, `supplement.py` | Figures, tables and supplement |
| `studies/` | Every study: registry, ledger, run records, sealed test files (model checkpoints omitted) |
| `results/` | Census, predictions, CV report, unsealed scores, analysis, audits, cost |
| `data/` | Derived data: CV folds, PQA-L pool, sealed evaluation inputs, SciFact files, LM tokenizer and splits |
| `excluded/` | The three DropKey runs excluded before the unseal (disclosed in the paper) |
| `paper/` | LaTeX sources, figures, tables and the compiled PDFs |
| `tests/` | Unit tests |

Omitted because of size or because they are public downloads: model checkpoints, census value tensors (`*.pt`), the
LM training stream `data/lm/train.bin` (145 MB; rebuilt by `prepare_lm_corpus.py` from the PubMedQA artificial subset),
the unlabelled and artificial PubMedQA subsets (fetched by `prepare_pubmedqa_extra.py`) and the pretrained models.

## Setup

Tested with Python 3.12, PyTorch 2.13 (CUDA 12.6), transformers 4.57 on one 4 GB laptop GPU (Windows 11).

```bash
pip install -r requirements.txt
```

The scripts expect this workspace layout (`env.WORKSPACE` is the parent folder of this repository):

```
<workspace>/
  unified/                      <- this repository
  domain_distill/models/biomedbert/   BiomedBERT-base (abstracts + full text), from Hugging Face
  domain_distill/data/          PubMedQA PQA-L train.json, dev.json, test.json (official split)
  strengthened/attentiondrop.py reference implementation, only for the g0_checks.py equivalence test
```

BioLinkBERT-base (pinned revision, hash-checked) is fetched by `prepare_biolinkbert.py`, SciFact by `prepare_scifact.py`,
and the PubMedQA unlabelled and artificial subsets by `prepare_pubmedqa_extra.py`. `prepare_lm_corpus.py` builds the
LM corpus from the artificial subset's PubMed abstracts.

## Reproducing

1. **Verify the records.** `python -m unittest discover tests` runs the 83 unit tests (all pass in the workspace
   layout above; `test_attnreg` and `test_pubmedqa_inputs` need BiomedBERT in `domain_distill/models/biomedbert`). The SHA-256 values in
   `FREEZE.json` and `FREEZE_ADDENDUM.jsonl` identify the exact code used for each stage.
2. **Re-run the analysis only (no GPU).** `results/unsealed.json` holds the scored test outputs, so
   `python analyze.py`, `python figures_results.py`, `python tables_results.py` and `python supplement.py` rebuild
   `results/analysis.json`, the figures, the tables and the supplement.
3. **Re-run the experiments (GPU, about 86 GPU-hours).** Build the data (`prepare_*.py`), run the census
   (`s0_census.py`), then the stages in protocol order through `drive.sh`, `drive_ext.sh`, `drive_ext2.sh` and
   `drive_e8.sh` (stages are declared by `campaign.py <stage>`). Finish with `cv_report.py`, record its hash in
   `FREEZE_ADDENDUM.jsonl`, run `unseal.py` once, then `analyze.py`. Kernels are non-deterministic in bf16; the
   deterministic replay audit (`audit.py`, `results/audit.json`) bounds the run-to-run variation.

## Integrity

The protocol, code and data manifests were frozen before any outcome run. Test labels and the LM test stream stayed
sealed until every declared run was complete and were scored once. Every run is reported, no seed was replaced, and
every deviation is in `DEVIATIONS.jsonl`.

## Data licences

PubMedQA (MIT; its abstracts are from PubMed), SciFact (claims CC BY 4.0, abstracts ODC-By 1.0), BiomedBERT (MIT),
BioLinkBERT (Apache-2.0).

## Citation

Citation details will be added on publication.

## Licence

Code: MIT (see `LICENSE`).
