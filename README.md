# Decoupled Fair Weight Decay for Over-Capacity Autoregressive Language Models

Code, results and manuscript generator for the paper on **τ-decay**: a decoupled,
weight-only decay whose rate saturates with the weight magnitude,

    w ← w − ρ · w / (1 + |w| / δ)

(the proximal step of the Fair penalty; ρ = maximal relative decay rate, δ = knee).
The ablation with δ → ∞ ("Tau(alpha=0)") is a constant decoupled decay at the same
scope and schedule, and the factorial with an AdamW-scope variant separates the
contribution of the decay's *scope and schedule* from that of its *magnitude adaptivity*.

Authors: Giuseppe Maulucci (corresponding, ORCID 0000-0002-2154-319X), Tommaso Marchetti,
Marco De Spirito — Università Cattolica del Sacro Cuore, Rome.

## What is here

| Path | Content |
|---|---|
| `experiment_utils.py` | Shared training loops, verified early stopping (`EarlyStopping`), PSO tuning (`find_best_hyperparams`, dimension-aware budgets: 12 evaluations for 1-D searches, 40 for ≥2-D), decay operators, `run_benchmark` with a resumable journal |
| `gpt2_wikitext_standardized.py` | From-scratch GPT-2 scale sweep on WikiText-2 (`--scale tiny|small|medium|large` = 2.1M / 7.4M / 18M / 66M parameters), 2×2 scope × adaptivity roster, robust-decay roster, data-quantity arm (`--data-fraction`), per-epoch instrumentation |
| `gpt2_wt103_standardized.py` | 124M GPT-2 on WikiText-103 (GPT-2 BPE, token-level perplexity) |
| `regression_benchmarks.py`, `classification_benchmarks.py`, `vit_cifar_standardized.py`, `bert_sst2_standardized.py`, `wikitext_benchmarks.py` | Single-scale benchmarks (sin / Friedman regression, MNIST, CIFAR-10 CNN, ViT-mini, BERT-tiny SST-2, SmolLM2-135M) |
| `new results/*_standardized_results.{csv,json}` | Aggregated statistics per method and per-seed metrics, one pair per benchmark (what the manuscript generator reads) |
| `results/journal/` | Append-only journals of every PSO evaluation, PSO winner and final run (full provenance; a benchmark resumes from them) |
| `results/instrumentation/` | Per-epoch train/validation perplexity and weight-magnitude statistics of every final run of the scale sweep (`pso/`: the tuning probes) |
| `analysis/` | Statistics (`stats_utils.py`), mechanism figures (`mechanism_analysis.py`), journal assembly (`assemble_from_journals.py`), best-epoch audit (`audit_early_stopping.py`), robust-family merge |
| `paper/` | `build_paper.py` generates the manuscript (`.docx`) from the CSVs; `create_paper_figures.py` the figures; `build_submission.py` assembles the journal submission folder (title page, highlights, declarations) |
| `tests/` | `pytest -q`: unit and end-to-end tests of the early-stopping restore and of the audit tool |
| `pod_*.sh`, `pod_create.py` | Launch, monitor, collect and assemble runs on RunPod GPU pods |

## Reproduce

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pytest -q                                              # correctness tests (CPU, ~1 min)

python gpt2_wikitext_standardized.py --scale large --seeds 10 --instrument   # 66M sweep point
python gpt2_wikitext_standardized.py --scale small                            # 7.4M, 10 methods
python regression_benchmarks.py --model all
python classification_benchmarks.py --model all
python vit_cifar_standardized.py
python bert_sst2_standardized.py
python wikitext_benchmarks.py --model smollm2
python gpt2_wt103_standardized.py --stage sweep && python gpt2_wt103_standardized.py --stage final

python analysis/mechanism_analysis.py --scale large   # mechanism tables and figures
python paper/build_paper.py                           # manuscript from the CSVs
```

Every benchmark tunes each method's hyperparameters by particle-swarm optimisation on the
validation set, then evaluates the winner over independent seeds; early stopping restores
the best-validation checkpoint and verifies the restore at run time (parameter fingerprint
and re-evaluated validation metric). Runs are journaled and resumable; a roster can be
split across machines with `--methods` and reassembled with
`analysis/assemble_from_journals.py`. See `RUN.md` for details.

## Protocol in one paragraph

Baseline is truly unregularised (no weight decay). Competitors: L1, L2, ElasticNet, SCAD,
MCP, LSP (loss penalties), WD-tuned (PSO-tuned AdamW decoupled weight decay), the
Tau(alpha=0) ablation, the Tau(AdamW-scope) 2×2 cell, and the robust-decay family
(Huber, pseudo-Huber, log-cosh) at 66M. Test perplexity / accuracy / MSE is reported as
mean ± standard deviation with 95% confidence intervals; n = 10 seeds at the 66M scale
point, n = 5 for the single-scale benchmarks, n = 3 for tiny, medium, SmolLM2 and
WikiText-103.
