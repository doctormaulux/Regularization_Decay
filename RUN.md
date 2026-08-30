# How to run

## Setup

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pytest -q          # correctness tests (CPU, ~1 min; Python >= 3.10)
```

Every benchmark writes `results/<name>_standardized_results.{csv,json}` (aggregates and
per-seed metrics) and journals every PSO evaluation and final run under
`results/journal/<name>_standardized_results.jsonl`, so an interrupted run resumes without
repeating finished work. The manuscript generator reads the copies in `new results/`.

## Benchmarks

```bash
# From-scratch GPT-2 scale sweep on WikiText-2 (word-level, custom model)
python gpt2_wikitext_standardized.py --scale tiny                 # 2.1M, core-6, n=3
python gpt2_wikitext_standardized.py --scale small                # 7.4M, 10 methods, n=5
python gpt2_wikitext_standardized.py --scale medium --instrument  # 18M, core-6, n=3
python gpt2_wikitext_standardized.py --scale large --seeds 10 --instrument \
    --methods "Baseline,L2,ElasticNet,WD-tuned,Tau(AdamW-scope),Tau(alpha=0),Tau(w)"   # 66M, n=10
python gpt2_wikitext_standardized.py --scale large --robust-new-only --seeds 5   # robust-decay family
python gpt2_wikitext_standardized.py --scale large --data-fraction 0.25 --seeds 5 \
    --hp-from results/gpt2_large_wikitext_standardized_results.json   # data-quantity arm (fixed hp)

# 124M GPT-2 on WikiText-103 (GPT-2 BPE): transfer + confirmation sweep, then finals
python gpt2_wt103_standardized.py --stage sweep
python gpt2_wt103_standardized.py --stage final --instrument

# Single-scale benchmarks
python regression_benchmarks.py --model all        # sin(x), Friedman #1
python classification_benchmarks.py --model all    # MNIST FC, CIFAR-10 CNN
python vit_cifar_standardized.py                   # ViT-mini on CIFAR-10
python bert_sst2_standardized.py                   # BERT-tiny on SST-2
python wikitext_benchmarks.py --model smollm2      # SmolLM2-135M on WikiText-2
```

Useful flags of `gpt2_wikitext_standardized.py`: `--methods A,B` (subset of the roster;
output files and journal unchanged, so a roster can be split across machines and
reassembled with `analysis/assemble_from_journals.py <stem> <journals...>`), `--seeds N`,
`--instrument` (per-epoch trajectories under `results/instrumentation/`), `--pso-budget`
(default `auto`: 12 evaluations per 1-D search, 40 per >= 2-D search), `--schedule-epochs`
(horizon of the linear LR decay, independent of the early-stopping ceiling `--epochs`).
`vit_cifar_standardized.py` and `wikitext_benchmarks.py` accept `--methods` too.

## Early stopping

All runs use patience-based early stopping and report the best-validation-epoch model.
`EarlyStopping.restore_best_model()` verifies a fingerprint of the restored parameters and
`check_restored(val)` re-evaluates the restored model against the recorded best epoch; a
mismatch raises instead of reporting a number. `analysis/audit_early_stopping.py` checks
the instrumentation trajectories of a run (`--min-ok 1`) or classifies result files by
whether training continued past the best epoch (`--legacy`).

## Analysis and manuscript

```bash
python analysis/mechanism_analysis.py --scale large      # tables + figures under figures/
python analysis/merge_robust_results.py --scale large    # robust-family head-to-head table
python paper/create_paper_figures.py
python paper/build_paper.py                              # -> paper/articolo.docx
```

## GPU pods (RunPod)

`pod_create.py` (REST API; injects the SSH key), `pod_bootstrap.sh` (upload + launch in
tmux), `pod_launch_*.sh` (each runs `pod_preflight.sh` first: tests + a short real run with
the restore checks live), `pod_status.sh [--pull [tag]]`, `pod_finish.sh`, `pod_collect.sh`
(pull, assemble split rosters from the journals, audit, copy to `new results/`).
`pods.conf` (git-ignored, see `pods.conf.example`) lists the pods.
