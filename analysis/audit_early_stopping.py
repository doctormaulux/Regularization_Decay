#!/usr/bin/env python3
"""Verify which result files report the best-epoch model.

Every results JSON stores `convergence_epoch = best_epoch + 1` per run, so a run that
trained on past its best epoch is one with `convergence_epoch < NUM_EPOCHS` (the last
epoch is min(NUM_EPOCHS, best_epoch + patience + 1)). Two modes:

  --legacy   classify results JSONs produced by a pipeline WITHOUT best-epoch restore by
             that rule (default directory: "new results"); for files produced by the
             current pipeline the rule is uninformative - use the instrumentation mode.

  (default)  verify instrumentation trajectories (results/instrumentation/*.json): the
             reported final.val_ppl must equal the trajectory value at the recorded best
             epoch. Files carrying phase='pso' are skipped. Exit code 1 if any final run
             reports a different epoch.

Usage:
    python analysis/audit_early_stopping.py --legacy
    python analysis/audit_early_stopping.py --instrumentation results/instrumentation --min-ok 1
"""
import argparse
import glob
import json
import math
import os
import re
import sys

# Epoch ceilings per benchmark stem (as configured in the scripts at the time the
# pre-fix results were produced). Keys are matched as prefixes of the file stem.
EPOCH_CEILINGS = [
    (r'^(sin|complex)_regression', 200),
    (r'^mnist', 50),
    (r'^cifar', 100),
    (r'^vit_cifar', 150),
    (r'^bert_sst2', 50),
    (r'^(roberta|albert|distilbert|modernbert|deberta|electra)_sst2', 10),
    (r'^t5_sst2', 10),
    (r'^(swin|deit)_cifar', 20),
    (r'^smollm2_wikitext', 5),
    (r'^(qwen25|llama32|phi2|gemma2)_wikitext', 3),
    (r'^(mamba|rwkv)_wikitext', 5),
    (r'^gpt2_wt103', 8),
    (r'^gpt2_(tiny|medium|large)_wikitext_standardized_results_data25c', 40),
    (r'^gpt2_(tiny|medium|large)_wikitext_standardized_results_data50c', 40),
    (r'^gpt2_(tiny|medium|large)_wikitext_standardized_results_data100c', 40),
    (r'^gpt2_(tiny|medium|large)_wikitext_standardized_results_data25', 48),
    (r'^gpt2_(tiny|medium|large)_wikitext_standardized_results_data50', 24),
    (r'^gpt2_(tiny|medium|large)_wikitext_standardized_results_data100', 12),
    (r'^gpt2_(tiny|medium|large)_wikitext', 12),
    (r'^gpt2_wikitext', 30),          # the 'small' reference (30 epochs, patience 8)
]


def ceiling_for(stem):
    for pat, n in EPOCH_CEILINGS:
        if re.match(pat, stem):
            return n
    return None


def audit_legacy(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, '*_results*.json')))
    if not files:
        print(f"no results JSON under {results_dir}")
        return 0
    print(f"{'file':<62} {'E':>4} {'runs':>5} {'affected':>9}  verdict / per-method mean conv_epoch")
    print('-' * 130)
    tot_runs = tot_aff = 0
    for f in files:
        stem = os.path.basename(f).replace('.json', '')
        ceiling = ceiling_for(stem)
        try:
            d = json.load(open(f, encoding='utf-8'))
        except Exception as exc:
            print(f"{stem:<62} {'?':>4} {'?':>5} {'?':>9}  unreadable ({exc})")
            continue
        res = d.get('results') or {}
        if ceiling is None:
            print(f"{stem:<62} {'?':>4} {sum(len(v) for v in res.values()):>5} {'?':>9}  "
                  f"no ceiling known for this stem - add it to EPOCH_CEILINGS")
            continue
        runs = aff = 0
        per_method = []
        for m, rr in res.items():
            ce = [r['metrics'].get('convergence_epoch') for r in rr
                  if isinstance(r, dict) and 'metrics' in r]
            ce = [c for c in ce if c is not None]
            a = sum(1 for c in ce if c < ceiling)
            runs += len(ce)
            aff += a
            if ce:
                per_method.append(f"{m}:{sum(ce)/len(ce):.1f}{'*' if a else ''}")
        tot_runs += runs
        tot_aff += aff
        verdict = ('CLEAN' if aff == 0 else
                   'AFFECTED' if aff == runs else f'PARTIAL ({aff}/{runs})')
        print(f"{stem:<62} {ceiling:>4} {runs:>5} {aff:>9}  {verdict}  "
              + ', '.join(per_method))
    print('-' * 130)
    print(f"total runs {tot_runs}, trained past the best epoch (last-epoch report): {tot_aff} "
          f"({100 * tot_aff / max(tot_runs, 1):.1f}%)")
    return 0


def audit_instrumentation(inst_dir, rel_tol=1e-6, min_ok=0):
    """Verify that every final run's reported val_ppl is the value at its recorded best
    epoch (convergence_epoch - 1), not the last epoch's.

    Note: EarlyStopping uses min_delta (0.01 PPL in the GPT-2 scripts), so the LAST epoch
    may be marginally better than the recorded best; the check
    therefore indexes the trajectory by the recorded best epoch instead of taking its
    minimum. A run is *informative* only if training continued past the best epoch.
    """
    files = sorted(glob.glob(os.path.join(inst_dir, '*.json')))
    if not files:
        print(f"no instrumentation JSON under {inst_dir}")
        return 0
    n_ok = n_bad = n_skip = n_noes = 0
    bad = []
    for f in files:
        try:
            d = json.load(open(f, encoding='utf-8'))
        except Exception:
            n_skip += 1
            continue
        if d.get('phase', 'eval') != 'eval':
            n_skip += 1
            continue
        traj = d.get('trajectory') or []
        final = d.get('final') or {}
        fval = final.get('val_ppl')
        vals = [t.get('val_ppl') for t in traj if t.get('val_ppl') is not None]
        conv = d.get('best_epoch', final.get('convergence_epoch'))
        if fval is None or not vals or conv is None:
            n_skip += 1
            continue
        k = int(conv) - 1
        if not 0 <= k < len(vals):
            n_bad += 1
            bad.append((os.path.basename(f), fval, float('nan'), vals[-1], conv))
            continue
        reports_best = math.isclose(fval, vals[k], rel_tol=rel_tol)
        informative = k != len(vals) - 1
        if not informative:
            n_noes += 1 if reports_best else 0
            if not reports_best:
                n_bad += 1
                bad.append((os.path.basename(f), fval, vals[k], vals[-1], conv))
            continue
        if reports_best:
            n_ok += 1
        else:
            n_bad += 1
            bad.append((os.path.basename(f), fval, vals[k], vals[-1], conv))
    print(f"instrumentation files: {len(files)}  |  report BEST epoch: {n_ok}  |  "
          f"report another epoch: {n_bad}  |  best==last (uninformative): {n_noes}  |  "
          f"skipped (pso/unreadable): {n_skip}")
    for name, fval, bval, lval, conv in bad:
        print(f"  NOT-BEST  {name}: final={fval:.3f} val@best_epoch({conv})={bval:.3f} "
              f"last={lval:.3f}")
    if n_bad:
        return 1
    if n_ok < min_ok:
        print(f"only {n_ok} informative final run(s) reported the best epoch; "
              f"--min-ok {min_ok} required (the restore guard was never exercised)")
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--legacy', action='store_true',
                    help='classify pre-fix results JSONs by the convergence_epoch rule')
    ap.add_argument('--dir', default='new results', help='results directory (legacy mode)')
    ap.add_argument('--instrumentation', default='results/instrumentation',
                    help='instrumentation directory (default mode)')
    ap.add_argument('--min-ok', type=int, default=0,
                    help='fail unless at least this many final runs that stopped past '
                         'their best epoch report the best-epoch metric (preflight use)')
    args = ap.parse_args()
    if args.legacy:
        return audit_legacy(args.dir)
    return audit_instrumentation(args.instrumentation, min_ok=args.min_ok)


if __name__ == '__main__':
    sys.exit(main())
