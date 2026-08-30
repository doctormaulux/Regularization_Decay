"""Merge the robust-decay head-to-head CSV from a partial (--robust-new-only) run.

The head-to-head of Section 5.10 needs seven methods at the same scale:

    Baseline, WD-tuned, Tau(alpha=0), tau(w)   <- already produced by the main
                                                  same-scale run (Section 5.6)
    Huber-decay, PseudoHuber-decay, LogCosh-decay  <- the three new competitors

Re-running the first four costs roughly half the GPU time of the whole comparison and
reproduces numbers we already have: the script, the scale config, the epoch budget, the
seeds and the PSO budget are identical between the two invocations, and the pipeline is
deterministic (verified when the 18M run was repeated and returned identical numbers).
So `--robust-new-only` trains only the three new methods, and this script splices the
four shared rows in from the existing CSV.

    python analysis/merge_robust_results.py --scale large

reads   new results/gpt2_large_wikitext_standardized_results.csv        (shared 4)
        new results/gpt2_large_wikitext_standardized_results_robust_new.csv  (new 3)
writes  new results/gpt2_large_wikitext_standardized_results_robust.csv

Provenance is recorded in a `source` column and in a sidecar .provenance.json, so the
merge can never be mistaken for a single self-contained run.
"""
import argparse
import json
import os
import sys

import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'new results')

SHARED_METHODS = ['Baseline', 'WD-tuned', 'Tau(alpha=0)', 'τ(w)']
NEW_METHODS = ['Huber-decay', 'PseudoHuber-decay', 'LogCosh-decay']
ROW_ORDER = ['Baseline', 'WD-tuned', 'Tau(alpha=0)', 'Huber-decay',
             'PseudoHuber-decay', 'LogCosh-decay', 'τ(w)']


def _path(scale, tag=''):
    return os.path.join(RESULTS_DIR,
                        f'gpt2_{scale}_wikitext_standardized_results{tag}.csv')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scale', default='large',
                    help="Scale tag of the run to merge (default: large).")
    ap.add_argument('--force', action='store_true',
                    help='Overwrite an existing merged CSV.')
    args = ap.parse_args()

    shared_path = _path(args.scale)
    new_path = _path(args.scale, '_robust_new')
    out_path = _path(args.scale, '_robust')

    for label, path in (('shared-methods', shared_path), ('new-methods', new_path)):
        if not os.path.exists(path):
            sys.exit(f"[ERROR] missing {label} CSV:\n        {path}")

    if os.path.exists(out_path) and not args.force:
        sys.exit(f"[ERROR] {out_path} already exists; pass --force to overwrite.")

    shared = pd.read_csv(shared_path)
    new = pd.read_csv(new_path)

    missing_shared = [m for m in SHARED_METHODS if m not in set(shared['method'])]
    missing_new = [m for m in NEW_METHODS if m not in set(new['method'])]
    if missing_shared:
        sys.exit(f"[ERROR] {shared_path}\n        lacks required methods: {missing_shared}")
    if missing_new:
        sys.exit(f"[ERROR] {new_path}\n        lacks required methods: {missing_new}")

    shared_rows = shared[shared['method'].isin(SHARED_METHODS)].copy()
    new_rows = new[new['method'].isin(NEW_METHODS)].copy()
    shared_rows['source'] = os.path.basename(shared_path)
    new_rows['source'] = os.path.basename(new_path)

    merged = pd.concat([shared_rows, new_rows], ignore_index=True)
    merged['_order'] = merged['method'].map({m: i for i, m in enumerate(ROW_ORDER)})
    merged = merged.sort_values('_order').drop(columns='_order').reset_index(drop=True)

    # A column present in one CSV but not the other would silently become NaN.
    only_shared = set(shared_rows.columns) - set(new_rows.columns)
    only_new = set(new_rows.columns) - set(shared_rows.columns)
    if only_shared or only_new:
        print(f"[WARN] column mismatch between the two CSVs — "
              f"only in shared: {sorted(only_shared)}; only in new: {sorted(only_new)}",
              file=sys.stderr)

    merged.to_csv(out_path, index=False)

    prov = {
        'merged_csv': os.path.basename(out_path),
        'scale': args.scale,
        'shared_methods': {'methods': SHARED_METHODS,
                           'from': os.path.basename(shared_path)},
        'new_methods': {'methods': NEW_METHODS, 'from': os.path.basename(new_path)},
        'note': ('Section 5.10 head-to-head assembled from two invocations of the same '
                 'script at the same scale, same seeds, same epoch budget and same PSO '
                 'budget. The four shared methods were not re-trained.'),
    }
    prov_path = out_path.replace('.csv', '.provenance.json')
    with open(prov_path, 'w', encoding='utf-8') as fh:
        json.dump(prov, fh, indent=2, ensure_ascii=False)

    print(f"[OK] wrote {out_path}")
    print(f"[OK] wrote {prov_path}")
    print(merged[['method', 'test_ppl_mean', 'test_ppl_std', 'source']].to_string(index=False))


if __name__ == '__main__':
    main()
