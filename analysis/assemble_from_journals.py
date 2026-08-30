#!/usr/bin/env python3
"""Assemble a benchmark's results CSV/JSON directly from RunJournal files.

    python analysis/assemble_from_journals.py <stem> <journal.jsonl> [...] [--out results]
    e.g. python analysis/assemble_from_journals.py gpt2_large_wikitext \\
             results/journal/gpt2_large_wikitext_standardized_results.A1_*.jsonl

Uses the same ExperimentTracker the benchmarks use, so the output is byte-for-byte the
format run_benchmark writes (mean/std/CI per metric, best_hyperparams, per-run metrics),
but nothing is trained: only `eval` and `best_hp` records are read. Rosters split across
pods are therefore merged by passing all their journals. Methods are ordered as in the
10-method roster, with any extra method (e.g. Tau(AdamW-scope)) appended.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment_utils import ExperimentTracker, DEFAULT_METHODS  # noqa: E402

ORDER = list(DEFAULT_METHODS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stem', help="e.g. gpt2_large_wikitext (-> <stem>_standardized_results.{csv,json})")
    ap.add_argument('journals', nargs='+')
    ap.add_argument('--out', default='results')
    ap.add_argument('--name', default=None, help='experiment_name stored in the JSON')
    args = ap.parse_args()

    files = []
    for j in args.journals:
        files += sorted(glob.glob(j)) if any(c in j for c in '*?[') else [j]
    evals, best_hp = {}, {}
    for f in files:
        for line in open(f, encoding='utf-8'):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get('kind') == 'eval':
                evals.setdefault(r['method'], {})[r['seed']] = r['metrics']   # last record wins
            elif r.get('kind') == 'best_hp':
                best_hp[r['method']] = r['hyperparams']
    if not evals:
        sys.exit(f'no eval records in {files}')

    tracker = ExperimentTracker(args.name or f'{args.stem}_assembled_from_journals')
    methods = [m for m in ORDER if m in evals] + [m for m in evals if m not in ORDER]
    for m in methods:
        tracker.set_hyperparams(m, best_hp.get(m, {}))
        for i, seed in enumerate(sorted(evals[m])):
            tracker.add_result(m, i, evals[m][seed])
    os.makedirs(args.out, exist_ok=True)
    tracker.save_to_csv(os.path.join(args.out, f'{args.stem}_standardized_results.csv'))
    tracker.save_to_json(os.path.join(args.out, f'{args.stem}_standardized_results.json'))
    print('assembled', args.stem, {m: len(evals[m]) for m in methods},
          'from', len(files), 'journal(s)')


if __name__ == '__main__':
    main()
