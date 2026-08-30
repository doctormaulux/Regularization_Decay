"""
Mechanism analysis (Workstream 2) for τ-decay.

Consumes the per-epoch trajectories written by the `--instrument` runs
(results/instrumentation/gpt2_<scale>_<method>_seed<seed>.json) and produces the
evidence a top-venue reviewer asks for — *what does the magnitude-adaptive term α
actually do?* — that the audit / pivot flagged as missing:

  1. Overfitting-delay curves: train vs val perplexity over epochs, per method,
     averaged across seeds. τ(w) should keep val-PPL falling for longer than the
     Baseline / tuned weight decay (over-capacity → τ delays the overfitting upturn).
  2. Weight-distribution dynamics: mean |w|, fraction of near-zero weights, and L2
     norm over training — the signature of magnitude-adaptive shrinkage (small-weight
     suppression with large-weight preservation) vs L2 / constant decay.
  3. A summary table: best val-PPL, epoch reached, and generalization gap (val−train).

Runs without a GPU. Figures require matplotlib; if it is unavailable the script
still prints the numeric summary so the analysis is usable headless.

Usage:
    python analysis/mechanism_analysis.py --scale medium
    python analysis/mechanism_analysis.py --scale medium --dir results/instrumentation
"""
import argparse
import glob
import json
import os
from collections import defaultdict

# Methods to contrast, in a fixed order (subset present in the data is used).
METHOD_ORDER = ['Baseline', 'L2', 'ElasticNet', 'WD-tuned', 'Tau(AdamW-scope)',
                'Tau(alpha=0)', 'τ(w)']
# Per-epoch weight-magnitude series to plot. median/max carry the adaptivity
# signature (small-weight suppression vs large-weight preservation); mean and
# frac_below_thr summarise the bulk; l2_norm the overall shrinkage.
WEIGHT_SERIES = ['median_abs_w', 'max_abs_w', 'frac_below_thr', 'l2_norm']

# Same method->color mapping as paper/create_paper_figures.py (keep in sync) so
# mechanism figures read as one system with the rest of the paper.
METHOD_COLORS = {
    'Baseline': '#1f77b4',
    'L1': '#ff7f0e',
    'L2': '#2ca02c',
    'ElasticNet': '#d62728',
    'SCAD': '#9467bd',
    'MCP': '#8c564b',
    'LSP': '#e377c2',
    'WD-tuned': '#17becf',
    'Tau(AdamW-scope)': '#7f7f7f',
    'Tau(alpha=0)': '#bcbd22',
    'τ(w)': '#FFD700',
    'Huber-decay': '#8c6d31',
    'PseudoHuber-decay': '#a55194',
    'LogCosh-decay': '#393b79',
}


def canonical(hp):
    """(rho, delta) of a tau-family hyperparameter dict, accepting both the canonical
    {'rho', 'delta'} form and the legacy {'decay_strength', 'tau0', 'tau_alpha'} triple.
    Mirrors experiment_utils.canonical_decay_params without importing torch."""
    if 'rho' in hp:
        return float(hp['rho']), float(hp.get('delta', float('inf')))
    ds, t0 = float(hp['decay_strength']), float(hp['tau0'])
    al = float(hp.get('tau_alpha', 10.0))
    return ds / t0, (float('inf') if al == 0 else t0 / al)


def load_trajectories(inst_dir, scale):
    """Return ({method: [trajectory_per_seed, ...]}, {method: hyperparams})."""
    by_method = defaultdict(list)
    hyper = {}
    pattern = os.path.join(inst_dir, f'gpt2_{scale}_*_seed*.json')
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding='utf-8') as fh:
            d = json.load(fh)
        # The glob also matches sibling tags such as 'gpt2_large_data25_*': keep only the
        # requested benchmark, and never a PSO probe (final runs only).
        if d.get('benchmark') != f'gpt2_{scale}' or d.get('phase', 'eval') != 'eval':
            continue
        traj = d.get('trajectory') or []
        if traj:
            by_method[d.get('method', '?')].append(traj)
            hyper.setdefault(d.get('method', '?'), d.get('hyperparams', {}))
    return by_method, hyper


def _mean_over_seeds(trajs, key):
    """Average `key` at each epoch index across seeds (truncate to shortest run)."""
    if not trajs:
        return [], []
    n = min(len(t) for t in trajs)
    epochs = [trajs[0][i].get('epoch', i + 1) for i in range(n)]
    means = []
    for i in range(n):
        vals = [t[i].get(key) for t in trajs if t[i].get(key) is not None]
        means.append(sum(vals) / len(vals) if vals else float('nan'))
    return epochs, means


def summarize(by_method):
    """Print a per-method summary: best val-PPL, epoch, generalization gap."""
    print(f"\n{'method':<14} {'best_val_ppl':>12} {'@epoch':>7} "
          f"{'gen_gap(val-train)':>18} {'n_seeds':>8}")
    print('-' * 62)
    rows = []
    for m in METHOD_ORDER + [k for k in by_method if k not in METHOD_ORDER]:
        trajs = by_method.get(m)
        if not trajs:
            continue
        _, val = _mean_over_seeds(trajs, 'val_ppl')
        _, tr = _mean_over_seeds(trajs, 'train_ppl')
        if not val:
            continue
        best_i = min(range(len(val)), key=lambda i: val[i])
        best_val = val[best_i]
        gap = (val[best_i] - tr[best_i]) if best_i < len(tr) else float('nan')
        print(f"{m:<14} {best_val:>12.3f} {best_i + 1:>7} {gap:>18.3f} {len(trajs):>8}")
        rows.append((m, best_val, best_i + 1, gap))
    return rows


def summarize_weights(by_method):
    """Weight-signature table: how the |w| distribution moved from epoch 1 to the
    last epoch (mean over seeds). The τ(w) signature is median DOWN, max UP."""
    print(f"\n{'method':<14} {'med|w| ep1':>10} {'med|w| end':>10} {'Δmed%':>7} "
          f"{'max|w| ep1':>10} {'max|w| end':>10} {'Δmax%':>7} {'frac<thr end':>13}")
    print('-' * 88)
    for m in METHOD_ORDER:
        trajs = by_method.get(m)
        if not trajs:
            continue
        def mean_at(key, idx):
            vals = [t[idx].get(key) for t in trajs if t[idx].get(key) is not None]
            return sum(vals) / len(vals) if vals else float('nan')
        med0, med1 = mean_at('median_abs_w', 0), mean_at('median_abs_w', -1)
        max0, max1 = mean_at('max_abs_w', 0), mean_at('max_abs_w', -1)
        frac1 = mean_at('frac_below_thr', -1)
        print(f"{m:<14} {med0:>10.5f} {med1:>10.5f} {100*(med1-med0)/med0:>+6.1f}% "
              f"{max0:>10.4f} {max1:>10.4f} {100*(max1-max0)/max0:>+6.1f}% {frac1:>13.4f}")


def make_figures(by_method, scale, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"\n[figures skipped] matplotlib unavailable ({e}). "
              f"Numeric summary above is still valid.")
        return

    os.makedirs(out_dir, exist_ok=True)
    present = [m for m in METHOD_ORDER if m in by_method]

    # 1. Overfitting-delay. Left: full train (dashed) vs val (solid) on log-y.
    #    Right: zoom on the late-epoch validation curves, where the methods
    #    separate — invisible on the full scale because of the epoch-1 PPL.
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(11, 4.5))
    half = None
    for m in present:
        ep, val = _mean_over_seeds(by_method[m], 'val_ppl')
        _, tr = _mean_over_seeds(by_method[m], 'train_ppl')
        c = METHOD_COLORS.get(m, '#999')
        ax.plot(ep, val, marker='o', markersize=3.5, linewidth=1.8, label=m, color=c)
        ax.plot(ep, tr, linestyle='--', alpha=0.45, linewidth=1.3, color=c)
        half = half or max(2, len(ep) // 2)
        axz.plot(ep[half:], val[half:], marker='o', markersize=4, linewidth=2,
                 label=m, color=c)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Perplexity (solid = validation, dashed = train)')
    ax.set_title('Full training (log scale)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')
    axz.set_xlabel('Epoch'); axz.set_ylabel('Validation perplexity')
    axz.set_title('Late epochs (zoom)')
    axz.grid(alpha=0.3)
    fig.suptitle(f'Train vs validation perplexity — GPT-2 {scale} from scratch')
    p = os.path.join(out_dir, f'mechanism_overfitting_delay_{scale}.png')
    fig.tight_layout(); fig.savefig(p, dpi=200); plt.close(fig)
    print(f'[OK] {p}')

    # 2. Weight-distribution dynamics (2x2: median/max = adaptivity signature,
    #    frac-below-threshold and L2 norm = bulk shrinkage).
    ylabels = {'median_abs_w': 'median |w|', 'max_abs_w': 'max |w|',
               'frac_below_thr': 'fraction |w| < threshold', 'l2_norm': '‖w‖₂'}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for ax, key in zip(axes.flat, WEIGHT_SERIES):
        for m in present:
            ep, ser = _mean_over_seeds(by_method[m], key)
            if ser:
                ax.plot(ep, ser, marker='.', markersize=5, linewidth=1.8,
                        label=m, color=METHOD_COLORS.get(m, '#999'))
        ax.set_xlabel('Epoch'); ax.set_ylabel(ylabels.get(key, key)); ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f'Weight-magnitude dynamics — GPT-2 {scale} from scratch')
    p = os.path.join(out_dir, f'mechanism_weight_dynamics_{scale}.png')
    fig.tight_layout(); fig.savefig(p, dpi=200); plt.close(fig)
    print(f'[OK] {p}')


def make_adaptivity_figure(by_method, hyper, scale, out_dir, lr=3e-4):
    """The analytic adaptivity signature implied by the TUNED hyperparameters.

    Left: effective per-step multiplicative decay rate as a function of |w| —
    τ(w) gives rate(w) = λ/(τ0+α|w|), i.e. strongest on small weights, capped
    force on large ones; Tau(alpha=0) and WD-tuned are horizontal lines (their
    rate is magnitude-independent). Right: the implicit penalty Ω(w) each rule
    is a (decoupled) gradient step on — τ(w) integrates to the Fair function,
    quadratic near 0 and linear in the tail.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"\n[adaptivity figure skipped] ({e})")
        return
    tau_hp = hyper.get('τ(w)')
    if not tau_hp:
        print('\n[adaptivity figure skipped] no τ(w) hyperparams in instrumentation data')
        return
    rho, delta = canonical(tau_hp)          # w <- w - rho * w / (1 + |w|/delta)
    w = np.logspace(-4, 0.5, 400)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # Left: per-step multiplicative decay rate vs |w|.
    ax1.plot(w, rho / (1.0 + w / delta), color=METHOD_COLORS['τ(w)'], linewidth=2.2,
             label=f'τ(w):  ρ/(1+|w|/δ)   (δ = {delta:.3g})')
    if 'Tau(alpha=0)' in hyper:
        r0, _ = canonical(hyper['Tau(alpha=0)'])
        ax1.axhline(r0, color=METHOD_COLORS['Tau(alpha=0)'], linewidth=1.8,
                    linestyle='--', label='Tau(α=0):  ρ (constant)')
    if 'WD-tuned' in hyper:
        rwd = lr * hyper['WD-tuned'].get('wd', 0.0)
        ax1.axhline(rwd, color=METHOD_COLORS['WD-tuned'], linewidth=1.8,
                    linestyle=':', label='WD-tuned:  η·wd (constant)')
    # Mark where the trained τ(w) weight distribution actually sits.
    trajs = by_method.get('τ(w)')
    if trajs:
        med = sum(t[-1]['median_abs_w'] for t in trajs) / len(trajs)
        mx = sum(t[-1]['max_abs_w'] for t in trajs) / len(trajs)
        for v, lab in ((med, 'median |w|'), (mx, 'max |w|')):
            ax1.axvline(v, color='#666', alpha=0.5, linewidth=1)
            ax1.text(v, ax1.get_ylim()[1], f' {lab}', fontsize=7, color='#444',
                     ha='left', va='top', rotation=90)
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_xlabel('|w|'); ax1.set_ylabel('per-step decay rate')
    ax1.set_title('Effective decay rate vs weight magnitude')
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3, which='both')

    # Right: implicit penalties, normalised to Ω(1)=1 for shape comparison.
    wl = np.linspace(0, 1.2, 400)
    fair = wl - delta * np.log1p(wl / delta)              # ∝ implicit Ω of τ(w) (Fair)
    ax2.plot(wl, fair / fair[-1] * (wl[-1]) , color=METHOD_COLORS['τ(w)'],
             linewidth=2.2, label='τ(w) implicit Ω (Fair): quad→linear')
    ax2.plot(wl, wl ** 2 / wl[-1] ** 2 * wl[-1], color=METHOD_COLORS['L2'],
             linewidth=1.8, linestyle='--', label='L2 / const. decay: w²')
    ax2.plot(wl, wl, color=METHOD_COLORS['L1'], linewidth=1.8, linestyle=':',
             label='L1: |w|')
    ax2.set_xlabel('|w|'); ax2.set_ylabel('penalty (normalised)')
    ax2.set_title('Implicit penalty shape')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    fig.suptitle(f'τ(w) adaptivity signature — tuned hyperparams, GPT-2 {scale}')
    p = os.path.join(out_dir, f'mechanism_adaptivity_{scale}.png')
    fig.tight_layout(); fig.savefig(p, dpi=200); plt.close(fig)
    print(f'[OK] {p}')


def main():
    ap = argparse.ArgumentParser(description='τ-decay mechanism analysis (Workstream 2)')
    ap.add_argument('--scale', default='medium',
                    help='From-scratch scale tag to analyse (tiny/small/medium/large).')
    ap.add_argument('--dir', default='results/instrumentation',
                    help='Directory of instrumentation JSONs.')
    ap.add_argument('--out', default='figures',
                    help='Output directory for figures.')
    args = ap.parse_args()

    by_method, hyper = load_trajectories(args.dir, args.scale)
    if not by_method:
        print(f"No instrumentation trajectories found for scale '{args.scale}' in "
              f"'{args.dir}'. Run e.g.:\n"
              f"  python gpt2_wikitext_standardized.py --scale {args.scale} --instrument")
        return
    print(f"Loaded methods: {', '.join(f'{m}({len(v)})' for m, v in by_method.items())}")
    summarize(by_method)
    summarize_weights(by_method)
    make_figures(by_method, args.scale, args.out)
    make_adaptivity_figure(by_method, hyper, args.scale, args.out)


if __name__ == '__main__':
    main()
