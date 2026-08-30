"""
Create publication-quality composite figures for the paper.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

COLORS = {
    'Baseline': '#1f77b4',
    'L1': '#ff7f0e',
    'L2': '#2ca02c',
    'ElasticNet': '#d62728',
    'SCAD': '#9467bd',
    'MCP': '#8c564b',
    'LSP': '#e377c2',
    'WD-tuned': '#17becf',
    'Tau(alpha=0)': '#bcbd22',
    'Tau(AdamW-scope)': '#7f7f7f',
    '\u03c4(w)': '#FFD700',
}

METHODS_ORDER = ['Baseline', 'L1', 'L2', 'ElasticNet', 'SCAD', 'MCP', 'LSP',
                 'WD-tuned', 'Tau(alpha=0)', '\u03c4(w)']

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
FIGURES_OUT = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(FIGURES_OUT, exist_ok=True)

# Allow importing from project root
sys.path.insert(0, ROOT_DIR)
from analysis.stats_utils import load_csv, welch_ttest, load_runs


def n_seeds_of(name, metric, default=5):
    """Per-benchmark n read from the per-seed JSON (the 66M member is n = 10 after the
    early-stopping fix); falls back to `default` when no JSON exists."""
    runs = load_runs(name, 'τ(w)', metric)
    return len(runs) if runs else default


# ============================================================================
# FIGURE 1: Multi-panel performance comparison (the 8 single-scale benchmarks)
# ============================================================================
def figure1_all_benchmarks():
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.5))
    fig.suptitle('Performance Comparison Across the 8 Single-Scale Benchmarks',
                 fontsize=16, fontweight='bold', y=1.00)

    # name, metric, title, mode, scale, ylabel
    benchmarks = [
        ('sin_regression',     'test_mse', 'sin(x)',        'min', 1e3, 'MSE \u00d710\u207b\u00b3'),
        ('complex_regression', 'test_mse', 'Friedman 10D',  'min', 1,   'MSE'),
        ('mnist',              'test_acc', 'MNIST',         'max', 100, 'Accuracy %'),
        ('cifar',              'test_acc', 'CIFAR-10 CNN',  'max', 100, 'Accuracy %'),
        ('vit_cifar',          'test_acc', 'ViT CIFAR-10',  'max', 100, 'Accuracy %'),
        ('bert_sst2',          'test_acc', 'BERT custom',   'max', 100, 'Accuracy %'),
        ('gpt2_wikitext',      'test_ppl', 'GPT-2 custom',  'min', 1,   'PPL'),
        ('smollm2_wikitext',   'test_ppl', 'SmolLM2-135M',  'min', 1,   'PPL'),
    ]

    for idx, (name, metric, title, mode, scale, ylabel) in enumerate(benchmarks):
        ax = axes[idx // 4][idx % 4]
        try:
            df = load_csv(name)
        except FileNotFoundError:
            ax.axis('off')
            continue

        methods = df['method'].tolist()
        means = df[f'{metric}_mean'].values * scale
        stds = df[f'{metric}_std'].values * scale

        # Reorder to canonical method order
        order = [methods.index(m) for m in METHODS_ORDER if m in methods]
        methods = [methods[i] for i in order]
        means = means[order]
        stds = stds[order]

        colors = [COLORS.get(m, '#999') for m in methods]
        x = np.arange(len(methods))

        bars = ax.bar(x, means, yerr=stds, capsize=2, color=colors, alpha=0.85,
                      edgecolor='black', linewidth=0.6)

        # Highlight best
        best_idx = int(np.argmax(means)) if mode == 'max' else int(np.argmin(means))
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(2.0)

        # Highlight tau
        if '\u03c4(w)' in methods:
            bars[methods.index('\u03c4(w)')].set_hatch('//')

        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=8)
        short_labels = [m.replace('ElasticNet', 'EN').replace('Baseline', 'Base') for m in methods]
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=7)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='y', labelsize=7)

        # Auto-zoom y-axis: exclude clear outliers (>3 std from median of the rest)
        # so a single method's catastrophic failure doesn't compress the rest.
        vals = means.astype(float)
        med = float(np.median(vals))
        # Robust spread = MAD * 1.4826
        mad = float(np.median(np.abs(vals - med)))
        spread = max(mad * 1.4826, 1e-12)
        good_mask = np.abs(vals - med) < 6 * spread  # very lenient outlier filter
        if good_mask.sum() < 3:
            good_mask = np.ones_like(vals, dtype=bool)
        good = vals[good_mask] + (stds[good_mask] if mode == 'max' else -stds[good_mask])
        if mode == 'max':
            ymin = max(0.0, vals[good_mask].min() - stds[good_mask].max() - 0.5)
            ymax = vals[good_mask].max() + stds[good_mask].max() + 0.5
            ax.set_ylim(ymin, ymax)
        else:
            ymin = max(0.0, vals[good_mask].min() - stds[good_mask].max() * 0.5)
            ymax = vals[good_mask].max() + stds[good_mask].max() + 0.5
            ax.set_ylim(ymin, ymax)

        # If any method's actual value is outside the zoomed range, annotate it
        for i, (m, v) in enumerate(zip(methods, vals)):
            if not good_mask[i]:
                ax.annotate(f'{m}={v:.1f}',
                            xy=(i, ymax * 0.98 if mode == 'max' else ymin + (ymax-ymin)*0.05),
                            xytext=(0, 0), textcoords='offset points',
                            ha='center', fontsize=6, color='red', fontweight='bold',
                            rotation=0)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(os.path.join(FIGURES_OUT, 'fig1_all_benchmarks.png'), dpi=200, bbox_inches='tight')
    print('[OK] fig1_all_benchmarks.png')
    plt.close()


# ============================================================================
# FIGURE 2: GPT-2 results (star result) with confidence intervals
# ============================================================================
def figure2_gpt2_detail():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('GPT-2 (7.4M) on WikiText-2, trained from scratch: the \u03c4 family vs the field',
                 fontsize=14, fontweight='bold')

    df = load_csv('gpt2_wikitext')

    # Reorder
    methods = df['method'].tolist()
    order = [methods.index(m) for m in METHODS_ORDER if m in methods]

    m_ordered = [methods[i] for i in order]
    means = df['test_ppl_mean'].values[order]
    stds = df['test_ppl_std'].values[order]
    ci_lo = df['test_ppl_ci_lower'].values[order]
    ci_hi = df['test_ppl_ci_upper'].values[order]

    # Panel A: Bar chart (all methods, including LSP \u2014 data-driven ylim)
    colors = [COLORS.get(m, '#999') for m in m_ordered]
    x = np.arange(len(m_ordered))

    bars = ax1.bar(x, means, yerr=stds, capsize=5, color=colors,
                   alpha=0.85, edgecolor='black', linewidth=1)

    # Highlight tau
    tau_idx = m_ordered.index('\u03c4(w)')
    bars[tau_idx].set_edgecolor('red')
    bars[tau_idx].set_linewidth(3)
    bars[tau_idx].set_hatch('//')

    # Add value labels
    for i, (bar, mean) in enumerate(zip(bars, means)):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_ylabel('Test Perplexity (lower is better)', fontweight='bold')
    ax1.set_title('A. Performance Comparison', fontweight='bold')
    short = [m.replace('ElasticNet', 'EN').replace('Baseline', 'Base') for m in m_ordered]
    ax1.set_xticks(x)
    ax1.set_xticklabels(short, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    # Data-driven ylim: zoom on the bulk of methods (MAD outlier filter), keep
    # LSP (or any other outlier) visible by annotating its actual value above
    # the zoomed range instead of hardcoding ranges.
    vals = means.astype(float)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    spread = max(mad * 1.4826, 1e-12)
    good_mask = np.abs(vals - med) < 6 * spread
    if good_mask.sum() < 3:
        good_mask = np.ones_like(vals, dtype=bool)
    good_vals = vals[good_mask]
    good_stds = stds[good_mask]
    pad = max(good_stds.max(), 1.0)
    ymin = max(0.0, float(good_vals.min()) - pad * 1.2)
    ymax = float(good_vals.max()) + pad * 2.5
    ax1.set_ylim(ymin, ymax)
    # Annotate outliers above the zoomed range
    for i, (m, v) in enumerate(zip(m_ordered, vals)):
        if not good_mask[i]:
            ax1.annotate(f'{m}: {v:.1f}', xy=(i, ymax * 0.96),
                         ha='center', va='top', fontsize=8, color='red',
                         fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.2',
                                   facecolor='lightyellow', edgecolor='red'))

    # Panel B: Confidence intervals (all methods including LSP)
    y = np.arange(len(m_ordered))
    colors2 = [COLORS.get(m, '#999') for m in m_ordered]
    for i, (m, mean, lo, hi, c) in enumerate(zip(m_ordered, means, ci_lo, ci_hi, colors2)):
        marker = 'D' if m == '\u03c4(w)' else 'o'
        size = 12 if m == '\u03c4(w)' else 8
        lw = 3 if m == '\u03c4(w)' else 2
        ax2.errorbar(mean, i, xerr=[[mean-lo], [hi-mean]], fmt=marker, color=c,
                    markersize=size, capsize=5, capthick=lw, elinewidth=lw, alpha=0.9)

    ax2.set_yticks(y)
    ax2.set_yticklabels(m_ordered)
    ax2.set_xlabel('Test Perplexity (lower is better)', fontweight='bold')
    ax2.set_title('B. 95% Confidence Intervals', fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    tau_mean = means[tau_idx]
    tau_std = stds[tau_idx]
    ax2.axvline(x=tau_mean, color='gold', linestyle='--', alpha=0.5, linewidth=2)
    # Data-driven xlim mirroring the Panel A bulk zoom
    ax2.set_xlim(ymin, ymax)

    # Data-driven significance: stars from Welch's t-test on summary statistics.
    # Marker reflects pre-correction p-value; section 5.6 reports the BH-FDR /
    # Bonferroni-corrected counts.
    def _stars(p):
        if p < 0.001:
            return '***'
        if p < 0.01:
            return '**'
        if p < 0.05:
            return '*'
        return 'ns'
    for i, m in enumerate(m_ordered):
        if m == '\u03c4(w)':
            continue
        _, p = welch_ttest(tau_mean, tau_std, means[i], stds[i],
                           n=n_seeds_of('gpt2_wikitext', 'test_ppl'))
        tag = _stars(p)
        ax2.text(means[i] + 0.3, i + 0.15, tag, fontsize=10,
                 color='red' if tag != 'ns' else 'gray', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_OUT, 'fig2_gpt2_detail.png'), dpi=300, bbox_inches='tight')
    print('[OK] fig2_gpt2_detail.png')
    plt.close()


# ============================================================================
# FIGURE 3: Architectural affinity heatmap
# ============================================================================
def figure3_architectural_affinity():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [2, 1]})
    fig.suptitle('\u03c4(w) rank by benchmark and by architecture family',
                 fontsize=14, fontweight='bold')

    # Panel A: Rank heatmap. Ranks are computed from the CSVs (1 = best);
    # ties are broken by lowest mean-position (pandas default).
    def _tau_rank(name, metric, mode):
        df = load_csv(name)
        col = f'{metric}_mean'
        ascending = (mode == 'min')          # lower is better → smaller value → rank 1
        df = df.copy()
        df['rank'] = df[col].rank(method='min', ascending=ascending).astype(int)
        row = df[df['method'] == 'τ(w)']
        if row.empty:
            return float('nan')
        return int(row.iloc[0]['rank'])

    bench_spec = [
        ('sin(x)',     'sin_regression',     'test_mse', 'min', 'Regression'),
        ('Complex',    'complex_regression', 'test_mse', 'min', 'Regression'),
        ('MNIST',      'mnist',              'test_acc', 'max', 'CNN'),
        ('CIFAR',      'cifar',              'test_acc', 'max', 'CNN'),
        ('ViT',        'vit_cifar',          'test_acc', 'max', 'ViT'),
        ('BERT',       'bert_sst2',          'test_acc', 'max', 'LM'),
        ('GPT-2',      'gpt2_wikitext',      'test_ppl', 'min', 'LM'),
        ('SmolLM2',    'smollm2_wikitext',   'test_ppl', 'min', 'LM'),
    ]
    benchmarks = [b[0] for b in bench_spec]
    tau_ranks = [_tau_rank(b[1], b[2], b[3]) for b in bench_spec]
    categories = [b[4] for b in bench_spec]
    cat_colors = {'Regression': '#AEC6CF', 'CNN': '#FFB347', 'ViT': '#B39EB5', 'LM': '#77DD77'}

    colors_bar = [cat_colors[c] for c in categories]
    x = np.arange(len(benchmarks))

    bars = ax1.bar(x, tau_ranks, color=colors_bar, edgecolor='black', linewidth=1, alpha=0.85)

    # Color bars by rank quality
    for i, (bar, rank) in enumerate(zip(bars, tau_ranks)):
        if rank == 1:
            bar.set_edgecolor('gold')
            bar.set_linewidth(3)
            ax1.text(i, rank - 0.3, '\u2605', ha='center', fontsize=14, color='gold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(benchmarks, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Rank (1 = best)', fontweight='bold')
    ax1.set_title('A. \u03c4(w) Ranking by Benchmark', fontweight='bold')
    ax1.set_ylim(0, 9)
    ax1.invert_yaxis()
    ax1.set_ylim(9, 0)
    ax1.axhline(y=1, color='gold', linestyle='--', alpha=0.5, linewidth=1)
    ax1.grid(axis='y', alpha=0.3)

    # Add vertical separators between categories
    ax1.axvline(x=1.5, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(x=3.5, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(x=4.5, color='gray', linestyle=':', alpha=0.5)

    # Legend for categories
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=cat_colors[c], edgecolor='black', label=c)
                      for c in ['Regression', 'CNN', 'ViT', 'LM']]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=9)

    # Panel B: Average rank by architecture (computed live from Panel A ranks).
    def _avg(cat):
        vals = [r for r, c in zip(tau_ranks, categories) if c == cat and r == r]  # filter NaN
        return float(np.mean(vals)) if vals else float('nan')
    arch_names = ['Feedforward\n(Regression)', 'CNN', 'Vision\nTransformer', 'Language\nModels']
    avg_ranks = [_avg('Regression'), _avg('CNN'), _avg('ViT'), _avg('LM')]
    arch_colors = ['#AEC6CF', '#FFB347', '#B39EB5', '#77DD77']

    bars2 = ax2.barh(arch_names, avg_ranks, color=arch_colors, edgecolor='black', linewidth=1, alpha=0.85)

    for bar, rank in zip(bars2, avg_ranks):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{rank:.1f}', va='center', fontweight='bold', fontsize=11)

    ax2.set_xlabel('Average Rank (lower is better)', fontweight='bold')
    ax2.set_title('B. Average Rank by Architecture', fontweight='bold')
    ax2.set_xlim(0, 8)
    ax2.axvline(x=1, color='gold', linestyle='--', alpha=0.5, linewidth=1)
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_OUT, 'fig3_architectural_affinity.png'), dpi=300, bbox_inches='tight')
    print('[OK] fig3_architectural_affinity.png')
    plt.close()


# ============================================================================
# FIGURE 4: Transformer benchmarks confidence intervals (panel)
# ============================================================================
def figure4_transformer_ci():
    # The four transformer benchmarks that remain in Table 1 (scope restricted 2026-08-29).
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Confidence Intervals for the Transformer Benchmarks of Table 1 (95% CI)',
                 fontsize=14, fontweight='bold')

    transformer_benchmarks = [
        ('bert_sst2', 'test_acc', 'BERT-tiny SST-2 - Accuracy', 100),
        ('gpt2_wikitext', 'test_ppl', 'GPT-2 WikiText-2 - Perplexity', 1),
        ('smollm2_wikitext', 'test_ppl', 'SmolLM2-135M WikiText-2 - Perplexity', 1),
        ('vit_cifar', 'test_acc', 'ViT CIFAR-10 - Accuracy', 100),
    ]

    for idx, (name, metric, title, scale) in enumerate(transformer_benchmarks):
        ax = axes[idx // 2][idx % 2]
        df = load_csv(name)

        methods = df['method'].tolist()
        order = [methods.index(m) for m in METHODS_ORDER if m in methods]

        m_ord = [methods[i] for i in order]
        means = df[f'{metric}_mean'].values[order] * scale
        ci_lo = df[f'{metric}_ci_lower'].values[order] * scale
        ci_hi = df[f'{metric}_ci_upper'].values[order] * scale

        # Exclude LSP if it has nan CI
        mask = ~(np.isnan(ci_lo) | np.isnan(ci_hi))
        # Also exclude LSP for GPT-2 (too far off)
        if name == 'gpt2_wikitext':
            mask = mask & np.array([m != 'LSP' for m in m_ord])

        m_filt = [m for m, k in zip(m_ord, mask) if k]
        means_f = means[mask]
        ci_lo_f = ci_lo[mask]
        ci_hi_f = ci_hi[mask]

        y = np.arange(len(m_filt))
        for i, (m, mean, lo, hi) in enumerate(zip(m_filt, means_f, ci_lo_f, ci_hi_f)):
            color = COLORS.get(m, '#999')
            marker = 'D' if m == '\u03c4(w)' else 'o'
            ms = 10 if m == '\u03c4(w)' else 7
            lw = 2.5 if m == '\u03c4(w)' else 1.5
            ax.errorbar(mean, i, xerr=[[mean-lo], [hi-mean]], fmt=marker, color=color,
                       markersize=ms, capsize=4, capthick=lw, elinewidth=lw, alpha=0.9)

        ax.set_yticks(y)
        ax.set_yticklabels([m.replace('ElasticNet', 'EN').replace('Baseline', 'Base') for m in m_filt], fontsize=9)
        ax.set_title(title, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Highlight tau line
        tau_idx = m_filt.index('\u03c4(w)') if '\u03c4(w)' in m_filt else -1
        if tau_idx >= 0:
            ax.axvline(x=means_f[tau_idx], color='gold', linestyle='--', alpha=0.4, linewidth=2)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_OUT, 'fig4_transformer_ci.png'), dpi=300, bbox_inches='tight')
    print('[OK] fig4_transformer_ci.png')
    plt.close()


# ============================================================================
# FIGURE 5: Statistical wins/losses summary
# ============================================================================
def figure5_wins_losses():
    """τ(w) wins and losses per single-scale benchmark, computed from the CSVs.

    Significance uses the same family-wide Benjamini-Hochberg correction (q = 0.05) as
    build_paper.py: the family is every τ(w)-vs-competitor Welch test over the single-scale
    benchmarks, the from-scratch scale sweep and the WikiText-103 confirmation, with the
    per-benchmark n. Tau(alpha=0) is an ablation of τ(w) and is never counted.
    """
    from analysis.stats_utils import benjamini_hochberg

    competitors = ['Baseline', 'L1', 'L2', 'ElasticNet', 'SCAD', 'MCP', 'LSP', 'WD-tuned']
    # n per benchmark from the per-seed JSONs (66M is n = 10 after the early-stopping fix).
    n_seeds = {}
    single_scale = [('sin(x)', 'sin_regression', 'test_mse', 'min'),
                    ('Complex', 'complex_regression', 'test_mse', 'min'),
                    ('MNIST', 'mnist', 'test_acc', 'max'),
                    ('CIFAR', 'cifar', 'test_acc', 'max'),
                    ('ViT', 'vit_cifar', 'test_acc', 'max'),
                    ('BERT', 'bert_sst2', 'test_acc', 'max'),
                    ('GPT-2', 'gpt2_wikitext', 'test_ppl', 'min'),
                    ('SmolLM2', 'smollm2_wikitext', 'test_ppl', 'min')]
    # Members of the correction family that are not plotted here.
    family_only = [(None, 'gpt2_tiny_wikitext', 'test_ppl', 'min'),
                   (None, 'gpt2_medium_wikitext', 'test_ppl', 'min'),
                   (None, 'gpt2_large_wikitext', 'test_ppl', 'min'),
                   (None, 'gpt2_wt103', 'test_ppl', 'min')]

    tests = []  # (plot label or None, tau better?, p)
    for label, name, metric, mode in single_scale + family_only:
        df = load_csv(name)
        n = n_seeds.get(name) or n_seeds_of(name, metric)
        tau = df[df['method'] == 'τ(w)'].iloc[0]
        tau_m, tau_s = float(tau[f'{metric}_mean']), float(tau[f'{metric}_std'])
        for m in competitors:
            row = df[df['method'] == m]
            if row.empty:
                continue
            m_m, m_s = float(row.iloc[0][f'{metric}_mean']), float(row.iloc[0][f'{metric}_std'])
            _, p = welch_ttest(tau_m, tau_s, m_m, m_s, n=n)
            better = (tau_m > m_m) if mode == 'max' else (tau_m < m_m)
            tests.append((label, better, p))
    sig = benjamini_hochberg([t[2] for t in tests], q=0.05)

    benchmarks = [b[0] for b in single_scale]

    def _count(label, want_better, want_sig):
        return sum(1 for (lab, better, _), ok in zip(tests, sig)
                   if lab == label and better == want_better and bool(ok) == want_sig)
    sig_wins = [_count(b, True, True) for b in benchmarks]
    ns_wins = [_count(b, True, False) for b in benchmarks]
    ns_losses = [_count(b, False, False) for b in benchmarks]
    sig_losses = [_count(b, False, True) for b in benchmarks]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(benchmarks))
    width = 0.2

    ax.bar(x - 1.5*width, sig_wins, width, label='Sig. wins (BH-FDR q=0.05)', color='#2ca02c', edgecolor='black', alpha=0.85)
    ax.bar(x - 0.5*width, ns_wins, width, label='Non-sig. wins', color='#98df8a', edgecolor='black', alpha=0.85)
    ax.bar(x + 0.5*width, ns_losses, width, label='Non-sig. losses', color='#ffbb78', edgecolor='black', alpha=0.85)
    ax.bar(x + 1.5*width, sig_losses, width, label='Sig. losses (BH-FDR q=0.05)', color='#d62728', edgecolor='black', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=45, ha='right')
    ax.set_ylabel('Number of methods', fontweight='bold')
    ax.set_title('τ(w) Statistical Wins and Losses vs. All Competitors (Welch t-test, family-wide BH-FDR)',
                fontweight='bold', fontsize=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 8)

    # Vertical separators: classic (0-3) | ViT (4) | language models (5-7)
    ax.axvline(x=3.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=4.5, color='gray', linestyle=':', alpha=0.5)
    ax.text(1.5, 7.5, 'Classic', ha='center', fontsize=9, style='italic', color='gray')
    ax.text(4, 7.5, 'ViT', ha='center', fontsize=9, style='italic', color='gray')
    ax.text(6, 7.5, 'Language Models', ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_OUT, 'fig5_wins_losses.png'), dpi=300, bbox_inches='tight')
    print('[OK] fig5_wins_losses.png')
    plt.close()


def figure2b_scale_sweep():
    """From-scratch GPT-2 scale sweep: test PPL vs parameter count (core-6).

    Left: absolute test perplexity per method across the four from-scratch
    scales (log-x). GPT-2 Small (7.4M) uses a longer 30-epoch budget: open
    markers + dotted connection, so within-scale comparisons stay honest.
    Right: τ(w) advantage (ΔPPL) over the Baseline and over tuned decoupled
    weight decay at each scale — the over-capacity trend itself.
    """
    # Trainable-parameter counts from the CSVs (total_params): 2.1M, 7.4M, 18.1M and
    # 65.6M (tied embeddings).
    scales = [('gpt2_tiny_wikitext', 2.1e6, 'tiny\n2M'),
              ('gpt2_wikitext', 7.4e6, 'Small\n7.4M'),
              ('gpt2_medium_wikitext', 18.1e6, 'medium\n18M'),
              ('gpt2_large_wikitext', 65.6e6, 'large\n66M')]
    methods = ['Baseline', 'L2', 'ElasticNet', 'WD-tuned', 'Tau(alpha=0)', 'τ(w)']
    dfs = {name: load_csv(name) for name, _, _ in scales}

    def val(name, m, stat='mean'):
        df = dfs[name]
        row = df[df['method'] == m]
        return float(row.iloc[0][f'test_ppl_{stat}']) if not row.empty else np.nan

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    xs = [x for _, x, _ in scales]
    for m in methods:
        ys = [val(name, m) for name, _, _ in scales]
        c = COLORS[m]
        lw = 2.4 if m == 'τ(w)' else 1.6
        # matched-recipe points (12-epoch budget): tiny, medium, large
        ax1.plot([xs[0], xs[2], xs[3]], [ys[0], ys[2], ys[3]], marker='o',
                 markersize=5, linewidth=lw, color=c, label=m)
        # Small (30-epoch budget): open marker, dotted link, same entity color
        ax1.plot(xs[1], ys[1], marker='o', markersize=6, markerfacecolor='white',
                 color=c, linestyle='none')
        ax1.plot(xs[:3], ys[:3], linestyle=':', linewidth=0.9, color=c, alpha=0.5)
    ax1.set_xscale('log')
    ax1.set_xticks(xs)
    ax1.set_xticklabels([lab for _, _, lab in scales], fontsize=8)
    from matplotlib.ticker import NullFormatter
    ax1.xaxis.set_minor_formatter(NullFormatter())   # log-scale minor labels collide with the size labels
    ax1.set_xlabel('Model size (from-scratch GPT-2)')
    ax1.set_ylabel('Test perplexity')
    ax1.set_title('Scale sweep (open marker = 30-epoch budget)')
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)

    labels = [lab for _, _, lab in scales]
    x = np.arange(len(scales))
    width = 0.38
    d_base = [val(n, 'Baseline') - val(n, 'τ(w)') for n, _, _ in scales]
    d_wd = [val(n, 'WD-tuned') - val(n, 'τ(w)') for n, _, _ in scales]
    ax2.bar(x - width / 2, d_base, width, color=COLORS['Baseline'], alpha=0.85,
            edgecolor='black', linewidth=0.6, label='vs Baseline')
    ax2.bar(x + width / 2, d_wd, width, color=COLORS['WD-tuned'], alpha=0.85,
            edgecolor='black', linewidth=0.6, label='vs WD-tuned')
    def _dlab(v):
        return '0.0' if abs(v) < 0.05 else f'{v:+.1f}'
    for xi, v in zip(x - width / 2, d_base):
        ax2.text(xi, max(v, 0) + 0.15, _dlab(v), ha='center', fontsize=7)
    for xi, v in zip(x + width / 2, d_wd):
        ax2.text(xi, max(v, 0) + 0.15, _dlab(v), ha='center', fontsize=7)
    ax2.axhline(0, color='#666', linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel('Δ test PPL (competitor − τ(w))')
    ax2.set_title('τ(w) margin tracks overfitting pressure, not size')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_OUT, 'fig2b_scale_sweep.png'), dpi=300,
                bbox_inches='tight')
    plt.close()
    print('[OK] fig2b_scale_sweep.png')


if __name__ == '__main__':
    print("Creating paper figures...")
    figure1_all_benchmarks()
    figure2_gpt2_detail()
    figure2b_scale_sweep()
    figure3_architectural_affinity()
    figure4_transformer_ci()
    figure5_wins_losses()
    print(f"\nAll figures saved to {FIGURES_OUT}")
