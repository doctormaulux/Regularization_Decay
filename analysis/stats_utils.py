"""
Shared statistical utility functions.
Used by statistical_analysis.py, build_paper.py, and create_paper_figures.py.
"""
import json
import os
import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'new results')


def load_csv(name):
    """Load a benchmark CSV from the results directory."""
    return pd.read_csv(os.path.join(RESULTS_DIR, f'{name}_standardized_results.csv'))


def get_val(df, method, metric, stat='mean'):
    """Get a specific statistic for a method from a benchmark DataFrame."""
    row = df[df['method'] == method]
    if row.empty:
        return float('nan')
    return row.iloc[0][f'{metric}_{stat}']


def welch_ttest(mean_a, std_a, mean_b, std_b, n=5):
    """Welch's t-test from summary statistics."""
    se = np.sqrt(std_a**2/n + std_b**2/n)
    if se == 0:
        return 0.0, 1.0
    t_stat = (mean_a - mean_b) / se
    df = (std_a**2/n + std_b**2/n)**2 / ((std_a**2/n)**2/(n-1) + (std_b**2/n)**2/(n-1))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    return t_stat, p_value


def load_runs(name, method, metric):
    """Per-seed values for one method, in seed order, from the benchmark's JSON.

    The CSVs carry only summary statistics, which forces every test to be an unpaired
    Welch test. But all methods on a benchmark are trained on the SAME seed list in the
    same order, so the runs are naturally paired and the pairing is worth exploiting:
    seed-to-seed variation is largely shared (same initialisation, same data order), and
    removing it is exactly what a paired test does.

    Returns a list ordered by run index, or [] if the method is absent.
    """
    path = os.path.join(RESULTS_DIR, f'{name}_standardized_results.json')
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as fh:
        blob = json.load(fh)
    runs = (blob.get('results') or {}).get(method) or []
    out = []
    for r in sorted(runs, key=lambda r: r.get('run', 0)):
        v = (r.get('metrics') or {}).get(metric)
        if v is not None:
            out.append(float(v))
    return out


def paired_ttest(a, b):
    """Paired t-test on per-seed values. Returns (t, p, mean_diff, n_pairs).

    a and b must be same-seed-aligned. Falls back to (nan, nan, nan, 0) if the two
    series cannot be paired, so callers can drop back to the unpaired Welch test rather
    than silently comparing mismatched runs.
    """
    n = min(len(a), len(b))
    if n < 2:
        return float('nan'), float('nan'), float('nan'), n
    d = np.asarray(a[:n], dtype=float) - np.asarray(b[:n], dtype=float)
    if np.allclose(d, d[0]):
        # zero variance in the differences: deterministic offset, p is degenerate
        return (float('inf') if d[0] else 0.0), (0.0 if d[0] else 1.0), float(d.mean()), n
    t, p = stats.ttest_rel(a[:n], b[:n])
    return float(t), float(p), float(d.mean()), n


def bootstrap_ci(a, b=None, statistic='mean_diff', n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CI, for the paired difference by default.

    Reported alongside the t-interval because at n = 3-10 the normal-theory interval
    leans on an assumption the sample cannot check. Resamples pairs, not the two series
    independently, so the pairing is preserved.
    """
    rng = np.random.default_rng(seed)
    if b is not None:
        n = min(len(a), len(b))
        if n < 2:
            return float('nan'), float('nan')
        d = np.asarray(a[:n], dtype=float) - np.asarray(b[:n], dtype=float)
    else:
        d = np.asarray(a, dtype=float)
        n = len(d)
        if n < 2:
            return float('nan'), float('nan')
    idx = rng.integers(0, n, size=(n_boot, n))
    stat = d[idx].mean(axis=1)
    lo, hi = np.percentile(stat, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def hedges_g(mean_a, std_a, mean_b, std_b, n=5):
    """Hedges' g: Cohen's d with the small-sample bias correction.

    Cohen's d overestimates the population effect at the sample sizes used here (n = 3-10),
    which is exactly where this study lives. The correction factor J = 1 - 3/(4*df - 1)
    removes most of that bias. Reported alongside d rather than instead of it, so the two
    can be compared against the published values.
    """
    d = cohens_d(mean_a, std_a, mean_b, std_b, n=n)
    df = 2 * n - 2
    if df <= 1:
        return d
    J = 1.0 - 3.0 / (4.0 * df - 1.0)
    return d * J


def tost_equivalence(mean_a, std_a, mean_b, std_b, margin, n=5, alpha=0.05):
    """Two one-sided tests (TOST) for equivalence within +/- margin.

    Non-significance of a difference is NOT evidence of equivalence — with n = 3-5 a test
    can fail to reject simply for want of power. TOST inverts the burden of proof: the null
    is 'the difference is at least `margin`', so rejecting it is positive evidence that the
    two methods are practically the same.

    Returns (equivalent, p_tost, ci_low, ci_high) where p_tost is the larger of the two
    one-sided p-values and the interval is the (1-2*alpha) CI of the difference a - b.
    Equivalence is declared when p_tost < alpha, i.e. when the whole CI sits inside
    (-margin, +margin).
    """
    se = np.sqrt(std_a**2 / n + std_b**2 / n)
    diff = mean_a - mean_b
    if se == 0:
        return abs(diff) < margin, 0.0 if abs(diff) < margin else 1.0, diff, diff
    df = (std_a**2/n + std_b**2/n)**2 / ((std_a**2/n)**2/(n-1) + (std_b**2/n)**2/(n-1))
    # H01: diff <= -margin   H02: diff >= +margin ; reject both => equivalent
    p_lower = stats.t.sf((diff + margin) / se, df)   # tests diff > -margin
    p_upper = stats.t.cdf((diff - margin) / se, df)  # tests diff < +margin
    p_tost = max(p_lower, p_upper)
    crit = stats.t.ppf(1 - alpha, df)
    return p_tost < alpha, p_tost, diff - crit * se, diff + crit * se


def cohens_d(mean_a, std_a, mean_b, std_b, n=5):
    """Cohen's d effect size with pooled standard deviation.

    Expects sample (ddof=1) standard deviations, matching the CSVs written by
    ExperimentTracker.get_statistics (CODICE-6).
    """
    pooled = np.sqrt(((n-1)*std_a**2 + (n-1)*std_b**2) / (2*n - 2))
    if pooled == 0:
        return 0.0
    return (mean_a - mean_b) / pooled


# ---------------------------------------------------------------------------
# Multiple-comparison correction (STAT-1)
#
# The study runs ~112 t-tests (τ(w) vs 7 competitors × 16 benchmarks). Counting
# raw significances as a scoreboard (e.g. "22 wins / 6 losses") without correction
# over-states the signal: under a global null at α=0.05, ~5.6 of 112 tests are
# significant by chance. The functions below let callers apply ONE correction
# symmetrically to wins AND losses, instead of narrating them asymmetrically.
# ---------------------------------------------------------------------------

def benjamini_hochberg(pvals, q=0.05):
    """Benjamini-Hochberg FDR control.

    Args:
        pvals: 1-D iterable of p-values for the whole comparison family
               (both directions — wins and losses — together).
        q: target false-discovery rate.

    Returns:
        Boolean numpy array; True where H0 is rejected at FDR q.
    """
    p = np.asarray(list(pvals), dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresh = q * (np.arange(1, m + 1) / m)
    below = ranked <= thresh
    if not below.any():
        return np.zeros(m, dtype=bool)
    kmax = np.max(np.where(below)[0])      # largest rank passing the BH line
    cutoff = ranked[kmax]
    return p <= cutoff


def bonferroni(pvals, alpha=0.05):
    """Bonferroni FWER control over a family of m p-values.

    Returns a boolean numpy array; True where p < alpha / m.
    """
    p = np.asarray(list(pvals), dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    return p < (alpha / m)
