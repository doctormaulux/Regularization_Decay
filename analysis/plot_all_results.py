"""
Plot All Results - Universal Plotting Script
Genera figure publication-ready dai file CSV/JSON degli esperimenti standardizzati

Supports:
- Classification tasks (test_acc): MNIST, CIFAR, ViT, BERT
- Regression tasks (test_mse): Sin, Complex Function
- Language modeling (test_ppl): GPT-2

Usage:
    python plot_all_results.py mnist_standardized_results.csv
    python plot_all_results.py cifar_standardized_results.csv
    python plot_all_results.py vit_cifar_standardized_results.csv
    python plot_all_results.py bert_sst2_standardized_results.csv
    python plot_all_results.py gpt2_wikitext_standardized_results.csv
    python plot_all_results.py sin_regression_standardized_results.csv
    python plot_all_results.py complex_regression_standardized_results.csv
"""

import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

# Color scheme
COLORS = {
    'Baseline': '#1f77b4',
    'L1': '#ff7f0e',
    'L2': '#2ca02c',
    'ElasticNet': '#d62728',
    'SCAD': '#9467bd',
    'MCP': '#8c564b',
    'LSP': '#e377c2',
    'τ(w)': '#bcbd22',
    'Tau(w)': '#bcbd22'
}


def load_results(csv_path):
    """Load results from CSV file"""
    df = pd.read_csv(csv_path)

    # Get JSON path (same name, different extension)
    json_path = csv_path.replace('.csv', '.json')

    json_data = None
    if Path(json_path).exists():
        with open(json_path, 'r') as f:
            json_data = json.load(f)

    return df, json_data


def get_metric_info(df):
    """Detect metric type (accuracy, MSE, or perplexity) from columns"""
    if 'test_acc_mean' in df.columns:
        return {
            'metric': 'test_acc',
            'metric_label': 'Test Accuracy',
            'mode': 'max',
            'format': '.4f'
        }
    elif 'test_mse_mean' in df.columns:
        return {
            'metric': 'test_mse',
            'metric_label': 'Test MSE',
            'mode': 'min',
            'format': '.6f'
        }
    elif 'test_ppl_mean' in df.columns:
        return {
            'metric': 'test_ppl',
            'metric_label': 'Test Perplexity',
            'mode': 'min',
            'format': '.2f'
        }
    else:
        raise ValueError("Unknown metric type in CSV")


def plot_performance_comparison(df, metric_info, output_prefix):
    """Bar chart: Performance comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))

    metric = metric_info['metric']
    methods = df['method'].tolist()
    means = df[f'{metric}_mean'].tolist()
    stds = df[f'{metric}_std'].tolist()

    x_pos = np.arange(len(methods))
    colors = [COLORS.get(m, '#999999') for m in methods]

    bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                  color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Highlight best
    if metric_info['mode'] == 'max':
        best_idx = np.argmax(means)
    else:
        best_idx = np.argmin(means)
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)

    ax.set_xlabel('Method', fontweight='bold')
    ax.set_ylabel(metric_info['metric_label'], fontweight='bold')
    ax.set_title(f'Performance Comparison ({metric_info["metric_label"]})',
                 fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        label = f'{mean:{metric_info["format"]}}\n±{std:.1e}'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"[PLOT] Saved {output_prefix}_performance_comparison.png")
    plt.close()


def plot_sparsity_comparison(df, output_prefix):
    """Bar chart: Sparsity comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = df['method'].tolist()
    sparsities = df['sparsity_mean'].tolist()

    x_pos = np.arange(len(methods))
    colors = [COLORS.get(m, '#999999') for m in methods]

    bars = ax.bar(x_pos, sparsities, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Method', fontweight='bold')
    ax.set_ylabel('Sparsity (%)', fontweight='bold')
    ax.set_title('Model Sparsity Comparison', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, sparsity in zip(bars, sparsities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{sparsity:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_sparsity_comparison.png', dpi=300, bbox_inches='tight')
    print(f"[PLOT] Saved {output_prefix}_sparsity_comparison.png")
    plt.close()


def plot_performance_vs_sparsity(df, metric_info, output_prefix):
    """Scatter plot: Performance vs Sparsity (Pareto front)"""
    fig, ax = plt.subplots(figsize=(10, 7))

    metric = metric_info['metric']
    methods = df['method'].tolist()
    performances = df[f'{metric}_mean'].tolist()
    sparsities = df['sparsity_mean'].tolist()

    # Plot points
    for method, perf, sparse in zip(methods, performances, sparsities):
        color = COLORS.get(method, '#999999')
        size = 200 if method in ['τ(w)', 'Tau(w)'] else 100
        marker = 'D' if method in ['τ(w)', 'Tau(w)'] else 'o'

        ax.scatter(sparse, perf, c=color, s=size, alpha=0.7,
                  edgecolors='black', linewidth=1.5, marker=marker, label=method)

    ax.set_xlabel('Sparsity (%)', fontweight='bold')
    ax.set_ylabel(metric_info['metric_label'], fontweight='bold')

    if metric_info['mode'] == 'max':
        title = f'{metric_info["metric_label"]} vs Sparsity\n(Ideal: Top-Left Corner)'
    else:
        title = f'{metric_info["metric_label"]} vs Sparsity\n(Ideal: Bottom-Left Corner)'

    ax.set_title(title, fontweight='bold')
    ax.legend(loc='best', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_performance_vs_sparsity.png', dpi=300, bbox_inches='tight')
    print(f"[PLOT] Saved {output_prefix}_performance_vs_sparsity.png")
    plt.close()


def plot_confidence_intervals(df, metric_info, output_prefix):
    """Plot with confidence intervals"""
    fig, ax = plt.subplots(figsize=(10, 6))

    metric = metric_info['metric']
    methods = df['method'].tolist()
    means = df[f'{metric}_mean'].tolist()
    ci_lowers = df[f'{metric}_ci_lower'].tolist()
    ci_uppers = df[f'{metric}_ci_upper'].tolist()

    x_pos = np.arange(len(methods))
    colors = [COLORS.get(m, '#999999') for m in methods]

    # Plot points with error bars
    for i, (method, mean, ci_low, ci_up, color) in enumerate(zip(methods, means, ci_lowers, ci_uppers, colors)):
        ci_range = [[mean - ci_low], [ci_up - mean]]
        marker = 'D' if method in ['τ(w)', 'Tau(w)'] else 'o'
        markersize = 10 if method in ['τ(w)', 'Tau(w)'] else 8

        ax.errorbar(i, mean, yerr=ci_range, fmt=marker, color=color,
                   markersize=markersize, capsize=5, capthick=2,
                   elinewidth=2, alpha=0.8, label=method)

    ax.set_xlabel('Method', fontweight='bold')
    ax.set_ylabel(metric_info['metric_label'], fontweight='bold')
    ax.set_title(f'{metric_info["metric_label"]} with 95% Confidence Intervals',
                 fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_confidence_intervals.png', dpi=300, bbox_inches='tight')
    print(f"[PLOT] Saved {output_prefix}_confidence_intervals.png")
    plt.close()


def plot_statistical_summary(df, metric_info, output_prefix):
    """Summary table as image"""
    fig, ax = plt.subplots(figsize=(12, len(df)*0.5 + 2))
    ax.axis('tight')
    ax.axis('off')

    metric = metric_info['metric']

    # Prepare table data
    table_data = []
    headers = ['Method', f'{metric_info["metric_label"]} (Mean ± Std)',
               '95% CI', 'Sparsity (%)']

    for _, row in df.iterrows():
        method = row['method']
        mean = row[f'{metric}_mean']
        std = row[f'{metric}_std']
        ci_low = row[f'{metric}_ci_lower']
        ci_up = row[f'{metric}_ci_upper']
        sparsity = row['sparsity_mean']

        table_data.append([
            method,
            f'{mean:{metric_info["format"]}} ± {std:.1e}',
            f'[{ci_low:{metric_info["format"]}}, {ci_up:{metric_info["format"]}}]',
            f'{sparsity:.2f}%'
        ])

    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.15, 0.3, 0.35, 0.2])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    plt.title('Statistical Summary Table', fontweight='bold', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_summary_table.png', dpi=300, bbox_inches='tight')
    print(f"[PLOT] Saved {output_prefix}_summary_table.png")
    plt.close()


def plot_all(csv_path):
    """Generate all plots from CSV file"""
    print("="*80)
    print(f"PLOTTING RESULTS FROM: {csv_path}")
    print("="*80)

    # Load data
    df, json_data = load_results(csv_path)
    metric_info = get_metric_info(df)

    print(f"Detected metric: {metric_info['metric_label']}")
    print(f"Number of methods: {len(df)}")
    methods_str = ', '.join(m.replace('\u03c4', 'Tau') for m in df['method'].tolist())
    print(f"Methods: {methods_str}")

    # Create figures directory if it doesn't exist
    figures_dir = 'figures'
    os.makedirs(figures_dir, exist_ok=True)
    print(f"Output directory: {figures_dir}/")

    # Get output prefix (remove _results.csv) and add figures/ path
    base_prefix = csv_path.replace('_results.csv', '').replace('.csv', '')
    base_prefix = os.path.basename(base_prefix)  # Remove any path components
    output_prefix = os.path.join(figures_dir, base_prefix)

    # Generate all plots
    print("\nGenerating plots...")

    plot_performance_comparison(df, metric_info, output_prefix)
    plot_sparsity_comparison(df, output_prefix)
    plot_performance_vs_sparsity(df, metric_info, output_prefix)
    plot_confidence_intervals(df, metric_info, output_prefix)
    plot_statistical_summary(df, metric_info, output_prefix)

    print("\n" + "="*80)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*80)
    print(f"\nOutput files (in {figures_dir}/):")
    print(f"  - {os.path.basename(output_prefix)}_performance_comparison.png")
    print(f"  - {os.path.basename(output_prefix)}_sparsity_comparison.png")
    print(f"  - {os.path.basename(output_prefix)}_performance_vs_sparsity.png")
    print(f"  - {os.path.basename(output_prefix)}_confidence_intervals.png")
    print(f"  - {os.path.basename(output_prefix)}_summary_table.png")
    print("="*80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_all_results.py <results.csv>")
        print("\nExamples (Classic benchmarks):")
        print("  python plot_all_results.py mnist_standardized_results.csv")
        print("  python plot_all_results.py cifar_standardized_results.csv")
        print("  python plot_all_results.py sin_regression_standardized_results.csv")
        print("  python plot_all_results.py complex_regression_standardized_results.csv")
        print("\nExamples (Transformer benchmarks):")
        print("  python plot_all_results.py vit_cifar_standardized_results.csv")
        print("  python plot_all_results.py bert_sst2_standardized_results.csv")
        print("  python plot_all_results.py gpt2_wikitext_standardized_results.csv")
        sys.exit(1)

    csv_path = sys.argv[1]

    if not Path(csv_path).exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    plot_all(csv_path)


if __name__ == "__main__":
    main()
