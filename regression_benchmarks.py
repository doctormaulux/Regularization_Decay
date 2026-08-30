"""
Regression Benchmarks - Config-driven
MLP regression with τ(w) regularization

Models: Sin (1D), Friedman #1 (10D)
Task: Function approximation with MSE loss

Usage:
    python regression_benchmarks.py --model sin
    python regression_benchmarks.py --model complex
    python regression_benchmarks.py --model all
    python regression_benchmarks.py --list
"""

import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

from experiment_utils import (
    EarlyStopping, set_seed, evaluate_model, measure_sparsity,
    run_benchmark, get_regularization, apply_tau_if_needed,
    optimizer_weight_decay
)

# ============================
# CONFIGURATION
# ============================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {DEVICE}")

DEFAULTS = dict(
    num_epochs=200, batch_size=64, lr=1e-3,
    val_fraction=0.2, patience=20,
    pso_budget='light',
    seeds=[42, 43, 44, 45, 46],
)

MODELS = {
    'sin': dict(
        input_dim=1, n_samples=1024, noise_std=0.05,
        experiment_name='SinRegression_Standardized',
        title='Sin Regression',
        csv='results/sin_regression_standardized_results.csv',
        json='results/sin_regression_standardized_results.json',
    ),
    'complex': dict(
        input_dim=10, n_samples=2048, noise_std=1.0,
        experiment_name='ComplexRegression_Standardized',
        title='Friedman #1 Regression (10D)',
        csv='results/complex_regression_standardized_results.csv',
        json='results/complex_regression_standardized_results.json',
    ),
}


# ============================
# MODEL
# ============================

class RegressionMLP(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


# ============================
# DATASET GENERATORS
# ============================

def make_sine_dataset(n_samples, noise_std, seed, device):
    torch.manual_seed(seed)
    x = torch.linspace(-3, 3, n_samples).unsqueeze(1)
    y = torch.sin(x)
    if noise_std > 0:
        y = y + noise_std * torch.randn_like(y)
    return x.to(device), y.to(device)


def make_friedman1_dataset(n_samples, noise_std, seed, device):
    torch.manual_seed(seed)
    x = torch.rand(n_samples, 10)
    y = (10.0 * torch.sin(np.pi * x[:, 0] * x[:, 1])
         + 20.0 * (x[:, 2] - 0.5) ** 2
         + 10.0 * x[:, 3]
         + 5.0 * x[:, 4])
    y = y.unsqueeze(1)
    if noise_std > 0:
        y = y + noise_std * torch.randn_like(y)
    return x.to(device), y.to(device)


DATASET_FNS = {
    'sin': make_sine_dataset,
    'complex': make_friedman1_dataset,
}


# ============================
# RUNNER
# ============================

def run_model(key, quiet=False):
    cfg = {**DEFAULTS, **MODELS[key]}
    dataset_fn = DATASET_FNS[key]

    def train_model(method_name, hyperparams, seed, verbose=True, prune_targets=None):
        set_seed(seed)

        x_all, y_all = dataset_fn(
            cfg['n_samples'], cfg['noise_std'], seed, DEVICE
        )

        n_val = int(len(x_all) * cfg['val_fraction'])
        n_train = len(x_all) - n_val
        dataset = TensorDataset(x_all, y_all)
        train_set, val_set = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(seed)
        )

        bs = cfg['batch_size']
        train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=bs, shuffle=False)

        x_test, y_test = dataset_fn(
            cfg['n_samples'], cfg['noise_std'], seed + 1000, DEVICE
        )
        test_loader = DataLoader(
            TensorDataset(x_test, y_test), batch_size=bs, shuffle=False
        )

        model = RegressionMLP(input_dim=cfg['input_dim']).to(DEVICE)
        # AdamW (≡ Adam when wd=0): only 'WD-tuned' gets a PSO-tuned decoupled wd (CODICE-1)
        optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'],
                                weight_decay=optimizer_weight_decay(method_name, hyperparams))
        loss_fn = nn.MSELoss()
        early_stopping = EarlyStopping(
            patience=cfg['patience'], mode='min'
        )

        for epoch in range(cfg['num_epochs']):
            model.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss = loss + get_regularization(model, method_name, hyperparams)

                loss.backward()
                optimizer.step()
                apply_tau_if_needed(model, method_name, hyperparams)

            val_mse = evaluate_model(model, val_loader, loss_fn, DEVICE)
            if early_stopping(val_mse, model, epoch):
                if verbose:
                    print(f"  Early stopping at epoch {epoch + 1}")
                break

        model = early_stopping.restore_best_model(model)
        test_mse = evaluate_model(model, test_loader, loss_fn, DEVICE)
        val_mse = evaluate_model(model, val_loader, loss_fn, DEVICE)
        # Outcome guard: the reported number must belong to the best epoch.
        early_stopping.check_restored(val_mse, context=f'{key} {method_name} seed={seed}')
        sparsity, total_params, _ = measure_sparsity(model)

        return {
            'test_mse': test_mse,
            'val_mse': val_mse,
            'sparsity': sparsity * 100,
            'convergence_epoch': early_stopping.best_epoch + 1,
            'total_params': total_params,
        }

    run_benchmark(
        experiment_name=cfg['experiment_name'],
        benchmark_title=cfg['title'],
        model_name=key, device=DEVICE,
        config={
            'Epochs': f"{cfg['num_epochs']} (with early stopping,"
                      f" patience={cfg['patience']})",
            'Batch size': cfg['batch_size'],
            'Learning rate': cfg['lr'],
            'Samples': cfg['n_samples'],
            'Noise std': cfg['noise_std'],
        },
        seeds=cfg['seeds'], train_fn=train_model,
        primary_metric='test_mse', metric_mode='min',
        pso_metric='val_mse', pso_mode='min',
        pso_budget=cfg['pso_budget'],
        csv_filename=cfg['csv'],
        json_filename=cfg['json'],
        quiet=quiet,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Regression Benchmarks'
    )
    parser.add_argument(
        '--model', choices=list(MODELS) + ['all'],
        help='Model to run (or "all")'
    )
    parser.add_argument(
        '--list', action='store_true',
        help='List available models'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Minimal output (model name, time estimate, elapsed)'
    )
    args = parser.parse_args()

    if args.list:
        print("Available regression models:")
        for k, v in MODELS.items():
            print(f"  {k:12s}  {v['title']}")
        sys.exit(0)

    if not args.model:
        parser.print_help()
        sys.exit(1)

    if args.model == 'all':
        for key in MODELS:
            run_model(key, quiet=args.quiet)
    else:
        run_model(args.model, quiet=args.quiet)
