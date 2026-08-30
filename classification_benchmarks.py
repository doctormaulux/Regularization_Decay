"""
Image Classification Benchmarks - Config-driven
MLP/CNN classification with τ(w) regularization

Models: MNIST (FC network), CIFAR-10 (CNN)
Task: Image classification with CrossEntropy loss

Usage:
    python classification_benchmarks.py --model mnist
    python classification_benchmarks.py --model cifar
    python classification_benchmarks.py --model all
    python classification_benchmarks.py --list
"""

import argparse
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

from experiment_utils import (
    EarlyStopping, set_seed, evaluate_classification,
    measure_sparsity, run_benchmark,
    get_regularization, apply_tau_if_needed,
    optimizer_weight_decay
)

# ============================
# CONFIGURATION
# ============================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {DEVICE}")

DEFAULTS = dict(
    batch_size=128, lr=1e-3, val_fraction=0.1,
    pso_budget='light',
    seeds=[42, 43, 44, 45, 46],
)

MODELS = {
    'mnist': dict(
        num_epochs=50, patience=10,
        experiment_name='MNIST_Standardized',
        title='MNIST Classification',
        csv='results/mnist_standardized_results.csv',
        json='results/mnist_standardized_results.json',
    ),
    'cifar': dict(
        num_epochs=100, patience=15,
        experiment_name='CIFAR10_Standardized',
        title='CIFAR-10 Classification',
        csv='results/cifar_standardized_results.csv',
        json='results/cifar_standardized_results.json',
    ),
}


# ============================
# MODELS
# ============================

class MNISTNet(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)


class CIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


MODEL_CLASSES = {
    'mnist': MNISTNet,
    'cifar': CIFAR10Net,
}


# ============================
# DATASETS
# ============================

MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

CIFAR_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    )
])

DATASET_INFO = {
    'mnist': dict(
        cls=torchvision.datasets.MNIST,
        transform=MNIST_TRANSFORM,
    ),
    'cifar': dict(
        cls=torchvision.datasets.CIFAR10,
        transform=CIFAR_TRANSFORM,
    ),
}


# ============================
# RUNNER
# ============================

def run_model(key, quiet=False):
    cfg = {**DEFAULTS, **MODELS[key]}
    ds_info = DATASET_INFO[key]

    train_full = ds_info['cls'](
        root="./data", train=True, download=True,
        transform=ds_info['transform'],
    )
    test_set = ds_info['cls'](
        root="./data", train=False, download=True,
        transform=ds_info['transform'],
    )

    def train_model(method_name, hyperparams, seed, verbose=True, prune_targets=None):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        set_seed(seed)

        n_val = int(len(train_full) * cfg['val_fraction'])
        n_train = len(train_full) - n_val
        train_set, val_set = random_split(
            train_full, [n_train, n_val],
            generator=torch.Generator().manual_seed(seed)
        )

        bs = cfg['batch_size']
        train_loader = DataLoader(
            train_set, batch_size=bs, shuffle=True
        )
        val_loader = DataLoader(
            val_set, batch_size=bs, shuffle=False
        )
        test_loader = DataLoader(
            test_set, batch_size=bs, shuffle=False
        )

        model = MODEL_CLASSES[key]().to(DEVICE)
        # AdamW (≡ Adam when wd=0): only 'WD-tuned' gets a PSO-tuned decoupled wd (CODICE-1)
        optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'],
                                weight_decay=optimizer_weight_decay(method_name, hyperparams))
        loss_fn = nn.CrossEntropyLoss()
        early_stopping = EarlyStopping(
            patience=cfg['patience'], mode='max'
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

            val_acc, _ = evaluate_classification(
                model, val_loader, DEVICE
            )
            if early_stopping(val_acc, model, epoch):
                if verbose:
                    print(f"  Early stopping at epoch {epoch + 1}")
                break

        model = early_stopping.restore_best_model(model)
        test_acc, _ = evaluate_classification(
            model, test_loader, DEVICE
        )
        val_acc, _ = evaluate_classification(
            model, val_loader, DEVICE
        )
        # Outcome guard: the reported number must belong to the best epoch.
        early_stopping.check_restored(val_acc, context=f'{key} {method_name} seed={seed}')
        sparsity, total_params, _ = measure_sparsity(model)

        if torch.cuda.is_available():
            model.cpu()
            del model, optimizer
            torch.cuda.empty_cache()

        return {
            'test_acc': test_acc,
            'val_acc': val_acc,
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
        },
        seeds=cfg['seeds'], train_fn=train_model,
        primary_metric='test_acc', metric_mode='max',
        pso_metric='val_acc', pso_mode='max',
        pso_budget=cfg['pso_budget'],
        csv_filename=cfg['csv'],
        json_filename=cfg['json'],
        quiet=quiet,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image Classification Benchmarks'
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
        print("Available classification models:")
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
