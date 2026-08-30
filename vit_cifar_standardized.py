"""
Vision Transformer (ViT) on CIFAR-10 - Standardized Benchmark
Custom ViT-mini with PSO hyperparameter tuning and statistical analysis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

from experiment_utils import (
    EarlyStopping, set_seed,
    evaluate_classification, measure_sparsity,
    run_benchmark, get_regularization, apply_tau_if_needed,
    optimizer_weight_decay
)

# ============================
# CONFIGURATION
# ============================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

NUM_EPOCHS = 150
BATCH_SIZE = 128
LR = 3e-4
VAL_FRACTION = 0.1
PATIENCE = 25
N_RUNS = 5

# ViT Architecture
PATCH_SIZE = 4    # 32x32 / 4 = 8x8 = 64 patches
EMBED_DIM = 128
DEPTH = 6
NUM_HEADS = 4
MLP_RATIO = 2
DROPOUT = 0.1

PSO_BUDGET = 'light'
SEEDS = [42 + i for i in range(N_RUNS)]

# ============================
# DATASET
# ============================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

train_full = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
test_set = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

# ============================
# VISION TRANSFORMER MODEL
# ============================

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=128):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.dropout(attn.softmax(dim=-1))

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.dropout(self.proj(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=2, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3,
                 num_classes=10, embed_dim=128, depth=6, num_heads=4,
                 mlp_ratio=2, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.head(x[:, 0])


# ============================
# TRAINING FUNCTION
# ============================

def train_model(method_name, hyperparams, seed, verbose=False, prune_targets=None, **_kwargs):
    # prune_targets is forwarded by run_benchmark for the post-training prune sweep;
    # ViT custom is not part of the prune-PSO suite, so the argument is ignored.
    set_seed(seed)

    n_val = int(len(train_full) * VAL_FRACTION)
    n_train = len(train_full) - n_val
    train_set, val_set = random_split(
        train_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    model = VisionTransformer(
        img_size=32, patch_size=PATCH_SIZE, in_channels=3, num_classes=10,
        embed_dim=EMBED_DIM, depth=DEPTH, num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO, dropout=DROPOUT
    ).to(DEVICE)

    # Option-1 protocol: only 'WD-tuned' carries a PSO-tuned decoupled weight decay
    # (Baseline stays unregularized, as before — CODICE-1).
    optimizer = optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=optimizer_weight_decay(method_name, hyperparams))
    loss_fn = nn.CrossEntropyLoss()

    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    early_stopping = EarlyStopping(patience=PATIENCE, mode='max')

    for epoch in range(NUM_EPOCHS):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            loss = loss_fn(model(inputs), targets)
            loss = loss + get_regularization(model, method_name, hyperparams)

            loss.backward()
            optimizer.step()
            apply_tau_if_needed(model, method_name, hyperparams)

        scheduler.step()

        val_acc, _ = evaluate_classification(model, val_loader, DEVICE)
        if early_stopping(val_acc, model, epoch):
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    model = early_stopping.restore_best_model(model)

    test_acc, _ = evaluate_classification(model, test_loader, DEVICE)
    val_acc, _ = evaluate_classification(model, val_loader, DEVICE)
    # Outcome guard: the reported number must belong to the best epoch.
    early_stopping.check_restored(val_acc, context=f'vit_cifar {method_name} seed={seed}')
    sparsity, total_params, _ = measure_sparsity(model)

    return {
        'test_acc': test_acc,
        'val_acc': val_acc,
        'sparsity': sparsity * 100,
        'convergence_epoch': early_stopping.best_epoch + 1,
        'total_params': total_params
    }


# ============================
# MAIN EXPERIMENT
# ============================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')
    parser.add_argument('--methods', type=str, default=None,
                        help="Comma-separated subset of the 10-method roster (output files and "
                             "journal unchanged, so the roster can be split across pods and "
                             "assembled afterwards from the concatenated journals).")
    args = parser.parse_args()

    methods_override = None
    if args.methods:
        from experiment_utils import DEFAULT_METHODS
        _alias = {'Tau(w)': 'τ(w)', 'tau(w)': 'τ(w)'}
        methods_override = [_alias.get(m.strip(), m.strip()) for m in args.methods.split(',')
                            if m.strip()]
        _bad = [m for m in methods_override if m not in DEFAULT_METHODS]
        if _bad:
            parser.error(f"--methods: unknown {_bad}; roster: {DEFAULT_METHODS}")
        print(f"[CONFIG] methods={methods_override}", flush=True)

    run_benchmark(
        experiment_name='ViT_CIFAR10_Standardized',
        benchmark_title='ViT-mini on CIFAR-10',
        model_name='ViT-mini (custom)', device=DEVICE,
        config={
            'Architecture': f'patch={PATCH_SIZE}, embed={EMBED_DIM}, '
                            f'depth={DEPTH}, heads={NUM_HEADS}',
            'Epochs': f'{NUM_EPOCHS} (early stopping, patience={PATIENCE})',
            'Batch size': BATCH_SIZE,
            'Learning rate': f'{LR} (with warmup)',
        },
        seeds=SEEDS, train_fn=train_model,
        primary_metric='test_acc', metric_mode='max',
        pso_metric='val_acc', pso_mode='max',
        pso_budget=PSO_BUDGET,
        csv_filename='vit_cifar_standardized_results.csv',
        json_filename='vit_cifar_standardized_results.json',
        methods=methods_override,
        quiet=args.quiet,
    )
