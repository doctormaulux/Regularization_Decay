"""
BERT-tiny on SST-2 Sentiment Analysis - Standardized Benchmark
Test tau(w) regularization on language understanding tasks

Architecture:
- BERT-tiny: 2 layers, 128 hidden, 2 heads
- Task: Binary sentiment classification
- Dataset: Stanford Sentiment Treebank v2 (SST-2)
- ~4M parameters

Methodology:
- Early stopping with patience
- Multi-run experiments (5 runs per method)
- PSO hyperparameter tuning for ALL methods
- Statistical analysis (mean, std, CI95%, t-tests)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from experiment_utils import (
    EarlyStopping, run_benchmark,
    get_regularization_transformer, apply_decoupled_decay_transformer,
    optimizer_weight_decay, tau_alpha_for, NO_LOSS_PENALTY, TAU_METHODS,
    evaluate_classification, measure_sparsity, set_seed, magnitude_prune,
)

# Try to import datasets library
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("[WARNING] Hugging Face datasets not available. Will use simplified dataset.")

# ============================
# CONFIGURATION
# ============================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model hyperparameters
VOCAB_SIZE = 10000  # Simplified vocabulary
MAX_SEQ_LENGTH = 128
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_HEADS = 2
INTERMEDIATE_SIZE = 512  # 4x hidden_size
DROPOUT = 0.1

# Training hyperparameters - OPTIMIZED for RTX 5000 (20GB)
NUM_EPOCHS = 50
BATCH_SIZE = 64  # Increased from 32 for faster training
LR = 5e-4  # Standard BERT learning rate
PATIENCE = 10
WARMUP_EPOCHS = 3
# LR-decay horizon (epochs), decoupled from the early-stopping ceiling NUM_EPOCHS.
SCHEDULE_EPOCHS = NUM_EPOCHS

# Experiment parameters
N_RUNS = 5
SEEDS = [42, 123, 456, 789, 1011]

# PSO hyperparameter tuning
PSO_BUDGET = 'light'

# ============================
# DATASET
# ============================

class SimpleTokenizer:
    """Simple word-level tokenizer for SST-2"""

    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[UNK]': 3}
        self.idx2word = {0: '[PAD]', 1: '[CLS]', 2: '[SEP]', 3: '[UNK]'}
        self.next_idx = 4

    def fit(self, texts):
        """Build vocabulary from texts"""
        word_freq = {}
        for text in texts:
            for word in text.lower().split():
                word_freq[word] = word_freq.get(word, 0) + 1

        # Add most frequent words to vocabulary
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, _ in sorted_words[:self.vocab_size - 4]:
            if word not in self.word2idx:
                self.word2idx[word] = self.next_idx
                self.idx2word[self.next_idx] = word
                self.next_idx += 1

    def encode(self, text, max_length=128):
        """Convert text to token indices"""
        tokens = [self.word2idx['[CLS]']]
        for word in text.lower().split():
            tokens.append(self.word2idx.get(word, self.word2idx['[UNK]']))
        tokens.append(self.word2idx['[SEP]'])

        # Pad or truncate
        if len(tokens) < max_length:
            tokens += [self.word2idx['[PAD]']] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]

        return tokens


class SST2Dataset(Dataset):
    """SST-2 sentiment analysis dataset"""

    def __init__(self, split='train', tokenizer=None, max_length=128):
        self.max_length = max_length
        self.tokenizer = tokenizer

        if HF_AVAILABLE:
            # Load from Hugging Face
            dataset = load_dataset('nyu-mll/glue', 'sst2', split=split)
            self.texts = dataset['sentence']
            self.labels = dataset['label']
        else:
            # Simple fallback dataset for testing
            print(f"[INFO] Using simplified {split} dataset")
            if split == 'train':
                self.texts = [
                    "this movie is great", "i love this film", "wonderful acting",
                    "terrible movie", "i hate this film", "awful performance",
                    "amazing story", "brilliant direction", "fantastic cinematography",
                    "boring plot", "waste of time", "disappointing ending"
                ] * 100
                self.labels = [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0] * 100
            elif split == 'validation':
                self.texts = [
                    "excellent film", "poor acting", "incredible movie", "bad script"
                ] * 50
                self.labels = [1, 0, 1, 0] * 50
            else:  # test
                self.texts = [
                    "superb direction", "mediocre plot", "outstanding performance"
                ] * 50
                self.labels = [1, 0, 1] * 50

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        if self.tokenizer:
            tokens = self.tokenizer.encode(text, self.max_length)
        else:
            # Fallback: simple character encoding
            tokens = [ord(c) % 256 for c in text[:self.max_length]]
            tokens += [0] * (self.max_length - len(tokens))

        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def get_dataloaders(tokenizer, batch_size=32, seed=42):
    """Create train, validation, and test dataloaders

    SST-2 test labels are not public, so we use:
    - Training set split: 90% train, 10% validation
    - Official validation set as test set
    """
    from torch.utils.data import random_split

    # Load full training set and official validation (used as test)
    full_train_dataset = SST2Dataset('train', tokenizer, MAX_SEQ_LENGTH)
    test_dataset = SST2Dataset('validation', tokenizer, MAX_SEQ_LENGTH)

    # Split training set into train/val (90/10)
    n_total = len(full_train_dataset)
    n_val = int(n_total * 0.1)
    n_train = n_total - n_val

    train_dataset, val_dataset = random_split(
        full_train_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ============================
# MODEL ARCHITECTURE
# ============================

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention"""

    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Linear projections
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        return self.out_linear(out)


class TransformerBlock(nn.Module):
    """BERT transformer block"""

    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.1):
        super().__init__()

        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out = self.attention(x, mask)
        x = self.norm1(x + attn_out)

        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class BERTTiny(nn.Module):
    """BERT-tiny for sequence classification"""

    def __init__(self, vocab_size, hidden_size, num_layers, num_heads,
                 intermediate_size, max_seq_length, num_classes=2, dropout=0.1):
        super().__init__()

        self.hidden_size = hidden_size

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape

        # Create position ids
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)
        x = self.norm(token_emb + pos_emb)
        x = self.dropout(x)

        # Create attention mask (0 for padding tokens)
        mask = (input_ids != 0).unsqueeze(1).unsqueeze(2)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Use [CLS] token (first token) for classification
        cls_output = x[:, 0, :]
        logits = self.classifier(cls_output)

        return logits



# ============================
# TRAINING
# ============================

def train_model(method, hyperparams, seed, verbose=True, prune_targets=None):
    """Train model with specified regularization method"""
    set_seed(seed)

    lambda_val = hyperparams.get('lambda_val', 0.0)
    extra_params = {k: v for k, v in hyperparams.items()
                    if k not in ('lambda_val', 'decay_strength', 'tau0', 'tau_alpha', 'delta', 'wd', 'rho')}

    # Build tokenizer
    tokenizer = SimpleTokenizer(VOCAB_SIZE)

    # Fit tokenizer on training data
    if HF_AVAILABLE:
        train_texts = load_dataset('nyu-mll/glue', 'sst2', split='train')['sentence']
    else:
        train_dataset = SST2Dataset('train')
        train_texts = train_dataset.texts
    tokenizer.fit(train_texts)

    # Get dataloaders (pass seed for reproducible train/val split)
    train_loader, val_loader, test_loader = get_dataloaders(tokenizer, BATCH_SIZE, seed)

    # Create model
    model = BERTTiny(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        max_seq_length=MAX_SEQ_LENGTH,
        num_classes=2,
        dropout=DROPOUT
    ).to(DEVICE)

    # Optimizer with warmup. Option-1 protocol: only 'WD-tuned' carries a PSO-tuned
    # decoupled weight decay; Baseline is truly unregularized (previously wd=0.01 for
    # ALL methods, stacking decay under τ — CODICE-1).
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR,
        weight_decay=optimizer_weight_decay(method, hyperparams))

    # Learning rate scheduler with warmup (horizon = SCHEDULE_EPOCHS, not the ceiling)
    total_steps = len(train_loader) * SCHEDULE_EPOCHS
    warmup_steps = len(train_loader) * WARMUP_EPOCHS

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / (total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=0.001, mode='max')

    if verbose:
        param_str = ', '.join([f'{k}={v:.1e}' if isinstance(v, float) else f'{k}={v}' for k, v in hyperparams.items()])
        print(f"[TRAIN] Method={method}, {param_str}, seed={seed}")

    for epoch in range(NUM_EPOCHS):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)

            if method not in NO_LOSS_PENALTY:
                loss = loss + get_regularization_transformer(
                    model, method, lambda_val, DEVICE, extra_params=extra_params)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Post-optimizer decoupled decay: tau family + robust-decay competitors.
            apply_decoupled_decay_transformer(model, method, hyperparams)

        val_acc, _ = evaluate_classification(model, val_loader, DEVICE)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{NUM_EPOCHS}: val_acc={val_acc:.4f}")

        if early_stopping(val_acc, model, epoch):
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    model = early_stopping.restore_best_model(model)

    test_acc, _ = evaluate_classification(model, test_loader, DEVICE)
    val_acc, _ = evaluate_classification(model, val_loader, DEVICE)
    # Outcome guard: the reported number must belong to the best epoch.
    early_stopping.check_restored(val_acc, context=f'bert_sst2 {method} seed={seed}')
    sparsity, total_params, _ = measure_sparsity(model)

    result = {
        'test_acc': test_acc,
        'val_acc': val_acc,
        'sparsity': sparsity * 100,
        'convergence_epoch': early_stopping.best_epoch + 1,
        'total_params': total_params
    }

    # Optional post-training magnitude pruning sweep
    if prune_targets:
        original_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for tgt in prune_targets:
            model.load_state_dict(original_state)
            info = magnitude_prune(model, sparsity_target=tgt)
            tag = int(round(tgt * 100))
            acc_pruned, _ = evaluate_classification(model, test_loader, DEVICE)
            result[f'test_acc@{tag}'] = acc_pruned
            result[f'sparsity@{tag}'] = info['achieved_sparsity'] * 100
        del original_state

    return result


# ============================
# MAIN EXPERIMENT
# ============================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')
    parser.add_argument('--prune-pso', action='store_true',
                        help='Full PSO on Baseline + tau(w), then multi-seed eval '
                             'with magnitude-pruning sweep at 25/50/75%%. Saves to '
                             '*_prune_pso.csv to keep the main results untouched.')
    parser.add_argument('--prune-targets', type=str, default=None,
                        help='Comma-separated sparsity targets in (0,1) for the sweep.')
    args = parser.parse_args()

    methods_override = None
    csv_filename = 'bert_sst2_standardized_results.csv'
    json_filename = 'bert_sst2_standardized_results.json'
    experiment_name = 'BERT_SST2_Standardized'

    prune_targets = None
    if args.prune_targets:
        prune_targets = [float(x) for x in args.prune_targets.split(',') if x.strip()]

    if args.prune_pso:
        methods_override = ['Baseline', 'τ(w)']
        if prune_targets is None:
            prune_targets = [0.25, 0.5, 0.75]
        csv_filename = 'bert_sst2_standardized_results_prune_pso.csv'
        json_filename = 'bert_sst2_standardized_results_prune_pso.json'
        experiment_name = experiment_name + '_PrunePSO'

    run_benchmark(
        experiment_name=experiment_name,
        benchmark_title='BERT-tiny on SST-2',
        model_name=f'BERT-tiny ({NUM_LAYERS}L, {HIDDEN_SIZE}H, {NUM_HEADS}heads)',
        device=DEVICE,
        config={
            'Epochs': f'{NUM_EPOCHS} (early stopping, patience={PATIENCE})',
            'Batch size': BATCH_SIZE,
            'Learning rate': LR,
        },
        seeds=SEEDS, train_fn=train_model,
        primary_metric='test_acc', metric_mode='max',
        pso_metric='val_acc', pso_mode='max',
        pso_budget=PSO_BUDGET,
        csv_filename=csv_filename,
        json_filename=json_filename,
        quiet=args.quiet,
        methods=methods_override,
        prune_targets=prune_targets,
    )
