"""
GPT-2 Small on WikiText-2 - Standardized Benchmark
Test tau(w) regularization on language modeling tasks

Architecture:
- GPT-2 Small: 6 layers, 256 hidden, 4 heads
- Task: Causal language modeling
- Dataset: WikiText-2
- ~7.4M parameters at the reference 'small' scale (see SCALES for the sweep)

Methodology:
- Early stopping with patience
- Multi-run experiments (5 runs per method)
- PSO hyperparameter tuning for ALL methods (multi-parameter)
- Statistical analysis (mean, std, CI95%, t-tests)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from experiment_utils import (
    EarlyStopping, run_benchmark,
    get_regularization_transformer, apply_decoupled_decay_transformer,
    optimizer_weight_decay, adamw_param_groups, tau_alpha_for, NO_LOSS_PENALTY, TAU_METHODS,
    measure_sparsity, set_seed, magnitude_prune, CORE_METHODS, ROBUST_METHODS,
    ROBUST_DECAY_METHODS, TWO_BY_TWO_METHODS, SEARCH_SPACES, DEFAULT_METHODS,
    weight_magnitude_stats,
)
import json as _json
import os as _os

# Mechanism-analysis instrumentation (Workstream 2). Off by default; enabled via
# --instrument. When on, each training run dumps its per-epoch trajectory
# (train/val perplexity + weight-magnitude stats) to INSTRUMENT_DIR so the
# overfitting-delay and weight-distribution-evolution figures can be produced
# WITHOUT re-running (the pod logs only kept [START]/[DONE] markers, not curves).
INSTRUMENT = False
INSTRUMENT_DIR = 'results/instrumentation'
SCALE_TAG = 'small'

# Fraction of the TRAINING split used (1.0 = all of it). Set by --data-fraction for the
# factorial arm that varies data quantity at fixed model size, so that "over-capacity
# relative to the corpus" can be identified rather than merely suggested by the scale
# sweep (which varies model size at fixed data — the other axis).
DATA_FRACTION = 1.0

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
MAX_SEQ_LENGTH = 256
HIDDEN_SIZE = 256
NUM_LAYERS = 6
NUM_HEADS = 4
INTERMEDIATE_SIZE = 1024  # 4x hidden_size
DROPOUT = 0.1

# Training hyperparameters - OPTIMIZED for RTX 5000 (20GB)
NUM_EPOCHS = 30
BATCH_SIZE = 32  # Increased from 16 for faster training
LR = 3e-4  # Standard GPT-2 learning rate
PATIENCE = 8
WARMUP_EPOCHS = 2
# Horizon of the linear LR decay, in epochs. Decoupled from NUM_EPOCHS (the early-
# stopping ceiling) so that raising the ceiling never stretches the annealing schedule.
# Set per scale in main(); see --schedule-epochs.
SCHEDULE_EPOCHS = NUM_EPOCHS

# Experiment parameters
N_RUNS = 5
# Seeds 6-10 appended (REVIEWER-9: n=3/5 is too few for the primary comparisons, and
# leaves Cohen's d hostage to unstable variance estimates). APPENDED, never reordered,
# so SEEDS[:3] and SEEDS[:5] still name exactly the runs behind the published numbers.
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
# Immutable master list. SEEDS gets truncated per scale further down; --seeds must
# select from the full list, never from an already-truncated one.
ALL_SEEDS = list(SEEDS)

# PSO budget for hyperparameter tuning
PSO_BUDGET = 'standard'

# From-scratch AR-LM scale presets. The 'small' preset is the reference config
# already integrated in the paper (commit 8a9ee4e); 'tiny' and 'medium' add extra
# from-scratch scales so the tau(w) advantage can be shown to *generalise across
# model size* rather than being a single-model (GPT-2 small) artefact — the core
# evidence for the "from-scratch AR-LM mechanism" claim. (hidden, layers, heads,
# intermediate) -> TRAINABLE parameter count with VOCAB_SIZE=10000, MAX_SEQ_LENGTH=256
# and tied input/output embeddings, exactly as measure_sparsity() reports it in the
# 'total_params' field of every results file (tied embeddings counted once).
SCALES = {
    'tiny':   dict(hidden=128, layers=4, heads=4, intermediate=512),   # 2,106,112  (~2.1M)
    'small':  dict(hidden=256, layers=6, heads=4, intermediate=1024),  # 7,364,608  (~7.4M, reference)
    'medium': dict(hidden=384, layers=8, heads=6, intermediate=1536),  # 18,134,784 (~18M)
    'large':  dict(hidden=640, layers=12, heads=10, intermediate=2560),  # 65,647,360 (~66M)
}

# ============================
# DATASET
# ============================

class SimpleTokenizer:
    """Simple word-level tokenizer for WikiText"""

    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word2idx = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<BOS>', 2: '<EOS>', 3: '<UNK>'}
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

    def encode(self, text, max_length=256):
        """Convert text to token indices"""
        tokens = [self.word2idx['<BOS>']]
        for word in text.lower().split():
            tokens.append(self.word2idx.get(word, self.word2idx['<UNK>']))
        tokens.append(self.word2idx['<EOS>'])

        # Truncate if too long
        if len(tokens) > max_length:
            tokens = tokens[:max_length]

        return tokens


class WikiTextDataset(Dataset):
    """WikiText-2 language modeling dataset"""

    def __init__(self, split='train', tokenizer=None, max_length=256):
        self.max_length = max_length
        self.tokenizer = tokenizer

        if HF_AVAILABLE:
            # Load from Hugging Face
            dataset = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split=split)
            self.texts = [text for text in dataset['text'] if len(text.strip()) > 0]
        else:
            # Simple fallback dataset for testing
            print(f"[INFO] Using simplified {split} dataset")
            if split == 'train':
                self.texts = [
                    "The quick brown fox jumps over the lazy dog.",
                    "Machine learning is a subset of artificial intelligence.",
                    "Natural language processing enables computers to understand human language.",
                    "Deep learning uses neural networks with multiple layers.",
                    "Transformers have revolutionized natural language understanding."
                ] * 200
            elif split == 'validation':
                self.texts = [
                    "Neural networks learn patterns from data.",
                    "Language models predict the next word in a sequence."
                ] * 100
            else:  # test
                self.texts = [
                    "Artificial intelligence continues to advance rapidly."
                ] * 100

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        if self.tokenizer:
            tokens = self.tokenizer.encode(text, self.max_length)
        else:
            # Fallback: simple character encoding
            tokens = [ord(c) % 256 for c in text[:self.max_length]]

        # Pad to max_length
        tokens += [0] * (self.max_length - len(tokens))

        # For language modeling: input = tokens[:-1], target = tokens[1:]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)

        return input_ids, target_ids


def get_dataloaders(tokenizer, batch_size=16):
    """Create train, validation, and test dataloaders.

    When DATA_FRACTION < 1 the TRAINING set is subsampled (validation and test are
    always kept whole, so the metric is comparable across fractions). The subset is
    drawn with an RNG seeded on the fraction alone — deliberately NOT on the run seed —
    so every method and every seed at a given fraction trains on exactly the same
    documents. Otherwise the comparison between methods would carry data-sampling noise
    on top of seed noise, and the runs would no longer be paired.

    Note that the tokenizer is fitted on the FULL training split by the caller, before
    this function is reached. That is intentional: holding the vocabulary fixed keeps
    the experiment about how much data the model trains on, rather than confounding it
    with how much of the vocabulary the model has ever seen.
    """
    train_dataset = WikiTextDataset('train', tokenizer, MAX_SEQ_LENGTH)
    val_dataset = WikiTextDataset('validation', tokenizer, MAX_SEQ_LENGTH)
    test_dataset = WikiTextDataset('test', tokenizer, MAX_SEQ_LENGTH)

    if DATA_FRACTION < 1.0:
        n_all = len(train_dataset.texts)
        n_keep = max(1, int(round(n_all * DATA_FRACTION)))
        rng = np.random.default_rng(int(round(DATA_FRACTION * 10000)))
        keep = rng.permutation(n_all)[:n_keep]
        train_dataset.texts = [train_dataset.texts[i] for i in sorted(keep)]
        print(f"[DATA] fraction={DATA_FRACTION:g}: training on {n_keep:,}/{n_all:,} "
              f"documents (val/test kept whole)", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ============================
# MODEL ARCHITECTURE
# ============================

class CausalSelfAttention(nn.Module):
    """Causal (masked) self-attention for GPT"""

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

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Linear projections
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        # Causal mask (prevent attending to future positions)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        return self.out_linear(out)


class TransformerBlock(nn.Module):
    """GPT transformer block"""

    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.1):
        super().__init__()

        self.attention = CausalSelfAttention(hidden_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # Self-attention with residual (pre-norm)
        x = x + self.attention(self.norm1(x))

        # Feed-forward with residual (pre-norm)
        x = x + self.ffn(self.norm2(x))

        return x


class GPT2Small(nn.Module):
    """GPT-2 Small for language modeling"""

    def __init__(self, vocab_size, hidden_size, num_layers, num_heads,
                 intermediate_size, max_seq_length, dropout=0.1):
        super().__init__()

        self.hidden_size = hidden_size

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(hidden_size)

        # Language modeling head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_embedding.weight

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
        x = self.dropout(token_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.norm(x)

        # Language modeling head
        logits = self.lm_head(x)

        return logits



# ============================
# EVALUATION
# ============================

def evaluate_lm(model, data_loader, device):
    """Evaluate language model on perplexity"""
    model.eval()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            # Reshape for cross-entropy
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    perplexity = math.exp(min(avg_loss, 20))  # Cap to avoid overflow

    return perplexity, avg_loss


# ============================
# TRAINING
# ============================

def train_model(method, hyperparams, seed, verbose=True, prune_targets=None, phase='eval'):
    """Train model with specified regularization method.

    `phase` is 'pso' for hyperparameter-search evaluations and 'eval' for the final
    per-seed runs (passed by run_benchmark). It only affects where the instrumentation
    trajectory is written, so tuning probes can never masquerade as final runs.
    """
    set_seed(seed)

    # Extract hyperparameters
    lambda_val = hyperparams.get('lambda_val', 0.0)
    extra_params = {k: v for k, v in hyperparams.items() if k not in ('lambda_val', 'decay_strength', 'tau0', 'tau_alpha', 'delta', 'wd', 'rho')}

    # Build tokenizer
    tokenizer = SimpleTokenizer(VOCAB_SIZE)

    # Fit tokenizer on training data
    if HF_AVAILABLE:
        train_texts = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train')['text']
        train_texts = [t for t in train_texts if len(t.strip()) > 0]
    else:
        train_dataset = WikiTextDataset('train')
        train_texts = train_dataset.texts
    tokenizer.fit(train_texts)

    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(tokenizer, BATCH_SIZE)

    # Create model
    model = GPT2Small(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        max_seq_length=MAX_SEQ_LENGTH,
        dropout=DROPOUT
    ).to(DEVICE)

    # Optimizer with warmup. Option-1 protocol: only 'WD-tuned' carries a PSO-tuned
    # decoupled weight decay; Baseline is truly unregularized (previously wd=0.01 was
    # applied to ALL methods here, stacking decay under τ — CODICE-1).
    optimizer = torch.optim.AdamW(adamw_param_groups(model, method, hyperparams), lr=LR)

    # Learning rate scheduler with warmup. The decay horizon is SCHEDULE_EPOCHS, NOT the
    # early-stopping ceiling NUM_EPOCHS: the two are decoupled on purpose.
    total_steps = len(train_loader) * SCHEDULE_EPOCHS
    warmup_steps = len(train_loader) * WARMUP_EPOCHS

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / (total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=0.01, mode='min')

    if verbose:
        params_str = ', '.join(f"{k}={v:.1e}" if isinstance(v, float) else f"{k}={v}" for k, v in hyperparams.items())
        print(f"[TRAIN] Method={method}, {params_str}, seed={seed}")

    traj = []  # per-epoch mechanism trajectory (only populated when INSTRUMENT)
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            loss = criterion(model(inputs).view(-1, VOCAB_SIZE), targets.view(-1))

            if method not in NO_LOSS_PENALTY:
                loss = loss + get_regularization_transformer(
                    model, method, lambda_val, DEVICE, extra_params=extra_params)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Post-optimizer decoupled decay: tau family, robust-decay competitors and
            # the AdamW-scope 2x2 cell. lr_scale carries the schedule to the only method
            # that is defined to follow it (Tau(AdamW-scope)); every other decay method
            # runs at a constant per-step rate and ignores it.
            apply_decoupled_decay_transformer(
                model, method, hyperparams,
                lr_scale=optimizer.param_groups[0]['lr'] / LR)

            train_loss += loss.item()

        val_ppl, _ = evaluate_lm(model, val_loader, DEVICE)
        train_ppl = math.exp(min(train_loss / len(train_loader), 20))

        if INSTRUMENT:
            traj.append({
                'epoch': epoch + 1,
                'train_ppl': train_ppl,
                'val_ppl': val_ppl,
                **weight_magnitude_stats(model, transformer=True),
            })

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{NUM_EPOCHS}: train_ppl={train_ppl:.2f}, val_ppl={val_ppl:.2f}")

        if early_stopping(val_ppl, model, epoch):
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    model = early_stopping.restore_best_model(model)

    test_ppl, test_loss = evaluate_lm(model, test_loader, DEVICE)
    val_ppl, _ = evaluate_lm(model, val_loader, DEVICE)
    # Outcome guard: the number we are about to report must belong to the best epoch.
    early_stopping.check_restored(
        val_ppl, context=f'gpt2_{SCALE_TAG} {method} seed={seed} phase={phase}')
    sparsity, total_params, _ = measure_sparsity(model)

    result = {
        'test_ppl': test_ppl,
        'val_ppl': val_ppl,
        'test_loss': test_loss,
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
            ppl_pruned, loss_pruned = evaluate_lm(model, test_loader, DEVICE)
            result[f'test_ppl@{tag}'] = ppl_pruned
            result[f'test_loss@{tag}'] = loss_pruned
            result[f'sparsity@{tag}'] = info['achieved_sparsity'] * 100
        del original_state

    # Dump per-epoch mechanism trajectory (Workstream 2) for offline figures.
    # Final runs go to INSTRUMENT_DIR under a (scale, method, seed) name; PSO probes go
    # to INSTRUMENT_DIR/pso/ with a hyperparameter hash in the name, so a probe can never
    # overwrite (or be mistaken for) a final run.
    if INSTRUMENT and traj:
        safe_method = method.replace('/', '').replace('(', '').replace(')', '').replace('=', '')
        if phase == 'pso':
            import hashlib as _hashlib
            _h = _hashlib.sha1(_json.dumps(hyperparams, sort_keys=True, default=float)
                               .encode('utf-8')).hexdigest()[:8]
            out_dir = _os.path.join(INSTRUMENT_DIR, 'pso')
            fn = _os.path.join(out_dir, f'gpt2_{SCALE_TAG}_{safe_method}_seed{seed}_pso{_h}.json')
        else:
            out_dir = INSTRUMENT_DIR
            fn = _os.path.join(out_dir, f'gpt2_{SCALE_TAG}_{safe_method}_seed{seed}.json')
        _os.makedirs(out_dir, exist_ok=True)
        with open(fn, 'w', encoding='utf-8') as fh:
            _json.dump({
                'benchmark': f'gpt2_{SCALE_TAG}',
                'method': method,
                'seed': seed,
                'phase': phase,
                'hyperparams': hyperparams,
                'best_epoch': early_stopping.best_epoch + 1,
                'last_epoch': (early_stopping.last_epoch or 0) + 1,
                'schedule_epochs': SCHEDULE_EPOCHS,
                'final': {k: v for k, v in result.items()
                          if not k.startswith('test_ppl@')},
                'trajectory': traj,
            }, fh, indent=2, default=str)

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
                        help='Comma-separated sparsity targets in (0,1) for the sweep. '
                             'Default if omitted: 0.25,0.5,0.75.')
    parser.add_argument('--scale', choices=list(SCALES), default='small',
                        help="From-scratch model scale: 'tiny' (2.1M), 'small' (7.4M, the "
                             "Table-1 reference, 10-method roster, n=5), 'medium' (18M) "
                             "and 'large' (66M). Non-'small' scales default to the core-6 "
                             "roster, 12 epochs, n=3. All scales use the 'auto' "
                             "(dimension-aware, 12/40 evals) PSO budget.")
    parser.add_argument('--full-roster', action='store_true',
                        help="Force the full 10-method roster even on tiny/medium "
                             "(default: core-6 on non-'small' scales).")
    parser.add_argument('--robust-roster', action='store_true',
                        help="Run the robust-decay head-to-head (REVIEWER-1): Baseline, "
                             "WD-tuned, Tau(alpha=0), Huber-decay (= AdamHD), "
                             "PseudoHuber-decay, LogCosh-decay, tau(w). Writes to "
                             "*_robust.csv/json so it never clobbers the main roster.")
    parser.add_argument('--data-fraction', type=float, default=None,
                        help="Fraction of the training split to use (default 1.0). Epochs "
                             "are scaled by 1/fraction so the number of optimizer steps is "
                             "held constant, isolating the amount of UNIQUE data from the "
                             "length of training. Roster is Baseline / Tau(alpha=0) / tau(w) "
                             "with hyperparameters transferred (via --hp-from, required) "
                             "from the full-data run at the same scale, so no PSO is "
                             "repeated. Writes *_dataNN.csv.")
    parser.add_argument('--converge', action='store_true',
                        help="With --data-fraction: give every condition an epoch ceiling\n"
                             "high enough that early stopping decides when to stop, instead\n"
                             "of holding optimizer steps constant. Fixes the confound where\n"
                             "the 100%% condition ran out of budget before converging.")
    parser.add_argument('--two-by-two', action='store_true',
                        help="Run the scope x adaptivity 2x2 (REVIEWER-5): Baseline, "
                             "WD-tuned, Tau(AdamW-scope), Tau(alpha=0), tau(w). "
                             "Decomposes tau-decay's margin over tuned AdamW into a "
                             "scope/schedule component and a magnitude-adaptivity "
                             "component. Writes to *_2x2.csv/json.")
    parser.add_argument('--robust-new-only', action='store_true',
                        help="Cheaper variant of --robust-roster: run ONLY the three new "
                             "competitors (Huber/PseudoHuber/LogCosh decay), writing "
                             "*_robust_new.csv. The four shared methods (Baseline, "
                             "WD-tuned, Tau(alpha=0), tau(w)) are reused from the existing "
                             "same-scale run and merged in by "
                             "analysis/merge_robust_results.py. Roughly halves GPU time; "
                             "valid because the protocol, config and seeds are identical "
                             "and the pipeline is deterministic.")
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override the early-stopping CEILING NUM_EPOCHS (default: 30 '
                             'for small, 12 otherwise). Does NOT change the LR-schedule '
                             'horizon; see --schedule-epochs.')
    parser.add_argument('--schedule-epochs', type=int, default=None,
                        help='Override SCHEDULE_EPOCHS, the horizon (in epochs) of the '
                             'linear LR decay. Default: the base epoch budget of the scale '
                             '(30 small / 12 otherwise), scaled by 1/fraction under '
                             '--data-fraction so every fraction anneals over the same '
                             'number of optimizer steps.')
    parser.add_argument('--pso-budget', choices=['light', 'standard', 'thorough', 'auto'],
                        default=None,
                        help="Override the PSO budget (default 'auto' at every scale: 12 "
                             "evals for 1-D searches, 40 for >=2-D).")
    parser.add_argument('--methods', type=str, default=None,
                        help="Comma-separated subset of methods to run (e.g. "
                             "'Baseline,WD-tuned'). Output file names and the journal are "
                             "unchanged, so a roster can be split across pods and later "
                             "assembled by re-running with the full roster on the "
                             "concatenated journals (every run is then served from cache).")
    parser.add_argument('--patience', type=int, default=None,
                        help='Override the early-stopping patience (default: 8 for small, '
                             '4 otherwise). Mainly for smoke tests.')
    parser.add_argument('--train-fraction', type=float, default=None,
                        help='SMOKE TESTS ONLY: subsample the training split to this '
                             'fraction without any other change (unlike --data-fraction, '
                             'which is the factorial data-quantity arm with its own '
                             'roster, epochs and output files).')
    parser.add_argument('--hp-from', type=str, default=None,
                        help='Path to a *_results.json whose best_hyperparams are used as '
                             'FIXED hyperparameters (no PSO) for the methods run. Required '
                             'by --data-fraction.')
    parser.add_argument('--seeds', type=int, default=None,
                        help='Use the first N seeds (default: 5 for small, 3 for tiny/medium).')
    parser.add_argument('--instrument', action='store_true',
                        help='Log per-epoch mechanism trajectory (train/val PPL + '
                             'weight-magnitude stats) to results/instrumentation/ for '
                             'the Workstream-2 overfitting-delay and weight-dynamics '
                             'figures. Adds a val/weight pass per epoch (small overhead).')
    args = parser.parse_args()

    # Enable mechanism instrumentation and tag output files by scale.
    INSTRUMENT = args.instrument
    SCALE_TAG = args.scale

    # Apply the chosen scale by overriding the architecture globals that train_model
    # reads. 'small' leaves the reference config untouched for backward compatibility.
    scale = args.scale
    cfg = SCALES[scale]
    HIDDEN_SIZE = cfg['hidden']
    NUM_LAYERS = cfg['layers']
    NUM_HEADS = cfg['heads']
    INTERMEDIATE_SIZE = cfg['intermediate']

    # Non-reference scales use the cheaper, claim-focused defaults; 'small' keeps the
    # exact protocol it was integrated under (10 methods, standard PSO).
    if scale == 'small':
        # 'auto' (was 'standard' = 80 evals in the 2026-06 reference run): the whole sweep
        # now shares one budget policy, 12 evals per 1-D search and 40 per >=2-D search.
        PSO_BUDGET = 'auto'
        default_methods = None                      # -> DEFAULT_METHODS (10)
        suffix = ''
        # Pin to the published n=5. Extending the master seed list to 10 (for the
        # higher-powered primary comparison at 'large') must not silently change the
        # reference configuration whose numbers are already in the paper.
        SEEDS = ALL_SEEDS[:N_RUNS]
    else:
        # Reduced-budget cross-scale runs: core-6 + auto PSO, plus a lighter training
        # budget (12 epochs with early stopping, 3 seeds) so the from-scratch scale
        # sweep is feasible in wall-clock. Each scale stays internally consistent (all
        # methods share the budget), so the per-scale tau-vs-competitor gap — the thing
        # the scale figure reports — is unaffected.
        PSO_BUDGET = 'auto'
        default_methods = list(CORE_METHODS)
        suffix = f'_{scale}'
        NUM_EPOCHS = 12
        PATIENCE = 4
        SEEDS = SEEDS[:3]
    if args.full_roster:
        default_methods = None
    # The LR-schedule horizon follows the scale's base budget; --epochs only moves the
    # early-stopping ceiling, --schedule-epochs only the horizon.
    SCHEDULE_EPOCHS = NUM_EPOCHS
    if args.epochs is not None:
        NUM_EPOCHS = args.epochs
    if args.schedule_epochs is not None:
        SCHEDULE_EPOCHS = args.schedule_epochs
    if args.pso_budget is not None:
        PSO_BUDGET = args.pso_budget
    if args.patience is not None:
        PATIENCE = args.patience
    if args.train_fraction is not None:
        if args.data_fraction is not None:
            parser.error("--train-fraction and --data-fraction are mutually exclusive")
        if not 0 < args.train_fraction <= 1.0:
            parser.error("--train-fraction must be in (0, 1]")
        DATA_FRACTION = args.train_fraction
        print(f"[SMOKE] training on a {DATA_FRACTION:g} fraction of the training split",
              flush=True)
    if args.seeds is not None:
        # Select from the FULL seed list, not from the scale-truncated one. Applying
        # this after `SEEDS = SEEDS[:3]` above would make `--seeds 10` silently mean
        # `SEEDS[:3][:10]` = 3 seeds — the run would look fine and quietly deliver the
        # underpowered n=3 that the whole point of `--seeds 10` is to escape.
        if args.seeds > len(ALL_SEEDS):
            parser.error(f"--seeds {args.seeds} exceeds the {len(ALL_SEEDS)} seeds "
                         f"defined in SEEDS; add more before asking for that many.")
        SEEDS = ALL_SEEDS[:args.seeds]

    fixed_hyperparams = None
    methods_override = default_methods
    csv_filename = f'gpt2{suffix}_wikitext_standardized_results.csv'
    json_filename = f'gpt2{suffix}_wikitext_standardized_results.json'
    experiment_name = f'GPT2{suffix}_WikiText2_Standardized'

    prune_targets = None
    if args.prune_targets:
        prune_targets = [float(x) for x in args.prune_targets.split(',') if x.strip()]

    # Sentinel default is None, NOT 1.0: passing --data-fraction 1.0 must still take
    # this branch, otherwise the 100% point of the factorial would silently run a
    # different experiment (PSO-tuned core roster) from the 25%/50% points.
    if args.data_fraction is not None:
        if not 0 < args.data_fraction <= 1.0:
            parser.error("--data-fraction must be in (0, 1]")
        if args.hp_from is None:
            parser.error("--data-fraction requires --hp-from <results.json>: the arm varies "
                         "the amount of data at FIXED hyperparameters, transferred from the "
                         "full-data run at the same scale (no PSO is repeated).")
        DATA_FRACTION = args.data_fraction
        # Hold optimizer steps constant: fewer documents -> proportionally more epochs,
        # and the LR schedule spans exactly that step budget, so every fraction anneals
        # over the same number of optimizer steps as the full-data run (the schedule is
        # deliberately independent of the early-stopping ceiling).
        steps_const_epochs = int(round(SCHEDULE_EPOCHS / DATA_FRACTION))
        SCHEDULE_EPOCHS = steps_const_epochs
        if args.converge:
            # Ceiling above the schedule horizon. Beyond SCHEDULE_EPOCHS the linear
            # schedule has reached 0, so the extra epochs only let early stopping
            # (patience 5) settle after annealing instead of being cut by the budget.
            NUM_EPOCHS = (args.epochs if args.epochs is not None
                          else max(40, steps_const_epochs))
            PATIENCE = 5
        else:
            NUM_EPOCHS = steps_const_epochs
            PATIENCE = max(PATIENCE, int(round(PATIENCE / DATA_FRACTION)))
        methods_override = ['Baseline', 'Tau(alpha=0)', 'τ(w)']
        _dtag = f"{int(round(args.data_fraction * 100))}{'c' if args.converge else ''}"
        csv_filename = f'gpt2{suffix}_wikitext_standardized_results_data{_dtag}.csv'
        json_filename = f'gpt2{suffix}_wikitext_standardized_results_data{_dtag}.json'
        experiment_name = experiment_name + f'_Data{_dtag}'
        SCALE_TAG = f'{scale}_data{_dtag}'
        _mode = ("convergence ceiling above a fixed schedule horizon"
                 if args.converge else "optimizer steps held constant across fractions")
        print(f"[DATA] fraction={DATA_FRACTION:g} epochs={NUM_EPOCHS} "
              f"schedule_epochs={SCHEDULE_EPOCHS} patience={PATIENCE} ({_mode})", flush=True)

    if args.two_by_two:
        methods_override = list(TWO_BY_TWO_METHODS)
        csv_filename = f'gpt2{suffix}_wikitext_standardized_results_2x2.csv'
        json_filename = f'gpt2{suffix}_wikitext_standardized_results_2x2.json'
        experiment_name = experiment_name + '_2x2'

    if args.robust_roster or args.robust_new_only:
        # Nearest-neighbour comparison: same decoupled post-optimizer mechanism for every
        # decay method, only the saturation profile differs. Separate output files.
        if args.robust_new_only:
            methods_override = [m for m in ROBUST_METHODS if m in ROBUST_DECAY_METHODS]
            tag = '_robust_new'
        else:
            methods_override = list(ROBUST_METHODS)
            tag = '_robust'
        csv_filename = f'gpt2{suffix}_wikitext_standardized_results{tag}.csv'
        json_filename = f'gpt2{suffix}_wikitext_standardized_results{tag}.json'
        experiment_name = experiment_name + '_Robust'

    if args.prune_pso:
        methods_override = ['Baseline', 'τ(w)']
        if prune_targets is None:
            prune_targets = [0.25, 0.5, 0.75]
        csv_filename = f'gpt2{suffix}_wikitext_standardized_results_prune_pso.csv'
        json_filename = f'gpt2{suffix}_wikitext_standardized_results_prune_pso.json'
        experiment_name = experiment_name + '_PrunePSO'

    if args.methods:
        _alias = {'Tau(w)': 'τ(w)', 'tau(w)': 'τ(w)', 'tau': 'τ(w)', 'WD-tuned-weights': 'WD-tuned(weights)'}
        _known = (set(SEARCH_SPACES) | set(DEFAULT_METHODS) | set(ROBUST_METHODS)
                  | set(TWO_BY_TWO_METHODS))
        _sel = [_alias.get(m.strip(), m.strip()) for m in args.methods.split(',')
                if m.strip()]
        _bad = [m for m in _sel if m not in _known]
        if _bad:
            parser.error(f"--methods: unknown method(s) {_bad}; known: {sorted(_known)}")
        methods_override = _sel

    if args.hp_from:
        with open(args.hp_from, encoding='utf-8') as _fh:
            _hp_all = _json.load(_fh).get('best_hyperparams', {})
        _roster = list(methods_override) if methods_override else list(DEFAULT_METHODS)
        _missing = [m for m in _roster if m != 'Baseline' and m not in _hp_all]
        if _missing:
            parser.error(f"--hp-from {args.hp_from}: no best_hyperparams for {_missing}")
        fixed_hyperparams = {m: dict(_hp_all.get(m, {})) for m in _roster}
        methods_override = _roster
        print(f"[HP] fixed hyperparameters from {args.hp_from}: "
              + ", ".join(f"{m}={fixed_hyperparams[m]}" for m in _roster), flush=True)

    print(f"[CONFIG] scale={scale} params={SCALES[scale]} epochs={NUM_EPOCHS} "
          f"schedule_epochs={SCHEDULE_EPOCHS} patience={PATIENCE} pso_budget={PSO_BUDGET} "
          f"n_seeds={len(SEEDS)} seeds={SEEDS} instrument={INSTRUMENT} "
          f"methods={methods_override or 'DEFAULT_METHODS'}", flush=True)

    run_benchmark(
        experiment_name=experiment_name,
        benchmark_title=f'GPT-2 ({scale}) on WikiText-2',
        model_name=f'GPT-2 {scale} ({NUM_LAYERS}L, {HIDDEN_SIZE}H, {NUM_HEADS}heads)',
        device=DEVICE,
        config={
            'Epochs': f'{NUM_EPOCHS} ceiling, LR schedule over {SCHEDULE_EPOCHS} '
                      f'(early stopping, patience={PATIENCE})',
            'Batch size': BATCH_SIZE,
            'Learning rate': LR,
        },
        seeds=SEEDS, train_fn=train_model,
        primary_metric='test_ppl', metric_mode='min',
        pso_metric='val_ppl', pso_mode='min',
        pso_budget=PSO_BUDGET,
        csv_filename=csv_filename,
        json_filename=json_filename,
        quiet=args.quiet,
        fixed_hyperparams=fixed_hyperparams,
        methods=methods_override,
        prune_targets=prune_targets,
        # With instrumentation on, the seeds[0] final run is trained explicitly instead of
        # being served from the PSO cache, so its trajectory file is a genuine final run.
        reuse_pso_seed0=not INSTRUMENT,
    )
