"""
GPT-2 Small (124M, standard architecture) on WikiText-103 — "insurance run".

Addresses the top-venue "toy setup" objection left open by the WikiText-2
word-level scale sweep (tiny 2M -> large 90M):
  - REAL corpus: WikiText-103-raw (103M training tokens), standard splits.
  - STANDARD tokenizer: GPT-2 BPE (vocab 50257) via HuggingFace tokenizers.
  - STANDARD metric: token-level perplexity on concatenated 512-token blocks,
    no padding and no ignore_index tricks -> numbers comparable to literature.
  - STANDARD architecture: GPT-2 small (12L, 768H, 12 heads, ~124M params
    with tied embeddings), trained from scratch.

Protocol (feasible at this scale — full PSO would take months of GPU time):
  1. Hyperparameter TRANSFER from the PSO-tuned 90M point of the scale sweep,
     with decay strengths rescaled by the step-count ratio (the cumulative
     effect of a per-step decay scales with the number of optimizer steps).
  2. Confirmation MINI-SWEEP: for each non-Baseline method, 3 candidate
     strengths ({1/4x, 1x, 4x} the transferred value), 1 seed, short budget;
     best val-PPL wins. Choice cached to results/wt103_sweep_choice.json.
  3. FINAL runs: core-6 roster x 3 seeds, early stopping, --instrument
     trajectories for the mechanism figures (scale tag 'wt103').

Usage:
    python gpt2_wt103_standardized.py --stage sweep      # step 2 only
    python gpt2_wt103_standardized.py --stage final      # step 3 (needs cache)
    python gpt2_wt103_standardized.py --stage all        # 2 then 3 (default)
    python gpt2_wt103_standardized.py --smoke            # tiny smoke test
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from experiment_utils import (
    EarlyStopping, run_benchmark,
    get_regularization_transformer, apply_decoupled_decay_transformer,
    optimizer_weight_decay, tau_alpha_for, NO_LOSS_PENALTY, TAU_METHODS,
    measure_sparsity, set_seed, CORE_METHODS, weight_magnitude_stats,
)

# ============================
# CONFIGURATION
# ============================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VOCAB_SIZE = 50257          # GPT-2 BPE
MAX_SEQ_LENGTH = 512
HIDDEN_SIZE = 768
NUM_LAYERS = 12
NUM_HEADS = 12
INTERMEDIATE_SIZE = 3072
DROPOUT = 0.1

NUM_EPOCHS = 8              # early stopping usually triggers before this
BATCH_SIZE = 16
LR = 3e-4
PATIENCE = 2
WARMUP_FRAC = 0.05          # fraction of total steps

SEEDS = [42, 123, 456]
SWEEP_SEED = 42
SWEEP_EPOCHS = 2            # per candidate in the confirmation mini-sweep

DATA_DIR = 'data'
CACHE = {s: os.path.join(DATA_DIR, f'wt103_{s}_gpt2bpe.npy')
         for s in ('train', 'validation', 'test')}
SWEEP_CHOICE_PATH = 'results/wt103_sweep_choice.json'

INSTRUMENT = False
INSTRUMENT_DIR = 'results/instrumentation'
SCALE_TAG = 'wt103'

# ── Hyperparameter transfer from the PSO-tuned 'large' scale-sweep point ──
# ("90M" below is that point's historical label; its measured trainable parameter
#  count is 65.6M - the embedding matrix is tied and was counted twice.)
# (new results/gpt2_large_wikitext_standardized_results.json, best_hyperparams).
# Per-step decays (tau family, AdamW wd) accumulate over optimizer steps, so the
# transferred strength is rescaled by STEP_RATIO = steps_90M / steps_wt103
# (~13.8k vs ~101k at the full budget). Loss penalties (L2/EN) set a per-step
# equilibrium, not a cumulative one -> transferred unscaled. The {1/4x, 1x, 4x}
# mini-sweep spans a 16x range around each transfer to absorb estimation error.
HP_90M = {
    'L2':           {'lambda_val': 1.1324654430748025e-05},
    'ElasticNet':   {'lambda_val': 1e-06, 'en_alpha': 0.5447956162689622},
    'WD-tuned':     {'wd': 0.09776957254503886},
    'Tau(alpha=0)': {'decay_strength': 0.00010253977986180159,
                     'tau0': 1.7509197307691162},
    'τ(w)':         {'decay_strength': 0.00019517224641449476,
                     'tau0': 1.000590589657011,
                     'tau_alpha': 21.256581564753876},
}
STEPS_90M = 1147 * 12       # steps/epoch x epochs of the 90M budget
# Rescaled per-step knob per method (None -> transfer unscaled).
STEP_SCALED_KEY = {'L2': None, 'ElasticNet': None, 'WD-tuned': 'wd',
                   'Tau(alpha=0)': 'decay_strength', 'τ(w)': 'decay_strength'}
SWEEP_FACTORS = (0.25, 1.0, 4.0)

# ============================
# DATA — GPT-2 BPE, concatenated 512-token blocks
# ============================

def _tokenize_split(split):
    """Tokenize a WikiText-103 split with GPT-2 BPE and cache as int32 .npy."""
    path = CACHE[split]
    if os.path.exists(path):
        return np.load(path, mmap_mode='r')
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f'[DATA] Tokenizing wikitext-103-raw-v1 {split} with GPT-2 BPE...',
          flush=True)
    ds = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split=split)
    tok = GPT2TokenizerFast.from_pretrained('gpt2')
    ids = []
    batch = []
    for t in ds['text']:
        batch.append(t)
        if len(batch) == 2000:
            for enc in tok(batch)['input_ids']:
                ids.extend(enc)
            batch = []
    if batch:
        for enc in tok(batch)['input_ids']:
            ids.extend(enc)
    arr = np.asarray(ids, dtype=np.int32)
    np.save(path, arr)
    print(f'[DATA] {split}: {len(arr):,} tokens -> {path}', flush=True)
    return np.load(path, mmap_mode='r')


class BlockDataset(Dataset):
    """Non-overlapping (MAX_SEQ_LENGTH+1)-token blocks of the concatenated corpus."""

    def __init__(self, tokens, block):
        self.tokens = tokens
        self.block = block
        self.n = (len(tokens) - 1) // block

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        chunk = torch.from_numpy(
            self.tokens[i * self.block: i * self.block + self.block + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]


def get_dataloaders(batch_size):
    loaders = {}
    for split in ('train', 'validation', 'test'):
        ds = BlockDataset(_tokenize_split(split), MAX_SEQ_LENGTH)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == 'train'),
            drop_last=(split == 'train'), num_workers=2, pin_memory=True)
    return loaders['train'], loaders['validation'], loaders['test']


# ============================
# MODEL — GPT-2 small with SDPA (flash) attention
# ============================

class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_linear = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_linear(out)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.1):
        super().__init__()
        self.attention = CausalSelfAttention(hidden_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class GPT2LM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads,
                 intermediate_size, max_seq_length, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # tied
        self._init_weights()

    def _init_weights(self):
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
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.dropout(self.token_embedding(input_ids)
                         + self.position_embedding(pos))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))


# ============================
# EVALUATION — standard token-level perplexity
# ============================

AMP_DTYPE = torch.bfloat16 if (torch.cuda.is_available()
                               and torch.cuda.is_bf16_supported()) else torch.float16


def evaluate_lm(model, data_loader, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.autocast('cuda', dtype=AMP_DTYPE, enabled=device.type == 'cuda'):
                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)).float(), targets.view(-1),
                    reduction='sum')
            total_loss += loss.item()
            total_tokens += targets.numel()
    avg = total_loss / max(total_tokens, 1)
    return math.exp(min(avg, 20)), avg


# ============================
# TRAINING
# ============================

def train_model(method, hyperparams, seed, verbose=True, prune_targets=None,
                num_epochs=None):
    set_seed(seed)
    epochs = num_epochs or NUM_EPOCHS

    lambda_val = hyperparams.get('lambda_val', 0.0)
    extra_params = {k: v for k, v in hyperparams.items()
                    if k not in ('lambda_val', 'decay_strength', 'tau0', 'tau_alpha', 'delta', 'wd', 'rho')}

    train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE)

    model = GPT2LM(VOCAB_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_HEADS,
                   INTERMEDIATE_SIZE, MAX_SEQ_LENGTH, DROPOUT).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR,
        weight_decay=optimizer_weight_decay(method, hyperparams))

    total_steps = len(train_loader) * epochs
    warmup_steps = max(1, int(total_steps * WARMUP_FRAC))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / (total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=0.01, mode='min')

    if verbose:
        params_str = ', '.join(f"{k}={v:.2e}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in hyperparams.items())
        print(f"[TRAIN] Method={method}, {params_str or 'no-hp'}, seed={seed}, "
              f"epochs<={epochs}", flush=True)

    traj = []
    for epoch in range(epochs):
        model.train()
        train_loss, n_batches = 0.0, 0
        t0 = time.time()
        for inputs, targets in train_loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=AMP_DTYPE,
                                enabled=DEVICE.type == 'cuda'):
                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.view(-1, VOCAB_SIZE).float(), targets.view(-1))
                if method not in NO_LOSS_PENALTY:
                    loss = loss + get_regularization_transformer(
                        model, method, lambda_val, DEVICE,
                        extra_params=extra_params)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Post-optimizer decoupled decay: tau family + robust-decay competitors.
            apply_decoupled_decay_transformer(model, method, hyperparams)

            train_loss += loss.item()
            n_batches += 1

        val_ppl, _ = evaluate_lm(model, val_loader, DEVICE)
        train_ppl = math.exp(min(train_loss / max(n_batches, 1), 20))

        if INSTRUMENT:
            traj.append({
                'epoch': epoch + 1,
                'train_ppl': train_ppl,
                'val_ppl': val_ppl,
                **weight_magnitude_stats(model, transformer=True),
            })

        if verbose:
            print(f"  Epoch {epoch+1}/{epochs}: train_ppl={train_ppl:.2f}, "
                  f"val_ppl={val_ppl:.2f} ({(time.time()-t0)/60:.1f} min)",
                  flush=True)

        if early_stopping(val_ppl, model, epoch):
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}", flush=True)
            break

    model = early_stopping.restore_best_model(model)

    test_ppl, test_loss = evaluate_lm(model, test_loader, DEVICE)
    val_ppl, _ = evaluate_lm(model, val_loader, DEVICE)
    # Outcome guard: the reported number must belong to the best epoch.
    early_stopping.check_restored(val_ppl, context=f'wt103 {method} seed={seed}')
    sparsity, total_params, _ = measure_sparsity(model)

    result = {
        'test_ppl': test_ppl,
        'val_ppl': val_ppl,
        'test_loss': test_loss,
        'sparsity': sparsity * 100,
        'convergence_epoch': early_stopping.best_epoch + 1,
        'total_params': total_params,
    }

    if INSTRUMENT and traj:
        os.makedirs(INSTRUMENT_DIR, exist_ok=True)
        safe = method.replace('/', '').replace('(', '').replace(')', '').replace('=', '')
        fn = os.path.join(INSTRUMENT_DIR, f'gpt2_{SCALE_TAG}_{safe}_seed{seed}.json')
        with open(fn, 'w', encoding='utf-8') as fh:
            json.dump({'benchmark': f'gpt2_{SCALE_TAG}', 'method': method,
                       'seed': seed, 'hyperparams': hyperparams,
                       'final': result, 'trajectory': traj}, fh,
                      indent=2, default=str)

    del model
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    return result


# ============================
# STAGE 1 — transfer + confirmation mini-sweep
# ============================

def transferred_hp(method, steps_new):
    """90M hyperparams with per-step decay knobs rescaled by the step ratio."""
    hp = dict(HP_90M[method])
    key = STEP_SCALED_KEY[method]
    if key is not None:
        hp[key] = hp[key] * STEPS_90M / steps_new
    return hp


def sweep_knob(method):
    """The single strength knob swept per method."""
    return {'L2': 'lambda_val', 'ElasticNet': 'lambda_val', 'WD-tuned': 'wd',
            'Tau(alpha=0)': 'decay_strength', 'τ(w)': 'decay_strength'}[method]


def run_sweep(methods, steps_new):
    """3-candidate confirmation sweep per method, 1 seed, short budget."""
    global INSTRUMENT
    saved_instrument, INSTRUMENT = INSTRUMENT, False
    choice = {}
    for method in methods:
        if method == 'Baseline':
            choice[method] = {}
            continue
        base = transferred_hp(method, steps_new)
        knob = sweep_knob(method)
        best_hp, best_val = None, float('inf')
        for f in SWEEP_FACTORS:
            hp = dict(base)
            hp[knob] = base[knob] * f
            print(f"\n[SWEEP] {method}: {knob}={hp[knob]:.3e} "
                  f"({f}x transfer)", flush=True)
            res = train_model(method, hp, SWEEP_SEED, verbose=True,
                              num_epochs=SWEEP_EPOCHS)
            print(f"[SWEEP] {method} {f}x -> val_ppl={res['val_ppl']:.3f}",
                  flush=True)
            if res['val_ppl'] < best_val:
                best_val, best_hp = res['val_ppl'], hp
        choice[method] = best_hp
        print(f"[SWEEP] BEST {method}: {best_hp} (val_ppl={best_val:.3f})",
              flush=True)
        os.makedirs(os.path.dirname(SWEEP_CHOICE_PATH), exist_ok=True)
        with open(SWEEP_CHOICE_PATH, 'w', encoding='utf-8') as fh:
            json.dump(choice, fh, indent=2, ensure_ascii=False)
    INSTRUMENT = saved_instrument
    print(f"\n[SWEEP] Choices saved to {SWEEP_CHOICE_PATH}", flush=True)
    return choice


# ============================
# MAIN
# ============================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['sweep', 'final', 'all'], default='all')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--seeds', type=int, default=None)
    parser.add_argument('--instrument', action='store_true')
    parser.add_argument('--smoke', action='store_true',
                        help='Tiny config + synthetic data: verifies the full '
                             'pipeline in minutes on CPU/small GPU.')
    args = parser.parse_args()

    INSTRUMENT = args.instrument

    if args.smoke:
        # Shrink everything; keep the real code paths.
        VOCAB_SIZE, MAX_SEQ_LENGTH = 512, 64
        HIDDEN_SIZE, NUM_LAYERS, NUM_HEADS, INTERMEDIATE_SIZE = 64, 2, 2, 128
        NUM_EPOCHS, BATCH_SIZE, SWEEP_EPOCHS = 2, 4, 1
        SEEDS = SEEDS[:2]
        os.makedirs(DATA_DIR, exist_ok=True)
        rng = np.random.default_rng(0)
        for s, n in (('train', 40000), ('validation', 6000), ('test', 6000)):
            np.save(CACHE[s].replace('.npy', '_smoke.npy'),
                    rng.integers(0, VOCAB_SIZE, n, dtype=np.int32))
        CACHE = {s: CACHE[s].replace('.npy', '_smoke.npy')
                 for s in CACHE}
        SCALE_TAG = 'wt103smoke'

    if args.epochs is not None:
        NUM_EPOCHS = args.epochs
    if args.seeds is not None:
        SEEDS = SEEDS[:args.seeds]

    methods = list(CORE_METHODS)

    # Steps at the full budget, for the transfer rescaling (needs train size).
    n_train_tokens = len(_tokenize_split('train'))
    steps_per_epoch = ((n_train_tokens - 1) // MAX_SEQ_LENGTH) // BATCH_SIZE
    steps_new = steps_per_epoch * NUM_EPOCHS
    print(f"[CONFIG] train tokens={n_train_tokens:,}, steps/epoch="
          f"{steps_per_epoch:,}, full-budget steps={steps_new:,}, "
          f"transfer step-ratio={STEPS_90M / steps_new:.3f}", flush=True)

    if args.stage in ('sweep', 'all'):
        choice = run_sweep(methods, steps_new)
    else:
        with open(SWEEP_CHOICE_PATH, encoding='utf-8') as fh:
            choice = json.load(fh)
        print(f"[CONFIG] Loaded sweep choices from {SWEEP_CHOICE_PATH}", flush=True)

    if args.stage in ('final', 'all'):
        run_benchmark(
            experiment_name='GPT2_WT103_Standardized',
            benchmark_title='GPT-2 small (124M) on WikiText-103 (BPE)',
            model_name=f'GPT-2 small ({NUM_LAYERS}L, {HIDDEN_SIZE}H, '
                       f'{NUM_HEADS}heads, BPE {VOCAB_SIZE})',
            device=DEVICE,
            config={
                'Epochs': f'{NUM_EPOCHS} (early stopping, patience={PATIENCE})',
                'Batch size': BATCH_SIZE,
                'Learning rate': LR,
                'Sequence length': MAX_SEQ_LENGTH,
                'Hyperparams': 'transfer from 90M + confirmation sweep',
            },
            seeds=SEEDS, train_fn=train_model,
            primary_metric='test_ppl', metric_mode='min',
            pso_metric='val_ppl', pso_mode='min',
            pso_budget='light',
            csv_filename='gpt2_wt103_standardized_results.csv',
            json_filename='gpt2_wt103_standardized_results.json',
            quiet=args.quiet,
            fixed_hyperparams=choice,
            methods=methods,
        )
