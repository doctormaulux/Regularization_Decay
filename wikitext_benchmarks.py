"""
WikiText-2 Language Modeling Benchmarks - Config-driven
Fine-tuning pretrained decoder LMs with τ(w) regularization

Models: SmolLM2-135M, Qwen2.5-0.5B, Llama-3.2-1B, Phi-2,
        Gemma-2-2B, Mamba-130M, RWKV-4-169M
Task: Causal language modeling on WikiText-2

Usage:
    python wikitext_benchmarks.py --model smollm2
    python wikitext_benchmarks.py --model all
    python wikitext_benchmarks.py --list
"""

import argparse
import sys
import torch
from experiment_utils import (
    DEFAULT_METHODS,
    train_hf_lm, run_benchmark, get_wikitext_dataloaders, CORE_METHODS
)

# Under the "from-scratch AR-LM mechanism" claim these pretrained fine-tuning runs
# are breadth/robustness evidence (graceful degradation to tuned weight decay), NOT
# the mechanism itself (which lives in the from-scratch GPT-2 scales). So 'all' runs
# only the three core LMs; the heavier giants (phi2/gemma2/mamba/rwkv) are droppable
# per the paper scope and remain runnable individually via --model <key>.
CORE_LMS = ['smollm2', 'qwen25', 'llama32']

try:
    from transformers import AutoModelForCausalLM
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("[ERROR] HuggingFace transformers not available.")

# ============================
# CONFIGURATION
# ============================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULTS = dict(
    # patience 3 (was 2) gives τ room to act within the short LM fine-tune (CODICE-12).
    # pso_budget 'auto' (was 'light', then 'standard'): the 5-particle/2-iter 'light'
    # search caused the SmolLM2/Qwen particle-0 artifact (τ never moved off the seed-42
    # init → null effect mis-read as a structural property, CODICE-2/8), but flat
    # 'standard' (80 evals) is ~10x overkill on the 1-D competitors and made the run
    # computationally infeasible. 'auto' scales evals with search-space dimensionality
    # (dimension_aware_budget): a genuine swarm on every method, no wasted GPU-hours.
    patience=3, warmup_ratio=0.1,
    max_seq_length=256, pso_budget='auto',
    # 3 seeds (was 5): retains a CI at ~40% less cost on these expensive fine-tunes,
    # in line with the reduced-budget reframe (breadth/robustness tier, not the flagship).
    seeds=[42, 123, 456],
    torch_dtype=None,
    use_mixed_precision=False,
    model_class=None,
)

MODELS = {
    'smollm2': dict(
        model_name='HuggingFaceTB/SmolLM2-135M',
        num_epochs=5, batch_size=8, lr=5e-5,
        experiment_name='SmolLM2_WikiText2_Standardized',
        title='SmolLM2-135M on WikiText-2',
        csv='results/smollm2_wikitext_standardized_results.csv',
        json='results/smollm2_wikitext_standardized_results.json',
    ),
    'qwen25': dict(
        model_name='Qwen/Qwen2.5-0.5B',
        num_epochs=3, batch_size=4, lr=2e-5,
        experiment_name='Qwen25_WikiText2_Standardized',
        title='Qwen2.5-0.5B on WikiText-2',
        csv='results/qwen25_wikitext_standardized_results.csv',
        json='results/qwen25_wikitext_standardized_results.json',
    ),
    'llama32': dict(
        model_name='meta-llama/Llama-3.2-1B',
        num_epochs=3, batch_size=4, lr=2e-5,
        experiment_name='Llama32_WikiText2_Standardized',
        title='Llama-3.2-1B on WikiText-2',
        csv='results/llama32_wikitext_standardized_results.csv',
        json='results/llama32_wikitext_standardized_results.json',
    ),
    'phi2': dict(
        model_name='microsoft/phi-2',
        num_epochs=3, batch_size=1, lr=1e-5,
        torch_dtype=torch.bfloat16,
        use_mixed_precision=False,
        experiment_name='Phi2_WikiText2_Standardized',
        title='Phi-2 on WikiText-2',
        csv='results/phi2_wikitext_standardized_results.csv',
        json='results/phi2_wikitext_standardized_results.json',
    ),
    'gemma2': dict(
        model_name='google/gemma-2-2b',
        num_epochs=3, batch_size=2, lr=1e-5,
        torch_dtype=torch.bfloat16,
        use_mixed_precision=False,
        experiment_name='Gemma2_WikiText2_Standardized',
        title='Gemma-2-2B on WikiText-2',
        csv='results/gemma2_wikitext_results.csv',
        json='results/gemma2_wikitext_results.json',
    ),
    'mamba': dict(
        model_name='state-spaces/mamba-130m-hf',
        num_epochs=5, batch_size=8, lr=5e-5,
        model_class='MambaForCausalLM',
        experiment_name='Mamba_WikiText2_Standardized',
        title='Mamba-130M on WikiText-2',
        csv='results/mamba_wikitext_results.csv',
        json='results/mamba_wikitext_results.json',
    ),
    'rwkv': dict(
        model_name='RWKV/rwkv-4-169m-pile',
        num_epochs=5, batch_size=8, lr=5e-5,
        model_class='RwkvForCausalLM',
        experiment_name='RWKV_WikiText2_Standardized',
        title='RWKV-4 169M on WikiText-2',
        csv='results/rwkv_wikitext_results.csv',
        json='results/rwkv_wikitext_results.json',
    ),
}


# ============================
# RUNNER
# ============================

def run_model(key, quiet=False, prune_only=False, prune_pso=False, prune_targets=None,
              methods=None):
    cfg = {**DEFAULTS, **MODELS[key]}
    fixed_hyperparams = None
    methods_override = None
    if prune_only:
        # Skip PSO; use canonical winning hyperparameters and only run
        # tau(w) + Baseline to produce a clean accuracy-vs-sparsity curve.
        fixed_hyperparams = {
            'Baseline': {},
            'τ(w)': {'decay_strength': 1e-3, 'tau0': 0.5, 'tau_alpha': 10.0},
        }
        methods_override = ['Baseline', 'τ(w)']
        if prune_targets is None:
            prune_targets = [0.25, 0.5, 0.75]
        # Save under a distinct filename so we do not overwrite the main run.
        cfg['csv'] = cfg['csv'].replace('.csv', '_prune.csv')
        cfg['json'] = cfg['json'].replace('.json', '_prune.json')
        cfg['experiment_name'] = cfg['experiment_name'] + '_Prune'
    elif prune_pso:
        # Full PSO on Baseline + tau(w), then multi-seed eval with pruning sweep.
        # Used to reproduce the paper's tau(w) result and then test prunability.
        methods_override = ['Baseline', 'τ(w)']
        if prune_targets is None:
            prune_targets = [0.25, 0.5, 0.75]
        cfg['csv'] = cfg['csv'].replace('.csv', '_prune_pso.csv')
        cfg['json'] = cfg['json'].replace('.json', '_prune_pso.json')
        cfg['experiment_name'] = cfg['experiment_name'] + '_PrunePSO'
    else:
        # Main run: core-6 roster (drop L1/SCAD/MCP/LSP sparsity penalties, orthogonal
        # to the training-dynamics story and pure cost on these expensive fine-tunes).
        methods_override = list(CORE_METHODS)
    if not quiet:
        print(f"\n[INFO] Using device: {DEVICE}")

    def load_model(model_name, device):
        cls_name = cfg.get('model_class')
        if cls_name:
            try:
                import transformers
                cls = getattr(transformers, cls_name)
            except (ImportError, AttributeError):
                cls = AutoModelForCausalLM
        else:
            cls = AutoModelForCausalLM

        kwargs = {}
        if cfg.get('torch_dtype'):
            kwargs['torch_dtype'] = cfg['torch_dtype']
        return cls.from_pretrained(model_name, **kwargs).to(device)

    def get_dataloaders(seed):
        return get_wikitext_dataloaders(
            cfg['model_name'], cfg['batch_size'],
            cfg['max_seq_length'], seed
        )

    def train_model(method, hyperparams, seed, verbose=True, prune_targets=None):
        return train_hf_lm(
            method=method, hyperparams=hyperparams, seed=seed,
            model_name=cfg['model_name'], device=DEVICE,
            num_epochs=cfg['num_epochs'], lr=cfg['lr'],
            patience=cfg['patience'],
            warmup_ratio=cfg['warmup_ratio'],
            load_model_fn=load_model,
            get_dataloaders_fn=get_dataloaders,
            verbose=verbose,
            use_mixed_precision=cfg.get('use_mixed_precision', False),
            prune_targets=prune_targets,
        )

    if methods:
        methods_override = list(methods)
        print(f"[CONFIG] methods={methods_override}", flush=True)

    run_benchmark(
        experiment_name=cfg['experiment_name'],
        benchmark_title=cfg['title'],
        model_name=cfg['model_name'], device=DEVICE,
        config={
            'Epochs': f"{cfg['num_epochs']} (with early stopping,"
                      f" patience={cfg['patience']})",
            'Batch size': cfg['batch_size'],
            'Learning rate': cfg['lr'],
        },
        seeds=cfg['seeds'], train_fn=train_model,
        primary_metric='test_ppl', metric_mode='min',
        pso_metric='val_ppl', pso_mode='min',
        pso_budget=cfg['pso_budget'],
        csv_filename=cfg['csv'],
        json_filename=cfg['json'],
        quiet=quiet,
        fixed_hyperparams=fixed_hyperparams,
        methods=methods_override,
        prune_targets=prune_targets,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='WikiText-2 Language Modeling Benchmarks'
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
    parser.add_argument(
        '--methods', type=str, default=None,
        help='Comma-separated subset of the roster (output files and journal unchanged, '
             'so a roster can be split across pods and assembled from the journals).'
    )
    parser.add_argument(
        '--prune-only', action='store_true',
        help='Skip PSO, run only tau(w) + Baseline with fixed hyperparameters '
             'and apply post-training magnitude pruning sweep at 25/50/75%%. '
             'Saves to *_prune.csv to keep the main results untouched.'
    )
    parser.add_argument(
        '--prune-pso', action='store_true',
        help='Full PSO on Baseline + tau(w), then multi-seed eval with pruning '
             'sweep at 25/50/75%%. Use this to reproduce the paper tau(w) result '
             'and then test prunability. Saves to *_prune_pso.csv.'
    )
    parser.add_argument(
        '--prune-targets', type=str, default=None,
        help='Comma-separated sparsity targets in (0,1) for the sweep, '
             'e.g. "0.25,0.5,0.75". Default if omitted: 0.25,0.5,0.75.'
    )
    args = parser.parse_args()

    if args.list:
        print("Available WikiText-2 models:")
        for k, v in MODELS.items():
            print(f"  {k:12s}  {v['model_name']}")
        sys.exit(0)

    if not args.model:
        parser.print_help()
        sys.exit(1)

    if not HF_AVAILABLE:
        print("[ERROR] Cannot run without HuggingFace libraries")
        sys.exit(1)

    prune_targets = None
    if args.prune_targets:
        prune_targets = [float(x) for x in args.prune_targets.split(',') if x.strip()]

    methods = None
    if args.methods:
        _alias = {'Tau(w)': 'τ(w)', 'tau(w)': 'τ(w)'}
        methods = [_alias.get(m.strip(), m.strip()) for m in args.methods.split(',') if m.strip()]
        _bad = [m for m in methods if m not in CORE_METHODS and m not in DEFAULT_METHODS]
        if _bad:
            parser.error(f"--methods: unknown {_bad}")

    if args.model == 'all':
        for key in CORE_LMS:
            run_model(key, quiet=args.quiet,
                      prune_only=args.prune_only,
                      prune_pso=args.prune_pso,
                      prune_targets=prune_targets, methods=methods)
    else:
        run_model(args.model, quiet=args.quiet,
                  prune_only=args.prune_only,
                  prune_pso=args.prune_pso,
                  prune_targets=prune_targets, methods=methods)
