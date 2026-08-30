"""
Utilità comuni per esperimenti standardizzati
Fornisce: early stopping, training loop, statistical reporting, PSO optimization
"""

import math
import inspect
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from scipy import stats
import csv
import json
import os
from collections.abc import Callable
import time
from contextlib import nullcontext


# ============================
# HYPERPARAMETER SEARCH SPACES
# ============================

SEARCH_SPACES = {
    'Baseline': {},
    'L1': {
        'lambda_val': {'low': 1e-6, 'high': 1e-1, 'log_scale': True},
    },
    'L2': {
        'lambda_val': {'low': 1e-6, 'high': 1e-1, 'log_scale': True},
    },
    'ElasticNet': {
        'lambda_val': {'low': 1e-6, 'high': 1e-1, 'log_scale': True},
        'en_alpha':   {'low': 0.1,  'high': 0.9,  'log_scale': False},
    },
    'SCAD': {
        'lambda_val': {'low': 1e-6, 'high': 1e-1, 'log_scale': True},
        'scad_a':     {'low': 2.5,  'high': 5.0,  'log_scale': False},
    },
    'MCP': {
        'lambda_val': {'low': 1e-6, 'high': 1e-1, 'log_scale': True},
        'mcp_gamma':  {'low': 1.5,  'high': 5.0,  'log_scale': False},
    },
    'LSP': {
        'lambda_val': {'low': 1e-6, 'high': 1e-1, 'log_scale': True},
        'lsp_theta':  {'low': 1e-6, 'high': 1e-1, 'log_scale': True},
    },
    # ---- tau-decay, canonical 2-parameter form (REVIEWER-2) ----
    # The historical (eta, tau0, alpha) triple is OVER-PARAMETERISED: the update
    #     w <- w - eta*w/(tau0 + alpha|w|) = w - (eta/tau0) * w/(1 + (alpha/tau0)|w|)
    # is invariant under (eta, tau0, alpha) -> (c*eta, c*tau0, c*alpha) for any c > 0,
    # so it has only TWO effective degrees of freedom. Searching all three gives PSO a
    # perfectly flat direction, makes numerically different configurations the same
    # method, and leaves the three symbols individually unidentifiable.
    #
    # Canonical form:   w <- w - rho * w / (1 + |w|/delta)
    #   rho   = eta/tau0   maximum relative decay rate (the rate as w -> 0)
    #   delta = tau0/alpha knee / magnitude scale (the reviewer's 1/kappa)
    #
    # Symbols: rho and delta rather than the reviewer's lambda and kappa, because
    # lambda is already the loss-penalty strength throughout this codebase and paper,
    # and delta is the standard knee symbol in the robust-loss literature \u2014 which makes
    # tau-decay's knee directly comparable to the Huber/pseudo-Huber/log-cosh delta.
    #
    # Bounds: exactly the image of the old 3-D box under (eta,tau0,alpha) -> (rho,delta),
    # i.e. rho in [1e-6/2.0, 1e-1/0.1] and delta in [0.1/50, 2.0/1.0], so the reachable
    # set of methods is preserved while the redundant dimension is removed.
    '\u03c4(w)': {
        'rho':   {'low': 5e-7, 'high': 1.0, 'log_scale': True},
        'delta': {'low': 2e-3, 'high': 2.0, 'log_scale': True},
    },
    # Option-1 protocol additions:
    # WD-tuned: explicit decoupled-weight-decay competitor (the fair baseline for \u03c4),
    # with its weight_decay PSO-tuned instead of fixed at 0.01.
    'WD-tuned': {
        # Upper bound 10: with the per-step AdamW decay lr*wd and a linear-to-zero
        # schedule, wd = 0.1 caps the decay at 3e-5 per step at peak LR, below the
        # per-step rates the tau family selects (~1e-4 at 66M); the search must be
        # allowed to exceed it.
        'wd': {'low': 1e-6, 'high': 10.0, 'log_scale': True},
    },
    # WD-tuned(weights): AdamW's decoupled decay (scaled by the LR schedule) restricted to
    # the parameters the tau family decays (every parameter named 'weight': matrices,
    # embeddings and LayerNorm gains; biases excluded). Separates the SCOPE factor from
    # the SCHEDULE factor of the scope x adaptivity decomposition.
    'WD-tuned(weights)': {
        'wd': {'low': 1e-6, 'high': 10.0, 'log_scale': True},
    },
    # Tau(alpha=0): ablation of \u03c4(w) with the magnitude term removed (alpha=0).
    # \u03c4(alpha=0) reduces to a constant decoupled decay (-decay/tau0 * w), so comparing
    # \u03c4(w) vs \u03c4(alpha=0) isolates the contribution of magnitude-adaptivity.
    # Tau(alpha=0) was over-parameterised too, and more starkly: with alpha = 0 the
    # update is (eta/tau0)*w, so the declared 2-D {decay_strength, tau0} box had ONE
    # real degree of freedom and a completely flat direction. Canonically it is 1-D.
    'Tau(alpha=0)': {
        'rho': {'low': 5e-7, 'high': 1.0, 'log_scale': True},
    },
    # Missing cell of the scope x adaptivity 2x2 (REVIEWER-5). WD-tuned and Tau(alpha=0)
    # differ in BOTH scope/schedule and nothing else; tau(w) adds adaptivity on top of
    # Tau(alpha=0). To separate the two factors we also need magnitude-adaptive decay
    # carrying AdamW's scope and schedule instead of tau's:
    #
    #                          constant profile      magnitude-adaptive profile
    #   AdamW scope+schedule   WD-tuned              Tau(AdamW-scope)   <- this entry
    #   tau scope+schedule     Tau(alpha=0)          tau(w)
    #
    # Column contrast = adaptivity; row contrast = scope/schedule/implementation.
    'Tau(AdamW-scope)': {
        'rho':   {'low': 5e-7, 'high': 1.0, 'log_scale': True},
        'delta': {'low': 2e-3, 'high': 2.0, 'log_scale': True},
    },
    # ---- Robust-decay family (nearest-neighbour competitors, REVIEWER-1) ----
    # Every member is a DECOUPLED post-optimizer decay w <- w - shrink(w) whose shrink
    # is the gradient of a robust loss transplanted onto the parameters: quadratic near
    # the origin, saturating in the tails. They differ ONLY in how the transition is
    # made, which is exactly the axis on which tau-decay's novelty now rests:
    #   Huber-decay        piecewise transition at |w| = delta   (= AdamHD, Guo & Fan 2025)
    #   PseudoHuber-decay  smooth, algebraic  (Charbonnier)
    #   LogCosh-decay      smooth, exponential
    #   tau(w)             smooth, algebraic  (Fair) -- shrink saturates at lambda*delta
    # Shared (lambda, delta) parameterisation: shrink(w) ~ lambda*w near 0 for all four,
    # knee at |w| = delta. tau(w)'s native (eta, tau0, alpha) maps onto it exactly via
    # lambda = eta/tau0 and delta = tau0/alpha -- see robust_weight_decay().
    # lambda's upper bound is 1.0, not 1e-1, because that is the largest small-weight rate
    # tau(w) itself can reach (eta=1e-1 over tau0=0.1); capping the competitors at 1e-1
    # would hand tau(w) a whole decade of tuning range the competitors never see.
    'Huber-decay': {
        'decay_strength': {'low': 1e-6, 'high': 1.0, 'log_scale': True},
        'delta':          {'low': 2e-3, 'high': 2.0, 'log_scale': True},
    },
    'PseudoHuber-decay': {
        'decay_strength': {'low': 1e-6, 'high': 1.0, 'log_scale': True},
        'delta':          {'low': 2e-3, 'high': 2.0, 'log_scale': True},
    },
    'LogCosh-decay': {
        'decay_strength': {'low': 1e-6, 'high': 1.0, 'log_scale': True},
        'delta':          {'low': 2e-3, 'high': 2.0, 'log_scale': True},
    },
}
SEARCH_SPACES['Tau(w)'] = SEARCH_SPACES['\u03c4(w)']

# ---- Method categories (single source of truth for dispatch) ----
# Methods that add NO loss penalty: Baseline (unregularized), WD-tuned (handled in the
# optimizer), and the \u03c4 family (post-optimizer weight decay).
NO_LOSS_PENALTY = {'Baseline', 'WD-tuned', 'WD-tuned(weights)', '\u03c4(w)', 'Tau(w)', 'Tau(alpha=0)',
                   'Tau(AdamW-scope)',
                   'Huber-decay', 'PseudoHuber-decay', 'LogCosh-decay'}
# Methods that apply post-optimizer \u03c4(w) weight decay.
TAU_METHODS = {'\u03c4(w)', 'Tau(w)', 'Tau(alpha=0)'}
# Magnitude-adaptive decay carrying AdamW's scope and schedule instead of tau's: the
# fourth cell of the scope x adaptivity 2x2 (REVIEWER-5). Same shrinkage profile as
# tau(w), but applied to EVERY trainable parameter (LayerNorm and biases included) and
# scaled by the current learning rate, exactly as AdamW's decoupled decay is.
ADAMW_SCOPE_ADAPTIVE = {'Tau(AdamW-scope)'}
# The 2x2 that decomposes tau-decay's margin over tuned AdamW into a scope/schedule
# component and a magnitude-adaptivity component.
SCOPE_ADAPTIVITY_2X2 = {
    ('adamw', 'constant'): 'WD-tuned',
    ('adamw', 'adaptive'): 'Tau(AdamW-scope)',
    ('tau',   'constant'): 'Tau(alpha=0)',
    ('tau',   'adaptive'): '\u03c4(w)',
}
# Robust-decay competitors: same decoupled post-optimizer mechanism as \u03c4(w), different
# saturation profile. Maps method name -> shrink shape understood by robust_weight_decay().
ROBUST_DECAY_SHAPES = {
    'Huber-decay': 'huber',
    'PseudoHuber-decay': 'pseudo_huber',
    'LogCosh-decay': 'logcosh',
}
ROBUST_DECAY_METHODS = set(ROBUST_DECAY_SHAPES)
# Default 10-method roster under the Option-1 protocol.
DEFAULT_METHODS = ['Baseline', 'L1', 'L2', 'ElasticNet', 'SCAD', 'MCP', 'LSP',
                   'WD-tuned', 'Tau(alpha=0)', '\u03c4(w)']
# Core-6 roster for the "from-scratch AR-LM mechanism" claim: keeps only the
# load-bearing comparisons (true unregularised Baseline, tuned weight decay as the
# fair competitor, an L2/ElasticNet standard-penalty reference, tau(w), and the
# Tau(alpha=0) ablation that isolates magnitude-adaptivity). Drops the L1/SCAD/MCP/LSP
# sparsity penalties, which are orthogonal to the training-dynamics story and only
# add cost on the expensive pretrained/large-scale runs.
CORE_METHODS = ['Baseline', 'L2', 'ElasticNet', 'WD-tuned', 'Tau(alpha=0)', '\u03c4(w)']
# Roster for the robust-decay head-to-head demanded by REVIEWER-1: the nearest
# neighbours in the literature (Huber decay = AdamHD, plus the two other standard
# smooth robust losses) against \u03c4(w) and the two decoupled-decay controls. SCAD/MCP/LSP
# are deliberately absent \u2014 they are functionally much further away than these.
ROBUST_METHODS = ['Baseline', 'WD-tuned', 'Tau(alpha=0)', 'Huber-decay',
                  'PseudoHuber-decay', 'LogCosh-decay', '\u03c4(w)']
# Roster for the scope x adaptivity decomposition demanded by REVIEWER-5. The primary
# inferential contrast of the revised paper is tau(w) vs Tau(alpha=0) (adaptivity at
# matched implementation); WD-tuned vs Tau(alpha=0) measures scope+schedule; the fourth
# cell completes the factorial. Baseline anchors the whole thing.
TWO_BY_TWO_METHODS = ['Baseline', 'WD-tuned', 'Tau(AdamW-scope)', 'Tau(alpha=0)', '\u03c4(w)']


def optimizer_weight_decay(method, hyperparams):
    """Decoupled weight decay for the AdamW optimizer under the Option-1 protocol.

    Only the 'WD-tuned' competitor uses a (PSO-tuned) weight decay; every other method
    \u2014 including Baseline \u2014 uses 0.0, so weight decay is never silently stacked on top of
    a loss penalty or \u03c4(w) (fixes the non-uniform 3-regime issue, CODICE-1).
    """
    if method in ('WD-tuned', 'WD-tuned(weights)'):
        return hyperparams.get('wd', 0.0)
    return 0.0


def is_decay_scope_param(name, param):
    """The parameter scope of the decoupled-decay family (tau(w), Tau(alpha=0), the robust
    decays): every trainable parameter whose name contains 'weight' and that is not a
    LayerNorm module named as such - i.e. weight matrices, embeddings and, in models whose
    normalization layers are named otherwise, their gains; biases are excluded."""
    if 'weight' not in name or not param.requires_grad:
        return False
    if 'LayerNorm' in name or 'layer_norm' in name.lower():
        return False
    return True


def adamw_param_groups(model, method, hyperparams):
    """Parameter groups for torch.optim.AdamW under the Option-1 protocol.

    'WD-tuned'          -> one group, decay on every parameter (AdamW's default scope).
    'WD-tuned(weights)' -> decay only on is_decay_scope_param() parameters, 0 elsewhere.
    every other method  -> one group, weight_decay 0.0.
    """
    wd = optimizer_weight_decay(method, hyperparams)
    if method != 'WD-tuned(weights)':
        return [{'params': [p for p in model.parameters() if p.requires_grad], 'weight_decay': wd}]
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if is_decay_scope_param(name, p) else no_decay).append(p)
    return [{'params': decay, 'weight_decay': wd}, {'params': no_decay, 'weight_decay': 0.0}]


# Stability of the decay step (REVIEWER-3). The decay alone maps
#     w -> w * (1 - rho/(1 + |w|/delta)),
# so the multiplier stays in (0, 1] — a monotone contraction towards zero, never a sign
# flip — exactly when 0 < rho <= 1. Magnitude contraction alone would allow rho < 2, but
# rho in (1, 2) overshoots through zero and oscillates, which is not a relaxation and is
# not what "time constant" should mean. We therefore impose rho <= 1 as a hard bound of
# the search space (see SEARCH_SPACES: 'rho' high = 1.0) rather than leaving it implicit.
# NOTE: this bounds the DECAY step only. It says nothing about the stability of the
# optimizer as a whole, which depends on the learning rate and the loss curvature.
RHO_MAX_STABLE = 1.0


def check_rho_stable(rho, method='', strict=False):
    """Warn (or raise) when a decay rate leaves the monotone-contraction regime."""
    if rho is None or rho != rho:
        return rho
    if rho > RHO_MAX_STABLE:
        msg = (f"[STABILITY] rho={rho:.4g} > {RHO_MAX_STABLE} for {method or 'decay'}: "
               f"the decay step overshoots zero and flips sign each step "
               f"(w -> w*(1-rho/(1+|w|/delta)) with a negative multiplier for small |w|).")
        if strict:
            raise ValueError(msg)
        print(msg, flush=True)
    return rho


def canonical_decay_params(method, hyperparams):
    """Return the identifiable (rho, delta) pair for any member of the tau family.

    Accepts BOTH parameterisations so that historical hyperparameter dicts — the 90M
    transfer values in gpt2_wt103_standardized.py, results/wt103_sweep_choice.json, and
    every published result — keep reproducing exactly:

      canonical : {'rho', 'delta'}
      legacy    : {'decay_strength', 'tau0', 'tau_alpha'} with
                  rho = decay_strength/tau0 and delta = tau0/tau_alpha

    delta = inf denotes no magnitude adaptivity (the alpha = 0 ablation), for which the
    update degenerates to the constant decoupled decay w <- w - rho*w.
    """
    if 'rho' in hyperparams:
        rho = hyperparams['rho']
        if method == 'Tau(alpha=0)':
            delta = float('inf')
        else:
            delta = hyperparams.get('delta', float('inf'))
        return check_rho_stable(rho, method), delta

    eta = hyperparams.get('decay_strength', 0.0)
    tau0 = hyperparams.get('tau0', 0.5)
    alpha = tau_alpha_for(method, hyperparams)
    if tau0 == 0:
        raise ValueError(f"tau0 must be > 0 (method={method})")
    rho = eta / tau0
    delta = float('inf') if alpha == 0 else tau0 / alpha
    return check_rho_stable(rho, method), delta


def tau_alpha_for(method, hyperparams):
    """\u03c4(w) alpha (magnitude scaling) for a given method: 0.0 for the Tau(alpha=0)
    ablation, otherwise the tuned tau_alpha (default 10.0)."""
    if method == 'Tau(alpha=0)':
        return 0.0
    return hyperparams.get('tau_alpha', 10.0)


class EarlyStopping:
    """Patience-based early stopping that restores the best-validation checkpoint and
    verifies the restore.

    Two independent run-time checks guard the step on which every reported number
    depends:

    * mechanism check - `restore_best_model()` recomputes a fingerprint of the model's
      parameters after `load_state_dict` and compares it with the fingerprint taken when
      the snapshot was stored; a mismatch raises RuntimeError.
    * outcome check - `check_restored(value)` compares the validation metric re-evaluated
      on the restored model with `best_value`; they must agree within a tolerance. Every
      training loop calls it right after restoring, so the number a benchmark reports is
      demonstrably the best-epoch model's.
    """

    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        """
        Args:
            patience: numero di epoche senza miglioramento prima di fermarsi
            min_delta: minimo cambiamento per considerare un miglioramento
            mode: 'min' per loss, 'max' per accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.best_model_state = None
        self.best_fingerprint = None
        self.early_stop = False
        self.best_epoch = 0
        self.last_epoch = None
        self.restored = False

    def __call__(self, value, model, epoch):
        self.last_epoch = epoch
        if self.best_value is None:
            self._store_best(value, model, epoch)
            return False

        if self.mode == 'min':
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)

        if improved:
            self._store_best(value, model, epoch)
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop

    def _store_best(self, value, model, epoch):
        self.best_value = value
        self.best_model_state = _snapshot_state(model)
        # Fingerprint of the SNAPSHOT (not of the live model), so that the restore check
        # below verifies that the stored parameters are the ones loaded back.
        self.best_fingerprint = _state_fingerprint(self.best_model_state)
        self.best_epoch = epoch

    def restore_best_model(self, model):
        """Load the best snapshot into `model` and verify that it actually landed."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            now = _state_fingerprint(model.state_dict())
            if not _fingerprints_match(now, self.best_fingerprint):
                raise RuntimeError(
                    "EarlyStopping.restore_best_model: the restored parameters do not "
                    "match the snapshot taken at the best epoch "
                    f"(epoch {self.best_epoch + 1}). The checkpoint was not preserved "
                    "across training - refusing to report a number from this model.")
        self.restored = True
        return model

    def check_restored(self, recomputed_value, rel_tol=5e-3, abs_tol=1e-6, context=''):
        """Outcome assertion: the validation metric re-evaluated on the restored model
        must reproduce `best_value` (the value that defined the best epoch).

        Tolerance covers evaluation non-determinism (mixed precision, CUDA reductions)
        and is far below any post-best-epoch degradation of practical relevance.
        """
        if self.best_value is None:
            return
        if not self.restored:
            raise RuntimeError("check_restored() called before restore_best_model()")
        tol = max(abs_tol, rel_tol * abs(self.best_value))
        if abs(recomputed_value - self.best_value) > tol:
            raise RuntimeError(
                "EarlyStopping outcome check failed"
                + (f" [{context}]" if context else "") + ": validation metric of the "
                f"restored model = {recomputed_value:.6g} but best epoch "
                f"{self.best_epoch + 1} recorded {self.best_value:.6g} "
                f"(|diff| > {tol:.3g}). The reported model is NOT the best-epoch model.")
        return True

# Failures that mean "the machine is in trouble", not "these hyperparameters are bad".
# Scoring these as a bad objective value would quietly corrupt a hyperparameter search
# and still return a plausible-looking winner, so the PSO lets them propagate.
_FATAL_EVAL_ERRORS = tuple(
    e for e in (
        getattr(getattr(torch, 'cuda', None), 'OutOfMemoryError', None),
        getattr(torch, 'OutOfMemoryError', None),
        MemoryError, OSError,
    ) if isinstance(e, type)
)

# A search that scored more than this fraction of its evaluations as "worst" because they
# crashed is not a search; refuse to report a winner from it.
PSO_MAX_FAILURE_RATE = 0.25


def _snapshot_state(model):
    """Deep copy of a model's parameters, detached from the live tensors.

    `model.state_dict()` returns a dict whose values are the live parameter tensors, so
    a shallow copy would keep tracking the weights as the optimizer updates them. The
    checkpoint used by EarlyStopping must be immutable, hence the per-tensor clone.
    """
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _state_fingerprint(state):
    """Cheap, order-stable summary of a state dict: (sum, L2-norm) per tensor in float64.

    Used by EarlyStopping to prove that the parameters present after
    `restore_best_model()` are the ones that were snapshotted at the best epoch.
    Reductions run on the tensor's own device, no full copies.
    """
    out = []
    for k in sorted(state):
        v = state[k]
        if not torch.is_tensor(v) or v.numel() == 0:
            out.append((k, 0.0, 0.0))
            continue
        v = v.detach()
        if v.is_floating_point() or v.is_complex():
            vf = v.to(torch.float64)
            out.append((k, float(vf.sum()), float(torch.linalg.vector_norm(vf))))
        else:
            vi = v.to(torch.float64)
            out.append((k, float(vi.sum()), float(torch.linalg.vector_norm(vi))))
    return out


def _fingerprints_match(a, b, rel_tol=1e-6, abs_tol=1e-9):
    if a is None or b is None or len(a) != len(b):
        return False
    for (ka, sa, na), (kb, sb, nb) in zip(a, b):
        if ka != kb:
            return False
        if not (math.isclose(sa, sb, rel_tol=rel_tol, abs_tol=abs_tol)
                and math.isclose(na, nb, rel_tol=rel_tol, abs_tol=abs_tol)):
            return False
    return True


class RunJournal:
    """Append-only journal that makes a benchmark run resumable and never loses work.

    Without it, `run_benchmark` holds every PSO evaluation in memory and writes results
    only after the last method's last seed. Stopping the pod at hour 79 of 80 therefore
    discards all 80 hours — and at the 90M scale a single PSO evaluation is ~39 minutes,
    so even losing one method's tuning costs most of a day.

    The journal records three kinds of event as they happen, one JSON object per line:

        pso      a completed PSO evaluation, keyed by (method, exact params)
        best_hp  the winning hyperparameters for a method, once its PSO is finished
        eval     a completed final run, keyed by (method, seed)

    On restart the journal is replayed and each of those becomes a skip:

      * a `pso` hit returns the stored metrics instead of retraining. This is exact, not
        approximate: PSO is seeded from seeds[0], so a replay generates the same particle
        positions in the same order, and every lookup hits until the point of interruption.
      * a `best_hp` hit skips that method's tuning entirely.
      * an `eval` hit skips that (method, seed) run.

    Append-only and flushed per line, so a hard kill mid-write costs at most the record
    being written; a truncated final line is discarded on load. Two pods running different
    rosters write different journals (the path derives from the CSV name), so they never
    collide.
    """

    def __init__(self, path, enabled=True):
        self.path = path
        self.enabled = enabled
        self.pso = {}       # (method, params_key) -> metrics
        self.best_hp = {}   # method -> hyperparams
        self.evals = {}     # (method, seed) -> metrics
        self._fh = None
        self._replayed = 0
        if not enabled:
            return
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        if os.path.exists(path):
            self._load()
        self._fh = open(path, 'a', encoding='utf-8')

    @staticmethod
    def _key(params):
        """Stable, exact key for a hyperparameter dict (repr keeps full float precision)."""
        return json.dumps({k: repr(v) for k, v in sorted((params or {}).items())},
                          sort_keys=True)

    def _load(self):
        bad = 0
        with open(self.path, encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1          # truncated tail from a hard kill: ignore
                    continue
                kind = rec.get('kind')
                if kind == 'pso':
                    self.pso[(rec['method'], rec['params_key'])] = rec['metrics']
                elif kind == 'best_hp':
                    self.best_hp[rec['method']] = rec['hyperparams']
                elif kind == 'eval':
                    self.evals[(rec['method'], rec['seed'])] = rec['metrics']
                self._replayed += 1
        print(f"[JOURNAL] resumed {self.path}: {len(self.pso)} PSO evals, "
              f"{len(self.best_hp)} tuned methods, {len(self.evals)} final runs"
              + (f" ({bad} truncated record(s) ignored)" if bad else ""), flush=True)

    def _write(self, rec):
        if not self.enabled or self._fh is None:
            return
        self._fh.write(json.dumps(rec, ensure_ascii=False, default=float) + '\n')
        self._fh.flush()
        os.fsync(self._fh.fileno())

    # -- PSO evaluations ----------------------------------------------------
    def lookup_pso(self, method, params):
        return self.pso.get((method, self._key(params)))

    def record_pso(self, method, params, metrics):
        k = self._key(params)
        self.pso[(method, k)] = metrics
        self._write({'kind': 'pso', 'method': method, 'params_key': k,
                     'params': {kk: vv for kk, vv in (params or {}).items()},
                     'metrics': metrics, 'ts': time.time()})

    # -- PSO winners --------------------------------------------------------
    def lookup_best_hp(self, method):
        return self.best_hp.get(method)

    def record_best_hp(self, method, hyperparams):
        self.best_hp[method] = hyperparams
        self._write({'kind': 'best_hp', 'method': method,
                     'hyperparams': hyperparams, 'ts': time.time()})

    # -- final evaluation runs ---------------------------------------------
    def lookup_eval(self, method, seed):
        return self.evals.get((method, seed))

    def record_eval(self, method, seed, metrics):
        self.evals[(method, seed)] = metrics
        self._write({'kind': 'eval', 'method': method, 'seed': seed,
                     'metrics': metrics, 'ts': time.time()})

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class ExperimentTracker:
    """Traccia e salva risultati degli esperimenti"""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.results = {}
        self.hyperparams = {}
        self.start_time = time.time()

    def set_hyperparams(self, method: str, hyperparams: dict):
        """Record the hyperparameters used for a method (PSO winner or fixed)."""
        self.hyperparams[method] = dict(hyperparams) if hyperparams else {}

    def add_result(self, method: str, run_idx: int, metrics: dict):
        """Aggiungi risultato di un singolo run"""
        if method not in self.results:
            self.results[method] = []
        self.results[method].append({
            'run': run_idx,
            'metrics': metrics
        })

    def get_statistics(self, method: str, metric_name: str) -> dict:
        """Calcola statistiche per un metodo e metrica"""
        if method not in self.results:
            return {}

        values = [r['metrics'][metric_name] for r in self.results[method]]

        mean = np.mean(values)
        # Sample std (ddof=1): unbiased estimator of the population std from n seeds.
        # Previously ddof=0 underestimated variance by (n-1)/n, inflating |t|/|d| and
        # shrinking p-values (~12% at n=5) → systematic over-statement of significance
        # (CODICE-6). ddof=1 is what Welch's t-test / Cohen's d downstream assume.
        std = np.std(values, ddof=1) if len(values) > 1 else 0.0

        # Confidence interval 95%
        if len(values) > 1:
            ci = stats.t.interval(
                0.95,
                len(values) - 1,
                loc=mean,
                scale=stats.sem(values)
            )
        else:
            ci = (mean, mean)

        return {
            'mean': mean,
            'std': std,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'values': values,
            'n_runs': len(values)
        }

    def compare_methods(self, method_a: str, method_b: str, metric_name: str) -> dict:
        """Confronto statistico tra due metodi"""
        if method_a not in self.results or method_b not in self.results:
            return {}

        values_a = [r['metrics'][metric_name] for r in self.results[method_a]]
        values_b = [r['metrics'][metric_name] for r in self.results[method_b]]

        # Welch's t-test (equal_var=False): does not assume equal variances, matching
        # the paper's stated test and stats_utils.welch_ttest. Previously used Student's
        # t-test (equal_var defaulted True), inconsistent with the paper (CODICE-7).
        t_stat, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)

        # Effect size (Cohen's d), sample std (ddof=1) to match get_statistics (CODICE-6)
        pooled_std = np.sqrt((np.std(values_a, ddof=1)**2
                              + np.std(values_b, ddof=1)**2) / 2)
        cohens_d = (np.mean(values_a) - np.mean(values_b)) / pooled_std if pooled_std > 0 else 0

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant_05': p_value < 0.05,
            'significant_01': p_value < 0.01
        }

    def save_to_csv(self, filepath: str):
        """Salva risultati aggregati in CSV"""
        # Create results directory if it doesn't exist
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)

        # Add results/ prefix if not already there
        if not filepath.startswith(results_dir + '/'):
            filepath = os.path.join(results_dir, os.path.basename(filepath))

        rows = []

        for method in self.results:
            # Get all metric names from first run
            if self.results[method]:
                metric_names = self.results[method][0]['metrics'].keys()

                row = {'method': method}
                for metric in metric_names:
                    stats_dict = self.get_statistics(method, metric)
                    row[f'{metric}_mean'] = stats_dict.get('mean', '')
                    row[f'{metric}_std'] = stats_dict.get('std', '')
                    row[f'{metric}_ci_lower'] = stats_dict.get('ci_lower', '')
                    row[f'{metric}_ci_upper'] = stats_dict.get('ci_upper', '')

                rows.append(row)

        if rows:
            fieldnames = rows[0].keys()
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"[SAVE] Results saved to {filepath}")

    def save_to_json(self, filepath: str):
        """Salva risultati completi in JSON"""
        # Create results directory if it doesn't exist
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)

        # Add results/ prefix if not already there
        if not filepath.startswith(results_dir + '/'):
            filepath = os.path.join(results_dir, os.path.basename(filepath))

        output = {
            'experiment_name': self.experiment_name,
            'total_time': time.time() - self.start_time,
            'best_hyperparams': self.hyperparams,
            'results': self.results
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"[SAVE] Detailed results saved to {filepath}")

    def print_summary(self, metric_name: str, mode='max'):
        """Stampa summary table ordinata per metrica"""
        print(f"\n{'='*80}")
        print(f"SUMMARY: {self.experiment_name} - {metric_name}")
        print(f"{'='*80}")

        summary = []
        for method in self.results:
            stats_dict = self.get_statistics(method, metric_name)
            summary.append({
                'method': method,
                'mean': stats_dict.get('mean', 0),
                'std': stats_dict.get('std', 0),
                'ci_lower': stats_dict.get('ci_lower', 0),
                'ci_upper': stats_dict.get('ci_upper', 0)
            })

        # Sort by metric
        summary.sort(key=lambda x: x['mean'], reverse=(mode == 'max'))

        # Print table
        print(f"{'Method':<15} {'Mean':<12} {'Std':<12} {'95% CI':<25}")
        print(f"{'-'*80}")
        for item in summary:
            ci_str = f"[{item['ci_lower']:.4f}, {item['ci_upper']:.4f}]"
            print(f"{item['method']:<15} {item['mean']:<12.4f} {item['std']:<12.4f} {ci_str:<25}")
        print(f"{'='*80}\n")


# ============================
# PARTICLE SWARM OPTIMIZATION
# ============================

PSO_BUDGETS = {
    # 'light' shrunk from 5×8=40 to 3×3=9 evals after the audit re-run showed each
    # eval costs a full training pass — on cheap benchmarks (MLP/CNN on MNIST/CIFAR,
    # ViT-mini, BERT-tiny) this cut wall-clock per method from ~2 h to ~30 min while
    # still resolving the small search spaces of the WD-tuned and Tau(alpha=0)
    # competitors introduced by the Opzione 1 protocol.
    'light':    {'n_particles': 3,  'n_iterations': 3,  'patience': 2},
    'standard': {'n_particles': 8,  'n_iterations': 10, 'patience': 3},
    'thorough': {'n_particles': 10, 'n_iterations': 16, 'patience': 4},
    # 'auto' is resolved per-method by dimension_aware_budget(): 12 evaluations for a
    # 1-D search (L2, WD-tuned, Tau(alpha=0)), 40 for every >=2-D search (tau(w),
    # ElasticNet, SCAD/MCP/LSP, the robust decays, Tau(AdamW-scope)). The entry below
    # is only a nominal placeholder for the wall-clock ESTIMATE at the benchmark level;
    # the real budget is computed in find_best_hyperparams once the search space (and
    # thus n_dims) is known.
    'auto':     {'n_particles': 8,  'n_iterations': 5,  'patience': 5},
}


# Evaluations per search-space dimensionality under the 'auto' budget.
#   0-D (Baseline)            -> no tuning
#   1-D (L1/L2/WD-tuned/rho)  -> 4 particles x 3 iterations = 12
#   >=2-D (every other method)-> 8 particles x 5 iterations = 40
# 40 evaluations for every >=2-D search (tau(w) in its canonical 2-D (rho, delta) form,
# ElasticNet, SCAD/MCP/LSP, the robust decays, Tau(AdamW-scope)) gives each of them the
# same tuning effort, so no method receives more budget than a competitor of equal
# dimensionality. The 1-D searches are saturated well below 12 evaluations (a
# log-scaled scalar).
PSO_AUTO_EVALS = {1: (4, 3), 2: (8, 5)}


def dimension_aware_budget(n_dims: int, evals_per_dim: int = None) -> dict:
    """PSO budget scaled to the search-space dimensionality (see PSO_AUTO_EVALS).

    The swarm's own early-stopping `patience` is set to the number of iterations, i.e.
    disabled: an 'auto' budget means exactly n_particles * n_iterations evaluations, so
    the budget reported in the paper's Appendix A.1 is the budget that was spent.
    `evals_per_dim` is accepted for backward compatibility and ignored.
    """
    if n_dims <= 0:                      # empty search space (Baseline): no tuning
        return {'n_particles': 1, 'n_iterations': 1, 'patience': 1}
    n_particles, n_iterations = PSO_AUTO_EVALS[min(n_dims, max(PSO_AUTO_EVALS))]
    return {'n_particles': n_particles, 'n_iterations': n_iterations,
            'patience': n_iterations}


class ParticleSwarmOptimizer:
    """
    Particle Swarm Optimization for hyperparameter tuning.
    Supports mixed log-scale and linear-scale parameters.
    """

    def __init__(
        self,
        search_space: dict[str, dict],
        n_particles: int = 8,
        n_iterations: int = 10,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        mode: str = 'min',
        patience: int = 3,
        seed: int = 42,
        verbose: bool = True
    ):
        self.search_space = search_space
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.mode = mode
        self.patience = patience
        self.seed = seed
        self.verbose = verbose
        # Count of evaluations that raised and were scored "worst". A search that ends
        # with n_failures > 0 produced a winner chosen partly from non-evidence, so the
        # caller must decide whether to trust it (see find_best_hyperparams).
        self.n_failures = 0

        self.param_names = list(search_space.keys())
        self.n_dims = len(self.param_names)

        self.lower = np.array([
            np.log10(search_space[p]['low']) if search_space[p]['log_scale']
            else search_space[p]['low']
            for p in self.param_names
        ])
        self.upper = np.array([
            np.log10(search_space[p]['high']) if search_space[p]['log_scale']
            else search_space[p]['high']
            for p in self.param_names
        ])

    def _decode_position(self, position: np.ndarray) -> dict[str, float]:
        params = {}
        for i, name in enumerate(self.param_names):
            if self.search_space[name]['log_scale']:
                params[name] = 10 ** position[i]
            else:
                params[name] = position[i]
        return params

    def optimize(self, objective_fn: Callable[[dict[str, float]], float]) -> tuple[dict[str, float], float]:
        rng = np.random.RandomState(self.seed)

        positions = rng.uniform(
            self.lower, self.upper,
            size=(self.n_particles, self.n_dims)
        )
        v_max = 0.2 * (self.upper - self.lower)
        velocities = rng.uniform(
            -v_max, v_max,
            size=(self.n_particles, self.n_dims)
        )

        worst = np.inf if self.mode == 'min' else -np.inf
        p_best_positions = positions.copy()
        p_best_scores = np.full(self.n_particles, worst)
        g_best_position = positions[0].copy()
        g_best_score = worst

        no_improve = 0
        total_evals = 0

        for iteration in range(self.n_iterations):
            improved = False
            for p in range(self.n_particles):
                params = self._decode_position(positions[p])
                try:
                    score = objective_fn(params)
                except _FATAL_EVAL_ERRORS:
                    # Infrastructure failures (CUDA OOM, out of disk, interrupted run) are
                    # NOT evidence that these hyperparameters are bad. Scoring them as
                    # "worst" would silently corrupt the search and still hand back a
                    # confident-looking winner, so they propagate and stop the benchmark.
                    raise
                except Exception as e:
                    # A genuinely divergent configuration: score it worst, but never
                    # silently. This used to be printed only when verbose, i.e. never in
                    # the --quiet mode used for long unattended runs.
                    self.n_failures += 1
                    params_str = ", ".join(f"{k}={v:.4g}" for k, v in params.items())
                    print(f"  [PSO][FAIL {self.n_failures}] {type(e).__name__} at "
                          f"{params_str}: {e} -> scored as worst", flush=True)
                    score = worst
                total_evals += 1

                is_better = (score < p_best_scores[p]) if self.mode == 'min' else (score > p_best_scores[p])
                if is_better:
                    p_best_scores[p] = score
                    p_best_positions[p] = positions[p].copy()

                is_global = (score < g_best_score) if self.mode == 'min' else (score > g_best_score)
                if is_global:
                    g_best_score = score
                    g_best_position = positions[p].copy()
                    improved = True

            if self.verbose:
                best_params = self._decode_position(g_best_position)
                params_str = ", ".join(f"{k}={v:.4g}" for k, v in best_params.items())
                print(f"  [PSO] Iter {iteration+1}/{self.n_iterations}: "
                      f"best={g_best_score:.6f} ({params_str})")

            if not improved:
                no_improve += 1
                if no_improve >= self.patience:
                    if self.verbose:
                        print(f"  [PSO] Early stop after {iteration+1} iterations ({total_evals} evals)")
                    break
            else:
                no_improve = 0

            # Update velocities and positions
            r1 = rng.uniform(0, 1, (self.n_particles, self.n_dims))
            r2 = rng.uniform(0, 1, (self.n_particles, self.n_dims))
            velocities = (self.w * velocities
                          + self.c1 * r1 * (p_best_positions - positions)
                          + self.c2 * r2 * (g_best_position - positions))
            velocities = np.clip(velocities, -v_max, v_max)
            positions = positions + velocities

            # Reflective boundaries
            for d in range(self.n_dims):
                below = positions[:, d] < self.lower[d]
                above = positions[:, d] > self.upper[d]
                positions[below, d] = self.lower[d]
                positions[above, d] = self.upper[d]
                velocities[below, d] *= -0.5
                velocities[above, d] *= -0.5

        best_params = self._decode_position(g_best_position)
        if self.verbose:
            params_str = ", ".join(f"{k}={v:.4g}" for k, v in best_params.items())
            print(f"  [PSO] Done: {total_evals} evals, best={g_best_score:.6f} ({params_str})")
        return best_params, g_best_score


def find_best_hyperparams(
    train_fn: Callable[[dict[str, float]], dict],
    method: str,
    metric_name: str = 'val_loss',
    mode: str = 'min',
    search_space: dict | None = None,
    pso_budget: str = 'standard',
    seed: int = 42,
    verbose: bool = True
) -> dict[str, float]:
    """
    Find best hyperparameters for a regularization method using PSO.

    Args:
        train_fn: function that accepts hyperparams dict and returns metrics dict
        method: regularization method name
        metric_name: metric to optimize
        mode: 'min' or 'max'
        search_space: override default search space
        pso_budget: 'light', 'standard', or 'thorough'
        seed: random seed for PSO
        verbose: print progress

    Returns:
        best hyperparameters dict
    """
    if method == 'Baseline':
        return {}

    space_key = '\u03c4(w)' if method == 'Tau(w)' else method
    if search_space is None:
        search_space = SEARCH_SPACES[space_key]

    if verbose:
        print(f"\n[PSO] Finding best hyperparameters for {method}...")

    if pso_budget == 'auto':
        budget = dimension_aware_budget(len(search_space))
        if verbose:
            print(f"  [PSO] auto budget for {len(search_space)}-D {method}: "
                  f"{budget['n_particles']}x{budget['n_iterations']}="
                  f"{budget['n_particles'] * budget['n_iterations']} evals")
    else:
        budget = PSO_BUDGETS[pso_budget]

    pso = ParticleSwarmOptimizer(
        search_space=search_space,
        mode=mode,
        seed=seed,
        verbose=verbose,
        **budget
    )

    def objective(params):
        return train_fn(params)[metric_name]

    best_params, best_score = pso.optimize(objective)

    n_evals = budget['n_particles'] * budget['n_iterations']
    if pso.n_failures:
        rate = pso.n_failures / max(n_evals, 1)
        msg = (f"[PSO] {method}: {pso.n_failures}/{n_evals} evaluations "
               f"({rate:.0%}) crashed and were scored as worst")
        if rate > PSO_MAX_FAILURE_RATE:
            raise RuntimeError(
                msg + f" — above the {PSO_MAX_FAILURE_RATE:.0%} tolerance. The winner "
                      f"would be selected mostly from failures rather than from "
                      f"measurements, so no hyperparameters are reported. Fix the "
                      f"underlying error and resume; completed evaluations are journaled "
                      f"and will not be repeated.")
        print(msg + " — winner reported, but treat this tuning as suspect.", flush=True)

    if verbose:
        params_str = ", ".join(f"{k}={v:.4g}" for k, v in best_params.items())
        print(f"[PSO] Best for {method}: {params_str} ({metric_name}={best_score:.6f})")

    return best_params


def set_seed(seed: int):
    """Set seed per riproducibilità completa (incluso determinismo GPU)"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate_model(model, data_loader, loss_fn, device):
    """Valuta modello su un dataset"""
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            total_count += inputs.size(0)

    return total_loss / total_count if total_count > 0 else 0.0


def evaluate_classification(model, data_loader, device):
    """Valuta accuracy e loss per classificazione"""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            total_loss += loss.item() * targets.size(0)

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0
    return accuracy, avg_loss


def measure_sparsity(model, threshold=1e-3):
    """Misura sparsità dei pesi"""
    all_weights = []
    for p in model.parameters():
        if p.requires_grad:
            all_weights.append(p.detach().cpu().flatten())

    if not all_weights:
        return 0.0, 0, 0

    all_weights = torch.cat(all_weights)
    total = all_weights.numel()
    small = (all_weights.abs() < threshold).sum().item()
    sparsity = small / total if total > 0 else 0.0

    return sparsity, total, small


def weight_magnitude_stats(model, threshold=1e-3, transformer=True):
    """Per-epoch weight-magnitude statistics for mechanism analysis (Workstream 2).

    Computed over the SAME parameter set the regularizers act on (transformer path:
    'weight' params excluding LayerNorm/bias; non-transformer: all requires_grad params),
    so the numbers are directly comparable to what τ(w)/penalties actually shrink.

    Returns a dict: mean/median/max |w|, L2 norm, fraction of |w| below `threshold`.
    These trajectories reveal *what magnitude-adaptive decay does to the weight
    distribution over training* (small-weight suppression, tail preservation) and,
    together with train/val perplexity, the overfitting-delay signature of τ(w).
    """
    with torch.no_grad():
        ws = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if transformer:
                if 'weight' not in name:
                    continue
                if 'LayerNorm' in name or 'layer_norm' in name.lower():
                    continue
            ws.append(p.detach().abs().flatten().float().cpu())
        if not ws:
            return {}
        w = torch.cat(ws)
        total = w.numel()
        return {
            'mean_abs_w': float(w.mean()),
            'median_abs_w': float(w.median()),
            'max_abs_w': float(w.max()),
            'l2_norm': float(torch.linalg.vector_norm(w)),
            'frac_below_thr': float((w < threshold).sum().item() / total) if total else 0.0,
            'n_params_tracked': int(total),
        }


def magnitude_prune(model, threshold=None, sparsity_target=None,
                    mode='weight', param_filter=None):
    """
    Post-training magnitude pruning (in-place).

    Args:
        model: the trained nn.Module to prune
        threshold: absolute |w| threshold; weights with |w| < threshold -> 0
        sparsity_target: float in (0, 1); prune the bottom fraction by magnitude
            globally across all eligible parameters. Ignored if threshold is set.
        mode: 'weight' (element-wise) or 'neuron' (row-wise by L2 norm).
            In 'neuron' mode the threshold/target applies to per-row norms
            of 2D weight matrices; rows below the cutoff are zeroed entirely.
        param_filter: optional callable (name, param) -> bool to include a
            parameter in pruning. Defaults to 2D weight tensors only
            (skips biases, LayerNorm, embeddings).

    Returns:
        dict with achieved_sparsity, zeroed_params, total_params, threshold_used.
    """
    if threshold is None and sparsity_target is None:
        raise ValueError("Provide either threshold or sparsity_target")
    if mode not in ('weight', 'neuron'):
        raise ValueError(f"mode must be 'weight' or 'neuron', got {mode}")

    if param_filter is None:
        def param_filter(name, p):
            return p.requires_grad and p.dim() >= 2 and 'embed' not in name.lower()

    targets = [(n, p) for n, p in model.named_parameters() if param_filter(n, p)]
    if not targets:
        return {'achieved_sparsity': 0.0, 'zeroed_params': 0,
                'total_params': 0, 'threshold_used': 0.0}

    if mode == 'weight':
        if threshold is None:
            flat = torch.cat([p.detach().abs().flatten() for _, p in targets])
            k = int(sparsity_target * flat.numel())
            threshold = torch.kthvalue(flat, max(k, 1)).values.item() if k > 0 else 0.0

        zeroed = 0
        total = 0
        with torch.no_grad():
            for _, p in targets:
                mask = p.abs() >= threshold
                p.mul_(mask)
                zeroed += (~mask).sum().item()
                total += p.numel()

    else:  # neuron mode: row-wise L2 norm on 2D weights
        row_norms = []
        for _, p in targets:
            if p.dim() == 2:
                row_norms.append(p.detach().norm(dim=1).flatten())
        if not row_norms:
            return {'achieved_sparsity': 0.0, 'zeroed_params': 0,
                    'total_params': 0, 'threshold_used': 0.0}

        if threshold is None:
            flat_norms = torch.cat(row_norms)
            k = int(sparsity_target * flat_norms.numel())
            threshold = torch.kthvalue(flat_norms, max(k, 1)).values.item() if k > 0 else 0.0

        zeroed = 0
        total = 0
        with torch.no_grad():
            for _, p in targets:
                if p.dim() != 2:
                    total += p.numel()
                    continue
                norms = p.norm(dim=1)
                dead_rows = norms < threshold
                p[dead_rows] = 0.0
                zeroed += dead_rows.sum().item() * p.size(1)
                total += p.numel()

    return {
        'achieved_sparsity': zeroed / total if total > 0 else 0.0,
        'zeroed_params': zeroed,
        'total_params': total,
        'threshold_used': float(threshold),
    }


# ============================
# REGULARIZATION PENALTY FUNCTIONS
# ============================

def l1_penalty(model):
    """L1 regularization (Lasso) - promotes sparsity"""
    l1 = 0.0
    for p in model.parameters():
        if p.requires_grad:
            l1 += p.abs().sum()
    return l1


def l2_penalty(model):
    """L2 regularization (Ridge) - weight decay"""
    l2 = 0.0
    for p in model.parameters():
        if p.requires_grad:
            l2 += (p ** 2).sum()
    return l2


def scad_penalty(model, lambda_scad, a=3.7):
    """SCAD penalty (Smoothly Clipped Absolute Deviation)"""
    penalty = 0.0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        w = p.view(-1)
        abs_w = w.abs()

        term1 = torch.where(abs_w <= lambda_scad, lambda_scad * abs_w, torch.zeros_like(abs_w))
        term2 = torch.where(
            (abs_w > lambda_scad) & (abs_w <= a * lambda_scad),
            (-abs_w**2 + 2*a*lambda_scad*abs_w - lambda_scad**2) / (2*(a-1)),
            torch.zeros_like(abs_w)
        )
        term3 = torch.where(
            abs_w > a * lambda_scad,
            torch.full_like(abs_w, (a+1)*lambda_scad**2 / 2),
            torch.zeros_like(abs_w)
        )

        penalty += term1.sum() + term2.sum() + term3.sum()
    return penalty


def mcp_penalty(model, lambda_mcp, gamma=3.0):
    """MCP penalty (Minimax Concave Penalty)"""
    penalty = 0.0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        w = p.view(-1)
        abs_w = w.abs()
        term = torch.where(
            abs_w < gamma * lambda_mcp,
            lambda_mcp * abs_w - (abs_w**2) / (2 * gamma),
            torch.full_like(abs_w, lambda_mcp**2 * gamma / 2)
        )
        penalty += term.sum()
    return penalty


def lsp_penalty(model, lambda_lsp, theta=1e-3):
    """LSP penalty (Log-Sum Penalty)"""
    penalty = 0.0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        w = p.view(-1)
        penalty += torch.log(1 + (w.abs()/theta)).sum()
    return lambda_lsp * penalty


def tau_weight_decay(model, decay_strength, tau0=0.5, alpha=10.0):
    """
    τ(w) weight-dependent regularization

    Applies decay: p.data -= decay_strength * p.data / (tau0 + alpha * |p.data|)
    Should be called AFTER optimizer.step()
    """
    if decay_strength == 0.0:
        return

    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad and p.data.numel() > 0:
                tau = tau0 + alpha * p.data.abs()
                decay = decay_strength * p.data / tau
                p.data -= decay


def robust_shrink(w, decay_strength, delta, shape):
    """Shrinkage vector of the robust-decay family, in the shared (lambda, delta) form.

    All members are gradients of a robust loss rho transplanted from residuals onto
    parameters, i.e. shrink(w) = lambda * rho'(w; delta). They agree to first order at
    the origin (shrink ~ lambda*w for |w| << delta) and saturate for |w| >> delta; they
    differ only in the transition:

        'huber'         lambda * w * min(1, delta/|w|)        piecewise (AdamHD)
        'pseudo_huber'  lambda * w / sqrt(1 + (w/delta)^2)    smooth, algebraic
        'logcosh'       lambda * delta * tanh(w/delta)        smooth, exponential
        'fair'          lambda * w / (1 + |w|/delta)          smooth, algebraic (= tau(w))

    The 'fair' branch is exactly tau_weight_decay() under the reparameterisation
    lambda = eta/tau0, delta = tau0/alpha, and is provided so that the four profiles can
    be driven through one code path in the head-to-head.
    """
    abs_w = w.abs()
    if shape == 'huber':
        # grad of the Huber loss: identity inside the quadratic bowl, clipped outside.
        return decay_strength * torch.clamp(w, min=-delta, max=delta)
    if shape == 'pseudo_huber':
        return decay_strength * w / torch.sqrt(1.0 + (w / delta) ** 2)
    if shape == 'logcosh':
        return decay_strength * delta * torch.tanh(w / delta)
    if shape == 'fair':
        return decay_strength * w / (1.0 + abs_w / delta)
    raise ValueError(f"Unknown robust-decay shape: {shape!r}")


def robust_weight_decay(model, decay_strength, delta, shape):
    """Decoupled robust weight decay for non-transformer models.

    Same mechanism and call site as tau_weight_decay() (AFTER optimizer.step()), with a
    different saturation profile. See robust_shrink() for the shapes.
    """
    if decay_strength == 0.0:
        return

    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad and p.data.numel() > 0:
                p.data -= robust_shrink(p.data, decay_strength, delta, shape)


def get_regularization(model, method, hyperparams):
    """Compute regularization penalty for non-transformer models.

    Returns penalty term to add to loss. Returns 0 for Baseline and τ(w).
    """
    if method in NO_LOSS_PENALTY:
        return 0.0

    lambda_val = hyperparams.get('lambda_val', 0.0)
    if lambda_val == 0:
        return 0.0

    if method == 'L1':
        return lambda_val * l1_penalty(model)
    elif method == 'L2':
        return lambda_val * l2_penalty(model)
    elif method == 'ElasticNet':
        alpha = hyperparams.get('en_alpha', 0.5)
        return lambda_val * (alpha * l1_penalty(model) + (1 - alpha) * l2_penalty(model))
    elif method == 'SCAD':
        return scad_penalty(model, lambda_val, a=hyperparams.get('scad_a', 3.7))
    elif method == 'MCP':
        return mcp_penalty(model, lambda_val, gamma=hyperparams.get('mcp_gamma', 3.0))
    elif method == 'LSP':
        return lsp_penalty(model, lambda_val, theta=hyperparams.get('lsp_theta', 1e-3))
    return 0.0


def apply_tau_if_needed(model, method, hyperparams, lr_scale=1.0):
    """Apply the post-optimizer decoupled decay for whichever decay family `method` is in.

    Covers the τ family (τ(w), Tau(alpha=0)) in the canonical (rho, delta) form, the
    AdamW-scope adaptive cell, and the robust-decay competitors (Huber/pseudo-Huber/
    log-cosh); a no-op for every loss-penalty method. Name kept for backwards
    compatibility with the existing call sites.

    lr_scale multiplies the decay, and is only ever != 1.0 for the AdamW-scope cell,
    which by construction couples its decay to the learning-rate schedule.
    """
    if method in TAU_METHODS:
        rho, delta = canonical_decay_params(method, hyperparams)
        robust_weight_decay(model, rho, delta=delta, shape='fair')
    elif method in ADAMW_SCOPE_ADAPTIVE:
        rho, delta = canonical_decay_params(method, hyperparams)
        robust_weight_decay(model, rho * lr_scale, delta=delta, shape='fair')
    elif method in ROBUST_DECAY_METHODS:
        robust_weight_decay(
            model,
            check_rho_stable(hyperparams.get('decay_strength', 0.0), method),
            delta=hyperparams.get('delta', 0.1),
            shape=ROBUST_DECAY_SHAPES[method]
        )


# ============================
# TRANSFORMER-SPECIFIC REGULARIZATION
# ============================

def get_regularization_transformer(model, method, lambda_val, device, extra_params=None):
    """
    Compute regularization penalty for transformer models.
    Excludes LayerNorm layers (both 'LayerNorm' and 'layer_norm' patterns).

    Args:
        model: transformer model
        method: one of 'Baseline', 'L1', 'L2', 'ElasticNet', 'SCAD', 'MCP', 'LSP', 'τ(w)', 'Tau(w)'
        lambda_val: regularization strength
        device: torch device
        extra_params: dict of method-specific hyperparameters (e.g. {'scad_a': 3.7})

    Returns:
        regularization term (tensor)
    """
    if extra_params is None:
        extra_params = {}

    reg = torch.tensor(0.0, device=device)

    if method == 'Baseline' or lambda_val == 0:
        return reg

    for name, param in model.named_parameters():
        # Skip non-weight params and LayerNorm layers
        if 'weight' not in name or not param.requires_grad:
            continue
        if 'LayerNorm' in name or 'layer_norm' in name.lower():
            continue

        if method == 'L1':
            reg += lambda_val * torch.sum(torch.abs(param))
        elif method == 'L2':
            reg += lambda_val * torch.sum(param ** 2)
        elif method == 'ElasticNet':
            alpha = extra_params.get('en_alpha', 0.5)
            reg += lambda_val * (alpha * torch.sum(torch.abs(param))
                                 + (1 - alpha) * torch.sum(param ** 2))
        elif method == 'SCAD':
            a = extra_params.get('scad_a', 3.7)
            abs_w = torch.abs(param)
            reg += torch.sum(torch.where(
                abs_w <= lambda_val,
                lambda_val * abs_w,
                torch.where(
                    abs_w <= a * lambda_val,
                    -(abs_w ** 2 - 2 * a * lambda_val * abs_w + lambda_val ** 2) / (2 * (a - 1)),
                    torch.full_like(abs_w, (a + 1) * lambda_val ** 2 / 2)
                )
            ))
        elif method == 'MCP':
            gamma = extra_params.get('mcp_gamma', 3.0)
            abs_w = torch.abs(param)
            reg += torch.sum(torch.where(
                abs_w <= gamma * lambda_val,
                lambda_val * abs_w - abs_w ** 2 / (2 * gamma),
                torch.full_like(abs_w, gamma * lambda_val ** 2 / 2)
            ))
        elif method == 'LSP':
            theta = extra_params.get('lsp_theta', 1e-3)
            reg += lambda_val * torch.sum(torch.log(1 + torch.abs(param) / theta))
        elif method in ['\u03c4(w)', 'Tau(w)']:
            # τ(w) uses weight decay, not loss penalty
            pass

    # λ is now applied exactly once per method above. SCAD/MCP already carry λ
    # internally (λ is both threshold and multiplier in their definitions), so they
    # are NOT re-multiplied here — fixes the previous λ² double-count (CODICE-3),
    # matching the non-transformer get_regularization() convention.
    return reg


def apply_tau_weight_decay_transformer(model, decay_strength, tau0=0.5, alpha=10.0):
    """
    Apply τ(w) weight-dependent decay for transformer models.
    Excludes LayerNorm layers (both 'LayerNorm' and 'layer_norm' patterns).
    Should be called AFTER optimizer.step().

    Args:
        model: transformer model
        decay_strength: decay strength (lambda)
        tau0: base tau value (default 0.5)
        alpha: scaling factor (default 10.0)
    """
    if decay_strength == 0.0:
        return

    with torch.no_grad():
        for name, param in model.named_parameters():
            # Skip non-weight params and LayerNorm layers
            if 'weight' not in name or not param.requires_grad:
                continue
            if 'LayerNorm' in name or 'layer_norm' in name.lower():
                continue

            tau = tau0 + alpha * param.data.abs()
            decay = decay_strength * param.data / tau
            param.data -= decay


def apply_robust_weight_decay_transformer(model, decay_strength, delta, shape):
    """Decoupled robust weight decay for transformer models.

    Identical scope and call site to apply_tau_weight_decay_transformer() — weight
    matrices only, LayerNorm excluded, applied AFTER optimizer.step() — so that the
    saturation profile is the ONLY difference between τ(w) and the Huber/pseudo-Huber/
    log-cosh competitors. See robust_shrink().
    """
    if decay_strength == 0.0:
        return

    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' not in name or not param.requires_grad:
                continue
            if 'LayerNorm' in name or 'layer_norm' in name.lower():
                continue

            param.data -= robust_shrink(param.data, decay_strength, delta, shape)


def apply_adaptive_decay_adamw_scope(model, rho, delta, lr_scale=1.0):
    """Magnitude-adaptive decay with AdamW's scope and schedule (2x2 cell, REVIEWER-5).

    Same Fair shrinkage profile as τ(w), but deliberately NOT τ(w)'s implementation:
      * applied to EVERY trainable parameter, LayerNorm and biases included, exactly as
        torch.optim.AdamW's weight_decay is;
      * scaled by the current learning rate (lr_scale = lr_t / lr_0), so the decay
        follows the warmup/decay schedule instead of running at a constant per-step rate.

    Contrasting this with τ(w) isolates scope+schedule; contrasting it with WD-tuned
    isolates magnitude adaptivity at AdamW's scope.
    """
    if rho == 0.0 or lr_scale == 0.0:
        return
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad and p.data.numel() > 0:
                p.data -= robust_shrink(p.data, rho * lr_scale, delta, 'fair')


def apply_decoupled_decay_transformer(model, method, hyperparams, lr_scale=1.0):
    """Single dispatch for every post-optimizer decoupled decay on transformers.

    τ family -> Fair shrinkage at τ's scope; Tau(AdamW-scope) -> Fair shrinkage at
    AdamW's scope and schedule; robust family -> the corresponding profile at τ's scope;
    anything else -> no-op.

    lr_scale (lr_t / lr_0) is used only by the AdamW-scope cell, whose defining feature
    is that its decay is coupled to the learning-rate schedule.
    """
    if method in TAU_METHODS:
        rho, delta = canonical_decay_params(method, hyperparams)
        apply_robust_weight_decay_transformer(model, rho, delta=delta, shape='fair')
    elif method in ADAMW_SCOPE_ADAPTIVE:
        rho, delta = canonical_decay_params(method, hyperparams)
        apply_adaptive_decay_adamw_scope(model, rho, delta, lr_scale=lr_scale)
    elif method in ROBUST_DECAY_METHODS:
        apply_robust_weight_decay_transformer(
            model,
            check_rho_stable(hyperparams.get('decay_strength', 0.0), method),
            delta=hyperparams.get('delta', 0.1),
            shape=ROBUST_DECAY_SHAPES[method]
        )


# ============================
# SHARED HF BENCHMARK FUNCTIONS
# ============================

# --- WikiText-2 LM Data Utilities ---

class WikiTextLMDataset(Dataset):
    """WikiText-2 dataset for causal language modeling with pretrained tokenizer."""

    def __init__(self, tokenized_chunks):
        self.chunks = tokenized_chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        input_ids = self.chunks[idx]
        return input_ids[:-1], input_ids[1:]


def prepare_wikitext_datasets(tokenizer, max_length=256):
    """Tokenize WikiText-2 and split into fixed-length chunks."""
    from datasets import load_dataset
    dataset = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1')

    def tokenize_and_chunk(split):
        texts = [t for t in dataset[split]['text'] if len(t.strip()) > 10]
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)
        chunks = []
        for i in range(0, len(all_tokens) - max_length, max_length):
            chunk = torch.tensor(all_tokens[i:i + max_length], dtype=torch.long)
            chunks.append(chunk)
        return chunks

    train_chunks = tokenize_and_chunk('train')
    val_chunks = tokenize_and_chunk('validation')
    test_chunks = tokenize_and_chunk('test')

    return (WikiTextLMDataset(train_chunks),
            WikiTextLMDataset(val_chunks),
            WikiTextLMDataset(test_chunks))


def get_wikitext_dataloaders(model_name, batch_size,
                             max_seq_length, seed):
    """Prepare WikiText-2 dataloaders for language modeling."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_ds, val_ds, test_ds = prepare_wikitext_datasets(
        tokenizer, max_seq_length)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def prepare_sst2_dataloaders(model_name, batch_size,
                              max_seq_length, seed):
    """Prepare SST-2 dataloaders for text classification."""
    from transformers import AutoTokenizer
    from datasets import load_dataset

    # Try fast tokenizer first (faster, supports most models). Fall back to
    # the slow Python tokenizer when conversion fails — e.g. DeBERTa-v3's
    # SentencePiece-based tokenizer triggers a TikToken fallback in newer
    # transformers releases that silently corrupts inputs.
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if 'deberta-v3' in model_name.lower():
        # Force the slow tokenizer for DeBERTa-v3 regardless of whether the
        # fast converter happened to succeed: the converter occasionally
        # produces a TikToken proxy that tokenizes everything to a single
        # OOV id, and that has produced 49% (random) accuracy in the past.
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    dataset = load_dataset('nyu-mll/glue', 'sst2')

    def tokenize_fn(examples):
        return tokenizer(
            examples['sentence'],
            padding='max_length',
            truncation=True,
            max_length=max_seq_length,
            return_tensors=None
        )

    train_dataset = dataset['train'].map(
        tokenize_fn, batched=True)
    test_dataset = dataset['validation'].map(
        tokenize_fn, batched=True)

    cols = ['input_ids', 'attention_mask', 'label']
    train_dataset.set_format(type='torch', columns=cols)
    test_dataset.set_format(type='torch', columns=cols)

    n_total = len(train_dataset)
    n_val = int(n_total * 0.1)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_sub, val_sub = random_split(
        train_dataset, [n_train, n_val],
        generator=generator
    )

    train_loader = DataLoader(
        train_sub, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        val_sub, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# --- Shared Evaluation Functions ---

def evaluate_text_classifier(model, dataloader, device):
    """Evaluate text classification model (SST-2 style dict batches)."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def evaluate_vision_classifier(model, dataloader, device):
    """Evaluate vision classification model (CIFAR style tuple batches)."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def evaluate_lm(model, data_loader, device):
    """Evaluate language model perplexity."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100
            )
            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    perplexity = math.exp(min(avg_loss, 20))
    return perplexity, avg_loss


# --- Shared Training Step Functions ---

def text_classification_step(model, batch, device):
    """Compute loss for text classification (SST-2 style dict batches)."""
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['label'].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return nn.CrossEntropyLoss()(outputs.logits, labels)


def vision_classification_step(model, batch, device):
    """Compute loss for vision classification (CIFAR style tuple batches)."""
    images, labels = batch
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
    return nn.CrossEntropyLoss()(logits, labels)


# --- Shared Training Loops ---

def train_hf_classifier(
    *, method, hyperparams, seed,
    model_name, device, num_epochs, lr, patience, warmup_ratio,
    load_model_fn,
    get_dataloaders_fn,
    evaluate_fn=None,
    compute_loss_fn=None,
    verbose=True, min_delta=0.001,
    prune_targets=None, schedule_epochs=None,
):
    """
    Shared training loop for HF pretrained classifier benchmarks.

    Handles Groups B (SST-2 text classification) and B2 (vision classification).
    τ(w) is applied as post-optimizer weight decay, guaranteeing consistency.

    Args:
        method: regularization method name
        hyperparams: dict of method-specific hyperparameters
        seed: random seed
        model_name: HuggingFace model name
        device: torch device
        num_epochs, lr, patience, warmup_ratio: training config
        load_model_fn: (model_name, device) -> model
        get_dataloaders_fn: (seed) -> (train_loader, val_loader, test_loader)
        evaluate_fn: (model, dataloader, device) -> accuracy_float
        compute_loss_fn: (model, batch, device) -> loss_tensor
        verbose: print progress
        min_delta: minimum improvement for early stopping

    Returns:
        dict with: test_acc, val_acc, sparsity, convergence_epoch, total_params
    """
    if evaluate_fn is None:
        evaluate_fn = evaluate_text_classifier
    if compute_loss_fn is None:
        compute_loss_fn = text_classification_step

    lambda_val = hyperparams.get('lambda_val', 0.0)
    extra_params = {k: v for k, v in hyperparams.items()
                    if k not in ('lambda_val', 'decay_strength', 'tau0', 'tau_alpha', 'delta', 'wd', 'rho')}

    set_seed(seed)

    model = load_model_fn(model_name, device)
    train_loader, val_loader, test_loader = get_dataloaders_fn(seed)

    # Optimizer weight decay (Option-1 protocol): only 'WD-tuned' carries a (PSO-tuned)
    # decoupled weight decay; Baseline is now truly unregularized — wd is never silently
    # stacked on top of a penalty or τ(w) (CODICE-1).
    wd = optimizer_weight_decay(method, hyperparams)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # LR scheduler with linear warmup + linear decay
    # The LR schedule anneals over `schedule_epochs` (default: the epoch budget), so
    # raising the early-stopping ceiling never stretches the annealing horizon.
    total_steps = len(train_loader) * (schedule_epochs or num_epochs)
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / (total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, mode='max')

    if verbose:
        param_str = ', '.join(
            [f'{k}={v:.1e}' if isinstance(v, float) else f'{k}={v}'
             for k, v in hyperparams.items()])
        print(f"[TRAIN] Method={method}, {param_str}, seed={seed}")

    step = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            loss = compute_loss_fn(model, batch, device)

            # Add loss-penalty regularization (skip Baseline / WD-tuned / τ family)
            if method not in NO_LOSS_PENALTY:
                loss = loss + get_regularization_transformer(
                    model, method, lambda_val, device, extra_params=extra_params)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Post-optimizer decoupled decay (single source of truth): τ(w),
            # the Tau(alpha=0) ablation, and the robust-decay competitors.
            apply_decoupled_decay_transformer(model, method, hyperparams)

            epoch_loss += loss.item()
            step += 1

        val_acc = evaluate_fn(model, val_loader, device)

        if verbose and (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: "
                  f"loss={epoch_loss/len(train_loader):.4f}, val_acc={val_acc:.4f}")

        if early_stopping(val_acc, model, epoch):
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    model = early_stopping.restore_best_model(model)

    test_acc = evaluate_fn(model, test_loader, device)
    val_acc = evaluate_fn(model, val_loader, device)
    early_stopping.check_restored(val_acc, context=f'{model_name} {method} seed={seed}')
    sparsity, total_params, _ = measure_sparsity(model)

    result = {
        'test_acc': test_acc * 100,
        'val_acc': val_acc * 100,
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
            result[f'test_acc@{tag}'] = evaluate_fn(model, test_loader, device) * 100
            result[f'sparsity@{tag}'] = info['achieved_sparsity'] * 100
        del original_state

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def train_hf_lm(
    *, method, hyperparams, seed,
    model_name, device, num_epochs, lr, patience, warmup_ratio,
    load_model_fn,
    get_dataloaders_fn,
    verbose=True, use_mixed_precision=False, min_delta=0.1,
    prune_targets=None, schedule_epochs=None,
):
    """
    Shared training loop for HF pretrained language model benchmarks.

    Handles Group C (WikiText-2 causal LM).
    τ(w) is applied as post-optimizer weight decay, guaranteeing consistency.

    Args:
        method: regularization method name
        hyperparams: dict of method-specific hyperparameters
        seed: random seed
        model_name: HuggingFace model name
        device: torch device
        num_epochs, lr, patience, warmup_ratio: training config
        load_model_fn: (model_name, device) -> model
        get_dataloaders_fn: (seed) -> (train_loader, val_loader, test_loader)
        verbose: print progress
        use_mixed_precision: use AMP GradScaler + autocast (for phi2, gemma2)
        min_delta: minimum improvement for early stopping

    Returns:
        dict with: test_ppl, val_ppl, test_loss, sparsity,
        convergence_epoch, total_params
    """
    lambda_val = hyperparams.get('lambda_val', 0.0)
    extra_params = {k: v for k, v in hyperparams.items()
                    if k not in ('lambda_val', 'decay_strength', 'tau0', 'tau_alpha', 'delta', 'wd', 'rho')}

    set_seed(seed)

    model = load_model_fn(model_name, device)
    train_loader, val_loader, test_loader = get_dataloaders_fn(seed)

    # Option-1 protocol: only 'WD-tuned' carries a (PSO-tuned) decoupled weight decay;
    # Baseline is truly unregularized (CODICE-1).
    wd = optimizer_weight_decay(method, hyperparams)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # The LR schedule anneals over `schedule_epochs` (default: the epoch budget), so
    # raising the early-stopping ceiling never stretches the annealing horizon.
    total_steps = len(train_loader) * (schedule_epochs or num_epochs)
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / (total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, mode='min')

    scaler = (torch.amp.GradScaler('cuda')
              if use_mixed_precision and device.type == 'cuda' else None)

    if verbose:
        param_str = ', '.join(
            [f'{k}={v:.1e}' if isinstance(v, float) else f'{k}={v}'
             for k, v in hyperparams.items()])
        print(f"[TRAIN] Method={method}, {param_str}, seed={seed}")

    step = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            amp_ctx = torch.amp.autocast('cuda') if scaler else nullcontext()
            with amp_ctx:
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=-100
                )
                if method not in NO_LOSS_PENALTY:
                    loss = loss + get_regularization_transformer(
                        model, method, lambda_val, device,
                        extra_params=extra_params)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            # Post-optimizer decoupled decay (single source of truth): τ(w),
            # the Tau(alpha=0) ablation, and the robust-decay competitors.
            apply_decoupled_decay_transformer(model, method, hyperparams)

            epoch_loss += loss.item()
            step += 1

        val_ppl, val_loss = evaluate_lm(model, val_loader, device)

        if verbose:
            avg = epoch_loss / len(train_loader)
            print(f"  Epoch {epoch+1}/{num_epochs}: "
                  f"train_loss={avg:.4f}, val_ppl={val_ppl:.2f}")

        if early_stopping(val_ppl, model, epoch):
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    model = early_stopping.restore_best_model(model)

    test_ppl, test_loss = evaluate_lm(model, test_loader, device)
    val_ppl, _ = evaluate_lm(model, val_loader, device)
    early_stopping.check_restored(val_ppl, context=f'{model_name} {method} seed={seed}')
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
            ppl_pruned, loss_pruned = evaluate_lm(model, test_loader, device)
            result[f'test_ppl@{tag}'] = ppl_pruned
            result[f'test_loss@{tag}'] = loss_pruned
            result[f'sparsity@{tag}'] = info['achieved_sparsity'] * 100
        del original_state

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# --- Shared Benchmark Orchestration ---

def _fmt_duration(seconds):
    """Format seconds as 'Xh Ym' or 'Ym Zs'."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m"
    return f"{m}m {s}s"


def run_benchmark(
    *, experiment_name, benchmark_title, model_name, device,
    config, seeds, train_fn,
    primary_metric, metric_mode,
    pso_metric, pso_mode, pso_budget,
    csv_filename, json_filename,
    quiet=False,
    fixed_hyperparams=None,
    methods=None,
    prune_targets=None,
    journal_enabled=True,
    reuse_pso_seed0=True,
):
    """
    Shared main() orchestration for all HF benchmarks.

    Runs: PSO tuning -> multi-run evaluation -> statistical analysis -> save results.

    Args:
        experiment_name: name for ExperimentTracker
        benchmark_title: display title
        model_name: HuggingFace model name
        device: torch device
        config: dict of config items to display (e.g. Epochs, Batch size, LR)
        seeds: list of random seeds
        train_fn: (method, hyperparams, seed, verbose, prune_targets=None) -> metrics dict
        primary_metric: metric name for ranking ('test_acc' or 'test_ppl')
        metric_mode: 'max' for accuracy, 'min' for perplexity
        pso_metric: metric for PSO optimization ('val_acc' or 'val_ppl')
        pso_mode: 'max' or 'min'
        pso_budget: PSO budget level ('light', 'standard', 'thorough')
        csv_filename: output CSV path
        json_filename: output JSON path
        quiet: if True, only print model name, time estimate, and elapsed time
        fixed_hyperparams: dict {method: hp_dict} to skip PSO and use directly.
            If provided, PSO is skipped and only the listed methods run.
        methods: list of methods to run. Defaults to all 8.
        prune_targets: list of sparsity fractions in (0, 1) for post-training
            magnitude pruning sweep applied during the multi-run eval (not PSO).
        reuse_pso_seed0: reuse the PSO evaluation of the winning hyperparameters as the
            seeds[0] final run (training is deterministic, so it IS that run). Set False
            when per-run side effects matter - e.g. instrumentation trajectories, which
            must come from a run tagged as a final evaluation - at the cost of one extra
            training run per tuned method.
    """
    n_runs = len(seeds)
    if methods is None:
        methods = list(DEFAULT_METHODS)
    verbose = not quiet

    # train_fn may optionally accept `phase` ('pso' | 'eval') so it can keep the side
    # effects of tuning evaluations apart from those of final runs (instrumentation
    # files). Passed only when the callable declares it.
    _sig = inspect.signature(train_fn).parameters
    _train_accepts_phase = ('phase' in _sig or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in _sig.values()))

    def _call_train(method, hp, seed, phase, **kw):
        if _train_accepts_phase:
            kw['phase'] = phase
        return train_fn(method, hp, seed, **kw)

    bench_start = time.time()

    if verbose:
        print("=" * 80)
        print(f"{benchmark_title} - STANDARDIZED BENCHMARK")
        print("=" * 80)
        print("Configuration:")
        print(f"  - Model: {model_name}")
        for key, value in config.items():
            print(f"  - {key}: {value}")
        print(f"  - Runs per method: {n_runs}")
        print(f"  - PSO budget: {pso_budget}")
        print(f"  - Device: {device}")
        print("=" * 80)

    if quiet:
        print(f"[START] {benchmark_title} ({model_name})", flush=True)

    tracker = ExperimentTracker(experiment_name)
    best_hyperparams = {}
    first_run_timed = False

    # Resumable journal + incremental saves. The journal path derives from the CSV name,
    # so different rosters (e.g. the 2x2 and the robust head-to-head running on two pods)
    # keep separate journals and never interfere.
    journal = RunJournal(
        os.path.join('results', 'journal',
                     os.path.basename(csv_filename).replace('.csv', '.jsonl')),
        enabled=journal_enabled)

    def _save_partial(tag=''):
        """Write the CSV/JSON as they currently stand. Called after every completed run,
        so whatever has finished is on disk even if the pod is stopped one second later."""
        try:
            tracker.save_to_csv(csv_filename)
            tracker.save_to_json(json_filename)
        except Exception as exc:                       # never let saving kill a run
            print(f"[PARTIAL] save failed{tag}: {exc}", flush=True)

    # Step 1: Hyperparameter tuning (skipped if fixed_hyperparams provided)
    if fixed_hyperparams is not None:
        if verbose or quiet:
            print("\n[SKIP-PSO] Using fixed hyperparameters:", flush=True)
            for m in methods:
                hp = fixed_hyperparams.get(m, {})
                print(f"  {m}: {hp}", flush=True)
        for m in methods:
            best_hyperparams[m] = fixed_hyperparams.get(m, {})
            tracker.set_hyperparams(m, best_hyperparams[m])
    else:
        if verbose:
            print("\n" + "=" * 80)
            print("STEP 1: HYPERPARAMETER TUNING (PSO)")
            print("=" * 80)

        for method in methods:
            if method == 'Baseline':
                best_hyperparams[method] = {}
                tracker.set_hyperparams(method, {})
                continue

            # Already tuned in an earlier, interrupted attempt: skip the whole search.
            _done_hp = journal.lookup_best_hp(method)
            if _done_hp is not None:
                print(f"[RESUME] {method}: reusing journaled PSO winner {_done_hp}",
                      flush=True)
                best_hyperparams[method] = _done_hp
                tracker.set_hyperparams(method, _done_hp)
                continue

            def pso_train_fn(params, _method=method):
                # Replay-exact cache: PSO is seeded, so a resumed run proposes the same
                # particles in the same order and every prior evaluation hits here.
                cached = journal.lookup_pso(_method, params)
                if cached is not None:
                    return cached
                metrics = _call_train(_method, params, seeds[0], 'pso', verbose=False)
                journal.record_pso(_method, params, metrics)
                return metrics

            # Time the first eval to estimate total duration
            if quiet and not first_run_timed:
                t0 = time.time()
                best_hyperparams[method] = find_best_hyperparams(
                    pso_train_fn, method,
                    metric_name=pso_metric,
                    mode=pso_mode,
                    pso_budget=pso_budget,
                    seed=seeds[0], verbose=False
                )
                first_method_time = time.time() - t0
                first_run_timed = True
                # Estimate: (n_methods-1) PSO-tuned methods + final runs
                est_total = first_method_time * max(1, len(methods) - 1) + (
                    first_method_time / PSO_BUDGETS[pso_budget]['n_iterations']
                    * n_runs * len(methods)
                )
                print(f"[EST]   ~{_fmt_duration(est_total)} "
                      f"(based on first method: {_fmt_duration(first_method_time)})",
                      flush=True)
            else:
                best_hyperparams[method] = find_best_hyperparams(
                    pso_train_fn, method,
                    metric_name=pso_metric,
                    mode=pso_mode,
                    pso_budget=pso_budget,
                    seed=seeds[0], verbose=verbose
                )
            tracker.set_hyperparams(method, best_hyperparams[method])
            journal.record_best_hp(method, best_hyperparams[method])

    # Step 2: Multi-run evaluation (with optional pruning sweep)
    if verbose:
        print("\n" + "=" * 80)
        print("STEP 2: MULTI-RUN EVALUATION (test set)")
        if prune_targets:
            print(f"  + post-training pruning sweep at {prune_targets}")
        print("=" * 80)

    for method in methods:
        hp = best_hyperparams[method]
        if verbose:
            params_str = (
                ", ".join(f"{k}={v:.4g}" for k, v in hp.items())
                if hp else "none")
            print(f"\n[EVAL] Running {method} "
                  f"({params_str})...")

        for run_idx, seed in enumerate(seeds):
            if verbose:
                print(f"  Run {run_idx+1}/{n_runs} "
                      f"(seed={seed})...", end=" ")

            cached = journal.lookup_eval(method, seed)
            if (cached is None and seed == seeds[0] and not prune_targets
                    and reuse_pso_seed0):
                # The PSO winner was, by construction, already trained at seeds[0] during
                # the search. Training is deterministic in (method, hyperparams, seed), so
                # that evaluation IS this one. Only safe when no pruning sweep is
                # requested, since the pruning metrics are not produced during PSO.
                reuse = journal.lookup_pso(method, hp)
                if reuse is not None:
                    cached = reuse
                    journal.record_eval(method, seed, reuse)
            if cached is not None:
                tracker.add_result(method, run_idx, cached)
                _save_partial(f' after {method}/seed{seed}')
                if verbose:
                    print(f"[REUSE] {primary_metric}="
                          f"{cached.get(primary_metric, float('nan')):.2f} "
                          f"(already trained during tuning)")
                continue

            metrics = _call_train(
                method, hp, seed, 'eval', verbose=False, prune_targets=prune_targets)
            if seed == seeds[0] and not reuse_pso_seed0 and hp:
                # Free determinism check: the PSO evaluation of the winning hyperparameters
                # was this very (method, hp, seed) run. Report the discrepancy.
                _pso_same = journal.lookup_pso(method, hp)
                if _pso_same is not None and primary_metric in _pso_same:
                    _d = metrics.get(primary_metric, float('nan')) - _pso_same[primary_metric]
                    print(f"[DETERMINISM] {method} seed={seed}: final-run {primary_metric}="
                          f"{metrics.get(primary_metric, float('nan')):.6g} vs PSO eval "
                          f"{_pso_same[primary_metric]:.6g} (diff {_d:+.3g})", flush=True)
            tracker.add_result(method, run_idx, metrics)
            journal.record_eval(method, seed, metrics)
            # Persist after EVERY run: stopping the pod now costs at most this one run.
            _save_partial(f' after {method}/seed{seed}')

            if verbose:
                val = metrics[primary_metric]
                sp = metrics['sparsity']
                if 'acc' in primary_metric:
                    print(f"{primary_metric}={val:.2f}%, "
                          f"sparsity={sp:.2f}%")
                else:
                    print(f"{primary_metric}={val:.2f}, "
                          f"sparsity={sp:.2f}%")

    # Step 3: Statistical analysis
    if verbose:
        print("\n" + "=" * 80)
        print("STEP 3: STATISTICAL ANALYSIS")
        print("=" * 80)

        tracker.print_summary(primary_metric, mode=metric_mode)
        tracker.print_summary('sparsity', mode='min')

        # Pairwise comparisons
        print("\n" + "=" * 80)
        print("PAIRWISE COMPARISONS (vs Baseline)")
        print("=" * 80)

        for method in methods:
            if method == 'Baseline':
                continue

            comparison = tracker.compare_methods(
                'Baseline', method, primary_metric)
            sig_marker = ("***" if comparison['significant_01']
                          else ("*" if comparison['significant_05'] else "ns"))

            print(f"{method:15} | t={comparison['t_statistic']:7.3f} | "
                  f"p={comparison['p_value']:7.4f} {sig_marker:3} | "
                  f"d={comparison['cohens_d']:6.3f}")

    # Save results
    tracker.save_to_csv(csv_filename)
    tracker.save_to_json(json_filename)

    if journal.enabled and os.path.exists(journal.path):
        _jsz = os.path.getsize(journal.path)
        print(f"[JOURNAL] {journal.path}: {_jsz/1024:.1f} KB "
              f"({len(journal.pso)} PSO evals + {len(journal.evals)} final runs). "
              f"Delete it to force a clean re-run.", flush=True)
    journal.close()

    elapsed = time.time() - bench_start
    if quiet:
        print(f"[DONE]  {benchmark_title} in {_fmt_duration(elapsed)}",
              flush=True)
    else:
        print("\n" + "=" * 80)
        print(f"EXPERIMENT COMPLETED in {_fmt_duration(elapsed)}")
        print("=" * 80)
