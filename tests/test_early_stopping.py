"""Outcome tests for early stopping.

They check the *outcome* (the number reported is the best-epoch model's), not the
configuration: a snapshot that aliased the live parameters, or a restore that did not
land, would fail these tests and the run-time guards they exercise.

Run:  pytest tests/test_early_stopping.py -q        (CPU only, a few seconds)
"""
import math
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiment_utils as eu  # noqa: E402
from experiment_utils import (  # noqa: E402
    EarlyStopping, canonical_decay_params, dimension_aware_budget,
)


# ----------------------------------------------------------------------------- unit
def _linear(fill):
    m = nn.Linear(4, 2)
    m.weight.data.fill_(fill)
    m.bias.data.zero_()
    return m


def test_snapshot_is_detached_from_live_parameters():
    """Restoring after further training must bring the best weights back."""
    m = _linear(1.0)
    es = EarlyStopping(patience=2, mode='min')
    es(10.0, m, 0)                      # best recorded here, weights = 1.0
    m.weight.data.fill_(4.0)            # training makes them worse
    es.restore_best_model(m)
    assert torch.all(m.weight == 1.0), "restore_best_model() must undo later training"


@pytest.mark.parametrize("mode,values,expected_best,expected_stop", [
    # min: best at index 3 (0.8), then two non-improvements -> stop at epoch 5
    ('min', [1.0, 0.9, 0.95, 0.8, 0.85, 0.86], 3, 5),
    # max: best at index 2 (0.93), then two non-improvements -> stop at epoch 4
    ('max', [0.80, 0.90, 0.93, 0.92, 0.93], 2, 4),
])
def test_best_epoch_and_patience(mode, values, expected_best, expected_stop):
    m = _linear(0.0)
    es = EarlyStopping(patience=2, mode=mode)
    stopped_at = None
    for epoch, v in enumerate(values):
        m.weight.data.fill_(float(epoch))          # weights identify the epoch
        if es(v, m, epoch):
            stopped_at = epoch
            break
    assert es.best_epoch == expected_best
    assert stopped_at == expected_stop
    es.restore_best_model(m)
    assert torch.all(m.weight == float(expected_best)), "restored weights = best epoch's"


def test_reimprovement_resets_counter():
    m = _linear(0.0)
    es = EarlyStopping(patience=2, mode='min')
    seq = [1.0, 1.1, 0.9, 1.0, 1.05]   # dip at 2 resets the counter; stop at 4
    stopped = [es(v, m, i) for i, v in enumerate(seq)]
    assert stopped == [False, False, False, False, True]
    assert es.best_epoch == 2


def test_restore_detects_aliased_snapshot(monkeypatch):
    """A snapshot that aliases live tensors must make the restore fail loudly."""
    monkeypatch.setattr(eu, '_snapshot_state', lambda model: model.state_dict().copy())
    m = _linear(1.0)
    es = EarlyStopping(patience=2, mode='min')
    es(10.0, m, 0)
    m.weight.data.fill_(4.0)
    with pytest.raises(RuntimeError, match="do not match the snapshot"):
        es.restore_best_model(m)


def test_check_restored_rejects_last_epoch_metric():
    m = _linear(1.0)
    es = EarlyStopping(patience=2, mode='min')
    es(10.0, m, 0)
    es(12.0, m, 1)
    es(13.0, m, 2)
    es.restore_best_model(m)
    assert es.check_restored(10.0)                          # best-epoch value: OK
    assert es.check_restored(10.0 * (1 + 1e-4))             # within tolerance
    with pytest.raises(RuntimeError, match="NOT the best-epoch model"):
        es.check_restored(13.0)                             # last-epoch value: refused


def test_check_restored_requires_restore_first():
    m = _linear(1.0)
    es = EarlyStopping(patience=2, mode='min')
    es(10.0, m, 0)
    with pytest.raises(RuntimeError, match="before restore_best_model"):
        es.check_restored(10.0)


def test_fingerprint_handles_buffers_and_integer_tensors():
    m = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3))
    m.train()
    m(torch.randn(8, 3))                       # updates running stats + num_batches_tracked
    es = EarlyStopping(patience=1, mode='min')
    es(1.0, m, 0)
    m(torch.randn(8, 3))                       # changes buffers again
    es.restore_best_model(m)                   # must not raise
    assert es.restored


# ---------------------------------------------------------------------- end-to-end
def _mse(model, x, y):
    model.eval()
    with torch.no_grad():
        return nn.functional.mse_loss(model(x), y).item()


def test_end_to_end_reported_metric_is_best_epoch_model():
    """Train a wide MLP past a clear overfitting peak on noisy 1-D data and assert that the
    metric reported after early stopping is the BEST-epoch model's, not the last epoch's."""
    torch.manual_seed(0)
    x_tr = torch.rand(24, 1) * 6 - 3
    y_tr = torch.sin(x_tr) + 0.4 * torch.randn_like(x_tr)          # noisy: invites overfit
    x_va = torch.linspace(-3, 3, 200).unsqueeze(1)
    y_va = torch.sin(x_va)
    x_te = torch.linspace(-2.9, 2.9, 150).unsqueeze(1)
    y_te = torch.sin(x_te)

    model = nn.Sequential(nn.Linear(1, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(),
                          nn.Linear(128, 1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    es = EarlyStopping(patience=15, mode='min')

    history = []                       # (val_mse, test_mse) per epoch, from the LIVE model
    last_epoch = None
    for epoch in range(800):
        model.train()
        opt.zero_grad()
        nn.functional.mse_loss(model(x_tr), y_tr).backward()
        opt.step()
        history.append((_mse(model, x_va, y_va), _mse(model, x_te, y_te)))
        last_epoch = epoch
        if es(history[-1][0], model, epoch):
            break

    model = es.restore_best_model(model)
    val_reported = _mse(model, x_va, y_va)
    test_reported = _mse(model, x_te, y_te)
    convergence_epoch = es.best_epoch + 1                # what every benchmark stores

    best_epoch = min(range(len(history)), key=lambda i: history[i][0])
    # The scenario is meaningful only if training went past the best epoch.
    assert es.early_stop and best_epoch < last_epoch, "test setup: no overfitting reached"
    assert history[last_epoch][0] > history[best_epoch][0] * 1.001, \
        "test setup: last epoch must be visibly worse than the best epoch"

    assert es.best_epoch == best_epoch
    assert convergence_epoch == best_epoch + 1
    assert es.check_restored(val_reported)
    assert math.isclose(val_reported, history[best_epoch][0], rel_tol=1e-6)
    assert math.isclose(test_reported, history[best_epoch][1], rel_tol=1e-6), \
        "reported TEST metric must be the best-epoch model's"
    assert not math.isclose(test_reported, history[last_epoch][1], rel_tol=1e-4), \
        "reported TEST metric must NOT be the last-epoch model's"


# ------------------------------------------------------------- protocol invariants
def test_auto_pso_budget_is_12_for_1d_and_40_for_every_higher_dim():
    assert dimension_aware_budget(0) == {'n_particles': 1, 'n_iterations': 1, 'patience': 1}
    b1 = dimension_aware_budget(1)
    b2 = dimension_aware_budget(2)
    b3 = dimension_aware_budget(3)
    assert b1['n_particles'] * b1['n_iterations'] == 12
    assert b2['n_particles'] * b2['n_iterations'] == 40
    assert b3['n_particles'] * b3['n_iterations'] == 40
    # the swarm must not stop early: the reported budget is the spent budget
    for b in (b1, b2, b3):
        assert b['patience'] >= b['n_iterations']


def test_canonical_form_reproduces_the_published_large_optimum():
    legacy = {'decay_strength': 0.00019517224641449476, 'tau0': 1.000590589657011,
              'tau_alpha': 21.256581564753876}
    rho, delta = canonical_decay_params('τ(w)', legacy)
    assert math.isclose(rho, 1.9506e-4, rel_tol=1e-3)
    assert math.isclose(delta, 0.04707, rel_tol=1e-3)
    rho0, delta0 = canonical_decay_params('Tau(alpha=0)', {'rho': 6e-5})
    assert rho0 == 6e-5 and delta0 == float('inf')
