"""Tests for analysis/audit_early_stopping.py (the pod preflight gate and the legacy audit)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'analysis'))
import audit_early_stopping as aud  # noqa: E402


def _write(tmp_path, name, vals, final_val, best_epoch, phase='eval'):
    d = {'benchmark': 'gpt2_tiny', 'method': name, 'seed': 42, 'phase': phase,
         'best_epoch': best_epoch,
         'final': {'val_ppl': final_val, 'convergence_epoch': best_epoch},
         'trajectory': [{'epoch': i + 1, 'val_ppl': v} for i, v in enumerate(vals)]}
    (tmp_path / f'gpt2_tiny_{name}_seed42.json').write_text(json.dumps(d), encoding='utf-8')


def test_instrumentation_audit_accepts_best_epoch_and_rejects_last_epoch(tmp_path):
    vals = [10.0, 8.0, 9.0, 9.5]                   # best at epoch 2, trained to epoch 4
    _write(tmp_path, 'good', vals, final_val=8.0, best_epoch=2)
    assert aud.audit_instrumentation(str(tmp_path), min_ok=1) == 0
    _write(tmp_path, 'lastepoch', vals, final_val=9.5, best_epoch=2)    # reports the last epoch
    assert aud.audit_instrumentation(str(tmp_path)) == 1


def test_instrumentation_audit_min_delta_tolerance_and_uninformative_runs(tmp_path):
    # Last epoch marginally better than the recorded best (min_delta): acceptable.
    _write(tmp_path, 'mindelta', [10.0, 8.0, 7.995], final_val=8.0, best_epoch=2)
    assert aud.audit_instrumentation(str(tmp_path), min_ok=1) == 0
    # Never stopped past the best epoch: uninformative, so --min-ok is not satisfied.
    _write(tmp_path, 'flat', [10.0, 9.0, 8.0], final_val=8.0, best_epoch=3)
    assert aud.audit_instrumentation(str(tmp_path), min_ok=2) == 1
    assert aud.audit_instrumentation(str(tmp_path), min_ok=1) == 0


def test_instrumentation_audit_ignores_pso_probes(tmp_path):
    _write(tmp_path, 'probe', [10.0, 8.0, 9.0], final_val=9.0, best_epoch=2, phase='pso')
    assert aud.audit_instrumentation(str(tmp_path)) == 0     # skipped, not a failure


def test_legacy_rule_ceiling_lookup():
    assert aud.ceiling_for('gpt2_large_wikitext_standardized_results') == 12
    assert aud.ceiling_for('gpt2_large_wikitext_standardized_results_data25c') == 40
    assert aud.ceiling_for('gpt2_large_wikitext_standardized_results_data25') == 48
    assert aud.ceiling_for('gpt2_wikitext_standardized_results') == 30
    assert aud.ceiling_for('gpt2_wt103_standardized_results') == 8
    assert aud.ceiling_for('cifar_standardized_results') == 100
    assert aud.ceiling_for('unknown_benchmark') is None
