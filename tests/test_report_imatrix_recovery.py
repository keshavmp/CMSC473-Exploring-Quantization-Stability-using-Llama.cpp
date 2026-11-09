import numpy as np
import pytest

import pandas as pd

from src.report_imatrix_recovery import holm, paired_effect, transfer_tables


def test_paired_effect_counts_disagreements_without_assuming_independent_models():
    control = {i: (float(i < 5), 'doc' + str(i), 'prompt') for i in range(10)}
    candidate = {i: (1.0, 'doc' + str(i), 'prompt') for i in range(10)}
    result = paired_effect(control, candidate, np.random.default_rng(0))
    assert result['gain_pp'] == 50
    assert result['wins'] == 5 and result['losses'] == 0 and result['ties'] == 5
    assert result['mcnemar_exact_p'] == pytest.approx(.0625)
    assert 0 <= result['paired_ci95_low_pp'] <= 50 <= result['paired_ci95_high_pp'] <= 100


def test_pairing_rejects_same_ids_with_different_questions():
    with pytest.raises(ValueError, match='question/prompt'):
        paired_effect({0: (0, 'a', 'p')}, {0: (1, 'b', 'p')}, np.random.default_rng(0))


def test_holm_adjusts_in_sorted_order_then_restores_original_order():
    np.testing.assert_allclose(holm([.04, .001, .03]), [.06, .003, .06])


def test_task_transfer_uses_matched_control_and_does_not_invent_code_benchmark():
    tables = {
        'model_metrics': pd.DataFrame([
            dict(precision='Q4', variant='control', gsm8k_score_pct=10),
            dict(precision='Q4', variant='code', gsm8k_score_pct=12)]),
        'benchmark_gains': pd.DataFrame([dict(precision='Q4', variant='code', task='gsm8k', gain_pp=2)]),
        'kl_by_domain': pd.DataFrame([dict(precision='Q4', variant='code', domain='code', kl_reduction_pct=5)])}
    result = transfer_tables(tables)
    row = result['calibration_task_transfer'].iloc[0]
    assert row.control_score_pct == 10 and row.imatrix_score_pct == 12
    assert 'no code benchmark' in row.relationship
    assert bool(result['calibration_domain_transfer'].iloc[0].same_domain)
    tables['benchmark_gains'].loc[0, 'gain_pp'] = 3
    with pytest.raises(ValueError, match='same-precision control'):
        transfer_tables(tables)
