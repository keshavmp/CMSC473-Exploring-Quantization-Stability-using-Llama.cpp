import numpy as np
import pandas as pd
import pytest

from src.compare_imatrix_recovery import pool_local, recovery


def test_recovery_retains_worsening_and_marks_zero_control_undefined():
    np.testing.assert_allclose(recovery([5, 10, 15, 0], [10, 10, 10, 0]),
                               [50, 0, -50, np.nan], equal_nan=True)


@pytest.mark.parametrize('error,control', [([-1], [1]), ([1], [-1]), ([np.inf], [1]), ([1], [np.nan])])
def test_recovery_rejects_invalid_errors(error, control):
    with pytest.raises(ValueError, match='finite and nonnegative'):
        recovery(error, control)


def test_pool_uses_matched_control_sums_instead_of_mean_tensor_percentages():
    # Tiny tensor improves 100%, large tensor worsens 10%: overall error increases.
    frame = pd.DataFrame(dict(model=['Llama', 'Llama', 'Qwen'], precision=['Q4']*3,
                              layer=[0, 0, 0], weighted_error=[0., 110., 5.],
                              control_error=[1., 100., 10.]))
    result = pool_local(frame, ['layer']).set_index('model')
    assert result.loc['Llama', 'recovery_pct'] == pytest.approx(100*(1-110/101))
    assert result.loc['Qwen', 'recovery_pct'] == 50


def test_pool_separates_calibrations_and_precisions():
    frame = pd.DataFrame(dict(model=['Gemma']*3, precision=['Q2', 'Q2', 'Q8'],
                              variant=['code', 'math', 'code'], weighted_error=[2., 5., 1.],
                              control_error=[10., 10., 1.]))
    result = pool_local(frame, ['variant']).set_index(['precision', 'variant'])
    assert result.loc[('Q2', 'code'), 'recovery_pct'] == 80
    assert result.loc[('Q2', 'math'), 'recovery_pct'] == 50
    assert result.loc[('Q8', 'code'), 'recovery_pct'] == 0
