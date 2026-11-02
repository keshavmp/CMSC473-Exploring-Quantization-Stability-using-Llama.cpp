import pandas as pd
import pytest

from src.important_imatrix_plots import TASKS, TYPES
from src.important_precision_plots import precision_averages


def fixture_data():
    stats, errors = [], []
    for model in ("BF16", "pure-q2"):
        for task_index, task in enumerate(TASKS):
            for index, kind in enumerate(TYPES):
                tensor = f"blk.0.{kind}.weight"
                stats.append(dict(model=model, task=task, tensor=tensor, layer=0,
                                  energy=task_index + 1, entropy_pct=90, kurtosis=10))
                if model != "BF16":
                    errors.append(dict(model=model, task=task, tensor=tensor, tensor_type=kind,
                        layer=0, precision="Q2", weighted_error=index + 1, reference_energy=10 if index == 0 else 1))
    return pd.DataFrame(stats), pd.DataFrame(errors)


def test_precision_averages_are_task_balanced_and_block_error_is_ratio_of_means():
    stats, errors = fixture_data()
    tensor, block, tensor_errors, block_errors = precision_averages(stats, errors, "BF16")
    assert tensor.energy.eq(2).all()
    assert tensor.n_tasks.eq(3).all()
    assert block.n_tensors.eq(7).all()
    assert block_errors.iloc[0].weighted_error == 4
    assert block_errors.iloc[0].relative_error_pct == pytest.approx(100 * 28 / 16)
    assert tensor_errors.query("tensor_type == 'attn_q'").iloc[0].relative_error_pct == 10


def test_incomplete_precision_tasks_are_rejected():
    stats, errors = fixture_data()
    with pytest.raises(ValueError, match="all three datasets"):
        precision_averages(stats.iloc[:-1], errors, "BF16")
