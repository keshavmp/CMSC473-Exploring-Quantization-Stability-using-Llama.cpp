from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.important_imatrix_plots import (
    TYPES, block_averages, column_squared_error, input_second_moment, relative_energy, weighted_error,
)


def test_relative_energy_normalizes_each_exact_tensor_across_domains():
    frame = pd.DataFrame([{"task": task, "tensor": tensor, "energy": energy}
                          for tensor, energies in (("blk.0.attn_q.weight", [1, 2, 3]),
                                                   ("blk.1.attn_q.weight", [10, 10, 10]))
                          for task, energy in zip(("code", "math", "chat"), energies)])
    result = relative_energy(frame)
    assert result["relative_energy"].tolist() == [0.5, 1, 1.5, 1, 1, 1]
    with pytest.raises(ValueError, match="all three"):
        relative_energy(frame.iloc[:-1])


def test_input_second_moment_uses_token_counts_and_rejects_partial_data():
    np.testing.assert_array_equal(input_second_moment([8, 12], [4]), [2, 3])
    with pytest.raises(ValueError, match="positive input count"):
        input_second_moment([8, 12], [0])


def test_weighted_error_matches_local_output_error_for_diagonal_second_moment():
    source = np.array([[1, 2], [3, 4]], dtype=np.float32)
    quant = np.array([[2, 2], [1, 5]], dtype=np.float32)
    original = SimpleNamespace(name="test", shape=np.array([2, 2]), data=source, tensor_type="F32")
    quantized = SimpleNamespace(name="test", shape=np.array([2, 2]), data=quant, tensor_type="F32")
    per_column = column_squared_error(original, quantized, lambda values, _: values, rows_per_batch=1)
    np.testing.assert_array_equal(per_column, [5, 1])
    assert weighted_error([2, 3], per_column) == 13
    inputs = np.array([[2, 0], [0, np.sqrt(6)]])
    local_error = inputs @ (quant - source).T
    assert np.mean(np.sum(local_error**2, axis=1)) == pytest.approx(13)


def test_dimension_mismatch_cannot_silently_broadcast():
    with pytest.raises(ValueError, match="must match"):
        weighted_error([1, 2], [1])


def test_identical_weights_have_zero_error():
    values = np.arange(6).reshape(2, 3)
    tensor = SimpleNamespace(name="test", shape=np.array([3, 2]), data=values, tensor_type="F32")
    result = column_squared_error(tensor, tensor, lambda data, _: data)
    np.testing.assert_array_equal(result, [0, 0, 0])


def test_block_means_normalize_after_averaging_raw_energy():
    stats, errors = [], []
    for task in ("code", "math", "chat"):
        for index, kind in enumerate(TYPES):
            energy = (400 if task == "chat" else 100) if index == 0 else 1
            stats.append(dict(task=task, layer=0, tensor_type=kind, energy=energy,
                              entropy_pct=100 - index, kurtosis=index * 2))
            errors.append(dict(task=task, layer=0, tensor_type=kind, model="pure-q2",
                               precision="Q2", weighted_error=index + 1))
    blocks, weighted = block_averages(pd.DataFrame(stats), pd.DataFrame(errors))
    code = blocks.set_index("task").loc["code"]
    assert code.mean_energy == pytest.approx(106 / 7)
    assert code.relative_energy == pytest.approx(106 / 206)
    assert code.mean_entropy_pct == 97
    assert code.mean_kurtosis == 6
    assert weighted.mean_weighted_error.eq(4).all()
    assert weighted.n_tensors.eq(7).all()
    with pytest.raises(ValueError, match="seven measured tensor types"):
        block_averages(pd.DataFrame(stats).iloc[:-1], pd.DataFrame(errors))
