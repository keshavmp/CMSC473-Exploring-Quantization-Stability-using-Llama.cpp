from types import SimpleNamespace
from unittest.mock import patch

import pytest
from requests.exceptions import ConnectionError

from src.gguf import GGUFLM, get_result


def make_payload(greedy=True):
    """Server logprobs: context is 8 chars long, continuation is 'c' then 'd'."""
    continuation_top = {"c": -0.1, "x": -9.0} if greedy else {"c": -9.0, "x": -0.1}
    return {
        "text_offset": [0, 4, 8],
        "tokens": ["a", "b", "c", "d"],
        "token_logprobs": [-1.0, -2.0, -3.0, -4.0],
        "top_logprobs": [
            {"a": -0.1},
            {"b": -0.1},
            continuation_top,
            {"d": -0.1},
        ],
    }


def make_lm():
    return GGUFLM(base_url="http://127.0.0.1:8080")


def test_get_result_sums_continuation_logprobs():
    total, _ = get_result(make_payload(), context_length=8)

    assert total == pytest.approx(-3.0)


def test_get_result_flags_greedy_and_non_greedy_choices():
    assert get_result(make_payload(greedy=True), context_length=8)[1] is True
    assert get_result(make_payload(greedy=False), context_length=8)[1] is False


@patch.object(GGUFLM, "gguf_completion")
def test_loglikelihood_returns_greedy_pairs(mock_complete):
    mock_complete.return_value = {"choices": [{"logprobs": make_payload()}]}
    request = SimpleNamespace(args=("abcdefgh", "zz"))

    assert make_lm().loglikelihood([request], disable_tqdm=True) == [(-3.0, True)]


@patch.object(GGUFLM, "gguf_completion")
def test_generate_until_strips_the_generated_text(mock_complete):
    mock_complete.return_value = {"choices": [{"text": "  hello  "}]}
    request = SimpleNamespace(args=("prompt", {"until": ["</s>"]}))

    assert make_lm().generate_until([request], disable_tqdm=True) == ["hello"]


@patch("src.gguf.time.sleep")  # don't really wait between retries
@patch("src.gguf.requests.post")
def test_gguf_completion_raises_after_all_retries_fail(mock_post, _sleep):
    mock_post.side_effect = ConnectionError("server is down")

    with pytest.raises(RuntimeError):
        make_lm().gguf_completion(context="hi", retries=2)

    assert mock_post.call_count == 2
