from types import SimpleNamespace
from threading import Barrier, Lock
from unittest.mock import Mock, patch

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
@patch("src.gguf.requests.Session.post")
def test_gguf_completion_raises_after_all_retries_fail(mock_post, _sleep):
    mock_post.side_effect = ConnectionError("server is down")

    with pytest.raises(RuntimeError):
        make_lm().gguf_completion(context="hi", retries=2)

    assert mock_post.call_count == 2


@patch("src.gguf.requests.Session.post")
def test_choice_scoring_uses_pre_sampling_answer_probability(post):
    payloads = [
        {"tokens": [1, 2]}, {"tokens": [1, 2, 3, 4]},
        {"completion_probabilities": [{"id": 3, "logprob": -2.0, "top_logprobs": [{"id": 8}]}]},
        {"completion_probabilities": [{"id": 4, "logprob": -1.5, "top_logprobs": [{"id": 4}]}]},
    ]
    post.side_effect = [Mock(json=Mock(return_value=payload)) for payload in payloads]
    model = GGUFLM(base_url="http://localhost:8080")
    assert model.score_continuation("prompt", " answer") == (-3.5, False)
    scoring = [call.kwargs["json"] for call in post.call_args_list[2:]]
    assert [request["prompt"] for request in scoring] == [[1, 2], [1, 2, 3]]
    assert all(request["post_sampling_probs"] is False for request in scoring)


def test_concurrent_generation_preserves_order_and_overlaps():
    barrier = Barrier(3)
    def complete(context, **kwargs):
        barrier.wait(timeout=5)  # Fails if requests run sequentially.
        return {"choices": [{"text": context}]}

    with GGUFLM(base_url="http://localhost:8080", request_workers=3) as model:
        model.gguf_completion = complete
        requests = [SimpleNamespace(args=(str(i), {})) for i in range(6)]
        assert model.generate_until(requests, disable_tqdm=True) == list(map(str, range(6)))


def test_workers_reuse_separate_sessions_and_close_them():
    barrier = Barrier(2)
    lock = Lock()
    sessions = []
    def session_factory():
        session = Mock()
        def post(*args, **kwargs):
            barrier.wait(timeout=5)
            return Mock(json=Mock(return_value={"choices": [{"text": "ok"}]}))
        session.post.side_effect = post
        with lock:
            sessions.append(session)
        return session

    with patch("src.gguf.requests.Session", side_effect=session_factory):
        with GGUFLM(base_url="http://localhost:8080", request_workers=2) as model:
            requests = [SimpleNamespace(args=("prompt", {})) for _ in range(4)]
            assert model.generate_until(requests, disable_tqdm=True) == ["ok"] * 4
    assert len(sessions) == 2
    for session in sessions:
        assert session.post.call_count == 2
        assert session.post.call_args.kwargs["timeout"] == (10, 300)
        session.close.assert_called_once()


def test_missing_prompt_logprobs_are_only_probed_once():
    with make_lm() as model:
        model.gguf_completion = Mock(return_value={"choices": [{"logprobs": None}]})
        model.score_continuation = Mock(return_value=(-2.0, True))
        requests = [SimpleNamespace(args=("prompt", " answer")) for _ in range(3)]
        assert model.loglikelihood(requests, disable_tqdm=True) == [(-2.0, True)] * 3
        assert model.gguf_completion.call_count == 1
        assert model.score_continuation.call_count == 3


def test_concurrent_request_failure_is_propagated():
    with GGUFLM(base_url="http://localhost:8080", request_workers=2) as model:
        model.gguf_completion = Mock(side_effect=RuntimeError("server failed"))
        with pytest.raises(RuntimeError, match="server failed"):
            model.generate_until([SimpleNamespace(args=("prompt", {}))], disable_tqdm=True)
