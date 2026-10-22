import logging
import time
from concurrent.futures import ThreadPoolExecutor
from threading import local

import requests
from requests.exceptions import RequestException
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


logger = logging.getLogger(__name__)


def get_result(logprobs, context_length):
    is_greedy = True
    offsets = logprobs["text_offset"]
    tokens = logprobs["tokens"]
    tokens_logprobs = logprobs["token_logprobs"]

    idx = 0
    while offsets[idx] < context_length:
        idx += 1
    continuation_logprobs = sum(tokens_logprobs[idx:-1])
    for i in range(idx, len(tokens)):
        token = tokens[i]
        top_tokens = logprobs["top_logprobs"][i]
        top_token = max(top_tokens.keys(), key=lambda x: top_tokens[x])
        if top_token != token:
            is_greedy = False
            break

    return continuation_logprobs, is_greedy


@register_model("gguf", "ggml")
class GGUFLM(LM):
    def __init__(self, base_url=None, max_length=2048, request_workers=1,
                 timeout=300, **kwargs):
        super().__init__()
        self.base_url = base_url
        assert self.base_url, "must pass `base_url` to use GGUF LM!"
        self.logprobs = 10
        self.temperature = 0.0
        self.max_length = max_length
        if request_workers < 1 or timeout <= 0:
            raise ValueError("request_workers and timeout must be positive")
        self.request_workers = request_workers
        self.timeout = timeout
        self._local = local()
        self._sessions = []
        self._executor = ThreadPoolExecutor(max_workers=request_workers)
        self._prompt_logprobs_supported = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._executor.shutdown(wait=True, cancel_futures=True)
        for session in self._sessions:
            session.close()
        self._sessions.clear()

    def _post(self, endpoint, payload):
        # Each worker owns its session; requests.Session is not shared by threads.
        if not hasattr(self._local, "session"):
            self._local.session = requests.Session()
            self._sessions.append(self._local.session)
        response = self._local.session.post(
            f"{self.base_url}/{endpoint}", json=payload, timeout=(10, self.timeout)
        )
        response.raise_for_status()
        return response.json()

    def _map_requests(self, function, requests, disable_tqdm):
        # Bound pending work, and return results in harness input order.
        results = []
        with tqdm(total=len(requests), disable=disable_tqdm) as progress:
            for start in range(0, len(requests), self.request_workers * 4):
                batch = requests[start:start + self.request_workers * 4]
                for result in self._executor.map(function, (req.args for req in batch)):
                    results.append(result)
                    progress.update()
        return results

    def gguf_completion(
        self, context, continuation=None, stop=None, retries=3, delay=5, **kwargs
    ):
        for _ in range(retries):
            try:
                prompt = context
                request = {
                    "prompt": prompt,
                    "temperature": self.temperature,
                }
                if continuation:
                    prompt += continuation
                    request.update({"prompt": prompt, "max_tokens": 1, "echo": True,
                                    "logprobs": self.logprobs})
                elif "max_gen_toks" in kwargs:
                    request["max_tokens"] = kwargs["max_gen_toks"]
                if stop is not None:
                    request["stop"] = stop
                return self._post("v1/completions", request)
            except RequestException as e:
                logger.error(f"RequestException: {e}")
                time.sleep(delay)  # wait before retrying
        else:
            raise RuntimeError(
                f"Failed to get a valid response after {retries} retries."
            )

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        if not requests:
            return []
        return self._map_requests(self._loglikelihood_one, requests, disable_tqdm)

    def _loglikelihood_one(self, args):
        context, continuation = args
        if self._prompt_logprobs_supported is False:
            return self.score_continuation(context, continuation)
        response = self.gguf_completion(context=context, continuation=continuation)
        if response and "choices" in response and response["choices"]:
            choice = response["choices"][0]
            logprobs = choice.get("logprobs")
            if (
                logprobs
                and "token_logprobs" in logprobs
                and logprobs["token_logprobs"]
            ):
                logprob, is_greedy = get_result(logprobs, len(context))
                return logprob, is_greedy
            else:
                # Recent llama-server builds ignore OpenAI echo/prompt logprobs.
                self._prompt_logprobs_supported = False
                return self.score_continuation(context, continuation)
        else:
            logger.error(
                f"Invalid response for loglikelihood. Response: {response}"
            )
            assert False

    def score_continuation(self, context, continuation):
        """Teacher-force each answer token and read its pre-sampling probability."""
        def post(endpoint, payload):
            return self._post(endpoint, payload)

        # Move trailing spaces into the continuation before tokenizing the pair.
        trailing = len(context) - len(context.rstrip())
        if trailing:
            continuation = context[-trailing:] + continuation
            context = context[:-trailing]
        prefix = post("tokenize", {"content": context, "add_special": True})["tokens"]
        tokens = post("tokenize", {"content": context + continuation, "add_special": True})["tokens"]
        total, greedy = 0.0, True
        for index in range(len(prefix), len(tokens)):
            token = tokens[index]
            result = post("completion", {
                "prompt": tokens[:index], "n_predict": 1, "temperature": 0,
                "n_probs": 1, "post_sampling_probs": False,
                "logit_bias": [[token, 1000]],
            })["completion_probabilities"][0]
            if result["id"] != token:
                raise RuntimeError("llama-server did not score the requested answer token")
            total += result["logprob"]
            greedy &= result["top_logprobs"][0]["id"] == token
        return total, greedy

    def generate_until(self, requests, disable_tqdm: bool = False):
        if not requests:
            return []

        return self._map_requests(self._generate_one, requests, disable_tqdm)

    def _generate_one(self, request):
        inp = request[0]
        request_args = request[1]
        until = request_args.get("until", ["</s>"])
        response = self.gguf_completion(context=inp, stop=until, **{
            key: value for key, value in request_args.items() if key != "until"
        })
        if response and "choices" in response and response["choices"]:
            choice = response["choices"][0]
            if "text" in choice:
                generated_text = choice["text"].strip()
                return generated_text
            else:
                logger.error(
                    f"Invalid response for greedy_until. Response: {response}"
                )
                raise RuntimeError(f"Missing completion text: {response}")
        else:
            logger.error(f"Invalid response for greedy_until. Response: {response}")
            raise RuntimeError(f"Invalid completion response: {response}")

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "loglikelihood_rolling not yet supported for GGUF models"
        )
