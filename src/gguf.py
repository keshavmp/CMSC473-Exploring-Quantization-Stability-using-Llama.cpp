
import logging
import time
import traceback
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
    while idx < len(offsets) and offsets[idx] < context_length:
        idx += 1

    continuation_logprobs = sum(
        p for p in tokens_logprobs[idx:-1] if p is not None
    )

    for i in range(idx, len(tokens)):
        token = tokens[i]
        top_tokens = logprobs["top_logprobs"][i]

        if not top_tokens:
            continue

        top_token = max(
            top_tokens.keys(),
            key=lambda x: top_tokens[x],
        )

        if top_token != token:
            is_greedy = False
            break

    return continuation_logprobs, is_greedy


@register_model("gguf", "ggml")
class GGUFLM(LM):

    def __init__(
        self,
        base_url=None,
        max_length=2048,
        request_workers=1,
        timeout=1200,
        **kwargs,
    ):
        super().__init__()

        self.base_url = base_url
        assert self.base_url, "must pass base_url to use GGUF LM!"

        self.logprobs = 10
        self.temperature = 0.0
        self.max_length = max_length

        if request_workers < 1 or timeout <= 0:
            raise ValueError(
                "request_workers and timeout must be positive"
            )

        self.request_workers = request_workers
        self.timeout = timeout

        self._local = local()
        self._sessions = []

        self._executor = ThreadPoolExecutor(
            max_workers=request_workers
        )

        self._prompt_logprobs_supported = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._executor.shutdown(
            wait=True,
            cancel_futures=True,
        )

        for session in self._sessions:
            session.close()

        self._sessions.clear()

    def _post(self, endpoint, payload):
        if not hasattr(self._local, "session"):
            self._local.session = requests.Session()
            self._sessions.append(self._local.session)

        response = self._local.session.post(
            f"{self.base_url.rstrip('/')}/{endpoint}",
            json=payload,
            timeout=(10, self.timeout),
        )

        response.raise_for_status()
        return response.json()

    def _map_requests(
        self,
        function,
        requests,
        disable_tqdm,
    ):
        results = []

        with tqdm(
            total=len(requests),
            disable=disable_tqdm,
        ) as progress:

            for start in range(
                0,
                len(requests),
                self.request_workers * 4,
            ):
                batch = requests[
                    start:start + self.request_workers * 4
                ]

                for result in self._executor.map(
                    function,
                    (req.args for req in batch),
                ):
                    results.append(result)
                    progress.update()

        return results

    def gguf_completion(
        self,
        context,
        continuation=None,
        stop=None,
        retries=3,
        delay=5,
        **kwargs,
    ):
        for attempt in range(retries):
            try:
                prompt = context

                request = {
                    "prompt": prompt,
                    "temperature": self.temperature,
                }

                if continuation:
                    prompt += continuation

                    request.update({
                        "prompt": prompt,
                        "max_tokens": 1,
                        "echo": True,
                        "logprobs": self.logprobs,
                    })

                else:
                    request["max_tokens"] = int(
                        kwargs.get("max_gen_toks", 512)
                    )

                if stop is not None:
                    request["stop"] = stop

                return self._post(
                    "v1/completions",
                    request,
                )

            except RequestException as error:
                logger.error(
                    "Request failed (attempt %d/%d): %s",
                    attempt + 1,
                    retries,
                    error,
                )

                if attempt < retries - 1:
                    time.sleep(delay)

        raise RuntimeError(
            f"Failed to get a valid response after {retries} retries."
        )

    # ---------------------------------------------------------
    # DEBUGGING: UNEXPECTED LOGLIKELIHOOD REQUESTS
    # ---------------------------------------------------------

    def loglikelihood(
        self,
        requests,
        disable_tqdm: bool = False,
    ):
        if not requests:
            return []

        stack = "".join(
            traceback.format_stack(limit=25)
        )

        raise RuntimeError(
            "\n\n"
            "========================================\n"
            "DEBUG: loglikelihood() WAS CALLED\n"
            "========================================\n"
            f"Number of requests: {len(requests)}\n"
            f"First request: {str(requests[0].args)[:500]}\n"
            "\nCALL STACK:\n"
            f"{stack}\n"
            "========================================\n"
        )

    def _loglikelihood_one(self, args):
        # Temporary diagnostic guard.
        # This function should not be reached by
        # the three generative benchmark tasks.

        raise RuntimeError(
            "Unexpected _loglikelihood_one() call.\n"
            f"Request arguments: {str(args)[:500]}\n"
            "Call stack:\n"
            + "".join(
                traceback.format_stack(limit=20)
            )
        )

    def score_continuation(
        self,
        context,
        continuation,
    ):
        """
        Disabled: the previous implementation used
        biased token forcing, which does not provide
        valid unmodified log-likelihood scores.
        """
        raise NotImplementedError(
            "Prompt log-likelihood scoring is not "
            "supported by this GGUF backend."
        )

    # ---------------------------------------------------------
    # TEXT GENERATION
    # ---------------------------------------------------------

    def generate_until(
        self,
        requests,
        disable_tqdm: bool = False,
    ):
        if not requests:
            return []

        print(
            f"DEBUG: generate_until() called "
            f"with {len(requests)} requests",
            flush=True,
        )

        return self._map_requests(
            self._generate_one,
            requests,
            disable_tqdm,
        )

    def _generate_one(self, request):
        inp = request[0]
        request_args = request[1]

        until = request_args.get(
            "until",
            ["</s>"],
        )

        response = self.gguf_completion(
            context=inp,
            stop=until,
            **{
                key: value
                for key, value in request_args.items()
                if key != "until"
            },
        )

        if (
            response
            and "choices" in response
            and response["choices"]
        ):
            choice = response["choices"][0]

            if "text" in choice:
                generated_text = choice["text"].strip()
                return generated_text

            logger.error(
                "Invalid response for generate_until: %s",
                response,
            )

            raise RuntimeError(
                f"Missing completion text: {response}"
            )

        logger.error(
            "Invalid completion response: %s",
            response,
        )

        raise RuntimeError(
            f"Invalid completion response: {response}"
        )

    def loglikelihood_rolling(
        self,
        requests,
        disable_tqdm: bool = False,
    ):
        raise NotImplementedError(
            "loglikelihood_rolling is not supported "
            "for GGUF models."
        )
