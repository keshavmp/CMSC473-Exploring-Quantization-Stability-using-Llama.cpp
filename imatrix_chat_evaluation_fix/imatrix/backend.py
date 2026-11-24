"""Generative lm-eval GGUF backend. Does NOT fabricate token likelihoods."""
from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor
from threading import local, Lock
from typing import Any
import requests
from lm_eval.api.model import LM

class GGUFGenerationLM(LM):
    def __init__(self, base_url: str, timeout: float=900, workers: int=1,
                 max_new_tokens: int=512, retries: int=0, prompt_format: str="chat"):
        super().__init__()
        if prompt_format not in ('chat', 'raw'):
            raise ValueError('prompt_format must be chat or raw')
        self.prompt_format=prompt_format
        self.url=base_url.rstrip('/') + (
            '/v1/chat/completions' if prompt_format=='chat' else '/v1/completions'
        )
        self.timeout=timeout
        self.workers=workers
        self.max_new_tokens=max_new_tokens
        self.retries=retries
        self._local=local()
        self._lock=Lock()
        self.requests_completed=0
        self.completion_tokens=0
        self.elapsed_generation_seconds=0.0

    def loglikelihood(self, requests, disable_tqdm=False):
        if not requests:return []
        raise NotImplementedError('This evaluator supports generation only. Use generative tasks; likelihood scores require a true scoring backend.')

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        raise NotImplementedError('Rolling likelihood is unsupported; never use logit_bias to force target tokens.')

    def _one(self, req:Any):
        prompt, opts=req.args
        opts=dict(opts)
        until=opts.get('until',[])
        if isinstance(until,str):until=[until]
        stops=[s for s in until if isinstance(s,str) and s]
        cap=int(opts.get('max_gen_toks',self.max_new_tokens))
        # Explicit global safety cap for smoke tests; benchmark prompts cannot override it.
        cap=min(cap,self.max_new_tokens)
        if cap<1: raise ValueError('Generation max tokens must be >= 1')
        # Chat endpoint formats this user prompt with the GGUF model's chat template.
        # Preserve the original lm-eval task prompt; do not rewrite answers or labels.
        payload={'temperature':0,'max_tokens':cap}
        if self.prompt_format=='chat':
            payload['messages']=[{'role':'user','content':prompt}]
        else:
            payload['prompt']=prompt
        if stops:payload['stop']=stops
        if not hasattr(self._local,'session'):self._local.session=requests.Session()
        started=time.monotonic()
        for attempt in range(self.retries+1):
            try:
                resp=self._local.session.post(self.url,json=payload,timeout=(10,self.timeout))
                resp.raise_for_status()
                output=resp.json()
                choices=output.get('choices') or []
                if not choices:
                    raise RuntimeError(f'No completion choices: {str(output)[:250]}')
                choice=choices[0]
                if self.prompt_format=='chat':
                    message=choice.get('message') or {}
                    generated=message.get('content')
                else:
                    generated=choice.get('text')
                if not isinstance(generated,str):
                    raise RuntimeError(f'Unexpected server response: {str(output)[:250]}')
                usage=output.get('usage') or {}
                tokens=int(usage.get('completion_tokens') or 0)
                with self._lock:
                    self.requests_completed+=1
                    self.completion_tokens+=tokens
                    self.elapsed_generation_seconds+=time.monotonic()-started
                return generated
            except requests.RequestException:
                if attempt>=self.retries:raise
                time.sleep(2)
        raise RuntimeError('Unreachable')

    def generate_until(self, requests, disable_tqdm=False):
        if not requests:return []
        from tqdm.auto import tqdm
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            results=[]
            for result in tqdm(pool.map(self._one,requests),total=len(requests),disable=disable_tqdm,desc='Generating'):
                results.append(result)
            return results
