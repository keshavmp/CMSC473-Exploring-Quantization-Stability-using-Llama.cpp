"""Explicit task metric selection. Values in lm-eval result JSON are proportions."""
from __future__ import annotations
import math

TASKS=('gsm8k','ifeval','mmlu_college_computer_science_generative')
KEYS={
    'gsm8k': {
        'gsm8k_strict':'exact_match,strict-match',
        'gsm8k_flexible':'exact_match,flexible-extract',
    },
    'ifeval': {
        'ifeval_strict':'prompt_level_strict_acc,none',
        'ifeval_loose':'prompt_level_loose_acc,none',
        'ifeval_instruction_strict':'inst_level_strict_acc,none',
    },
    'mmlu_college_computer_science_generative': {
        'mmlu_generative':'exact_match,get_response',
    },
}
PRIMARY=('gsm8k_strict','ifeval_strict','mmlu_generative')

def extract(results:dict)->dict:
    """No positional metric guesses, no sample_len mistaken for accuracy."""
    out={}
    for task, keys in KEYS.items():
        data=results.get(task,{})
        for name,key in keys.items():
            val=data.get(key)
            if val is None:out[name]=None;continue
            val=float(val)
            if not math.isfinite(val) or not (0<=val<=1):
                raise ValueError(f'{task}/{key} invalid accuracy proportion: {val}')
            out[name]=100*val
        out[f'{task}_n']=data.get('sample_len')
    return out
