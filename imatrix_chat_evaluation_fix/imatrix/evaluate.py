"""Measure actual generative task scores per GGUF, one model at a time.

For a quick test use --limit 1 --models uniform-q4 --max-new-tokens 64.
For research keep --max-new-tokens consistent across all models; save BF16 separately.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import pandas as pd

from .metrics import TASKS, PRIMARY, extract
from .server import discover_server,managed_server

ROOT=Path(__file__).resolve().parents[1]

def inventory(root:Path)->dict[str,Path]:
    result={}
    base=root/'models/Llama-3.2-1B-Instruct-BF16.gguf'
    if base.is_file():result['bf16']=base
    # Existing recovery and earlier pilot checkpoints; preserve their names.
    sources=[root/'models/imatrix_experiments',root/'models/imatrix_quants']
    aliases={
        'chat-q4':'Llama-3.2-1B-Instruct-chat_q4_imatrix.gguf',
        'math-q4':'Llama-3.2-1B-Instruct-math_q4_imatrix.gguf',
        'code-q4':'Llama-3.2-1B-Instruct-code_q4_imatrix.gguf',
        'mixed-q4-q6':'Llama-3.2-1B-Instruct-chat_mixed_q4_q6.gguf',
        'uniform-q4':'Llama-3.2-1B-Instruct-uniform_q4.gguf',
        'uniform-q6':'Llama-3.2-1B-Instruct-uniform_q6.gguf',
        'chat-q6':'Llama-3.2-1B-Instruct-chat_q6_imatrix.gguf',
        'math-q6':'Llama-3.2-1B-Instruct-math_q6_imatrix.gguf',
        'code-q6':'Llama-3.2-1B-Instruct-code_q6_imatrix.gguf',
    }
    for folder in sources:
        if not folder.is_dir():continue
        for path in sorted(folder.glob('*.gguf')):
            name=path.stem
            # Prefer human-readable names for common previously-produced artifacts.
            for alias,fname in aliases.items():
                if path.name==fname:name=alias;break
            # Recognize both old and newly-generated uniform names.
            if path.name=='uniform-q4.gguf':name='uniform-q4'
            if path.name=='uniform-q6.gguf':name='uniform-q6'
            result.setdefault(name,path)
    return result

def load_results(folder:Path, models:dict[str,Path]):
    records=[]
    for name,path in models.items():
        metrics=folder/name/'metrics.json'
        if not metrics.exists():continue
        with metrics.open(encoding='utf-8') as f:raw=json.load(f)
        record={'model':name,'size_mib':round(path.stat().st_size/(1024**2),3)}
        record.update(extract(raw))
        perf=folder/name/'performance.json'
        if perf.exists():record.update(json.loads(perf.read_text(encoding='utf-8')))
        records.append(record)
    return pd.DataFrame(records)

def write_reports(folder:Path,models:dict[str,Path]):
    from .reports import build_reports
    data=load_results(folder,models)
    if data.empty:
        print('No completed results to summarize.');return
    data.to_csv(folder/'benchmark_scores.csv',index=False)
    build_reports(data,folder)
    print('Saved',folder/'benchmark_scores.csv',flush=True)

def run_one(name:str,path:Path,folder:Path,args):
    import lm_eval
    from .backend import GGUFGenerationLM
    from lm_eval.tasks import TaskManager
    start=time.monotonic()
    manager=TaskManager()
    tasks=manager.load(list(TASKS))['tasks']
    for task_name,task in tasks.items():
        if task.dump_config().get('output_type')!='generate_until':
            raise ValueError(f'{task_name} is not generative; cannot evaluate with this backend')
    sample_indices={}
    for task_name,task in tasks.items():
        available=len(task.eval_docs)
        cap=available if args.limit==0 else min(args.limit,available)
        sample_indices[task_name]=list(range(cap))
        print(f'{task_name}: {cap}/{available} examples',flush=True)
    out=folder/name
    out.mkdir(parents=True,exist_ok=True)
    with managed_server(path,args.server,args.port,args.threads,args.context,args.gpu_layers,out/'server.log') as url:
        model=GGUFGenerationLM(base_url=url,timeout=args.timeout,workers=1,max_new_tokens=args.max_new_tokens,retries=args.retries,prompt_format=args.prompt_format)
        result=lm_eval.simple_evaluate(
            model=model,
            tasks=list(tasks.values()),
            task_manager=manager,
            samples=sample_indices,
            num_fewshot=0,
            random_seed=0,
            numpy_random_seed=1234,
            torch_random_seed=1234,
            fewshot_random_seed=1234,
            batch_size=1,
            gen_kwargs={'temperature':0,'do_sample':False},
            log_samples=True,
        )
    if not isinstance(result,dict) or 'results' not in result:
        raise RuntimeError('lm-eval returned no result metrics')
    mapped={task_name:result['results'].get(task_name,{}) for task_name in TASKS}
    # Validate metric selection BEFORE marking the model as complete.
    selected=extract(mapped)
    for primary in PRIMARY:
        if selected[primary] is None:raise RuntimeError(f'Missing benchmark metric {primary}; saved nothing for this model')
    (out/'samples.json').write_text(
        json.dumps(result.get('samples',{}),indent=2,ensure_ascii=False,default=str)+'\n',
        encoding='utf-8'
    )
    (out/'metrics.json').write_text(json.dumps(mapped,indent=2,default=str)+'\n',encoding='utf-8')
    performance={
        'evaluation_seconds':round(time.monotonic()-start,3),
        'completed_requests':model.requests_completed,
        'reported_completion_tokens':model.completion_tokens,
        'seconds_per_request':round(model.elapsed_generation_seconds/max(model.requests_completed,1),3),
        'tokens_per_second':round(model.completion_tokens/max(model.elapsed_generation_seconds,1e-9),3) if model.completion_tokens else None,
    }
    (out/'performance.json').write_text(json.dumps(performance,indent=2)+'\n',encoding='utf-8')
    (out/'protocol.json').write_text(json.dumps({
        'tasks':list(TASKS),'limit_per_task':args.limit,'max_new_tokens':args.max_new_tokens,
        'context_size':args.context,'fewshot':0,'temperature':0,'model':str(path.resolve()),
        'threads':args.threads,'gpu_layers':args.gpu_layers,
        'prompt_format':args.prompt_format,'benchmark_metrics':{
            'gsm8k':['strict-match','flexible-extract'],
            'ifeval':['prompt_level_strict_acc'],
            'mmlu_college_computer_science_generative':['get_response']
        },
    },indent=2)+'\n',encoding='utf-8')
    print('Completed:',name,selected,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--experiment',default='imatrix-pilot')
    p.add_argument('--models',nargs='+',help='Names from --list-models; default all (BF16 last)')
    p.add_argument('--list-models',action='store_true')
    p.add_argument('--limit',type=int,default=2,help='Examples per task; 0=full benchmark')
    p.add_argument('--max-new-tokens',type=int,default=512)
    p.add_argument('--prompt-format',choices=['chat','raw'],default='chat',
                   help='chat applies model GGUF template via /v1/chat/completions; raw uses plain completions')
    p.add_argument('--context',type=int,default=2048)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--gpu-layers',type=int,default=0)
    p.add_argument('--port',type=int,default=8080)
    p.add_argument('--timeout',type=int,default=900)
    p.add_argument('--retries',type=int,default=0,help='Default 0 avoids repeating long timed-out prompts')
    p.add_argument('--server',type=str)
    p.add_argument('--rerun',action='store_true')
    p.add_argument('--report-only',action='store_true')
    a=p.parse_args()
    root=a.root.expanduser().resolve()
    models=inventory(root)
    if a.list_models:
        for name,path in models.items():print(f'{name}: {path}')
        return
    if a.models:
        missing=set(a.models)-set(models)
        if missing:p.error(f'Models not found: {sorted(missing)}; use --list-models')
        models={name:models[name] for name in a.models}
    else:
        models={name:models[name] for name in sorted(models,key=lambda x:(x=='bf16',x))}
    if not models:p.error('No GGUF models discovered')
    if a.limit<0 or a.threads<1 or a.max_new_tokens<1:p.error('Invalid limit, threads, or max-new-tokens')
    out=root/'results'/a.experiment
    out.mkdir(parents=True,exist_ok=True)
    if a.report_only:
        write_reports(out,models);return
    a.server=discover_server(a.server)
    failed=[]
    for name,path in models.items():
        if (out/name/'metrics.json').is_file() and not a.rerun:
            protocol_file=out/name/'protocol.json'
            prior=json.loads(protocol_file.read_text()) if protocol_file.is_file() else {}
            expected={'prompt_format':a.prompt_format,'limit_per_task':a.limit,
                      'max_new_tokens':a.max_new_tokens,'fewshot':0,
                      'context_size':a.context}
            differences={k:(prior.get(k),v) for k,v in expected.items() if prior.get(k)!=v}
            if differences:
                raise SystemExit(
                    f'Existing results for {name} use different settings: {differences}. '
                    'Use a NEW --experiment name or --rerun to replace them.'
                )
            print('Reusing completed result:',name,flush=True);continue
        print('\nEvaluating',name,flush=True)
        try:run_one(name,path,out,a)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            failed.append((name,str(exc)))
            print('Failed:',name,exc,flush=True)
    write_reports(out,models)
    if failed:raise SystemExit('Incomplete models: '+', '.join(n for n,_ in failed))

if __name__=='__main__':main()
