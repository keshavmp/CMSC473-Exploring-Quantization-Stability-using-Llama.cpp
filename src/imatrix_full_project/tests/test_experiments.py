"""Synthetic integration checks. Never treat synthetic scores as measurements."""
import importlib.util
from pathlib import Path
import pandas as pd

BASE=Path(__file__).resolve().parents[1]/'src'

def mod(name):
    spec=importlib.util.spec_from_file_location(name,BASE/f'{name}.py')
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def test_ablations_and_reports(tmp_path):
    a=mod('imatrix_ablations')
    r=mod('imatrix_results')
    out=tmp_path/'results'
    out.mkdir()
    tensors=[]
    layers=[]
    for domain in a.DOMAINS:
        for layer in range(4):
            layers.append({'layer':layer,'domain':domain,'heuristic_score':layer/10})
            for suffix in ('attn_q.weight','ffn_down.weight'):
                tensors.append({'tensor':f'blk.{layer}.{suffix}', 'layer':layer, 'domain':domain})
    pd.DataFrame(tensors).to_csv(out/'shared_tensor_features.csv',index=False)
    pd.DataFrame(layers).pivot(index='layer',columns='domain',values='heuristic_score').to_csv(out/'layer_scores.csv')
    plan=a.plan(out,tmp_path/'models',.25,'Q4_K','Q6_K',True)
    assert len(plan)==5+3+1+4
    assert (out/'overrides'/'mixed-math.txt').exists()
    assert (out/'overrides'/'p1-layer-00.txt').exists()
    records=[]
    for p in plan.itertuples():
        records.append({'model':p.name,'gsm8k':0.8-(.02 if p.method=='layer-ablation' else 0),
                        'ifeval':0.7-(.03 if p.method=='layer-ablation' else 0),
                        'mmlu':0.6-(.04 if p.method=='layer-ablation' else 0),
                        'size_mib':500 if p.method=='mixed-imatrix' else 550})
    scorespath=out/'benchmarks.csv'
    pd.DataFrame(records).to_csv(scorespath,index=False)
    scores=r.load_scores(scorespath)
    p1=r.p1_sensitivity(scores,plan,out)
    p2=r.p2_summary(scores,out)
    r.combined_report(out,p1,p2,scores)
    assert len(p1)==4
    assert abs(p1.iloc[0].sensitivity-.03)<1e-9
    assert (out/'p1_layer_ablation.png').is_file()
    assert (out/'p2_pareto.png').is_file()
    assert (out/'EXPERIMENT_REPORT.md').is_file()
