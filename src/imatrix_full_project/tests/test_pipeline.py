"""Synthetic tests only: do not represent actual model measurements."""
import importlib.util
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd

MODULE_PATH=Path(__file__).resolve().parents[1]/'src'/'imatrix_research.py'
spec=importlib.util.spec_from_file_location('imatrix_research',MODULE_PATH)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_pipeline():
    with tempfile.TemporaryDirectory() as tmp:
        out=Path(tmp)/'results'
        def fake_domain(path, domain):
            shift={'chat':0.0,'math':0.3,'code':-0.2}[domain]
            return pd.DataFrame([
                dict(domain=domain,tensor=f'blk.{i}.{kind}.weight',layer=i,
                     coordinates=20,observations=200,
                     mean_energy=(i+1)*(.2 if kind=='attn_q' else .4)+shift+.5,
                     median_energy=.3,max_energy=1,std_energy=.3,p95_energy=.7)
                for i in range(6) for kind in ['attn_q','ffn_up']])
        with patch.object(m,'load_domain',side_effect=fake_domain):
            ranks=m.analyze(Path(tmp),out)
        assert len(ranks)==6
        assert (out/'layer_importance.png').is_file()
        assert (out/'tensor_statistics.csv').is_file()
        m.recommend(out,.3,'Q4_K','Q6_K')
        assert (out/'tensor_types.txt').is_file()
        assert len(pd.read_csv(out/'precision_plan.csv'))==12
        p1=Path(tmp)/'p1.csv'
        pd.DataFrame({'layer':range(6),'sensitivity':range(6)}).to_csv(p1,index=False)
        m.validate(out,p1)
        assert (out/'p1_validation.png').is_file()
        scores=Path(tmp)/'scores.csv'
        pd.DataFrame([
            dict(model='a',method='uniform',size_mib=500,gsm8k=.4,ifeval=.5,mmlu=.6),
            dict(model='b',method='mixed',size_mib=525,gsm8k=.6,ifeval=.5,mmlu=.7)
        ]).to_csv(scores,index=False)
        m.compare(out,scores)
        assert (out/'pareto_frontier.png').is_file()

if __name__=='__main__':
    test_pipeline()
    print('Synthetic pipeline tests passed')
