"""Create actual-score CSV/plots with controlled BF16 deltas."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .metrics import PRIMARY

def build_reports(data:pd.DataFrame,out:Path):
    data=data.copy()
    if 'bf16' in set(data.model):
        base=data[data.model=='bf16'].iloc[0]
        for col in PRIMARY:
            data[col+'_drop_pp']=base[col]-data[col]
            data[col+'_retained_pct']=np.where(base[col]>0,100*data[col]/base[col],np.nan)
        data['compression_ratio']=base.size_mib/data.size_mib
    data['mean_primary_accuracy']=data[list(PRIMARY)].mean(axis=1)
    data.to_csv(out/'accuracy_retention.csv',index=False)
    fig,ax=plt.subplots(figsize=(max(9,len(data)*.8),5))
    for offset,col in zip((-.23,0,.23),PRIMARY):
        ax.scatter(np.arange(len(data))+offset,data[col],label=col,s=35)
    ax.set_xticks(np.arange(len(data)),data.model,rotation=45,ha='right')
    ax.set_ylim(0,100)
    ax.set(ylabel='Accuracy (%)',title='Measured accuracy; consistent generative protocol')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(out/'benchmark_accuracy.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    ax.scatter(data.size_mib,data.mean_primary_accuracy)
    for row in data.itertuples():ax.annotate(row.model,(row.size_mib,row.mean_primary_accuracy),fontsize=8)
    ax.set(xlabel='GGUF file size (MiB)',ylabel='Average of three primary task accuracies (%)',title='File size vs measured quality')
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'size_quality.png',dpi=180);plt.close(fig)
    if 'bf16' in set(data.model):
        diff=data[data.model!='bf16'].set_index('model')[[c+'_drop_pp' for c in PRIMARY]]
        if not diff.empty:
            ax=diff.plot.bar(figsize=(10,5))
            ax.axhline(0,color='black');ax.set_ylabel('Accuracy loss vs BF16 (percentage points)')
            plt.xticks(rotation=45,ha='right');plt.tight_layout();plt.savefig(out/'accuracy_delta.png',dpi=180);plt.close()
    lines=['# Imatrix benchmark comparison','','Measured scores only. No synthetic accuracies or forced-token probabilities.','',
        '## Evaluation','','- GSM8K strict exact match (flexible extraction also saved)',
        '- IFEval prompt-level strict accuracy (loose/instruction-level also saved)',
        '- MMLU College CS **generative** exact match; NOT standard multiple-choice likelihood accuracy',
        '- Scores are percentages; identical task sample indices and seed across models',
        '- A short generation cap can truncate answers; smoke-test accuracy is not a research claim',
        '- Model size is GGUF file bytes, not peak RAM or inference throughput','',
        '## Results','', '```text',data.round(3).to_string(index=False),'```','']
    if 'bf16' not in set(data.model):lines.append('BF16 baseline not yet measured; no accuracy-loss or recovery conclusion is possible.')
    else:lines.append('Accuracy drop and retention relative to BF16 are in accuracy_retention.csv.')
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
