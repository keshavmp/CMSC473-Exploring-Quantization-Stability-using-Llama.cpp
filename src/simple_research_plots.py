"""One metric per figure, compared across precision, family, task or text domain."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.research_questions import ROOT, PRECISIONS, TASKS, SWEEPS, sha

OUT=ROOT/'results/research_questions/simple_comparisons'
FAMILIES=['Llama','Qwen','Gemma']
FAMILY_COLORS={'Llama':'#2878ad','Qwen':'#db8730','Gemma':'#49936a'}
DOMAINS=['code','math','chat']
DOMAIN_COLORS={'code':'#2878ad','math':'#db8730','chat':'#49936a'}
TASK_LABELS={'gsm8k':'GSM8K · math reasoning','ifeval':'IFEval · instructions','hellaswag':'HellaSwag · commonsense'}
SOURCES={}


def csv(path):
    SOURCES[str(path.relative_to(ROOT))]=sha(path)
    return pd.read_csv(path)


def read_json(path):
    SOURCES[str(path.relative_to(ROOT))]=sha(path)
    return json.loads(path.read_text(encoding='utf-8'))


def finish(fig,axes,title,ylabel,footer,name,legend='Model family'):
    for ax in np.asarray(axes).reshape(-1):
        ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    for ax in np.asarray(axes).reshape(-1):ax.set_ylabel('')
    fig.supylabel(ylabel,fontsize=12,x=.008)
    handles,labels=np.asarray(axes).reshape(-1)[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.91),ncol=3,frameon=False,title=legend)
    fig.suptitle(title,fontsize=18,fontweight='bold',y=.99)
    fig.text(.5,.015,footer,ha='center',va='bottom',fontsize=9,color='#444444')
    fig.tight_layout(rect=(.025,.12,1,.88))
    fig.savefig(OUT/f'{name}.png',dpi=180,bbox_inches='tight');plt.close(fig)


def accuracy():
    rows=[]
    for model,folder in SWEEPS.items():
        base=(ROOT/'results/Pure_Sweep/BF16-pure-sweep' if model=='Llama'
              else ROOT/'results'/folder/'source-reference')
        source=read_json(base/'metrics.json')
        for precision in ['Source']+PRECISIONS:
            metrics=source if precision=='Source' else read_json(ROOT/'results'/folder/f'mixed-{precision.lower()}-control/metrics.json')
            for task,(metric,filt,_,_) in TASKS.items():
                score=100*metrics[task][metric+','+filt]
                rows.append({'model':model,'precision':precision,'task':task,'accuracy_pct':score,
                             'source_accuracy_pct':100*source[task][metric+','+filt]})
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'accuracy.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(13,5.4),sharey=True)
    order=['Source']+PRECISIONS
    for ax,task in zip(axes,TASKS):
        for model in FAMILIES:
            g=frame[frame.task.eq(task)&frame.model.eq(model)].set_index('precision').loc[order]
            # Source is a reference marker; the precision curve starts at Q2.
            ax.scatter([0],[g.loc['Source','accuracy_pct']],color=FAMILY_COLORS[model],marker='D',s=32)
            ax.plot(range(1,7),g.loc[PRECISIONS,'accuracy_pct'],'o-',color=FAMILY_COLORS[model],label=model)
        ax.set(xticks=range(7),xticklabels=['Source','Q2','Q3','Q4','Q5','Q6','Q8'],title=TASK_LABELS[task],ylim=(0,100),xlabel='Weight precision')
        ax.tick_params(axis='x',labelsize=10)
    finish(fig,axes,'1. How does accuracy change with precision?','Correct answers (%)',
           'Higher is better. Each panel shows one task; every line shows one model family.\nSource = original BF16 (Llama/Gemma) or F16 (Qwen). Quantized points are no-imatrix standard recipes; descriptive single executions.',
           '01_accuracy_by_precision')
    # Use a separate figure for loss: never put accuracy and accuracy loss on one axis.
    fig,axes=plt.subplots(1,3,figsize=(13,5.4),sharey=True)
    for ax,task in zip(axes,TASKS):
        for model in FAMILIES:
            g=frame[frame.task.eq(task)&frame.model.eq(model)].set_index('precision').loc[PRECISIONS]
            ax.plot(range(6),g.source_accuracy_pct-g.accuracy_pct,'o-',color=FAMILY_COLORS[model],label=model)
        ax.axhline(0,color='#555555',lw=1)
        ax.set(xticks=range(6),xticklabels=PRECISIONS,title=TASK_LABELS[task],xlabel='Weight precision')
    finish(fig,axes,'1. How much accuracy is lost?','Accuracy lost versus source (percentage points)',
           'Lower is better. Zero = source accuracy; negative = observed accuracy above source.\nEach family uses its own original source. Standard recipes; historical source references; no repeated-execution uncertainty.',
           '02_accuracy_loss_by_precision')


def task_gains():
    frame=csv(ROOT/'results/research_questions/imatrix_task_effects.csv')
    averaged=frame.groupby(['model','precision','task'],as_index=False).gain_pp.mean()
    averaged.to_csv(OUT/'imatrix_accuracy_gain.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(13,5.4),sharey=True)
    for ax,task in zip(axes,TASKS):
        for model in FAMILIES:
            g=averaged[averaged.model.eq(model)&averaged.task.eq(task)].set_index('precision').loc[PRECISIONS]
            ax.plot(range(6),g.gain_pp,'o-',color=FAMILY_COLORS[model],label=model)
        ax.axhline(0,color='#555555',lw=1)
        ax.set(xticks=range(6),xticklabels=PRECISIONS,title=TASK_LABELS[task],xlabel='Weight precision')
    finish(fig,axes,'2. How much accuracy does imatrix recover?','Accuracy gained over control (percentage points)',
           'Above zero = improvement; below zero = worsening. Each point averages code/math/chat calibration gains equally.\nMatched same-precision controls. Averages are descriptive, not independent repeats; detailed paired intervals are in the parent folder.',
           '03_imatrix_accuracy_gain')


def kl_recovery():
    frame=csv(ROOT/'results/imatrix_recovery_tables/output_by_domain.csv')
    # Equal coefficients on calibration variants; divide only after pooling errors.
    frame=frame.groupby(['model','precision','domain'],as_index=False)[['error','control_error']].sum()
    frame['recovery_pct']=100*(1-frame.error/frame.control_error)
    frame.to_csv(OUT/'kl_recovery.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(13,5.4),sharey=True)
    for ax,domain in zip(axes,DOMAINS):
        for model in FAMILIES:
            g=frame[frame.model.eq(model)&frame.domain.eq(domain)].set_index('precision').loc[PRECISIONS]
            ax.plot(range(6),g.recovery_pct,'o-',color=FAMILY_COLORS[model],label=model)
        ax.axhline(0,color='#555555',lw=1)
        ax.set(xticks=range(6),xticklabels=PRECISIONS,title=f'{domain.title()} text',xlabel='Weight precision')
    finish(fig,axes,'2. How much output KL does imatrix remove?','Control output KL removed (%)',
           'Higher is better. Zero = unchanged; negative = worse. Calibration errors are pooled before computing recovery.\n1,020 scored tokens per domain and model; tokenizers differ. KL recovery measures output-distribution preservation, not accuracy.',
           '04_imatrix_kl_recovery')


def block_recovery():
    frame=csv(ROOT/'results/imatrix_recovery_tables/block_recovery.csv')
    frame.to_csv(OUT/'block_recovery.csv',index=False)
    fig,axes=plt.subplots(2,3,figsize=(13,8),sharex=True,sharey=True)
    for ax,precision in zip(axes.flat,PRECISIONS):
        for model in FAMILIES:
            g=frame[frame.model.eq(model)&frame.precision.eq(precision)].sort_values('depth_pct')
            ax.plot(g.depth_pct,g.recovery_pct,'o-',markersize=2.7,lw=1.5,color=FAMILY_COLORS[model],label=model)
        ax.axhline(0,color='#555555',lw=.8)
        ax.set(title=precision,xticks=[0,50,100],xlabel='Block position through model (%)')
    finish(fig,axes,'3. Which blocks benefit across model families?','Original local error removed (%)',
           'Higher is more recovery. Each panel shows one precision; each line shows one family.\nCode/math/chat calibration and measurement errors are pooled before division. Models have different block counts; no interpolation is used.',
           '05_block_recovery_across_families')


def calibration_blocks():
    from itertools import combinations
    frame=csv(ROOT/'results/imatrix_recovery_tables/local_tensor_detail.csv')
    frame=frame[frame.precision.eq('Q4')].groupby(
        ['model','precision','variant','domain','layer','depth_pct'],as_index=False)[['weighted_error','control_error']].sum()
    frame['recovery_pct']=100*(1-frame.weighted_error/frame.control_error)
    frame.to_csv(OUT/'q4_calibration_block_recovery.csv',index=False)
    profiles=[]
    for (model,domain),group in frame.groupby(['model','domain']):
        for a,b in combinations(DOMAINS,2):
            x,y=[group[group.variant.eq(v)].set_index('layer').recovery_pct for v in [a,b]]
            profiles.append({'model':model,'measurement_domain':domain,'calibration_a':a,'calibration_b':b,
                             'spearman_recovery_rank_agreement':x.corr(y,method='spearman')})
    pd.DataFrame(profiles).to_csv(OUT/'q4_calibration_profile_agreement.csv',index=False)
    fig,axes=plt.subplots(3,3,figsize=(13,10),sharey=True)
    for row,model in enumerate(FAMILIES):
        for col,domain in enumerate(DOMAINS):
            ax=axes[row,col]
            for variant in DOMAINS:
                g=frame[frame.model.eq(model)&frame.domain.eq(domain)&frame.variant.eq(variant)].sort_values('layer')
                ax.plot(g.layer,g.recovery_pct,'o-',markersize=2.5,color=DOMAIN_COLORS[variant],label=variant.title())
            ax.axhline(0,color='#555555',lw=.8)
            ax.set(title=f'{model} · measured on {domain}',xlabel='Transformer block')
            last=int(g.layer.max())
            ax.set_xticks(sorted(set(list(range(0,last+1,5))+[last])))
    finish(fig,axes,'3. Does calibration data change which blocks recover?','Original local error removed (%)',
           'Q4 | Rows = model families; columns = measurement domains; lines = calibration domains. Native block indices are retained.\nHigher means more proportional local reconstruction recovery. Historical calibration budgets differ; content and quantity are confounded.',
           '06_calibration_block_recovery',legend='Imatrix calibration data')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.titlesize':12})
    accuracy();task_gains();kl_recovery();block_recovery();calibration_blocks()
    notes=['# Simple comparison plots','',
           '**One metric per figure.** Model-family colors stay blue (Llama), orange (Qwen), green (Gemma).',
           'Figure 06 instead colors calibration domains, explicitly labeled in its legend.','',
           '| Figure | One quantity measured | What to compare |',
           '|---|---|---|',
           '| 01 | Accuracy (%) | Precisions and families, separated by benchmark task |',
           '| 02 | Accuracy lost from source (pp) | Precisions and families, separated by benchmark task |',
           '| 03 | Accuracy gained from imatrix (pp) | Precisions and families, separated by benchmark task |',
           '| 04 | Output KL removed (%) | Precisions and families, separated by measurement domain |',
           '| 05 | Local reconstruction error removed (%) | Blocks and families, separated by precision |',
           '| 06 | Local reconstruction error removed (%) | Blocks and calibration datasets, separated by model family and measurement domain (Q4) |','',
           'At Q4, block-recovery rankings are similar across calibration choices: mean pairwise Spearman agreement is approximately 0.994 (Llama), 0.914 (Qwen), and 0.971 (Gemma). These are descriptive averages across related groups, not significance tests or proof of dataset-specific targeting.',
           'These figures summarize saved historical results. They do not include the unfinished equal-budget calibration or execution-repeat study.',
           'Source checkpoints differ in size, architecture, and training; family differences cannot be assigned solely to architecture.',
           'Figures 01/02 use standard mixed-recipe no-imatrix controls; they are a separate series from the pure Llama sweep.',
           'Question-resampling intervals are in the parent results folder. Simple averages here are descriptive and hide calibration-specific variation.',
           'All CSVs preserve plotted values. Rebuild: `python -m src.simple_research_plots`.']
    (OUT/'README.md').write_text('\n'.join(notes)+'\n',encoding='utf-8')
    manifest={'inputs':SOURCES,'code_sha256':sha(Path(__file__)),
              'outputs':{p.name:sha(p) for p in OUT.iterdir() if p.suffix in ['.csv','.png','.md']}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(f'Wrote 6 simple comparison figures: {OUT}',flush=True)


if __name__=='__main__':main()
