"""Build manuscript figures and LaTeX tables from verified experimental CSVs."""
from pathlib import Path
import json, re, os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/tmlr-mpl')
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

HERE=Path(__file__).resolve().parent; OUT=HERE.parent; ROOT=Path(os.environ.get('EXPERIMENT_RESULTS_DIR',str(OUT.parent)))
S=pd.read_csv(HERE/'metric_summary.csv'); P=pd.read_csv(HERE/'paired_main.csv')
L=pd.read_csv(HERE/'lr_stability.csv'); W=pd.read_csv(HERE/'lr_windows.csv')
C=pd.read_csv(HERE/'calibration_choices.csv'); RP=pd.read_csv(HERE/'paired_ratios.csv')
ORDER=['relu','leaky_relu','tanh','sigmoid','tanh+relu','tanh+leaky_relu','sigmoid+relu']
NAMES=['ReLU','Leaky-ReLU','tanh','sigmoid','tanh + ReLU','tanh + leaky-ReLU','sigmoid + ReLU']
NAME=dict(zip(ORDER,NAMES)); SHORT=dict(zip(ORDER,['R','L','T','S','T+R','T+L','S+R']))
COL=dict(zip(ORDER,['#263b53','#5b8cb1','#a56a24','#a78cae','#007e75','#8aa535','#ce6a57']))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':180})
raw=[]
for f in sorted(ROOT.glob('*.csv')):
    x=pd.read_csv(f); x=x[x.seed.astype(str)!='seed'].copy()
    for c in ['seed','hidden','ratio','lr','val_acc','best_val_acc','test_acc_best','test_f1_best','test_acc','test_f1','seconds','best_val_epoch']:
        x[c]=pd.to_numeric(x[c])
    x['stage']=re.sub(r'^(c10_)?(.*?)_\d+\.csv$',r'\2',f.name)
    x['config']=np.where(x.ratio.eq(0),x.act1,x.act1+'+'+x.act2)
    raw.append(x)
D=pd.concat(raw,ignore_index=True)
def sel(df,**kwargs):
    for k,v in kwargs.items(): df=df[df[k].eq(v)]
    return df
def save(fig,name):
    fig.savefig(OUT/'figures'/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(OUT/'figures'/f'{name}.png',bbox_inches='tight')
    plt.close(fig)
def legend(fig,ax,ncol=4):
    h,l=ax.get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=ncol,frameon=False,bbox_to_anchor=(.5,0))
def fmt(x,n=3):return f'{x:.{n}f}'
def ci(a,b):return f'[{a:+.3f}, {b:+.3f}]'
def rate(v):return f'{v:g}'
def pair(a,b,metric='test_acc_best'):
    z=a[['seed',metric]].merge(b[['seed',metric]],on='seed',suffixes=('_a','_b'),validate='one_to_one').sort_values('seed')
    d=(z[metric+'_a']-z[metric+'_b']).to_numpy();r=np.random.default_rng(20260915)
    lo,hi=np.quantile(d[r.integers(0,len(d),(10000,len(d)))].mean(1),[.025,.975])
    return d.mean(),lo,hi
def table(name,caption,label,heads,rows,align=None,note='',size='small'):
    align=align or ('l'+'r'*(len(heads)-1))
    text='\\begin{table}[htbp]\n\\centering\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\'+size+'\n'
    text+='\\begin{tabular}{@{}'+align+'@{}}\n\\toprule\n'+' & '.join(heads)+r' \\'+'\n\\midrule\n'
    text+='\n'.join(' & '.join(map(str,r))+r' \\' for r in rows)
    text+='\n\\bottomrule\n\\end{tabular}\n'
    if note:text+='\\par\\smallskip{\\footnotesize '+note+'}\n'
    text+='\\end{table}\n';(OUT/'tables'/f'{name}.tex').write_text(text)

# Diagram: a fixed mask assigns a single function to each hidden unit.
fig,ax=plt.subplots(figsize=(7,2.4));ax.axis('off');ax.set_xlim(-.5,6.5);ax.set_ylim(-.5,4.5)
xs=[0,2,4,6]; ys=[np.linspace(.3,3.7,4),np.linspace(.3,3.7,6),np.linspace(.3,3.7,6),np.linspace(.3,3.7,3)]
for k in range(3):
    for y1 in ys[k]:
        for y2 in ys[k+1]:ax.plot([xs[k],xs[k+1]],[y1,y2],c='#bac5ce',lw=.35,zorder=0)
for k,x in enumerate(xs):
    for j,y in enumerate(ys[k]):
        col=('#007e75' if j%2 else '#a56a24') if k in [1,2] else '#b8c7d5'
        ax.add_patch(Circle((x,y),.16,fc=col,ec='white',lw=.8))
for x,l in zip(xs,['Input: D','Hidden: H','Hidden: H','Output: 10']):ax.text(x,4.17,l,ha='center',fontsize=9)
ax.text(3,-.28,'Brown: tanh     Green: ReLU     Fixed 50:50 mask in both hidden layers',ha='center',fontsize=9)
save(fig,'network')

fig,axs=plt.subplots(1,2,figsize=(7,2.75),sharey=True)
for ax,h in zip(axs,[64,128]):
    for c in ['relu','leaky_relu','tanh','tanh+relu','tanh+leaky_relu']:
        q=sel(D,stage='lrsweep',dataset='mnist',hidden=h,config=c,lr=.03)
        a=np.stack([np.fromstring(v,sep='|') for v in q.epoch_val_acc]); mu=a.mean(0);sd=a.std(0,ddof=1)
        ax.plot(np.arange(1,41),mu,color=COL[c],label=NAME[c],lw=1.65)
        ax.fill_between(np.arange(1,41),np.maximum(0,mu-sd),np.minimum(100,mu+sd),color=COL[c],alpha=.10)
    ax.axhline(15,color='#8a8a8a',ls=':',lw=.8);ax.set_title(f'MNIST, H={h}, learning rate 0.03');ax.set_xlabel('Epoch');ax.set_ylim(0,101);ax.grid(alpha=.12)
axs[0].set_ylabel('Validation accuracy (%)');legend(fig,axs[0],3);fig.tight_layout(rect=[0,.18,1,1]);save(fig,'training_curves')

fig,axs=plt.subplots(1,3,figsize=(7,2.9),sharey=True)
for ax,(ds,h) in zip(axs,[('mnist',64),('mnist',128),('cifar10',128)]):
    for c in ORDER:
        q=sel(L,dataset=ds,hidden=h,config=c).sort_values('lr');ax.plot(q.lr,q.final_val,'o-',ms=2.4,lw=1.1,color=COL[c],label=NAME[c])
    ax.set_xscale('log');ax.set_xlabel('Learning rate');ax.set_title(f'{"MNIST" if ds=="mnist" else "CIFAR-10"}, H={h}');ax.grid(alpha=.12);ax.set_ylim(0,101)
axs[0].set_ylabel('Final validation accuracy (%)');legend(fig,axs[0],4);fig.tight_layout(rect=[0,.19,1,1]);save(fig,'lr_scores')

fig,axs=plt.subplots(3,1,figsize=(7,6.0))
for ax,(ds,h) in zip(axs,[('mnist',64),('mnist',128),('cifar10',128)]):
    q=sel(L,dataset=ds,hidden=h);m=q.pivot(index='config',columns='lr',values='fail15').loc[ORDER]
    ax.imshow(m,vmin=0,vmax=30,cmap='OrRd',aspect='auto')
    for i in range(7):
        for j in range(len(m.columns)):
            n=int(m.iloc[i,j]);ax.text(j,i,str(n),ha='center',va='center',fontsize=7.5,color='white' if n>20 else '#253442')
    ax.set_yticks(range(7),[SHORT[c] for c in ORDER]);ax.set_xticks(range(len(m.columns)),[rate(x) for x in m.columns]);ax.set_title(f'{"MNIST" if ds=="mnist" else "CIFAR-10"}, H={h}: failed runs out of 30',loc='left');ax.tick_params(length=0)
axs[-1].set_xlabel('Learning rate');fig.tight_layout(h_pad=1.3);save(fig,'failure_maps')

fig,axs=plt.subplots(1,2,figsize=(7,2.9),sharey=True)
for ax,h in zip(axs,[64,128]):
    for j,c in enumerate(ORDER):
        q=sel(D,stage='lrsweep',dataset='mnist',hidden=h,config=c,lr=.03).sort_values('seed');v=q.best_val_acc-q.val_acc
        ax.scatter(j+np.linspace(-.16,.16,len(v)),v,s=10,color=COL[c],alpha=.65)
        ax.plot([j-.24,j+.24],[v.mean()]*2,color='black',lw=1.7)
    ax.set_xticks(range(7),[SHORT[c] for c in ORDER]);ax.set_title(f'MNIST, H={h}, learning rate 0.03');ax.grid(axis='y',alpha=.15)
axs[0].set_ylabel('Peak-to-final loss (pp)');fig.tight_layout();save(fig,'drop_seeds')

fig,axs=plt.subplots(2,2,figsize=(7,4.8),sharex=True)
for j,ds in enumerate(['mnist','cifar10']):
    for i,metric in enumerate(['test_acc_best','test_f1_best']):
        ax=axs[i,j]
        for c in ORDER:
            q=sel(S,stage='main',dataset=ds,config=c).sort_values('hidden');ax.plot(q.hidden,q[metric+'_mean'],'o-',color=COL[c],ms=3,lw=1.2,label=NAME[c])
        ax.set_xscale('log',base=2);ax.set_xticks([12,32,64,128,256,512],['12','32','64','128','256','512']);ax.grid(alpha=.15)
        if i==0:ax.set_title('MNIST' if ds=='mnist' else 'CIFAR-10')
        if i==1:ax.set_xlabel('Hidden width')
        if j==0:ax.set_ylabel('Test accuracy (%)' if i==0 else 'Test macro-F1 (%)')
legend(fig,axs[0,0],4);fig.tight_layout(rect=[0,.12,1,1]);save(fig,'width_scores')

fig,axs=plt.subplots(1,2,figsize=(7,2.8))
for ax,ds in zip(axs,['mnist','cifar10']):
    q=sel(P,dataset=ds,config='tanh+relu',baseline='relu',metric='test_acc_best').sort_values('hidden')
    ax.errorbar(range(6),q.delta,yerr=[q.delta-q.ci_low,q.ci_high-q.delta],fmt='o',color=COL['tanh+relu'],capsize=3)
    ax.axhline(0,color='gray',ls='--',lw=.8);ax.set_xticks(range(6),q.hidden);ax.set_xlabel('Hidden width');ax.set_title('MNIST' if ds=='mnist' else 'CIFAR-10');ax.grid(axis='y',alpha=.12)
axs[0].set_ylabel('tanh + ReLU minus ReLU (pp)');fig.tight_layout();save(fig,'paired_width')

fig,axs=plt.subplots(1,2,figsize=(7,2.85))
for ax,c in zip(axs,['tanh+relu','tanh+leaky_relu']):
    q=sel(S,stage='layout',config=c,layout='interleave').sort_values('ratio')
    ax.errorbar(q.ratio,q.test_acc_best_mean,yerr=q.test_acc_best_std,fmt='o-',color=COL[c],capsize=3)
    parent=c.split('+')[1]
    q0=sel(S,stage='lrsweep',dataset='mnist',hidden=128,config='tanh',lr=.01).iloc[0]
    q1=sel(S,stage='lrsweep',dataset='mnist',hidden=128,config=parent,lr=.01).iloc[0]
    ax.errorbar([0,1],[q0.test_acc_best_mean,q1.test_acc_best_mean],yerr=[q0.test_acc_best_std,q1.test_acc_best_std],fmt='s',color='#697b89',capsize=3)
    ax.set_title(NAME[c]);ax.set_xticks([0,.25,.5,.75,1],['0','25','50','75','100']);ax.set_xlabel('Second activation share (%)');ax.grid(alpha=.12);ax.set_ylim(97.65,98.22)
axs[0].set_ylabel('Checkpoint test accuracy (%)');fig.tight_layout();save(fig,'ratio_scores')

fig,axs=plt.subplots(1,2,figsize=(7,2.9))
for ax,ds in zip(axs,['mnist','cifar10']):
    for c in ['relu','tanh','tanh+relu','tanh+leaky_relu']:
        q=sel(S,stage='main',dataset=ds,config=c).sort_values('hidden');ax.plot(q.parameters,q.test_acc_best_mean,'o-',color=COL[c],ms=3,lw=1.2,label=NAME[c])
    ax.set_xscale('log');ax.set_xlabel('Trainable parameters');ax.set_title('MNIST' if ds=='mnist' else 'CIFAR-10');ax.grid(alpha=.15)
axs[0].set_ylabel('Checkpoint test accuracy (%)');legend(fig,axs[0],4);fig.tight_layout(rect=[0,.12,1,1]);save(fig,'parameter_cost')

fig,axs=plt.subplots(1,2,figsize=(7,2.9))
for ax,ds in zip(axs,['mnist','cifar10']):
    for c in ['relu','tanh','tanh+relu','tanh+leaky_relu']:
        q=sel(S,stage='main',dataset=ds,config=c).sort_values('hidden');ax.plot(q.hidden,q.gap_best_mean,'o-',color=COL[c],ms=3,label=NAME[c])
    ax.set_xscale('log',base=2);ax.set_xticks([12,32,64,128,256,512],['12','32','64','128','256','512']);ax.set_title('MNIST' if ds=='mnist' else 'CIFAR-10');ax.set_xlabel('Hidden width');ax.grid(alpha=.15)
axs[0].set_ylabel('Train minus test accuracy (pp)');legend(fig,axs[0],4);fig.tight_layout(rect=[0,.12,1,1]);save(fig,'train_test_gap')

# Class-level diagnostics for all seven methods, at H=128.
fig,axs=plt.subplots(1,2,figsize=(7,3.5))
for ax,ds in zip(axs,['mnist','cifar10']):
    values=[]
    for c in ORDER:
        q=sel(D,stage='main',dataset=ds,hidden=128,config=c)
        cms=np.stack([np.fromstring(x,sep='|').reshape(10,10) for x in q.test_confusion_best])
        values.append((100*np.diagonal(cms,axis1=1,axis2=2)/cms.sum(2)).mean(0))
    a=np.array(values).T
    im=ax.imshow(a,aspect='auto',cmap='YlGnBu',vmin=90 if ds=='mnist' else 20,vmax=100 if ds=='mnist' else 80)
    for i in range(10):
        for j in range(7):ax.text(j,i,f'{a[i,j]:.1f}',ha='center',va='center',fontsize=7,color='white' if a[i,j]>(96 if ds=='mnist' else 58) else 'black')
    ax.set_xticks(range(7),[SHORT[c] for c in ORDER]);ax.set_yticks(range(10),[str(i) for i in range(10)] if ds=='mnist' else ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']);ax.set_title(('MNIST' if ds=='mnist' else 'CIFAR-10')+': class recall (%)');ax.tick_params(length=0)
fig.tight_layout();save(fig,'class_recall')

# Tables used in the main text.
table('stress','MNIST at learning rate 0.03. Each row has 30 seeds. The loss is peak minus final validation accuracy.','tab:stress',['H','Model','Final acc. (SD)','Peak acc.','Loss (pp)','Below 15\%'],[[h,NAME[c],f'{q.final_val:.3f} ({q.sd:.3f})',fmt(q.best_val),fmt(q['drop']),f'{int(q.fail15)}/30'] for h in [64,128] for c in ORDER for _,q in sel(L,dataset='mnist',hidden=h,config=c,lr=.03).iterrows()],align='rlrrrr')
rows=[]; onset=[]
for h in [64,128]:
    for c in ['relu','tanh+relu']:
        q=sel(D,stage='lrsweep',dataset='mnist',hidden=h,config=c,lr=.03)
        hits=[];shares=[]
        for _,r in q.iterrows():
            a=np.fromstring(r.epoch_val_acc,sep='|');z=np.flatnonzero(a<15)
            if len(z):hits.append(int(z[0]+1))
            cm=np.fromstring(r.test_confusion,sep='|').reshape(10,10);shares.append(100*cm.sum(0).max()/cm.sum())
        val='--' if not hits else f'{min(hits)} / {np.median(hits):g} / {max(hits)}'
        rows.append([h,NAME[c],f'{len(hits)}/30',val,f'{np.mean(shares):.1f}'])
        onset.append(dict(hidden=h,config=c,n_hits=len(hits),epochs=hits,largest_class_share=np.mean(shares)))
table('onset','When failure first appears at MNIST learning rate 0.03. Class share uses final test predictions.','tab:onset',['H','Model','Ever below 15\%','First epoch: min/median/max','Largest class (\%)'],rows,align='rllrr',size='footnotesize')
(HERE/'failure_onset.json').write_text(json.dumps(onset,indent=2))
rows=[]
for ds,h in [('mnist',64),('mnist',128),('cifar10',128)]:
    for c in ORDER:
        cells=[]
        for criterion in ['Within 1 pp of own best','No final validation <15%','No peak-to-final drop >10 pp']:
            q=sel(W,dataset=ds,hidden=h,config=c,criterion=criterion).iloc[0]
            cells.append('--' if q.grid_points==0 else f'{rate(q.low)}--{rate(q.high)} ({int(q.grid_points)})')
        rows.append(['MNIST' if ds=='mnist' else 'CIFAR-10',h,SHORT[c]]+cells)
table('windows','Passing learning-rate ranges on the tested grid. Parentheses give the number of sampled rates.','tab:windows',['Data','H','Model','Near best','No final $<15\\%$','No drop $>10$ pp'],rows,align='lrlrrr',size='footnotesize',note='Near best: mean final score within 1 pp of the model\'s own best grid mean. Each no-event rule requires all 30 seeds to pass. Ranges are the widest consecutive passing sets, chosen by log span; untested rates are unknown. Small drops alone do not imply high accuracy. Model codes follow Figure~\\ref{fig:drops}.')
table('cifar_stress','CIFAR-10, H=128. Leaky-ReLU is an important control. Each row has 30 seeds.','tab:cifarstress',['LR','Model','Final acc. (\%)','Peak acc. (\%)','Loss (pp)','Below 15\%'],[[rate(lr),NAME[c],fmt(q.final_val),fmt(q.best_val),fmt(q['drop']),f'{int(q.fail15)}/30'] for lr in [.0003,.003,.01,.03] for c in ['relu','tanh+relu','leaky_relu'] for _,q in sel(L,dataset='cifar10',hidden=128,config=c,lr=lr).iterrows()],align='llrrrr')
for ds in ['mnist','cifar10']:
    for metric,mlabel in [('test_acc_best','accuracy'),('test_f1_best','macro-F1')]:
        rows=[]
        for c in ORDER:
            row=[NAME[c]]
            for h in [12,32,64,128,256,512]:
                q=sel(S,stage='main',dataset=ds,hidden=h,config=c).iloc[0]
                row.append(f'{q[metric+"_mean"]:.2f} $\\pm$ {q[metric+"_std"]:.2f}')
            rows.append(row)
        table(ds+'_'+metric,f'{"MNIST" if ds=="mnist" else "CIFAR-10"} checkpoint test {mlabel} (\%). Mean $\\pm$ seed SD, $n=30$.',f'tab:{ds}:{metric}',['Model','12','32','64','128','256','512'],rows,size='footnotesize',note='Main hybrids use a 50:50 split. These cells keep the original learning-rate choices.')
rows=[]
for ds in ['mnist','cifar10']:
    for _,q in sel(P,dataset=ds,config='tanh+relu',baseline='relu',metric='test_acc_best').sort_values('hidden').iterrows():
        rows.append(['MNIST' if ds=='mnist' else 'CIFAR-10',q.hidden,f'{q.delta:+.3f}',ci(q.ci_low,q.ci_high),f'{q.wins}/30',f'{q.p_holm:.3g}'])
table('paired','tanh + ReLU minus ReLU at the original learning rates. Intervals are pointwise; adjusted p-values cover 120 main accuracy tests.','tab:paired',['Data','H','Difference (pp)','95\% paired interval','Wins','Holm p'],rows,align='lrrrrr')
rows=[]
for lr in [.003,.01]:
    a=sel(D,stage='lrsweep',dataset='mnist',hidden=128,config='tanh+relu',lr=.01);b=sel(D,stage='lrsweep',dataset='mnist',hidden=128,config='relu',lr=lr)
    delta,lo,hi=pair(a,b);rows.append([rate(lr),fmt(b.test_acc_best.mean()),fmt(a.test_acc_best.mean()),f'{delta:+.3f}',ci(lo,hi)])
table('aligned','MNIST, H=128: aligning the ReLU learning rate with checkpoint selection. The hybrid uses LR=0.01 in both rows.','tab:aligned',['ReLU LR','ReLU acc. (\%)','Hybrid acc. (\%)','Difference (pp)','95\% paired interval'],rows,align='rrrrr')
rows=[]
for c in ['tanh+relu','tanh+leaky_relu']:
    for _,q in sel(S,stage='layout',config=c,layout='interleave').sort_values('ratio').iterrows():
        rows.append([NAME[c],f'{q.ratio:.2f}',f'{q.test_acc_best_mean:.3f} $\\pm$ {q.test_acc_best_std:.3f}',f'{q.test_f1_best_mean:.3f}',f'{q.best_val_epoch_mean:.1f}'])
table('ratios','Allocation ratio at MNIST H=128 and LR=0.01. The second function is ReLU or leaky-ReLU. Each row has 30 seeds.','tab:ratios',['Model','$\\rho$','Accuracy (\%), mean $\\pm$ SD','Macro-F1 (\%)','Selected epoch'],rows,align='lrrrr',size='footnotesize')
table('ratio_pairs','Paired ratio comparisons. Holm correction covers these six tests only.','tab:ratiopairs',['Model','Ratio change','Difference (pp)','95\% paired interval','Holm p'],[[NAME[q.config],f'{q.low_ratio:.2f} to {q.high_ratio:.2f}',f'{q.delta:+.3f}',ci(q.ci_low,q.ci_high),f'{q.p_holm_6:.3g}'] for _,q in RP.iterrows()],align='llrrr',size='footnotesize')
rows=[]
for c in ['tanh+relu','tanh+leaky_relu']:
    base=sel(D,stage='layout',config=c,layout='interleave',ratio=.5)
    for layout in ['interleave','block','random']:
        q=sel(D,stage='layout',config=c,layout=layout,ratio=.5);delta,lo,hi=pair(q,base)
        rows.append([NAME[c],layout,f'{q.test_acc_best.mean():.3f} $\\pm$ {q.test_acc_best.std():.3f}',f'{delta:+.3f}',ci(lo,hi)])
table('layouts','Mask layout at MNIST H=128, LR=0.01 and $\\rho=0.5$. Differences use the interleave mask as the reference.','tab:layouts',['Model','Layout','Accuracy (\%), mean $\\pm$ SD','Difference (pp)','95\% interval'],rows,align='llrrr',size='footnotesize')
rows=[]
for h in [64,128]:
    base=sel(D,stage='mse',hidden=h,config='relu')
    for c in ['relu','tanh','tanh+relu']:
        q=sel(D,stage='mse',hidden=h,config=c);delta,lo,hi=pair(q,base)
        rows.append([h,NAME[c],rate(q.lr.iloc[0]),f'{q.test_acc_best.mean():.3f} $\\pm$ {q.test_acc_best.std():.3f}',fmt(q.test_f1_best.mean()),f'{delta:+.3f}',ci(lo,hi)])
table('mse','MNIST with the implemented MSE loss and sigmoid output. Mean and SD use 30 seeds. Differences are against ReLU at the same width.','tab:mse',['H','Model','LR','Accuracy (\%)','F1 (\%)','$\\Delta$ (pp)','95\% interval'],rows,align='rllrrrr',size='footnotesize')
rows=[]
for ds in ['mnist','cifar10']:
    for h,c in [(128,'tanh+relu'),(256,'relu'),(512,'relu'),(512,'tanh+relu')]:
        q=sel(S,stage='main',dataset=ds,hidden=h,config=c).iloc[0]
        rows.append(['MNIST' if ds=='mnist' else 'CIFAR-10',h,NAME[c],fmt(q.test_acc_best_mean),f'{int(q.parameters):,}',f'{q.seconds_mean:.1f}'])
table('cost','Accuracy and size at selected operating points. Seconds are full-run elapsed time under the logged shared workload, not inference time.','tab:cost',['Data','H','Model','Accuracy (\%)','Parameters','Seconds'],rows,align='lrlrrr',size='small')
rows=[]
for _,q in C[C.changed].sort_values(['dataset','stage','hidden','config']).iterrows():
    rows.append(['MNIST' if q.dataset=='mnist' else 'CIFAR-10',q.hidden,NAME[q.config],'MSE' if q.stage=='calib_mse' else 'CE',rate(q.lr_final),rate(q.lr_best),'Yes' if q.available_evaluation_n==30 else 'No'])
table('calibration_changes','Learning-rate choices that change when tuning uses each run\'s best validation score. Existing evaluation runs are available for four of the 17 changed CE choices.','tab:calchanges',['Data','H','Model','Loss','Final rule','Peak rule','Runs exist'],rows,align='lrllrrl',size='footnotesize')
for ds in ['mnist','cifar10']:
    rows=[]
    for c in ORDER:rows.append([NAME[c]]+[rate(sel(S,stage='main',dataset=ds,hidden=h,config=c).lr.iloc[0]) for h in [12,32,64,128,256,512]])
    table('lrs_'+ds,f'Original selected learning rates: {"MNIST" if ds=="mnist" else "CIFAR-10"}.',f'tab:lrs:{ds}',['Model','12','32','64','128','256','512'],rows,size='footnotesize')

# Compact complete main-grid appendix: 84 rows and both test endpoints.
lines=[r'\small',r'\begin{longtable}{@{}lrlrrrrrr@{}}',r'\caption{Complete main-grid results. Selected and final refer to the checkpoint and epoch 40. All cells have 30 seeds.}\label{tab:allmain}\\',r'\toprule Data & H & Model & Selected acc. & Final acc. & Selected F1 & Train--test & Epoch & Seconds \\',r'\midrule\endfirsthead',r'\toprule Data & H & Model & Selected acc. & Final acc. & Selected F1 & Train--test & Epoch & Seconds \\',r'\midrule\endhead',r'\bottomrule\endfoot']
for ds in ['mnist','cifar10']:
    for h in [12,32,64,128,256,512]:
        for c in ORDER:
            q=sel(S,stage='main',dataset=ds,hidden=h,config=c).iloc[0]
            row=['M' if ds=='mnist' else 'C',str(h),SHORT[c],fmt(q.test_acc_best_mean),fmt(q.test_acc_mean),fmt(q.test_f1_best_mean),fmt(q.gap_best_mean),f'{q.best_val_epoch_mean:.1f}',f'{q.seconds_mean:.1f}']
            lines.append(' & '.join(row)+r' \\')
        lines.append(r'\addlinespace')
lines+=[r'\end{longtable}',r'\normalsize']
(OUT/'tables'/'all_main.tex').write_text('\n'.join(lines))

# Complete final-validation LR matrices, one table per dataset/width.
for ds,h in [('mnist',64),('mnist',128),('cifar10',128)]:
    q=sel(L,dataset=ds,hidden=h);rates=sorted(q.lr.unique())
    rows=[[SHORT[c]]+[f'{sel(q,config=c,lr=lr).final_val.iloc[0]:.2f}' for lr in rates] for c in ORDER]
    table(f'lr_matrix_{ds}_{h}',f'Mean final validation accuracy (\%) across the full evaluation LR grid: {"MNIST" if ds=="mnist" else "CIFAR-10"}, H={h}.',f'tab:lrmat:{ds}:{h}',['Model']+[rate(x) for x in rates],rows,size='scriptsize')

# Separate dataset panels keep calibration values readable at print size.
for loss,stage,datasets,name in [('CE','calib',['mnist'],'calibration_ce_mnist'),('CE','calib',['cifar10'],'calibration_ce_cifar10'),('MSE','calib_mse',['mnist'],'calibration_mse')]:
    cells=[(ds,h) for ds in datasets for h in ([12,32,64,128,256,512] if loss=='CE' else [64,128])]
    fig,axs=plt.subplots(len(cells)//2,2,figsize=(7,6.0 if loss=='CE' else 2.3),squeeze=False)
    for ax,(ds,h) in zip(axs.flat,cells):
        q=sel(S,stage=stage,dataset=ds,hidden=h);configs=ORDER if loss=='CE' else ['relu','tanh','tanh+relu']
        m=q.pivot(index='config',columns='lr',values='val_acc_mean').loc[configs]
        ax.imshow(m,cmap='viridis',vmin=0,vmax=100,aspect='auto')
        for i in range(len(configs)):
            for j in range(len(m.columns)):ax.text(j,i,f'{m.iloc[i,j]:.1f}',fontsize=7,ha='center',va='center',color='white' if m.iloc[i,j]<65 else '#132a30')
        ax.set_xticks(range(len(m.columns)),[rate(x) for x in m.columns],rotation=35,ha='right',fontsize=7)
        ax.set_yticks(range(len(configs)),[SHORT[c] for c in configs],fontsize=7);ax.tick_params(length=0);ax.set_title(f'{"MNIST" if ds=="mnist" else "CIFAR-10"}, H={h}',fontsize=9,loc='left')
    fig.tight_layout(h_pad=1,w_pad=1);save(fig,name)
print('Generated manuscript figures and tables.')
