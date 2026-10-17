from pathlib import Path
import csv, json, re
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research_audit'
KEY = ['dataset','seed','act1','act2','ratio','layout','hidden','lr','epochs','loss']
frames=[]; inventory=[]
for path in sorted(ROOT.glob('*.csv')):
    raw=pd.read_csv(path)
    headers=raw.seed.astype(str).eq('seed')
    df=raw[~headers].copy()
    for c in ['seed','ratio','hidden','lr','epochs','seconds','train_acc','val_acc','val_f1','test_acc','test_f1','best_val_acc','best_val_epoch','train_acc_clean','test_acc_best','test_f1_best']:
        df[c]=pd.to_numeric(df[c], errors='raise')
    dup=df.duplicated(KEY,keep=False)
    conflict=0
    for _,g in df[dup].groupby(KEY):
        if len(g[['test_acc','test_acc_best','test_confusion_best','epoch_val_acc']].drop_duplicates())>1: conflict+=1
    inventory.append(dict(file=path.name,rows=len(df),repeated_headers=int(headers.sum()),duplicates=int(df.duplicated(KEY).sum()),conflicting_keys=conflict,status=df.status.value_counts().to_dict(),missing_best=int(df.test_acc_best.isna().sum())))
    df=df.drop_duplicates(KEY,keep='first')
    df['file']=path.name
    df['stage']=re.sub(r'^(c10_)?(.*?)_\d+\.csv$',r'\2',path.name)
    df['config']=np.where(df.ratio.eq(0),df.act1,df.act1+'+'+df.act2)
    frames.append(df)
d=pd.concat(frames,ignore_index=True)
checks={'inventory':inventory,'total_unique_within_files':len(d),'status':d.status.value_counts().to_dict()}
class_rows=[]; supports={}; errors=[]
for i,r in d.iterrows():
    cm=np.fromstring(r.test_confusion_best,sep='|').reshape(10,10)
    acc=100*np.trace(cm)/cm.sum()
    den=cm.sum(0)+cm.sum(1)
    f1=np.divide(200*cm.diagonal(),den,out=np.zeros(10),where=den>0)
    if cm.sum()!=10000 or abs(acc-r.test_acc_best)>0.0002 or abs(f1.mean()-r.test_f1_best)>0.0002: errors.append(int(i))
    supports.setdefault(r.dataset,set()).add(tuple(cm.sum(1).astype(int)))
    va=np.fromstring(r.epoch_val_acc,sep='|'); tr=np.fromstring(r.epoch_train_acc_clean,sep='|')
    if len(va)!=40 or abs(va.max()-r.best_val_acc)>0.0002 or va.argmax()+1!=r.best_val_epoch: errors.append(int(i))
    d.loc[i,'train_best']=tr[int(r.best_val_epoch)-1]
    d.loc[i,'loss_final']=float(r.epoch_loss.split('|')[-1])
    d.loc[i,'checkpoint_gain']=r.test_acc_best-r.test_acc
    d.loc[i,'gap_best']=tr[int(r.best_val_epoch)-1]-r.test_acc_best
    if r.stage in ['main','mse','layout']:
        for c in range(10):
            class_rows.append(dict(dataset=r.dataset,hidden=r.hidden,config=r.config,stage=r.stage,layout=r.layout,ratio=r.ratio,seed=r.seed,cls=c,precision=100*cm[c,c]/cm[:,c].sum() if cm[:,c].sum() else 0,recall=100*cm[c,c]/cm[c,:].sum(),f1=f1[c]))
checks['metric_or_curve_errors']=errors
checks['class_supports']={k:[[int(a) for a in x] for x in v] for k,v in supports.items()}
checks['cross_file_duplicate_rows']=int(d.duplicated(KEY).sum())
OUT.joinpath('integrity.json').write_text(json.dumps(checks,indent=2))
group=['dataset','stage','hidden','config','layout','ratio','lr']
metrics=['test_acc_best','test_f1_best','test_acc','test_f1','val_acc','val_f1','best_val_acc','train_acc','train_acc_clean','train_best','gap_best','checkpoint_gain','best_val_epoch','seconds','loss_final']
s=d.groupby(group)[metrics].agg(['mean','std','median','min','max'])
s.columns=['_'.join(x) for x in s.columns]; s=s.reset_index()
s['n']=d.groupby(group).size().values
s['parameters']=s.hidden**2+s.hidden*(np.where(s.dataset.eq('mnist'),784,3072)+12)+10
s.to_csv(OUT/'all_metrics.csv',index=False)
main=d[d.stage.eq('main')]
comparisons=[]; rng=np.random.default_rng(20260913)
for (dataset,width),g in main.groupby(['dataset','hidden']):
    for cfg,h in g.groupby('config'):
        parents=['relu']
        if '+' in cfg: parents+=cfg.split('+')
        for parent in dict.fromkeys(parents):
            if cfg==parent: continue
            pair=h.merge(g[g.config.eq(parent)],on='seed',suffixes=('_h','_b'),validate='one_to_one')
            for metric in ['test_acc_best','test_f1_best','test_acc','test_f1']:
                diff=(pair[metric+'_h']-pair[metric+'_b']).to_numpy()
                lo,hi=np.quantile(diff[rng.integers(0,len(diff),(10000,len(diff)))].mean(1),[.025,.975])
                comparisons.append(dict(dataset=dataset,hidden=width,config=cfg,baseline=parent,metric=metric,n=len(diff),delta=diff.mean(),ci_low=lo,ci_high=hi,p=stats.ttest_1samp(diff,0).pvalue,wins=int((diff>0).sum()),ties=int((diff==0).sum()),relative_metric_pct=100*diff.mean()/pair[metric+'_b'].mean(),relative_error_reduction_pct=100*diff.mean()/(100-pair[metric+'_b'].mean())))
p=pd.DataFrame(comparisons)
# Holm across all primary checkpoint-accuracy contrasts (120 comparisons).
idx=p.index[p.metric.eq('test_acc_best')]; order=p.loc[idx].sort_values('p').index
p.loc[order,'p_holm']=np.minimum(1,np.maximum.accumulate(p.loc[order,'p'].to_numpy()*np.arange(len(order),0,-1)))
p.to_csv(OUT/'paired_comparisons.csv',index=False)
pd.DataFrame(class_rows).groupby(['dataset','stage','hidden','config','layout','ratio','cls'])[['precision','recall','f1']].agg(['mean','std']).to_csv(OUT/'class_metrics.csv')
cal=[]
for (dataset,stage,width,cfg,layout,ratio),g in d[d.stage.isin(['calib','calib_mse'])].groupby(['dataset','stage','hidden','config','layout','ratio']):
    a=g.groupby('lr')[['val_acc','best_val_acc']].mean()
    cal.append(dict(dataset=dataset,stage=stage,hidden=width,config=cfg,lr_final=a.val_acc.idxmax(),lr_best=a.best_val_acc.idxmax(),selection_changed=a.val_acc.idxmax()!=a.best_val_acc.idxmax()))
pd.DataFrame(cal).to_csv(OUT/'calibration_audit.csv',index=False)
lr=d[d.stage.eq('lrsweep')].copy()
lr['below50']=lr.val_acc.lt(50); lr['below15']=lr.val_acc.lt(15)
lr.groupby(group).agg(n=('seed','size'),below50=('below50','sum'),below15=('below15','sum'),final_val=('val_acc','mean'),best_val=('best_val_acc','mean'),final_test=('test_acc','mean'),best_test=('test_acc_best','mean')).to_csv(OUT/'lr_diagnostics.csv')
print(json.dumps(checks,indent=2))
print('\nMAIN CHECKPOINT ACCURACY\n',main.pivot_table(index='config',columns=['dataset','hidden'],values='test_acc_best').round(3).to_string())
print('\nMAIN SUMMARY\n',s[s.stage.eq('main')][['dataset','hidden','config','lr','test_acc_best_mean','test_acc_best_std','test_acc_mean','test_f1_best_mean','gap_best_mean','best_val_epoch_mean','seconds_mean']].round(4).to_string(index=False))
print('\nCALIBRATION CHANGES\n',pd.DataFrame(cal).query('selection_changed').to_string(index=False))
