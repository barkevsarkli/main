from pathlib import Path
import json, hashlib, re, itertools, os
import numpy as np
import pandas as pd
from scipy import stats
ROOT=Path(os.environ.get('EXPERIMENT_RESULTS_DIR',str(Path(__file__).resolve().parents[2])))
OUT=Path(__file__).resolve().parent
KEY=['dataset','seed','act1','act2','ratio','layout','hidden','lr','epochs','loss']
NUM=['seed','ratio','hidden','lr','epochs','seconds','train_acc','val_acc','val_f1','test_acc','test_f1','best_val_acc','best_val_epoch','train_acc_clean','test_acc_best','test_f1_best']
frames=[]; inventory=[]; errors=[]; classrows=[]
for path in sorted(ROOT.glob('*.csv')):
    raw=pd.read_csv(path); headers=raw.seed.astype(str).eq('seed'); x=raw.loc[~headers].copy()
    for c in NUM:x[c]=pd.to_numeric(x[c],errors='raise')
    assert not x.duplicated(KEY).any(),path.name
    x['file']=path.name;x['stage']=re.sub(r'^(c10_)?(.*?)_\d+\.csv$',r'\2',path.name)
    x['config']=np.where(x.ratio.eq(0),x.act1,x.act1+'+'+x.act2)
    inventory.append(dict(file=path.name,rows=len(x),repeated_headers=int(headers.sum()),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    frames.append(x)
d=pd.concat(frames,ignore_index=True)
for i,r in d.iterrows():
    va=np.fromstring(r.epoch_val_acc,sep='|');tr=np.fromstring(r.epoch_train_acc_clean,sep='|')
    assert len(va)==40 and len(tr)==40
    assert abs(va.max()-r.best_val_acc)<.0002 and va.argmax()+1==r.best_val_epoch
    for field,a,f in [('test_confusion_best','test_acc_best','test_f1_best'),('test_confusion','test_acc','test_f1')]:
        cm=np.fromstring(r[field],sep='|').reshape(10,10);den=cm.sum(0)+cm.sum(1)
        f1=np.divide(200*cm.diagonal(),den,out=np.zeros(10),where=den>0)
        assert cm.sum()==10000 and abs(100*np.trace(cm)/cm.sum()-r[a])<.0002 and abs(f1.mean()-r[f])<.0002
        if field=='test_confusion_best' and r.stage in ['main','layout','mse']:
            for c in range(10):
                classrows.append(dict(dataset=r.dataset,stage=r.stage,hidden=r.hidden,config=r.config,layout=r.layout,ratio=r.ratio,seed=r.seed,cls=c,support=cm[c].sum(),recall=100*cm[c,c]/cm[c].sum(),f1=f1[c]))
    d.loc[i,'train_best']=tr[int(r.best_val_epoch)-1]
    d.loc[i,'gap_best']=tr[int(r.best_val_epoch)-1]-r.test_acc_best
    d.loc[i,'val_drop']=r.best_val_acc-r.val_acc
    d.loc[i,'test_checkpoint_gain']=r.test_acc_best-r.test_acc
    d.loc[i,'below15']=int(r.val_acc<15)
    d.loc[i,'below50']=int(r.val_acc<50)
    d.loc[i,'drop10']=int(r.best_val_acc-r.val_acc>10)
    d.loc[i,'epoch40']=int(r.best_val_epoch==40)
main=d[d.stage.eq('main')];layout=d[d.stage.eq('layout')];lr=d[d.stage.eq('lrsweep')];cal=d[d.stage.isin(['calib','calib_mse'])];mse=d[d.stage.eq('mse')]
for _,g in main.groupby(['dataset','hidden','config']):assert set(g.seed)==set(range(1,31))
for _,g in cal.groupby(['dataset','hidden','config','loss','lr']):assert set(g.seed)==set(range(101,107))
G=['dataset','stage','hidden','config','layout','ratio','lr']
metrics=['test_acc_best','test_f1_best','test_acc','test_f1','val_acc','best_val_acc','val_drop','train_best','gap_best','test_checkpoint_gain','best_val_epoch','seconds']
s=d.groupby(G)[metrics].agg(['mean','std','median','min','max']);s.columns=['_'.join(c) for c in s.columns];s=s.reset_index()
s['n']=d.groupby(G).size().values;s['parameters']=s.hidden**2+s.hidden*(np.where(s.dataset.eq('mnist'),784,3072)+12)+10
s.to_csv(OUT/'metric_summary.csv',index=False)
def subset(frame,**kw):
    for k,v in kw.items():frame=frame[frame[k].eq(v)]
    return frame

def compare(h,b,label='',metric='test_acc_best'):
    p=h[['seed',metric]].merge(b[['seed',metric]],on='seed',suffixes=('_h','_b'),validate='one_to_one').sort_values('seed')
    assert len(p)==len(h)==len(b)
    delta=(p[metric+'_h']-p[metric+'_b']).to_numpy();rng=np.random.default_rng(20260915)
    lo,hi=np.quantile(delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1),[.025,.975])
    return dict(comparison=label,n=len(p),delta=delta.mean(),ci_low=lo,ci_high=hi,p=float(stats.ttest_1samp(delta,0).pvalue) if np.any(delta) else 1.,wins=int((delta>1e-9).sum()),ties=int((abs(delta)<1e-9).sum()),error_reduction=100*delta.mean()/(100-p[metric+'_b'].mean()))
pairs=[]
for (ds,h),g in main.groupby(['dataset','hidden']):
    for cfg in g.config.unique():
        for parent in dict.fromkeys(['relu']+(cfg.split('+') if '+' in cfg else [])):
            if cfg==parent:continue
            for metric in ['test_acc_best','test_f1_best']:
                pairs.append(dict(dataset=ds,hidden=h,config=cfg,baseline=parent,metric=metric,**compare(subset(g,config=cfg),subset(g,config=parent),cfg+' vs '+parent,metric)))
pairs=pd.DataFrame(pairs); ix=pairs[pairs.metric.eq('test_acc_best')].sort_values('p').index
assert len(ix)==120
pairs.loc[ix,'p_holm']=np.minimum(1,np.maximum.accumulate(pairs.loc[ix,'p'].to_numpy()*np.arange(len(ix),0,-1)))
pairs.to_csv(OUT/'paired_main.csv',index=False)
old_audit_count=None
old_path=ROOT/'research_audit/paired_comparisons.csv'
if old_path.exists():
    old=pd.read_csv(old_path)
    merged=pairs.merge(old,on=['dataset','hidden','config','baseline','metric'],suffixes=('_new','_old'))
    assert len(merged)==len(pairs) and np.allclose(merged.delta_new,merged.delta_old,atol=1e-9)
    old_audit_count=len(merged)
# Original calibration criterion and retrospective checkpoint-aligned criterion.
choices=[]
for (ds,stage,h,cfg),g in cal.groupby(['dataset','stage','hidden','config']):
    m=g.groupby('lr')[['val_acc','best_val_acc']].mean();a=m.val_acc.idxmax();b=m.best_val_acc.idxmax()
    ev=subset(d,dataset=ds,hidden=h,config=cfg,lr=b);ev=ev[ev.stage.isin(['main','lrsweep'] if stage=='calib' else ['mse'])].drop_duplicates(KEY)
    choices.append(dict(dataset=ds,stage=stage,hidden=h,config=cfg,lr_final=a,lr_best=b,changed=a!=b,available_evaluation_n=len(ev)))
choices=pd.DataFrame(choices);choices.to_csv(OUT/'calibration_choices.csv',index=False)
L=lr.groupby(G).agg(n=('seed','size'),final_val=('val_acc','mean'),sd=('val_acc','std'),best_val=('best_val_acc','mean'),drop=('val_drop','mean'),fail15=('below15','sum'),fail50=('below50','sum'),drop10=('drop10','sum'),best_test=('test_acc_best','mean')).reset_index();L.to_csv(OUT/'lr_stability.csv',index=False)
def window(rates,ok):
    segments=[];start=None
    for i,b in enumerate(list(ok)+[False]):
        if b and start is None:start=i
        if not b and start is not None:
            segments.append((rates[start],rates[i-1],i-start));start=None
    return max(segments,key=lambda x:(np.log10(x[1]/x[0]),x[2])) if segments else (np.nan,np.nan,0)
windows=[]
for (ds,h,cfg),g in L.groupby(['dataset','hidden','config']):
    g=g.sort_values('lr');rates=g.lr.to_numpy()
    for rule,ok in [('Within 1 pp of own best',g.final_val>=g.final_val.max()-1),('No final validation <15%',g.fail15.eq(0)),('No peak-to-final drop >10 pp',g.drop10.eq(0)),('No final validation <50%',g.fail50.eq(0)),('Mean final validation >=95%',g.final_val.ge(95))]:
        a,b,n=window(rates,ok);windows.append(dict(dataset=ds,hidden=h,config=cfg,criterion=rule,low=a,high=b,grid_points=n,decades=np.log10(b/a) if n else np.nan))
windows=pd.DataFrame(windows);windows.to_csv(OUT/'lr_windows.csv',index=False)
ratio_pairs=[]
for cfg,g in layout[layout.layout.eq('interleave')].groupby('config'):
    for a,b in [(0.5,0.25),(0.75,0.25),(0.75,0.5)]:
        ratio_pairs.append(dict(config=cfg,high_ratio=a,low_ratio=b,**compare(subset(g,ratio=a),subset(g,ratio=b),f'{a:g} vs {b:g}')))
ratio_pairs=pd.DataFrame(ratio_pairs); ix=ratio_pairs.sort_values('p').index
ratio_pairs.loc[ix,'p_holm_6']=np.minimum(1,np.maximum.accumulate(ratio_pairs.loc[ix,'p'].to_numpy()*np.arange(len(ix),0,-1)))
ratio_pairs.to_csv(OUT/'paired_ratios.csv',index=False)
classes=pd.DataFrame(classrows);classes.groupby(['dataset','stage','hidden','config','layout','ratio','cls'])[['support','recall','f1']].agg(['mean','std']).to_csv(OUT/'class_summary.csv')
checks=dict(raw_files=len(inventory),data_rows=len(d),repeated_headers=sum(x['repeated_headers'] for x in inventory),cross_stage_duplicate_rows=int(d.duplicated(KEY).sum()),main_rows=len(main),main_comparisons=120,old_audit_deltas_reproduced=old_audit_count,all_status_ok=bool(d.status.eq('ok').all()),all_final_and_best_confusion_metrics_verified=True,all_validation_histories_verified=True,bootstrap_seed=20260915,bootstrap_resamples=10000,main_lr_changes=int(choices.query("stage=='calib'").changed.sum()),inventory=inventory)
(OUT/'verification.json').write_text(json.dumps(checks,indent=2))
if __name__=='__main__':
    print(json.dumps({k:v for k,v in checks.items() if k!='inventory'},indent=2))
    print('\nRATIO PAIRS\n',ratio_pairs.round(6).to_string(index=False))
    print('\nWIDTH128 HIGH LR\n',L.query("hidden==128 and lr in [.01,.03,.1] and config in ['relu','tanh+relu','leaky_relu','tanh+leaky_relu']").round(4).to_string(index=False))
    print('\nWINDOWS\n',windows.query("config in ['relu','tanh+relu']").round(4).to_string(index=False))
    print('\nCHANGED CALIBRATION\n',choices[choices.changed].to_string(index=False))
