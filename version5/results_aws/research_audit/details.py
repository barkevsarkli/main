from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
R=Path(__file__).resolve().parents[1]; O=R/'research_audit'
def read(f):
    x=pd.read_csv(R/f); x=x[x.seed.astype(str)!='seed'].copy()
    for c in ['seed','hidden','ratio','lr','test_acc','test_acc_best','test_f1_best','val_acc','best_val_acc','seconds','best_val_epoch']:
        x[c]=pd.to_numeric(x[c])
    x['config']=np.where(x.ratio.eq(0),x.act1,x.act1+'+'+x.act2)
    return x
rng=np.random.default_rng(20260913)
def paired(a,b):
    z=a.merge(b,on='seed',suffixes=('_a','_b'),validate='one_to_one')
    d=(z.test_acc_best_a-z.test_acc_best_b).to_numpy()
    return dict(n=len(z),delta=d.mean(),ci=np.quantile(d[rng.integers(0,len(d),(10000,len(d)))].mean(1),[.025,.975]).tolist(),p=stats.ttest_1samp(d,0).pvalue,wins=int((d>0).sum()))
out={}
x=read('lrsweep_128.csv')
out['mnist128_matched_lr']=paired(x[(x.config=='tanh+relu')&(x.lr==.01)],x[(x.config=='relu')&(x.lr==.01)])
out['mnist128_calibration']=read('calib_128.csv').query('config in ["relu","tanh+relu"] and lr in [0.003,0.01]').groupby(['config','lr'])[['val_acc','best_val_acc']].mean().reset_index().to_dict('records')
out['mse']={}
for w in [64,128]:
    t=read(f'mse_{w}.csv'); out['mse'][w]=paired(t[t.config=='tanh+relu'],t[t.config=='relu'])
out['layout']={}
t=read('layout_128.csv')
for c in ['tanh+relu','tanh+leaky_relu']:
    g=t[t.config==c]
    out['layout'][c]=paired(g[(g.layout=='interleave')&(g.ratio==.75)],g[(g.layout=='interleave')&(g.ratio==.5)])
out['lr_cells']=[]
for f in ['lrsweep_64.csv','lrsweep_128.csv','c10_lrsweep_128.csv']:
    t=read(f)
    for (cfg,lr),g in t.groupby(['config','lr']):
        if (f.startswith('c10') and lr in [.0003,.003,.03]) or (not f.startswith('c10') and lr in [.01,.03,.1]):
            out['lr_cells'].append(dict(file=f,config=cfg,lr=lr,n=len(g),below50=int((g.val_acc<50).sum()),below15=int((g.val_acc<15).sum()),final_mean=g.val_acc.mean(),best_mean=g.best_val_acc.mean()))
out['timing_repeats']=[]
for f,sw in [('main_128.csv','lrsweep_128.csv'),('c10_main_128.csv','c10_lrsweep_128.csv')]:
    a=read(f); b=read(sw)
    m=a.merge(b,on=['seed','config','lr'],suffixes=('_a','_b'))
    out['timing_repeats'].append(dict(file=f,n=len(m),accuracy_differences=int((m.test_acc_best_a!=m.test_acc_best_b).sum()),mean_time_ratio=(m.seconds_b/m.seconds_a).mean(),min_ratio=(m.seconds_b/m.seconds_a).min(),max_ratio=(m.seconds_b/m.seconds_a).max()))
out['class_delta']={}
for f,cfg in [('main_128.csv','tanh+relu'),('c10_main_512.csv','tanh+relu')]:
    t=read(f); groups={}
    for c in ['relu',cfg]:
        cms=np.stack([np.fromstring(v,sep='|').reshape(10,10) for v in t[t.config==c].test_confusion_best])
        rec=100*np.diagonal(cms,axis1=1,axis2=2)/cms.sum(2)
        prec=100*np.diagonal(cms,axis1=1,axis2=2)/cms.sum(1)
        groups[c]={'recall':rec.mean(0).tolist(),'precision':prec.mean(0).tolist()}
    out['class_delta'][f]=groups
out['main_boundary']={}
for dataset,prefix in [('mnist',''),('cifar10','c10_')]:
    t=pd.concat([read(f'{prefix}main_{w}.csv') for w in [12,32,64,128,256,512]])
    out['main_boundary'][dataset]=dict(best_epoch40=int((t.best_val_epoch==40).sum()),n=len(t),val_below50=int((t.val_acc<50).sum()),val_below15=int((t.val_acc<15).sum()))
O.joinpath('diagnostic_details.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
