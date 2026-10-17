import lib,pandas as pd,numpy as np
out=open('out/F_class.md','w')
CN={'cifar10':['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'],'mnist':[str(i) for i in range(10)]}
def perclass(conf):
    c=np.array(conf.split('|'),float).reshape(10,10)  # assume row=true
    tp=np.diag(c); prec=tp/np.maximum(c.sum(0),1); rec=tp/np.maximum(c.sum(1),1)
    return 100*2*prec*rec/np.maximum(prec+rec,1e-12), 100*rec
rows=[]
for ds,pre,ws in [('cifar10','c10_',[64,128,256,512]),('mnist','',[128])]:
    for w in ws:
        d=pd.read_csv(f'{lib.R}/{pre}main_{w}.csv',dtype=str,usecols=['seed','act1','act2','test_confusion_best','test_acc_best']); d=d[d.seed!='seed']
        d['cfg']=np.where(d.act1==d.act2,d.act1,d.act1+'+'+d.act2); d['seed']=d.seed.astype(int)
        # check orientation: row sums should be class supports (1000 for cifar)
        c0=np.array(d.test_confusion_best.iloc[0].split('|'),float).reshape(10,10)
        if w==ws[0]: print(ds,'row sums',c0.sum(1)[:4],'col sums',c0.sum(0)[:4])
        F={c:{} for c in ['relu','tanh','tanh+relu']}
        for _,r in d[d.cfg.isin(F)].iterrows(): F[r.cfg][r.seed]=perclass(r.test_confusion_best)[0]
        seeds=sorted(F['relu'])
        dr=np.array([F['tanh+relu'][s]-F['relu'][s] for s in seeds]); dt=np.array([F['tanh+relu'][s]-F['tanh'][s] for s in seeds])
        for k in range(10):
            lo,hi=lib.boot(dr[:,k])
            rows.append({'dataset':'CIFAR-10' if ds=='cifar10' else 'MNIST','W':w,'class':CN[ds][k],'F1 relu':f"{np.mean([F['relu'][s][k] for s in seeds]):.2f}",'F1 tanh+relu':f"{np.mean([F['tanh+relu'][s][k] for s in seeds]):.2f}",
                         'Δ vs relu [95% CI]':f"{dr[:,k].mean():+.2f} [{lo:+.2f}, {hi:+.2f}]",'wins':f"{int((dr[:,k]>0).sum())}/30",'Δ vs tanh':f"{dt[:,k].mean():+.2f}"})
t=pd.DataFrame(rows); out.write("\n### CLASS\n"+lib.md(t)); print(t.to_string()); out.close()
