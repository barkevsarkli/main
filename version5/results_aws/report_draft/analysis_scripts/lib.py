import pandas as pd, numpy as np, math, os
R=os.path.expanduser('~/mnt/results_aws')
NUM=['seed','ratio','hidden','lr','epochs','seconds','train_acc','val_acc','val_f1','test_acc','test_f1','best_val_acc','best_val_epoch','train_acc_clean','test_acc_best','test_f1_best']
ORDER=['relu','leaky_relu','tanh','sigmoid','tanh+relu','tanh+leaky_relu','sigmoid+relu']
HYB={'tanh+relu':('tanh','relu'),'tanh+leaky_relu':('tanh','leaky_relu'),'sigmoid+relu':('sigmoid','relu')}
HOMO=['relu','leaky_relu','tanh','sigmoid']
WID=[12,32,64,128,256,512]
def load(f):
    df=pd.read_csv(os.path.join(R,f),dtype=str); df=df[df.seed!='seed'].copy()
    for c in NUM:
        if c in df: df[c]=df[c].astype(float)
    df['seed']=df.seed.astype(int); df['hidden']=df.hidden.astype(int)
    df['cfg']=np.where(df.act1==df.act2,df.act1,df.act1+'+'+df.act2)
    ev=df.epoch_val_acc.str.split('|').apply(lambda x:[float(v) for v in x])
    df['ev']=ev
    df['best_epoch_idx']=df.best_val_epoch.astype(int)
    def tr(r):
        a=r.epoch_train_acc_clean.split('|'); i=int(r.best_val_epoch)-1
        return float(a[i]) if 0<=i<len(a) else np.nan
    df['train_clean_best']=df.apply(tr,axis=1)
    df['drop']=df.best_val_acc-df.val_acc
    return df
def betacf(a,b,x):
    MAXIT=300;EPS=3e-16;FPMIN=1e-300
    qab=a+b;qap=a+1;qam=a-1;c=1;d=1-qab*x/qap
    if abs(d)<FPMIN:d=FPMIN
    d=1/d;h=d
    for m in range(1,MAXIT+1):
        m2=2*m;aa=m*(b-m)*x/((qam+m2)*(a+m2))
        d=1+aa*d; d=FPMIN if abs(d)<FPMIN else d; c=1+aa/c; c=FPMIN if abs(c)<FPMIN else c; d=1/d;h*=d*c
        aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2))
        d=1+aa*d; d=FPMIN if abs(d)<FPMIN else d; c=1+aa/c; c=FPMIN if abs(c)<FPMIN else c; d=1/d;de=d*c;h*=de
        if abs(de-1)<EPS:break
    return h
def betai(a,b,x):
    if x<=0:return 0.0
    if x>=1:return 1.0
    bt=math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log(1-x))
    if x<(a+1)/(a+b+2): return bt*betacf(a,b,x)/a
    return 1-bt*betacf(b,a,1-x)/b
def ttest_p(d):
    d=np.asarray(d,float);n=len(d);sd=d.std(ddof=1)
    if sd==0: return 1.0 if d.mean()==0 else 0.0
    t=d.mean()/(sd/math.sqrt(n));df=n-1
    return betai(df/2,0.5,df/(df+t*t))
RNG=np.random.default_rng(12345)
def boot(d,B=10000):
    d=np.asarray(d,float);n=len(d);idx=RNG.integers(0,n,(B,n));m=d[idx].mean(1)
    return np.percentile(m,2.5),np.percentile(m,97.5)
def paired(a,b,col='test_acc_best'):
    x=a.set_index('seed')[col];y=b.set_index('seed')[col];s=x.index.intersection(y.index)
    d=(x[s]-y[s]).values;lo,hi=boot(d)
    return dict(n=len(d),delta=d.mean(),lo=lo,hi=hi,wins=int((d>0).sum()),p=ttest_p(d))
def fp(p):
    return f"{p:.3f}" if p>=0.001 else f"{p:.1e}"
def holm(ps):
    ps=np.asarray(ps);o=np.argsort(ps);m=len(ps);adj=np.empty(m);run=0
    for k,i in enumerate(o):
        run=max(run,min(1,(m-k)*ps[i]));adj[i]=run
    return adj
def md(df,floatfmt=None):
    cols=list(df.columns);s='| '+' | '.join(map(str,cols))+' |\n|'+'|'.join(['---']*len(cols))+'|\n'
    for _,r in df.iterrows(): s+='| '+' | '.join(str(v) for v in r.values)+' |\n'
    return s
