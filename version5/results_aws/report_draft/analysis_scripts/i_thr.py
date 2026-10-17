import lib,pandas as pd,numpy as np
df=pd.read_pickle('all.pkl'); S=df[df.stage=='lrsweep']; C=df[df.stage=='calib']; rows=[]
def add(name,s,thr,lab):
    cr=int((s[s.cfg=='relu'].val_acc<thr).sum()); ch=int((s[s.cfg=='tanh+relu'].val_acc<thr).sum()); cl=int((s[s.cfg=='leaky_relu'].val_acc<thr).sum()); ct=int((s[s.cfg=='tanh'].val_acc<thr).sum()); n=int((s.cfg=='relu').sum())
    rows.append({'setting':name,'failure definition':lab,'runs/config':n,'ReLU':cr,'tanh':ct,'leaky-ReLU':cl,'tanh+ReLU':ch,'ReLU ÷ tanh+ReLU':(f"{cr/ch:.2f}×" if ch else '∞')})
for name,ds,w in [('MNIST W=64 sweep (30 seeds × 11 LRs)','mnist',64),('MNIST W=128 sweep (30 seeds × 11 LRs)','mnist',128)]:
    s=S[(S.ds==ds)&(S.hidden==w)]
    for thr,lab in [(15,'final val < 15% (near chance)'),(50,'final val < 50%'),(90,'final val < 90% (useful accuracy lost)')]: add(name,s,thr,lab)
s=S[(S.ds=='cifar10')&(S.hidden==128)]
for thr,lab in [(15,'final val < 15% (near chance)'),(25,'final val < 25%')]: add('CIFAR-10 W=128 sweep (30 seeds × 8 LRs)',s,thr,lab)
add('CIFAR-10 W=128 sweep, LR = 0.03 only',s[s.lr==0.03],15,'final val < 15% (near chance)')
for ds,nm,thrs in [('mnist','MNIST calibration, 6 widths (6 seeds × 8 LRs)',[(15,'final val < 15%'),(50,'final val < 50%'),(90,'final val < 90%')]),('cifar10','CIFAR-10 calibration, 6 widths (6 seeds × 8 LRs)',[(15,'final val < 15%'),(25,'final val < 25%')])]:
    for thr,lab in thrs: add(nm,C[C.ds==ds],thr,lab)
t=pd.DataFrame(rows); open('out/I_thr.md','w').write("\n### THR\n"+lib.md(t)); print(t.to_string())
