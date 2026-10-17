import lib,pandas as pd,numpy as np
df=pd.read_pickle('all.pkl'); out=open('out/G_align.md','w')
C=df[df.stage=='calib']; S=df[df.stage=='lrsweep']; M=df[df.stage=='main']
sel=[]
for (ds,w,c),x in C.groupby(['ds','hidden','cfg']):
    a=x.groupby('lr').val_acc.mean(); b=x.groupby('lr').best_val_acc.mean()
    sel.append(dict(ds=ds,W=w,cfg=c,lr_final=a.idxmax(),lr_best=b.idxmax()))
sel=pd.DataFrame(sel); ch=sel[sel.lr_final!=sel.lr_best]
print(len(ch)); print(ch.to_string())
t=ch.copy(); t['dataset']=t.ds.map({'mnist':'MNIST','cifar10':'CIFAR-10'}); t=t[['dataset','W','cfg','lr_final','lr_best']]; t.columns=['dataset','W','config','LR (final val criterion, used)','LR (best-val criterion)']
out.write("\n### ALIGNCHANGES\n"+lib.md(t))
rows=[]
for name,ds,w in [('MNIST','mnist',64),('MNIST','mnist',128),('CIFAR-10','cifar10',128)]:
    s=S[(S.ds==ds)&(S.hidden==w)]; ss=sel[(sel.ds==ds)&(sel.W==w)].set_index('cfg')
    for h,p in [('tanh+relu','relu'),('tanh+relu','tanh'),('tanh+leaky_relu','leaky_relu'),('tanh+leaky_relu','tanh'),('sigmoid+relu','relu')]:
        for crit,col in [('original (final val)','lr_final'),('aligned (best val)','lr_best')]:
            a=s[(s.cfg==h)&(s.lr==ss.loc[h,col])]; b=s[(s.cfg==p)&(s.lr==ss.loc[p,col])]
            q=lib.paired(a,b)
            rows.append({'setting':f'{name} W={w}','comparison':f'{h} − {p}','LR criterion':crit,'LRs (hybrid / parent)':f"{ss.loc[h,col]:g} / {ss.loc[p,col]:g}",'Δ acc (ckpt)':f"{q['delta']:+.3f}",'95% CI':f"[{q['lo']:+.3f}, {q['hi']:+.3f}]",'wins':f"{q['wins']}/30",'p':lib.fp(q['p'])})
    # oracle: each config at its own best eval-seed LR (by mean best val) - upper bound
    g=s.groupby(['cfg','lr']).best_val_acc.mean()
    for h,p in [('tanh+relu','relu'),('tanh+leaky_relu','leaky_relu')]:
        lh=g[h].idxmax(); lp=g[p].idxmax(); q=lib.paired(s[(s.cfg==h)&(s.lr==lh)],s[(s.cfg==p)&(s.lr==lp)])
        rows.append({'setting':f'{name} W={w}','comparison':f'{h} − {p}','LR criterion':'dense 11-pt sweep, best val on eval seeds (optimistic)','LRs (hybrid / parent)':f"{lh:g} / {lp:g}",'Δ acc (ckpt)':f"{q['delta']:+.3f}",'95% CI':f"[{q['lo']:+.3f}, {q['hi']:+.3f}]",'wins':f"{q['wins']}/30",'p':lib.fp(q['p'])})
t=pd.DataFrame(rows); out.write("\n### ALIGN\n"+lib.md(t)); print(t.to_string())
# peak accuracy across whole LR sweep: best mean test_acc_best over LRs
rows=[]
for name,ds,w in [('MNIST','mnist',64),('MNIST','mnist',128),('CIFAR-10','cifar10',128)]:
    s=S[(S.ds==ds)&(S.hidden==w)]; g=s.groupby(['cfg','lr']).test_acc_best.mean()
    r={'setting':f'{name} W={w}'}
    for c in lib.ORDER: r[c]=f"{g[c].max():.3f} @ {g[c].idxmax():g}"
    rows.append(r)
t=pd.DataFrame(rows); out.write("\n### ENVELOPE best test acc over sweep (descriptive, test-selected)\n"+lib.md(t)); print(t.to_string())
out.close()
