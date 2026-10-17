import lib,pandas as pd,numpy as np
df=pd.read_pickle('all.pkl'); out=open('out/B_rob.md','w')
CH=15.0
def cell(x):
    return dict(final=x.val_acc.mean(),best=x.best_val_acc.mean(),drop=x['drop'].mean(),coll=int((x.val_acc<CH).sum()),n=len(x),
                deg5=int((x['drop']>5).sum()),test=x.test_acc_best.mean())
for name,sel in [('MNIST W=64','(df.stage=="lrsweep")&(df.ds=="mnist")&(df.hidden==64)'),('MNIST W=128','(df.stage=="lrsweep")&(df.ds=="mnist")&(df.hidden==128)'),('CIFAR-10 W=128','(df.stage=="lrsweep")&(df.ds=="cifar10")&(df.hidden==128)')]:
    s=df[eval(sel)]; lrs=sorted(s.lr.unique())
    res={(c,l):cell(s[(s.cfg==c)&(s.lr==l)]) for c in lib.ORDER for l in lrs}
    for key,lab,fmt in [('final','mean final val acc (%)','{:.2f}'),('drop','mean peak−final val drop (pts)','{:.2f}'),('coll','near-chance collapses (final val <15%), /30','{}'),('deg5','seeds losing >5 pts from own peak, /30','{}')]:
        rows=[]
        for c in lib.ORDER:
            r={'config':c}
            for l in lrs: r[f'{l:g}']=fmt.format(res[(c,l)][key])
            rows.append(r)
        out.write(f"\n### SWEEP {name} {lab}\n"+lib.md(pd.DataFrame(rows)))
    # summary robustness
    rows=[]
    for c in lib.ORDER:
        fin=np.array([res[(c,l)]['final'] for l in lrs]); co=np.array([res[(c,l)]['coll'] for l in lrs]); dr=np.array([res[(c,l)]['deg5'] for l in lrs])
        pk=fin.max()
        def maxlr(ok):
            # largest LR such that all LRs from best-LR up to it satisfy ok (contiguous above optimum)
            i=int(fin.argmax()); j=i
            while j+1<len(lrs) and ok[j+1]: j+=1
            return lrs[j]
        m1=maxlr(fin>=pk-1); mc=maxlr(co==0); m5=maxlr((fin>=pk-5)&(co==0))
        hi=[k for k,l in enumerate(lrs) if l>=0.003]
        rows.append({'config':c,'best LR':f"{lrs[int(fin.argmax())]:g}",'peak final val':f"{pk:.2f}",'max LR within 1 pt':f"{m1:g}",'max LR within 5 pts & 0 collapses':f"{m5:g}",
          'max LR with 0 collapses':f"{mc:g}",'total collapses (all LRs)':int(co.sum()),'total >5-pt degradations':int(dr.sum()),
          'mean final val, LR≥0.003':f"{fin[hi].mean():.2f}",'mean final val, all LRs':f"{fin.mean():.2f}"})
    out.write(f"\n### SWEEPSUM {name}\n"+lib.md(pd.DataFrame(rows)))
    print(name); print(pd.DataFrame(rows).to_string())
# calibration grids: all widths, both datasets (6 seeds)
rows=[]
for ds in ['mnist','cifar10']:
    for w in lib.WID:
        s=df[(df.stage=='calib')&(df.ds==ds)&(df.hidden==w)]; lrs=sorted(s.lr.unique())
        for c in lib.ORDER:
            fin=np.array([s[(s.cfg==c)&(s.lr==l)].val_acc.mean() for l in lrs]); co=np.array([int((s[(s.cfg==c)&(s.lr==l)].val_acc<CH).sum()) for l in lrs])
            i=int(fin.argmax());pk=fin[i]
            def mx(ok):
                j=i
                while j+1<len(lrs) and ok[j+1]: j+=1
                return lrs[j]
            rows.append(dict(ds=ds,W=w,cfg=c,bestlr=lrs[i],peak=pk,m1=mx(fin>=pk-1),m2=mx(fin>=pk-2),m5=mx((fin>=pk-5)&(co==0)),mc=mx(co==0),coll=int(co.sum()),nrun=len(lrs)*6))
cal=pd.DataFrame(rows); cal.to_pickle('calrob.pkl')
for ds in ['mnist','cifar10']:
  for key,lab in [('m2','largest LR (≥ own best LR) keeping mean final val within 2 pts of peak'),('mc','largest LR (≥ own best LR) with no calibration seed below 15%'),('coll','near-chance collapsed calibration runs (of 48 per cell)')]:
    t=cal[cal.ds==ds].pivot(index='cfg',columns='W',values=key).reindex(lib.ORDER)
    t=t.map(lambda v:f"{v:g}"); t.columns=[f'W={w}' for w in t.columns]; t=t.reset_index().rename(columns={'cfg':'config'})
    out.write(f"\n### CAL {ds} {lab}\n"+lib.md(t)); print(ds,key); print(t.to_string())
out.close()
