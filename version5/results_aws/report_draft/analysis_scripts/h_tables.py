import lib,pandas as pd,numpy as np
df=pd.read_pickle('all.pkl'); pc=pd.read_pickle('paired_main.pkl'); out=open('out/H_tables.md','w')
DS={'mnist':'MNIST','cifar10':'CIFAR-10'}
pc=pc[~((pc.ref=='best_homo')&(pc.refname.isin(['relu','leaky_relu','tanh','sigmoid'])) & False)]
def ci(r): return f"{r.delta:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]"
# paired vs parents per dataset
for ds in ['mnist','cifar10']:
    rows=[]
    for _,r in pc[(pc.ds==ds)&pc.ref.isin(['relu','leaky_relu','tanh','sigmoid'])].iterrows():
        rows.append({'W':r.W,'comparison':f"{r.hybrid} − {r.refname}",'Δ acc (pp) [95% CI]':ci(r),'wins':f"{r.wins}/30",'p':lib.fp(r.p),'Holm p (72 tests)':lib.fp(r.holm),'Δ macro-F1 (pp) [95% CI]':f"{r.dF1:+.3f} [{r.F1lo:+.3f}, {r.F1hi:+.3f}]"})
    out.write(f"\n### PAIRED {ds}\n"+lib.md(pd.DataFrame(rows)))
# synergy: hybrid - mean of parents, compact wide
for key,lab in [('parent_mean','SYNERGY'),('best_homo','BESTHOMO')]:
    rows=[]
    for ds in ['mnist','cifar10']:
        for h in lib.HYB:
            r_={'dataset':DS[ds],'hybrid':h}
            for w in lib.WID:
                r=pc[(pc.ds==ds)&(pc.W==w)&(pc.hybrid==h)&(pc.ref==key)].iloc[0]
                sig='*' if (r.lo>0 or r.hi<0) else ''
                r_[f'W={w}']=f"{r.delta:+.2f}{sig} ({r.wins}/30)" + (f" vs {r.refname}" if key=='best_homo' else '')
            rows.append(r_)
    out.write(f"\n### {lab}\n"+lib.md(pd.DataFrame(rows)))
# scoreboard
rows=[]
for ds in ['mnist','cifar10']:
    for h,(p1,p2) in lib.HYB.items():
        sub=pc[(pc.ds==ds)&(pc.hybrid==h)]
        def cnt(ref,cond):
            s=sub[sub.ref==ref]; return int(cond(s).sum())
        rect=p2
        rows.append({'dataset':DS[ds],'hybrid':h,
          'widths sig. better than bounded parent (tanh or sigmoid)':f"{cnt(p1,lambda s:s.lo>0)}/6",
          f'widths sig. better than rectifier parent':f"{cnt(rect,lambda s:s.lo>0)}/6",
          'widths not sig. worse than rectifier parent':f"{cnt(rect,lambda s:s.hi>=0)}/6",
          'widths sig. above parent mean':f"{cnt('parent_mean',lambda s:s.lo>0)}/6",
          'widths ranked #1 of 7':''})
M=df[df.stage=='main']
t=M.groupby(['ds','hidden','cfg']).test_acc_best.mean()
for r in rows:
    ds={'MNIST':'mnist','CIFAR-10':'cifar10'}[r['dataset']]
    r['widths ranked #1 of 7']=f"{sum(t[ds][w].idxmax()==r['hybrid'] for w in lib.WID)}/6"
out.write("\n### SCOREBOARD\n"+lib.md(pd.DataFrame(rows))); print(pd.DataFrame(rows).to_string())
# final-epoch appendix wide + paired vs relu
for ds in ['mnist','cifar10']:
    m=M[M.ds==ds]; g=m.groupby(['cfg','hidden'])
    tt=g.test_acc.mean().unstack().reindex(lib.ORDER); ss=g.test_acc.std().unstack().reindex(lib.ORDER)
    rows=[{'config':c,**{f'W={w}':f"{tt.loc[c,w]:.2f} ± {ss.loc[c,w]:.2f}" for w in lib.WID}} for c in lib.ORDER]
    out.write(f"\n### FINALWIDE {ds}\n"+lib.md(pd.DataFrame(rows)))
    rows=[]
    for h in lib.HYB:
        for ref in ['relu',lib.HYB[h][0]] + ([lib.HYB[h][1]] if lib.HYB[h][1]!='relu' else []):
            r_={'comparison':f'{h} − {ref}'}
            for w in lib.WID:
                x=m[m.hidden==w]; q=lib.paired(x[x.cfg==h],x[x.cfg==ref],'test_acc')
                r_[f'W={w}']=f"{q['delta']:+.2f} [{q['lo']:+.2f}, {q['hi']:+.2f}]"
            rows.append(r_)
    out.write(f"\n### FINALPAIRED {ds}\n"+lib.md(pd.DataFrame(rows)))
# calibration grids appendix
C=df[df.stage=='calib']
for ds in ['mnist','cifar10']:
    for w in lib.WID:
        x=C[(C.ds==ds)&(C.hidden==w)]; tt=x.groupby(['cfg','lr']).val_acc.mean().unstack().reindex(lib.ORDER)
        co=x.assign(c=x.val_acc<15).groupby(['cfg','lr']).c.sum().unstack().reindex(lib.ORDER)
        rows=[{'config':c,**{f'{l:g}':(f"{tt.loc[c,l]:.2f}"+(f" ({int(co.loc[c,l])}✗)" if co.loc[c,l]>0 else '')) for l in tt.columns}} for c in lib.ORDER]
        out.write(f"\n### CALGRID {ds} W={w}\n"+lib.md(pd.DataFrame(rows)))
# clean SAMELR
S=df[df.stage=='lrsweep']; rows=[]
for name,ds,w in [('MNIST W=64','mnist',64),('MNIST W=128','mnist',128),('CIFAR-10 W=128','cifar10',128)]:
    s=S[(S.ds==ds)&(S.hidden==w)]
    for l in sorted(s.lr.unique()):
        q1=lib.paired(s[(s.cfg=='tanh+relu')&(s.lr==l)],s[(s.cfg=='relu')&(s.lr==l)],'val_acc')
        q2=lib.paired(s[(s.cfg=='tanh+leaky_relu')&(s.lr==l)],s[(s.cfg=='leaky_relu')&(s.lr==l)],'val_acc')
        rows.append({'setting':name,'LR':f'{l:g}','tanh+ReLU − ReLU [95% CI]':f"{q1['delta']:+.2f} [{q1['lo']:+.2f}, {q1['hi']:+.2f}]",'wins (1)':f"{q1['wins']}/30",
                     'tanh+leaky − leaky [95% CI]':f"{q2['delta']:+.2f} [{q2['lo']:+.2f}, {q2['hi']:+.2f}]",'wins (2)':f"{q2['wins']}/30"})
out.write("\n### SAMELR2\n"+lib.md(pd.DataFrame(rows)))
# clean CONV (MNIST thr 0.5, CIFAR thr 1.0)
def ep_to(ev,thr):
    b=max(ev)
    for i,v in enumerate(ev):
        if v>=b-thr: return i+1
rows=[]
for ds,thr in [('mnist',0.5),('cifar10',1.0)]:
    for w in lib.WID:
        m=M[(M.ds==ds)&(M.hidden==w)]; r={'dataset':DS[ds],'W':w,'tolerance':f'{thr} pt'}
        for c in ['relu','tanh','tanh+relu','tanh+leaky_relu']:
            x=m[m.cfg==c]; r[f'{c}: epoch-1 val']=f"{np.mean([e[0] for e in x.ev]):.2f}"; r[f'{c}: epochs to tol.']=f"{np.mean([ep_to(e,thr) for e in x.ev]):.1f}"
        rows.append(r)
out.write("\n### CONV2\n"+lib.md(pd.DataFrame(rows)))
out.close()
