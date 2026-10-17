import lib,pandas as pd,numpy as np
df=pd.read_pickle('all.pkl'); out=open('out/D_ratio.md','w')
L=df[df.stage=='layout']; S=df[(df.stage=='lrsweep')&(df.ds=='mnist')&(df.hidden==128)&(df.lr==0.01)]
M=df[(df.stage=='main')&(df.ds=='mnist')&(df.hidden==128)]
def grp(pair):
    act2=pair.split('+')[1]
    g={0.0:S[S.cfg=='tanh']}
    for r in [0.25,0.5,0.75]: g[r]=L[(L.cfg==pair)&(L.ratio==r)&(L.layout=='interleave')]
    g[1.0]=S[S.cfg==act2]
    return g
rows=[];prow=[]
for pair in ['tanh+relu','tanh+leaky_relu']:
    act2=pair.split('+')[1]; g=grp(pair)
    for r,x in g.items():
        lab={0.0:'0% (pure tanh)',1.0:f'100% (pure {act2})'}.get(r,f'{int(r*100)}%')
        rows.append({'pair':pair,f'rectifier share ρ':lab,'n':len(x),'test acc (ckpt)':f"{x.test_acc_best.mean():.3f} ± {x.test_acc_best.std():.3f}",
          'macro-F1 (ckpt)':f"{x.test_f1_best.mean():.3f} ± {x.test_f1_best.std():.3f}",'min seed':f"{x.test_acc_best.min():.2f}",'max seed':f"{x.test_acc_best.max():.2f}",'final-epoch test acc':f"{x.test_acc.mean():.3f}",
          'best val acc':f"{x.best_val_acc.mean():.2f}",'sel. epoch':f"{x.best_val_epoch.mean():.1f}",'peak−final drop':f"{x['drop'].mean():.3f}",'train−test gap':f"{(x.train_clean_best-x.test_acc_best).mean():.2f}"})
    keys=list(g.keys())
    comps=[(0.25,0.0),(0.5,0.25),(0.75,0.5),(1.0,0.75),(0.5,0.0),(0.75,0.0),(0.25,1.0),(0.5,1.0),(0.75,1.0),(0.75,0.25)]
    for a,b in comps:
        for col,nm in [('test_acc_best','acc'),('test_f1_best','F1')]:
            q=lib.paired(g[a],g[b],col)
            prow.append({'pair':pair,'comparison':f"ρ={int(a*100)}% − ρ={int(b*100)}%",'metric':nm,'Δ (pp)':f"{q['delta']:+.3f}",'95% CI':f"[{q['lo']:+.3f}, {q['hi']:+.3f}]",'wins':f"{q['wins']}/30",'paired t p':lib.fp(q['p'])})
out.write("\n### RATIO table (MNIST W=128, all at LR=0.01)\n"+lib.md(pd.DataFrame(rows)))
pr=pd.DataFrame(prow); out.write("\n### RATIO paired\n"+lib.md(pr[pr.metric=='acc'])); out.write("\n### RATIO paired F1\n"+lib.md(pr[pr.metric=='F1']))
print(pd.DataFrame(rows).to_string()); print(pr.to_string())
# versus calibrated homogeneous baselines from main grid
rows=[]
for pair in ['tanh+relu','tanh+leaky_relu']:
    for r in [0.25,0.5,0.75]:
        x=L[(L.cfg==pair)&(L.ratio==r)&(L.layout=='interleave')]
        r_={'pair':pair,'ρ':f'{int(r*100)}%'}
        for ref in ['relu','leaky_relu','tanh']:
            q=lib.paired(x,M[M.cfg==ref]); r_[f'Δ vs calibrated {ref} [95% CI]']=f"{q['delta']:+.3f} [{q['lo']:+.3f}, {q['hi']:+.3f}] ({q['wins']}/30)"
        rows.append(r_)
out.write("\n### RATIO vs calibrated baselines\n"+lib.md(pd.DataFrame(rows))); print(pd.DataFrame(rows).to_string())
# layout
rows=[]
for pair in ['tanh+relu','tanh+leaky_relu']:
    base=L[(L.cfg==pair)&(L.ratio==0.5)&(L.layout=='interleave')]
    for lay in ['interleave','block','random']:
        x=L[(L.cfg==pair)&(L.ratio==0.5)&(L.layout==lay)]
        q=lib.paired(x,base) if lay!='interleave' else None
        rows.append({'pair':pair,'layout':lay,'test acc (ckpt)':f"{x.test_acc_best.mean():.3f} ± {x.test_acc_best.std():.3f}",'macro-F1':f"{x.test_f1_best.mean():.3f}",
                     'Δ vs interleave [95% CI]':'—' if q is None else f"{q['delta']:+.3f} [{q['lo']:+.3f}, {q['hi']:+.3f}]",'wins':'—' if q is None else f"{q['wins']}/30",'p':'—' if q is None else lib.fp(q['p'])})
out.write("\n### LAYOUT\n"+lib.md(pd.DataFrame(rows))); print(pd.DataFrame(rows).to_string())
out.close()
