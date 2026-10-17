import lib,pandas as pd,numpy as np
df=pd.read_pickle('all.pkl'); out=open('out/E_misc.md','w')
M=df[df.stage=='main']
# E: seed-level stability: SD and worst seed
rows=[]
for ds in ['mnist','cifar10']:
    for w in lib.WID:
        m=M[(M.ds==ds)&(M.hidden==w)]
        g=m.groupby('cfg').test_acc_best
        sd=g.std(); mn=g.min(); p10=g.quantile(0.1); rng=g.max()-g.min()
        rows.append({'dataset':'MNIST' if ds=='mnist' else 'CIFAR-10','W':w,
          'SD relu':f"{sd['relu']:.3f}",'SD tanh+relu':f"{sd['tanh+relu']:.3f}",'SD tanh+leaky':f"{sd['tanh+leaky_relu']:.3f}",
          'worst seed relu':f"{mn['relu']:.2f}",'worst seed tanh+relu':f"{mn['tanh+relu']:.2f}",'worst seed tanh+leaky':f"{mn['tanh+leaky_relu']:.2f}",
          '10th pct relu':f"{p10['relu']:.2f}",'10th pct tanh+relu':f"{p10['tanh+relu']:.2f}"})
t=pd.DataFrame(rows); out.write("\n### SEEDSTAB\n"+lib.md(t)); print(t.to_string())
# homogeneous-mean SD vs hybrid-mean SD summary
s=M.groupby(['ds','hidden','cfg']).test_acc_best.std().unstack()
print((s['tanh+relu']/s['relu']).round(2).to_string()); print((s['tanh+leaky_relu']/s['leaky_relu']).round(2).to_string())
# K: convergence - epochs to reach within 0.5 pt (mnist) / 1pt (cifar) of own run's best val
def ep_to(ev,thr):
    b=max(ev)
    for i,v in enumerate(ev):
        if v>=b-thr: return i+1
rows=[]
for ds,thr in [('mnist',0.5),('cifar10',1.0)]:
    for w in lib.WID:
        m=M[(M.ds==ds)&(M.hidden==w)]
        r={'dataset':'MNIST' if ds=='mnist' else 'CIFAR-10','W':w}
        for c in ['relu','tanh','tanh+relu','tanh+leaky_relu']:
            x=m[m.cfg==c]; r[f'{c}: epoch 1 val']=f"{np.mean([e[0] for e in x.ev]):.2f}"; r[f'{c}: epochs to −{thr} of best']=f"{np.mean([ep_to(e,thr) for e in x.ev]):.1f}"
        rows.append(r)
t=pd.DataFrame(rows); out.write("\n### CONV\n"+lib.md(t)); print(t.to_string())
# generalization gap
rows=[]
for ds in ['mnist','cifar10']:
    g=M[M.ds==ds].assign(gap=lambda x:x.train_clean_best-x.test_acc_best).groupby(['cfg','hidden']).gap.mean().unstack().reindex(lib.ORDER).round(2)
    g.columns=[f'W={w}' for w in g.columns]; g=g.reset_index().rename(columns={'cfg':'config'})
    out.write(f"\n### GAP {ds}\n"+lib.md(g)); print(g.to_string())
# I: MSE
MS=df[df.stage=='mse']; rows=[];prow=[]
for w in [64,128]:
    m=MS[MS.hidden==w]
    for c in ['relu','tanh','tanh+relu']:
        x=m[m.cfg==c]; rows.append({'W':w,'config':c,'LR':f"{x.lr.iloc[0]:g}",'test acc (ckpt)':f"{x.test_acc_best.mean():.3f} ± {x.test_acc_best.std():.3f}",'macro-F1':f"{x.test_f1_best.mean():.3f}",'final-epoch acc':f"{x.test_acc.mean():.3f}",'sel. epoch':f"{x.best_val_epoch.mean():.1f}"})
    for ref in ['relu','tanh']:
        q=lib.paired(m[m.cfg=='tanh+relu'],m[m.cfg==ref]); prow.append({'W':w,'comparison':f'tanh+relu − {ref}','Δ acc':f"{q['delta']:+.3f}",'95% CI':f"[{q['lo']:+.3f}, {q['hi']:+.3f}]",'wins':f"{q['wins']}/30",'p':lib.fp(q['p'])})
    ce=M[(M.ds=='mnist')&(M.hidden==w)]
    for c in ['relu','tanh','tanh+relu']:
        q=lib.paired(ce[ce.cfg==c],m[m.cfg==c]); prow.append({'W':w,'comparison':f'{c}: CE − MSE','Δ acc':f"{q['delta']:+.3f}",'95% CI':f"[{q['lo']:+.3f}, {q['hi']:+.3f}]",'wins':f"{q['wins']}/30",'p':lib.fp(q['p'])})
out.write("\n### MSE\n"+lib.md(pd.DataFrame(rows))+"\n"+lib.md(pd.DataFrame(prow))); print(pd.DataFrame(rows).to_string()); print(pd.DataFrame(prow).to_string())
# MSE calib grid 64
CM=df[df.stage=='calib_mse']
for w in [64,128]:
    t=CM[CM.hidden==w].groupby(['cfg','lr']).val_acc.mean().unstack().round(2); t.columns=[f'lr={l:g}' for l in t.columns]
    co=CM[CM.hidden==w].assign(c=lambda x:x.val_acc<15).groupby(['cfg']).c.sum()
    t['collapsed runs /48']=co; t=t.reset_index().rename(columns={'cfg':'config'})
    out.write(f"\n### MSECAL W={w}\n"+lib.md(t)); print(t.to_string())
out.close()
