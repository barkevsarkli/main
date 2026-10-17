import lib,pandas as pd,numpy as np
df=pd.read_pickle('all.pkl'); out=open('out/C_more.md','w')
S=df[df.stage=='lrsweep']
# same-LR paired final val: tanh+relu - relu, tanh+leaky - leaky
rows=[]
for name,ds,w in [('MNIST W=64','mnist',64),('MNIST W=128','mnist',128),('CIFAR-10 W=128','cifar10',128)]:
    s=S[(S.ds==ds)&(S.hidden==w)]
    for l in sorted(s.lr.unique()):
        r={'setting':name,'LR':f'{l:g}'}
        for h,p in [('tanh+relu','relu'),('tanh+leaky_relu','leaky_relu')]:
            a=s[(s.cfg==h)&(s.lr==l)];b=s[(s.cfg==p)&(s.lr==l)]
            q=lib.paired(a,b,'val_acc')
            r[f'{h} − {p}: Δ final val [95% CI]']=f"{q['delta']:+.2f} [{q['lo']:+.2f}, {q['hi']:+.2f}]"
            r[f'wins']=f"{q['wins']}/30" if h=='tanh+relu' else r['wins']
            if h!='tanh+relu': r['wins ']=f"{q['wins']}/30"
        rows.append(r)
out.write("\n### SAMELR paired final val\n"+lib.md(pd.DataFrame(rows)))
# 3x summary
rows=[]
for name,ds,w in [('MNIST W=64','mnist',64),('MNIST W=128','mnist',128),('CIFAR-10 W=128','cifar10',128)]:
    s=S[(S.ds==ds)&(S.hidden==w)]
    for rng,lab in [(lambda l:l>=0,'all LRs'),(lambda l:l>=0.01,'LR ≥ 0.01')]:
        ss=s[rng(s.lr)]
        cr=int((ss[ss.cfg=='relu'].val_acc<15).sum()); ch=int((ss[ss.cfg=='tanh+relu'].val_acc<15).sum()); n=int((ss.cfg=='relu').sum())
        rows.append({'setting':name,'LR range':lab,'runs per config':n,'ReLU collapses':cr,'tanh+ReLU collapses':ch,'ratio ReLU/hybrid':f"{cr/ch:.2f}×" if ch else '∞',
                     'ReLU collapse rate':f"{100*cr/n:.1f}%",'hybrid collapse rate':f"{100*ch/n:.1f}%"})
    l=0.03; a=s[(s.lr==l)]
    cr=int((a[a.cfg=='relu'].val_acc<15).sum()); ch=int((a[a.cfg=='tanh+relu'].val_acc<15).sum())
    rows.append({'setting':name,'LR range':'LR = 0.03','runs per config':30,'ReLU collapses':cr,'tanh+ReLU collapses':ch,'ratio ReLU/hybrid':f"{cr/ch:.2f}×" if ch else '∞','ReLU collapse rate':f"{100*cr/30:.1f}%",'hybrid collapse rate':f"{100*ch/30:.1f}%"})
C=df[df.stage=='calib']
for ds in ['mnist','cifar10']:
    ss=C[C.ds==ds]; cr=int((ss[ss.cfg=='relu'].val_acc<15).sum()); ch=int((ss[ss.cfg=='tanh+relu'].val_acc<15).sum()); n=int((ss.cfg=='relu').sum())
    rows.append({'setting':f"{'MNIST' if ds=='mnist' else 'CIFAR-10'} calibration, all 6 widths",'LR range':'all LRs','runs per config':n,'ReLU collapses':cr,'tanh+ReLU collapses':ch,'ratio ReLU/hybrid':f"{cr/ch:.2f}×",'ReLU collapse rate':f"{100*cr/n:.1f}%",'hybrid collapse rate':f"{100*ch/n:.1f}%"})
out.write("\n### THREEX collapses\n"+lib.md(pd.DataFrame(rows))); print(pd.DataFrame(rows).to_string())
# LR tolerance ratio table from calibration
cal=pd.read_pickle('calrob.pkl')
rows=[]
for ds in ['mnist','cifar10']:
    for w in lib.WID:
        c=cal[(cal.ds==ds)&(cal.W==w)].set_index('cfg')
        rows.append({'dataset':'MNIST' if ds=='mnist' else 'CIFAR-10','W':w,
          'ReLU: max LR within 2 pts':f"{c.loc['relu','m2']:g}",'tanh+ReLU: max LR within 2 pts':f"{c.loc['tanh+relu','m2']:g}",'tolerance ratio':f"{c.loc['tanh+relu','m2']/c.loc['relu','m2']:g}×",
          'ReLU: max LR, no collapse':f"{c.loc['relu','mc']:g}",'tanh+ReLU: max LR, no collapse':f"{c.loc['tanh+relu','mc']:g}",'no-collapse ratio':f"{c.loc['tanh+relu','mc']/c.loc['relu','mc']:g}×",
          'collapsed calib runs ReLU / hybrid':f"{c.loc['relu','coll']} / {c.loc['tanh+relu','coll']}"})
out.write("\n### LRTOL calibration\n"+lib.md(pd.DataFrame(rows))); print(pd.DataFrame(rows).to_string())
# Pareto: accuracy vs robustness, MNIST128 sweep
rows=[]
M=df[df.stage=='main']
for name,ds,w in [('MNIST W=64','mnist',64),('MNIST W=128','mnist',128),('CIFAR-10 W=128','cifar10',128)]:
    s=S[(S.ds==ds)&(S.hidden==w)]; m=M[(M.ds==ds)&(M.hidden==w)]
    for c in lib.ORDER:
        x=s[s.cfg==c]; g=x.groupby('lr').val_acc.mean()
        hi=x[x.lr>=0.003] if ds=='mnist' else x[x.lr>=0.001]
        rows.append({'setting':name,'config':c,'calibrated test acc (ckpt)':f"{m[m.cfg==c].test_acc_best.mean():.2f}",'collapse rate, all LRs':f"{100*(x.val_acc<15).mean():.1f}%",
                     'mean final val at LR ≥ '+('0.003' if ds=='mnist' else '0.001'):f"{hi.val_acc.mean():.2f}",'worst-LR-cell mean final val (≤0.03)':f"{g[g.index<=0.03].min():.2f}"})
out.write("\n### PARETO\n"+lib.md(pd.DataFrame(rows))); print(pd.DataFrame(rows).to_string())
out.close()
