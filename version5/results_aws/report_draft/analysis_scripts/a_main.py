import lib,pandas as pd,numpy as np
df=pd.read_pickle('all.pkl'); M=df[df.stage=='main']
out=open('out/A_main.md','w')
P=lambda H,D:H*H+(D+12)*H+10
for ds,D in [('mnist',784),('cifar10',3072)]:
    m=M[M.ds==ds]
    g=m.groupby(['cfg','hidden'])
    for col,name in [('test_acc_best','accuracy'),('test_f1_best','macro-F1')]:
        t=g[col].mean().unstack().reindex(lib.ORDER); s=g[col].std().unstack().reindex(lib.ORDER)
        rows=[]
        for c in lib.ORDER:
            row={'config':c}
            for w in lib.WID:
                best=t[w].max(); v=f"{t.loc[c,w]:.2f} ± {s.loc[c,w]:.2f}"
                row[f'W={w}']=f"**{v}**" if t.loc[c,w]==best else v
            rows.append(row)
        out.write(f"\n### WIDE {ds} {name}\n"+lib.md(pd.DataFrame(rows)))
    # ranks
    t=g.test_acc_best.mean().unstack().reindex(lib.ORDER); rk=t.rank(ascending=False).astype(int)
    rk['mean rank']=rk.mean(1).round(2); rk=rk.reset_index().rename(columns={'cfg':'config'})
    rk.columns=['config']+[f'W={w}' for w in lib.WID]+['mean rank']
    out.write(f"\n### RANK {ds}\n"+lib.md(rk))
    # detailed per width
    for w in lib.WID:
        rows=[]
        for c in lib.ORDER:
            x=m[(m.cfg==c)&(m.hidden==w)]
            rows.append({'config':c,'LR':f"{x.lr.iloc[0]:g}",'acc (ckpt)':f"{x.test_acc_best.mean():.3f} ± {x.test_acc_best.std():.3f}",
              'F1 (ckpt)':f"{x.test_f1_best.mean():.3f} ± {x.test_f1_best.std():.3f}",'min seed acc':f"{x.test_acc_best.min():.2f}",
              'final-epoch acc':f"{x.test_acc.mean():.3f}",'sel. epoch':f"{x.best_val_epoch.mean():.1f}",
              'train−test gap':f"{(x.train_clean_best-x.test_acc_best).mean():.2f}",'peak−final val drop':f"{x['drop'].mean():.2f}",'time (s)':f"{x.seconds.mean():.0f}"})
        out.write(f"\n### DETAIL {ds} W={w} params={P(w,D):,}\n"+lib.md(pd.DataFrame(rows)))
# paired comparisons
rows=[]
for ds in ['mnist','cifar10']:
    for w in lib.WID:
        m=M[(M.ds==ds)&(M.hidden==w)]
        means=m.groupby('cfg').test_acc_best.mean()
        besthomo=means[lib.HOMO].idxmax()
        for h,(p1,p2) in lib.HYB.items():
            for ref in [p1,p2,'best_homo','parent_mean']:
                if ref=='best_homo': rname=besthomo; b=m[m.cfg==besthomo]
                elif ref=='parent_mean':
                    a1=m[m.cfg==p1].set_index('seed').test_acc_best; a2=m[m.cfg==p2].set_index('seed').test_acc_best
                    b=pd.DataFrame({'seed':a1.index,'test_acc_best':((a1+a2)/2).values}); rname='mean of parents'
                else: rname=ref; b=m[m.cfg==ref]
                r=lib.paired(m[m.cfg==h],b); r.update(ds=ds,W=w,hybrid=h,ref=ref,refname=rname)
                rf=lib.paired(m[m.cfg==h],b if ref!='parent_mean' else None,'test_f1_best') if ref!='parent_mean' else None
                if rf: r.update(dF1=rf['delta'],F1lo=rf['lo'],F1hi=rf['hi'],pF1=rf['p'])
                rows.append(r)
pc=pd.DataFrame(rows)
fam=pc.ref.isin(['tanh','relu','leaky_relu','sigmoid'])
pc.loc[fam,'holm']=lib.holm(pc.loc[fam,'p'].values)
pc.to_pickle('paired_main.pkl')
out.close()
print(pc[pc.ref!='parent_mean'][['ds','W','hybrid','refname','delta','lo','hi','wins','p','holm']].round(3).to_string())
print(pc[pc.ref=='parent_mean'][['ds','W','hybrid','delta','lo','hi','wins','p']].round(3).to_string())
