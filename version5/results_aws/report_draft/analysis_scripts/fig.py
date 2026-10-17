import lib,pandas as pd,numpy as np,matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
df=pd.read_pickle('all.pkl'); pc=pd.read_pickle('paired_main.pkl'); FD=lib.R+'/report_draft/figures/'
COL={'relu':'#1f77b4','tanh+relu':'#d62728','leaky_relu':'#2ca02c','tanh+leaky_relu':'#ff7f0e','tanh':'#7f7f7f','sigmoid':'#9467bd','sigmoid+relu':'#8c564b'}
# Fig1 hybrid - parents vs width
fig,ax=plt.subplots(1,2,figsize=(11,4))
for i,ds in enumerate(['mnist','cifar10']):
    for h,ref,c,ls in [('tanh+relu','relu','#d62728','-'),('tanh+relu','tanh','#d62728','--'),('tanh+leaky_relu','leaky_relu','#ff7f0e','-'),('tanh+leaky_relu','tanh','#ff7f0e','--')]:
        s=pc[(pc.ds==ds)&(pc.hybrid==h)&(pc.ref==ref)].sort_values('W')
        x=np.arange(6); ax[i].plot(x,s.delta,ls,color=c,marker='o',label=f'{h} − {ref}'); ax[i].fill_between(x,s.lo,s.hi,color=c,alpha=.12)
    ax[i].axhline(0,color='k',lw=.8); ax[i].set_xticks(range(6)); ax[i].set_xticklabels(lib.WID); ax[i].set_xlabel('hidden width W'); ax[i].set_ylabel('paired Δ test accuracy (pp), checkpoint'); ax[i].set_title({'mnist':'MNIST','cifar10':'CIFAR-10'}[ds]); ax[i].grid(alpha=.3)
ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(FD+'fig1_paired_delta_vs_width.png',dpi=160); plt.close()
# Fig2 collapse rate vs LR
S=df[df.stage=='lrsweep']
fig,ax=plt.subplots(1,3,figsize=(14,4))
for i,(name,ds,w) in enumerate([('MNIST W=64','mnist',64),('MNIST W=128','mnist',128),('CIFAR-10 W=128','cifar10',128)]):
    s=S[(S.ds==ds)&(S.hidden==w)]
    for c in ['relu','tanh','tanh+relu','leaky_relu','tanh+leaky_relu']:
        g=s[s.cfg==c].groupby('lr').val_acc.mean(); ax[i].plot(g.index,g.values,marker='o',ms=3,color=COL[c],label=c,lw=2 if c in('relu','tanh+relu') else 1)
    ax[i].set_xscale('log'); ax[i].set_title(name); ax[i].set_xlabel('learning rate'); ax[i].set_ylabel('mean final validation accuracy (%)'); ax[i].grid(alpha=.3)
ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(FD+'fig2_lr_sweep_final_val.png',dpi=160); plt.close()
fig,ax=plt.subplots(1,3,figsize=(14,4))
for i,(name,ds,w) in enumerate([('MNIST W=64','mnist',64),('MNIST W=128','mnist',128),('CIFAR-10 W=128','cifar10',128)]):
    s=S[(S.ds==ds)&(S.hidden==w)]; lrs=sorted(s.lr.unique()); x=np.arange(len(lrs))
    for j,c in enumerate(['relu','tanh+relu']):
        thr=50 if ds=='mnist' else 15; g=s[s.cfg==c].assign(k=lambda d:d.val_acc<thr).groupby('lr').k.sum().reindex(lrs)
        ax[i].bar(x+(j-.5)*.4,g.values,.4,color=COL[c],label=c)
    ax[i].set_xticks(x); ax[i].set_xticklabels([f'{l:g}' for l in lrs],rotation=45); ax[i].set_title(f'{name}  (failure: final val < {thr}%)'); ax[i].set_ylabel('failed seeds (/30)'); ax[i].set_xlabel('learning rate')
ax[0].legend(); fig.tight_layout(); fig.savefig(FD+'fig3_collapse_counts.png',dpi=160); plt.close()
# Fig4 ratio curve
L=df[df.stage=='layout']; T=df[(df.stage=='lrsweep')&(df.ds=='mnist')&(df.hidden==128)&(df.lr==0.01)]
fig,ax=plt.subplots(figsize=(6,4))
for pair,c in [('tanh+relu','#d62728'),('tanh+leaky_relu','#ff7f0e')]:
    a2=pair.split('+')[1]; xs=[0,25,50,75,100]
    grp=[T[T.cfg=='tanh']]+[L[(L.cfg==pair)&(L.ratio==r)&(L.layout=='interleave')] for r in [.25,.5,.75]]+[T[T.cfg==a2]]
    m=[g.test_acc_best.mean() for g in grp]; se=[g.test_acc_best.std()/np.sqrt(len(g)) for g in grp]
    ax.errorbar(xs,m,yerr=[1.96*e for e in se],marker='o',color=c,capsize=3,label=f'tanh → {a2}')
ax.set_xlabel('rectifier share ρ (%) — 0 = pure tanh, 100 = pure rectifier'); ax.set_ylabel('test accuracy (%), checkpoint'); ax.set_title('MNIST W=128, LR=0.01 (mean ± 95% normal CI)'); ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(FD+'fig4_ratio_curve.png',dpi=160); plt.close()
print('ok')
