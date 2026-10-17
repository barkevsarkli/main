"""Rebuild all assets, then add the verified 1 October 2026 diagnostics."""
from build_assets import *

arr=lambda s:np.fromstring(s,sep='|')
summary={}
# Six-seed calibration data are reported separately from 30-seed evaluation.
rows=[];calrows=[]
for ds in ['mnist','cifar10']:
    for c in ['relu','tanh','tanh+relu']:
        row=['MNIST' if ds=='mnist' else 'CIFAR-10',NAME[c]]
        for h in [12,32,64,128,256,512]:
            q=sel(D,stage='calib',dataset=ds,hidden=h,config=c,lr=.03)
            row.append(f'{q.val_acc.mean():.1f} ({int((q.val_acc<15).sum())})')
            calrows.append(dict(dataset=ds,hidden=h,config=c,n=len(q),final_mean=q.val_acc.mean(),fail15=int((q.val_acc<15).sum()),drop_mean=(q.best_val_acc-q.val_acc).mean()))
        rows.append(row)
table('width_stability','CE calibration at LR=0.03: mean final validation accuracy (\\%) and final failures below 15\\% in parentheses. Each cell has six seeds.','tab:widthstability',['Data','Model','12','32','64','128','256','512'],rows,align='llrrrrrr',size='footnotesize',note='These are tuning seeds 101--106, used here for an exploratory width check. They are not additional 30-seed tests.')
pd.DataFrame(calrows).to_csv(HERE/'width_stability.csv',index=False)

# Loss is accumulated online, before each per-example update.
rows=[];lossrows=[]
for c in ['relu','leaky_relu','tanh+leaky_relu','tanh+relu']:
    q=sel(D,stage='lrsweep',dataset='mnist',hidden=128,config=c,lr=.03).sort_values('seed')
    a=np.stack(q.epoch_loss.map(arr));v=np.stack(q.epoch_val_acc.map(arr));jump=np.diff(a,axis=1);ep=jump.argmax(1)+2;mini=a.argmin(1)+1
    rows.append([NAME[c],f'{a[:,-1].mean():.4f} $\\pm$ {a[:,-1].std(ddof=1):.4f}',f'{mini.min()}--{mini.max()} ({np.median(mini):g})',f'{ep.min()}--{ep.max()}',f'{jump.max(1).mean():.3f}'])
    for j,(_,r) in enumerate(q.iterrows()):
        hits=np.flatnonzero(v[j]<15);first=int(hits[0]+1) if len(hits) else None
        lossrows.append(dict(config=c,seed=int(r.seed),final_loss=float(a[j,-1]),min_loss_epoch=int(mini[j]),largest_jump_epoch=int(ep[j]),largest_jump=float(jump[j].max()),first_below15=first,min_to_failure=None if first is None else int(first-mini[j])))
table('loss_signatures','MNIST H=128, LR=0.03: online CE loss patterns over 30 seeds. SD is the sample standard deviation.','tab:losssignatures',['Model','Final CE, mean $\\pm$ SD','Min. epoch (median)','Max. jump epoch','Mean jump'],rows,align='lrrrr',size='footnotesize',note='The largest jump is the maximum consecutive-epoch increase, computed separately for each seed. Its epoch range is meaningful only when the jump is large. Loss is collected while weights change; it is not a fixed-checkpoint validation loss.')
pd.DataFrame(lossrows).to_csv(HERE/'loss_signatures.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(7,3.1),sharey=True)
for ax,h in zip(axs,[64,128]):
    for c in ['relu','leaky_relu','tanh+leaky_relu','tanh+relu']:
        q=sel(D,stage='lrsweep',dataset='mnist',hidden=h,config=c,lr=.03);a=np.stack(q.epoch_loss.map(arr));mu=a.mean(0);sd=a.std(0,ddof=1)
        ax.plot(np.arange(1,41),mu,color=COL[c],label=NAME[c],lw=1.5);ax.fill_between(np.arange(1,41),np.maximum(0,mu-sd),mu+sd,color=COL[c],alpha=.12)
    ax.axhline(np.log(10),color='gray',ls=':',lw=1);ax.set_title(f'MNIST, H={h}, LR=0.03');ax.set_xlabel('Epoch');ax.grid(alpha=.12)
axs[0].set_ylabel('Online training CE');axs[1].text(2,np.log(10)+.12,'ln(10)',fontsize=8,color='gray');legend(fig,axs[0],4);fig.tight_layout(rect=[0,.14,1,1]);save(fig,'loss_dynamics')

# Large-drop counts across the full evaluation grid, not only favorable rates.
rows=[]
for h in [64,128]:
    for c in ['relu','tanh','leaky_relu','tanh+relu']:
        row=[h,NAME[c]]
        for lr in [.02,.03,.05,.1]:
            q=sel(D,stage='lrsweep',dataset='mnist',hidden=h,config=c,lr=lr);row.append(f'{int(((q.best_val_acc-q.val_acc)>10).sum())}/30')
        rows.append(row)
table('drop_threshold','MNIST runs losing more than 10 pp from peak to final validation accuracy.','tab:dropthreshold',['H','Model','LR=0.02','0.03','0.05','0.1'],rows,align='rlrrrr',size='small',note='A zero count does not ensure a high score: a run that never learns can also have a small drop.')

# Normalize each curve by its own empirical best mean-final-validation LR.
fig,axs=plt.subplots(1,2,figsize=(7,3.1),sharey=True);normrows=[]
for ax,h in zip(axs,[64,128]):
    for c in ['relu','tanh','leaky_relu','sigmoid','tanh+relu']:
        q=sel(L,dataset='mnist',hidden=h,config=c).sort_values('lr');best=q.loc[q.final_val.idxmax()];rel=q.lr/best.lr
        ax.plot(rel,q.final_val-best.final_val,'o-',ms=2.5,lw=1.2,label=NAME[c],color=COL[c])
        edge=best.lr
        for _,r in q[q.lr>=best.lr].iterrows():
            if r.final_val<best.final_val-5:break
            edge=r.lr
        normrows.append(dict(hidden=h,config=c,best_lr=best.lr,upper_5pp=edge,relative_upper=edge/best.lr))
    ax.axhline(-5,ls=':',color='gray');ax.axvline(1,ls=':',color='gray');ax.set_xscale('log');ax.set_title(f'MNIST, H={h}');ax.set_xlabel('LR / own best sampled LR');ax.set_ylim(-90,2);ax.grid(alpha=.12)
axs[0].set_ylabel('Final minus best score (pp)');legend(fig,axs[0],3);fig.tight_layout(rect=[0,.20,1,1]);save(fig,'normalized_lr')
pd.DataFrame(normrows).to_csv(HERE/'normalized_lr.csv',index=False)

fig,axs=plt.subplots(1,2,figsize=(7,3.0),sharey=True);cifrows=[]
for ax,lr in zip(axs,[.01,.03]):
    for c in ['relu','tanh','leaky_relu','tanh+relu']:
        q=sel(D,stage='lrsweep',dataset='cifar10',hidden=128,config=c,lr=lr);a=np.stack(q.epoch_val_acc.map(arr));mu=a.mean(0);sd=a.std(0,ddof=1)
        ax.plot(np.arange(1,41),mu,color=COL[c],label=NAME[c],lw=1.3);ax.fill_between(np.arange(1,41),np.maximum(0,mu-sd),mu+sd,color=COL[c],alpha=.10)
        cifrows.append(dict(lr=lr,config=c,epoch1_fail15=int((a[:,0]<15).sum()),ever_fail15=int((a<15).any(1).sum()),final_fail15=int((a[:,-1]<15).sum()),peak_mean=q.best_val_acc.mean(),final_mean=q.val_acc.mean(),best_epoch_median=q.best_val_epoch.median(),mean_curve_peak=a.mean(0).max()))
    ax.axhline(15,color='gray',ls=':');ax.set_title(f'CIFAR-10, H=128, LR={lr:g}');ax.set_xlabel('Epoch');ax.set_ylim(0,55);ax.grid(alpha=.12)
axs[0].set_ylabel('Validation accuracy (%)');legend(fig,axs[0],4);fig.tight_layout(rect=[0,.14,1,1]);save(fig,'cifar_dynamics')
pd.DataFrame(cifrows).to_csv(HERE/'cifar_dynamics.csv',index=False)
table('cifar_onset','CIFAR-10 H=128: final failure and failure at any logged epoch are distinct measures. Each row has 30 seeds.','tab:cifaronset',['LR','Model','Epoch 1 $<15\\%$','Any epoch $<15\\%$','Final $<15\\%$','Median best epoch'],[[rate(q['lr']),NAME[q['config']],q['epoch1_fail15'],q['ever_fail15'],q['final_fail15'],f"{q['best_epoch_median']:g}"] for q in cifrows],align='llrrrr',size='footnotesize')

rows=[]
for h in [64,128]:
    for c in ['relu','tanh','tanh+relu']:
        cells=[]
        for lr in [.03,.1]:
            q=sel(D,stage='calib_mse',dataset='mnist',hidden=h,config=c,lr=lr);cells.append(f'{q.val_acc.mean():.3f} ({int((q.val_acc<15).sum())}/6)')
        rows.append([h,NAME[c]]+cells)
table('mse_stability','MNIST MSE calibration: mean final validation accuracy (\\%) and final failure counts below 15\\%. Each cell has six tuning seeds.','tab:msestability',['H','Model','LR=0.03','LR=0.1'],rows,align='rlrr',size='small',note='These runs use sigmoid outputs and the implemented MSE derivative scale. They do not isolate loss choice from output activation or update scale.')

# Additional endpoint comparisons use the same documented bootstrap convention.
rows=[];endrows=[]
for h in [12,32,64,128,256,512]:
    a=sel(D,stage='main',dataset='cifar10',hidden=h,config='tanh+relu');b=sel(D,stage='main',dataset='cifar10',hidden=h,config='relu')
    d,lo,hi=pair(a,b,'test_acc');dc,lc,hc=pair(a,b,'test_acc_best')
    rows.append([h,rate(b.lr.iloc[0]),rate(a.lr.iloc[0]),f'{d:+.3f}',ci(lo,hi),f'{dc:+.3f}',ci(lc,hc)])
    endrows.append(dict(hidden=h,final_delta=d,final_ci_low=lo,final_ci_high=hi,checkpoint_delta=dc,checkpoint_ci_low=lc,checkpoint_ci_high=hc))
table('endpoints','CIFAR-10 tanh--ReLU minus ReLU: final and selected test checkpoints. All differences are in pp; intervals are pointwise paired 95\\% bootstrap intervals.','tab:endpoints',['H','R LR','T+R LR','Final diff.','95\\% interval','Selected diff.','95\\% interval'],rows,align='rrrrrrr',size='footnotesize',note='The added final-epoch comparisons are exploratory and have no multiple-test correction. H=64 also changes LR. Main claims of selected-checkpoint accuracy retain the original Holm correction.')
pd.DataFrame(endrows).to_csv(HERE/'endpoint_comparisons.csv',index=False)
a=sel(D,stage='lrsweep',dataset='mnist',hidden=128,config='tanh+relu',lr=.03);b=sel(D,stage='lrsweep',dataset='mnist',hidden=128,config='relu',lr=.03)
d,lo,hi=pair(a,b);bc=np.stack(b.test_confusion.map(lambda s:arr(s).reshape(10,10)));shares=100*bc.sum(1).max(1)/bc.sum((1,2))
summary['mnist_checkpoint_lr003']=dict(relu=b.test_acc_best.mean(),hybrid=a.test_acc_best.mean(),delta=d,ci_low=lo,ci_high=hi,wins=int((a.sort_values('seed').test_acc_best.to_numpy()>b.sort_values('seed').test_acc_best.to_numpy()).sum()))
summary['relu_prediction_share']=dict(exactly_one_class=int((shares==100).sum()),minimum_share=float(shares.min()),maximum_share=float(shares.max()))
summary['cifar_first_epoch_below15']=next(q['epoch1_fail15'] for q in cifrows if q['lr']==.03 and q['config']=='relu')
assert summary['relu_prediction_share']['exactly_one_class']==26
assert summary['cifar_first_epoch_below15']==29
assert summary['mnist_checkpoint_lr003']['wins']==30
(HERE/'revision_verification.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
