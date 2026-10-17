import lib,pandas as pd
frames=[]
for ds,pre in [('mnist',''),('cifar10','c10_')]:
    for w in lib.WID:
        for st in ['main','calib']:
            d=lib.load(f'{pre}{st}_{w}.csv'); d['stage']=st; d['ds']=ds; frames.append(d)
for f,st,ds in [('lrsweep_64.csv','lrsweep','mnist'),('lrsweep_128.csv','lrsweep','mnist'),('c10_lrsweep_128.csv','lrsweep','cifar10'),('layout_128.csv','layout','mnist'),('mse_64.csv','mse','mnist'),('mse_128.csv','mse','mnist'),('calib_mse_64.csv','calib_mse','mnist'),('calib_mse_128.csv','calib_mse','mnist')]:
    d=lib.load(f); d['stage']=st; d['ds']=ds; frames.append(d)
df=pd.concat(frames,ignore_index=True)
keep=[c for c in df.columns if c not in ('epoch_loss','epoch_train_acc','epoch_val_f1','test_confusion','epoch_train_acc_clean')]
df[keep].to_pickle('all.pkl')
print(df.groupby(['ds','stage']).size())
