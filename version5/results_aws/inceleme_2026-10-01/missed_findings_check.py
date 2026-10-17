"""Reproduces the numbers in kacirilan_bulgular_inceleme.md from the raw results_aws CSVs.
Run: python3 missed_findings_check.py   (needs pandas, numpy, scipy)"""
import glob, os, math
import numpy as np, pandas as pd
from scipy import stats
R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
pd.set_option('display.width', 250)

def load():
    out = []
    for f in sorted(glob.glob(R + '/*.csv')):
        d = pd.read_csv(f, dtype=str); d = d[d.seed != 'seed']; d['file'] = os.path.basename(f); out.append(d)
    d = pd.concat(out, ignore_index=True)
    for c in ['seed','ratio','hidden','lr','seconds','val_acc','test_acc','best_val_acc','best_val_epoch','test_acc_best']:
        d[c] = pd.to_numeric(d[c])
    d['ds'] = np.where(d.file.str.startswith('c10'), 'C', 'M')
    d['model'] = [a if (a == b or r == 0) else f'{a}+{b}' for a, b, r in zip(d.act1, d.act2, d.ratio)]
    d['group'] = d.file.str.replace(r'_\d+\.csv', '', regex=True)
    d['drop'] = d.best_val_acc - d.val_acc
    return d
arr = lambda s: np.array([float(x) for x in s.split('|')])
d = load()

print('\n[1] Calibration grid (6 seeds): mean final val and #final<15, CE')
c = d[d.group.isin(['calib', 'c10_calib']) & d.model.isin(['relu','tanh+relu','tanh','leaky_relu','sigmoid'])]
for ds in 'MC':
    x = c[(c.ds == ds) & c.lr.isin([0.01, 0.03, 0.1])]
    print(ds); print(x.pivot_table(index=['lr','model'], columns='hidden', values='val_acc', aggfunc='mean').round(1))
    print(x.assign(f=x.val_acc < 15).pivot_table(index=['lr','model'], columns='hidden', values='f', aggfunc='sum'))

print('\n[2-3] Loss signatures at MNIST LR=0.03 (lrsweep)')
for H in [64, 128]:
    for m in ['relu','leaky_relu','tanh+leaky_relu','tanh+relu']:
        y = d[(d.group == 'lrsweep') & (d.lr == 0.03) & (d.hidden == H) & (d.model == m)]
        L = np.stack([arr(s) for s in y.epoch_loss]); V = np.stack([arr(s) for s in y.epoch_val_acc])
        jump_ep = np.diff(L, axis=1).argmax(1) + 2; minep = L.argmin(1) + 1
        ff = np.array([np.where(v < 15)[0][0] + 1 if (v < 15).any() else np.nan for v in V])
        print(H, m, 'finalL mean %.4f sd %.4f' % (L[:, -1].mean(), L[:, -1].std()), '| ln10=%.4f' % math.log(10),
              '| min-loss epoch median', np.median(minep), '| biggest jump epoch range', jump_ep.min(), jump_ep.max(),
              'size %.2f' % np.diff(L, axis=1).max(1).mean(), '| lead to fail', np.nanmean(ff - minep).round(1) if (~np.isnan(ff)).any() else '-')
y = d[(d.group == 'lrsweep') & (d.hidden == 128) & (d.lr == 0.03) & (d.model == 'relu')]
cols = [np.array([float(v) for v in s.split('|') if v]).reshape(10, 10).sum(0) for s in y.test_confusion]
print('ReLU H128 runs predicting exactly one class:', sum(c.max() == c.sum() for c in cols), '/30')

print('\n[4] Runs with peak-final drop >10 pp (of 30), MNIST lrsweep')
x = d[d.group == 'lrsweep']
print(x.assign(b=x['drop'] > 10).pivot_table(index=['hidden','model'], columns='lr', values='b', aggfunc='sum'))

print('\n[5] Best LR and last LR within 5 pp of own best')
for g in ['lrsweep', 'c10_lrsweep']:
    x = d[d.group == g]
    for H in sorted(x.hidden.unique()):
        for m in ['relu','tanh','sigmoid','leaky_relu','tanh+relu']:
            s = x[(x.hidden == H) & (x.model == m)].groupby('lr').val_acc.mean(); b = s.idxmax()
            up = max(l for l in s.index if l >= b and (s.loc[b:l] >= s.max() - 5).all())
            print(g, H, m, 'best', b, 'edge', up, 'ratio %.1f' % (up / b))

print('\n[6] MSE calibration (6 seeds) mean final val')
print(d[d.group == 'calib_mse'].pivot_table(index=['hidden','model'], columns='lr', values='val_acc', aggfunc='mean').round(1))

print('\n[7] CIFAR lrsweep H=128 dynamics')
for lr in [0.01, 0.03]:
    for m in ['relu','tanh+relu','tanh','leaky_relu']:
        y = d[(d.group == 'c10_lrsweep') & (d.lr == lr) & (d.model == m)]
        V = np.stack([arr(s) for s in y.epoch_val_acc])
        print(lr, m, 'val ep1/10/40', V.mean(0)[[0, 9, 39]].round(1), 'best epoch median', y.best_val_epoch.median(),
              'ever<15', int((V < 15).any(1).sum()), 'final<15', int((V[:, -1] < 15).sum()))

print('\n[8] Shared init/order: cross-model correlation of final-val residuals by seed')
for g, H in [('c10_main', 512), ('c10_main', 128), ('main', 128)]:
    P = d[(d.group == g) & (d.hidden == H)].pivot_table(index='seed', columns='model', values='val_acc')
    C = (P - P.mean()).corr().values[np.triu_indices(7, 1)]
    print(g, H, 'mean r %.2f' % C.mean())

print('\n[9] Final-epoch vs checkpoint paired T+R - ReLU (main grids)')
rng = np.random.default_rng(20261001)
x = d[d.group.isin(['main', 'c10_main'])]
for ds in 'MC':
    for H in [12, 32, 64, 128, 256, 512]:
        a = x[(x.ds == ds) & (x.hidden == H) & (x.model == 'relu')].sort_values('seed')
        b = x[(x.ds == ds) & (x.hidden == H) & (x.model == 'tanh+relu')].sort_values('seed')
        df = b.test_acc.values - a.test_acc.values
        bs = [df[rng.integers(0, 30, 30)].mean() for _ in range(10000)]
        print(ds, H, 'LRs', a.lr.iloc[0], b.lr.iloc[0], 'final diff %+.3f' % df.mean(), np.percentile(bs, [2.5, 97.5]).round(3),
              'ckpt diff %+.3f' % (b.test_acc_best.values - a.test_acc_best.values).mean())
y = d[(d.group == 'lrsweep') & (d.hidden == 128) & (d.lr == 0.03)]
a = y[y.model == 'relu'].sort_values('seed').test_acc_best.values; b = y[y.model == 'tanh+relu'].sort_values('seed').test_acc_best.values
print('MNIST H128 LR0.03 checkpoint test: ReLU %.2f, T+R %.2f, diff %+.3f, wins %d/30' % (a.mean(), b.mean(), (b - a).mean(), (b > a).sum()))
