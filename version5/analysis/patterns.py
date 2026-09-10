#!/usr/bin/env python3
"""Pattern mining for the article: goes beyond summary_extended.md.
numpy + pandas only (no scipy): p-values via exact sign test / permutation.

The evaluation grid is read at BALANCED_N seeds for every width and both datasets, matching
analysis/analyze.py, so pooled tests weight each width equally instead of counting whichever
width happened to be run longest."""
import os, pandas as pd, numpy as np, math, itertools, collections
rng = np.random.default_rng(0)
BALANCED_N = 20
RES = os.environ.get("V5_RESULTS", "results")
def load(fn, cap=True):
    df = pd.read_csv(f"{RES}/{fn}", dtype=str)
    df = df[(df.seed != "seed") & (df.status == "ok")].copy()
    for c in ["seed","hidden"]: df[c] = df[c].astype(int)
    if cap:
        df = df[df.seed <= BALANCED_N]
    for c in ["ratio","lr","test_acc","test_f1","val_acc","val_f1","train_acc"]: df[c] = df[c].astype(float)
    df["cfg"] = np.where(df.ratio == 0, df.act1, df.act1 + "+" + df.act2)
    df["ep_val"] = df.epoch_val_acc.apply(lambda s: [float(x) for x in s.split("|")])
    df["ep_loss"] = df.epoch_loss.apply(lambda s: [float(x) for x in s.split("|")])
    df["conf"] = df.test_confusion.apply(lambda s: np.array([int(x) for x in s.split("|")]).reshape(10,10))
    return df.drop_duplicates(["seed","cfg","layout","ratio","hidden","lr","loss"])
main = {12:"main_12_v2.csv",32:"main_32.csv",64:"main_64.csv",128:"main_128_v2.csv",
        256:"main_256.csv",512:"main_512.csv"}
M = {w: load(f) for w,f in main.items()}
# CIFAR-10 replication; widths appear here only once their stage has finished.
c10 = {w: f"c10_main_{w}.csv" for w in (12,32,64,128,256,512)}
C = {w: load(f) for w,f in c10.items() if os.path.exists(f"{RES}/{f}")}
WM = sorted(M); WC = sorted(C)
def paired_d(df, a, b, col="test_acc"):
    x = df[df.cfg==a].set_index("seed")[col]; y = df[df.cfg==b].set_index("seed")[col]
    s = x.index.intersection(y.index); return (x[s]-y[s]).values
def sign_p(d):  # two-sided exact sign test ignoring ties
    d = d[d!=0]; n=len(d); k=(d>0).sum()
    return min(1.0, 2*sum(math.comb(n,i) for i in range(min(k,n-k)+1))/2**n)
def perm_p(d, n=20000):  # paired permutation test on the mean
    m = abs(d.mean()); flips = rng.choice([-1,1], size=(n,len(d)))
    return ((abs((flips*d).mean(1)) >= m-1e-12).mean())
def ci(d): 
    b = rng.choice(d, size=(10000,len(d))).mean(1); return np.percentile(b,[2.5,97.5])

print("="*90); print(f"P1. Pooled (stratified by width) paired tests -- hybrid minus parent, test accuracy (n={BALANCED_N}/width)")
def pooled(D, a, b, ws, tag):
    ws = [w for w in ws if w in D]
    if not ws: return
    d = np.concatenate([paired_d(D[w],a,b) for w in ws])
    lo,hi = ci(d)
    print(f"  {tag:9s} {a:16s} - {b:11s} widths {str(ws):26s} n={len(d):3d} meanΔ={d.mean():+.3f} "
          f"CI[{lo:+.2f},{hi:+.2f}] wins={int((d>0).sum())}/{len(d)} sign p={sign_p(d):.4f} perm p={perm_p(d):.4f}")
POOLS = [("tanh+relu","tanh",[32,64,128,256,512]), ("tanh+relu","relu",[64,128,256,512]),
         ("tanh+relu","relu",[12,32]), ("tanh+leaky_relu","leaky_relu",[64,128,256,512]),
         ("tanh+leaky_relu","tanh",[12,32,64,128,256,512]), ("tanh+relu","tanh",[12,32,64,128,256,512])]
for a,b,ws in POOLS: pooled(M, a, b, ws, "MNIST")
if C:
    print("  -- CIFAR-10, same pools over the widths finished so far:")
    for a,b,ws in POOLS: pooled(C, a, b, ws, "CIFAR-10")

print("\n"+"="*90); print("P2. Seed-to-seed standard deviation of test accuracy (is the hybrid more or less variable?)")
cfgs = ["relu","tanh","leaky_relu","tanh+relu","tanh+leaky_relu","sigmoid","sigmoid+relu"]
print("  width " + " ".join(f"{c:>16s}" for c in cfgs))
for w,df in list(M.items()) + [(f"c10-{w}", d) for w,d in C.items()]:
    print(f"  {str(w):>5s} " + " ".join(f"{df[df.cfg==c].test_acc.std(ddof=1):16.3f}" if (df.cfg==c).any() else f"{'-':>16s}" for c in cfgs))

print("\n"+"="*90); print("P3. Generalisation gap: mean(train_acc - test_acc) at final epoch")
print("  width " + " ".join(f"{c:>16s}" for c in cfgs))
for w,df in list(M.items()) + [(f"c10-{w}", d) for w,d in C.items()]:
    print(f"  {str(w):>5s} " + " ".join(f"{(df[df.cfg==c].train_acc-df[df.cfg==c].test_acc).mean():16.2f}" if (df.cfg==c).any() else f"{'-':>16s}" for c in cfgs))

print("\n"+"="*90); print("P4. Matched-learning-rate comparison at width 128 (lrsweep_128, 10 seeds, SAME lr for all configs)")
L = load("lrsweep_128.csv")
for lr in sorted(L.lr.unique()):
    sub = L[L.lr==lr]
    line = f"  lr={lr:<6g} final val: " + " ".join(f"{c}={sub[sub.cfg==c].val_acc.mean():6.2f}" for c in ["relu","tanh","leaky_relu","tanh+relu","tanh+leaky_relu"])
    if lr <= 0.03:
        d1 = paired_d(sub,"tanh+relu","relu","val_acc"); d2 = paired_d(sub,"tanh+relu","tanh","val_acc")
        line += f" | Δ(tr-relu)={d1.mean():+.2f} wins {int((d1>0).sum())}/10 p={sign_p(d1):.3f}; Δ(tr-tanh)={d2.mean():+.2f} wins {int((d2>0).sum())}/10 p={sign_p(d2):.3f}"
    print(line)
print("  -- same at width 12 (lrsweep_12):")
L12 = load("lrsweep_12.csv")
for lr in sorted(L12.lr.unique()):
    sub = L12[L12.lr==lr]
    print(f"  lr={lr:<6g} final val: " + " ".join(f"{c}={sub[sub.cfg==c].val_acc.mean():6.2f}" for c in ["relu","tanh","leaky_relu","tanh+relu","tanh+leaky_relu"]))

print("\n"+"="*90); print("P5. Convergence speed at MATCHED lr (lrsweep_128, lr=0.003 and 0.01): mean val acc after epoch 1, 2, 3, 10")
for lr in [0.003, 0.01]:
    sub = L[L.lr==lr]
    for c in ["relu","tanh","leaky_relu","tanh+relu","tanh+leaky_relu"]:
        ep = np.array(sub[sub.cfg==c].ep_val.tolist())
        print(f"  lr={lr:<6g} {c:16s} ep1={ep[:,0].mean():6.2f} ep2={ep[:,1].mean():6.2f} ep3={ep[:,2].mean():6.2f} ep10={ep[:,9].mean():6.2f}  epochs-to-97%={np.mean([(np.argmax(r>=97.0)+1) if (r>=97.0).any() else 11 for r in ep]):.1f}")

print("\n"+"="*90); print("P6. Per-class F1 difference tanh+relu - relu, pooled over widths 64/128/256 (mean over seeds, points)")
def per_class_f1(conf):
    tp = np.diag(conf); fp = conf.sum(0)-tp; fn = conf.sum(1)-tp
    p = np.where(tp+fp>0, tp/np.maximum(tp+fp,1), 0); r = np.where(tp+fn>0, tp/np.maximum(tp+fn,1), 0)
    return np.where(p+r>0, 2*p*r/np.maximum(p+r,1e-9), 0)*100
acc = collections.defaultdict(list)
for w in [w for w in (64,128,256,512) if w in M]:
    df = M[w]; A = df[df.cfg=="tanh+relu"].set_index("seed"); B = df[df.cfg=="relu"].set_index("seed")
    for s in A.index.intersection(B.index): acc[w].append(per_class_f1(A.loc[s,"conf"]) - per_class_f1(B.loc[s,"conf"]))
allw = np.concatenate([np.array(v) for v in acc.values()])
print("  digit:      " + " ".join(f"{d:6d}" for d in range(10)))
for w,v in acc.items(): print(f"  width {w:3d}:  " + " ".join(f"{x:+6.2f}" for x in np.array(v).mean(0)))
print("  pooled:     " + " ".join(f"{x:+6.2f}" for x in allw.mean(0)) + "   (sign-test p per digit: " + " ".join(f"{sign_p(allw[:,d]):.2f}" for d in range(10)) + ")")

print("\n"+"="*90); print("P7. Hybrid relative to its two parents: below both / between / above both (test acc means)")
for w,df in list(M.items()) + [(f"c10-{w}", d) for w,d in C.items()]:
    m = df.groupby("cfg").test_acc.mean()
    for hyb,p1,p2 in [("tanh+relu","relu","tanh"),("tanh+leaky_relu","leaky_relu","tanh"),("sigmoid+relu","relu","sigmoid")]:
        if hyb not in m: continue
        lo,hi = sorted([m[p1],m[p2]]); pos = "BELOW both" if m[hyb]<lo else ("ABOVE both" if m[hyb]>hi else "between")
        print(f"  W={str(w):>8s} {hyb:16s} {m[hyb]:6.2f} vs parents [{lo:6.2f},{hi:6.2f}] -> {pos}")

print("\n"+"="*90); print("P8. Layout / ratio at width 128 (layout_128.csv, 10 seeds) vs interleave 0.5")
Lay = load("layout_128.csv")
for pair in ["tanh+relu","tanh+leaky_relu"]:
    base = Lay[(Lay.cfg==pair)&(Lay.layout=="interleave")&(Lay.ratio==0.5)].set_index("seed").test_acc
    for (lay,r),g in Lay[Lay.cfg==pair].groupby(["layout","ratio"]):
        x = g.set_index("seed").test_acc; s = x.index.intersection(base.index); d = (x[s]-base[s]).values
        print(f"  {pair:16s} {lay:10s} ratio={r:<5g} mean={x.mean():6.2f} sd={x.std(ddof=1):.2f}  Δ vs interleave0.5={d.mean():+.2f} wins={int((d>0).sum())}/{len(d)}")

print("\n"+"="*90); print("P9. Calibrated LR vs width (from best_lr.json)")
import json
for tag, path in (("MNIST", f"{RES}/best_lr.json"), ("CIFAR-10", f"{RES}/best_lr_c10.json")):
    if not os.path.exists(path): continue
    bl = json.load(open(path))
    print(f"  -- {tag} ({path})")
    for c in ["relu","tanh","leaky_relu","tanh+relu","tanh+leaky_relu"]:
        a1,a2,r = (c,c,0) if "+" not in c else (c.split("+")[0], c.split("+")[1], 0.5)
        print(f"    {c:16s} " + " ".join(f"W{w}={bl.get(f'{a1}/{a2}/{r:g}/interleave/{w}/ce','-'):<8}" for w in [12,32,64,128,256,512]))

print("\n"+"="*90); print("P10. Degradation at aggressive lr=0.03 relative to each config's own best lr (mean val acc, lrsweep)")
for name,Lx in [("W=12",L12),("W=128",L)]:
    for c in ["relu","tanh","leaky_relu","tanh+relu","tanh+leaky_relu"]:
        g = Lx[Lx.cfg==c].groupby("lr").val_acc.mean()
        print(f"  {name} {c:16s} best={g.max():.2f}@{g.idxmax():g}  at0.03={g[0.03]:.2f} (drop {g.max()-g[0.03]:.2f})  at0.1={g[0.1]:.2f} (drop {g.max()-g[0.1]:.2f})")
