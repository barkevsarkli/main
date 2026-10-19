#!/usr/bin/env python3
"""A7: consistency checks -> analysis_out/partA/checks/checks.txt (and claims/checks.json).

    python3 analysis/checks_partA.py        exit status 1 if any check FAILs

  anchors      the sanity anchors of the Part A prompt (section 1.4), re-derived from the data
  clamp code   the cross-entropy clamp -log(max(p, 1e-12)) in both trainers
               (version6/src/train_util.h, version5/main.cpp)
  clamp ratio  final train loss / (1 - final train acc / 100) for every leaky-ReLU run with final
               train loss > ln 10 (v6 Stage 0 and Stage 2), per stage / dataset / architecture
  T2 vs A2     the paper-table paired differences agree with the A2 summary: means, wins and
               paired-t p identical; bootstrap CI ends within 10% of the CI width (the two use
               different resampling streams)
The "no hand-typed numbers" check is in report_partA.py, which renders FINDINGS_partA.md from
claims.json and fails on any placeholder it cannot resolve.
"""
import math
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from partA_common import OUT, V6, Claims, out_path  # noqa: E402

CLAMP = -math.log(1e-12)
LINES = []
RESULTS = []
C = Claims("checks")


def check(name, ok, detail):
    RESULTS.append(bool(ok))
    LINES.append(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    C.put(f"{re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_').lower()}.pass", bool(ok), "see checks.txt", desc=detail)


def near(x, target, tol):
    return abs(x - target) <= tol


def cell(r, **kw):
    m = np.ones(len(r), bool)
    for k, v in kw.items():
        m &= np.isclose(r[k], v) if isinstance(v, float) else (r[k] == v).to_numpy()
    return r[m]


def anchors():
    from mech_series import series
    from stats_io import runs
    s0 = runs(0)
    for m, want in (("relu", 10), ("tanh+relu", 0)):
        g = cell(s0, hidden=128, lr=0.03, model=m)
        k = int((g.final_val < 15).sum())
        check(f"S0 mlp2 MNIST H128 LR0.03 {m} final<15", k == want and len(g) == 10, f"{k}/{len(g)} (expected {want}/10)")
    ids = cell(s0, hidden=128, lr=0.03, model="relu").run_id
    se = series(0)
    last = se[se.run_id.isin(ids)].sort_values("epoch").groupby("run_id").tail(1)
    d2, d1 = last["dead|hidden2|relu"].median(), last["dead|hidden1|relu"].median()
    lbs, npred = last["lbs"].median(), last["npred"]
    check("S0 ReLU H128 LR0.03 hidden2 dead (last epoch)", near(d2, 1.0, 0.01), f"median {d2:.4f} (expected ~1.00)")
    check("S0 ReLU H128 LR0.03 hidden1 dead (last epoch)", near(d1, 0.04, 0.02), f"median {d1:.4f} (expected ~0.04)")
    check("S0 ReLU H128 LR0.03 logit_between_std (last epoch)", lbs < 1e-6,
          f"median {lbs:.3g} (expected ~3e-9)")
    check("S0 ReLU H128 LR0.03 n_pred_classes (last epoch)", bool((npred == 1).all()),
          f"values {sorted(set(npred.astype(int)))} (expected 1)")
    s2 = runs(2)
    s2 = s2[s2.seed > 10]
    for m, want_k, want_mean in (("relu", 20, 10.9), ("tanh+relu", 0, 98.2)):
        g = cell(s2, dataset="mnist", arch_label="cfg1", hidden=128, lr=0.02, model=m)
        k, mu = int((g.final_val < 15).sum()), g.final_val.mean()
        check(f"S2 ext MNIST cfg1 H128 LR0.02 {m}", k == want_k and len(g) == 20 and near(mu, want_mean, 0.1),
              f"{k}/{len(g)} final<15, mean {mu:.3f} (expected {want_k}/20, ~{want_mean})")
    for m, want in (("tanh+relu@hidden", 19), ("tanh+relu@conv", 1)):
        g = cell(s2, dataset="cifar10", arch_label="cfg2", hidden=128, lr=0.02, model=m)
        k = int((g.final_val < 15).sum())
        check(f"S2 ext CIFAR cfg2 H128 LR0.02 {m} final<15", k == want and len(g) == 20, f"{k}/{len(g)} (expected {want}/20)")
    s3 = runs(3)
    for m, want in (("tanh+relu", 97.2), ("tanh", 96.0)):
        g = cell(s3, dataset="mnist", lr=0.03, model=m)
        mu = g.final_val.mean()
        check(f"S3 MNIST LR0.03 {m} mean final", near(mu, want, 0.1), f"{mu:.3f} over n = {len(g)} (expected ~{want})")
    g = cell(s3, dataset="mnist", lr=0.03, model="relu")
    k = int((g.final_val < 15).sum())
    check("S3 MNIST LR0.03 relu final<15", k == 15 and len(g) == 15, f"{k}/{len(g)} (expected 15/15)")


def clamp_code():
    v6 = os.path.join(V6, "src", "train_util.h")
    v5 = os.path.join(os.path.dirname(V6), "version5", "main.cpp")
    for path, line, pat in ((v6, 24, r"-std::log\(std::max\(prob\[label\],\s*T\(1e-12f\)\)\)"),
                            (v5, 229, r"-std::log\(std::max\(prob\[label\],\s*1e-12f\)\)")):
        with open(path) as f:
            src = f.read().splitlines()
        hits = [i + 1 for i, s in enumerate(src) if re.search(pat, s)]
        rel = os.path.relpath(path, os.path.dirname(V6))
        check(f"clamp in {rel}", line in hits, f"-log(max(p_label, 1e-12)) at line(s) {hits} (expected {line})")


def clamp_ratio():
    """Share of exploded leaky runs whose loss / training error equals the clamp value (+-0.3)."""
    from stats_io import runs
    rows = []
    for st in (0, 2):
        r = runs(st)
        r = r[(r.model == "leaky_relu") & (r.final_train_loss > math.log(10))]
        err = 1 - r.train_acc.astype(float) / 100
        rows.append(pd.DataFrame({"stage": st, "dataset": r.dataset, "arch": r.arch_label,
                                  "ratio": r.final_train_loss / err}))
    t = pd.concat(rows)
    t["at_clamp"] = (t.ratio - CLAMP).abs() <= 0.3
    g = t.groupby(["stage", "dataset", "arch"], sort=True).agg(n=("ratio", "size"), n_at_clamp=("at_clamp", "sum"),
                                                                 ratio_min=("ratio", "min"),
                                                                 ratio_median=("ratio", "median"),
                                                                 ratio_max=("ratio", "max")).reset_index()
    for _, q in g.iterrows():
        LINES.append(f"INFO  clamp ratio S{q.stage} {q.dataset} {q.arch}: {q.n_at_clamp}/{q.n} within 0.3 of "
                     f"{CLAMP:.3f}; min {q.ratio_min:.2f}, median {q.ratio_median:.2f}, max {q.ratio_max:.2f}")
        tag = f"clamp_ratio.s{q.stage}.{q.dataset}.{q.arch}"
        C.put(f"{tag}.n_at_clamp", f"{q.n_at_clamp}/{q.n}", f"S{q.stage} (all seeds)")
        C.put(f"{tag}.ratio_min", q.ratio_min, f"S{q.stage} (all seeds)", n=q.n)
        C.put(f"{tag}.ratio_median", q.ratio_median, f"S{q.stage} (all seeds)", n=q.n)
    s0 = t[t.stage == 0]
    cnn = t[(t.stage == 2) & (t.arch != "mlp1")]
    check("clamp ratio, exploded leaky runs, S0 mlp2", bool(s0.at_clamp.all()),
          f"{int(s0.at_clamp.sum())}/{len(s0)} with final train loss / (1 - train acc) within 0.3 of -ln(1e-12) = "
          f"{CLAMP:.3f}")
    check("clamp ratio, exploded leaky runs, S2 CNN", cnn.at_clamp.mean() >= 0.9,
          f"{int(cnn.at_clamp.sum())}/{len(cnn)} within 0.3 (median {cnn.ratio.median():.3f}); the rest "
          f"{cnn[~cnn.at_clamp].ratio.min():.2f}-{cnn[~cnn.at_clamp].ratio.max():.2f} (loss partly from unclamped "
          "errors); S2 mlp1 listed as INFO, not checked")
    C.put("clamp_ratio.n", len(t), "S0 + S2 (all seeds)")
    C.put("clamp_ratio.n_at_clamp", int(t.at_clamp.sum()), "S0 + S2 (all seeds)", n=len(t))


def t2_vs_a2():
    from stats_io import stage1_selection
    t2 = pd.read_csv(os.path.join(OUT, "tables", "T2_paired_hybrid.csv"))
    a2 = pd.read_csv(os.path.join(V6, "results", "stage2", "summary_seeds11-30", "paired.csv"))
    a2 = a2[a2.metric == "final_val"]
    sel = stage1_selection()

    def cfg_label(ds, arch):
        if arch == "mlp1":
            return "mlp1"
        c = sel[ds][arch]
        return f"{c['kernel_mode']}/{c['kernel_family']}/K{c['conv_k']}" if c["kernel_mode"] != "random" \
            else f"random/K{c['conv_k']}"

    t2 = t2[t2.comparison.isin(["tanh+relu - relu", "tanh+relu - tanh"])].copy()
    t2["config"] = [cfg_label(d, a) for d, a in zip(t2.dataset, t2.arch)]
    m = t2.merge(a2, on=["dataset", "config", "hidden", "lr", "comparison"], suffixes=("", "_a2"))
    width = (m.ci_hi_a2 - m.ci_lo_a2).clip(lower=1e-9)
    rel = np.maximum(np.abs(m.ci_lo - m.ci_lo_a2), np.abs(m.ci_hi - m.ci_hi_a2)) / width
    ok = len(m) == len(t2) and np.allclose(m.mean_diff, m.mean_diff_a2, atol=1e-3) and (m.wins == m.wins_a2).all() \
        and np.allclose(m.p_t, m.p_t_a2, rtol=1e-3, atol=1e-6) and rel.max() <= 0.1
    check("T2 agrees with A2 paired.csv (hybrid - relu / tanh)", ok,
          f"{len(m)}/{len(t2)} rows matched; means, wins and paired-t p identical (max |mean diff| "
          f"{np.abs(m.mean_diff - m.mean_diff_a2).max():.2g}); bootstrap CI ends differ by at most "
          f"{rel.max():.3f} of the CI width (different resampling streams, both 10,000 resamples)")


def main():
    anchors()
    clamp_code()
    clamp_ratio()
    t2_vs_a2()
    n_fail = RESULTS.count(False)
    LINES.append(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    text = "\n".join(LINES) + "\n"
    with open(out_path("checks", "checks.txt"), "w") as f:
        f.write(text)
    C.put("n_checks", len(RESULTS), "see checks.txt")
    C.put("n_failed", n_fail, "see checks.txt")
    C.save()
    print(text, end="")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
