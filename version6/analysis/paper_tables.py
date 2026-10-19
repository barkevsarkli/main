#!/usr/bin/env python3
"""A5: the tables the article uses (CSV + LaTeX booktabs) -> analysis_out/partA/tables/.

    python3 analysis/paper_tables.py

  T1  Stage 2 CNN + mlp1 at the stress and below-stress LR, seeds 11-30 (CONFIRMATORY).
  T2  paired tanh+relu - {relu, tanh, leaky_relu, sigmoid}, final val, same cells (CONFIRMATORY).
  T3  mask placement (@conv / @hidden / both), cfg2 H128/256, seeds 11-30 (CONFIRMATORY).
  T4  Stage 3 ratio x LR, seeds 31-45 (pre-registered slice, fresh seeds).
  T5  tuned accuracy (checkpoint test) at each model's selected LR, seeds 11-30 (CONFIRMATORY).
  T6  critical-LR ratios from A4 (exploratory).
  T7  version5 vs v6 Stage 0 replication, mlp2 MNIST (exploratory).
  T8  Stage 1 kernel screen and the robustness of the cfg1 / cfg2 selection (screen, seeds 101-103).

Paired statistics (partA_common.paired): mean difference, 95% paired bootstrap CI (10,000
resamples), wins / n, paired t-test p, Holm-adjusted within each family (dataset, metric,
comparison) of the table. One random generator (seed 20261018) is used per table, in a fixed
iteration order, so the output is byte-identical on re-run.

T2 recomputes the A2 comparisons (results/stage2/summary_seeds11-30/paired.csv) instead of
reading them, because A2 has no tanh+relu - leaky_relu / - sigmoid rows; its Holm family is
the T2 cells (stress and below-stress LR) only, so p_holm differs from A2's, which adjusts over
every LR of the extension. Means, CIs and wins agree with A2 where both exist (checked in A7).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from partA_common import BOOT_N, OUT, SEED, Claims, holm, paired, write_table  # noqa: E402

SUB = "tables"
MAIN_MODELS = ["relu", "tanh+relu", "tanh", "leaky_relu", "sigmoid"]
ARCHS = ["cfg1", "cfg2", "mlp1"]
CONF = "S2 ext seeds 11-30 (confirmatory)"
C = Claims("tables")


def gen():
    return np.random.default_rng(SEED)


def lr_role(ds, arch, H, lr):
    from stats_io import stage2_lrs
    s = stage2_lrs(ds, arch, H)
    roles = [k for k in ("stress", "below") if np.isclose(s[k], lr)]
    return roles[0] if roles else ""


def ext_runs():
    from stats_io import runs
    r = runs(2)
    return r[r.seed > 10].copy()


def cell_stats(g):
    return dict(n=len(g), final_mean=g.final_val.mean(), final_sd=g.final_val.std(ddof=1),
                n_final_lt15=int((g.final_val < 15).sum()), best_mean=g.best_val.mean(),
                test_best_mean=g.test_acc_best.mean(),
                n_late_collapse=int((g.outcome == "late_collapse").sum()),
                n_never_learned=int((g.outcome == "never_learned").sum()))


def paired_rows(g, comps, metric, rng, keys):
    rows = []
    for a, b in comps:
        ga, gb = g[g.model == a].set_index("seed"), g[g.model == b].set_index("seed")
        seeds = sorted(set(ga.index) & set(gb.index))
        if len(seeds) < 2:
            continue
        r = paired(ga.loc[seeds, metric].to_numpy(), gb.loc[seeds, metric].to_numpy(), rng)
        rows.append(keys | dict(metric=metric, comparison=f"{a} - {b}") | r)
    return rows


def add_holm(t, family=("dataset", "metric", "comparison")):
    t = t.copy()
    t["p_holm"] = np.nan
    for _, idx in t.groupby(list(family)).groups.items():
        t.loc[idx, "p_holm"] = holm(t.loc[idx, "p_t"].to_numpy())
    return t


def stress_cells(e):
    """(dataset, arch, H, lr, role) of the stress and below-stress LRs of every Stage 2 cell."""
    from stats_io import stage2_lrs
    out = []
    for ds in ("mnist", "cifar10"):
        for arch in ARCHS:
            for H in (64, 128, 256):
                s = stage2_lrs(ds, arch, H)
                for role in ("below", "stress"):
                    out.append((ds, arch, H, s[role], role))
    return out


# ----------------------------------------------------------------------------- T1, T2

def t1(e):
    rows = []
    for ds, arch, H, lr, role in stress_cells(e):
        g = e[(e.dataset == ds) & (e.arch_label == arch) & (e.hidden == H) & np.isclose(e.lr, lr)]
        for m in MAIN_MODELS:
            gm = g[g.model == m]
            if len(gm):
                rows.append(dict(dataset=ds, arch=arch, hidden=H, lr=lr, lr_role=role, model=m) | cell_stats(gm))
    t = pd.DataFrame(rows)
    write_table(t, SUB, "T1_stage2_stress",
                "T1 (CONFIRMATORY, Stage 2 extension seeds 11-30). CNN cfg1 / cfg2 and mlp1 at the pre-registered "
                "below-stress and stress LR: per model the mean final validation accuracy and its SD, the number of "
                "runs with final < 15% out of n, mean best validation accuracy, mean checkpoint test accuracy, and the "
                "outcome classes late_collapse / never_learned (outcomes.py).", digits=2)
    for _, r in t.iterrows():
        tag = f"T1.{r.dataset}.{r.arch}.h{r.hidden}.{r.lr_role}.{r.model}"
        C.put(f"{tag}.final_mean", r.final_mean, CONF, n=r.n, unit="%")
        C.put(f"{tag}.n_final_lt15", r.n_final_lt15, CONF, n=r.n)
    return t


def t2(e):
    rng = gen()
    rows = []
    comps = [("tanh+relu", m) for m in ("relu", "tanh", "leaky_relu", "sigmoid")]
    for ds, arch, H, lr, role in stress_cells(e):
        g = e[(e.dataset == ds) & (e.arch_label == arch) & (e.hidden == H) & np.isclose(e.lr, lr)]
        rows += paired_rows(g, comps, "final_val", rng, dict(dataset=ds, arch=arch, hidden=H, lr=lr, lr_role=role))
    t = add_holm(pd.DataFrame(rows))
    write_table(t, SUB, "T2_paired_hybrid",
                "T2 (CONFIRMATORY, seeds 11-30). Paired differences in final validation accuracy (pp), tanh+relu minus "
                "each single activation, at the below-stress and stress LR: mean, 95% paired bootstrap CI (10,000 "
                "resamples), wins (hybrid higher) and losses out of n, paired t-test p, Holm-adjusted within (dataset, "
                "comparison) over the 18 cells of this table.", digits=3)
    for _, r in t.iterrows():
        tag = f"T2.{r.dataset}.{r.arch}.h{r.hidden}.{r.lr_role}.{r.comparison.split(' - ')[1]}"
        C.put(f"{tag}.mean_diff", r.mean_diff, CONF, n=r.n, unit="pp")
        C.put(f"{tag}.ci", f"[{r.ci_lo:.2f}, {r.ci_hi:.2f}]", CONF, n=r.n)
        C.put(f"{tag}.wins", f"{r.wins}/{r.n}", CONF, n=r.n)
        C.put(f"{tag}.p_holm", r.p_holm, CONF, n=r.n)
    return t


# ----------------------------------------------------------------------------- T3

def t3(e):
    rng = gen()
    models = ["relu", "tanh", "tanh+relu", "tanh+relu@conv", "tanh+relu@hidden"]
    comps = [("tanh+relu@conv", "relu"), ("tanh+relu@hidden", "relu"), ("tanh+relu@conv", "tanh+relu@hidden"),
             ("tanh+relu", "tanh+relu@conv"), ("tanh+relu", "tanh+relu@hidden")]
    rows, prow = [], []
    for ds, arch, H, lr, role in stress_cells(e):
        g = e[(e.dataset == ds) & (e.arch_label == arch) & (e.hidden == H) & np.isclose(e.lr, lr)]
        if not (g.model == "tanh+relu@conv").any():
            continue
        for m in models:
            gm = g[g.model == m]
            rows.append(dict(dataset=ds, hidden=H, lr=lr, lr_role=role, model=m) | cell_stats(gm))
        prow += paired_rows(g, comps, "final_val", rng, dict(dataset=ds, hidden=H, lr=lr, lr_role=role))
    t = pd.DataFrame(rows)
    p = add_holm(pd.DataFrame(prow))
    write_table(t, SUB, "T3_mask_placement",
                "T3 (CONFIRMATORY, seeds 11-30). Mask placement in cfg2: the 50/50 tanh+relu mask in the conv layer "
                "only (@conv, hidden layer all ReLU), in the hidden layer only (@hidden, conv all ReLU), or in both "
                "(tanh+relu), with relu and tanh, at the below-stress and stress LR. Columns as T1.", digits=2)
    write_table(p, SUB, "T3_mask_placement_paired",
                "T3b (CONFIRMATORY, seeds 11-30). Paired differences in final validation accuracy (pp) between mask "
                "placements; 95% paired bootstrap CI, wins/losses, paired t p, Holm within (dataset, comparison).",
                digits=3)
    for _, r in t.iterrows():
        tag = f"T3.{r.dataset}.h{r.hidden}.{r.lr_role}.{r.model}"
        C.put(f"{tag}.n_final_lt15", r.n_final_lt15, CONF, n=r.n)
        C.put(f"{tag}.final_mean", r.final_mean, CONF, n=r.n, unit="%")
    for _, r in p.iterrows():
        tag = f"T3.{r.dataset}.h{r.hidden}.{r.lr_role}.{r.comparison.replace(' - ', '_minus_')}"
        C.put(f"{tag}.mean_diff", r.mean_diff, CONF, n=r.n, unit="pp")
        C.put(f"{tag}.ci", f"[{r.ci_lo:.2f}, {r.ci_hi:.2f}]", CONF, n=r.n)
        C.put(f"{tag}.p_holm", r.p_holm, CONF, n=r.n)
    return t, p


# ----------------------------------------------------------------------------- T4

def t4():
    from stats_io import runs
    s3 = runs(3)
    rng = gen()
    label = {"tanh+relu@0.25": "rho=0.25", "tanh+relu": "rho=0.5", "tanh+relu@0.75": "rho=0.75",
             "relu": "relu (rho=1)", "tanh": "tanh (rho=0)"}
    order = ["tanh", "tanh+relu@0.25", "tanh+relu", "tanh+relu@0.75", "relu"]
    rows, prow = [], []
    comps = [("tanh+relu", "relu"), ("tanh+relu", "tanh"), ("tanh+relu@0.25", "relu"), ("tanh+relu@0.75", "relu"),
             ("tanh+relu@0.25", "tanh"), ("tanh+relu@0.75", "tanh")]
    for (ds, lr), g in s3.groupby(["dataset", "lr"], sort=True):
        from stats_io import stage2_lrs
        s = stage2_lrs(ds, "cfg2", 128)
        role = "stress" if np.isclose(lr, s["stress"]) else ("below" if np.isclose(lr, s["below"]) else "above")
        for m in order:
            rows.append(dict(dataset=ds, lr=lr, lr_role=role, model=label[m]) | cell_stats(g[g.model == m]))
        for metric in ("final_val", "test_acc_best"):
            prow += paired_rows(g, comps, metric, rng, dict(dataset=ds, lr=lr, lr_role=role))
    t = pd.DataFrame(rows)
    p = add_holm(pd.DataFrame(prow))
    p["comparison"] = p.comparison.replace({f"{a} - {b}": f"{label[a]} - {label[b]}" for a, b in comps})
    ss = "S3 seeds 31-45 (pre-registered slice, fresh seeds)"
    write_table(t, SUB, "T4_stage3_ratio",
                "T4 (Stage 3, seeds 31-45: pre-registered slice with fresh seeds; LRs from Stage 2 seeds 1-10). "
                "cfg2 H128, share of ReLU rho = 0 (tanh), 0.25, 0.5, 0.75, 1 (relu) at the below-stress, stress and "
                "above-stress LR. Columns as T1; test_best_mean = test accuracy at the best-validation checkpoint.",
                digits=2)
    write_table(p, SUB, "T4_stage3_paired",
                "T4b (Stage 3, seeds 31-45). Paired differences (pp) in final validation accuracy and checkpoint test "
                "accuracy; 95% paired bootstrap CI, wins/losses, paired t p, Holm within (dataset, metric, comparison) "
                "over the three LRs.", digits=3)
    for _, r in t.iterrows():
        tag = f"T4.{r.dataset}.{r.lr_role}.{r.model.split(' ')[0]}"
        C.put(f"{tag}.final_mean", r.final_mean, ss, n=r.n, unit="%")
        C.put(f"{tag}.n_final_lt15", r.n_final_lt15, ss, n=r.n)
        C.put(f"{tag}.test_best_mean", r.test_best_mean, ss, n=r.n, unit="%")
    for _, r in p.iterrows():
        a, b = (x.split(" ")[0] for x in r.comparison.split(" - "))
        tag = f"T4.{r.dataset}.{r.lr_role}.{r.metric}.{a}_minus_{b}"
        C.put(f"{tag}.mean_diff", r.mean_diff, ss, n=r.n, unit="pp")
        C.put(f"{tag}.ci", f"[{r.ci_lo:.2f}, {r.ci_hi:.2f}]", ss, n=r.n)
        C.put(f"{tag}.wins", f"{r.wins}/{r.n}", ss, n=r.n)
        C.put(f"{tag}.p_holm", r.p_holm, ss, n=r.n)
    return t, p


# ----------------------------------------------------------------------------- T5

def t5(e):
    from stats_io import runs, stage2_lrs
    sweep = runs(2)
    sweep = sweep[sweep.seed <= 10]
    rng = gen()
    rows, prow = [], []
    for ds in ("mnist", "cifar10"):
        for arch in ARCHS:
            for H in (64, 128, 256):
                s = stage2_lrs(ds, arch, H)
                grid = np.sort(sweep[(sweep.dataset == ds) & (sweep.arch_label == arch) & (sweep.hidden == H)].lr.unique())
                g = e[(e.dataset == ds) & (e.arch_label == arch) & (e.hidden == H)]
                sel = []
                for m in MAIN_MODELS:
                    lr = s["selected"][m]
                    gm = g[(g.model == m) & np.isclose(g.lr, lr)]
                    edge = "low" if np.isclose(lr, grid.min()) else ("high" if np.isclose(lr, grid.max()) else "")
                    rows.append(dict(dataset=ds, arch=arch, hidden=H, model=m, selected_lr=lr, grid_edge=edge,
                                     n=len(gm), test_best_mean=gm.test_acc_best.mean(),
                                     test_best_sd=gm.test_acc_best.std(ddof=1), final_mean=gm.final_val.mean(),
                                     n_final_lt15=int((gm.final_val < 15).sum())))
                    sel.append(gm.assign(model=m))
                gs = pd.concat(sel)
                prow += paired_rows(gs, [("tanh+relu", "relu"), ("tanh+relu", "tanh")], "test_acc_best", rng,
                                    dict(dataset=ds, arch=arch, hidden=H))
    t = pd.DataFrame(rows)
    p = add_holm(pd.DataFrame(prow))
    write_table(t, SUB, "T5_tuned_accuracy",
                "T5 (CONFIRMATORY, seeds 11-30). Tuned accuracy: test accuracy at the best-validation checkpoint, each "
                "model at its own selected LR (chosen from seeds 1-10). grid_edge = 'low' / 'high' when the selected LR "
                "is the smallest / largest LR of the Stage 2 grid, so the tuned optimum may lie outside the grid "
                "(censored).", digits=2)
    write_table(p, SUB, "T5_tuned_paired",
                "T5b (CONFIRMATORY, seeds 11-30). Paired differences in checkpoint test accuracy (pp), each model at its "
                "own selected LR: tanh+relu - relu and tanh+relu - tanh; 95% paired bootstrap CI, wins/losses, paired "
                "t p, Holm within (dataset, comparison).", digits=3)
    for _, r in t.iterrows():
        tag = f"T5.{r.dataset}.{r.arch}.h{r.hidden}.{r.model}"
        C.put(f"{tag}.test_best_mean", r.test_best_mean, CONF, n=r.n, unit="%")
        C.put(f"{tag}.selected_lr", r.selected_lr, "S2 sweep seeds 1-10 (selection)")
        C.put(f"{tag}.grid_edge", r.grid_edge, "S2 sweep seeds 1-10 (selection)")
    for _, r in p.iterrows():
        tag = f"T5.{r.dataset}.{r.arch}.h{r.hidden}.{r.comparison.replace(' - ', '_minus_')}"
        C.put(f"{tag}.mean_diff", r.mean_diff, CONF, n=r.n, unit="pp")
        C.put(f"{tag}.ci", f"[{r.ci_lo:.2f}, {r.ci_hi:.2f}]", CONF, n=r.n)
        C.put(f"{tag}.wins", f"{r.wins}/{r.n}", CONF, n=r.n)
        C.put(f"{tag}.p_holm", r.p_holm, CONF, n=r.n)
    C.put("T5.n_grid_edge_low", int((t.grid_edge == "low").sum()), "S2 sweep seeds 1-10 (selection)", n=len(t))
    for ds in ("mnist", "cifar10"):
        C.put(f"T5.{ds}.n_grid_edge_low", int(((t.grid_edge == "low") & (t.dataset == ds)).sum()),
              "S2 sweep seeds 1-10 (selection)", n=int((t.dataset == ds).sum()))
    return t, p


# ----------------------------------------------------------------------------- T6

def t6():
    parts = []
    for oc in ("fail", "late_collapse", "degraded"):
        t = pd.read_csv(os.path.join(OUT, "critical_lr", f"lr50_ratios_{oc}.csv"))
        parts.append(t.assign(outcome=oc))
    t = pd.concat(parts, ignore_index=True)
    cols = ["outcome", "source", "dataset", "arch", "hidden", "lr50_relu", "lr50_tanh+relu", "lr50_tanh",
            "ratio_tanh+relu_relu", "ratio_tanh_relu", "fail_is_never_learned"]
    write_table(t[cols], SUB, "T6_critical_lr",
                "T6 (exploratory). Critical LR (LR50, the LR at which half the runs reach the outcome) per model and "
                "the ratios LR50(tanh+relu)/LR50(relu) and LR50(tanh)/LR50(relu), for fail (final < 15%), "
                "late_collapse (never_learned runs excluded) and degraded (final < 50% of the best grid mean). "
                "Fitted values with seed-bootstrap 95% CI; brackets 'in [a, b]' or bounds where the outcomes are "
                "separated; 'non-monotone' where the outcome becomes significantly less frequent at a higher LR (first "
                "crossing given). fail_is_never_learned: most pure-ReLU failures of the cell never learned. Full "
                "details: critical_lr/lr50_all.csv.", digits=3)
    return t


# ----------------------------------------------------------------------------- T7

def t7():
    from stats_io import runs
    from v5_io import v5_runs
    s0 = runs(0)
    v5 = v5_runs()
    v5 = v5[(v5.kind == "main") & (v5.dataset == "mnist")]
    rng = gen()
    rows = []
    for (H, m, lr), g6 in s0.groupby(["hidden", "model", "lr"], sort=True):
        g5 = v5[(v5.hidden == H) & (v5.model == m) & np.isclose(v5.lr, lr)]
        f6, f5 = g6.final_val.to_numpy(), g5.final_val.to_numpy()
        b6 = f6[rng.integers(0, len(f6), (BOOT_N, len(f6)))]
        b5 = f5[rng.integers(0, len(f5), (BOOT_N, len(f5)))]
        dm = b6.mean(1) - b5.mean(1)
        dp = (b6 < 15).mean(1) - (b5 < 15).mean(1)
        g5s = g5[g5.seed <= 10]
        rows.append(dict(hidden=H, model=m, lr=lr,
                         v5_n=len(f5), v5_final_mean=f5.mean(), v5_final_sd=f5.std(ddof=1), v5_final_median=np.median(f5),
                         v5_n_lt15=int((f5 < 15).sum()), v5_seeds1_10_n_lt15=int((g5s.final_val < 15).sum()),
                         v6_n=len(f6), v6_final_mean=f6.mean(), v6_final_sd=f6.std(ddof=1), v6_final_median=np.median(f6),
                         v6_n_lt15=int((f6 < 15).sum()),
                         diff_mean=f6.mean() - f5.mean(), diff_mean_ci_lo=np.percentile(dm, 2.5),
                         diff_mean_ci_hi=np.percentile(dm, 97.5),
                         diff_fail_share=(f6 < 15).mean() - (f5 < 15).mean(),
                         diff_fail_ci_lo=np.percentile(dp, 2.5), diff_fail_ci_hi=np.percentile(dp, 97.5)))
    t = pd.DataFrame(rows)
    write_table(t, SUB, "T7_v5_v6_replication",
                "T7 (exploratory). Replication of the version5 mlp2 MNIST sweeps (lrsweep_64 / lrsweep_128, seeds 1-30) "
                "by the version6 code (Stage 0, seeds 1-10; different implementation and random streams, so runs are "
                "not identical seed by seed): final validation accuracy mean, SD, median and the number of runs with "
                "final < 15% (version5 also for seeds 1-10 alone); v6 - v5 difference in the mean and in the share of "
                "final < 15% runs with unpaired bootstrap 95% CIs (10,000 resamples).", digits=2)
    ss = "S0 seeds 1-10 vs v5 seeds 1-30"
    for _, r in t.iterrows():
        tag = f"T7.h{r.hidden}.{r.model}.lr{r.lr:g}"
        C.put(f"{tag}.v5_n_lt15", f"{r.v5_n_lt15}/{r.v5_n}", ss)
        C.put(f"{tag}.v6_n_lt15", f"{r.v6_n_lt15}/{r.v6_n}", ss)
        C.put(f"{tag}.diff_fail_ci", f"[{r.diff_fail_ci_lo:.2f}, {r.diff_fail_ci_hi:.2f}]", ss)
    C.put("T7.n_cells_fail_share_ci_excludes_0", int(((t.diff_fail_ci_lo > 0) | (t.diff_fail_ci_hi < 0)).sum()), ss,
          n=len(t))
    return t


# ----------------------------------------------------------------------------- T8

FAM_ORDER = ["none", "gauss", "sharpen", "sobel", "laplacian", "gabor", "random_frozen"]
MODE_ORDER = ["random", "frozen", "bank_init"]


def select(s1, seeds):
    """The pre-registered Stage 1 rule (sweep/select_stage2.py) on a subset of the seeds.
    Returns (cfg1, cfg2, info DataFrame)."""
    d = s1[s1.seed.isin(seeds)]
    lrs = sorted(d.lr.unique())
    rows = []
    for (mode, K, fam), g in d.groupby(["kernel_mode", "conv_k", "kernel_family"], sort=False):
        rel = g[g.model == "relu"].groupby("lr")
        bv, fv = rel.best_val.mean(), rel.final_val.mean()
        best_lr = max(lrs, key=lambda lr: (round(bv[lr], 9), -lr))
        best = fv.max()
        stress = next((lr for lr in lrs if fv[lr] < 0.5 * best), lrs[-1])
        gs = g[np.isclose(g.lr, stress)]
        h = gs[gs.model == "tanh+relu"].set_index("seed").final_val
        r = gs[gs.model == "relu"].set_index("seed").final_val
        rows.append(dict(kernel_mode=mode, conv_k=int(K), kernel_family=fam, relu_best_lr=best_lr,
                         relu_best_val_mean=bv[best_lr], stress_lr=stress, diff_final_at_stress=(h - r).mean()))
    info = pd.DataFrame(rows)
    info["_o"] = [(k, FAM_ORDER.index(f), MODE_ORDER.index(m)) for m, k, f in
                  zip(info.kernel_mode, info.conv_k, info.kernel_family)]
    info = info.sort_values("_o").drop(columns="_o").reset_index(drop=True)
    rnd = info[info.kernel_mode == "random"]
    i1 = max(rnd.index, key=lambda i: (round(rnd.relu_best_val_mean[i], 9), -rnd.conv_k[i]))
    ref = info.relu_best_val_mean[i1]
    info["eligible"] = (info.kernel_mode != "random") & (info.relu_best_val_mean >= ref - 2.0)
    el = info[info.eligible]
    i2 = max(el.index, key=lambda i: (round(el.diff_final_at_stress[i], 9), -el.conv_k[i],
                                      -FAM_ORDER.index(el.kernel_family[i]), -MODE_ORDER.index(el.kernel_mode[i])))
    name = lambda i: f"{info.kernel_mode[i]}/{info.kernel_family[i]}/K{info.conv_k[i]}"  # noqa: E731
    return name(i1), name(i2), info, i1, i2


def t8():
    from stats_io import runs, stage1_selection
    s1 = runs(1)
    s1 = s1[(s1.arch == "cnn") & (s1.hidden == 128) & (s1.model.isin(["relu", "tanh+relu", "tanh"]))]
    seeds = sorted(s1.seed.unique())
    sel = stage1_selection()
    screen, robust = [], []
    ss = "S1 seeds 101-103 (screen)"
    for ds in ("mnist", "cifar10"):
        d = s1[s1.dataset == ds]
        c1, c2, info, i1, i2 = select(d, seeds)
        exp = {k: "{kernel_mode}/{kernel_family}/K{conv_k}".format(**sel[ds][k]) for k in ("cfg1", "cfg2")}
        assert (c1, c2) == (exp["cfg1"], exp["cfg2"]), (ds, c1, c2, exp)
        rnd = info[info.kernel_mode == "random"].sort_values("relu_best_val_mean", ascending=False)
        el = info[info.eligible].sort_values("diff_final_at_stress", ascending=False)
        m1 = rnd.relu_best_val_mean.iloc[0] - rnd.relu_best_val_mean.iloc[1]
        m2 = el.diff_final_at_stress.iloc[0] - el.diff_final_at_stress.iloc[1]
        ru1 = f"random/none/K{rnd.conv_k.iloc[1]}"
        ru2 = f"{el.kernel_mode.iloc[1]}/{el.kernel_family.iloc[1]}/K{el.conv_k.iloc[1]}"
        screen.append(info.assign(dataset=ds, cfg=["cfg1" if i == i1 else ("cfg2" if i == i2 else "")
                                                   for i in info.index]))
        row = dict(dataset=ds, seeds="101-103", cfg1=c1, cfg2=c2, cfg1_runner_up=ru1, cfg1_margin_pp=m1,
                   cfg2_runner_up=ru2, cfg2_margin_pp=m2)
        robust.append(row)
        C.put(f"T8.{ds}.cfg1_margin_pp", m1, ss, desc=f"ReLU best-val margin of {c1} over {ru1}", unit="pp")
        C.put(f"T8.{ds}.cfg2_margin_pp", m2, ss, desc=f"diff@stress margin of {c2} over {ru2}", unit="pp")
        n_same1 = n_same2 = 0
        for drop in seeds:
            keep = [s for s in seeds if s != drop]
            a1, a2, inf2, j1, j2 = select(d, keep)
            n_same1 += a1 == c1
            n_same2 += a2 == c2
            r2 = inf2[inf2.eligible].sort_values("diff_final_at_stress", ascending=False)
            robust.append(dict(dataset=ds, seeds=f"without {drop}", cfg1=a1, cfg2=a2,
                               cfg1_margin_pp=np.nan, cfg2_margin_pp=r2.diff_final_at_stress.iloc[0]
                               - r2.diff_final_at_stress.iloc[1]))
            C.put(f"T8.{ds}.without_{drop}.cfg1", a1, f"S1 seeds 101-103 without {drop}")
            C.put(f"T8.{ds}.without_{drop}.cfg2", a2, f"S1 seeds 101-103 without {drop}")
        C.put(f"T8.{ds}.loso_cfg1_same", f"{n_same1}/{len(seeds)}", ss)
        C.put(f"T8.{ds}.loso_cfg2_same", f"{n_same2}/{len(seeds)}", ss)
    sc = pd.concat(screen, ignore_index=True)
    sc = sc[["dataset", "kernel_mode", "kernel_family", "conv_k", "relu_best_val_mean", "relu_best_lr", "stress_lr",
             "diff_final_at_stress", "eligible", "cfg"]]
    write_table(sc, SUB, "T8_stage1_screen",
                "T8 (Stage 1 screen, seeds 101-103, H = 128, validation only). Per kernel configuration: ReLU mean best "
                "validation accuracy at its best grid LR, that LR, the stress LR (first LR where ReLU's mean final "
                "accuracy < 50% of its best) and the mean paired tanh+relu - relu difference in final validation "
                "accuracy there; eligible = bank configuration within 2 pp of cfg1. cfg marks the pre-registered "
                "selection.", digits=2)
    rb = pd.DataFrame(robust)
    write_table(rb, SUB, "T8_selection_robustness",
                "T8b (Stage 1, seeds 101-103). Robustness of the pre-registered selection: the choice with all three "
                "seeds (reproduces results/stage1/selection.json) and with each seed left out; the margin of the chosen "
                "configuration over the runner-up (cfg1: ReLU mean best validation accuracy; cfg2: mean tanh+relu - relu "
                "final-accuracy difference at the stress LR).", digits=2)
    return sc, rb


def main():
    e = ext_runs()
    t1(e)
    t2(e)
    t3(e)
    t4()
    t5(e)
    t6()
    t7()
    t8()
    C.save()
    print("paper_tables: T1-T8 written to", os.path.join(OUT, SUB))


if __name__ == "__main__":
    main()
