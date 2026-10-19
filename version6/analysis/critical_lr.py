#!/usr/bin/env python3
"""A4: the effect of mixing as a shift of the stability threshold (critical LR).

    python3 analysis/critical_lr.py   -> analysis_out/partA/critical_lr/, figures/crit_*

Per cell (source, dataset, arch, H, model) and outcome, LR50 = the LR at which P(outcome) = 0.5,
from a logistic regression of the per-run outcome on log10(LR):

  fail           final validation accuracy < 15%;
  late_collapse  outcome class late_collapse (outcomes.py), never_learned runs left out of
                 the fit (they cannot collapse late);
  degraded       final validation accuracy < 50% of the model's best grid mean final accuracy
                 (highest mean over the cell's LRs).

Separation is handled before any fit, on the per-LR counts:
  above_grid   P(outcome) never reaches 0.5 inside the grid (no event, or the fit puts LR50
               above the largest LR): reported as "> largest LR";
  below_grid   the outcome already has P >= 0.5 at the smallest LR: "< smallest LR";
  bracket      complete or quasi-complete separation (every LR is all-0 or all-1 except at
               most one): reported as [largest LR with P = 0, smallest LR with P = 1];
  non_monotone the outcome is significantly LESS frequent at some higher LR than at a lower one
               (one-sided Fisher exact test, p < 0.05, any LR pair), or the fitted slope is <= 0:
               a single LR50 does not describe the cell; the first-crossing interval
               [LR before the first LR with P >= 0.5, that LR] is reported with the reversal;
  fit          otherwise: the maximum-likelihood LR50 with a seed-bootstrap 95% CI.
A fitted number is never reported for the first four.

Seed bootstrap: 2,000 resamples of the cell's seeds with replacement (a seed carries all of its
runs, at every LR and for every model of the cell, so model ratios are paired), refit each
resample. A separated resample has a bracket or bound instead of a number; it enters the
percentile CI at the end of its bracket that widens the interval (so the CI is conservative and can
be open on one side), and the share of separated resamples is reported. Random numbers: seed 20261018 combined with a
CRC32 of the cell name, so each cell's draws do not depend on the other cells.

Sources: version5 lrsweep_64 / lrsweep_128 (mlp2 MNIST) and c10_lrsweep_128 (mlp2 CIFAR-10),
30 seeds; v6 Stage 0 (mlp2 MNIST, seeds 1-10); v6 Stage 2 (CNN cfg1/cfg2 and mlp1, seeds 1-30
at the selected / stress / below-stress LR and 1-10 elsewhere). The version5 width calibration
files (calib_*, c10_calib_*; 6 seeds) are analysed separately (never pooled with the sweeps).
"""
import os
import sys
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

N_BOOT = 2000
NONMONO_ALPHA = 0.05
OUTCOMES = ("fail", "late_collapse", "degraded")
HYBRID = "tanh+relu"


# ----------------------------------------------------------------------------- fitting

def _irls(x, k, n, iters=60):
    """Vectorised binomial logistic regression of k/n on x.

    x: (L,), k and n: (R, L). Returns (a, b) arrays of shape (R,) and a convergence flag."""
    R = k.shape[0]
    xc = x - x.mean()
    beta = np.zeros((R, 2))
    ok = np.ones(R, bool)
    for _ in range(iters):
        eta = beta[:, :1] + beta[:, 1:] * xc
        p = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
        w = n * p * (1 - p)
        r = k - n * p
        g = np.stack([r.sum(1), (r * xc).sum(1)], 1)
        h00, h01, h11 = w.sum(1), (w * xc).sum(1), (w * xc * xc).sum(1)
        det = h00 * h11 - h01 ** 2
        bad = det <= 1e-12
        det = np.where(bad, 1.0, det)
        step = np.stack([(h11 * g[:, 0] - h01 * g[:, 1]) / det, (-h01 * g[:, 0] + h00 * g[:, 1]) / det], 1)
        step[bad] = 0
        ok &= ~bad
        beta += np.clip(step, -5, 5)
        if np.abs(step).max() < 1e-10:
            break
    a = beta[:, 0] - beta[:, 1] * x.mean()
    return a, beta[:, 1], ok


def classify_counts(x, k, n):
    """Separation status of per-LR counts (x sorted ascending), vectorised over rows.

    Returns status codes: 0 = estimable, 1 = no event at any LR, 2 = event at every run,
    3 = (quasi-)complete separation with events at the high-LR end, 4 = separation with
    events at the low-LR end."""
    present = n > 0
    has_ev = (k > 0) & present
    has_non = ((n - k) > 0) & present
    big = np.inf
    xf_min = np.where(has_ev, x, big).min(1)
    xf_max = np.where(has_ev, x, -big).max(1)
    xn_min = np.where(has_non, x, big).min(1)
    xn_max = np.where(has_non, x, -big).max(1)
    st = np.zeros(k.shape[0], int)
    none_ = ~has_ev.any(1)
    all_ = ~has_non.any(1)
    st[none_] = 1
    st[all_ & ~none_] = 2
    rest = ~none_ & ~all_
    st[rest & (xn_max <= xf_min)] = 3
    st[rest & (xf_max <= xn_min) & ~(xn_max <= xf_min)] = 4
    return st


def reversal(lrs, k, n, alpha=NONMONO_ALPHA):
    """First LR pair (i < j) where the outcome is significantly LESS frequent at the higher LR
    (one-sided Fisher exact test, p < alpha), or None. Used to flag non-monotone cells."""
    from scipy.stats import fisher_exact
    best = None
    for i in range(len(lrs)):
        for j in range(i + 1, len(lrs)):
            if k[i] / n[i] <= k[j] / n[j]:
                continue
            p = fisher_exact([[k[i], n[i] - k[i]], [k[j], n[j] - k[j]]], alternative="greater")[1]
            if p < alpha and (best is None or p < best[2]):
                best = (lrs[i], lrs[j], p)
    return best


def first_crossing(lrs, p):
    """[LR before the first LR with P >= 0.5, first LR with P >= 0.5] (0 / inf where open)."""
    f = np.flatnonzero(p >= 0.5)
    if not len(f):
        return lrs.max(), np.inf
    return (lrs[f[0] - 1] if f[0] > 0 else 0.0), lrs[f[0]]


def fit_counts(lrs, k, n):
    """LR50 of one set of per-LR counts. Returns dict(status, lr50, lo, hi) where lo/hi are the
    bracket or bound (0 / inf where open) and lr50 is NaN unless status == 'fit'. For a fit,
    env = (largest LR below LR50 with P = 0, smallest LR above it with P = 1) is the fallback
    interval when the bootstrap CI is not available. A non-monotone cell gets the first-crossing
    interval and the reversal (lr_i, lr_j, Fisher p)."""
    lrs = np.asarray(lrs, float)
    k, n = np.asarray(k, float), np.asarray(n, float)
    order = np.argsort(lrs)
    lrs, k, n = lrs[order], k[order], n[order]
    keep = n > 0
    lrs, k, n = lrs[keep], k[keep], n[keep]
    x = np.log10(lrs)
    p = k / n
    st = int(classify_counts(x, k[None], n[None])[0])
    if st == 1:
        return dict(status="above_grid", lr50=np.nan, lo=lrs.max(), hi=np.inf)
    if st == 2:
        return dict(status="below_grid", lr50=np.nan, lo=0.0, hi=lrs.min())
    rev = reversal(lrs, k, n)
    if rev is not None or st == 4:
        lo, hi = first_crossing(lrs, p)
        return dict(status="non_monotone", lr50=np.nan, lo=lo, hi=hi, rev=rev)
    if st == 3:
        lo = lrs[p == 0].max() if (p == 0).any() else 0.0
        hi = lrs[p == 1].min() if (p == 1).any() else np.inf
        return dict(status="bracket", lr50=np.nan, lo=lo, hi=hi)
    a, b, ok = _irls(x, k[None], n[None])
    if not ok[0] or b[0] <= 0:
        lo, hi = first_crossing(lrs, p)
        return dict(status="non_monotone", lr50=np.nan, lo=lo, hi=hi, rev=None)
    l50 = 10 ** (-a[0] / b[0])
    if l50 > lrs.max():
        return dict(status="above_grid", lr50=np.nan, lo=lrs.max(), hi=np.inf)
    if l50 < lrs.min():
        return dict(status="below_grid", lr50=np.nan, lo=0.0, hi=lrs.min())
    z, o = (lrs < l50) & (p == 0), (lrs > l50) & (p == 1)
    env = (lrs[z].max() if z.any() else 0.0, lrs[o].min() if o.any() else np.inf)
    return dict(status="fit", lr50=l50, lo=np.nan, hi=np.nan, a=a[0], b=b[0], env=env)


def boot_lr50(lrs, F, W):
    """log10 LR50 range of each bootstrap resample, shape (R, 2) = (low, high).

    F: (S, L) outcome per seed and LR (NaN = no run), W: (R, S) multiplicity of each seed in each
    resample. A resample that can be fitted gives low = high = its log10 LR50. A separated resample
    gives its own bracket or bound (open ends = -inf / +inf), and a resample with no positive slope
    gives (-inf, +inf). Taking the 2.5th percentile of `low` and the 97.5th of `high` therefore puts
    every separated resample at the end of its bracket that widens the interval (conservative)."""
    lrs = np.asarray(lrs, float)
    x = np.log10(lrs)
    pres = ~np.isnan(F)
    k = W @ np.where(pres, F, 0.0)
    n = W @ pres.astype(float)
    st = classify_counts(x, k, n)
    a, b, ok = _irls(x, k, n)
    good = (st == 0) & ok & (b > 0)
    l = np.where(good, -a / np.where(b > 0, b, 1), np.nan)
    lo = np.full(len(st), -np.inf)
    hi = np.full(len(st), np.inf)
    fit = good & (l >= x.min()) & (l <= x.max())
    lo[fit], hi[fit] = l[fit], l[fit]
    above = (st == 1) | (good & (l > x.max()))
    lo[above] = x.max()
    below = (st == 2) | (good & (l < x.min()))
    hi[below] = x.min()
    m3 = st == 3
    if m3.any():
        with np.errstate(invalid="ignore", divide="ignore"):
            p = k[m3] / n[m3]
        has = n[m3] > 0
        lo[m3] = np.where(has & (p == 0), x, -np.inf).max(1)
        hi[m3] = np.where(has & (p == 1), x, np.inf).min(1)
    return np.stack([lo, hi], 1)


def boot_ci(lb):
    """Conservative percentile 95% CI (log10) from boot_lr50 output; open ends stay infinite."""
    return (np.quantile(lb[:, 0], 0.025, method="inverted_cdf"),
            np.quantile(lb[:, 1], 0.975, method="inverted_cdf"))


def sep_share(lb):
    return float((lb[:, 0] != lb[:, 1]).mean())


def seed_matrix(df, outcome):
    """(seeds, lrs, F) for one model of one cell: F[s, l] = outcome of seed s at LR l (NaN if absent)."""
    seeds = np.sort(df.seed.unique())
    lrs = np.sort(df.lr.unique())
    F = np.full((len(seeds), len(lrs)), np.nan)
    si = {s: i for i, s in enumerate(seeds)}
    li = {v: i for i, v in enumerate(lrs)}
    for s, v, y in zip(df.seed, df.lr, df[outcome]):
        if y == y:
            F[si[s], li[v]] = float(y)
    return seeds, lrs, F


def cell_rng(*key):
    from partA_common import SEED
    return np.random.default_rng([SEED, zlib.crc32("|".join(map(str, key)).encode())])


# ----------------------------------------------------------------------------- data

def load_runs():
    """All runs of the sources, with outcome columns fail / late_collapse / degraded."""
    from stats_io import runs
    from v5_io import v5_runs
    parts = []
    v5 = v5_runs()
    parts.append(pd.DataFrame({"source": v5.source, "kind": v5.kind, "dataset": v5.dataset, "arch": "mlp2",
                               "hidden": v5.hidden, "model": v5.model, "lr": v5.lr, "seed": v5.seed,
                               "seed_set": v5.seed_set, "final_val": v5.final_val, "outcome": v5.outcome}))
    for st in (0, 2):
        r = runs(st)
        parts.append(pd.DataFrame({"source": f"v6 stage{st}", "kind": "main", "dataset": r.dataset,
                                   "arch": r.arch_label, "hidden": r.hidden, "model": r.model, "lr": r.lr,
                                   "seed": r.seed, "seed_set": r.seed_set, "final_val": r.final_val,
                                   "outcome": r.outcome}))
    d = pd.concat(parts, ignore_index=True)
    d["fail"] = (d.final_val < 15).astype(float)
    d["late_collapse"] = np.where(d.outcome == "never_learned", np.nan, (d.outcome == "late_collapse").astype(float))
    cell = ["source", "dataset", "arch", "hidden", "model"]
    best = d.groupby(cell + ["lr"]).final_val.mean().groupby(cell).max().rename("best_grid_final")
    d = d.join(best, on=cell)
    d["degraded"] = (d.final_val < 0.5 * d.best_grid_final).astype(float)
    return d


CELL = ["source", "kind", "dataset", "arch", "hidden"]


def seed_label(g):
    ss = sorted(g.seed_set.unique())
    return "; ".join(ss)


def estimate_all(d, n_boot=N_BOOT):
    """One row per (cell, model, outcome), plus the bootstrap log10 LR50 draws for the ratios."""
    rows, draws = [], {}
    for key, g in d.groupby(CELL, sort=True):
        seeds_all = np.sort(g.seed.unique())
        gen = cell_rng(*key)
        idx = gen.integers(0, len(seeds_all), size=(n_boot, len(seeds_all)))
        W = np.zeros((n_boot, len(seeds_all)))
        np.add.at(W, (np.repeat(np.arange(n_boot), len(seeds_all)), idx.ravel()), 1.0)
        for model, gm in g.groupby("model", sort=True):
            for oc in OUTCOMES:
                seeds, lrs, F = seed_matrix(gm, oc)
                pos = np.searchsorted(seeds_all, seeds)
                Wm = W[:, pos]
                k = np.nansum(F, 0)
                n = (~np.isnan(F)).sum(0)
                res = fit_counts(lrs, k, n)
                lb = boot_lr50(lrs, F, Wm)
                sep = sep_share(lb)
                ci_lo = ci_hi = np.nan
                if res["status"] == "fit":
                    ci_lo, ci_hi = 10.0 ** np.array(boot_ci(lb))
                draws[key + (model, oc)] = (res, lb)
                nl = gm[(gm.fail == 1)]
                rows.append(dict(zip(CELL, key)) | dict(
                    model=model, outcome=oc, seed_set=seed_label(gm), n_runs=int(n.sum()), n_seeds=len(seeds),
                    lrs=" ".join(f"{v:g}" for v in lrs), n_by_lr=" ".join(str(int(v)) for v in n),
                    events_by_lr=" ".join(str(int(v)) for v in k), status=res["status"], lr50=res["lr50"],
                    ci_lo=ci_lo, ci_hi=ci_hi, bound_lo=res["lo"], bound_hi=res["hi"], boot_sep_share=sep,
                    share_fail_never_learned=float((nl.outcome == "never_learned").mean()) if len(nl) else np.nan,
                    display=display(res, ci_lo, ci_hi)))
    return pd.DataFrame(rows), draws


def fmt(v):
    return f"{v:.3g}"


def display(res, lo=np.nan, hi=np.nan):
    s = res["status"]
    if s == "fit":
        lo_s = "0" if lo == 0 else fmt(lo)
        hi_s = "inf" if np.isinf(hi) else fmt(hi)
        return f"{fmt(res['lr50'])} [{lo_s}, {hi_s}]"
    if s == "above_grid":
        return f"> {fmt(res['lo'])}"
    if s == "below_grid":
        return f"< {fmt(res['hi'])}"
    if s == "bracket":
        hi = "inf" if np.isinf(res["hi"]) else fmt(res["hi"])
        return f"in [{fmt(res['lo'])}, {hi}]"
    hi = "inf" if np.isinf(res["hi"]) else fmt(res["hi"])
    rev = res.get("rev")
    tail = f"; falls from {fmt(rev[0])} to {fmt(rev[1])}" if rev else ""
    return f"non-monotone: first crossing in [{fmt(res['lo'])}, {hi}]{tail}"


def interval(res, lb):
    """(lo, hi) range of LR50 implied by a result: the conservative bootstrap CI for a fit, the
    bracket, bound or first-crossing interval otherwise."""
    if res["status"] == "fit":
        return tuple(10.0 ** np.array(boot_ci(lb)))
    return (res["lo"], res["hi"])


def _range_text(lo, hi):
    if np.isinf(hi) and lo == 0:
        return "unbounded"
    if np.isinf(hi):
        return f"> {fmt(lo)}"
    if lo == 0:
        return f"< {fmt(hi)}"
    return f"in [{fmt(lo)}, {fmt(hi)}]"


def ratio(num, den):
    """LR50(num) / LR50(den). Both fits: point and paired seed-bootstrap 95% CI (same resamples for
    both models; a separated resample enters with the bracket end that widens the interval).
    Otherwise the interval implied by the two CIs / brackets / bounds. Returns dict(status, point,
    lo, hi, display)."""
    (rn, ln), (rd, ld) = num, den
    if rn["status"] == "fit" and rd["status"] == "fit":
        point = rn["lr50"] / rd["lr50"]
        lo = np.quantile(ln[:, 0] - ld[:, 1], 0.025, method="inverted_cdf")
        hi = np.quantile(ln[:, 1] - ld[:, 0], 0.975, method="inverted_cdf")
        lo, hi = 10.0 ** lo, 10.0 ** hi
        hi_s = "inf" if np.isinf(hi) else fmt(hi)
        lo_s = "0" if lo == 0 else fmt(lo)
        return dict(status="fit", point=point, lo=lo, hi=hi, display=f"{fmt(point)} [{lo_s}, {hi_s}]")
    a, b = interval(rn, ln)
    c, d_ = interval(rd, ld)
    lo = a / d_ if 0 < d_ < np.inf else 0.0
    hi = b / c if c > 0 else np.inf
    return dict(status="interval", point=np.nan, lo=lo, hi=hi, display=_range_text(lo, hi))


def ratio_table(est, draws, kind):
    rows = []
    for key, g in est[est.kind == kind].groupby(CELL + ["outcome"], sort=True):
        ck, oc = tuple(key[:-1]), key[-1]
        models = set(g.model)
        if "relu" not in models:
            continue
        d = dict(zip(CELL, ck)) | dict(outcome=oc, seed_set=g[g.model == "relu"].seed_set.iloc[0])
        for m in ("relu", HYBRID, "tanh", "leaky_relu", "sigmoid"):
            d[f"lr50_{m}"] = g[g.model == m].display.iloc[0] if m in models else ""
        for m in ("relu", HYBRID, "tanh"):
            if m in models:
                d[f"lr50_{m}_lo"], d[f"lr50_{m}_hi"] = interval(*draws[ck + (m, oc)])
                d[f"nl_share_{m}"] = g[g.model == m].share_fail_never_learned.iloc[0]
        d["fail_is_never_learned"] = bool(d.get("nl_share_relu", 0) > 0.5)
        for m in (HYBRID, "tanh"):
            if m in models:
                r = ratio(draws[ck + (m, oc)], draws[ck + ("relu", oc)])
                d[f"ratio_{m}_relu"] = r["display"]
                d[f"ratio_{m}_relu_point"], d[f"ratio_{m}_relu_lo"], d[f"ratio_{m}_relu_hi"] = r["point"], r["lo"], r["hi"]
        rows.append(d)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures

def curve_figure(d, est, draws, panels, name, caption, models=("relu", HYBRID, "tanh")):
    import matplotlib.pyplot as plt
    from partA_common import COLORS, INK, MARKERS, MUTED, add_caption, savefig, style
    ncol = 4
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.3 * nrow), squeeze=False, sharey=True)
    for ax, key in zip(axes.ravel(), panels):
        g = d[(d.source == key[0]) & (d.dataset == key[2]) & (d.arch == key[3]) & (d.hidden == key[4])]
        for m in models:
            gm = g[g.model == m]
            if gm.empty:
                continue
            p = gm.groupby("lr").fail.mean()
            ax.plot(p.index, p.values, ls="none", marker=MARKERS[m], ms=4, color=COLORS[m], label=m)
            res, lb = draws[key + (m, "fail")]
            if res["status"] == "fit":
                xs = np.logspace(np.log10(p.index.min()), np.log10(p.index.max()), 80)
                ax.plot(xs, 1 / (1 + np.exp(-(res["a"] + res["b"] * np.log10(xs)))), color=COLORS[m], lw=1.2)
                lo, hi = boot_ci(lb)
                lo, hi = max(lo, np.log10(p.index.min())), min(hi, np.log10(p.index.max()))
                ax.axvspan(10 ** lo, 10 ** hi, color=COLORS[m], alpha=0.12, lw=0)
            else:
                ax.plot(p.index, p.values, color=COLORS[m], lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.axhline(0.5, color=MUTED, lw=0.6, ls="--")
        src = key[0].replace("v5 ", "v5 ").replace("v6 ", "v6 ")
        ax.set_title(f"{src} {key[2]} {key[3]} H={key[4]}", loc="left", color=INK, fontsize=7)
        style(ax)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("P(final < 15%)", color=MUTED)
    for ax in axes[-1]:
        ax.set_xlabel("learning rate", color=MUTED)
    axes[0, 0].legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout()
    savefig(fig, name)
    add_caption(name, caption)


def width_figure(est, name, caption):
    import matplotlib.pyplot as plt
    from partA_common import COLORS, INK, MARKERS, MUTED, add_caption, savefig, style
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.6), sharey=True)
    for ax, ds in zip(axes, ("mnist", "cifar10")):
        g = est[(est.kind == "calib") & (est.dataset == ds) & (est.outcome == "fail")]
        for j, m in enumerate(("relu", HYBRID, "tanh")):
            q = g[g.model == m].sort_values("hidden")
            off = (j - 1) * 0.06
            for _, r in q.iterrows():
                h = r.hidden * 10 ** off
                if r.status == "fit":
                    ax.plot([h, h], [max(r.ci_lo, 5e-5), min(r.ci_hi, 1.0)], color=COLORS[m], lw=0.8)
                    ax.plot([h], [r.lr50], marker=MARKERS[m], color=COLORS[m], ms=5)
                elif r.status in ("bracket", "above_grid", "below_grid", "non_monotone"):
                    lo = max(r.bound_lo, 5e-5)
                    hi = min(r.bound_hi, 1.0)
                    ax.plot([h, h], [lo, hi], color=COLORS[m], lw=2, alpha=0.8)
                    if r.status == "above_grid":
                        ax.plot([h], [lo], marker="^", color=COLORS[m], ms=5)
                    if r.status == "below_grid":
                        ax.plot([h], [hi], marker="v", color=COLORS[m], ms=5)
            ax.plot([], [], color=COLORS[m], marker=MARKERS[m], label=m)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{ds}: LR50 (final < 15%) vs width", loc="left", color=INK)
        ax.set_xlabel("hidden width H", color=MUTED)
        style(ax)
    axes[0].set_ylabel("LR50 (point, or bracket / bound)", color=MUTED)
    axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    savefig(fig, name)
    add_caption(name, caption)


# ----------------------------------------------------------------------------- main

def main():
    from partA_common import Claims, write_table
    from v5_io import v5_duplicates
    C = Claims("critical_lr")
    d = load_runs()
    est, draws = estimate_all(d)
    sub = "critical_lr"
    write_table(est, sub, "lr50_all",
                "A4. LR50 per cell, model and outcome (fail = final < 15%; late_collapse = outcomes.py class, "
                "never_learned runs excluded; degraded = final < 50% of the model's best grid mean final accuracy). "
                "status: fit (maximum likelihood, seed-bootstrap 95% CI, 2,000 resamples), bracket / above_grid / "
                "below_grid (separation: no fitted value), non_monotone. boot_sep_share = share of resamples that were "
                "separated (they enter the CI at their widening bracket end). share_fail_never_learned = share of the cell's final < 15% "
                "runs that never learned (outcomes.py).", digits=4)
    main_rt = ratio_table(est, draws, "main")
    for oc in OUTCOMES:
        t = main_rt[main_rt.outcome == oc].drop(columns="outcome")
        write_table(t, sub, f"lr50_ratios_{oc}",
                    f"A4 main table ({oc}). LR50 per model and the ratios LR50(tanh+relu)/LR50(relu) and "
                    "LR50(tanh)/LR50(relu): point with paired seed-bootstrap 95% CI when both are fits, otherwise the "
                    "interval implied by the brackets / bounds. Sources: version5 30-seed sweeps, v6 Stage 0 (seeds "
                    "1-10) and v6 Stage 2 (seeds 1-30 at three LRs, 1-10 elsewhere); all exploratory. nl_share_<model> = "
                    "share of the model's final < 15% runs that never learned; fail_is_never_learned marks cells where "
                    "most pure-ReLU failures are never_learned (a threshold of learning, not of late collapse).", digits=4)
    cal_rt = ratio_table(est, draws, "calib")
    for oc in ("fail", "late_collapse"):
        t = cal_rt[cal_rt.outcome == oc].drop(columns="outcome").sort_values(["dataset", "hidden"])
        write_table(t, sub, f"calib_width_{oc}",
                    f"Critical LR vs width, H = 12..512, 6 seeds, exploratory ({oc}). version5 calib_* (MNIST) and "
                    "c10_calib_* (CIFAR-10) files, mlp2, seeds 101-106; analysed separately from the 30-seed sweeps.",
                    digits=4)
    dup = v5_duplicates()
    C.put("v5_duplicate_runs", int(sum(dup.values())), "version5", desc="duplicate runs across the version5 files")
    for _, r in est.iterrows():
        tag = f"{r.source.replace(' ', '_')}.{r.dataset}.{r.arch}.h{r.hidden}.{r.model}.{r.outcome}"
        C.put(f"{tag}.status", r.status, r.seed_set)
        C.put(f"{tag}.display", r.display, r.seed_set, n=r.n_runs)
        if r.status == "fit":
            C.put(f"{tag}.lr50", r.lr50, r.seed_set, n=r.n_runs)
    for _, r in pd.concat([main_rt, cal_rt]).iterrows():
        tag = f"ratio.{r.source.replace(' ', '_')}.{r.dataset}.{r.arch}.h{r.hidden}.{r.outcome}"
        for m in (HYBRID, "tanh"):
            if f"ratio_{m}_relu" in r and isinstance(r[f"ratio_{m}_relu"], str):
                C.put(f"{tag}.{m}_relu", r[f"ratio_{m}_relu"], r.seed_set)
                C.put(f"{tag}.{m}_relu_lo", r[f"ratio_{m}_relu_lo"], r.seed_set)
                C.put(f"{tag}.{m}_relu_hi", r[f"ratio_{m}_relu_hi"], r.seed_set)
        for m in ("relu", HYBRID):
            if f"lr50_{m}_lo" in r and r[f"lr50_{m}_lo"] == r[f"lr50_{m}_lo"]:
                C.put(f"{tag}.lr50_{m}_lo", r[f"lr50_{m}_lo"], r.seed_set)
                C.put(f"{tag}.lr50_{m}_hi", r[f"lr50_{m}_hi"], r.seed_set)
                C.put(f"{tag}.nl_share_{m}", r[f"nl_share_{m}"], r.seed_set,
                      desc="share of final < 15% runs that never learned")
    C.save()
    panels = sorted({tuple(k[:5]) for k in draws if k[1] == "main"},
                    key=lambda k: (k[0].startswith("v6"), k[0], k[2], k[3], k[4]))
    curve_figure(d, est, draws, panels, "crit_lr_curves",
                 "Share of runs with final validation accuracy < 15% versus learning rate, per model (points), with the "
                 "logistic fit (line) and the seed-bootstrap 95% CI of LR50 (band) where the outcomes are not separated; "
                 "dotted lines join the points of separated cells (no fit reported). Panels: version5 30-seed sweeps "
                 "(mlp2), v6 Stage 0 (mlp2, seeds 1-10), v6 Stage 2 (cfg1, cfg2, mlp1; seeds 1-30 at three LRs per cell, "
                 "1-10 elsewhere). All exploratory.")
    width_figure(est, "crit_lr_width",
                 "Critical LR vs width, H = 12..512, 6 seeds, exploratory (version5 calib_* and c10_calib_*, mlp2). "
                 "Points: fitted LR50 of P(final < 15%) with its conservative seed-bootstrap 95% CI (thin line); thick "
                 "bars: brackets (separated outcomes) or first-crossing intervals (non-monotone cells); up / down "
                 "triangles: one-sided bounds (no failure inside the grid / failure already at the smallest LR).")
    print(f"critical_lr: {len(est)} estimates")


if __name__ == "__main__":
    main()
