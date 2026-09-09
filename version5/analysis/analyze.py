#!/usr/bin/env python3
"""Statistical analysis of the version5 hybrid-activation sweeps.

Discovers every stage CSV in results/ and writes results/summary_extended.md plus
results/figures/*.png.  (results/summary.md is the frozen first-look report and is
never rewritten by this script.)

Discovery rules
  main_<W>.csv / main_<W>_v2.csv       evaluation runs; _v2 wins when both exist
  calib_<W>.csv / calib_<W>_v2.csv     LR calibration grid (seeds 101-103)
  lrsweep_<W>.csv                      seed-replicated LR robustness (seeds 1-10)
  layout_<W>.csv                       layout / ratio variants
  mse_<W>.csv, calib_mse*.csv          MSE-loss replication

Statistics (all paired by seed, since every config was trained with the same seeds
and the same fixed train/val/test split):
  * mean / sd / median of test accuracy and macro-F1 per config
  * paired difference hybrid - parent, with a 95% bootstrap CI (10k resamples),
    a paired t-test and a Wilcoxon signed-rank test (scipy when available),
    plus the number of seeds on which the hybrid wins
  * crossover of those paired differences against hidden width
  * learning-rate robustness: collapse fractions and a within-1-point LR range

Row hygiene, applied everywhere: status must be "ok", test_confusion must hold
100 entries (a truncated row means a worker was killed mid-write), repeated header
lines are dropped, and duplicate run keys keep the first occurrence.

Runs with numpy + matplotlib only; scipy is optional (p-values are skipped without it).
"""
import csv, glob, json, os, re, sys, collections
import numpy as np
import matplotlib, matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import stats
except ImportError:  # p-values become "n/a"
    stats = None

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(0)

# The evaluation grid is reported at one seed count for every width, so the width axis is a
# like-for-like comparison and the pooled tests carry equal weight per cell.  Widths 12 and 32
# hold 30 seeds on disk from an earlier pass; those extra seeds are kept and reported
# separately (section 2x) rather than deleted or silently mixed in.
BALANCED_N = 20

COLLAPSE_ACC = 50.0     # final validation accuracy below this counts as a collapsed run
WITHIN_PTS = 1.0        # robustness score: stay within this many points of the config's best

# ---------------------------------------------------------------- palette (dataviz reference instance, light mode)
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; GRID = "#e6e5e1"
COLOR = {  # fixed hue per configuration, never cycled
    "tanh+relu":        "#2a78d6",  # blue    - the main hybrid
    "relu":             "#eb6834",  # orange
    "tanh":             "#1baf7a",  # aqua
    "leaky_relu":       "#eda100",  # yellow
    "sigmoid":          "#e87ba4",  # magenta
    "sigmoid+relu":     "#008300",  # green
    "tanh+leaky_relu":  "#4a3aa7",  # violet
}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "semibold", "axes.titlelocation": "left",
})

CFG_ORDER = ["relu", "tanh", "leaky_relu", "sigmoid", "tanh+relu", "tanh+leaky_relu", "sigmoid+relu"]
PAIRS = [("tanh+relu", "relu"), ("tanh+relu", "tanh"),
         ("tanh+leaky_relu", "leaky_relu"), ("tanh+leaky_relu", "tanh"),
         ("sigmoid+relu", "relu"), ("sigmoid+relu", "sigmoid")]
# The four comparisons that carry the article's crossover claim.
CROSS_PAIRS = [("tanh+relu", "relu"), ("tanh+relu", "tanh"),
               ("tanh+leaky_relu", "leaky_relu"), ("tanh+leaky_relu", "tanh")]

def order_for(groups):
    return [k for k in CFG_ORDER if k in groups] + [k for k in groups if k not in CFG_ORDER]

# ---------------------------------------------------------------- loading
def cfg_name(r):
    ratio = float(r["ratio"])
    if ratio == 0: return r["act1"]
    if ratio == 1: return r["act2"]
    return f'{r["act1"]}+{r["act2"]}'

BAD = []        # rows that were rejected, for the sanity section: (file, cfg, seed, lr, why)

def load(name):
    """Load one results CSV.  Rejected rows are recorded in BAD, never silently dropped."""
    path = name if os.path.isabs(name) else os.path.join(RES, name)
    rows = []
    if not os.path.exists(path):
        return rows
    base = os.path.basename(path)
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("seed") == "seed":
                continue                                   # repeated header from a parallel worker
            why = None
            if r.get("status") != "ok":
                why = f'status={r.get("status")}'
            elif not r.get("test_confusion") or r["test_confusion"].count("|") != 99:
                why = "truncated row"
            if why:
                BAD.append((base, cfg_name(r), r.get("seed"), r.get("lr"), r.get("hidden"), why))
                continue
            r["file"] = base
            r["cfg"] = cfg_name(r)
            for k in ("test_acc", "test_f1", "val_acc", "val_f1", "train_acc", "lr", "ratio"):
                r[k] = float(r[k])
            r["seed"] = int(r["seed"]); r["hidden"] = int(r["hidden"])
            r["epoch_val_acc"] = [float(x) for x in r["epoch_val_acc"].split("|")]
            r["epoch_loss"] = [float(x) for x in r["epoch_loss"].split("|")]
            rows.append(r)
    # de-duplicate (a run re-executed after an interrupted chunk): keep the first
    seen, out = set(), []
    for r in rows:
        k = (r["seed"], r["cfg"], r["layout"], r["ratio"], r["hidden"], r["lr"], r["loss"])
        if k not in seen:
            seen.add(k); out.append(r)
    return out

def by_cfg(rows, key="cfg"):
    g = collections.OrderedDict()
    for r in rows:
        g.setdefault(r[key], []).append(r)
    return g

def discover(prefix):
    """{width: (filename, superseded_filename_or_None)} for results/<prefix>_<W>[_v2].csv."""
    found = {}
    for p in sorted(glob.glob(os.path.join(RES, f"{prefix}_*.csv"))):
        m = re.fullmatch(rf"{prefix}_(\d+)(_v2)?\.csv", os.path.basename(p))
        if not m:
            continue
        w, v2 = int(m.group(1)), bool(m.group(2))
        cur = found.get(w)
        if cur is None:
            found[w] = (os.path.basename(p), None, v2)
        elif v2 and not cur[2]:                 # _v2 supersedes the plain file
            found[w] = (os.path.basename(p), cur[0], True)
        elif not v2 and cur[2]:
            found[w] = (cur[0], os.path.basename(p), True)
    return collections.OrderedDict(sorted((w, (f, sup)) for w, (f, sup, _) in found.items()))

# ---------------------------------------------------------------- statistics
def boot_ci(d, n=10000):
    d = np.asarray(d)
    if len(d) == 0:
        return (float("nan"), float("nan"))
    m = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)

def paired(hyb, base, metric):
    """hyb/base: lists of rows.  Returns dict with paired-by-seed statistics."""
    hb = {r["seed"]: r[metric] for r in hyb}
    bb = {r["seed"]: r[metric] for r in base}
    seeds = sorted(set(hb) & set(bb))
    d = np.array([hb[s] - bb[s] for s in seeds])
    out = dict(n=len(d), mean=d.mean() if len(d) else float("nan"),
               sd=d.std(ddof=1) if len(d) > 1 else 0.0,
               ci=boot_ci(d), wins=int((d > 0).sum()), ties=int((d == 0).sum()), d=d)
    if stats is not None and len(d) > 1:
        out["t_p"] = stats.ttest_rel([hb[s] for s in seeds], [bb[s] for s in seeds]).pvalue
        try:
            out["w_p"] = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        except ValueError:
            out["w_p"] = float("nan")
    return out

def fmt_p(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.3f}" if x >= 0.001 else f"{x:.1e}")

def summary_table(groups, order=None):
    lines = ["| config | lr | n | test acc mean ± sd | median | test macro-F1 mean ± sd | val acc mean |",
             "|---|---|---|---|---|---|---|"]
    for k in (order or groups):
        g = groups.get(k)
        if not g: continue
        acc = np.array([r["test_acc"] for r in g]); f1 = np.array([r["test_f1"] for r in g])
        va = np.array([r["val_acc"] for r in g])
        lrs = sorted({r["lr"] for r in g})
        lrtxt = "/".join(f"{l:g}" for l in lrs)
        lines.append(f"| {k} | {lrtxt} | {len(g)} | {acc.mean():.2f} ± {acc.std(ddof=1):.2f} | {np.median(acc):.2f} "
                     f"| {f1.mean():.2f} ± {f1.std(ddof=1):.2f} | {va.mean():.2f} |")
    return "\n".join(lines)

def paired_table(groups, pairs):
    lines = ["| comparison | metric | n | mean Δ (hybrid − parent) | 95% bootstrap CI | hybrid wins | paired t p | Wilcoxon p |",
             "|---|---|---|---|---|---|---|---|"]
    for hyb, base in pairs:
        if hyb not in groups or base not in groups: continue
        for metric, label in (("test_acc", "accuracy"), ("test_f1", "macro-F1")):
            p = paired(groups[hyb], groups[base], metric)
            if p["n"] == 0: continue
            lines.append(f"| {hyb} vs {base} | {label} | {p['n']} | {p['mean']:+.2f} | [{p['ci'][0]:+.2f}, {p['ci'][1]:+.2f}] "
                         f"| {p['wins']}/{p['n']} | {fmt_p(p.get('t_p'))} | {fmt_p(p.get('w_p'))} |")
    return "\n".join(lines)

# ---------------------------------------------------------------- figures
def strip_box(groups, order, metric, title, fname, ylabel):
    order = [k for k in order if k in groups and groups[k]]
    if not order: return
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    ymax_all = max(max(r[metric] for r in groups[k]) for k in order)
    ymin_all = min(min(r[metric] for r in groups[k]) for k in order)
    pad = max((ymax_all - ymin_all) * 0.06, 1e-3)
    for i, k in enumerate(order):
        v = np.array([r[metric] for r in groups[k]])
        c = COLOR.get(k, INK2)
        ax.boxplot(v, positions=[i], widths=0.5, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="none", edgecolor=c, linewidth=1.2),
                   medianprops=dict(color=c, linewidth=2), whiskerprops=dict(color=c, linewidth=1),
                   capprops=dict(color=c, linewidth=1))
        jitter = rng.uniform(-0.16, 0.16, len(v))
        ax.scatter(i + jitter, v, s=16, color=c, alpha=0.55, edgecolor=SURFACE, linewidth=0.6, zorder=3)
        ax.text(i, v.max() + pad * 0.4, f"mean {v.mean():.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_ylim(top=ymax_all + pad * 2.4)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([k.replace("+", "\n+ ") for k in order], fontsize=9)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=160); plt.close(fig)

def paired_delta_plot(groups, pairs, metric, title, fname):
    use = [(h, b) for h, b in pairs if h in groups and b in groups and paired(groups[h], groups[b], metric)["n"]]
    if not use: return
    fig, ax = plt.subplots(figsize=(7.2, 0.9 + 0.75 * len(use)))
    labels = []
    for i, (hyb, base) in enumerate(use):
        p = paired(groups[hyb], groups[base], metric)
        c = COLOR.get(hyb, INK2)
        ax.scatter(p["d"], np.full(len(p["d"]), i) + rng.uniform(-0.12, 0.12, len(p["d"])),
                   s=16, color=c, alpha=0.5, edgecolor=SURFACE, linewidth=0.6, zorder=3)
        ax.plot(p["ci"], [i, i], color=c, linewidth=2.2, zorder=4, solid_capstyle="round")
        ax.scatter([p["mean"]], [i], s=70, color=c, edgecolor=SURFACE, linewidth=1.2, zorder=5)
        ax.text(max(p["ci"][1], p["d"].max()) + 0.15, i, f"{p['mean']:+.2f}  [{p['ci'][0]:+.2f}, {p['ci'][1]:+.2f}]",
                va="center", fontsize=8, color=INK2)
        labels.append(f"{hyb}\nvs {base}")
    ax.axvline(0, color=INK2, linewidth=1, linestyle="--")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis(); ax.set_xlabel(f"paired difference in test {metric.replace('test_', '').replace('_', ' ')} (points), one dot = one seed")
    ax.set_title(title); ax.grid(axis="y", visible=False)
    xmax = ax.get_xlim()[1]; ax.set_xlim(right=xmax + 1.6)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=160); plt.close(fig)

def spread_labels(ax, labels, min_gap_frac=0.045):
    """labels: list of (x, y, text, color).  Pushes y positions apart (in axis fraction) so text does not collide."""
    if not labels: return
    lo, hi = ax.get_ylim(); span = hi - lo
    order_ = sorted(range(len(labels)), key=lambda i: labels[i][1])
    ys = [labels[i][1] for i in order_]
    gap = min_gap_frac * span
    for j in range(1, len(ys)):
        if ys[j] - ys[j - 1] < gap: ys[j] = ys[j - 1] + gap
    for j in range(len(ys) - 2, -1, -1):          # keep inside the axis by pushing back down
        if ys[j] > hi - gap: ys[j] = min(ys[j], ys[j + 1] - gap)
    for j, i in enumerate(order_):
        x, _, t, c = labels[i]
        ax.text(x, ys[j], t, fontsize=8, color=c, va="center")

def lr_curves(rows, title, fname, order, nseed_label=None):
    if not rows: return
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    g = by_cfg(rows); labels = []
    for k in order:
        if k not in g: continue
        per_lr = collections.defaultdict(list)
        for r in g[k]: per_lr[r["lr"]].append(r["val_acc"])
        lrs = sorted(per_lr); m = [np.mean(per_lr[l]) for l in lrs]
        c = COLOR.get(k, INK2)
        ax.plot(lrs, m, marker="o", markersize=5, linewidth=2, color=c, label=k, markeredgecolor=SURFACE)
        labels.append((lrs[-1] * 1.12, m[-1], k, c))
    spread_labels(ax, labels)
    all_lrs = sorted({r["lr"] for r in rows})
    ax.set_xscale("log"); ax.set_xticks(all_lrs); ax.set_xticklabels([f"{l:g}" for l in all_lrs])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_xlabel("learning rate")
    ax.set_ylabel(f"validation accuracy (%), {nseed_label or 'mean over calibration seeds'}")
    ax.set_title(title); ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
    ax.set_xlim(right=ax.get_xlim()[1] * 2.2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=160); plt.close(fig)

def lrsweep_plot(rows, width, fname):
    """Stage C: mean val accuracy vs LR with a shaded 10-90 percentile band over seeds."""
    if not rows: return
    g = by_cfg(rows)
    nseed = max(len({r["seed"] for r in v}) for v in g.values())
    fig, ax = plt.subplots(figsize=(7.6, 4.2)); labels = []
    for k in order_for(g):
        per_lr = collections.defaultdict(list)
        for r in g[k]: per_lr[r["lr"]].append(r["val_acc"])
        lrs = sorted(per_lr)
        m = np.array([np.mean(per_lr[l]) for l in lrs])
        lo = np.array([np.percentile(per_lr[l], 10) for l in lrs])
        hi = np.array([np.percentile(per_lr[l], 90) for l in lrs])
        c = COLOR.get(k, INK2)
        ax.plot(lrs, m, marker="o", markersize=5, linewidth=2, color=c, label=k, markeredgecolor=SURFACE)
        ax.fill_between(lrs, lo, hi, color=c, alpha=0.13, linewidth=0)
        labels.append((lrs[-1] * 1.12, m[-1], k, c))
    spread_labels(ax, labels)
    all_lrs = sorted({r["lr"] for r in rows})
    ax.set_xscale("log"); ax.set_xticks(all_lrs); ax.set_xticklabels([f"{l:g}" for l in all_lrs])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.axhline(COLLAPSE_ACC, color=INK2, linewidth=1, linestyle=":")
    ax.text(all_lrs[0], COLLAPSE_ACC + 1, "collapse threshold", fontsize=7.5, color=INK2)
    ax.set_xlabel("learning rate")
    ax.set_ylabel(f"validation accuracy (%), mean and 10–90 pct band ({nseed} seeds)")
    ax.set_title(f"Learning-rate robustness, {width} hidden, CE ({nseed} seeds per point)")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
    ax.set_xlim(right=ax.get_xlim()[1] * 2.2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=160); plt.close(fig)

def learning_curves(groups, order, title, fname):
    order = [k for k in order if k in groups and groups[k]]
    if not order: return
    fig, ax = plt.subplots(figsize=(7.6, 4.0)); labels = []
    for k in order:
        m = np.array([r["epoch_val_acc"] for r in groups[k]])
        mean = m.mean(0); sd = m.std(0, ddof=1) if len(m) > 1 else np.zeros(m.shape[1])
        x = np.arange(1, m.shape[1] + 1); c = COLOR.get(k, INK2)
        ax.plot(x, mean, color=c, linewidth=2, label=k)
        ax.fill_between(x, mean - sd, mean + sd, color=c, alpha=0.12, linewidth=0)
        labels.append((x[-1] + 0.15, mean[-1], k, c))
    spread_labels(ax, labels)
    ax.set_xlabel("epoch"); ax.set_ylabel("validation accuracy (%), mean ± sd over seeds")
    ax.set_title(title); ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_xlim(right=ax.get_xlim()[1] + 1.8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=160); plt.close(fig)

def crossover_plot(cross, widths, metric, title, fname, ylabel):
    """cross[(hyb, base)][width] = paired() dict.  x = width (log), y = paired delta with 95% CI."""
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    for (hyb, base) in CROSS_PAIRS:
        pts = [(w, cross[(hyb, base)][w]) for w in widths
               if w in cross.get((hyb, base), {}) and cross[(hyb, base)][w]["n"] > 1]
        if not pts: continue
        xs = [w for w, _ in pts]
        ys = np.array([p["mean"] for _, p in pts])
        lo = ys - np.array([p["ci"][0] for _, p in pts])
        hi = np.array([p["ci"][1] for _, p in pts]) - ys
        c = COLOR.get(hyb, INK2)
        vs_tanh = (base == "tanh")
        ax.errorbar(xs, ys, yerr=[lo, hi], color=c, linewidth=2, capsize=3, elinewidth=1.2,
                    marker=("s" if vs_tanh else "o"), markersize=6, markeredgecolor=SURFACE,
                    linestyle=("--" if vs_tanh else "-"), label=f"{hyb} − {base}")
    ax.axhline(0, color=INK2, linewidth=1, linestyle="--")
    ax.set_xscale("log", base=2); ax.set_xticks(widths)
    ax.set_xticklabels([str(w) for w in widths])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("hidden width (neurons per hidden layer, log scale)")
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(frameon=False, fontsize=8.5, loc="best")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=160); plt.close(fig)

def width_acc_plot(per_width, widths, fname):
    fig, ax = plt.subplots(figsize=(7.8, 4.4)); labels = []
    for k in CFG_ORDER:
        pts = [(w, per_width[w][k]) for w in widths if k in per_width.get(w, {}) and per_width[w][k]]
        if not pts: continue
        xs = [w for w, _ in pts]
        m = np.array([np.mean([r["test_acc"] for r in g]) for _, g in pts])
        sd = np.array([np.std([r["test_acc"] for r in g], ddof=1) if len(g) > 1 else 0.0 for _, g in pts])
        c = COLOR.get(k, INK2)
        ax.plot(xs, m, marker="o", markersize=5, linewidth=2, color=c, label=k, markeredgecolor=SURFACE)
        ax.fill_between(xs, m - sd, m + sd, color=c, alpha=0.13, linewidth=0)
        labels.append((xs[-1] * 1.09, m[-1], k, c))
    spread_labels(ax, labels)
    ax.set_xscale("log", base=2); ax.set_xticks(widths); ax.set_xticklabels([str(w) for w in widths])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("hidden width (neurons per hidden layer, log scale)")
    ax.set_ylabel("test accuracy (%), mean ± sd over seeds")
    ax.set_title("Test accuracy against hidden width, per configuration")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    ax.set_xlim(right=widths[-1] * 1.9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=160); plt.close(fig)

# ---------------------------------------------------------------- LR robustness tables
def collapse_table(rows):
    g = by_cfg(rows)
    lrs = sorted({r["lr"] for r in rows})
    lines = ["| config | " + " | ".join(f"lr={l:g}" for l in lrs) + " |", "|---|" + "---|" * len(lrs)]
    for k in order_for(g):
        cells = []
        per = collections.defaultdict(list)
        for r in g[k]: per[r["lr"]].append(r["val_acc"])
        for l in lrs:
            v = per.get(l)
            if not v: cells.append("–"); continue
            n_bad = sum(1 for x in v if x < COLLAPSE_ACC)
            cells.append(f"{n_bad}/{len(v)}")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    return "\n".join(lines)

def lr_mean_table(rows, order=None):
    g = by_cfg(rows)
    lrs = sorted({r["lr"] for r in rows})
    lines = ["| config | " + " | ".join(f"lr={l:g}" for l in lrs) + " |", "|---|" + "---|" * len(lrs)]
    for k in (order or order_for(g)):
        if k not in g: continue
        per = collections.defaultdict(list)
        for r in g[k]: per[r["lr"]].append(r["val_acc"])
        lines.append(f"| {k} | " + " | ".join(f"{np.mean(per[l]):.2f}" if l in per else "–" for l in lrs) + " |")
    return "\n".join(lines)

def _widest_run(lrs, ok):
    """Widest contiguous block of True in `ok`; returns (lo_lr, hi_lr, n_points, decades) or None."""
    bi = bj = -1; i = 0
    while i < len(ok):
        if ok[i]:
            j = i
            while j + 1 < len(ok) and ok[j + 1]: j += 1
            if bi < 0 or (j - i) > (bj - bi): bi, bj = i, j
            i = j + 1
        else:
            i += 1
    if bi < 0: return None
    dec = np.log10(lrs[bj] / lrs[bi]) if lrs[bi] > 0 else 0.0
    return (lrs[bi], lrs[bj], bj - bi + 1, dec)

def complete_configs(rows):
    """Split an lrsweep stage into configs whose LR grid is complete and those that are not.
    A partially-run config must not appear in a table or a robustness score -- a missing
    high-LR cell would silently read as "never collapsed". Complete configs are still
    reported, so a stage that is only missing one config still yields a usable width."""
    cells = collections.Counter((r["cfg"], r["lr"]) for r in rows)
    lrs = {l for _, l in cells}
    per = collections.defaultdict(dict)
    for (c, l), n in cells.items():
        per[c][l] = n
    full = max((max(d.values()) for d in per.values()), default=0)
    ok, bad = [], []
    for c, d in per.items():
        (ok if set(d) == lrs and set(d.values()) == {full} else bad).append(c)
    return set(ok), sorted(bad), sorted(lrs)

def robustness_score(rows):
    """Two robustness measures per config, both on the tested LR grid:
      "within"   widest contiguous interval where mean val accuracy stays within
                 WITHIN_PTS of that config's own best  (the measure the plan asked for)
      "nocollapse" widest contiguous interval where no seed collapsed at all.
    The first is dominated by the low-LR end and barely separates configs, because a
    collapse lands far outside a 1-point band; the second is what "tolerates a large
    learning rate" actually means, so both are reported.
    Returns {cfg: {"within": tuple|None, "nocollapse": tuple|None, "best": float}}."""
    g = by_cfg(rows)
    out = {}
    for k, v in g.items():
        per = collections.defaultdict(list)
        for r in v: per[r["lr"]].append(r["val_acc"])
        lrs = sorted(per)
        m = [np.mean(per[l]) for l in lrs]
        best = max(m)
        within = _widest_run(lrs, [x >= best - WITHIN_PTS for x in m])
        nocol = _widest_run(lrs, [all(x >= COLLAPSE_ACC for x in per[l]) for l in lrs])
        out[k] = {"within": within, "nocollapse": nocol, "best": best}
    return out

def robustness_table(per_width_lr):
    """per_width_lr: {width: rows}.  Returns (markdown, scores)."""
    widths = sorted(per_width_lr)
    scores = {w: robustness_score(per_width_lr[w]) for w in widths}
    cfgs = [k for k in CFG_ORDER if any(k in scores[w] for w in widths)]
    def fmt(s):
        return "–" if s is None else f"{s[0]:g}–{s[1]:g} ({s[2]} pts, {s[3]:.2f} dec)"
    lines = ["| config | measure | " + " | ".join(f"W={w}" for w in widths) + " |",
             "|---|---|" + "---|" * len(widths)]
    for k in cfgs:
        lines.append(f"| {k} | within {WITHIN_PTS:g} pt of own best | "
                     + " | ".join(fmt(scores[w].get(k, {}).get("within")) for w in widths) + " |")
        lines.append(f"| {k} | no seed collapsed | "
                     + " | ".join(fmt(scores[w].get(k, {}).get("nocollapse")) for w in widths) + " |")
    return "\n".join(lines), scores

# ---------------------------------------------------------------- per-dataset sections
def width_sections(per_width, widths, ds, tag, sec):
    """Per-width tables and figures plus the crossover block, for one dataset.

    `ds` is the dataset name shown in headings, `tag` a figure-filename suffix ("" for
    MNIST, so its figure names are unchanged), `sec` the section number to use.  MNIST
    and CIFAR-10 go through this same code, so a difference between the two reports is a
    difference in the data and never in how it was reduced.
    """
    md = []
    cross = {p: {} for p in CROSS_PAIRS}
    for w in widths:
        g = per_width[w]
        order = order_for(g)
        nseeds = sorted({len(v) for v in g.values()})
        nlabel = str(nseeds[0]) if len(nseeds) == 1 else f"{min(nseeds)}–{max(nseeds)}"
        md += [f"## {sec}.{w} {ds}, {w} hidden neurons, cross-entropy ({nlabel} seeds per config)", "",
               summary_table(g, order), "",
               "Paired comparisons (hybrid minus each homogeneous parent, same seeds):", "",
               paired_table(g, PAIRS), ""]
        strip_box(g, order, "test_acc", f"{ds}, {w} hidden, CE: test accuracy per configuration",
                  f"fig_acc_{w}{tag}.png", "test accuracy (%)")
        strip_box(g, order, "test_f1", f"{ds}, {w} hidden, CE: test macro-F1 per configuration",
                  f"fig_f1_{w}{tag}.png", "test macro-F1 (%)")
        paired_delta_plot(g, PAIRS, "test_acc", f"Paired differences, {w} hidden, CE — accuracy",
                          f"fig_paired_acc_{w}{tag}.png")
        paired_delta_plot(g, PAIRS, "test_f1", f"Paired differences, {w} hidden, CE — macro-F1",
                          f"fig_paired_f1_{w}{tag}.png")
        learning_curves(g, order, f"Validation accuracy per epoch, {w} hidden, CE", f"fig_curves_{w}{tag}.png")
        md += [f"![](figures/fig_acc_{w}{tag}.png)", f"![](figures/fig_f1_{w}{tag}.png)",
               f"![](figures/fig_paired_acc_{w}{tag}.png)", f"![](figures/fig_paired_f1_{w}{tag}.png)",
               f"![](figures/fig_curves_{w}{tag}.png)", ""]
        for (hyb, base) in CROSS_PAIRS:
            if hyb in g and base in g:
                cross[(hyb, base)][w] = {"acc": paired(g[hyb], g[base], "test_acc"),
                                         "f1": paired(g[hyb], g[base], "test_f1")}
    return md, cross


def crossover_sections(cross, per_width, widths, ds, tag, sec):
    md = [f"## {sec}. {ds} crossover: paired hybrid − parent against width", "",
          "One row per comparison and width. Δ is the paired mean difference in points "
          "(positive = the hybrid is ahead), CI is the 95% bootstrap interval.", ""]
    for metric, label in (("acc", "accuracy"), ("f1", "macro-F1")):
        md += [f"**{label}**", "",
               "| comparison | width | n | Δ | 95% CI | wins | paired t p | Wilcoxon p |",
               "|---|---|---|---|---|---|---|---|"]
        for (hyb, base) in CROSS_PAIRS:
            for w in widths:
                c = cross[(hyb, base)].get(w)
                if not c: continue
                p = c[metric]
                md.append(f"| {hyb} − {base} | {w} | {p['n']} | {p['mean']:+.2f} | "
                          f"[{p['ci'][0]:+.2f}, {p['ci'][1]:+.2f}] | {p['wins']}/{p['n']} | "
                          f"{fmt_p(p.get('t_p'))} | {fmt_p(p.get('w_p'))} |")
        md.append("")
    if widths:
        crossover_plot({k: {w: v[w]["acc"] for w in v} for k, v in cross.items()}, widths, "acc",
                       f"{ds}: hybrid − parent against hidden width (test accuracy, 95% bootstrap CI)",
                       f"fig_crossover{tag}.png", "paired Δ test accuracy (points)")
        crossover_plot({k: {w: v[w]["f1"] for w in v} for k, v in cross.items()}, widths, "f1",
                       f"{ds}: hybrid − parent against hidden width (test macro-F1, 95% bootstrap CI)",
                       f"fig_crossover_f1{tag}.png", "paired Δ test macro-F1 (points)")
        width_acc_plot(per_width, widths, f"fig_width_acc{tag}.png")
        md += [f"![](figures/fig_crossover{tag}.png)", f"![](figures/fig_crossover_f1{tag}.png)",
               f"![](figures/fig_width_acc{tag}.png)", ""]
    return md


# ---------------------------------------------------------------- main
def main():
    md = ["# version5 results (extended sweep): hybrid activations across width, learning rate and layout", ""]
    md += ["Protocol: MLP D→H→H→10 (D = 784 on MNIST, 3072 on CIFAR-10), batch-size-1 SGD, 10 epochs, "
           "fixed 45k/5k/10k train/val/test split "
           "shared by all runs, per-epoch shuffling, He/Kaiming-uniform init. Hidden layers carry per-neuron activation "
           "masks; the output layer is uniform (linear+softmax with cross-entropy, or sigmoid with MSE). Learning rate "
           "chosen per configuration on validation accuracy using calibration seeds (101–103) that are disjoint "
           "from the evaluation seeds (1–N). All comparisons are paired by seed. Bootstrap CIs use 10,000 resamples "
           f"and a collapsed run means final validation accuracy below {COLLAPSE_ACC:g}%.", ""]

    mains = discover("main")
    calibs = discover("calib")
    lrsweeps = discover("lrsweep")
    layouts = discover("layout")
    c10_mains = discover("c10_main")
    c10_calibs = discover("c10_calib")

    superseded = [(w, f, s) for w, (f, s) in mains.items() if s]
    md += ["## 0. Files used", "", "| stage | width | file | superseded |", "|---|---|---|---|"]
    for label, d in (("main", mains), ("calib", calibs), ("lrsweep", lrsweeps), ("layout", layouts),
                     ("main (CIFAR-10)", c10_mains), ("calib (CIFAR-10)", c10_calibs)):
        for w, (f, s) in d.items():
            md.append(f"| {label} | {w} | `{f}` | {('`' + s + '`') if s else '—'} |")
    md.append("")
    if superseded:
        md += ["Where both `main_<W>.csv` and `main_<W>_v2.csv` exist the `_v2` file is used and the older one ignored: "
               + ", ".join(f"W={w} uses `{f}`, ignores `{s}`" for w, f, s in superseded) + ".", ""]

    main_rows_all = {w: load(f) for w, (f, _) in mains.items()}
    main_rows = {w: [r for r in rows if r["seed"] <= BALANCED_N] for w, rows in main_rows_all.items()}
    widths = [w for w in sorted(main_rows) if main_rows[w]]
    per_width = {w: by_cfg(main_rows[w]) for w in widths}
    over_n = {w: sorted({r["seed"] for r in rows}) for w, rows in main_rows_all.items()
              if max((r["seed"] for r in rows), default=0) > BALANCED_N}

    c10_rows = {w: [r for r in load(f) if r["seed"] <= BALANCED_N] for w, (f, _) in c10_mains.items()}
    c10_widths = [w for w in sorted(c10_rows) if c10_rows[w]]
    c10_per_width = {w: by_cfg(c10_rows[w]) for w in c10_widths}

    # ---- run inventory (config x width -> n)
    md += [f"## 1. Run inventory (evaluation runs, config × width → n)", "",
           f"Every width is reported at the same seed count, n = {BALANCED_N} (seeds 1–{BALANCED_N}), "
           "so the width axis compares like with like.", ""]
    md += [
           "| config | " + " | ".join(f"W={w}" for w in widths) + " |", "|---|" + "---|" * len(widths)]
    for k in CFG_ORDER:
        cells = [str(len(per_width[w].get(k, []))) if per_width[w].get(k) else "–" for w in widths]
        if all(c == "–" for c in cells): continue
        md.append(f"| {k} | " + " | ".join(cells) + " |")
    md.append("")
    md += ["| other stage | file | ok runs |", "|---|---|---|"]
    other = collections.OrderedDict()
    for label, d in (("calib", calibs), ("lrsweep", lrsweeps), ("layout", layouts)):
        for w, (f, _) in d.items():
            other[f] = load(f)
    for extra in ("mse_12.csv", "calib_mse.csv", "mse_128.csv", "calib_mse_128.csv"):
        if os.path.exists(os.path.join(RES, extra)):
            other[extra] = load(extra)
    for f, rows in other.items():
        md.append(f"| {f.split('_')[0]} | `{f}` | {len(rows)} |")
    md.append("")

    # ---- per-width sections (MNIST, then CIFAR-10 through the same code path)
    block, cross = width_sections(per_width, widths, "MNIST", "", 2)
    md += block
    md += crossover_sections(cross, per_width, widths, "MNIST", "", 3)

    if c10_widths:
        md += [f"## 3c. CIFAR-10 replication", "",
               "The same protocol on a harder dataset: 3072 inputs instead of 784, the official "
               "10 000-image test batch instead of a slice of the training file, and only the 5 000 "
               "validation images drawn out of the 50 000 training images by the same fixed "
               "`SPLIT_SEED`. Learning rates are calibrated separately for CIFAR-10 "
               "(`results/best_lr_c10.json`) because they do not transfer between datasets. "
               "Absolute accuracy is far below a convolutional network's — an MLP on raw pixels is "
               "not a competitive CIFAR-10 model — which does not affect the paired within-dataset "
               "comparisons this report is built on.", "",
               "| config | " + " | ".join(f"W={w}" for w in c10_widths) + " |",
               "|---|" + "---|" * len(c10_widths)]
        for k in CFG_ORDER:
            cells = [str(len(c10_per_width[w].get(k, []))) if c10_per_width[w].get(k) else "–"
                     for w in c10_widths]
            if all(c == "–" for c in cells): continue
            md.append(f"| {k} | " + " | ".join(cells) + " |")
        md.append("")
        block, c10_cross = width_sections(c10_per_width, c10_widths, "CIFAR-10", "_c10", 3.1)
        md += block
        md += crossover_sections(c10_cross, c10_per_width, c10_widths, "CIFAR-10", "_c10", 3.2)

        # ---- the article's central comparison: does the crossover replicate?
        md += ["## 3.3 Does the crossover replicate across datasets?", "",
               "Paired Δ in test accuracy points at each width, MNIST beside CIFAR-10. A crossover "
               "that is a property of hybrid activation rather than of MNIST should change sign at "
               "a similar width in both columns.", ""]
        shared = [w for w in widths if w in c10_widths]
        for (hyb, base) in CROSS_PAIRS:
            rows_ = []
            for w in shared:
                a = cross[(hyb, base)].get(w)
                b = c10_cross[(hyb, base)].get(w)
                if not a or not b: continue
                rows_.append((w, a["acc"], b["acc"]))
            if not rows_: continue
            md += [f"**{hyb} − {base}**", "",
                   "| width | MNIST Δ | MNIST wins | CIFAR-10 Δ | CIFAR-10 wins | same sign |",
                   "|---|---|---|---|---|---|"]
            for w, a, b in rows_:
                same = "yes" if (a["mean"] > 0) == (b["mean"] > 0) else "**no**"
                md.append(f"| {w} | {a['mean']:+.2f} [{a['ci'][0]:+.2f}, {a['ci'][1]:+.2f}] | "
                          f"{a['wins']}/{a['n']} | {b['mean']:+.2f} [{b['ci'][0]:+.2f}, {b['ci'][1]:+.2f}] | "
                          f"{b['wins']}/{b['n']} | {same} |")
            md.append("")

    # ---- widths that hold more seeds than the balanced grid uses
    if over_n:
        md += ["## 3x. Widths with more seeds than the balanced grid reports", "",
               f"The tables above cap every width at n = {BALANCED_N} so the width axis is "
               "like-for-like. Widths 12 and 32 were run to 30 seeds in an earlier pass and those "
               "runs are kept. Below, each such width is shown at its full seed count next to the "
               f"capped n = {BALANCED_N}. A paired difference that moves materially between the two "
               "is a warning that the cell is noise-limited, not a reason to prefer either number.", ""]
        for w, seeds in sorted(over_n.items()):
            gf = by_cfg(main_rows_all[w])
            gc = per_width[w]
            md += [f"**MNIST, width {w} — n = {len(seeds)} (full) against n = {BALANCED_N} (reported)**", "",
                   f"| comparison | Δ at n={BALANCED_N} | Δ at n={len(seeds)} | wins (full) | t p (full) |",
                   "|---|---|---|---|---|"]
            for (hyb, base) in CROSS_PAIRS:
                if hyb not in gf or base not in gf:
                    continue
                a = paired(gc[hyb], gc[base], "test_acc")
                b = paired(gf[hyb], gf[base], "test_acc")
                md.append(f"| {hyb} − {base} | {a['mean']:+.2f} [{a['ci'][0]:+.2f}, {a['ci'][1]:+.2f}] | "
                          f"{b['mean']:+.2f} [{b['ci'][0]:+.2f}, {b['ci'][1]:+.2f}] | "
                          f"{b['wins']}/{b['n']} | {fmt_p(b.get('t_p'))} |")
            md.append("")

    # ---- LR robustness (stage C)
    md += ["## 4. Learning-rate robustness (seed-replicated, evaluation seeds)", ""]
    lr_rows = {}
    partial_lr = []
    for w, (f, _) in lrsweeps.items():
        rows = load(f)
        if not rows: continue
        ok, bad, _ = complete_configs(rows)
        if not ok:
            partial_lr.append((w, f, len(rows), bad))
            continue
        if bad:
            partial_lr.append((w, f, None, bad))
        lr_rows[w] = [r for r in rows if r["cfg"] in ok]
        nseed = len({r["seed"] for r in rows})
        lrsweep_plot(rows, w, f"fig_lrsweep_{w}.png")
        md += [f"**{w} hidden, CE — mean validation accuracy ({nseed} seeds)**", "", lr_mean_table(rows), "",
               f"**{w} hidden, CE — collapsed seeds (final val acc < {COLLAPSE_ACC:g}%), count / n**", "",
               collapse_table(rows), "", f"![](figures/fig_lrsweep_{w}.png)", ""]
    for w, f, n, bad in partial_lr:
        names = ", ".join(f"`{c}`" for c in bad)
        if n is None:
            md += [f"_At width {w} (`{f}`), {names} is omitted from the tables and the robustness "
                   f"score: its learning-rate grid is not fully run, and a missing high-LR cell "
                   f"would read as \"never collapsed\". The other configs are complete._", ""]
        else:
            md += [f"_Width {w} (`{f}`) is excluded entirely: no config has a complete "
                   f"learning-rate grid ({n} rows)._", ""]
    if lr_rows:
        tbl, _ = robustness_table(lr_rows)
        md += [f"**Robustness score** — widest contiguous LR interval on the tested grid over which a config's "
               f"mean validation accuracy stays within {WITHIN_PTS:g} point of its own best "
               f"(shown as lo–hi, number of grid points, decades spanned):", "", tbl, ""]

    # ---- calibration LR grids
    md += ["## 5. Calibration LR grids (validation accuracy, mean over calibration seeds)", ""]
    for w, (f, _) in calibs.items():
        rows = load(f)
        if not rows: continue
        md += [f"**{w} hidden, CE — `{f}`**", "", lr_mean_table(rows, CFG_ORDER), ""]
        lr_curves(rows, f"LR calibration, {w} hidden, CE", f"fig_calib_{w}.png", CFG_ORDER)
        md += [f"![](figures/fig_calib_{w}.png)", ""]
    for w, (f, _) in c10_calibs.items():
        rows = load(f)
        if not rows: continue
        md += [f"**CIFAR-10, {w} hidden, CE — `{f}`**", "", lr_mean_table(rows, CFG_ORDER), ""]
        lr_curves(rows, f"LR calibration, CIFAR-10, {w} hidden, CE", f"fig_calib_{w}_c10.png", CFG_ORDER)
        md += [f"![](figures/fig_calib_{w}_c10.png)", ""]

    def selected_lr_table(path, caption):
        if not os.path.exists(path):
            return []
        best = json.load(open(path))
        ws = sorted({int(k.split("/")[4]) for k in best if k.endswith("/ce")})
        if not ws:
            return []
        out = [caption, "", "| config | " + " | ".join(f"W={w}" for w in ws) + " |",
               "|---|" + "---|" * len(ws)]
        for k in CFG_ORDER:
            a1, a2, r = (k.split("+")[0], k.split("+")[1], 0.5) if "+" in k else (k, k, 0.0)
            cells = [f"{best[f'{a1}/{a2}/{r:g}/interleave/{w}/ce']:g}"
                     if f"{a1}/{a2}/{r:g}/interleave/{w}/ce" in best else "–" for w in ws]
            if all(c == "–" for c in cells): continue
            out.append(f"| {k} | " + " | ".join(cells) + " |")
        out.append("")
        return out

    md += selected_lr_table(os.path.join(RES, "best_lr.json"),
                            "**Selected learning rate per config and width (cross-entropy), "
                            "from `results/best_lr.json`**")
    md += selected_lr_table(os.path.join(RES, "best_lr_c10.json"),
                            "**Selected learning rate per config and width, CIFAR-10 "
                            "(cross-entropy), from `results/best_lr_c10.json`**")

    # ---- layout / ratio
    md += ["## 6. Layout and ratio", ""]
    for w, (f, _) in layouts.items():
        rows = load(f)
        if not rows: continue
        g = per_width.get(w, {})
        lay = collections.OrderedDict()
        for r in rows:
            lay.setdefault((r["cfg"], r["layout"], r["ratio"]), []).append(r)
        md += [f"**{w} hidden, CE — `{f}`.** Each pair uses the LR calibrated for its interleave/0.5 variant, "
               "so ratios 0.25 and 0.75 are not separately tuned.", "",
               "| pair | layout | ratio | lr | n | test acc mean ± sd | test macro-F1 mean ± sd "
               "| Δ acc vs non-tanh parent | Δ acc vs tanh | Δ acc vs interleave 0.5 |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for (cfg, layout, ratio), v in sorted(lay.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
            parent = cfg.split("+")[1] if "+" in cfg else cfg
            acc = np.array([r["test_acc"] for r in v]); f1 = np.array([r["test_f1"] for r in v])
            dp = paired(v, g.get(parent, []), "test_acc")
            dt = paired(v, g.get("tanh", []), "test_acc")
            base_int = lay.get((cfg, "interleave", 0.5), [])
            db = paired(v, base_int, "test_acc")
            def d(x):
                return "–" if x["n"] == 0 else f"{x['mean']:+.2f} [{x['ci'][0]:+.2f}, {x['ci'][1]:+.2f}]"
            md.append(f"| {cfg} | {layout} | {ratio:g} | {v[0]['lr']:g} | {len(v)} | "
                      f"{acc.mean():.2f} ± {acc.std(ddof=1):.2f} | {f1.mean():.2f} ± {f1.std(ddof=1):.2f} | "
                      f"{d(dp)} | {d(dt)} | {d(db)} |")
        md.append("")

    # ---- MSE replications
    md += ["## 7. Replication under the original loss (MSE, uniform sigmoid output)", ""]
    for w in (12, 128):
        rows = load(f"mse_{w}.csv")
        if not rows: continue
        gm = by_cfg(rows); om = order_for(gm)
        md += [f"**{w} hidden, MSE — `mse_{w}.csv` ({len(gm[om[0]])} seeds)**", "", summary_table(gm, om), "",
               paired_table(gm, [("tanh+relu", "relu"), ("tanh+relu", "tanh")]), ""]
        strip_box(gm, om, "test_acc", f"MNIST, {w} hidden, MSE: test accuracy", f"fig_acc_{w}_mse.png", "test accuracy (%)")
        md += [f"![](figures/fig_acc_{w}_mse.png)", ""]
        cal = load(f"calib_mse_{w}.csv") or (load("calib_mse.csv") if w == 12 else [])
        if cal:
            md += [f"MSE calibration grid, {w} hidden (validation accuracy, mean over calibration seeds):", "",
                   lr_mean_table(cal, CFG_ORDER), ""]
            lr_curves(cal, f"LR calibration, {w} hidden, MSE + sigmoid output", f"fig_calib_{w}_mse.png", CFG_ORDER)
            md += [f"![](figures/fig_calib_{w}_mse.png)", ""]

    # ---- rejected rows
    md += ["## 8. Rejected rows", ""]
    if BAD:
        md += ["| file | config | width | seed | lr | reason |", "|---|---|---|---|---|---|"]
        for f, cfg, seed, lr, hid, why in BAD:
            md.append(f"| `{f}` | {cfg} | {hid} | {seed} | {lr} | {why} |")
        md.append("")
    else:
        md += ["No row was rejected: every run in every CSV has `status=ok` and a complete confusion matrix.", ""]

    with open(os.path.join(RES, "summary_extended.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[wrote {os.path.join(RES, 'summary_extended.md')} and figures in {FIG}]", file=sys.stderr)

if __name__ == "__main__":
    main()
