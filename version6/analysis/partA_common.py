"""Shared helpers for the Part A analysis: output paths, deterministic tables and figures,
paired statistics, and the claims store that FINDINGS_partA.md reads its numbers from.

Every random draw uses SEED (20261018). Tables are written as CSV and as LaTeX (booktabs);
figures as PDF and 300-dpi PNG with no timestamps, so a second run is byte-identical.
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402

V6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(V6, "analysis_out", "partA")
SEED = 20261018
BOOT_N = 10000

# Model colours: the validated reference palette already used by analysis/figures.py,
# fixed per model (never cycled). Markers give a second, colour-independent encoding.
from figures import COLORS, INK, MUTED, GRID  # noqa: E402,F401
MARKERS = {"relu": "o", "tanh+relu": "s", "tanh": "^", "leaky_relu": "v", "sigmoid": "D",
           "tanh+relu@conv": "P", "tanh+relu@hidden": "X", "tanh+relu@0.25": "<", "tanh+relu@0.75": ">"}
MODEL_ORDER = list(COLORS)

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.hashsalt": "partA", "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
})


def out_path(*parts):
    p = os.path.join(OUT, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


# ----------------------------------------------------------------------------- tables

def _tex_escape(s):
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 ("$", r"\$"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")):
        s = s.replace(a, b)
    return s


def _fmt(v, digits):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if v != 0 and (abs(v) >= 1e5 or abs(v) < 10 ** (-digits)):
            return f"{v:.{max(1, digits - 1)}e}"
        return f"{v:.{digits}f}"
    return str(v)


def write_table(df, subdir, name, caption, digits=3, label=None):
    """Write <OUT>/<subdir>/<name>.csv and .tex (booktabs). Returns the CSV path."""
    df = df.reset_index(drop=True)
    csv = out_path(subdir, f"{name}.csv")
    df.to_csv(csv, index=False, float_format="%.6g", lineterminator="\n")
    cols = list(df.columns)
    align = "".join("r" if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]) else "l"
                    for c in cols)
    lines = [r"\begin{table}[htbp]\centering\small", rf"\caption{{{_tex_escape(caption)}}}",
             rf"\label{{{label or 'tab:' + name}}}", r"\resizebox{\linewidth}{!}{%",
             rf"\begin{{tabular}}{{@{{}}{align}@{{}}}}", r"\toprule",
             " & ".join(_tex_escape(c) for c in cols) + r" \\", r"\midrule"]
    for row in df.itertuples(index=False):
        lines.append(" & ".join(_tex_escape(_fmt(v, digits)) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    with open(out_path(subdir, f"{name}.tex"), "w") as f:
        f.write("\n".join(lines))
    return csv


# ----------------------------------------------------------------------------- figures

def style(ax):
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def savefig(fig, name, subdir="figures"):
    """PDF + 300-dpi PNG without timestamps (byte-identical on re-run)."""
    fig.savefig(out_path(subdir, f"{name}.pdf"), metadata={"CreationDate": None, "ModDate": None, "Producer": None,
                                                            "Creator": None})
    fig.savefig(out_path(subdir, f"{name}.png"), dpi=300, metadata={"Software": None})
    plt.close(fig)


def add_caption(name, text):
    """Record a figure caption; figures_partA.py writes figures/CAPTIONS.md from these."""
    p = out_path("figures", "captions", f"{name}.txt")
    with open(p, "w") as f:
        f.write(text.strip() + "\n")


# ----------------------------------------------------------------------------- statistics

def rng():
    return np.random.default_rng(SEED)


def paired(a, b, gen, n_boot=BOOT_N):
    """Paired a - b (aligned arrays): n, mean, 95% bootstrap CI, wins (a > b), paired t p."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = a[ok] - b[ok]
    n = len(d)
    if n == 0:
        return dict(n=0, mean_diff=np.nan, ci_lo=np.nan, ci_hi=np.nan, wins=0, losses=0, p_t=np.nan)
    idx = gen.integers(0, n, size=(n_boot, n))
    m = d[idx].mean(axis=1)
    p = sps.ttest_rel(a[ok], b[ok]).pvalue if n > 1 and d.std() > 0 else (1.0 if n > 1 else np.nan)
    return dict(n=n, mean_diff=d.mean(), ci_lo=np.percentile(m, 2.5), ci_hi=np.percentile(m, 97.5),
                wins=int((d > 0).sum()), losses=int((d < 0).sum()), p_t=float(p))


def holm(p):
    p = np.asarray(p, dtype=float)
    adj = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    order = np.argsort(q, kind="stable")
    out = np.empty_like(q)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(q) - rank) * q[i])
        out[i] = min(1.0, running)
    adj[ok] = out
    return adj


def auc(pos, neg):
    """ROC AUC = P(score_pos > score_neg) + 0.5 P(tie) (Mann-Whitney)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = sps.rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


# ----------------------------------------------------------------------------- claims

class Claims:
    """Numbers quoted in FINDINGS_partA.md. Each module writes claims/<module>.json;
    report_partA.py merges them into claims.json."""

    def __init__(self, module):
        self.module = module
        self.d = {}

    def put(self, key, value, seed_set, n=None, desc="", unit=""):
        if isinstance(value, (np.floating, float)):
            value = None if math.isnan(float(value)) else float(value)
        elif isinstance(value, (np.integer,)):
            value = int(value)
        elif isinstance(value, (np.bool_,)):
            value = bool(value)
        self.d[f"{self.module}.{key}"] = dict(value=value, seed_set=seed_set,
                                              n=None if n is None else int(n), desc=desc, unit=unit)

    def save(self):
        p = out_path("claims", f"{self.module}.json")
        with open(p, "w") as f:
            json.dump(self.d, f, indent=1, sort_keys=True)
            f.write("\n")
