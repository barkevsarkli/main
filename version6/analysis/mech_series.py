"""Per-run, per-epoch diagnostic series and per-run events for the Part A mechanism analysis.

series(stage) -> one row per (run_id, epoch), epochs 0..40 (0 = initialisation), columns
  dead|<layer>|<group>    dead_units / n_units, group relu or leaky_relu
  ndead|<layer>|<group>   nearly_dead_units / n_units
  n|<layer>|<group>       n_units of that group (units = neurons, or channels in conv)
  sat|<layer>             tanh frac_sat (share of probe values with |h| > 0.99)
  zmax|<layer>            z_maxabs over the whole layer (max over groups)
  zp99|<layer>            z_p99 (group `all` in mixed layers)
  w|<layer>, dw|<layer>, clip|<layer>   w_norm, dw_rel, clip_frac (layer = conv, hidden1,
                          hidden2, output)
  lbs, npred, maxshare, maxprob, logit_std   output-layer probe statistics
  val, loss, train_acc    from the merged CSV (epoch_val_acc, epoch_loss, epoch_train_acc;
                          NaN at epoch 0)

events(stage) -> one row per run (definitions in the function docstrings).
"""
import functools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_io import _cache_dir, load_metrics, runs  # noqa: E402

GROUP_METRICS = ["dead_units", "nearly_dead_units", "n_units", "frac_sat", "z_maxabs", "z_p99"]
LAYER_METRICS = ["w_norm", "dw_rel", "clip_frac"]
OUTPUT_METRICS = {"logit_between_std": "lbs", "n_pred_classes": "npred", "max_class_share": "maxshare",
                  "mean_max_prob": "maxprob", "logit_std": "logit_std"}
RELU_GROUPS = ("relu", "leaky_relu")
EPOCHS = 40
LAYERS = ("conv", "hidden1", "hidden2")


def hidden_layers(arch):
    return {"mlp1": ["hidden1"], "mlp2": ["hidden1", "hidden2"], "cnn": ["conv", "hidden1"]}[arch]


def last_hidden(arch):
    return "hidden2" if arch == "mlp2" else "hidden1"


SERIES_VERSION = 2


@functools.lru_cache(maxsize=None)
def series(stage):
    """Wide per-epoch table (module docstring); cached as parquet next to the metric cache."""
    p = os.path.join(_cache_dir(stage), f"_series_v{SERIES_VERSION}.parquet")
    if os.path.exists(p):
        return pd.read_parquet(p)
    df = _build_series(stage)
    df.to_parquet(p + ".tmp", index=False)
    os.replace(p + ".tmp", p)
    return df


def _build_series(stage):
    st = load_metrics(stage, GROUP_METRICS + LAYER_METRICS + list(OUTPUT_METRICS))
    key = ["run_id", "epoch"]
    parts = []

    g = st[st.metric.isin(["dead_units", "nearly_dead_units", "n_units"]) & st.group.isin(RELU_GROUPS)]
    w = g.pivot_table(index=key + ["layer", "group"], columns="metric", values="value", aggfunc="first").reset_index()
    w["dead"] = w["dead_units"] / w["n_units"]
    w["ndead"] = w["nearly_dead_units"] / w["n_units"]
    for col in ("dead", "ndead", "n_units"):
        p = w.pivot_table(index=key, columns=["layer", "group"], values=col, aggfunc="first")
        p.columns = [f"{'n' if col == 'n_units' else col}|{a}|{b}" for a, b in p.columns]
        parts.append(p)

    s = st[(st.metric == "frac_sat") & (st.group == "tanh")]
    p = s.pivot_table(index=key, columns="layer", values="value", aggfunc="first")
    p.columns = [f"sat|{c}" for c in p.columns]
    parts.append(p)

    z = st[(st.metric == "z_maxabs") & (st.layer != "output")]
    p = z.groupby(key + ["layer"])["value"].max().unstack("layer")
    p.columns = [f"zmax|{c}" for c in p.columns]
    parts.append(p)

    q = st[(st.metric == "z_p99") & (st.layer != "output")].copy()
    q["pref"] = (q.group == "all").astype(int)
    q = q.sort_values("pref", kind="stable").groupby(key + ["layer"])["value"].last().unstack("layer")
    q.columns = [f"zp99|{c}" for c in q.columns]
    parts.append(q)

    for m, short in (("w_norm", "w"), ("dw_rel", "dw"), ("clip_frac", "clip")):
        t = st[(st.metric == m) & (st.group == "all")]
        p = t.pivot_table(index=key, columns="layer", values="value", aggfunc="first")
        p.columns = [f"{short}|{c}" for c in p.columns]
        parts.append(p)

    o = st[st.metric.isin(list(OUTPUT_METRICS)) & (st.layer == "output")]
    p = o.pivot_table(index=key, columns="metric", values="value", aggfunc="first").rename(columns=OUTPUT_METRICS)
    parts.append(p)

    df = pd.concat(parts, axis=1).sort_index().reset_index()
    # Curves from the merged CSV (epochs 1..40).
    r = runs(stage)
    rec = []
    for rid, v, l, a in zip(r.run_id, r.epoch_val_acc, r.epoch_loss, r.epoch_train_acc):
        e = np.arange(1, len(v) + 1)
        rec.append(pd.DataFrame({"run_id": rid, "epoch": e, "val": v, "loss": l, "train_acc": a}))
    curves = pd.concat(rec, ignore_index=True)
    df = df.merge(curves, on=key, how="left")
    return df.sort_values(key, kind="stable").reset_index(drop=True)


def first_epoch(e, mask):
    """First epoch where mask holds, NaN if never."""
    idx = np.flatnonzero(mask)
    return float(e[idx[0]]) if len(idx) else np.nan


def run_events(g, arch):
    """Events of one run (g: its series rows, sorted by epoch).

    e_star       first epoch with validation accuracy < 15% (as load.py).
    e_fail       first epoch e >= 2 with val(e) < max(val(1..e-1)) - 10 (pp).
    e_lossrise   first epoch e >= 2 with train loss(e) > 2 x min(loss(1..e-1)).
    e_loss_min   epoch of the minimum training loss.
    peak_before  max validation accuracy before e_star (NaN if e_star is 1 or missing).
    For every ReLU-type group (relu, leaky_relu) of every hidden layer:
      d0, dfinal, dmax   dead fraction at epoch 0, 40 and its maximum;
      e10, e50, e90      first epoch (>= 1) with dead fraction >= 0.1 / 0.5 / 0.9.
    """
    e = g["epoch"].to_numpy()
    v = g["val"].to_numpy()
    loss = g["loss"].to_numpy()
    out = {}
    ok = e >= 1
    ev, vv, lv = e[ok], v[ok], loss[ok]
    out["e_star"] = first_epoch(ev, vv < 15)
    runmax = np.maximum.accumulate(vv)
    prev_max = np.concatenate([[np.nan], runmax[:-1]])
    out["e_fail"] = first_epoch(ev, vv < prev_max - 10)
    runmin = np.minimum.accumulate(lv)
    prev_min = np.concatenate([[np.nan], runmin[:-1]])
    out["e_lossrise"] = first_epoch(ev, lv > 2 * prev_min)
    out["e_loss_min"] = float(ev[np.nanargmin(lv)])
    es = out["e_star"]
    out["peak_before"] = float(vv[ev < es].max()) if es == es and es > 1 else np.nan
    for L in hidden_layers(arch):
        for grp in RELU_GROUPS:
            c = f"dead|{L}|{grp}"
            if c not in g or g[c].isna().all():
                continue
            d = g[c].to_numpy()
            pre = f"{L}|{grp}"
            out[f"d0|{pre}"] = d[e == 0][0] if (e == 0).any() else np.nan
            out[f"dfinal|{pre}"] = d[e == EPOCHS][0] if (e == EPOCHS).any() else np.nan
            out[f"dmax|{pre}"] = np.nanmax(d[ok])
            for thr in (10, 50, 90):
                out[f"e{thr}|{pre}"] = first_epoch(ev, d[ok] >= thr / 100)
    return out


@functools.lru_cache(maxsize=None)
def events(stage):
    s = series(stage)
    r = runs(stage).set_index("run_id")
    rows = []
    for rid, g in s.groupby("run_id", sort=True):
        d = run_events(g, r.at[rid, "arch"])
        d["run_id"] = rid
        rows.append(d)
    ev = pd.DataFrame(rows)
    cols = ["run_id", "stage", "dataset", "arch", "arch_label", "config", "hidden", "model", "lr", "seed",
            "seed_set", "final_val", "best_val", "drop", "final_lt15", "drop_gt10", "ever_lt15", "outcome"]
    rr = runs(stage).copy()
    rr["stage"] = stage
    return rr[cols].merge(ev, on="run_id", how="left", validate="one_to_one")
