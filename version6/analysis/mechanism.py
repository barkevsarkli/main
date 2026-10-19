#!/usr/bin/env python3
"""A3: what collapses, where and when (Stages 0, 2 and 3; per-epoch diagnostics).

    python3 analysis/mechanism.py      -> analysis_out/partA/mechanism_tables/, figures/mech_*

Q1  pure ReLU collapse: which layer dies, how fast, and when relative to e* and the
    training-loss rise (event study aligned at e*).
Q2  tanh+relu: does the ReLU group die? Paired against pure ReLU (same cell and seed);
    regimes "protected" / "carried" (ReLU half dead, tanh carries) / "failed".
Q3  mask placement (Stage 2 cfg2, H 128/256): which layer's ReLU death precedes failure.
Q4  leaky-ReLU weight explosion: spike epoch, jump factors, spread, candidate predictors,
    and the loss-clamp check after the spike.
Q5  partial collapse (final validation accuracy in [15, 50)).
Q6  early warning: lead times and ROC AUC of four signals.

Definitions (also in mech_series.py): e* = first epoch with val < 15%; failure epoch =
first epoch with val more than 10 pp below its running maximum; loss rise = first epoch
with training loss > 2 x its running minimum; dead fraction = dead_units / n_units of a
ReLU-type group (a unit is dead when z <= 0 on all 1,000 probe inputs; conv units are
channels). Seed sets: Stage 0 seeds 1-10 and Stage 2 sweep seeds 1-10 are exploratory;
Stage 2 seeds 11-30 exist only at the selected, stress and below-stress LRs, and the
mechanism tables pool seeds 1-30 there (descriptive, labelled); Stage 3 seeds 31-45.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from mech_series import EPOCHS, events, hidden_layers, last_hidden, series  # noqa: E402
from partA_common import (COLORS, INK, MARKERS, MUTED, Claims, add_caption, auc, paired, rng,  # noqa: E402
                          savefig, style, write_table)
from stats_io import stage2_lrs  # noqa: E402

TAB = "mechanism_tables"
OUTCOMES = ["never_learned", "late_collapse", "late_degradation", "stable"]
FAILING = ("late_collapse", "late_degradation")
WINDOW = (-8, 3)
CLAMP = -np.log(1e-12)      # loss of a sample clamped at p = 1e-12 (src/train_util.h)
HYBRIDS = {0: ["tanh+relu"], 2: ["tanh+relu"], 3: ["tanh+relu@0.25", "tanh+relu", "tanh+relu@0.75"]}
C = Claims("mechanism")


# ----------------------------------------------------------------------------- helpers

def seed_label(stage, seeds):
    seeds = sorted(set(int(s) for s in seeds))
    if not seeds:
        return ""
    rng_ = f"{seeds[0]}-{seeds[-1]}" if seeds == list(range(seeds[0], seeds[-1] + 1)) else ",".join(map(str, seeds))
    tag = {0: "S0", 1: "S1", 2: "S2", 3: "S3"}[stage]
    kind = "exploratory" if stage in (0, 1) or (stage == 2 and seeds[-1] <= 10) else (
        "descriptive, pooled sweep+ext" if stage == 2 and seeds[0] <= 10 else (
            "confirmatory seeds" if stage == 2 else "ratio slice"))
    return f"{tag} seeds {rng_} ({kind})"


def lr_role(row):
    if row["stage"] != 2:
        return ""
    L = stage2_lrs(row["dataset"], row["arch_label"], row["hidden"])
    roles = []
    if np.isclose(row["lr"], L["stress"]):
        roles.append("stress")
    if L["below"] is not None and np.isclose(row["lr"], L["below"]):
        roles.append("below-stress")
    if np.isclose(row["lr"], L["selected"].get(row["model"], -1)):
        roles.append("selected")
    return "+".join(roles) if roles else "sweep"


def all_events():
    ev = pd.concat([events(s) for s in (0, 2, 3)], ignore_index=True)
    ev["lr_role"] = ev.apply(lr_role, axis=1)
    return ev


def med(x):
    x = pd.Series(x, dtype=float).dropna()
    return float(x.median()) if len(x) else np.nan


def q25(x):
    x = pd.Series(x, dtype=float).dropna()
    return float(x.quantile(0.25)) if len(x) else np.nan


def q75(x):
    x = pd.Series(x, dtype=float).dropna()
    return float(x.quantile(0.75)) if len(x) else np.nan


CELL = ["stage", "dataset", "arch_label", "hidden", "lr"]


def first_layer(arch):
    return hidden_layers(arch)[0]


# ----------------------------------------------------------------------------- Q1

def q1(ev):
    rel = ev[(ev.model == "relu") & ((ev.stage == 0) | ev.lr_role.str.contains("stress"))].copy()
    rows = []
    for key, g in rel.groupby(CELL, sort=True):
        arch = g.arch.iloc[0]
        lh, fl = last_hidden(arch), first_layer(arch)
        col = g[g.e_star.notna()]
        late = col[col.outcome != "never_learned"]          # fell below 15% after learning
        dur = late[f"e90|{lh}|relu"] - late[f"e10|{lh}|relu"]
        first = []
        for _, r in late.iterrows():
            e50 = {L: r.get(f"e50|{L}|relu", np.nan) for L in hidden_layers(arch)}
            e50 = {L: v for L, v in e50.items() if v == v}
            if not e50:
                first.append("none >=50% dead")
            else:
                m = min(e50.values())
                ws = [L for L, v in e50.items() if v == m]
                first.append(ws[0] if len(ws) == 1 else "same epoch")
        fc = pd.Series(first).value_counts().to_dict()
        rows.append(dict(zip(CELL, key)) | dict(
            seed_set=seed_label(key[0], g.seed), lr_role=g.lr_role.iloc[0], n=len(g),
            n_final_lt15=int(g.final_lt15.sum()), n_ever_lt15=int(col.shape[0]), n_late=len(late),
            n_late_recovered=int((~late.final_lt15).sum()),
            **{f"oc_{c}": int((g.outcome == c).sum()) for c in OUTCOMES},
            n_revived=int(((g[f"dmax|{lh}|relu"] >= 0.99) & (g[f"dfinal|{lh}|relu"] < 0.5)).sum()),
            e_star_med=med(col.e_star), e_lossrise_med=med(late.e_lossrise),
            lossrise_before_estar_med=med(late.e_star - late.e_lossrise),
            lossmin_before_estar_med=med(late.e_star - late.e_loss_min),
            last_layer=lh, dead_last_final_med=med(g[f"dfinal|{lh}|relu"]),
            e90_last_minus_estar_med=med(late[f"e90|{lh}|relu"] - late.e_star),
            dur10_90_last_med=med(dur), n_abrupt_le1=int((dur <= 1).sum()), n_reach90=int(dur.notna().sum()),
            first_layer=fl, dead_first_final_med=med(g[f"dfinal|{fl}|relu"]) if fl != lh else np.nan,
            first_to_50pct=";".join(f"{k}:{v}" for k, v in sorted(fc.items()))))
    t = pd.DataFrame(rows)
    write_table(t, TAB, "q1_relu_collapse",
                "Q1. Pure-ReLU collapse per cell (Stage 0 all cells; Stage 2 at the stress and below-stress LR, "
                "seeds 1-30 pooled). Late = runs with an e* that are not never_learned (outcomes.py). "
                "oc_* = outcome classes of all runs of the cell. Late recovered = late runs "
                "that end at >= 15%. Revived = the last hidden layer reached >= 99% dead (on the probe set) and "
                "ended < 50% dead. Medians over runs.")
    # claims: Stage 0 H128 LR 0.03 and pooled CNN cells
    s0 = t[(t.stage == 0) & (t.hidden == 128) & np.isclose(t.lr, 0.03)].iloc[0]
    for k in ("n", "n_late", "e_star_med", "lossrise_before_estar_med", "lossmin_before_estar_med",
              "dead_last_final_med", "dead_first_final_med", "dur10_90_last_med", "n_abrupt_le1", "n_reach90"):
        C.put(f"q1.s0_h128_lr0.03.{k}", s0[k], s0.seed_set, n=s0.n)
    late_all = rel[rel.e_star.notna() & (rel.outcome != "never_learned")].copy()
    late_all["lh"] = late_all.arch.map(last_hidden)
    late_all["dur"] = [r[f"e90|{r.lh}|relu"] - r[f"e10|{r.lh}|relu"] for _, r in late_all.iterrows()]
    late_all["grp"] = np.where(late_all.stage == 0, "s0_mlp2", late_all.dataset + "_" +
                               np.where(late_all.arch == "cnn", "cnn", late_all.arch))
    for grp, g in late_all.groupby("grp"):
        ss = "S0 seeds 1-10 (exploratory)" if grp.startswith("s0") else "S2 seeds 1-30 at stress/below-stress LR (descriptive)"
        C.put(f"q1.{grp}.n_late", len(g), ss, n=len(g))
        C.put(f"q1.{grp}.n_late_recovered", int((~g.final_lt15).sum()), ss, n=len(g),
              desc="runs that fell below 15% after learning (not never_learned) and end at >= 15%")
        C.put(f"q1.{grp}.share_abrupt_le1", float((g.dur <= 1).mean()) if g.dur.notna().any() else np.nan, ss,
              n=int(g.dur.notna().sum()), desc="share of late collapses whose last hidden layer goes 10%->90% dead in <=1 epoch")
        C.put(f"q1.{grp}.share_reach90", float(g.dur.notna().mean()), ss, n=len(g))
        C.put(f"q1.{grp}.lossrise_lead_med", med(g.e_star - g.e_lossrise), ss, n=len(g), unit="epochs")
        C.put(f"q1.{grp}.lossmin_lead_med", med(g.e_star - g.e_loss_min), ss, n=len(g), unit="epochs")
    return t, rel


def event_study(rel):
    s_all = {s: series(s).set_index("run_id") for s in (0, 2)}
    groups = [("Stage 0 mlp2, MNIST", (rel.stage == 0)),
              ("Stage 2 CNN, MNIST", (rel.stage == 2) & (rel.arch == "cnn") & (rel.dataset == "mnist")),
              ("Stage 2 CNN, CIFAR-10", (rel.stage == 2) & (rel.arch == "cnn") & (rel.dataset == "cifar10")),
              ("Stage 2 mlp1, CIFAR-10", (rel.stage == 2) & (rel.arch == "mlp1") & (rel.dataset == "cifar10"))]
    panels = [("val", "validation acc. (%)", False), ("loss", "training loss (online)", False),
              ("dead_first", "dead frac., first layer", False), ("dead_last", "dead frac., last hidden", False),
              ("lbs", "between-input logit std", True)]
    fig, axes = plt.subplots(len(groups), len(panels), figsize=(11, 2.0 * len(groups)), squeeze=False)
    rel_e = np.arange(WINDOW[0], WINDOW[1] + 1)
    summary = []
    for gi, (title, mask) in enumerate(groups):
        g = rel[mask & rel.e_star.notna() & (rel.e_star >= 2) & (rel.outcome == "late_collapse")]
        mats = {p[0]: [] for p in panels}
        for _, r in g.iterrows():
            s = s_all[r.stage].loc[[r.run_id]].set_index("epoch")
            fl, lh = first_layer(r.arch), last_hidden(r.arch)
            e = r.e_star + rel_e
            for key, _, _ in panels:
                col = {"dead_first": f"dead|{fl}|relu", "dead_last": f"dead|{lh}|relu"}.get(key, key)
                v = s[col].reindex(e).to_numpy(dtype=float) if col in s else np.full(len(e), np.nan)
                mats[key].append(v)
        for pi, (key, lab, logy) in enumerate(panels):
            ax = axes[gi, pi]
            M = np.array(mats[key]) if mats[key] else np.full((0, len(rel_e)), np.nan)
            if len(M):
                cnt = (~np.isnan(M)).sum(axis=0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN columns -> NaN
                    md = np.nanmedian(M, axis=0)
                    lo = np.nanpercentile(M, 25, axis=0)
                    hi = np.nanpercentile(M, 75, axis=0)
                ax.fill_between(rel_e, lo, hi, color=COLORS["relu"], alpha=0.2, lw=0)
                ax.plot(rel_e, md, color=COLORS["relu"], lw=1.5, marker="o", ms=3)
                for j, e in enumerate(rel_e):
                    summary.append(dict(group=title, metric=key, rel_epoch=int(e), n=int(cnt[j]),
                                        median=md[j], q25=lo[j], q75=hi[j]))
            ax.axvline(0, color=MUTED, lw=0.8, ls="--")
            if logy:
                ax.set_yscale("log")
            if gi == 0:
                ax.set_title(lab, loc="left", color=INK)
            if pi == 0:
                ax.set_ylabel(f"{title}\n(n={len(g)})", color=INK)
            if gi == len(groups) - 1:
                ax.set_xlabel("epoch relative to e* (first val < 15%)", color=MUTED)
            style(ax)
    fig.tight_layout()
    savefig(fig, "mech_q1_event_study")
    add_caption("mech_q1_event_study",
                "Pure-ReLU collapse aligned at e*, the first epoch with validation accuracy below 15% "
                "(dashed line), window -8..+3 epochs. Line: median across runs; band: interquartile range. "
                "Rows: Stage 0 mlp2 MNIST (seeds 1-10, all H and LR; exploratory) and Stage 2 at the stress and "
                "below-stress LR (seeds 1-30 pooled, descriptive) for the CNN (cfg1 and cfg2, all H) and mlp1. "
                "Only late_collapse runs (outcomes.py) are shown; runs with e* = 1 have no pre-collapse window and are "
                "excluded (n per row in the label). Runs that fall below 15% and recover are counted in "
                "q1_relu_collapse (n_late_recovered, n_revived). "
                "Dead fraction: share of ReLU units (channels in conv) with z <= 0 on all 1,000 probe inputs; "
                "first layer = hidden1 (mlp2) or conv (CNN).")
    write_table(pd.DataFrame(summary), TAB, "q1_event_study_values",
                "Values plotted in the Q1 event-study figure (median and IQR per relative epoch).")


# ----------------------------------------------------------------------------- Q1 follow-ups

SPEED_BINS = [("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4", 4, 4), ("5-9", 5, 9), ("10+", 10, 99)]


def collapse_speed(ev):
    """Epochs from 10% to 90% dead in the last hidden layer, and e90 - e*, for every pure-ReLU
    late_collapse run (all LRs), per (stage, dataset, arch, H)."""
    x = ev[(ev.model == "relu") & (ev.outcome == "late_collapse")].copy()
    x["lh"] = x.arch.map(last_hidden)
    x["e10"] = [r[f"e10|{r.lh}|relu"] for _, r in x.iterrows()]
    x["e90"] = [r[f"e90|{r.lh}|relu"] for _, r in x.iterrows()]
    x["dur"] = x.e90 - x.e10
    x["off"] = x.e90 - x.e_star
    rows = []
    for key, g in x.groupby(["stage", "dataset", "arch_label", "hidden"], sort=True):
        d = dict(zip(["stage", "dataset", "arch_label", "hidden"], key)) | dict(
            seed_set=seed_label(key[0], g.seed), lrs=" ".join(f"{v:g}" for v in sorted(g.lr.unique())),
            n=len(g), layer=g.lh.iloc[0])
        for lab, lo, hi in SPEED_BINS:
            d[f"dur_{lab}"] = int(((g.dur >= lo) & (g.dur <= hi)).sum())
        d["dur_no90"] = int(g.dur.isna().sum())
        d |= dict(dur_q25=q25(g.dur), dur_median=med(g.dur), dur_q75=q75(g.dur), dur_max=g.dur.max(),
                  off_median=med(g.off), off_q25=q25(g.off), off_q75=q75(g.off), off_min=g.off.min(), off_max=g.off.max(),
                  n_e90_before_estar=int((g.off < 0).sum()), n_e90_at_estar=int((g.off == 0).sum()),
                  n_e90_after_estar=int((g.off > 0).sum()))
        rows.append(d)
    t = pd.DataFrame(rows)
    write_table(t, TAB, "q1_collapse_speed",
                "Q1. Speed of ReLU death in pure-ReLU late collapses (outcomes.py), all LRs: number of runs by epochs "
                "from 10% to 90% dead in the last hidden layer (dur_*; no90 = never reached 90%), quantiles, and the "
                "offset e90 - e* (epoch reaching 90% dead minus first epoch with validation accuracy < 15%).")
    for _, q in t.iterrows():
        tag = f"q1.speed.s{q.stage}_{q.dataset}_{q.arch_label}_h{q.hidden}"
        for f in ("n", "dur_median", "dur_q25", "dur_q75", "dur_max", "dur_0", "dur_1", "off_median",
                  "n_e90_before_estar", "n_e90_at_estar", "n_e90_after_estar"):
            C.put(f"{tag}.{f}", q[f], q.seed_set, n=q.n)
    allx = x[x.dur.notna()]
    C.put("q1.speed.all.share_le1", float((allx.dur <= 1).mean()), "S0 + S2 seeds 1-30 + S3", n=len(allx))
    C.put("q1.speed.all.share_le3", float((allx.dur <= 3).mean()), "S0 + S2 seeds 1-30 + S3", n=len(allx))
    C.put("q1.speed.all.dur_median", med(allx.dur), "S0 + S2 seeds 1-30 + S3", n=len(allx))
    C.put("q1.speed.all.share_off_0", float((allx.off == 0).mean()), "S0 + S2 seeds 1-30 + S3", n=len(allx))
    return t


def revival(rel):
    """Pure-ReLU runs of the Q1 cells whose last hidden layer reached >= 99% dead on the probe
    set and ended < 50% dead. "Dead" = z <= 0 on all 1,000 probe (validation) inputs; units
    can still fire on training inputs. dw_rel(e) is the relative weight change during epoch e,
    so the epochs e_dead+1 .. e_revive are the ones in which the layer came back."""
    s2 = series(2).set_index("run_id")
    x = rel[rel.stage == 2].copy()
    x["lh"] = x.arch.map(last_hidden)
    rows = []
    for _, r in x.iterrows():
        g = s2.loc[[r.run_id]].set_index("epoch").loc[1:]
        dead = g[f"dead|{r.lh}|relu"]
        hit = dead.index[dead >= 0.99]
        if not len(hit):
            continue
        ed = int(hit[0])
        after = dead.loc[ed + 1:]
        rv = after.index[after < 0.5]
        came_back = bool(len(rv))
        revived = came_back and dead.loc[EPOCHS] < 0.5
        status = "revived" if revived else ("revived, then died again" if came_back else "stayed dead")
        er = int(rv[0]) if came_back else np.nan
        win = g.loc[ed + 1: er] if came_back else g.loc[ed + 1: ed + 1]
        rows.append(dict(run_id=r.run_id, dataset=r.dataset, arch_label=r.arch_label, hidden=r.hidden, lr=r.lr,
                         seed=r.seed, outcome=r.outcome, revived=revived, status=status, e_dead=ed, e_revive=er,
                         epochs_dead=(er - ed) if came_back else np.nan, val_at_dead=g.at[ed, "val"],
                         max_val_after=g.loc[ed + 1:, "val"].max() if ed < EPOCHS else np.nan,
                         final_val=r.final_val, reaches_15_after=bool(g.loc[ed + 1:, "val"].max() > 15) if ed < EPOCHS else False,
                         dw_conv=win["dw|conv"].median() if "dw|conv" in win else np.nan,
                         dw_hidden=win[f"dw|{r.lh}"].median(), dw_output=win["dw|output"].median(),
                         dw_conv_zero=bool((win["dw|conv"] == 0).all()) if "dw|conv" in win else np.nan,
                         w_conv_ratio=(g.at[er, "w|conv"] / g.at[ed, "w|conv"]) if came_back else
                         (g.at[min(ed + 1, EPOCHS), "w|conv"] / g.at[ed, "w|conv"])))
    t = pd.DataFrame(rows)
    write_table(t, TAB, "q1_revival_runs",
                "Q1. Pure-ReLU runs of the Q1 cells (Stage 2, stress and below-stress LR, seeds 1-30) whose last hidden "
                "layer reached >= 99% dead on the 1,000 probe inputs (e_dead). Revived = ended < 50% dead; e_revive = "
                "first later epoch < 50% dead (status 'revived, then died again' if it ends >= 50% dead). dw_* = "
                "median relative weight change in epochs e_dead+1..e_revive, or in epoch e_dead+1 for runs that "
                "stayed dead; w_conv_ratio = conv weight norm at e_revive (or "
                "e_dead+1) over e_dead.", digits=4)
    summ = (t.groupby(["dataset", "status"], sort=True)
            .agg(n=("run_id", "size"), n_reach_15_after=("reaches_15_after", "sum"),
                 epochs_dead_median=("epochs_dead", "median"), dw_conv_median=("dw_conv", "median"),
                 dw_hidden_median=("dw_hidden", "median"), dw_output_median=("dw_output", "median"),
                 n_dw_conv_zero=("dw_conv_zero", "sum"), w_conv_ratio_median=("w_conv_ratio", "median"),
                 final_val_median=("final_val", "median"))
            .reset_index())
    write_table(summ, TAB, "q1_revival_summary",
                "Q1. Revival summary (from q1_revival_runs): runs whose last hidden layer reached >= 99% dead on the "
                "probe set, by status. 'Dead' means z <= 0 on all 1,000 probe (validation) inputs, not on the "
                "training set: a non-zero dw of the layer or of the conv layer below it means some units still "
                "fired on training inputs. Stage 2 seeds 1-30 pooled (descriptive).", digits=4)
    rv = t[t.revived]
    nr = t[t.status == "stayed dead"]
    for (ds, stt), q in summ.set_index(["dataset", "status"]).iterrows():
        tag = f"q1.revival.{ds}.{stt.replace(' ', '_').replace(',', '')}"
        for f in ("n", "dw_conv_median", "dw_hidden_median", "n_dw_conv_zero", "n_reach_15_after"):
            C.put(f"{tag}.{f}", q[f], "S2 seeds 1-30 at stress/below-stress LR (descriptive)", n=q.n)
    ss = "S2 seeds 1-30 at stress/below-stress LR (descriptive)"
    C.put("q1.revival.n_revived", len(rv), ss, n=len(t))
    C.put("q1.revival.n_revived_reach15", int(rv.reaches_15_after.sum()), ss, n=len(rv))
    C.put("q1.revival.n_revived_final_ge15", int((rv.final_val >= 15).sum()), ss, n=len(rv))
    C.put("q1.revival.epochs_dead_median", med(rv.epochs_dead), ss, n=len(rv))
    C.put("q1.revival.dw_conv_median_revived", med(rv.dw_conv), ss, n=len(rv))
    C.put("q1.revival.dw_conv_median_not_revived", med(nr.dw_conv), ss, n=len(nr))
    C.put("q1.revival.n_dw_conv_zero_revived", int(rv.dw_conv_zero.sum()), ss, n=len(rv))
    C.put("q1.revival.n_dw_conv_zero_not_revived", int(nr.dw_conv_zero.sum()), ss, n=len(nr))
    return t, summ


# ----------------------------------------------------------------------------- post-hoc choices

def posthoc(ev, rel, sp):
    """Each analysis choice made after looking at the data, with the result under the original
    definition where it is cheap to compute."""
    rows = []
    # 1. Q4 spread: spike epoch >= 2 (used) vs >= 1 (original).
    for key, g in sp[sp.spiked].groupby(["stage", "dataset", "arch_label"], sort=True):
        g2 = g[g.spike_epoch >= 2]
        a = spread_row(g2) if len(g2) else {}
        b = spread_row(g)
        rows.append(dict(choice="Q4 spike-epoch spread: spikes at epoch >= 2 only",
                         reason="a spike in epoch 1 is growth from the initialisation, with no epoch before it",
                         measure=f"ICC S{key[0]} {key[1]} {key[2]}", value_used=a.get("icc", np.nan),
                         value_original=b["icc"], n_used=a.get("n_spiking_runs", 0), n_original=b["n_spiking_runs"]))
    # 2. Q6 zmax_jump: from epoch 2 (used) vs epoch 1 (original).
    lead1, P1, N1 = q6_collect(ev, zmax_from=1)
    lead2, P2, N2 = q6_collect(ev, zmax_from=2)
    lt1, au1 = q6_summaries(lead1, P1, N1, ["zmax_jump"])
    lt2, au2 = q6_summaries(lead2, P2, N2, ["zmax_jump"])
    for st in (0, 2, 3):
        for meas, t1, t2, col in (("median lead", lt1, lt2, "lead_median"), ("AUC", au1, au2, "auc")):
            rows.append(dict(choice="Q6 zmax_jump signal from epoch 2",
                             reason="the max |z| always more than doubles in epoch 1 (growth from the initialisation)",
                             measure=f"{meas} S{st}", value_used=float(t2[t2.stage == st][col].iloc[0]),
                             value_original=float(t1[t1.stage == st][col].iloc[0]), n_used=np.nan, n_original=np.nan))
    # 3. Q1 event-study filter: late_collapse runs only (used) vs every run with e* >= 2 (original).
    used = rel[rel.e_star.notna() & (rel.e_star >= 2) & (rel.outcome == "late_collapse")]
    orig = rel[rel.e_star.notna() & (rel.e_star >= 2)]
    rows.append(dict(choice="Q1 event study: late_collapse runs only",
                     reason="runs that dip below 15% and recover made the median curve rise after e*",
                     measure="runs in the event study", value_used=len(used), value_original=len(orig),
                     n_used=len(used), n_original=len(orig)))
    # 4. Q5 partial collapse: late_degradation in [15, 50) (used) vs final in [15, 50) (original).
    band = ev[(ev.final_val >= 15) & (ev.final_val < 50)]
    rows.append(dict(choice="Q5 partial collapse = late_degradation with final < 50%",
                     reason="stable CIFAR-10 runs also end between 15% and 50%",
                     measure="runs", value_used=int((band.outcome == "late_degradation").sum()), value_original=len(band),
                     n_used=np.nan, n_original=np.nan))
    # 5. Q6 failing runs: taxonomy (used) vs drop > 10 pp (original).
    rows.append(dict(choice="Q6 failing = late_collapse or late_degradation",
                     reason="the outcome taxonomy; drop > 10 pp also counted never_learned runs",
                     measure="failing runs", value_used=int(ev.outcome.isin(FAILING).sum()),
                     value_original=int(ev.drop_gt10.sum()), n_used=np.nan, n_original=np.nan))
    t = pd.DataFrame(rows)
    write_table(t, TAB, "posthoc_choices",
                "Post-hoc analysis choices: each definition changed after looking at the data, the reason, and the "
                "result under the definition used and under the original one.")
    for i, q in t.iterrows():
        C.put(f"posthoc.{i}.value_used", q.value_used, "see table", desc=f"{q.choice}: {q.measure}")
        C.put(f"posthoc.{i}.value_original", q.value_original, "see table", desc=f"{q.choice}: {q.measure}")
    return t


# ----------------------------------------------------------------------------- Q2

def regime(r, lh):
    if r.final_val < 15:
        return "failed"
    d = r.get(f"dfinal|{lh}|relu", np.nan)
    return "carried" if d >= 0.5 else "protected"


def q2(ev):
    sat = {s: series(s)[["run_id", "epoch"] + [c for c in series(s).columns if c.startswith("sat|")]]
           for s in (0, 2, 3)}
    sat_final = {s: v[v.epoch == EPOCHS].set_index("run_id") for s, v in sat.items()}
    gen = rng()
    rows, regs = [], []
    for stage in (0, 2, 3):
        e = ev[ev.stage == stage]
        for key, g in e.groupby(CELL[1:], sort=True):
            arch = g.arch.iloc[0]
            relu = g[g.model == "relu"].set_index("seed")
            tanh = g[g.model == "tanh"]
            for hyb in HYBRIDS[stage]:
                h = g[g.model == hyb].set_index("seed")
                if h.empty:
                    continue
                seeds = sorted(set(h.index) & set(relu.index))
                lh = last_hidden(arch)
                for L in hidden_layers(arch):
                    col = f"dfinal|{L}|relu"
                    if col not in h or h[col].isna().all():
                        continue
                    pr = paired(h.loc[seeds, col], relu.loc[seeds, col], gen) if seeds else paired([], [], gen)
                    sat_h = sat_final[stage].reindex(h.run_id)[f"sat|{L}"] if f"sat|{L}" in sat_final[stage] else pd.Series(dtype=float)
                    sat_t = sat_final[stage].reindex(tanh.run_id)[f"sat|{L}"] if f"sat|{L}" in sat_final[stage] else pd.Series(dtype=float)
                    rows.append(dict(stage=stage, **dict(zip(CELL[1:], key)), hybrid=hyb, layer=L,
                                     seed_set=seed_label(stage, h.index), lr_role=g.lr_role.iloc[0] if stage == 2 else "",
                                     hybrid_relu_dead_final=h[col].mean(), relu_dead_final=relu[col].mean() if len(relu) else np.nan,
                                     hybrid_relu_dead_max=h[f"dmax|{L}|relu"].mean(),
                                     relu_dead_max=relu[f"dmax|{L}|relu"].mean() if len(relu) else np.nan,
                                     **{f"diff_{k}": v for k, v in pr.items()},
                                     hybrid_tanh_sat_final=float(sat_h.mean()) if len(sat_h) else np.nan,
                                     tanh_sat_final=float(sat_t.mean()) if len(sat_t) else np.nan))
                cnt = h.apply(lambda r: regime(r, lh), axis=1).value_counts()
                regs.append(dict(stage=stage, **dict(zip(CELL[1:], key)), hybrid=hyb, seed_set=seed_label(stage, h.index),
                                 n=len(h), protected=int(cnt.get("protected", 0)), carried=int(cnt.get("carried", 0)),
                                 failed=int(cnt.get("failed", 0)), last_layer=lh,
                                 hybrid_relu_dead_last_med=med(h[f"dfinal|{lh}|relu"]),
                                 hybrid_final_val=h.final_val.mean(),
                                 relu_final_val=relu.final_val.mean() if len(relu) else np.nan,
                                 relu_n_lt15=int(relu.final_lt15.sum()) if len(relu) else np.nan))
    t = pd.DataFrame(rows).rename(columns={"diff_n": "n_pairs"})
    write_table(t, TAB, "q2_hybrid_relu_dead_paired",
                "Q2. Dead fraction of the ReLU units in tanh+relu versus pure ReLU in the same layer, cell and seed "
                "(final epoch). diff = hybrid - relu: mean, 95% paired bootstrap CI (10,000 resamples), wins = seeds "
                "where the hybrid has MORE dead ReLU units, losses = fewer. tanh frac_sat: share of probe values with "
                "|h| > 0.99 in the tanh units.")
    r = pd.DataFrame(regs)
    # Regime switch: smallest LR where most hybrid runs have the last hidden layer's ReLU half >= 50% dead.
    sw = []
    for key, g in r.groupby(["stage", "dataset", "arch_label", "hidden", "hybrid"], sort=True):
        g = g.sort_values("lr")
        carried_major = g[g.carried > g.n / 2]
        lr_c = carried_major.lr.min() if len(carried_major) else np.nan
        lr_dead = g[g.hybrid_relu_dead_last_med >= 0.5].lr.min() if (g.hybrid_relu_dead_last_med >= 0.5).any() else np.nan
        tested = sorted(g.lr)
        sw.append(dict(zip(["stage", "dataset", "arch_label", "hidden", "hybrid"], key)) | dict(
            lrs_tested=" ".join(f"{x:g}" for x in tested),
            lr_switch_relu_half_dead=lr_dead, lr_majority_carried=lr_c,
            regime_at_switch=("" if lr_dead != lr_dead else
                              g[g.lr == lr_dead].apply(lambda q: "carried" if q.carried >= q.failed else "failed", axis=1).iloc[0])))
    sw = pd.DataFrame(sw)
    write_table(r, TAB, "q2_hybrid_regimes",
                "Q2. Regimes of tanh+relu runs: protected = final val >= 15% and the last hidden layer's ReLU units "
                "< 50% dead; carried = final val >= 15% with >= 50% of them dead (the tanh units carry the network); "
                "failed = final val < 15%.")
    write_table(sw, TAB, "q2_regime_switch",
                "Q2. Smallest tested LR at which the median tanh+relu run has >= 50% of its last-hidden-layer ReLU "
                "units dead (lr_switch_relu_half_dead), and the regime of the majority there.")
    x = t[(t.stage == 0) & (t.hidden == 128) & (t.layer == "hidden2")]
    for lr in (0.02, 0.03, 0.05):
        q = x[np.isclose(x.lr, lr)].iloc[0]
        C.put(f"q2.s0_h128_lr{lr:g}.hybrid_relu_dead_hidden2", q.hybrid_relu_dead_final, q.seed_set, n=q.n_pairs)
        C.put(f"q2.s0_h128_lr{lr:g}.relu_dead_hidden2", q.relu_dead_final, q.seed_set, n=q.n_pairs)
        C.put(f"q2.s0_h128_lr{lr:g}.hybrid_tanh_sat_hidden2", q.hybrid_tanh_sat_final, q.seed_set, n=q.n_pairs)
    for _, q in sw.iterrows():
        C.put(f"q2.switch.s{q.stage}_{q.dataset}_{q.arch_label}_h{q.hidden}_{q.hybrid}", q.lr_switch_relu_half_dead,
              f"S{q.stage}", desc=f"regime at switch: {q.regime_at_switch}")
    return t, r, sw


# ----------------------------------------------------------------------------- Q3

def q3(ev):
    e = ev[(ev.stage == 2) & (ev.arch_label == "cfg2") & ev.hidden.isin([128, 256]) &
           ev.model.isin(["relu", "tanh+relu", "tanh+relu@conv", "tanh+relu@hidden"])]
    rows, order = [], []
    for key, g in e.groupby(["dataset", "hidden", "lr", "model"], sort=True):
        fail = g[g.drop_gt10 | g.final_lt15]
        rows.append(dict(zip(["dataset", "hidden", "lr", "model"], key)) | dict(
            seed_set=seed_label(2, g.seed), lr_role=g.lr_role.iloc[0], n=len(g), n_final_lt15=int(g.final_lt15.sum()),
            n_drop_gt10=int(g.drop_gt10.sum()), final_val=g.final_val.mean(),
            conv_relu_dead_final=g["dfinal|conv|relu"].mean(), hidden_relu_dead_final=g["dfinal|hidden1|relu"].mean(),
            conv_relu_dead_max=g["dmax|conv|relu"].mean(), hidden_relu_dead_max=g["dmax|hidden1|relu"].mean()))
        for _, r in fail.iterrows():
            ef = r.e_fail if r.e_fail == r.e_fail else r.e_star
            c50, h50 = r["e50|conv|relu"], r["e50|hidden1|relu"]
            if c50 != c50 and h50 != h50:
                lab = "neither layer >=50% dead"
            elif h50 != h50 or (c50 == c50 and c50 < h50):
                lab = "conv first"
            elif c50 != c50 or h50 < c50:
                lab = "hidden first"
            else:
                lab = "same epoch"
            first = np.nanmin([c50, h50]) if not (c50 != c50 and h50 != h50) else np.nan
            order.append(dict(zip(["dataset", "hidden", "lr", "model"], key)) | dict(
                seed=r.seed, failure_epoch=ef, conv_e50=c50, hidden_e50=h50, order=lab,
                death_before_or_at_failure=bool(first == first and ef == ef and first <= ef)))
    t = pd.DataFrame(rows)
    o = pd.DataFrame(order)
    write_table(t, TAB, "q3_mask_placement",
                "Q3. Mask placement, Stage 2 cfg2 H 128/256: failures and final dead fractions of the ReLU units in "
                "the conv layer (channels) and the hidden layer. Seeds 1-30 at the stress, below-stress and selected "
                "LRs, 1-10 elsewhere.")
    if not o.empty:
        oc = (o.groupby(["dataset", "model", "order"]).size().rename("n_runs").reset_index())
        pre = o.groupby(["dataset", "model"]).death_before_or_at_failure.agg(["sum", "size"]).reset_index()
        pre.columns = ["dataset", "model", "n_death_at_or_before_failure", "n_failing"]
        oc = oc.merge(pre, on=["dataset", "model"])
        write_table(oc, TAB, "q3_failure_order",
                    "Q3. Failing runs (drop > 10 pp or final < 15%), Stage 2 cfg2 H 128/256, all LRs pooled: which "
                    "layer's ReLU units first reach 50% dead, and whether that happens at or before the failure epoch.")
    for ds in ("mnist", "cifar10"):
        for H in (128, 256):
            for m in ("tanh+relu@conv", "tanh+relu@hidden", "tanh+relu", "relu"):
                q = t[(t.dataset == ds) & (t.hidden == H) & np.isclose(t.lr, 0.02) & (t.model == m)]
                if len(q):
                    q = q.iloc[0]
                    C.put(f"q3.{ds}_h{H}_lr0.02.{m}.n_final_lt15", q.n_final_lt15, q.seed_set, n=q.n)
                    C.put(f"q3.{ds}_h{H}_lr0.02.{m}.conv_relu_dead", q.conv_relu_dead_final, q.seed_set, n=q.n)
                    C.put(f"q3.{ds}_h{H}_lr0.02.{m}.hidden_relu_dead", q.hidden_relu_dead_final, q.seed_set, n=q.n)
    return t, o


# ----------------------------------------------------------------------------- Q4

def spike_table(stage, ev):
    s = series(stage)
    lk = ev[(ev.stage == stage) & (ev.model == "leaky_relu")]
    s = s[s.run_id.isin(lk.run_id)]
    info = lk.set_index("run_id")
    rows = []
    for rid, g in s.groupby("run_id", sort=True):
        g = g.set_index("epoch").sort_index()
        r = info.loc[rid]
        arch = r.arch
        layers = hidden_layers(arch) + ["output"]
        ratio = g["w|output"] / g["w|output"].shift(1)
        ratio = ratio.loc[1:]
        es = int(ratio.idxmax())
        jump = float(ratio.max())
        d = dict(run_id=rid, stage=stage, dataset=r.dataset, arch=arch, arch_label=r.arch_label, hidden=r.hidden,
                 lr=r.lr, seed=r.seed, final_val=r.final_val, best_val=r.best_val, spike_epoch=es, jump_output=jump,
                 spiked=jump >= 10)
        for L in layers:
            d[f"jump_{L}"] = float(g.at[es, f"w|{L}"] / g.at[es - 1, f"w|{L}"])
            d[f"w_{L}_before"] = float(g.at[es - 1, f"w|{L}"])
            d[f"clip_{L}_before"] = float(g.at[es - 1, f"clip|{L}"]) if es - 1 >= 1 else np.nan
            d[f"clip_{L}_at"] = float(g.at[es, f"clip|{L}"])
        for L in hidden_layers(arch):
            d[f"zmax_{L}_before"] = float(g.at[es - 1, f"zmax|{L}"])
            d[f"dead_{L}_at"] = float(g.at[es, f"dead|{L}|leaky_relu"])
            d[f"dead_{L}_max"] = float(g[f"dead|{L}|leaky_relu"].loc[1:].max())
        d["maxprob_before"] = float(g.at[es - 1, "maxprob"])
        d["maxprob_at"] = float(g.at[es, "maxprob"])
        d["maxprob_final"] = float(g.at[EPOCHS, "maxprob"])
        d["lbs_at"] = float(g.at[es, "lbs"])
        d["logit_std_at"] = float(g.at[es, "logit_std"])
        d["w_prod_before"] = float(np.prod([g.at[es - 1, f"w|{L}"] for L in layers]))
        d["lr_w_output2_before"] = r.lr * d["w_output_before"] ** 2
        lh = last_hidden(arch)
        d["lr_w_last2_before"] = r.lr * d[f"w_{lh}_before"] ** 2
        d["lr_zmax_last_before"] = r.lr * d[f"zmax_{lh}_before"]
        rows.append(d)
    return pd.DataFrame(rows)


def cv(x):
    x = pd.Series(x, dtype=float).dropna()
    return float(x.std(ddof=1) / abs(x.mean())) if len(x) > 1 and x.mean() != 0 else np.nan


def spread_row(g):
    """Spike-epoch spread of one (stage, dataset, arch) group: pooled within-cell SD over cells
    (H, LR) with >= 2 spiking runs, SD of the cell means, ICC = between / (between + within)."""
    by = g.groupby(["hidden", "lr"])
    sizes = by.size()
    valid = sizes[sizes >= 2].index
    gv = g.set_index(["hidden", "lr"]).loc[valid].reset_index() if len(valid) else g.iloc[0:0]
    within = np.sqrt(gv.groupby(["hidden", "lr"]).spike_epoch.var(ddof=1).mean()) if len(valid) else np.nan
    means = by.spike_epoch.mean()
    between = means.std(ddof=1) if len(means) > 1 else np.nan
    vb, vw = (between ** 2 if between == between else np.nan), (within ** 2 if within == within else np.nan)
    return dict(n_cells=len(means), n_spiking_runs=len(g), within_cell_sd=within, between_cell_sd_of_means=between,
                icc=vb / (vb + vw) if vb == vb and vw == vw and vb + vw > 0 else np.nan,
                within_cell_range_max=float(by.spike_epoch.agg(lambda x: x.max() - x.min()).max()))


def q4(ev):
    sp = pd.concat([spike_table(0, ev), spike_table(2, ev)], ignore_index=True)
    cells = []
    for key, g in sp.groupby(CELL, sort=True):
        k = g[g.spiked]
        cells.append(dict(zip(CELL, key)) | dict(
            seed_set=seed_label(key[0], g.seed), n=len(g), n_spike=len(k),
            spike_epoch_med=med(k.spike_epoch), spike_epoch_min=k.spike_epoch.min() if len(k) else np.nan,
            spike_epoch_max=k.spike_epoch.max() if len(k) else np.nan,
            jump_output_med=med(k.jump_output), jump_hidden1_med=med(k.jump_hidden1),
            jump_last_med=med([r[f"jump_{last_hidden(r.arch)}"] for _, r in k.iterrows()]),
            maxprob_before_med=med(k.maxprob_before), maxprob_at_med=med(k.maxprob_at),
            clip_output_before_med=med(k.clip_output_before), clip_output_at_med=med(k.clip_output_at),
            leaky_dead_max=float(np.nanmax([r[f"dead_{last_hidden(r.arch)}_max"] for _, r in g.iterrows()])),
            final_val_mean=g.final_val.mean(), best_val_mean=g.best_val.mean()))
    cells = pd.DataFrame(cells)
    write_table(cells, TAB, "q4_leaky_spikes",
                "Q4. Leaky-ReLU weight explosion. Spike epoch = epoch with the largest one-epoch growth of the output "
                "layer's weight norm; a run has a spike if that growth is >= 10x. Jump = w_norm(spike)/w_norm(spike-1). "
                "maxprob = mean max softmax probability on the probe set; clip = share of clipped dL/dz.")
    write_table(sp, TAB, "q4_leaky_runs", "Q4. Per-run leaky-ReLU spike diagnostics (all leaky runs of Stages 0 and 2).")
    # Spread of the spike epoch and candidate predictors: spiking runs with spike epoch >= 2
    # (a spike in epoch 1 is growth from the initialisation, with nothing measured before it).
    k = sp[sp.spiked & (sp.spike_epoch >= 2)].copy()
    spread, pred = [], []
    for key, g in k.groupby(["stage", "dataset", "arch_label"], sort=True):
        by = g.groupby(["hidden", "lr"])
        sizes = by.size()
        valid = sizes[sizes >= 2].index
        gv = g.set_index(["hidden", "lr"]).loc[valid].reset_index() if len(valid) else g.iloc[0:0]
        means = by.spike_epoch.mean()
        spread.append(dict(zip(["stage", "dataset", "arch_label"], key)) | spread_row(g))
        lh = last_hidden(g.arch.iloc[0])
        cands = [f"w_{L}_before" for L in hidden_layers(g.arch.iloc[0]) + ["output"]] + [
            "w_prod_before", "lr_w_output2_before", "lr_w_last2_before", f"zmax_{lh}_before", "lr_zmax_last_before"]
        if g.arch.iloc[0] in ("mlp2", "cnn"):
            cands.insert(len(hidden_layers(g.arch.iloc[0])), f"zmax_{hidden_layers(g.arch.iloc[0])[0]}_before")
        for c in dict.fromkeys(cands):
            w_cv = [cv(x) for _, x in gv.groupby(["hidden", "lr"])[c]] if len(valid) else []
            pred.append(dict(zip(["stage", "dataset", "arch_label"], key)) | dict(
                predictor=c.replace("_before", " at spike-1"), n_runs=len(g), n_cells=len(means),
                cv_within_cells_mean=float(np.nanmean(w_cv)) if w_cv else np.nan,
                cv_across_cell_means=cv(g.groupby(["hidden", "lr"])[c].mean()), cv_all_runs=cv(g[c])))
    spread, pred = pd.DataFrame(spread), pd.DataFrame(pred)
    write_table(spread, TAB, "q4_spike_epoch_spread",
                "Q4. Spread of the spike epoch among spiking leaky-ReLU runs with spike epoch >= 2: pooled within-cell SD (cells = H x LR "
                "with >= 2 spiking runs), SD of the cell means, and ICC = between / (between + within) variance.")
    write_table(pred, TAB, "q4_spike_predictors",
                "Q4. Candidate threshold predictors measured at the epoch before the spike (spike epoch >= 2): coefficient of variation "
                "within cells (mean over cells), across cell means, and across all spiking runs. A quantity that "
                "triggers the spike at a fixed threshold would have a small CV across all runs.")
    for _, q in cells[cells.stage == 0].iterrows():
        tag = f"q4.s0_h{q.hidden}_lr{q.lr:g}"
        for f in ("n_spike", "spike_epoch_min", "spike_epoch_max", "jump_output_med", "maxprob_at_med",
                  "leaky_dead_max", "final_val_mean"):
            C.put(f"{tag}.{f}", q[f], q.seed_set, n=q.n)
    for _, q in spread.iterrows():
        tag = f"q4.spread.s{q.stage}_{q.dataset}_{q.arch_label}"
        for f in ("within_cell_sd", "between_cell_sd_of_means", "icc", "n_spiking_runs", "n_cells"):
            C.put(f"{tag}.{f}", q[f], f"S{q.stage}")
    for _, q in pred.iterrows():
        C.put(f"q4.pred.s{q.stage}_{q.dataset}_{q.arch_label}.{q.predictor.replace(' ', '_')}.cv_all",
              q.cv_all_runs, f"S{q.stage}", n=q.n_runs)
    s2 = sp[sp.stage == 2]
    C.put("q4.s2.n_leaky_runs", len(s2), "S2 seeds 1-30", n=len(s2))
    C.put("q4.s2.n_spiked", int(s2.spiked.sum()), "S2 seeds 1-30", n=len(s2))
    C.put("q4.n_spike_at_epoch1", int((sp.spiked & (sp.spike_epoch == 1)).sum()), "S0 + S2", n=int(sp.spiked.sum()))
    clamp_check(sp)
    spike_recovery(sp)
    return sp, cells, spread, pred


def spike_recovery(sp):
    """Validation and training accuracy at spike-1, at the spike and at epoch 40 (spike epoch >= 2)."""
    rows = []
    for stage in (0, 2):
        k = sp[(sp.stage == stage) & sp.spiked & (sp.spike_epoch >= 2)]
        s = series(stage).set_index(["run_id", "epoch"])
        for _, r in k.iterrows():
            e = int(r.spike_epoch)
            g = s.loc[r.run_id]
            rows.append(dict(stage=stage, dataset=r.dataset, arch_label=r.arch_label, lr=r.lr, seed=r.seed,
                             spike_epoch=e, val_before=g.at[e - 1, "val"], val_at=g.at[e, "val"],
                             val_final=g.at[EPOCHS, "val"], train_before=g.at[e - 1, "train_acc"],
                             train_at=g.at[e, "train_acc"], train_final=g.at[EPOCHS, "train_acc"],
                             lbs_at=g.at[e, "lbs"]))
    d = pd.DataFrame(rows)
    t = (d.groupby(["stage", "dataset", "arch_label", "lr"], sort=True)
         .agg(n=("seed", "size"), spike_epoch_med=("spike_epoch", "median"), val_before=("val_before", "median"),
              val_at=("val_at", "median"), val_final=("val_final", "median"), train_before=("train_before", "median"),
              train_at=("train_at", "median"), train_final=("train_final", "median"),
              logit_between_std_at=("lbs_at", "median"))
         .reset_index())
    t.insert(4, "seed_set", [seed_label(st, d[d.stage == st].seed) for st in t.stage])
    write_table(t, TAB, "q4_spike_recovery",
                "Q4. Spiking leaky-ReLU runs (spike epoch >= 2): validation and training accuracy (%) at the epoch "
                "before the spike, at the spike and at epoch 40, and the between-input logit std at the spike. "
                "Medians over runs. Stage 2 pools H 64/128/256 and seeds 1-30.")
    for _, q in t.iterrows():
        tag = f"q4.recovery.s{q.stage}_{q.dataset}_{q.arch_label}_lr{q.lr:g}"
        for f in ("val_before", "val_at", "val_final", "train_at", "train_final", "spike_epoch_med"):
            C.put(f"{tag}.{f}", q[f], q.seed_set, n=q.n)
    return t


def clamp_check(sp):
    """After a spike, is the epoch training loss exactly the clamp value times the training error rate?

    With loss = -log(max(p, 1e-12)), a sample whose correct-class probability is below 1e-12 costs
    exactly CLAMP = -ln(1e-12) = 27.631; a ratio loss / (CLAMP x error rate) of 1 means every
    misclassified training sample is clamped and every correct one costs ~0.
    """
    rows = []
    for stage in (0, 2):
        k = sp[(sp.stage == stage) & sp.spiked]
        x = series(stage).merge(k[["run_id", "dataset", "arch_label", "lr", "spike_epoch"]], on="run_id")
        x = x[(x.epoch > x.spike_epoch) & (x.train_acc < 100)]
        x = x.assign(ratio=x.loss / (CLAMP * (1 - x.train_acc / 100)))
        for key, g in x.groupby(["dataset", "arch_label", "lr"], sort=True):
            dev = (g.ratio - 1).abs()
            rows.append(dict(stage=stage, dataset=key[0], arch_label=key[1], lr=key[2],
                             seed_set=seed_label(stage, sp[sp.run_id.isin(g.run_id)].seed), n_runs=g.run_id.nunique(),
                             n_epochs=len(g), ratio_median=g.ratio.median(), ratio_min=g.ratio.min(),
                             ratio_max=g.ratio.max(), max_abs_dev=dev.max()))
    t = pd.DataFrame(rows)
    write_table(t, TAB, "q4_clamp_check",
                "Q4. Spiked leaky-ReLU runs, every epoch after the spike: training loss divided by "
                "-ln(1e-12) x training error rate. A ratio of 1 means the loss is set by the probability clamp "
                "(every misclassified training sample has p < 1e-12, every correct one costs ~0).", digits=6)
    C.put("q4.clamp.n_runs", int(t.n_runs.sum()), "S0 + S2", desc="spiked leaky runs checked")
    C.put("q4.clamp.n_epochs", int(t.n_epochs.sum()), "S0 + S2")
    C.put("q4.clamp.max_abs_dev", float(t.max_abs_dev.max()), "S0 + S2", desc="max |ratio - 1| over all post-spike epochs")
    return t


# ----------------------------------------------------------------------------- Q5

def q5(ev):
    fin = {s: series(s)[series(s).epoch == EPOCHS].set_index("run_id") for s in (0, 2, 3)}
    # CIFAR-10 runs that train normally also end below 50%, so a run counts as a partial
    # collapse only if it also lost more than 10 pp from its peak; n_band_only counts the rest.
    band = ev[(ev.final_val >= 15) & (ev.final_val < 50)]
    p = band[band.outcome == "late_degradation"]
    rows = []
    for key, g in p.groupby(["stage", "dataset", "arch_label", "hidden", "model", "lr"], sort=True):
        f = fin[key[0]].reindex(g.run_id)
        total = ev[(ev.stage == key[0]) & (ev.dataset == key[1]) & (ev.arch_label == key[2]) & (ev.hidden == key[3]) &
                   (ev.model == key[4]) & np.isclose(ev.lr, key[5])]
        d = dict(zip(["stage", "dataset", "arch_label", "hidden", "model", "lr"], key)) | dict(
            seed_set=seed_label(key[0], g.seed), n_partial=len(g), n_cell=len(total),
            n_band_only=int(((total.final_val >= 15) & (total.final_val < 50) &
                             (total.outcome != "late_degradation")).sum()),
            final_val_mean=g.final_val.mean(), best_val_mean=g.best_val.mean(),
            n_pred_classes_med=med(f.npred), max_class_share_med=med(f.maxshare))
        for L in hidden_layers(g.arch.iloc[0]):
            for grp in ("relu", "leaky_relu"):
                c = f"dead|{L}|{grp}"
                if c in f and f[c].notna().any():
                    d[f"{L}_{grp}_dead"] = f[c].mean()
                    d[f"{L}_{grp}_alive_units"] = float(((1 - f[c]) * f[f"n|{L}|{grp}"]).mean())
            if f"sat|{L}" in f and f[f"sat|{L}"].notna().any():
                d[f"{L}_tanh_sat"] = f[f"sat|{L}"].mean()
        rows.append(d)
    t = pd.DataFrame(rows)
    write_table(t, TAB, "q5_partial_collapse",
                "Q5. Partial collapse: late_degradation runs (learned, final >= 15%, peak-to-final drop > 10 pp; "
                "outcomes.py) that end below 50% (n_band_only = other runs of the cell in the same accuracy band: "
                "stable CIFAR-10 runs or never_learned). Predicted classes and the final dead "
                "fraction and number of surviving ReLU-type units per layer (conv units are channels), means over "
                "the partial runs of each cell.")
    C.put("q5.n_partial_runs_s0", int((p.stage == 0).sum()), "S0 seeds 1-10")
    C.put("q5.n_partial_runs_s2", int((p.stage == 2).sum()), "S2 seeds 1-30")
    C.put("q5.n_partial_runs_s3", int((p.stage == 3).sum()), "S3 seeds 31-45")
    C.put("q5.n_band_only_runs", int(len(band) - len(p)), "S0 + S2 + S3",
          desc="final in [15, 50) but not late_degradation (excluded)")
    return t


# ----------------------------------------------------------------------------- Q6

SIGNALS = ["loss_rise", "zmax_jump", "dead_gt20", "dw_spike"]


def signal_scores(g, arch, zmax_from=2):
    """Per-epoch continuous scores (index = epoch) and the first-crossing epoch of each signal.

    loss_rise  loss(e) / min(loss(1..e-1));                  fires at > 2
    zmax_jump  max over hidden layers of zmax(e) / zmax(e-1);  fires at > 2 (e >= zmax_from)
    dead_gt20  max dead fraction over ReLU-type groups;        fires at > 0.2 (after <= 0.2)
    dw_spike   max over layers of dw_rel(e) / median(dw_rel(1..e-1)); fires at > 3 (e >= 3)

    zmax_from = 2 (used): the first epoch's growth from the initialisation is not a warning
    sign; zmax_from = 1 is the original definition, kept for the post-hoc table.
    """
    g = g.set_index("epoch").sort_index()
    e = g.index.to_numpy()
    sc = pd.DataFrame(index=e)
    loss = g["loss"]
    sc["loss_rise"] = loss / loss.cummin().shift(1)
    zcols = [f"zmax|{L}" for L in hidden_layers(arch) if f"zmax|{L}" in g]
    sc["zmax_jump"] = pd.concat([g[c] / g[c].shift(1) for c in zcols], axis=1).max(axis=1).where(g.index >= zmax_from)
    dcols = [c for c in g.columns if c.startswith("dead|") and g[c].notna().any()]
    sc["dead_gt20"] = g[dcols].max(axis=1) if dcols else np.nan
    wcols = [c for c in g.columns if c.startswith("dw|") and g[c].notna().any()]
    ratios = []
    for c in wcols:
        x = g[c].where(g.index >= 1)
        ratios.append(x / x.expanding().median().shift(1).where(g.index >= 3))
    sc["dw_spike"] = pd.concat(ratios, axis=1).max(axis=1) if ratios else np.nan
    sc.loc[0, "loss_rise"] = np.nan
    first = {}
    thr = {"loss_rise": 2, "zmax_jump": 2, "dw_spike": 3}
    for k, t in thr.items():
        hit = sc.index[(sc.index >= 1) & (sc[k] > t)]
        first[k] = float(hit[0]) if len(hit) else np.nan
    d = sc["dead_gt20"]
    hit = sc.index[(sc.index >= 1) & (d > 0.2) & (d.shift(1) <= 0.2)]
    first["dead_gt20"] = float(hit[0]) if len(hit) else np.nan
    return sc, first


Q6_CELL = ["stage", "dataset", "arch_label", "hidden", "lr", "model"]


def failure_epoch(r):
    """First epoch > 10 pp below the running maximum; e* for a late collapse without one."""
    return r.e_fail if r.e_fail == r.e_fail else r.e_star


def q6_collect(ev, zmax_from=2):
    """Lead times of failing runs and per-epoch scores of positives / negatives.

    Failing = late_collapse or late_degradation (outcomes.py), in cells (dataset, arch, H, LR,
    model) that contain at least one failing run; negatives = the stable runs of those cells."""
    lead_rows, pos, neg = [], [], []
    for stage in (0, 2, 3):
        s = series(stage)
        e = ev[ev.stage == stage].copy()
        e["failing"] = e.outcome.isin(FAILING)
        cells_with_fail = e[e.failing].groupby(Q6_CELL[1:]).size().index
        e = e.set_index(Q6_CELL[1:]).loc[cells_with_fail].reset_index() if len(cells_with_fail) else e.iloc[0:0]
        sg = s[s.run_id.isin(e.run_id)].groupby("run_id")
        for _, r in e.iterrows():
            sc, first = signal_scores(sg.get_group(r.run_id), r.arch, zmax_from)
            base = {k: r[k] for k in Q6_CELL} | {"seed": r.seed, "run_id": r.run_id, "outcome": r.outcome}
            if r.failing:
                ef = failure_epoch(r)
                lead_rows.append(base | {"failure_epoch": ef, "e_star": r.e_star} |
                                 {f"lead_{k}": ef - v for k, v in first.items()} |
                                 {"lead_loss_rise_to_estar": r.e_star - first["loss_rise"]})
                win = sc.loc[(sc.index >= max(1, ef - 3)) & (sc.index <= ef - 1)]
                for ep, row in win.iterrows():
                    pos.append(base | {"epoch": ep} | row.to_dict())
            elif r.outcome == "stable":
                win = sc.loc[(sc.index >= 1) & (sc.index <= EPOCHS - 1)]
                for ep, row in win.iterrows():
                    neg.append(base | {"epoch": ep} | row.to_dict())
    return pd.DataFrame(lead_rows), pd.DataFrame(pos), pd.DataFrame(neg)


def q6_summaries(lead, P, N, signals=SIGNALS):
    lt, au = [], []
    for (stage,), g in lead.groupby(["stage"]):
        for k in signals:
            x = g[f"lead_{k}"]
            lt.append(dict(stage=stage, signal=k, n_failing=len(g), n_signal_fired=int(x.notna().sum()),
                           n_lead_ge1=int((x >= 1).sum()), n_lead_ge3=int((x >= 3).sum()), n_lead_le0=int((x <= 0).sum()),
                           lead_median=med(x), lead_q25=q25(x), lead_q75=q75(x)))
    for stage in (0, 2, 3):
        p_, n_ = P[P.stage == stage], N[N.stage == stage]
        both = set(map(tuple, p_[Q6_CELL].drop_duplicates().to_numpy())) & set(
            map(tuple, n_[Q6_CELL].drop_duplicates().to_numpy()))
        kp = p_[p_[Q6_CELL].apply(tuple, axis=1).isin(both)] if both else p_.iloc[0:0]
        kn = n_[n_[Q6_CELL].apply(tuple, axis=1).isin(both)] if both else n_.iloc[0:0]
        for k in signals:
            au.append(dict(stage=stage, signal=k, pooling="same cell incl. model", n_cells=len(both),
                           n_pos_epochs=int(kp[k].notna().sum()) if len(kp) else 0,
                           n_neg_epochs=int(kn[k].notna().sum()) if len(kn) else 0,
                           auc=auc(kp[k], kn[k]) if len(kp) else np.nan))
    return pd.DataFrame(lt), pd.DataFrame(au)


def q6(ev):
    lead, P, N = q6_collect(ev)
    lt, au = q6_summaries(lead, P, N)
    write_table(lt, TAB, "q6_lead_times",
                "Q6. Lead time = failure epoch - first epoch the signal fires (positive = warning before failure). "
                "Failing runs = late_collapse or late_degradation (outcomes.py); failure epoch = first epoch more than "
                "10 pp below the running maximum (e* for a late collapse without one). Signals: loss_rise = training "
                "loss > 2x its running minimum; zmax_jump = a hidden layer's max |z| more than doubles in one epoch "
                "(from epoch 2 on); dead_gt20 = a ReLU-type group crosses 20% dead; dw_spike = an epoch's relative "
                "weight change > 3x its running median.")
    write_table(lead, TAB, "q6_lead_runs", "Q6. Per-run lead times (epochs).")
    write_table(au, TAB, "q6_auc",
                "Q6. ROC AUC of each signal's value at epoch e for 'the failure epoch is e+1..e+3'. Positives: those "
                "epochs of failing runs; negatives: every epoch 1-39 of the stable runs (outcomes.py) of the same cell "
                "(dataset, arch, H, LR, model).")
    for _, q in lt.iterrows():
        C.put(f"q6.s{q.stage}.{q.signal}.lead_median", q.lead_median, f"S{q.stage}", n=q.n_failing, unit="epochs")
        C.put(f"q6.s{q.stage}.{q.signal}.n_lead_ge1", q.n_lead_ge1, f"S{q.stage}", n=q.n_failing)
        C.put(f"q6.s{q.stage}.{q.signal}.n_fired", q.n_signal_fired, f"S{q.stage}", n=q.n_failing)
    for _, q in au.iterrows():
        C.put(f"q6.s{q.stage}.{q.signal}.auc", q.auc, f"S{q.stage}", n=q.n_pos_epochs + q.n_neg_epochs)
    loss_rise_reconcile(ev, lead)
    lead_fig(lead)
    return lead, lt, au


def loss_rise_reconcile(ev, lead):
    """Q1 vs Q6 for the loss-rise signal: hit rate and lead in failing runs, false-positive rate
    (fires at any epoch) in the stable runs of the same cells and of all cells."""
    rows = []
    groups = [("all", lambda d: d.stage >= 0)] + [
        (f"S{st} {ds} {al}", (lambda st, ds, al: lambda d: (d.stage == st) & (d.dataset == ds) & (d.arch_label == al))(st, ds, al))
        for st, ds, al in sorted(set(zip(lead.stage, lead.dataset, lead.arch_label)))]
    stable = ev[ev.outcome == "stable"]
    fail_cells = set(map(tuple, lead[Q6_CELL].drop_duplicates().to_numpy()))
    stable_same = stable[stable[Q6_CELL].apply(tuple, axis=1).isin(fail_cells)]
    for name, f in groups:
        g = lead[f(lead)]
        lc = g[g.outcome == "late_collapse"]
        ss, sa = stable_same[f(stable_same)], stable[f(stable)]
        x = g.lead_loss_rise
        rows.append(dict(group=name, n_failing=len(g), share_fires_before_failure=float((x >= 1).mean()) if len(g) else np.nan,
                         share_fires_ever=float(x.notna().mean()) if len(g) else np.nan,
                         lead_to_failure_median=med(x), n_late_collapse=len(lc),
                         lead_to_estar_median=med(lc.lead_loss_rise_to_estar),
                         failure_before_estar_median=med(lc.e_star - lc.failure_epoch),
                         n_stable_same_cells=len(ss), fpr_same_cells=float(ss.e_lossrise.notna().mean()) if len(ss) else np.nan,
                         n_stable_all_cells=len(sa), fpr_all_cells=float(sa.e_lossrise.notna().mean()) if len(sa) else np.nan))
    t = pd.DataFrame(rows)
    write_table(t, TAB, "q6_loss_rise_reconcile",
                "Q6. The loss-rise signal (training loss > 2x its running minimum). Failing runs: share where it fires "
                "at least one epoch before the failure epoch, share where it fires at all, median lead to the failure "
                "epoch and (late collapses) to e*; median gap between the failure epoch and e*. False-positive rate = "
                "share of stable runs (outcomes.py) in which it fires at any epoch, in the cells that contain failing "
                "runs and in all cells. Stage 0 seeds 1-10, Stage 2 seeds 1-30 pooled, Stage 3 seeds 31-45 "
                "(exploratory).")
    for _, q in t.iterrows():
        tag = "q6.lossrise." + q.group.replace(" ", "_")
        for f in ("share_fires_before_failure", "lead_to_failure_median", "lead_to_estar_median",
                  "failure_before_estar_median", "fpr_same_cells", "fpr_all_cells", "n_failing",
                  "n_stable_same_cells", "n_stable_all_cells"):
            C.put(f"{tag}.{f}", q[f], "S0 + S2 seeds 1-30 + S3")
    return t


def lead_fig(lead):
    fig, axes = plt.subplots(1, len(SIGNALS), figsize=(10, 2.4), sharey=True)
    bins = np.arange(-15.5, 25.5, 1)
    stage_col = {0: COLORS["relu"], 2: COLORS["tanh+relu"], 3: COLORS["tanh"]}
    for ax, k in zip(axes, SIGNALS):
        for st, g in lead.groupby("stage"):
            x = g[f"lead_{k}"].dropna()
            ax.hist(x, bins=bins, histtype="step", lw=1.5, color=stage_col[st], label=f"Stage {st} (fired {len(x)}/{len(g)})")
        ax.axvline(0, color=MUTED, ls="--", lw=0.8)
        ax.set_title(k, loc="left", color=INK)
        ax.set_xlabel("lead (epochs before failure)", color=MUTED)
        ax.legend(frameon=False, fontsize=6)
        style(ax)
    axes[0].set_ylabel("failing runs", color=MUTED)
    fig.tight_layout()
    savefig(fig, "mech_q6_lead_times")
    add_caption("mech_q6_lead_times",
                "Lead time of four candidate warning signals before the failure epoch (first epoch more than 10 pp below "
                "the running maximum of validation accuracy), per stage (Stage 0 seeds 1-10, Stage 2 seeds 1-30 where "
                "available, Stage 3 seeds 31-45; all exploratory). Positive = the signal fired before failure. Runs where "
                "a signal never fired are counted in the legend.")


# ----------------------------------------------------------------------------- figures for Q2-Q4

def q2_fig(t):
    x = t[(t.layer.isin(["hidden2", "hidden1", "conv"]))]
    panels = [(0, "mnist", "mlp2", 64), (0, "mnist", "mlp2", 128)]
    for ds in ("mnist", "cifar10"):
        panels.append((2, ds, "cfg2", 128))
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 2.6), sharey=True)
    for ax, (st, ds, al, H) in zip(axes, panels):
        g = x[(x.stage == st) & (x.dataset == ds) & (x.arch_label == al) & (x.hidden == H) & (x.hybrid == "tanh+relu")]
        for L, ls in (("conv", ":"), ("hidden1", "--"), ("hidden2", "-")):
            q = g[g.layer == L].sort_values("lr")
            if q.empty:
                continue
            ax.plot(q.lr, q.relu_dead_final, color=COLORS["relu"], ls=ls, marker=MARKERS["relu"], ms=4, label=f"relu {L}")
            ax.plot(q.lr, q.hybrid_relu_dead_final, color=COLORS["tanh+relu"], ls=ls, marker=MARKERS["tanh+relu"], ms=4,
                    label=f"tanh+relu (ReLU half) {L}")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks(sorted(g.lr.unique()))
        ax.set_title(f"S{st} {ds} {al} H={H}", loc="left", color=INK)
        ax.set_xlabel("learning rate", color=MUTED)
        style(ax)
    axes[0].set_ylabel("final dead fraction (mean)", color=MUTED)
    handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], ms=4, label=lab)
               for m, lab in (("relu", "relu"), ("tanh+relu", "tanh+relu, ReLU half"))]
    handles += [Line2D([], [], color=MUTED, ls=ls, label=L) for L, ls in (("conv", ":"), ("hidden1", "--"),
                                                                         ("hidden2", "-"))]
    axes[0].legend(handles=handles, frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout()
    savefig(fig, "mech_q2_hybrid_dead")
    add_caption("mech_q2_hybrid_dead",
                "Final dead fraction of ReLU units versus learning rate: pure ReLU (blue) and the ReLU half of tanh+relu "
                "(orange), per layer (line style). Stage 0 mlp2 MNIST (seeds 1-10, exploratory) and Stage 2 cfg2 H=128 "
                "(seeds 1-30 at the stress/below-stress/selected LRs, 1-10 elsewhere). Means over seeds.")


def q4_fig(sp, ev):
    s0 = series(0)
    k = sp[(sp.stage == 0) & sp.spiked]
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.6))
    for _, r in k.iterrows():
        g = s0[s0.run_id == r.run_id].set_index("epoch").loc[1:]      # epoch 0 = initialisation
        rel = g.index - r.spike_epoch
        for ax, L in zip(axes[:2], ("hidden2", "output")):
            ax.plot(rel, g[f"w|{L}"], color=COLORS["leaky_relu"], lw=0.6, alpha=0.5)
    for ax, L in zip(axes[:2], ("hidden2", "output")):
        ax.set_yscale("log")
        ax.set_xlim(-8, 5)
        ax.axvline(0, color=MUTED, ls="--", lw=0.8)
        ax.set_title(f"w_norm {L} (each spiking run)", loc="left", color=INK)
        ax.set_xlabel("epoch relative to spike", color=MUTED)
        style(ax)
    ax = axes[2]
    cells = sorted(set(zip(k.hidden, k.lr)))
    for i, (H, lr) in enumerate(cells):
        y = k[(k.hidden == H) & np.isclose(k.lr, lr)].spike_epoch
        ax.scatter(np.full(len(y), i) + np.linspace(-0.15, 0.15, len(y)), y, s=12, color=COLORS["leaky_relu"],
                   edgecolor=INK, lw=0.3)
    ax.set_xticks(range(len(cells)), [f"H{H}\n{lr:g}" for H, lr in cells])
    ax.set_title("spike epoch per cell (H, LR)", loc="left", color=INK)
    style(ax)
    fig.tight_layout()
    savefig(fig, "mech_q4_leaky_spike")
    add_caption("mech_q4_leaky_spike",
                "Leaky-ReLU weight explosion in Stage 0 mlp2 MNIST (seeds 1-10, exploratory). Left and middle: weight "
                "norms of the second hidden layer and the output layer, aligned at each run's spike epoch (largest "
                "one-epoch growth of the output weight norm, >= 10x). Right: spike epoch of every spiking run per (H, LR) "
                "cell.")


def main():
    ev = all_events()
    t1, rel = q1(ev)
    event_study(rel)
    collapse_speed(ev)
    revival(rel)
    t2, r2, sw = q2(ev)
    q2_fig(t2)
    q3(ev)
    sp, cells, spread, pred = q4(ev)
    q4_fig(sp, ev)
    q5(ev)
    q6(ev)
    posthoc(ev, rel, sp)
    C.save()
    print("mechanism: wrote tables and figures")


if __name__ == "__main__":
    main()
