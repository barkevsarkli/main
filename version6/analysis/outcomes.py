#!/usr/bin/env python3
"""Outcome taxonomy used by every Part A table from A3 on: each run gets exactly one class.

    never_learned     best validation accuracy < 50% of REF, where REF is the mean best
                      validation accuracy at the model's own best LR in the same cell
                      (source/stage, dataset, config, H, model; best LR = highest mean best
                      validation accuracy over the LR grid, ties -> lower LR);
    late_collapse     learned (not never_learned) and final validation accuracy < 15%;
    late_degradation  learned, final >= 15% and peak-to-final drop > 10 pp;
    stable            everything else (learned, final >= 15%, drop <= 10 pp).

Why the relative rule: one threshold works for both datasets and every architecture,
because it is measured against what the same model reaches at its best LR (about 98% on
MNIST, 45-65% on CIFAR-10, so the cut-off is about 49% and 23-33%). The absolute rule
(best < 50% on MNIST, < 20% on CIFAR-10) is kept as a sensitivity check (`rule="absolute"`);
the run counts that change class are reported in outcomes/rule_sensitivity.csv.

    python3 analysis/outcomes.py   -> analysis_out/partA/outcomes/
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CLASSES = ["never_learned", "late_collapse", "late_degradation", "stable"]
ABSOLUTE = {"mnist": 50.0, "cifar10": 20.0}
REL = 0.5


def reference(df, keys):
    """Mean best validation accuracy at each cell's own best LR (ties -> lower LR)."""
    m = df.groupby(keys + ["lr"], sort=True)["best_val"].mean().reset_index()
    m = m.sort_values(keys + ["best_val", "lr"], ascending=[True] * len(keys) + [False, True], kind="stable")
    return m.groupby(keys, sort=True).head(1).set_index(keys)["best_val"].rename("ref_best_val")


def classify(df, keys, rule="relative"):
    """Outcome class per row of df (needs dataset, lr, best_val, final_val and `keys`)."""
    if rule == "relative":
        ref = df[keys].merge(reference(df, keys).reset_index(), on=keys, how="left")["ref_best_val"].to_numpy()
        never = df["best_val"].to_numpy() < REL * ref
    else:
        never = df["best_val"].to_numpy() < df["dataset"].map(ABSOLUTE).to_numpy()
    final = df["final_val"].to_numpy()
    drop = (df["best_val"] - df["final_val"]).to_numpy()
    out = np.where(never, "never_learned",
                   np.where(final < 15, "late_collapse", np.where(drop > 10, "late_degradation", "stable")))
    return pd.Series(out, index=df.index, name="outcome")


V6_KEYS = ["dataset", "config", "hidden", "model"]


def counts(df, by):
    """One row per group with the number of runs in each class."""
    t = df.groupby(by + ["outcome"], sort=True).size().unstack("outcome", fill_value=0)
    for c in CLASSES:
        if c not in t:
            t[c] = 0
    t = t[CLASSES].reset_index()
    t.insert(len(by), "n", t[CLASSES].sum(axis=1))
    return t


def all_runs():
    """v6 stages 0-3 and the version5 sources, with outcome (relative and absolute rule)."""
    from stats_io import runs
    from v5_io import v5_runs
    parts = []
    for st in (0, 1, 2, 3):
        r = runs(st)
        parts.append(pd.DataFrame({
            "source": f"v6 stage{st}", "dataset": r.dataset, "arch": r.arch, "arch_label": r.arch_label,
            "config": r.config, "hidden": r.hidden, "model": r.model, "lr": r.lr, "seed": r.seed,
            "seed_set": r.seed_set, "best_val": r.best_val, "final_val": r.final_val, "outcome": r.outcome,
            "outcome_abs": classify(r, V6_KEYS, rule="absolute")}))
    v5 = v5_runs()
    parts.append(v5[["source", "dataset", "arch", "arch_label", "config", "hidden", "model", "lr", "seed", "seed_set",
                     "best_val", "final_val", "outcome", "outcome_abs"]])
    return pd.concat(parts, ignore_index=True)


def depth_check(a):
    """Late collapse by architecture: mlp1 (one ReLU layer) vs mlp2 / CNN (two ReLU layers)."""
    rows = []
    for key, g in a.groupby(["source", "dataset", "arch"], sort=True):
        for model in ("relu", "any"):
            x = g if model == "any" else g[g.model == "relu"]
            lc = x[x.outcome == "late_collapse"]
            rows.append(dict(source=key[0], dataset=key[1], arch=key[2], model=model, n=len(x),
                             n_late_collapse=len(lc), n_never_learned=int((x.outcome == "never_learned").sum()),
                             n_late_degradation=int((x.outcome == "late_degradation").sum()),
                             lrs_with_late_collapse=" ".join(f"{v:g}" for v in sorted(lc.lr.unique()))))
    t = pd.DataFrame(rows)
    # Matched comparison inside v6 Stage 2: same dataset, H and LR, ReLU only.
    s2 = a[(a.source == "v6 stage2") & (a.model == "relu")]
    m = []
    for (ds, H, lr), g in s2.groupby(["dataset", "hidden", "lr"], sort=True):
        d = dict(dataset=ds, hidden=H, lr=lr)
        for lab in ("cfg1", "cfg2", "mlp1"):
            x = g[g.arch_label == lab]
            d[f"{lab}_n"] = len(x)
            d[f"{lab}_late_collapse"] = int((x.outcome == "late_collapse").sum())
            d[f"{lab}_never_learned"] = int((x.outcome == "never_learned").sum())
        m.append(d)
    return t, pd.DataFrame(m)


def main():
    from partA_common import Claims, write_table
    C = Claims("outcomes")
    a = all_runs()
    by = ["source", "dataset", "arch_label", "hidden", "model", "lr", "seed_set"]
    write_table(counts(a, by), "outcomes", "outcome_counts",
                "Outcome classes (never_learned, late_collapse, late_degradation, stable; rules in outcomes.py) per "
                "source, dataset, architecture, H, model, LR and seed set. Reference for never_learned: the cell's "
                "mean best validation accuracy at the model's own best LR, all seeds of the source.")
    sens = a.groupby(["source", "dataset", "outcome", "outcome_abs"], sort=True).size().rename("n").reset_index()
    sens = sens[sens.outcome != sens.outcome_abs]
    write_table(sens, "outcomes", "rule_sensitivity",
                "Runs whose class differs between the relative never_learned rule (used) and the absolute rule "
                "(best validation accuracy < 50% on MNIST, < 20% on CIFAR-10).")
    C.put("n_runs", len(a), "v6 stages 0-3 + version5")
    C.put("n_rule_changes", int(sens.n.sum()), "v6 stages 0-3 + version5", desc="runs classed differently by the absolute rule")
    for (src, cls), n in a.groupby(["source", "outcome"]).size().items():
        C.put(f"{src.replace(' ', '_')}.{cls}", int(n), src)
    t, m = depth_check(a)
    write_table(t, "outcomes", "depth_check",
                "Late collapse by architecture (model = relu, or any model). mlp1 has one ReLU layer; mlp2 and the "
                "CNN (conv + hidden) have two.")
    write_table(m, "outcomes", "depth_matched_stage2",
                "v6 Stage 2, pure ReLU, matched dataset x H x LR: late_collapse and never_learned counts for the CNN "
                "(cfg1, cfg2) and mlp1. Seeds 1-30 where run (1-10 off the three extension LRs); exploratory.")
    for _, q in t.iterrows():
        C.put(f"depth.{q.source.replace(' ', '_')}.{q.dataset}.{q.arch}.{q.model}.n_late_collapse", q.n_late_collapse,
              q.source, n=q.n)
        C.put(f"depth.{q.source.replace(' ', '_')}.{q.dataset}.{q.arch}.{q.model}.n_never_learned", q.n_never_learned,
              q.source, n=q.n)
    C.save()
    print(f"outcomes: {len(a)} runs classified")


if __name__ == "__main__":
    main()
