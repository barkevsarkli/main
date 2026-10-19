#!/usr/bin/env python3
"""A7-A9: merge claims and render the Part A reports from templates.

    python3 analysis/report_partA.py

  claims/*.json  (written by outcomes, mechanism, critical_lr, paper_tables, checks_partA)
  + report.*     (derived here from the Part A tables: counts and ranges over cells)
  -> analysis_out/partA/claims.json

  analysis/templates/FINDINGS_partA.md.tmpl          -> analysis_out/partA/FINDINGS_partA.md
  analysis/templates/manuscript_corrections.md.tmpl  -> analysis_out/partA/manuscript_corrections.md

Every number in the two reports comes from claims.json through a placeholder:

    {{key}}           value, default format (str as is, int, float with 2 decimals)
    {{key:SPEC}}      SPEC = a Python format spec (.1f, +.2f, .3g, ...), or
                      pct0 / pct1 (share x 100, 0 / 1 decimals), p (p-value, "< 0.001" below
                      0.001), lr (%g), int, n (the claim's n), seed (the claim's seed set)

The script fails (exit 1) on an unknown key, a null value or any placeholder it cannot
resolve, so no number in the reports is typed by hand. Quoted draft text (main.tex) in
manuscript_corrections.md is quoted, not computed.
"""
import glob
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from partA_common import OUT  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATES = {"FINDINGS_partA.md.tmpl": "FINDINGS_partA.md",
             "manuscript_corrections.md.tmpl": "manuscript_corrections.md"}
PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_.+=@\-]+)(?::([^}]+))?\s*\}\}")


# ----------------------------------------------------------------------------- claims

def load_claims():
    claims = {}
    for f in sorted(glob.glob(os.path.join(OUT, "claims", "*.json"))):
        with open(f) as fh:
            d = json.load(fh)
        dup = set(d) & set(claims)
        if dup:
            raise SystemExit(f"report_partA: duplicate claim keys in {f}: {sorted(dup)[:5]}")
        claims.update(d)
    return claims


def put(claims, key, value, seed_set, n=None, desc=""):
    if isinstance(value, (np.floating, float)):
        value = None if math.isnan(float(value)) else float(value)
    elif isinstance(value, np.integer):
        value = int(value)
    claims[f"report.{key}"] = dict(value=value, seed_set=seed_set, n=None if n is None else int(n), desc=desc, unit="")


def table(sub, name):
    return pd.read_csv(os.path.join(OUT, sub, f"{name}.csv"))


def derived(claims):
    """Counts and ranges over cells, computed from the Part A tables."""
    conf = "S2 ext seeds 11-30 (confirmatory)"
    # Run counts per stage (outcome classes sum to the stage size).
    for st in range(4):
        n = sum(v["value"] for k, v in claims.items() if k.startswith(f"outcomes.v6_stage{st}.")
                and k.split(".")[-1] in ("never_learned", "late_collapse", "late_degradation", "stable"))
        put(claims, f"n_runs.stage{st}", n, f"v6 stage{st}")

    # Sigmoid at the stress LR (T1).
    t1 = table("tables", "T1_stage2_stress")
    s = t1[t1.lr_role == "stress"]
    top, exc = 0, []
    cells = list(s.groupby(["dataset", "arch", "hidden"], sort=True))
    for (ds, arch, H), g in cells:
        g = g[~g.model.str.contains("@")]
        best = g.sort_values("final_mean", ascending=False).iloc[0]
        if best.model == "sigmoid":
            top += 1
        else:
            sg = g[g.model == "sigmoid"].iloc[0]
            exc.append(f"{ds} {arch} H{H} ({best.model} {best.final_mean:.1f}% vs sigmoid {sg.final_mean:.1f}%)")
    sig = s[s.model == "sigmoid"]
    put(claims, "sigmoid_stress.n_top", top, conf, n=len(cells), desc="cells where sigmoid has the highest mean final")
    put(claims, "sigmoid_stress.n_cells", len(cells), conf)
    put(claims, "sigmoid_stress.exceptions", "; ".join(exc) if exc else "none", conf)
    put(claims, "sigmoid_stress.n_fail", f"{int(sig.n_final_lt15.sum())}/{int(sig.n.sum())}", conf)
    put(claims, "sigmoid_stress.final_min_cnn", sig[sig.arch != "mlp1"].final_mean.min(), conf)
    for ds in ("mnist", "cifar10"):
        q = s[(s.dataset == ds) & (s.arch != "mlp1")]
        sg, hy = q[q.model == "sigmoid"], q[q.model == "tanh+relu"]
        put(claims, f"sigmoid_stress.{ds}_cnn_final_min", sg.final_mean.min(), conf, n=sg.n.sum())
        put(claims, f"sigmoid_stress.{ds}_cnn_final_max", sg.final_mean.max(), conf, n=sg.n.sum())
        put(claims, f"hybrid_stress.{ds}_cnn_final_min", hy.final_mean.min(), conf, n=hy.n.sum())
        put(claims, f"hybrid_stress.{ds}_cnn_final_max", hy.final_mean.max(), conf, n=hy.n.sum())
        put(claims, f"hybrid_stress.{ds}_cnn_n_fail", f"{int(hy.n_final_lt15.sum())}/{int(hy.n.sum())}", conf)
        r = q[q.model == "relu"]
        put(claims, f"relu_stress.{ds}_cnn_n_fail", f"{int(r.n_final_lt15.sum())}/{int(r.n.sum())}", conf)

    # Hybrid minus sigmoid / tanh at the stress LR (T2, final validation accuracy).
    t2 = table("tables", "T2_paired_hybrid")
    st = t2[(t2.lr_role == "stress") & (t2.metric == "final_val")]
    for comp, tag in (("tanh+relu - sigmoid", "sigmoid"), ("tanh+relu - tanh", "tanh"), ("tanh+relu - relu", "relu")):
        q = st[st.comparison == comp]
        put(claims, f"t2_stress.minus_{tag}.n_neg", int((q.mean_diff < 0).sum()), conf, n=len(q))
        put(claims, f"t2_stress.minus_{tag}.n_cells", len(q), conf)
        put(claims, f"t2_stress.minus_{tag}.n_sig_pos", int(((q.mean_diff > 0) & (q.p_holm < 0.05)).sum()), conf,
            n=len(q), desc="positive with p_holm < 0.05")
        put(claims, f"t2_stress.minus_{tag}.n_sig_neg", int(((q.mean_diff < 0) & (q.p_holm < 0.05)).sum()), conf,
            n=len(q), desc="negative with p_holm < 0.05")
        for ds in ("mnist", "cifar10"):
            for part, m in (("cnn", q.arch != "mlp1"), ("mlp1", q.arch == "mlp1")):
                x = q[(q.dataset == ds) & m]
                put(claims, f"t2_stress.minus_{tag}.{ds}_{part}_min", x.mean_diff.min(), conf, n=len(x))
                put(claims, f"t2_stress.minus_{tag}.{ds}_{part}_max", x.mean_diff.max(), conf, n=len(x))

    # Tuned accuracy (T5): hybrid - ReLU at the selected LRs, and the grid edge.
    t5p = table("tables", "T5_tuned_paired")
    q = t5p[t5p.comparison == "tanh+relu - relu"]
    for ds in ("mnist", "cifar10"):
        x = q[q.dataset == ds]
        put(claims, f"t5.{ds}.min", x.mean_diff.min(), conf, n=len(x))
        put(claims, f"t5.{ds}.max", x.mean_diff.max(), conf, n=len(x))
        put(claims, f"t5.{ds}.n_sig", int((x.p_holm < 0.05).sum()), conf, n=len(x), desc="cells with p_holm < 0.05")
    x = q[(q.dataset == "cifar10") & (q.arch != "mlp1") & q.hidden.isin([64, 128])]
    put(claims, "t5.cifar10_cnn_h64_128.min", x.mean_diff.min(), conf, n=len(x))
    put(claims, "t5.cifar10_cnn_h64_128.max", x.mean_diff.max(), conf, n=len(x))
    put(claims, "t5.cifar10_cnn_h64_128.n_sig", int((x.p_holm < 0.05).sum()), conf, n=len(x))
    put(claims, "t5.cifar10_cnn_h64_128.cost_min", -x.mean_diff.max(), conf, n=len(x), desc="ReLU - hybrid, smallest")
    put(claims, "t5.cifar10_cnn_h64_128.cost_max", -x.mean_diff.min(), conf, n=len(x), desc="ReLU - hybrid, largest")
    x = q[(q.dataset == "cifar10") & (q.arch != "mlp1") & (q.hidden == 256)]
    put(claims, "t5.cifar10_cnn_h256.min", x.mean_diff.min(), conf, n=len(x))
    put(claims, "t5.cifar10_cnn_h256.max", x.mean_diff.max(), conf, n=len(x))
    t5 = table("tables", "T5_tuned_accuracy")
    t5 = t5[~t5.model.str.contains("@")]
    sel = "S2 sweep seeds 1-10 (selection)"
    for ds in ("mnist", "cifar10"):
        x = t5[t5.dataset == ds]
        put(claims, f"t5.{ds}.n_grid_edge_low", f"{int((x.grid_edge == 'low').sum())}/{len(x)}", sel)
        for arch in ("cfg1", "cfg2", "mlp1"):
            y = x[x.arch == arch]
            put(claims, f"t5.{ds}.{arch}.n_grid_edge_low", f"{int((y.grid_edge == 'low').sum())}/{len(y)}", sel)

    # Collapse speed (mechanism Q1): 10% -> 90% dead, last hidden layer, and its timing vs e*.
    sp = table("mechanism_tables", "q1_collapse_speed")
    put(claims, "speed.n_runs", int(sp.n.sum()), "S0 seeds 1-10 + S2 seeds 1-30 + S3 seeds 31-45")
    for tag, m in (("s2_mnist", (sp.stage == 2) & (sp.dataset == "mnist")),
                   ("s2_cifar10", (sp.stage == 2) & (sp.dataset == "cifar10")),
                   ("s2_h256", (sp.stage == 2) & (sp.hidden == 256)),
                   ("s2_h64_128", (sp.stage == 2) & sp.hidden.isin([64, 128]))):
        x = sp[m]
        seeds = "S2 seeds 1-30 (descriptive, pooled sweep+ext)"
        put(claims, f"speed.{tag}.n", int(x.n.sum()), seeds)
        put(claims, f"speed.{tag}.median_min", x.dur_median.min(), seeds)
        put(claims, f"speed.{tag}.median_max", x.dur_median.max(), seeds)
        put(claims, f"speed.{tag}.share_le1", (x.dur_0.sum() + x.dur_1.sum()) / x.n.sum(), seeds, n=x.n.sum())
        put(claims, f"speed.{tag}.share_ge5", (x["dur_5-9"].sum() + x["dur_10+"].sum()) / x.n.sum(), seeds,
            n=x.n.sum())
        put(claims, f"speed.{tag}.dur_max", x.dur_max.max(), seeds)
    allr = "S0 seeds 1-10 + S2 seeds 1-30 + S3 seeds 31-45"
    put(claims, "speed.e90_at_estar", f"{int(sp.n_e90_at_estar.sum())}/{int(sp.n.sum())}", allr)
    put(claims, "speed.e90_before_estar", f"{int(sp.n_e90_before_estar.sum())}/{int(sp.n.sum())}", allr)
    put(claims, "speed.e90_after_estar", f"{int(sp.n_e90_after_estar.sum())}/{int(sp.n.sum())}", allr)
    put(claims, "speed.off_min", sp.off_min.min(), allr)

    # Hybrid regimes at LR 0.02 in the Stage 2 CNN (mechanism Q2) and which layer dies first (Q3).
    q2 = table("mechanism_tables", "q2_hybrid_regimes")
    pooled = "S2 seeds 1-30 (descriptive, pooled sweep+ext)"
    for ds in ("mnist", "cifar10"):
        x = q2[(q2.stage == 2) & (q2.dataset == ds) & (q2.arch_label != "mlp1") & np.isclose(q2.lr, 0.02)
               & (q2.hybrid == "tanh+relu")]
        for reg in ("protected", "carried", "failed"):
            put(claims, f"q2.{ds}_cnn_lr0.02.{reg}", f"{int(x[reg].sum())}/{int(x.n.sum())}", pooled)
        put(claims, f"q2.{ds}_cnn_lr0.02.relu_dead_min", x.hybrid_relu_dead_last_med.min(), pooled)
        put(claims, f"q2.{ds}_cnn_lr0.02.relu_dead_max", x.hybrid_relu_dead_last_med.max(), pooled)
    q3 = table("mechanism_tables", "q3_failure_order")
    for (ds, model), g in q3.groupby(["dataset", "model"], sort=True):
        hf = int(g.loc[g.order == "hidden first", "n_runs"].sum())
        tag = f"q3.{ds}.{model}"
        put(claims, f"{tag}.hidden_first", f"{hf}/{int(g.n_failing.iloc[0])}", "S2 seeds 1-30, cfg2 H 128/256")
        put(claims, f"{tag}.death_by_failure", f"{int(g.n_death_at_or_before_failure.iloc[0])}/"
            f"{int(g.n_failing.iloc[0])}", "S2 seeds 1-30, cfg2 H 128/256")

    # Leaky-ReLU spike (Q4): spread of the logits across inputs at the spike epoch.
    rec = table("mechanism_tables", "q4_spike_recovery")
    for _, r in rec.iterrows():
        put(claims, f"q4.s{r.stage}_{r.dataset}_{r.arch_label}_lr{r.lr:g}.lbs_at", r.logit_between_std_at, r.seed_set,
            n=r.n, desc="median logit_between_std at the spike epoch")

    # Depth, matched within Stage 2 (pure ReLU, same dataset / H / LR): CNN vs mlp1 at LR 0.02.
    dm = table("outcomes", "depth_matched_stage2")
    x = dm[np.isclose(dm.lr, 0.02)]
    for ds in ("mnist", "cifar10"):
        y = x[x.dataset == ds]
        lc = int(y.cfg1_late_collapse.sum() + y.cfg2_late_collapse.sum())
        put(claims, f"depth_lr0.02.{ds}.cnn_late_collapse", f"{lc}/{int(y.cfg1_n.sum() + y.cfg2_n.sum())}",
            "S2 seeds 1-30 where run (descriptive)")
        put(claims, f"depth_lr0.02.{ds}.mlp1_late_collapse", f"{int(y.mlp1_late_collapse.sum())}/{int(y.mlp1_n.sum())}",
            "S2 seeds 1-30 where run (descriptive)")
        put(claims, f"depth_lr0.02.{ds}.mlp1_never_learned", f"{int(y.mlp1_never_learned.sum())}/{int(y.mlp1_n.sum())}",
            "S2 seeds 1-30 where run (descriptive)")

    # Clamp ratio, S2 CNN groups pooled (checks_partA).
    k = n = 0
    for ds in ("mnist", "cifar10"):
        for arch in ("cfg1", "cfg2"):
            a, b = claims[f"checks.clamp_ratio.s2.{ds}.{arch}.n_at_clamp"]["value"].split("/")
            k, n = k + int(a), n + int(b)
    put(claims, "clamp.s2_cnn_n_at_clamp", f"{k}/{n}", "S2 seeds 1-30 (all leaky runs with final train loss > ln 10)")
    for ds in ("mnist", "cifar10"):
        k = n = 0
        for arch in ("cfg1", "cfg2"):
            a, b = claims[f"checks.clamp_ratio.s2.{ds}.{arch}.n_at_clamp"]["value"].split("/")
            k, n = k + int(a), n + int(b)
        put(claims, f"clamp.s2_{ds}_cnn_n_at_clamp", f"{k}/{n}", "S2 seeds 1-30")

    # Critical LR (T6): how many hybrid/relu ratios exclude 1 (lower end > 1).
    rt = table("critical_lr", "lr50_ratios_fail")
    v6 = rt[rt.source == "v6 stage2"]
    lo = v6["ratio_tanh+relu_relu_lo"]
    put(claims, "lr50.v6_s2_fail.n_ratio_gt1", int((lo > 1).sum()), "S2 seeds 1-30", n=int(lo.notna().sum()))
    put(claims, "lr50.v6_s2_fail.n_cells", len(v6), "S2 seeds 1-30")
    for ds in ("mnist", "cifar10"):
        x = v6[(v6.dataset == ds) & (v6.arch != "mlp1")]["ratio_tanh+relu_relu_lo"]
        put(claims, f"lr50.v6_s2_fail.{ds}_cnn_ratio_lo_min", x.min(), "S2 seeds 1-30", n=len(x))
        put(claims, f"lr50.v6_s2_fail.{ds}_cnn_ratio_lo_max", x.max(), "S2 seeds 1-30", n=len(x))
    return claims


# ----------------------------------------------------------------------------- rendering

def fmt(claim, spec, key):
    v = claim["value"]
    if spec == "n":
        v = claim["n"]
        if v is None:
            raise KeyError(f"{key}: no n")
        return str(v)
    if spec == "seed":
        return claim["seed_set"]
    if v is None:
        raise KeyError(f"{key}: value is null")
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, str):
        if spec:
            raise KeyError(f"{key}: format {spec!r} for a text value")
        return v
    v = float(v)
    if spec in (None, ""):
        return str(int(v)) if v == int(v) and abs(v) < 1e15 and isinstance(claim["value"], int) else f"{v:.2f}"
    if spec == "int":
        return str(int(round(v)))
    if spec == "lr":
        return f"{v:g}"
    if spec in ("pct0", "pct1"):
        return f"{100 * v:.{spec[-1]}f}"
    if spec == "p":
        return "< 0.001" if v < 0.001 else f"{v:.3f}"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return format(v, spec)


def render(text, claims, used):
    missing = []

    def sub(m):
        key, spec = m.group(1), (m.group(2) or "").strip() or None
        if key not in claims:
            missing.append(f"unknown key {key}")
            return m.group(0)
        try:
            out = fmt(claims[key], spec, key)
        except (KeyError, ValueError) as e:
            missing.append(str(e))
            return m.group(0)
        used.add(key)
        return out

    out = PLACEHOLDER.sub(sub, text)
    if "{{" in out or "}}" in out:
        missing.append("unparsed placeholder left: " + out[out.find("{{"):out.find("{{") + 60])
    return out, missing


def main():
    claims = derived(load_claims())
    with open(os.path.join(OUT, "claims.json"), "w") as f:
        json.dump(claims, f, indent=1, sort_keys=True)
        f.write("\n")
    used, errors = set(), []
    for tmpl, out in TEMPLATES.items():
        with open(os.path.join(HERE, "templates", tmpl)) as f:
            text, miss = render(f.read(), claims, used)
        errors += [f"{tmpl}: {m}" for m in miss]
        with open(os.path.join(OUT, out), "w") as f:
            f.write(text)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(f"report_partA: {len(errors)} unresolved placeholder(s)")
    print(f"report_partA: {len(claims)} claims in claims.json; {len(used)} distinct claims quoted in "
          f"{', '.join(TEMPLATES.values())}")


if __name__ == "__main__":
    main()
