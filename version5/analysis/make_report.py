#!/usr/bin/env python3
"""Assemble version5/REPORT_EXTENDED.md from the extended-sweep CSVs.

Reuses the loaders, statistics and table builders in analyze.py, so the numbers here
and in results/summary_extended.md come from exactly the same code path.
Section 9 (the plain-language reading) and section 10 (deviations) are read from
analysis/report_prose.md, which is written by hand.
"""
import collections, json, os, subprocess, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A

RES = A.RES
ROOT = A.ROOT

def h(n, t): return f"\n## {n}. {t}\n"

def load_prose():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_prose.md")
    if not os.path.exists(p):
        return {}
    out, key, buf = {}, None, []
    for line in open(p):
        if line.startswith("@@"):
            if key: out[key] = "".join(buf).strip()
            key, buf = line[2:].strip(), []
        else:
            buf.append(line)
    if key: out[key] = "".join(buf).strip()
    return out

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

def main():
    prose = load_prose()
    md = ["# version5 extended sweep — report", "",
          "Produced by `analysis/make_report.py` from the CSVs in `results/`. "
          "The full auto-generated tables and every figure live in "
          "[`results/summary_extended.md`](results/summary_extended.md); this file is the digest.", ""]

    # ---- 1. machine, flags, timing
    md += [h(1, "Machine, build and timing").strip(), "", prose.get("machine", "_(missing)_"), ""]

    # ---- 2. run inventory
    mains = A.discover("main"); calibs = A.discover("calib")
    lrsweeps = A.discover("lrsweep"); layouts = A.discover("layout")
    main_rows = {w: A.load(f) for w, (f, _) in mains.items()}
    widths = [w for w in sorted(main_rows) if main_rows[w]]
    per_width = {w: A.by_cfg(main_rows[w]) for w in widths}

    md += [h(2, "Run inventory").strip(), "",
           "Evaluation runs (stage B), config × width → number of seeds:", "",
           "| config | " + " | ".join(f"W={w}" for w in widths) + " |", "|---|" + "---|" * len(widths)]
    for k in A.CFG_ORDER:
        cells = [str(len(per_width[w].get(k, []))) if per_width[w].get(k) else "–" for w in widths]
        if all(c == "–" for c in cells): continue
        md.append(f"| {k} | " + " | ".join(cells) + " |")
    md += ["", "All other stages:", "", "| stage | file | ok runs |", "|---|---|---|"]
    for label, d in (("A calibration", calibs), ("C LR robustness", lrsweeps), ("D layout/ratio", layouts)):
        for w, (f, _) in d.items():
            md.append(f"| {label} (W={w}) | `{f}` | {len(A.load(f))} |")
    for f, label in (("calib_mse_128.csv", "E MSE calibration (W=128)"), ("mse_128.csv", "E MSE main (W=128)")):
        if os.path.exists(os.path.join(RES, f)):
            md.append(f"| {label} | `{f}` | {len(A.load(f))} |")
    md.append("")
    sup = [(w, f, s) for w, (f, s) in mains.items() if s]
    if sup:
        md += ["Where a width has both `main_<W>.csv` and `main_<W>_v2.csv`, the `_v2` file is used and the "
               "older one ignored: " + "; ".join(f"W={w} uses `{f}`, ignores `{s}`" for w, f, s in sup) + ".", ""]
    md += [prose.get("inventory_note", ""), ""]

    # ---- 3. best_lr
    md += [h(3, "Calibrated learning rates").strip(), ""]
    best = json.load(open(os.path.join(RES, "best_lr.json")))
    oldp = os.path.join(RES, "best_lr_firstlook.json")
    old = json.load(open(oldp)) if os.path.exists(oldp) else {}
    ws = sorted({int(k.split("/")[4]) for k in best if k.endswith("/ce")})
    md += ["Cross-entropy, selected on mean validation accuracy over the calibration seeds "
           "(101–103; 101–102 at width 512), grid {0.001, 0.003, 0.01, 0.03}:", "",
           "| config | " + " | ".join(f"W={w}" for w in ws) + " |", "|---|" + "---|" * len(ws)]
    def keyof(k, w, loss="ce"):
        if "+" in k:
            a1, a2 = k.split("+"); return f"{a1}/{a2}/0.5/interleave/{w}/{loss}"
        return f"{k}/{k}/0/interleave/{w}/{loss}"
    for k in A.CFG_ORDER:
        cells = [f"{best[keyof(k, w)]:g}" if keyof(k, w) in best else "–" for w in ws]
        if all(c == "–" for c in cells): continue
        md.append(f"| {k} | " + " | ".join(cells) + " |")
    md.append("")
    changed = []
    for w in (12, 128):
        for k in A.CFG_ORDER:
            kk = keyof(k, w)
            if kk in best and kk in old and best[kk] != old[kk]:
                changed.append(f"W={w} {k}: {old[kk]:g} → {best[kk]:g}")
            elif kk in best and kk not in old:
                changed.append(f"W={w} {k}: (not calibrated in the first-look sweep) → {best[kk]:g}")
    md += ["Changes against the first-look selections (`results/best_lr_firstlook.json`) at widths 12 and 128:", ""]
    md += ([f"- {c}" for c in changed] if changed else ["- none; every width-12 and width-128 selection is unchanged."])
    md += ["", prose.get("lr_note", ""), ""]
    mse_keys = [k for k in best if k.endswith("/mse") and "/128/" in k]
    if mse_keys:
        md += ["MSE + sigmoid output at width 128, grid {0.003, 0.01, 0.03}:", "",
               "| config | lr |", "|---|---|"]
        for k in A.CFG_ORDER:
            kk = keyof(k, 128, "mse")
            if kk in best: md.append(f"| {k} | {best[kk]:g} |")
        md.append("")

    # ---- 4. per-width tables
    md += [h(4, "Per-width results").strip(), ""]
    cross = {p: {} for p in A.CROSS_PAIRS}
    for w in widths:
        g = per_width[w]; order = A.order_for(g)
        ns = sorted({len(v) for v in g.values()})
        nlab = str(ns[0]) if len(ns) == 1 else f"{min(ns)}–{max(ns)}"
        md += [f"### Width {w} — {nlab} seeds per config, cross-entropy", "", A.summary_table(g, order), "",
               A.paired_table(g, A.PAIRS), "",
               f"Figures: `results/figures/fig_acc_{w}.png`, `fig_f1_{w}.png`, "
               f"`fig_paired_acc_{w}.png`, `fig_paired_f1_{w}.png`, `fig_curves_{w}.png`", "",
               f"![](results/figures/fig_acc_{w}.png)", f"![](results/figures/fig_paired_acc_{w}.png)", ""]
        for (hyb, base) in A.CROSS_PAIRS:
            if hyb in g and base in g:
                cross[(hyb, base)][w] = {"acc": A.paired(g[hyb], g[base], "test_acc"),
                                         "f1": A.paired(g[hyb], g[base], "test_f1")}

    # ---- 5. crossover
    md += [h(5, "Crossover: hybrid − parent against width").strip(), ""]
    for metric, label in (("acc", "test accuracy"), ("f1", "test macro-F1")):
        md += [f"**{label}** (Δ in points, positive = hybrid ahead)", "",
               "| comparison | width | n | Δ | 95% bootstrap CI | wins | paired t p | Wilcoxon p |",
               "|---|---|---|---|---|---|---|---|"]
        for (hyb, base) in A.CROSS_PAIRS:
            for w in widths:
                c = cross[(hyb, base)].get(w)
                if not c: continue
                p = c[metric]
                md.append(f"| {hyb} − {base} | {w} | {p['n']} | {p['mean']:+.2f} | "
                          f"[{p['ci'][0]:+.2f}, {p['ci'][1]:+.2f}] | {p['wins']}/{p['n']} | "
                          f"{A.fmt_p(p.get('t_p'))} | {A.fmt_p(p.get('w_p'))} |")
        md.append("")
    md += ["![](results/figures/fig_crossover.png)", "![](results/figures/fig_crossover_f1.png)",
           "![](results/figures/fig_width_acc.png)", ""]

    # ---- 6. LR robustness
    md += [h(6, "Learning-rate robustness").strip(), ""]
    lr_rows = {}
    partial_lr = []
    for w, (f, _) in lrsweeps.items():
        rows = A.load(f)
        if not rows: continue
        ok, bad, _ = complete_configs(rows)
        if not ok:
            partial_lr.append((w, f, len(rows), bad)); continue
        if bad:
            partial_lr.append((w, f, None, bad))
        lr_rows[w] = [r for r in rows if r["cfg"] in ok]
    for w, f, n, bad in partial_lr:
        names = ", ".join(f"`{c}`" for c in bad)
        if n is None:
            md += [f"At width {w} (`{f}`), {names} is **omitted** from the tables and the "
                   f"robustness score: its learning-rate grid is not fully run, and a missing "
                   f"high-LR cell would read as \"never collapsed\". Every other config is complete.", ""]
        else:
            md += [f"Width {w} (`{f}`) is **excluded**: no config has a complete learning-rate "
                   f"grid ({n} rows).", ""]
    for w in sorted(lr_rows):
        rows = lr_rows[w]; ns = len({r["seed"] for r in rows})
        md += [f"**Width {w} — mean validation accuracy over {ns} seeds**", "", A.lr_mean_table(rows), "",
               f"**Width {w} — collapsed seeds (final validation accuracy < {A.COLLAPSE_ACC:g}%), count / n**", "",
               A.collapse_table(rows), "", f"![](results/figures/fig_lrsweep_{w}.png)", ""]
    if lr_rows:
        tbl, _ = A.robustness_table(lr_rows)
        md += [f"**Robustness score** — widest contiguous interval of the tested LR grid on which a config's mean "
               f"validation accuracy stays within {A.WITHIN_PTS:g} point of its own best "
               f"(lo–hi, grid points spanned, decades):", "", tbl, ""]

    # ---- 7. layout / ratio
    md += [h(7, "Layout and ratio").strip(), ""]
    for w, (f, _) in layouts.items():
        rows = A.load(f)
        if not rows: continue
        g = per_width.get(w, {})
        lay = collections.OrderedDict()
        for r in rows: lay.setdefault((r["cfg"], r["layout"], r["ratio"]), []).append(r)
        md += [f"**Width {w}** — each pair uses the LR calibrated for its interleave/0.5 variant, so ratios "
               "0.25 and 0.75 are not separately tuned.", "",
               "| pair | layout | ratio | lr | n | test acc mean ± sd | test macro-F1 mean ± sd "
               "| Δ acc vs non-tanh parent | Δ acc vs tanh | Δ acc vs interleave 0.5 |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for (cfg, layout, ratio), v in sorted(lay.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
            parent = cfg.split("+")[1] if "+" in cfg else cfg
            acc = np.array([r["test_acc"] for r in v]); f1 = np.array([r["test_f1"] for r in v])
            def d(x): return "–" if x["n"] == 0 else f"{x['mean']:+.2f} [{x['ci'][0]:+.2f}, {x['ci'][1]:+.2f}]"
            md.append(f"| {cfg} | {layout} | {ratio:g} | {v[0]['lr']:g} | {len(v)} | "
                      f"{acc.mean():.2f} ± {acc.std(ddof=1):.2f} | {f1.mean():.2f} ± {f1.std(ddof=1):.2f} | "
                      f"{d(A.paired(v, g.get(parent, []), 'test_acc'))} | "
                      f"{d(A.paired(v, g.get('tanh', []), 'test_acc'))} | "
                      f"{d(A.paired(v, lay.get((cfg, 'interleave', 0.5), []), 'test_acc'))} |")
        md.append("")

    # ---- 8. MSE at 128
    md += [h(8, "MSE replication at width 128").strip(), ""]
    rows = A.load("mse_128.csv")
    if rows:
        gm = A.by_cfg(rows); om = A.order_for(gm)
        md += [A.summary_table(gm, om), "",
               A.paired_table(gm, [("tanh+relu", "relu"), ("tanh+relu", "tanh")]), "",
               "![](results/figures/fig_acc_128_mse.png)", ""]
        cal = A.load("calib_mse_128.csv")
        if cal:
            md += ["Calibration grid (validation accuracy, mean of seeds 101–103):", "",
                   A.lr_mean_table(cal, A.CFG_ORDER), ""]
    else:
        md += ["_Not run._", ""]

    # ---- 9 / 10 prose
    md += [h(9, "What the data show").strip(), "", prose.get("reading", "_(missing)_"), ""]
    md += [h(10, "Deviations, problems and anything skipped").strip(), "", prose.get("deviations", "_(missing)_"), ""]

    # ---- appendix: sanity checks
    md += ["\n## Appendix — sanity checks\n", prose.get("sanity", "_(missing)_"), ""]

    path = os.path.join(ROOT, "REPORT_EXTENDED.md")
    with open(path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"wrote {path}", file=sys.stderr)

    # The prompt asks for items 5, 6 and 9 on the terminal as well.
    def section(n):
        starts = [i for i, l in enumerate(md) if l.startswith(f"## {n}.")]
        if not starts: return ""
        i = starts[0]
        j = next((k for k in range(i + 1, len(md)) if md[k].startswith("## ")), len(md))
        return "\n".join(md[i:j]).strip()
    print("\n\n".join(section(n) for n in (5, 6, 9)))
    return md

if __name__ == "__main__":
    main()
