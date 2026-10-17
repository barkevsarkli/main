#!/usr/bin/env python3
"""Per-stage run counts and wall-clock estimates from results/bench/bench.json.

    python3 tools/estimate.py [--aws-cores 192] [--aws-price 2.6]

Wall time = sum over runs of (epochs x seconds_per_epoch + per-run overhead) / workers,
plus 5%. seconds_per_epoch comes from the benchmark cell with the same dataset, arch,
H, K and conv cost class (frozen = no conv backward; random and bank_init = learned
conv), measured with all workers busy, so contention is included.

AWS: the benchmark ran version5's own code path (mlp2 --rng-mode legacy, CIFAR-10,
H = 128) under the same load; version5 measured 19.3 s/epoch for that job on a
c8g.48xlarge at 190 concurrent workers. Every time is divided by that ratio and the sum
is spread over --aws-cores workers. This assumes the CNN kernels speed up by the same
factor as the MLP between the two machines -- an approximation (+-30% is a fair
guess), to be replaced by a 20-minute benchmark on the instance itself.

Stages 2 and 3 use make_jobs.ASSUMPTION for the not-yet-selected configurations.
Reduced variants follow PROMPT.md section 11.5 and are applied one step at a time until
the stage fits its target on this machine.
"""
import argparse
import json
import os
import sys

V6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V6, "sweep"))
import common  # noqa: E402
import make_jobs as mj  # noqa: E402

TARGET_H = {"stage0": 4, "stage1": 72, "stage2": 120, "stage3": 24}


class Cost:
    def __init__(self, bench):
        self.cells = {}
        for c in bench["cost"]:
            cls = "frozen" if c["kernel_mode"] == "frozen" else "learned"
            self.cells[(c["dataset"], c["arch"], c["hidden"], c["conv_k"], cls)] = c
        self.hs = sorted({c["hidden"] for c in bench["cost"]})

    def run_seconds(self, job):
        c = common.canonical(job)
        cls = "frozen" if c["kernel_mode"] == "frozen" else "learned"
        key = (c["dataset"], c["arch"], c["hidden"], c["conv_k"], cls)
        if key not in self.cells:
            raise KeyError(f"no benchmark cell for {key}")
        cell = self.cells[key]
        return c["epochs"] * cell["epoch_s"] + cell["overhead_s"]


def total(jobs, cost):
    return sum(cost.run_seconds(j) for j in jobs)


def hours(sec, workers):
    return sec / workers / 3600 * 1.05


def variants(stage, sel1, sel2_assumed):
    """[(label, jobs)] full first, then the cumulative reduction steps."""
    A = mj.ASSUMPTION
    if stage == "stage0":
        return [("full", mj.stage0())]
    if stage == "stage1":
        return [("full", mj.stage1()),
                ("seeds 3 -> 2", mj.stage1(seeds=(1, 2))),
                ("+ drop K = 7", mj.stage1(seeds=(1, 2), ks=(3, 5, 9))),
                ("+ drop lowest LR", mj.stage1(seeds=(1, 2), ks=(3, 5, 9), drop_low_lr=True))]
    if stage == "stage2":
        def s2(sweep_seeds, ext_seeds, controls_h=(128, 256), drop_sig=False):
            kw = dict(controls_h=controls_h, drop_sigmoid_c10_256=drop_sig)
            return mj.stage2_sweep(sel1, seeds=sweep_seeds, **kw) + mj.stage2_ext(sel1, sel2_assumed, seeds=ext_seeds, **kw)
        return [("full", s2(range(1, 11), range(11, 31))),
                ("sweep seeds 10 -> 6", s2(range(1, 7), range(7, 31))),
                ("+ extension 30 -> 20 seeds", s2(range(1, 7), range(7, 21))),
                ("+ drop @conv/@hidden at H=256", s2(range(1, 7), range(7, 21), controls_h=(128,))),
                ("+ drop sigmoid CIFAR H=256", s2(range(1, 7), range(7, 21), controls_h=(128,), drop_sig=True))]
    if stage == "stage3":
        return [("full", mj.stage3(sel1, sel2_assumed)),
                ("seeds 15 -> 10 (not in PROMPT; proposed)", mj.stage3(sel1, sel2_assumed, seeds=range(1, 11)))]


def assumed_sel2():
    """Three distinct LRs per cell (upper bound for the extension's size)."""
    out = {}
    for ds in mj.DATASETS:
        g = mj.STAGE2_LRS[ds]
        rec = {"selected_lr": {m: g[2] for m in mj.STAGE2_MODELS + mj.CONTROLS},
               "stress_lr": g[4], "below_stress_lr": g[3], "above_stress_lr": g[5]}
        out[ds] = {lab: {str(H): rec for H in (64, 128, 256)} for lab in ("cfg1", "cfg2", "mlp1")}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", default=os.path.join(V6, "results", "bench", "bench.json"))
    ap.add_argument("--aws-cores", type=int, default=192)
    ap.add_argument("--aws-price", type=float, default=2.6, help="USD per instance hour (version5 spot price)")
    a = ap.parse_args()
    bench = json.load(open(a.bench))
    cost = Cost(bench)
    W = bench["chosen_workers"]
    ratio = bench["aws_reference"]["ratio_this_over_aws"]
    sel1 = {ds: mj.ASSUMPTION for ds in mj.DATASETS}
    sel2 = assumed_sel2()
    report, lines = {}, []
    lines.append(f"Workers here: {W}. AWS: {a.aws_cores} workers, per-run time / {ratio:.2f} "
                 f"(this machine vs c8g.48xlarge on the version5 code path).\n")
    lines.append("| stage | variant | runs | epochs | here (h) | target (h) | AWS (h) | AWS cost (USD) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for stage in ("stage0", "stage1", "stage2", "stage3"):
        report[stage] = []
        fits = False
        for label, jobs in variants(stage, sel1, sel2):
            jobs = mj.dedupe(jobs)
            sec = total(jobs, cost)
            here, aws = hours(sec, W), hours(sec / ratio, a.aws_cores)
            rec = {"variant": label, "runs": len(jobs), "epochs": sum(int(j.get("epochs", 40)) for j in jobs),
                   "hours_here": here, "hours_aws": aws, "usd_aws": aws * a.aws_price}
            report[stage].append(rec)
            lines.append(f"| {stage} | {label} | {len(jobs)} | {rec['epochs']} | {here:.1f} | {TARGET_H[stage]} | "
                         f"{aws:.1f} | {aws * a.aws_price:.0f} |")
            if label == "full" and here <= TARGET_H[stage]:
                fits = True
                break
            if label != "full" and here <= TARGET_H[stage]:
                break
        report[stage + "_fits_here_in_full"] = fits
    out = os.path.join(V6, "results", "bench")
    with open(os.path.join(out, "estimates.json"), "w") as f:
        json.dump(report, f, indent=1)
    with open(os.path.join(out, "estimates.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
