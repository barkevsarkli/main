#!/usr/bin/env python3
"""Benchmark and budget measurements (PROMPT.md section 11). Writes results/bench/.

    python3 tools/bench.py [--workers-list 4,6,8] [--max-hours 2]

1. Worker scaling: W concurrent copies of one mid-size job (CIFAR-10, H = 128, cnn, K = 5,
   random, 1 epoch, diagnostics on as in the plan) for each W; throughput in jobs/hour
   and the spread of per-worker epoch times. The W with the best throughput is chosen.
2. Per-configuration cost: with the chosen W, all workers running copies of the same job,
   seconds per epoch (the trainer's epoch_seconds: training + evaluation + diagnostics)
   and the per-run fixed overhead (load, split, final test evaluations) for every
   (dataset, H) of the plan x {mlp1, mlp2, cnn K in {3,5,7,9} x {random, frozen}}.
   If the measured part-1 cost says this would exceed --max-hours, only K in {3, 9} is
   measured and the conv part is interpolated in K^2 (reported as such).
3. Drift: the part-1 job for 3 epochs with W workers; epoch-to-epoch time change
   (thermal throttling shows as rising epoch times).
4. AWS reference: `mlp2 --rng-mode legacy` (the version5 code path, bit-identical) on
   CIFAR-10 H = 128 under the same load. version5 measured 773-775 s for 40 epochs at
   190 concurrent workers on a c8g.48xlarge (Graviton4), i.e. ~19.3 s per epoch; the
   ratio converts this machine's timings into AWS estimates (tools/estimate.py).
"""
import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time

V6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V6, "sweep"))
import common  # noqa: E402

OUT = os.path.join(V6, "results", "bench")
RAW = os.path.join(OUT, "raw")
AWS_V5_SECONDS_PER_EPOCH = 774.0 / 40      # version5 c8g.48xlarge, P=190, CIFAR H=128 mlp2


def job_args(ds, arch, H, K=None, mode="random", epochs=1, seed=201, legacy=False):
    a = ["--arch", arch, "--dataset", ds, "--hidden", str(H), "--epochs", str(epochs), "--seed", str(seed),
         "--lr", "0.003", "--act1", "tanh", "--act2", "relu", "--ratio", "0.5"]
    if arch == "cnn":
        a += ["--conv-k", str(K), "--kernel-mode", mode]
        if mode != "random":
            a += ["--kernel-family", "gabor"]
    if legacy:
        a += ["--rng-mode", "legacy"]
    return a


def run_parallel(name, args, W):
    """W concurrent copies; returns wall time and the rows."""
    os.makedirs(RAW, exist_ok=True)
    procs, outs = [], []
    t0 = time.time()
    for i in range(W):
        out = os.path.join(RAW, f"{name}_w{W}_{i}.csv")
        st = os.path.join(RAW, f"{name}_w{W}_{i}_stats.csv")
        outs.append(out)
        cmd = [common.default_binary(), "--quiet", "--data", os.path.join(V6, ".."), "--kernel-dir",
               os.path.join(V6, "kernels"), "--out", out, "--stats-out", st] + args
        procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
    for p in procs:
        _, err = p.communicate()
        if p.returncode != 0:
            raise RuntimeError(f"{name}: {err.decode()[-500:]}")
    wall = time.time() - t0
    rows = [next(csv.DictReader(open(o))) for o in outs]
    for r, o in zip(rows, outs):
        st = o.replace(".csv", "_stats.csv")
        r["_stats_bytes"] = os.path.getsize(st)
        os.remove(st)
    return wall, rows


def epoch_secs(r):
    return [float(x) for x in r["epoch_seconds"].split("|")]


def conditions():
    c = {"platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
         "python": platform.python_version(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    if sys.platform == "darwin":
        for k, cmd in (("cpu", ["sysctl", "-n", "machdep.cpu.brand_string"]),
                       ("perf_cores", ["sysctl", "-n", "hw.perflevel0.physicalcpu"]),
                       ("eff_cores", ["sysctl", "-n", "hw.perflevel1.physicalcpu"]),
                       ("power", ["pmset", "-g", "batt"]), ("therm", ["pmset", "-g", "therm"])):
            try:
                c[k] = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
            except Exception:
                pass
        try:
            c["low_power_mode"] = [l for l in subprocess.run(["pmset", "-g"], capture_output=True, text=True)
                                   .stdout.splitlines() if "lowpowermode" in l]
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        try:
            c["cpu"] = [l for l in open("/proc/cpuinfo") if "model name" in l or "CPU part" in l][:1]
        except Exception:
            pass
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    n = os.cpu_count() or 1
    ap.add_argument("--workers-list", default=",".join(str(x) for x in sorted({max(1, n // 2), max(1, 3 * n // 4), n})))
    ap.add_argument("--max-hours", type=float, default=2.0)
    ap.add_argument("--hidden", default="64,128,256")
    a = ap.parse_args()
    if os.path.isdir(RAW):
        shutil.rmtree(RAW)
    res = {"conditions": conditions(), "binary_provenance": None}
    t_start = time.time()

    # 1. worker scaling
    scaling = []
    for W in [int(x) for x in a.workers_list.split(",")]:
        wall, rows = run_parallel("scale", job_args("cifar10", "cnn", 128, 5), W)
        es = [epoch_secs(r)[0] for r in rows]
        scaling.append({"workers": W, "wall_s": wall, "jobs_per_hour": W / wall * 3600,
                        "epoch_s_min": min(es), "epoch_s_max": max(es), "epoch_s_mean": sum(es) / len(es)})
        print(f"[scale] W={W}: {W / wall * 3600:.1f} jobs/h, epoch {min(es):.1f}-{max(es):.1f} s", flush=True)
    res["scaling"] = scaling
    res["binary_provenance"] = {k: rows[0][k] for k in ("git_hash", "git_dirty", "compiler", "build_flags", "platform")}
    W = max(scaling, key=lambda s: s["jobs_per_hour"])["workers"]
    res["chosen_workers"] = W
    print(f"[scale] chosen W = {W}", flush=True)

    # 2. per-configuration cost
    Hs = [int(x) for x in a.hidden.split(",")]
    per_job = scaling[[s["workers"] for s in scaling].index(W)]["wall_s"]
    n_cfg_full = 2 * len(Hs) * (2 + 8)
    ks = [3, 5, 7, 9]
    if n_cfg_full * per_job * 1.3 / 3600 > a.max_hours:
        ks = [3, 9]
        res["k_interpolated"] = True
        print(f"[cost] full grid would exceed {a.max_hours} h: measuring K in {{3, 9}}, interpolating in K^2")
    cost = []
    for ds in ("mnist", "cifar10"):
        for H in Hs:
            cfgs = [("mlp1", None, "random"), ("mlp2", None, "random")]
            cfgs += [("cnn", K, m) for K in ks for m in ("random", "frozen")]
            for arch, K, mode in cfgs:
                wall, rows = run_parallel(f"cost_{ds}_{arch}_{H}_{K}_{mode}", job_args(ds, arch, H, K, mode), W)
                es = [epoch_secs(r)[0] for r in rows]
                ov = [float(r["seconds"]) - sum(epoch_secs(r)) for r in rows]
                rec = {"dataset": ds, "arch": arch, "hidden": H, "conv_k": K or 0, "kernel_mode": mode if arch == "cnn" else "none",
                       "epoch_s": sum(es) / len(es), "epoch_s_max": max(es), "overhead_s": sum(ov) / len(ov),
                       "measured": True}
                cost.append(rec)
                print(f"[cost] {ds} {arch} H={H} K={K} {mode}: {rec['epoch_s']:.1f} s/epoch, overhead {rec['overhead_s']:.1f} s",
                      flush=True)
    if ks == [3, 9]:
        for c3 in [c for c in cost if c["conv_k"] == 3]:
            c9 = next(c for c in cost if c["conv_k"] == 9 and all(c[k] == c3[k] for k in ("dataset", "hidden", "kernel_mode")))
            for K in (5, 7):
                f = (K * K - 9) / (81 - 9)
                cost.append(dict(c3, conv_k=K, measured=False,
                                 epoch_s=c3["epoch_s"] + f * (c9["epoch_s"] - c3["epoch_s"]),
                                 epoch_s_max=None, overhead_s=c3["overhead_s"] + f * (c9["overhead_s"] - c3["overhead_s"])))
    res["cost"] = cost

    # 3. drift
    wall, rows = run_parallel("drift", job_args("cifar10", "cnn", 128, 5, epochs=3), W)
    res["drift"] = [epoch_secs(r) for r in rows]
    # stats size: 4 measurement points (epochs 0-3) -> per epoch, and extrapolated to 40 epochs
    per_point = rows[0]["_stats_bytes"] / 4
    res["stats_bytes_per_epoch"] = per_point
    res["stats_bytes_per_40_epoch_run"] = per_point * 41
    print("[drift] epoch seconds per worker:", [[round(x, 1) for x in e] for e in res["drift"]], flush=True)

    # 4. AWS reference
    wall, rows = run_parallel("aws_ref", job_args("cifar10", "mlp2", 128, legacy=True), W)
    es = [epoch_secs(r)[0] for r in rows]
    res["aws_reference"] = {"this_machine_epoch_s": sum(es) / len(es), "aws_v5_epoch_s": AWS_V5_SECONDS_PER_EPOCH,
                            "ratio_this_over_aws": (sum(es) / len(es)) / AWS_V5_SECONDS_PER_EPOCH}
    print(f"[aws] mlp2 legacy CIFAR H=128: {sum(es) / len(es):.1f} s/epoch here vs "
          f"{AWS_V5_SECONDS_PER_EPOCH:.1f} on c8g.48xlarge", flush=True)
    res["bench_wall_s"] = time.time() - t_start
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "bench.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {OUT}/bench.json ({res['bench_wall_s'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
