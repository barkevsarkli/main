#!/usr/bin/env python3
"""Run a jobs file with N parallel trainer processes (standard library only).

    python3 sweep/run_jobs.py results/stage1/jobs.jsonl --workers 24

* Each job is one JSON object per line (see sweep/common.py for the fields).
* A run is finished when results/<stage>/rows/<run_id>.csv holds a valid row whose
  identity columns equal the job's canonical identity; finished runs are skipped, so
  re-running the same command continues where it stopped.
* Every process writes its own row file and stats file (atomically, via a temp file and a
  rename), so nothing is ever appended concurrently and an interrupted run leaves no
  half-written row. sweep/merge.py builds one CSV per stage afterwards.
* Progress: one line per finished job, and a summary (done/remaining, failures, elapsed,
  ETA from observed throughput) every 5 minutes.
* Failed jobs (non-zero exit other than status=nan/collapsed_abort, or no valid row) are
  logged with their stderr tail to results/<stage>/failures.log.
* Keeps the machine awake: SetThreadExecutionState on Windows, caffeinate on macOS,
  systemd-inhibit on Linux when available.
* Ctrl-C kills the children and exits; partial temp files are removed.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

V6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_jobs(path):
    jobs = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "<" in line:
                sys.exit(f"{path}:{n}: job contains an unresolved placeholder; finalise the stage first")
            j = json.loads(line)
            jobs.append((j, common.canonical(j)))
    return jobs


class KeepAwake:
    def __init__(self):
        self.proc = None

    def __enter__(self):
        try:
            if sys.platform == "win32":
                import ctypes
                ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
            elif sys.platform == "darwin" and shutil.which("caffeinate"):
                self.proc = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
            elif shutil.which("systemd-inhibit"):
                self.proc = subprocess.Popen(
                    ["systemd-inhibit", "--what=sleep:idle", "--who=v6-run_jobs", "--why=training sweep",
                     "--mode=block", "sleep", "infinity"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:  # never fatal
            print(f"[keep-awake] not available: {e}", flush=True)
        return self

    def __exit__(self, *exc):
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            except Exception:
                pass
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


def fmt_dur(s):
    s = int(s)
    return f"{s // 3600}h{(s % 3600) // 60:02d}m{s % 60:02d}s"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--binary", default=common.default_binary())
    ap.add_argument("--data", default=os.path.join(V6, ".."))
    ap.add_argument("--kernel-dir", default=os.path.join(V6, "kernels"))
    ap.add_argument("--results", default=os.path.join(V6, "results"), help="results root")
    ap.add_argument("--limit", type=int, default=0, help="run at most this many pending jobs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summary-every", type=float, default=300.0, help="seconds between summaries")
    a = ap.parse_args()

    jobs = load_jobs(a.jobs)
    pending, seen, n_done = [], set(), 0
    for j, c in jobs:
        rid = common.run_id(c)
        stage = j.get("stage", "manual")
        if (stage, rid) in seen:
            continue
        seen.add((stage, rid))
        row = common.read_row(os.path.join(a.results, stage, "rows", rid + ".csv"))
        if row is not None and row.get("run_id") == rid and common.row_matches(row, c):
            n_done += 1
        else:
            pending.append((j, c, rid, stage))
    total = n_done + len(pending)
    if a.limit:
        pending = pending[:a.limit]
    print(f"[{time.strftime('%H:%M:%S')}] {a.jobs}: {total} unique jobs, {n_done} finished, "
          f"{len(pending)} to run with {a.workers} workers", flush=True)
    if a.dry_run or not pending:
        return 0
    if not os.path.exists(a.binary):
        sys.exit(f"trainer binary not found: {a.binary} (build first: build.sh / build.ps1)")

    procs, lock = set(), threading.Lock()
    stop = threading.Event()
    t0 = time.time()
    counters = {"ok": 0, "fail": 0}

    def run_one(item):
        j, c, rid, stage = item
        sdir = os.path.join(a.results, stage)
        out = os.path.join(sdir, "rows", rid + ".csv")
        stats_every = int(j.get("stats_every", 1))
        stats = os.path.join(sdir, "stats", rid + ".csv") if stats_every > 0 else None
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if stats:
            os.makedirs(os.path.dirname(stats), exist_ok=True)
        argv = common.argv_for(c, a.binary, a.data, a.kernel_dir, out, stats, stats_every, rid, stage)
        if stop.is_set():
            return item, None, "", 0.0
        ts = time.time()
        p = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        with lock:
            procs.add(p)
        _, err = p.communicate()
        with lock:
            procs.discard(p)
        return item, p.returncode, err, time.time() - ts

    def on_int(signum, frame):
        stop.set()
        with lock:
            for p in list(procs):
                try:
                    p.kill()
                except Exception:
                    pass
        print("\ninterrupted: children killed; finished runs are kept, re-run to continue", flush=True)

    signal.signal(signal.SIGINT, on_int)
    last_summary = t0
    finished = 0
    with KeepAwake(), ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_one, it) for it in pending]
        for fut in as_completed(futs):
            item, rc, err, dt = fut.result()
            if rc is None or stop.is_set():
                continue
            j, c, rid, stage = item
            row = common.read_row(os.path.join(a.results, stage, "rows", rid + ".csv"))
            good = row is not None and row.get("run_id") == rid and common.row_matches(row, c) and rc in (0, 2)
            finished += 1
            if good:
                counters["ok"] += 1
                tag = row["status"]
            else:
                counters["fail"] += 1
                tag = f"FAILED rc={rc}"
                with open(os.path.join(a.results, stage, "failures.log"), "a") as f:
                    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {rid} rc={rc} key={common.run_key(c)}\n")
                    f.write("    " + " ".join(common.argv_for(c, a.binary, a.data, a.kernel_dir, "<out>")) + "\n")
                    for line in (err or "").strip().splitlines()[-20:]:
                        f.write("    | " + line + "\n")
            el = time.time() - t0
            rate = finished / el
            eta = (len(pending) - finished) / rate if rate > 0 else 0
            print(f"[{time.strftime('%H:%M:%S')}] {finished}/{len(pending)} {rid} {c['arch']} {c['dataset']} "
                  f"H={c['hidden']} lr={common.fmt(c['lr'])} seed={c['seed']} {tag} {dt:.0f}s | ETA {fmt_dur(eta)}",
                  flush=True)
            if time.time() - last_summary >= a.summary_every:
                last_summary = time.time()
                print(f"=== summary: done {n_done + counters['ok']}/{total}, remaining {len(pending) - finished}, "
                      f"failures {counters['fail']}, elapsed {fmt_dur(el)}, "
                      f"throughput {rate * 3600:.1f} jobs/h, ETA {fmt_dur(eta)} ===", flush=True)
    # Clean any temp files a killed child left behind.
    for stage in {it[3] for it in pending}:
        for sub in ("rows", "stats"):
            d = os.path.join(a.results, stage, sub)
            if os.path.isdir(d):
                for fn in os.listdir(d):
                    if fn.endswith(".tmp"):
                        os.remove(os.path.join(d, fn))
    el = time.time() - t0
    print(f"=== finished: ok {counters['ok']}, failed {counters['fail']}, elapsed {fmt_dur(el)} ===", flush=True)
    return 1 if counters["fail"] or stop.is_set() else 0


if __name__ == "__main__":
    sys.exit(main())
