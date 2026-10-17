"""Shared job logic: canonical run identity, run key, run id, command line.

A job is a dict of trainer settings (one JSON object per line in a jobs file). The *run
key* is built from every identity column of the result row except run_id and stage, after
the same canonicalisation the trainer applies (src/main.cpp, make_config), so that a run
is recognised as finished whatever spelling produced it. The run id is FNV-1a 64 of the
run key, as 16 hex digits.
"""
import csv
import os
import sys

IDENTITY = [
    "arch", "dataset", "seed", "lr", "epochs", "loss", "rng_mode",
    "conv_k", "conv_channels", "pool", "kernel_mode", "kernel_family", "kernel_seed",
    "conv_act1", "conv_act2", "conv_ratio", "conv_mask_offset",
    "hid_act1", "hid_act2", "hid_ratio", "layout", "hidden", "abort_on_collapse",
]

DEFAULTS = {
    "arch": "cnn", "dataset": "mnist", "seed": 1, "lr": 0.01, "epochs": 40, "loss": "ce",
    "rng_mode": "split", "conv_k": 5, "conv_channels": 8, "pool": 2, "kernel_mode": "random",
    "kernel_family": "none", "kernel_seed": 20261018, "conv_act1": "relu", "conv_act2": "relu",
    "conv_ratio": 0.0, "conv_mask_offset": 0, "hid_act1": "relu", "hid_act2": "relu",
    "hid_ratio": 0.0, "layout": "interleave", "hidden": 128, "abort_on_collapse": 0,
}

INT_FIELDS = {"seed", "epochs", "conv_k", "conv_channels", "pool", "kernel_seed",
              "conv_mask_offset", "hidden", "abort_on_collapse"}
FLOAT_FIELDS = {"lr", "conv_ratio", "hid_ratio"}

# Non-identity job fields the runner understands.
RUN_FIELDS = {"stage", "stats_every"}

ACTS = ("relu", "leaky_relu", "tanh", "sigmoid")


def fmt(v):
    """Format a value exactly as the trainer prints it (C++ ostream, precision 6)."""
    if isinstance(v, float):
        s = "%.6g" % v
        return s
    return str(v)


def canon_mask(a1, a2, r):
    if r <= 0 or a1 == a2:
        return a1, a1, 0.0
    if r >= 1:
        return a2, a2, 0.0
    return a1, a2, r


def canonical(job):
    """Identity fields of a job after the trainer's canonicalisation rules."""
    unknown = set(job) - set(IDENTITY) - RUN_FIELDS
    if unknown:
        raise ValueError(f"unknown job fields: {sorted(unknown)}")
    c = dict(DEFAULTS)
    c.update({k: v for k, v in job.items() if k in IDENTITY})
    for k in INT_FIELDS:
        c[k] = int(c[k])
    for k in FLOAT_FIELDS:
        c[k] = float(c[k])
    # float32 round trip: the trainer parses with stof and prints float32.
    import struct
    for k in FLOAT_FIELDS:
        c[k] = struct.unpack("f", struct.pack("f", c[k]))[0]
    c["hid_act1"], c["hid_act2"], c["hid_ratio"] = canon_mask(c["hid_act1"], c["hid_act2"], c["hid_ratio"])
    if c["arch"] == "cnn":
        if c["kernel_mode"] == "random":
            c["kernel_family"] = "none"
        if c["kernel_family"] != "random_frozen":
            c["kernel_seed"] = 0
        c["conv_act1"], c["conv_act2"], c["conv_ratio"] = canon_mask(c["conv_act1"], c["conv_act2"], c["conv_ratio"])
        if c["conv_ratio"] == 0.0:
            c["conv_mask_offset"] = 0
    else:
        c.update(conv_k=0, conv_channels=0, pool=0, kernel_mode="none", kernel_family="none",
                 kernel_seed=0, conv_act1="none", conv_act2="none", conv_ratio=0.0, conv_mask_offset=0)
    return c


def run_key(c):
    return "|".join(f"{k}={fmt(c[k])}" for k in IDENTITY)


def fnv1a64(s):
    h = 0xcbf29ce484222325
    for b in s.encode():
        h ^= b
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def run_id(c):
    return fnv1a64(run_key(c))


def argv_for(c, binary, data, kernel_dir, out, stats_out=None, stats_every=1, run_id_=None, stage="manual"):
    """Trainer command line for canonical job c."""
    a = [binary, "--quiet", "--data", data, "--out", out, "--stage", stage, "--run-id", run_id_ or run_id(c)]
    for k in ("arch", "dataset", "seed", "lr", "epochs", "loss", "rng_mode", "hid_act1", "hid_act2",
              "hid_ratio", "layout", "hidden", "abort_on_collapse"):
        a += ["--" + k.replace("_", "-"), fmt(c[k])]
    if c["arch"] == "cnn":
        for k in ("conv_k", "conv_channels", "pool", "kernel_mode", "conv_act1", "conv_act2",
                  "conv_ratio", "conv_mask_offset"):
            a += ["--" + k.replace("_", "-"), fmt(c[k])]
        if c["kernel_mode"] != "random":
            a += ["--kernel-family", c["kernel_family"], "--kernel-dir", kernel_dir]
        if c["kernel_family"] == "random_frozen":
            a += ["--kernel-seed", fmt(c["kernel_seed"])]
    if stats_out:
        a += ["--stats-out", stats_out, "--stats-every", str(stats_every)]
    return a


def read_row(path):
    """The single result row of a per-run file, or None if absent/invalid."""
    try:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    except (FileNotFoundError, OSError):
        return None
    if len(rows) != 1:
        return None
    r = rows[0]
    if r.get("test_confusion", "").count("|") != 99:
        return None
    return r


def row_matches(row, c):
    """True if the row's identity columns equal canonical job c."""
    return all(row.get(k) == fmt(c[k]) for k in IDENTITY)


def default_binary():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exe = "main.exe" if sys.platform == "win32" else "main"
    return os.path.join(here, "bin", exe)
