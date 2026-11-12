#!/usr/bin/env python
"""Progress of the SLURM grid: job states, records per (set, run), errors.

    python experiments/slurm/status.py [--job ARRAY_JOB_ID] [--ev 0]
"""
import argparse
import collections
import glob
import json
import os
import re
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXACT_THRESHOLD = 15


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except Exception as e:  # noqa: BLE001
        return f"<{e}>"


def n_vars(path):
    m = re.search(r"_n(\d+)_", os.path.basename(path))
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", help="array job id (default: all lcn_grid jobs)")
    ap.add_argument("--ev", type=int, default=0)
    args = ap.parse_args()
    os.chdir(REPO)

    # --- SLURM states
    if args.job:
        out = sh(["sacct", "-n", "-X", "-P", "-j", args.job, "--format=JobID,State,Elapsed"])
        states = collections.Counter()
        bad = []
        for line in out.split("\n"):
            if not line.strip():
                continue
            jid, st, el = line.split("|")[:3]
            st = st.split()[0]
            # a pending array range ("123_[5-99%20]") stands for many tasks
            m = re.search(r"\[(\d+)-(\d+)", jid)
            k = int(m.group(2)) - int(m.group(1)) + 1 if m and st == "PENDING" else 1
            states[st] += k
            if st not in ("PENDING", "RUNNING", "COMPLETED"):
                bad.append(f"{jid} {st} {el}")
        print("SLURM:", ", ".join(f"{s}={c}" for s, c in sorted(states.items())) or "none")
        for b in bad:
            print("  not ok:", b)

    # --- records per (set, run)
    manifest = os.path.join("experiments", "slurm", "manifest.txt")
    pairs = [l.split() for l in open(manifest) if l.strip()]
    print(f"\n{'set':20s} {'run':14s} {'done/exp':>9s}  ok tmo mem err")
    tot_done = tot_exp = 0
    for d, r in pairs:
        inst = glob.glob(os.path.join("benchmarks", d, "**", "*.lcn"), recursive=True)
        if r.startswith("exact"):
            inst = [p for p in inst if (n_vars(p) or 0) <= EXACT_THRESHOLD]
        exp = len(inst)
        c = collections.Counter()
        for f in glob.glob(os.path.join("results", d, f"ev{args.ev}", f"{r}.shard*.jsonl")):
            for line in open(f):
                try:
                    c[json.loads(line).get("status", "?")] += 1
                except json.JSONDecodeError:
                    c["corrupt"] += 1
        done = sum(c.values())
        tot_done += done
        tot_exp += exp
        if done == 0:
            continue
        print(f"{d:20s} {r:14s} {done:4d}/{exp:<4d}  {c['ok']:3d} {c['timeout']:3d} "
              f"{c['memory']:3d} {c['error']:3d}")
    print(f"TOTAL records {tot_done}/{tot_exp} ({100 * tot_done / max(tot_exp, 1):.1f}%)")

    # --- errors in logs
    logs = sorted(glob.glob(os.path.join("results", "slurm_logs",
                                         f"{args.job or '*'}_*.err")))
    hits = []
    for f in logs:
        try:
            txt = open(f, errors="replace").read()
        except OSError:
            continue
        for pat in ("Traceback", "oom-kill", "OUT_OF_MEMORY", "DUE TO TIME LIMIT", "CANCELLED"):
            n = txt.count(pat)
            if n:
                hits.append(f"  {os.path.basename(f)}: {pat} x{n}")
    print("\nLog problems:" if hits else "\nLog problems: none")
    print("\n".join(hits))


if __name__ == "__main__":
    main()
