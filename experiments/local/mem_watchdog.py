"""Emulate the grid's 8 GB per-worker cap on macOS (RLIMIT_AS is not enforced).

Every few seconds, checks the RSS of each run_experiment worker and SIGKILLs it
(and its solver subprocesses) when it exceeds the cap; run_algorithm.py then
records the instance as status "memory". As on the grid, where RLIMIT_AS caps
the worker process only, the solver subprocesses (ipopt, scip) are not counted;
pass --with-solvers to count them too.
Usage: python experiments/local/mem_watchdog.py [CAP_GB] [PERIOD_S] [--with-solvers]
"""
import os, signal, subprocess, sys, time

WITH_SOLVERS = "--with-solvers" in sys.argv
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
CAP = float(ARGS[0] if len(ARGS) > 0 else 8) * 2**30
PERIOD = float(ARGS[1] if len(ARGS) > 1 else 3)

def procs():
    out = subprocess.run(["ps", "-axo", "pid=,ppid=,rss=,command="],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        pid, ppid, rss, cmd = line.split(None, 3)
        yield int(pid), int(ppid), int(rss) * 1024, cmd

while True:
    table = {p: (pp, r, c) for p, pp, r, c in procs()}
    runners = {p for p, (pp, r, c) in table.items() if "run_experiment.py" in c}
    workers = {p for p, (pp, r, c) in table.items() if pp in runners}
    total = {w: table[w][1] for w in workers}
    for p, (pp, r, c) in table.items():          # solver children of a worker
        if not WITH_SOLVERS:
            break
        q = pp
        while q in table and q not in workers and q > 1:
            q = table[q][0]
        if q in workers and p != q:
            total[q] += r
    for w, rss in total.items():
        if rss > CAP:
            print(f"{time.strftime('%H:%M:%S')} kill worker {w} "
                  f"({rss / 2**30:.1f} GB): {table[w][2][:120]}", flush=True)
            for p, (pp, r, c) in table.items():  # solvers first, then worker
                if pp == w:
                    os.kill(p, signal.SIGKILL)
            os.kill(w, signal.SIGKILL)
    time.sleep(PERIOD)
