# -*- coding: utf-8 -*-
"""
compare_rl.py - paired tests between storage methods, from studies20_runs.csv (no simulation is run).

For every split there is ONE run per method (study Q1, FCFS batching, dynamic assignment, FCFS sequence), so the methods
can be compared split by split (paired t-test over the splits, two-sided, same function paired_p as in the report).

Usage (in the folder with manual_sim.py and studies20_runs.csv):
    python compare_rl.py
    python compare_rl.py --csv studies20_runs.csv --out compare_rl_out.txt
"""
import argparse
import csv
import statistics
from collections import defaultdict

import manual_sim as M

METRICS = [("walk_m", "Walking distance"), ("makespan", "Makespan"), ("avg_order_time", "Avg order time")]
# (new method, reference method): "improvement" = reduction of the new method compared with the reference
PAIRS = [("rl_v2", "frequency"), ("rl_v2", "cluster"), ("rl_v2", "rl_v1"), ("cluster", "frequency"), ("frequency", "random")]


def holm(ps):
    """Holm-adjusted p-values for a list of p-values."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="studies20_runs.csv")
    ap.add_argument("--out", default="compare_rl_out.txt")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv, newline="", encoding="utf-8")))
    # Q1 runs with the default operating policy only (batching/assign/seq are empty in Q1)
    q1 = [r for r in rows if r["study"] == "Q1" and r.get("batching", "") == ""]
    by = defaultdict(lambda: defaultdict(list))            # by[strategy][split] -> list of runs (random has several layouts)
    for r in q1:
        by[r["strategy"]][int(r["split"])].append(r)
    splits = sorted(set.intersection(*[set(by[s]) for s in {x for p in PAIRS for x in p}]))
    print(f"splits with all methods: {len(splits)}")

    def vals(strategy, metric):
        return [statistics.mean(float(x[metric]) for x in by[strategy][sp]) for sp in splits]

    lines = []
    for metric, label in METRICS:
        res = []
        for new, ref in PAIRS:
            vn, vr = vals(new, metric), vals(ref, metric)
            imp = [100 * (b - n) / b for n, b in zip(vn, vr)]          # % reduction of `new` vs `ref`, per split
            res.append((new, ref, statistics.mean(vr), statistics.mean(vn), statistics.mean(imp), statistics.stdev(imp), M.paired_p(vr, vn)))
        adj = holm([r[-1] for r in res])
        lines.append(f"\n{label} - {len(splits)} splits (improvement = reduction of the first method compared with the second; + = better)")
        lines.append(f"{'comparison':28s} | {'second':>9s} | {'first':>9s} | {'improvement (mean +- sd)':>26s} | {'p (paired t)':>12s} | {'p (Holm)':>9s}")
        for (new, ref, mr, mn, mi, sd, p), pa in zip(res, adj):
            lines.append(f"{new + ' vs ' + ref:28s} | {mr:9.1f} | {mn:9.1f} | {mi:+9.1f}% +- {sd:4.1f}        | {M.fmt_p(p):>12s} | {M.fmt_p(pa):>9s}")
    text = "\n".join(lines)
    print(text)
    open(a.out, "w", encoding="utf-8").write(text + "\n")
    print(f"\nwritten to {a.out}")


if __name__ == "__main__":
    main()