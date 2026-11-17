# -*- coding: utf-8 -*-
"""
sensitivity_level.py - does the ranking of the storage strategies depend on the size of the extra picking steps per level?

The model of the report adds extra steps for picking a SKU on level 1 (bottom) and level 3 (top); level 2 has none. The size of
these extra steps (default 1, 0, 2) is an assumption. This script repeats the comparison of the storage strategies for other
sizes. Plans are learned ONCE per split (training orders only) and then evaluated for every scenario, so the only thing that
changes between scenarios is the picking time.

Scenarios (extra steps on level 1, 2, 3):  0,0,0 (no level effect) | 1,0,2 (report) | 2,0,4 (double) | 2,0,2 (symmetric)
Operating policies:  A = FCFS batching + dynamic + FCFS sequence | D = optimized batching + dynamic + SPT sequence
Storage:  random (mean of 3 layouts) | frequency | cluster (data mining, k validated) | RL v2

Needs studies20.py, manual_sim.py, wh_sim.py, rl2.py, rl_storage.py, instacart_patch.py in the folder.
Run:  python sensitivity_level.py --level-extra 1,0,2 --out sensitivity_runs.csv > sensitivity_log.txt 2>&1
"""
import argparse
import csv
import statistics
import time
from collections import defaultdict
from types import SimpleNamespace

import numpy as np

import manual_sim as M
import rl2
import rl_storage as R1
import studies20 as S          # patches manual_sim.make_batches with the 'optimized' method
import wh_sim as W

CFGS = [("A", "fcfs", "dynamic", "fcfs"), ("D", "optimized", "dynamic", "spt")]
STORAGE = [("random", "Random"), ("frequency", "Frequency"), ("cluster", "Data mining (k validated)"), ("rl_v2", "RL v2")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="archive/order_products__train.csv")
    ap.add_argument("--products", default="archive/products.csv")
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--orders", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=3, help="random layouts per split")
    ap.add_argument("--lam", type=float, default=6.4)
    ap.add_argument("--scen", default="0,0,0;1,0,2;2,0,4;2,0,2", help="scenarios of extra steps (level 1,2,3), separated by ;")
    M.add_picking_args(ap)
    ap.add_argument("--out", default="sensitivity_runs.csv")
    a = ap.parse_args()
    a.synthetic, a.split_seed = False, 0
    scen = [tuple(int(x) for x in s.split(",")) for s in a.scen.split(";")]
    data = M.load_data(a)
    ctx = M.Ctx(data, M.mcfg_from_args(a), SimpleNamespace(split_seed=0))
    S._WALK["w"] = ctx.walk
    secs, slots_of, D, dst, cap = R1.section_tables(ctx)
    sidx = {s: k for k, s in enumerate(secs)}
    res = defaultdict(list)
    t_start = time.time()
    for sp in range(a.splits):
        skus, train, test = ctx.split(a.orders, sp)
        fasg = W.assign_frequency(skus, ctx.slots, train)
        casg, _ = M.cluster_plan(ctx, skus, train, n_val=a.orders)
        f, C = rl2.dense_stats(skus, train)
        s0 = np.array([sidx[fasg[x].sec] for x in skus])
        s, _, _ = rl2.train_improve(f, C, D, dst, s0, lam=a.lam, seed=sp)
        v2 = rl2.to_assignment(s, skus, f, secs, slots_of)
        rnd = [W.assign_random(skus, ctx.slots, 1000 + sd) for sd in range(a.seeds)]
        for sc in scen:
            for label, bt, asg_m, sq in CFGS:
                def kw():
                    return dict(batching=bt, assign=asg_m, seq=sq, level_extra=sc)
                key = ("-".join(map(str, sc)), label)
                res[key + ("random",)].append(S.avg([ctx.run("sens", sp, "random", r, test, **kw()) for r in rnd]))
                res[key + ("frequency",)].append(ctx.run("sens", sp, "frequency", fasg, test, **kw()))
                res[key + ("cluster",)].append(ctx.run("sens", sp, "cluster", casg, test, **kw()))
                res[key + ("rl_v2",)].append(ctx.run("sens", sp, "rl_v2", v2, test, **kw()))
        print(f"split {sp + 1}/{a.splits} done ({(time.time() - t_start) / 60:.1f} min)", flush=True)

    for label, bt, asg_m, sq in CFGS:
        rows = []
        for sc in scen:
            sk = "-".join(map(str, sc))
            base = res[(sk, label, "random")]
            freq = res[(sk, label, "frequency")]
            for st, stl in STORAGE:
                r = res[(sk, label, st)]
                is_b = st == "random"
                rows.append([sk, stl, M.num(S.mean_key(r, "walk_m")), M.num(S.mean_key(r, "makespan")), M.num(S.mean_key(r, "avg_order_time")),
                             "-" if is_b else S.fi(S.improvements(base, r, "makespan")),
                             "-" if is_b else M.fmt_p(M.paired_p([x["makespan"] for x in base], [x["makespan"] for x in r])),
                             "-" if st in ("random", "frequency") else S.fi(S.improvements(freq, r, "makespan")),
                             "-" if st in ("random", "frequency") else M.fmt_p(M.paired_p([x["makespan"] for x in freq], [x["makespan"] for x in r]))])
        M.table(f"Table L{label}. Sensitivity to the extra picking steps per level (level 1-2-3), configuration {label} "
                f"({bt} batching, {asg_m} assignment, {sq} sequence), {a.orders} orders, {ctx.mcfg.n_pickers} pickers, {a.splits} splits",
                ["Extra steps", "Storage", "Walking (m)", "Makespan", "Avg order time", "Makespan impr. vs random", "p",
                 "Makespan impr. vs frequency", "p"], rows,
                "Improvements are reductions in makespan (+ = better), mean +- sd over the splits; p = paired t-test over the splits. "
                "Walking distance does not depend on the extra steps.")
    keys = ["study", "split", "strategy", "batching", "assign", "seq", "level_extra"] + \
           [k for k in ctx.runs[0] if k not in ("study", "split", "strategy", "batching", "assign", "seq", "level_extra")]
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(ctx.runs)
    print(f"\nAll runs written to {a.out} ({len(ctx.runs)} runs, {(time.time() - t_start) / 60:.1f} min)")


if __name__ == "__main__":
    main()