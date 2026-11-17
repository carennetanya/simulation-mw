# -*- coding: utf-8 -*-
"""
combined20.py - what happens when the operational decisions of Q3, Q4 and Q5 are COMBINED?

Setting (fixed): 4 pickers, 1 order start station + 1 packing station (tour: order start -> stops -> packing -> walk back), 1 batch = as many orders as fit into the item capacity (default 20 items), 100 test orders per wave, 1 step = 1 grid cell.
Decisions combined: batching (fcfs | optimized) x assignment (dynamic | static) x sequence (fcfs | spt)
Storage: random (mean of 3 layouts) | cluster (data mining, k chosen on validation orders) | RL v2
Every number is the mean over all splits (a split = one random train/test partition of the orders).
Needs studies20.py (batching 'optimized' is defined there), manual_sim.py, wh_sim.py, rl2.py, rl_storage.py.
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

CONFIGS = [  # (label, batching, assignment, sequence)
    ("A  baseline: FCFS batching + dynamic + FCFS sequence", "fcfs", "dynamic", "fcfs"),
    ("B  + SPT sequence", "fcfs", "dynamic", "spt"),
    ("C  + optimized batching", "optimized", "dynamic", "fcfs"),
    ("D  optimized batching + SPT sequence", "optimized", "dynamic", "spt"),
    ("E  D but static assignment", "optimized", "static", "spt"),
]
ST = [("random", "Random"), ("cluster", "Cluster (data mining, k validated)"), ("rl_v2", "RL v2")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="archive/order_products__train.csv")
    ap.add_argument("--products", default="archive/products.csv")
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--orders", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lam", type=float, default=6.4,
                    help="RL v2 pair weight; 6.4 was selected by lam_validation.py (validation orders held out from TRAIN data)")
    M.add_picking_args(ap)
    ap.add_argument("--out", default="combined20_runs.csv")
    a = ap.parse_args()
    a.synthetic, a.split_seed = False, 0
    data = M.load_data(a)
    ctx = M.Ctx(data, M.mcfg_from_args(a), SimpleNamespace(split_seed=0))
    S._WALK["w"] = ctx.walk
    secs, slots_of, D, dst, cap = R1.section_tables(ctx)
    print("LAYOUT: 15 racks R1-R15 (R1-R13 horizontal, R14/R15 vertical along the side walls), 91 sections = 546 slots = 546 SKUs; tour: ORDER START station -> pick stops -> PACKING station -> walk back to the order start station", flush=True)
    sidx = {s: k for k, s in enumerate(secs)}
    res = defaultdict(list)
    t_start = time.time()
    for sp in range(a.splits):
        skus, train, test = ctx.split(a.orders, sp)
        fasg = W.assign_frequency(skus, ctx.slots, train)
        casg, _ = M.cluster_plan(ctx, skus, train, n_val=a.orders)      # k chosen on validation orders from TRAIN
        f, C = rl2.dense_stats(skus, train)
        s0 = np.array([sidx[fasg[x].sec] for x in skus])
        s, _, _ = rl2.train_improve(f, C, D, dst, s0, lam=a.lam, seed=sp)
        v2 = rl2.to_assignment(s, skus, f, secs, slots_of)
        rnd = [W.assign_random(skus, ctx.slots, 1000 + sd) for sd in range(a.seeds)]
        for label, bt, asg_m, sq in CONFIGS:
            kw = dict(batching=bt, assign=asg_m, seq=sq)
            res[("random", label)].append(S.avg([ctx.run("combined", sp, "random", r, test, **kw) for r in rnd]))
            res[("cluster", label)].append(ctx.run("combined", sp, "cluster", casg, test, **kw))
            res[("rl_v2", label)].append(ctx.run("combined", sp, "rl_v2", v2, test, **kw))
        print(f"split {sp + 1}/{a.splits} done ({(time.time() - t_start) / 60:.1f} min)", flush=True)

    base_rand = res[("random", CONFIGS[0][0])]
    rows = []
    for st, stl in ST:
        base = res[(st, CONFIGS[0][0])]
        for label, _, _, _ in CONFIGS:
            r = res[(st, label)]
            is_b = label == CONFIGS[0][0]
            rows.append([stl, label[:1], M.num(S.mean_key(r, "walk_m")), M.num(S.mean_key(r, "makespan")),
                         M.num(S.mean_key(r, "avg_order_time")), M.num(S.mean_key(r, "station_wait")),
                         f"{S.mean_key(r, 'picker_util'):.2f}",
                         "-" if is_b else S.fi(S.improvements(base, r, "walk_m")),
                         "-" if is_b else S.fi(S.improvements(base, r, "makespan")),
                         "-" if is_b else M.fmt_p(M.paired_p([x["makespan"] for x in base], [x["makespan"] for x in r])),
                         "-" if is_b else S.fi(S.improvements(base, r, "avg_order_time")),
                         "-" if is_b else M.fmt_p(M.paired_p([x["avg_order_time"] for x in base], [x["avg_order_time"] for x in r]))])
    M.table(f"Table C1. Combined operating policy, {a.orders} orders, {ctx.mcfg.n_pickers} pickers, {a.station_mode} stations, {a.splits} splits. "
            f"Improvements (reductions) are vs configuration A of the SAME storage method (+ = better)",
            ["Storage", "Cfg", "Walking (m)", "Makespan", "Avg order time", "Station wait", "Picker util",
             "Walking improvement", "Makespan improvement", "p", "Order-time improvement", "p"], rows,
            "A = FCFS batching + dynamic + FCFS sequence | B = A + SPT | C = A + optimized batching | "
            "D = optimized batching + SPT | E = D with static assignment")
    rows = []
    for st, stl in ST:
        for label in (CONFIGS[0][0], CONFIGS[3][0]):
            r = res[(st, label)]
            rows.append([stl, label[:1], M.num(S.mean_key(r, "walk_m")), M.num(S.mean_key(r, "makespan")),
                         M.num(S.mean_key(r, "avg_order_time")),
                         S.fi(S.improvements(base_rand, r, "walk_m")), S.fi(S.improvements(base_rand, r, "makespan")),
                         M.fmt_p(M.paired_p([x["makespan"] for x in base_rand], [x["makespan"] for x in r])),
                         S.fi(S.improvements(base_rand, r, "avg_order_time")),
                         M.fmt_p(M.paired_p([x["avg_order_time"] for x in base_rand], [x["avg_order_time"] for x in r]))])
    M.table("Table C2. Overall effect: storage method + operating policy vs the simplest system "
            "(Random storage, configuration A)",
            ["Storage", "Cfg", "Walking (m)", "Makespan", "Avg order time", "Walking improvement", "Makespan improvement", "p",
             "Order-time improvement", "p"], rows)
    keys = ["study", "split", "strategy", "batching", "assign", "seq"] + \
           [k for k in ctx.runs[0] if k not in ("study", "split", "strategy", "batching", "assign", "seq")]
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(ctx.runs)
    print(f"\nAll runs written to {a.out} ({len(ctx.runs)} runs, {(time.time() - t_start) / 60:.1f} min)")


if __name__ == "__main__":
    main()