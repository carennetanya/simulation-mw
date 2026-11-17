# -*- coding: utf-8 -*-
"""
batching_vs_exact.py - how far are the batching heuristics from the EXACT optimum? (small instances)

Batching model (item capacity): a batch may contain any set of orders whose total number of items (order lines) is at most the
capacity Q (default 20; an order larger than Q forms a batch of its own). Cost of a batch = length of the tour
order start station -> all stops of the batch -> packing station, plus the walk back to the order start station (tsp(): exact Held-Karp up to 9 stops, nearest neighbour + 2-opt + or-opt above).
Set-partitioning model (the "MIP"):   minimize  sum_S c_S x_S   s.t.  sum_{S contains i} x_S = 1 for every order i,  x_S in {0,1}
over ALL feasible batches S. The model is solved exactly by dynamic programming over subsets of the orders.
Lower bound LB: the LP relaxation of the same model (0 <= x_S <= 1 instead of x_S in {0,1}) over the same feasible batches,
solved with scipy.optimize.linprog (HiGHS); LB <= optimum.
Table E1 compares fcfs / similarity / section / optimized batching with this optimum (same tour function, so the gap is only due to
the batching decision) and reports the lower bound LB.
Table E2 compares the tour heuristic with the exact Held-Karp tour for 10-12 and 13-15 stops.
Usage:  python batching_vs_exact.py --instances 10 --tsp-batches 40
Needs studies20.py (batching 'optimized' is defined there), manual_sim.py, wh_sim.py, instacart_patch.py (or the data in archive/), scipy.
"""
import argparse
import csv
import itertools
import random
import statistics
import time
from types import SimpleNamespace

import numpy as np

import manual_sim as M
import studies20 as S          # patches manual_sim.make_batches with the 'optimized' method
import wh_sim as W


def tour_len(walk, cells):
    """tour: order start station -> all stops -> packing station, plus the walk back to the order start station."""
    cells = sorted(set(cells))
    if not cells:
        return 0
    stp = walk.svc[0]
    order = M.tsp(walk, walk.start, cells, end=stp)
    return M.route_total(walk, walk.start, order, stp) + walk.d[(stp, walk.start)]


def optimum(walk, order_cells, items, cap):
    """exact optimum of the item-capacity batching: set partitioning over ALL feasible batches, solved by dynamic programming
    over subsets (the lowest order that is not yet batched must be in the next batch).
    -> (optimum, LB) where LB is the optimum of the LP relaxation over the same feasible batches."""
    n = len(order_cells)
    cache = {}
    masks = [[] for _ in range(n)]
    costs = [[] for _ in range(n)]
    for r in range(1, n + 1):
        feas_any = False
        for S_ in itertools.combinations(range(n), r):
            if r > 1 and sum(items[i] for i in S_) > cap:
                continue
            feas_any = True
            cells = frozenset().union(*[order_cells[i] for i in S_])
            if cells not in cache:
                cache[cells] = tour_len(walk, cells)
            masks[S_[0]].append(sum(1 << i for i in S_))
            costs[S_[0]].append(cache[cells])
        if not feas_any and r > 1:
            break
    masks = [np.array(m, dtype=np.int64) for m in masks]
    costs = [np.array(c, dtype=float) for c in costs]
    full = (1 << n) - 1
    g = np.full(1 << n, np.inf)
    g[0] = 0.0
    for R in range(full):
        if not np.isfinite(g[R]):
            continue
        u = ((R + 1) & -(R + 1)).bit_length() - 1          # lowest order not batched yet
        ok = (masks[u] & R) == 0
        np.minimum.at(g, R | masks[u][ok], g[R] + costs[u][ok])
    lb = lp_bound(n, np.concatenate(masks), np.concatenate(costs))
    return float(round(g[full], 6)), lb


def lp_bound(n, masks, costs):
    """LP relaxation of the set-partitioning model: min sum c_S x_S, sum_{S contains i} x_S = 1, 0 <= x_S <= 1."""
    from scipy.optimize import linprog
    A = ((masks[None, :] >> np.arange(n, dtype=np.int64)[:, None]) & 1).astype(float)   # A[i, S] = 1 if order i is in batch S
    res = linprog(costs, A_eq=A, b_eq=np.ones(n), bounds=(0, 1), method="highs")
    if res.status != 0:
        raise RuntimeError(f"LP relaxation not solved: {res.message}")
    return float(round(res.fun, 6))


def held_karp_path(walk, cells):
    """exact tour: order start station -> all cells -> packing station (numpy Held-Karp, up to ~15 stops)."""
    start, end = walk.start, walk.svc[0]
    m = len(cells)
    d0 = np.array([walk.d[(start, c)] for c in cells], dtype=float)
    dE = np.array([walk.d[(c, end)] for c in cells], dtype=float)
    D = np.array([[walk.d[(a, b)] for b in cells] for a in cells], dtype=float)
    INF = 1e18
    dp = np.full((1 << m, m), INF)
    for j in range(m):
        dp[1 << j, j] = d0[j]
    bits = np.arange(m)
    for mask in range(1, 1 << m):
        row = dp[mask]
        if row.min() >= INF:
            continue
        v = (row[:, None] + D).min(axis=0)
        free = [k for k in range(m) if not (mask >> k) & 1]
        for k in free:
            nm = mask | (1 << k)
            if v[k] < dp[nm, k]:
                dp[nm, k] = v[k]
    return float((dp[(1 << m) - 1] + dE).min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="archive/order_products__train.csv")
    ap.add_argument("--products", default="archive/products.csv")
    ap.add_argument("--sizes", type=int, nargs="+", default=[12, 16])
    ap.add_argument("--instances", type=int, default=10)
    ap.add_argument("--tsp-batches", type=int, default=40)
    M.add_picking_args(ap)
    ap.add_argument("--out", default="batching_vs_exact.csv")
    a = ap.parse_args()
    a.synthetic, a.split_seed = False, 0
    ctx = M.Ctx(M.load_data(a), M.mcfg_from_args(a), SimpleNamespace(split_seed=0))
    S._WALK["w"] = ctx.walk
    walk, cfg, st0 = ctx.walk, ctx.mcfg, ctx.walk.svc[0]
    cap = cfg.batch_items
    skus, train, test = ctx.split(100, 0)
    casg, _ = M.cluster_plan(ctx, skus, train, n_val=100)
    t0 = time.time()
    rows = []
    for n in a.sizes:
        for inst in range(a.instances):
            ords = random.Random(1000 * n + inst).sample(test, n)
            for stname in ("cluster", "random"):
                asg = casg if stname == "cluster" else W.assign_random(skus, ctx.slots, 1000 + inst)
                cells = [frozenset(asg[s].access for s in o) for o in ords]
                items = [len(set(o)) for o in ords]
                opt, lb = optimum(walk, cells, items, cap)
                for bt in ("fcfs", "similarity", "section", "optimized"):
                    groups = M.make_batches(ords, asg, cfg.batch_orders, bt, cap=cap)
                    tot = sum(tour_len(walk, frozenset().union(*[cells[i] for i in g])) for g in groups)
                    rows.append(dict(orders=n, storage=stname, instance=inst, batching=bt, tour_length=tot, optimum=opt, lb=lb,
                                     gap_pct=100 * (tot - opt) / opt, opt_vs_lb_pct=100 * (opt - lb) / lb))
            print(f"n={n} instance {inst + 1}/{a.instances} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    out = []
    for n in a.sizes:
        for stname in ("cluster", "random"):
            for bt in ("fcfs", "optimized", "section", "similarity"):
                r = [x for x in rows if x["orders"] == n and x["storage"] == stname and x["batching"] == bt]
                out.append([n, stname, bt, M.num(statistics.mean(x["tour_length"] for x in r)), M.num(statistics.mean(x["optimum"] for x in r)),
                            f"{statistics.mean(x['gap_pct'] for x in r):.1f}%", f"{max(x['gap_pct'] for x in r):.1f}%",
                            f"{sum(1 for x in r if x['gap_pct'] < 1e-6)}/{len(r)}"])
    M.table(f"Table E1. Batching heuristics vs the exact optimum (set-partitioning model) (item capacity {cap}; total tour length order start -> stops -> packing -> back to order start, steps; "
            f"{a.instances} instances per size and storage plan)",
            ["Orders", "Storage", "Batching", "Tour length (steps)", "Optimum (steps)", "Mean gap to optimum", "Max gap", "Optimal found"], out,
            "gap = (heuristic - optimum) / optimum, lower = better; 0 means the heuristic found the optimal batching.\n"
            "optimized = section batches improved by swapping orders (uses a nearest-neighbour tour estimate inside the search).")
    # ---- E1b: lower bound LB (LP relaxation) vs optimum, one value per instance
    out_lb = []
    for n in a.sizes:
        for stname in ("cluster", "random"):
            r = [x for x in rows if x["orders"] == n and x["storage"] == stname and x["batching"] == "fcfs"]
            out_lb.append([n, stname, M.num(statistics.mean(x["lb"] for x in r)), M.num(statistics.mean(x["optimum"] for x in r)),
                           f"{statistics.mean(x['opt_vs_lb_pct'] for x in r):.2f}%", f"{max(x['opt_vs_lb_pct'] for x in r):.2f}%",
                           f"{sum(1 for x in r if x['opt_vs_lb_pct'] < 1e-6)}/{len(r)}"])
    M.table(f"Table E1b. Lower bound LB (LP relaxation of the set-partitioning model) vs the exact optimum ({a.instances} instances per size and storage plan)",
            ["Orders", "Storage", "LB (steps)", "Optimum (steps)", "Mean gap optimum vs LB", "Max gap", "LB = optimum"], out_lb,
            "gap = (optimum - LB) / LB; 0 means the LP relaxation already gives an integer (optimal) batching.")
    # ---- E2: tour heuristic vs exact Held-Karp
    rng = random.Random(7)
    asg = W.assign_random(skus, ctx.slots, 1000)
    rows2 = {"10-12": [], "13-15": []}
    allords = test + train[:3000]
    tries = 0
    while min(len(v) for v in rows2.values()) < a.tsp_batches and tries < 200000:
        tries += 1
        g = rng.sample(allords, rng.choice([2, 3, 4, 5, 6]))
        cells = sorted({asg[s].access for o in g for s in o})
        k = len(cells)
        cls = "10-12" if 10 <= k <= 12 else ("13-15" if 13 <= k <= 15 else None)
        if cls is None or len(rows2[cls]) >= a.tsp_batches:
            continue
        heur = M.route_total(walk, walk.start, M.tsp(walk, walk.start, cells, end=st0), st0)
        exact = held_karp_path(walk, cells)
        rows2[cls].append((heur, exact))
        if len(rows2[cls]) == a.tsp_batches:
            print(f"TSP check {cls} stops done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    out2 = []
    for cls, v in rows2.items():
        gaps = [100 * (h - e) / e for h, e in v]
        out2.append([cls, len(v), f"{statistics.mean(gaps):.2f}%", f"{max(gaps):.2f}%", f"{sum(1 for g in gaps if g < 1e-6)}/{len(v)}"])
    M.table("Table E2. TSP heuristic (nearest neighbour + 2-opt + or-opt) vs exact Held-Karp, by number of stops of the batch",
            ["Stops", "Batches", "Mean gap", "Max gap", "Optimal found"], out2,
            "tour order start station -> all stops -> packing station (no walk back); batches = random groups of 2-6 orders of the data (random storage);\n"
            "tsp() uses exact Held-Karp up to 9 stops, so these classes are exactly where the heuristic is used.")
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["orders", "storage", "instance", "batching", "tour_length", "optimum", "lb", "gap_pct", "opt_vs_lb_pct"])
        w.writeheader()
        w.writerows(rows)
    print(f"written {a.out} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()