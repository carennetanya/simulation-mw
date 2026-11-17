# -*- coding: utf-8 -*-
"""
studies20.py - answers the professor's questions for MANUAL picking, all with many splits.

Setting (fixed): P pickers (default 4) walk in the aisles of the real warehouse layout (order start station -> stops -> packing station -> walk back), 1 batch = as many orders as fit into the item capacity (default 20 items),
1 step = walking 1 grid cell. Only the decision under study changes.

  Q1  number of clusters for 546 SKUs, data mining vs reinforcement learning (RL)
  Q2  how many orders have to be picked (orders per wave)
  Q3  how to batch orders: FCFS vs similarity vs section vs OPTIMIZED (local search on the estimated tour length)
  Q4  how to assign batches to pickers: dynamic vs static
  Q5  how to sequence the batches of each picker: FCFS vs SPT (shortest tour first)

A split = one random partition of all orders into TRAIN (80%, used to build clusters / learn the RL plan) and TEST
(unseen orders that are picked). Every number is the mean over all splits.
Storage plans used in Q2-Q5: random (baseline, mean of 3 layouts), cluster (data mining), RL v2.
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
from kmeans_plan import kmeans_clusters
import wh_sim as W

# --------------------------------------------------------------------------
# OPTIMIZED batching (Q3): start from the section batches, then swap orders between batches
# as long as the total estimated walking tour gets shorter.
# --------------------------------------------------------------------------
_orig_make_batches = M.make_batches
_WALK = {}


def est_tour(walk, cells):
    """nearest-neighbour estimate of the tour: ORDER START station -> all stops -> PACKING station (in steps).
    Same tour definition as the simulation (manual_sim.simulate), only the visiting order is found by nearest neighbour."""
    d = walk.d
    rem = set(cells)
    cur, tot = walk.start, 0
    while rem:
        nx = min(rem, key=lambda c: (d[(cur, c)], c))
        tot += d[(cur, nx)]
        rem.discard(nx)
        cur = nx
    return tot + walk.end_cost[cur]


def batches_optimized(walk, order_skus, slot_of, B, max_pass=4, cap=0):
    batches = [list(b) for b in _orig_make_batches(order_skus, slot_of, B, "section", cap=cap)]
    n_items = [len(set(o)) for o in order_skus]

    def fits(batch):          # item capacity (an order larger than the capacity stays alone in its batch)
        return cap <= 0 or len(batch) == 1 or sum(n_items[i] for i in batch) <= cap

    cells = [frozenset(slot_of[s].access for s in o) for o in order_skus]
    cache = {}

    def cost(batch):
        key = tuple(sorted(batch))
        if key not in cache:
            cache[key] = est_tour(walk, frozenset().union(*[cells[i] for i in batch]))
        return cache[key]

    for _ in range(max_pass):
        improved = False
        for a in range(len(batches)):
            for b in range(a + 1, len(batches)):
                for ia in range(len(batches[a])):
                    for ib in range(len(batches[b])):
                        A, Bb = batches[a], batches[b]
                        old = cost(A) + cost(Bb)
                        A2, B2 = list(A), list(Bb)
                        A2[ia], B2[ib] = Bb[ib], A[ia]
                        if fits(A2) and fits(B2) and cost(A2) + cost(B2) < old - 1e-9:
                            batches[a], batches[b] = A2, B2
                            improved = True
        if not improved:
            break
    return batches


def make_batches_patched(order_skus, slot_of, B, method, cap=0):
    if method == "optimized":
        return batches_optimized(_WALK["w"], order_skus, slot_of, B, cap=cap)
    return _orig_make_batches(order_skus, slot_of, B, method, cap=cap)


M.make_batches = make_batches_patched

# --------------------------------------------------------------------------
KEYS = ["walk_m", "walk_m_per_order", "makespan", "avg_order_time", "avg_batch_time", "station_wait",
        "avg_stops", "picker_util", "station_util", "batches"]


def avg(rr):
    return {k: statistics.mean(r[k] for r in rr) for k in KEYS}


def improvements(base, new, key):
    return [M.pct(b[key], n[key]) for b, n in zip(base, new)]

def fi(g):
    return f"{statistics.mean(g):.1f}% ± {statistics.pstdev(g):.1f}"     

def mean_key(lst, key):
    return statistics.mean(r[key] for r in lst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="archive/order_products__train.csv")
    ap.add_argument("--products", default="archive/products.csv")
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--orders", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=3, help="random layouts per split")
    ap.add_argument("--lam", type=float, default=6.4,
                    help="RL v2 pair weight; 6.4 was selected by lam_validation.py (validation orders held out from TRAIN data)")
    ap.add_argument("--v1-episodes", type=int, default=4000)
    M.add_picking_args(ap)
    ap.add_argument("--out", default="studies20_runs.csv")
    a = ap.parse_args()
    a.synthetic, a.split_seed = False, 0
    data = M.load_data(a)
    ctx = M.Ctx(data, M.mcfg_from_args(a), SimpleNamespace(split_seed=0))
    _WALK["w"] = ctx.walk
    secs, slots_of, D, dst, cap = R1.section_tables(ctx)
    print("LAYOUT: 15 racks R1-R15 (R1-R13 horizontal, R14/R15 vertical along the side walls), 91 sections = 546 slots = 546 SKUs; tour: ORDER START station -> pick stops -> PACKING station -> walk back to the order start station", flush=True)
    sidx = {s: k for k, s in enumerate(secs)}
    print(f"SKU={len(ctx.slots)} slots | pickers={ctx.mcfg.n_pickers} stations={ctx.mcfg.n_stations} "
          f"batch capacity={ctx.mcfg.batch_items} items (0 = {ctx.mcfg.batch_orders} orders) | splits={a.splits} | test orders per wave={a.orders}", flush=True)

    KS = M.K_GRID                                  # Q1: number of clusters k (cluster sizes are free)
    SIZES = [50, 100, 200, 300]
    BATCH = ["fcfs", "similarity", "section", "optimized"]
    COMBO = [(x, y) for x in ("dynamic", "static", "balanced") for y in ("fcfs", "spt")]
    MODES = ["pooled"]                             # Q4/Q5: one packing station, so "dedicated" would be identical to "pooled"
    Q1, Q2, Q3, Q4 = (defaultdict(list) for _ in range(4))
    kchosen = []
    t_start = time.time()
    for sp in range(a.splits):
        t0 = time.time()
        skus, train, test = ctx.split(a.orders, sp)
        fasg = W.assign_frequency(skus, ctx.slots, train)
        ck, fq = W.cluster_skus_k(skus, train, KS)
        cl = {k: W.pack_clusters(ck[k], fq, ctx.slots) for k in KS}
        km = {k: W.pack_clusters(kmeans_clusters(skus, train, k, sp), fq, ctx.slots) for k in KS}   # k-means comparison
        k_sel, _ = M.select_k(ctx, skus, train, a.orders)          # k of the 'cluster' plan: chosen on validation orders (TRAIN only)
        kchosen.append(k_sel)
        csel, _ = W.assign_cluster_k(skus, ctx.slots, train, k_sel)
        v1, _, _ = R1.train_rl(skus, train, ctx, episodes=a.v1_episodes, seed=sp, verbose=False)
        f, C = rl2.dense_stats(skus, train)
        s0 = np.array([sidx[fasg[x].sec] for x in skus])
        s, _, _ = rl2.train_improve(f, C, D, dst, s0, lam=a.lam, seed=sp)
        v2 = rl2.to_assignment(s, skus, f, secs, slots_of)
        rnd = [W.assign_random(skus, ctx.slots, 1000 + sd) for sd in range(a.seeds)]
        t1 = time.time()

        # ---- Q1 : clusters / data mining vs RL (FCFS batching, dynamic, FCFS sequence)
        Q1["random"].append(avg([ctx.run("Q1", sp, "random", r, test) for r in rnd]))
        Q1["frequency"].append(ctx.run("Q1", sp, "frequency", fasg, test))
        for k in KS:
            Q1[f"k{k}"].append(ctx.run("Q1", sp, f"cluster_k{k}", cl[k], test))
        for k in KS:
            Q1[f"km{k}"].append(ctx.run("Q1", sp, f"kmeans_k{k}", km[k], test))
        Q1["cluster"].append(ctx.run("Q1", sp, "cluster", csel, test))
        Q1["rl_v1"].append(ctx.run("Q1", sp, "rl_v1", v1, test))
        Q1["rl_v2"].append(ctx.run("Q1", sp, "rl_v2", v2, test))

        plans = {"cluster": csel, "rl_v2": v2}

        # ---- Q2 : number of orders per wave
        for n in SIZES:
            _, _, tn = ctx.split(n, sp)
            Q2[(n, "random")].append(avg([ctx.run(f"Q2_n{n}", sp, "random", r, tn) for r in rnd]))
            for name, asg in plans.items():
                Q2[(n, name)].append(ctx.run(f"Q2_n{n}", sp, name, asg, tn))

        # ---- Q3 : batching
        for bt in BATCH:
            Q3[("random", bt)].append(avg([ctx.run("Q3", sp, "random", r, test, batching=bt) for r in rnd]))
            for name, asg in plans.items():
                Q3[(name, bt)].append(ctx.run("Q3", sp, name, asg, test, batching=bt))

        # ---- Q4 + Q5 : assignment of batches to pickers and sequence per picker
        for md in MODES:
            for asg_m, sq in COMBO:
                kw = dict(assign=asg_m, seq=sq, station_mode=md)
                Q4[("random", md, asg_m, sq)].append(avg([ctx.run("Q4Q5", sp, "random", r, test, **kw) for r in rnd]))
                for name, asg in plans.items():
                    Q4[(name, md, asg_m, sq)].append(ctx.run("Q4Q5", sp, name, asg, test, **kw))
        print(f"split {sp + 1}/{a.splits} done  (plans {t1 - t0:.0f}s, simulations {time.time() - t1:.0f}s, "
              f"total {(time.time() - t_start) / 60:.1f} min)", flush=True)

    ST = {"random": "Random", "cluster": "Cluster (data mining)", "rl_v2": "RL v2"}
    # ------------------------------ Table Q1
    rows = []
    names = [("random", "Random (baseline)", "-"), ("frequency", "Frequency (no clusters)", "-")] + \
            [(f"k{k}", f"Data mining: k = {k} clusters", str(k)) for k in KS] + \
            [(f"km{k}", f"K-means: k = {k} clusters", str(k)) for k in KS] + \
            [("cluster", "Data mining: k chosen on validation orders", f"{statistics.mean(kchosen):.0f} (mean)"),
             ("rl_v1", "RL v1 (constructive Q-learning)", "-"), ("rl_v2", "RL v2 (improvement Q-learning)", "-")]
    for key, label, k in names:
        row = [label, k, M.num(mean_key(Q1[key], "walk_m")), M.num(mean_key(Q1[key], "makespan")),
               M.num(mean_key(Q1[key], "avg_order_time"))]
        if key == "random":
            row += ["-", "-", "-"]
        else:
            row += [fi(improvements(Q1["random"], Q1[key], "walk_m")),
                    M.fmt_p(M.paired_p([r["walk_m"] for r in Q1["random"]], [r["walk_m"] for r in Q1[key]])),
                    fi(improvements(Q1["random"], Q1[key], "makespan"))]
        rows.append(row)
    M.table(f"Table Q1. Number of clusters and storage method (data mining vs RL) - {a.orders} orders, {a.splits} splits",
            ["Storage method", "Clusters", "Walking (m)", "Makespan (steps)", "Avg order time (steps)",
             "Walking improvement vs Random (mean ± sd)", "p (paired t)", "Makespan improvement vs Random"], rows,
            "Improvement = reduction in %, a negative value = worse. RL does not form clusters: it places every SKU directly into a slot.")

    # ------------------------------ Table Q2
    rows = []
    for n in SIZES:
        row = [n, M.num(mean_key(Q2[(n, "random")], "batches"))]
        for st in ST:
            row.append(M.num(mean_key(Q2[(n, st)], "walk_m") / n))
        for st in ("cluster", "rl_v2"):
            row.append(fi(improvements(Q2[(n, "random")], Q2[(n, st)], "walk_m")))
        for st in ("cluster", "rl_v2"):
            row.append(fi(improvements(Q2[(n, "random")], Q2[(n, st)], "makespan")))
        rows.append(row)
    M.table(f"Table Q2. Number of orders per wave - {a.splits} splits",
            ["Orders", "Batches", "Random (m/order)", "Cluster (m/order)", "RL v2 (m/order)",
             "Walking improvement: Cluster", "Walking improvement: RL v2", "Makespan improvement: Cluster", "Makespan improvement: RL v2"], rows,
            "Improvement (reduction) vs Random, mean ± sd over splits. A smaller sd means a more stable result.")

    # ------------------------------ Table Q3
    rows = []
    for st in ST:
        for bt in BATCH:
            r, b0 = Q3[(st, bt)], Q3[(st, "fcfs")]
            rows.append([ST[st], bt, M.num(mean_key(r, "walk_m")), M.num(mean_key(r, "avg_stops")),
                         M.num(mean_key(r, "makespan")), M.num(mean_key(r, "avg_order_time")),
                         "-" if bt == "fcfs" else fi(improvements(b0, r, "walk_m")),
                         "-" if bt == "fcfs" else M.fmt_p(M.paired_p([x["walk_m"] for x in b0], [x["walk_m"] for x in r])),
                         "-" if bt == "fcfs" else fi(improvements(b0, r, "makespan")),
                         "-" if bt == "fcfs" else fi(improvements(b0, r, "avg_order_time"))])
    M.table(f"Table Q3. Batching method (batch capacity {ctx.mcfg.batch_items} items) - {a.orders} orders, {a.splits} splits",
            ["Storage", "Batching", "Walking (m)", "Stops/batch", "Makespan", "Avg order time",
             "Walking improvement vs FCFS", "p (walking)", "Makespan improvement", "Avg order time improvement"], rows,
            "fcfs = orders 1-4, 5-8, ... | similarity = Jaccard similarity of SKUs | section = Jaccard similarity of aisle stops\n"
            "optimized = section batches improved by swapping orders between batches while the estimated tour gets shorter.\n"
            "A negative improvement means worse than FCFS.")

    # ------------------------------ Table Q4 / Q5 (+ station design)
    rows = []
    P0 = ctx.mcfg.n_pickers
    for st in ST:
        base = Q4[(st, "pooled", "dynamic", "fcfs")]
        for md in MODES:
            for asg_m, sq in COMBO:
                r = Q4[(st, md, asg_m, sq)]
                is_b = (md, asg_m, sq) == ("pooled", "dynamic", "fcfs")
                rows.append([ST[st], md, asg_m, sq, M.num(mean_key(r, "makespan")), M.num(mean_key(r, "avg_order_time")),
                             M.num(mean_key(r, "walk_m")), M.num(mean_key(r, "station_wait")),
                             f"{mean_key(r, 'picker_util'):.2f}",
                             "-" if is_b else fi(improvements(base, r, "makespan")),
                             "-" if is_b else M.fmt_p(M.paired_p([x["makespan"] for x in base], [x["makespan"] for x in r])),
                             "-" if is_b else fi(improvements(base, r, "avg_order_time")),
                             "-" if is_b else M.fmt_p(M.paired_p([x["avg_order_time"] for x in base], [x["avg_order_time"] for x in r]))])
    M.table(f"Table Q4/Q5. Station design, assignment of batches to pickers (Q4) and sequence per picker (Q5) - {P0} pickers, "
            f"{ctx.mcfg.n_stations} stations, {a.orders} orders, {a.splits} splits",
            ["Storage", "Station design", "Assignment", "Sequence", "Makespan", "Avg order time", "Walking (m)", "Station wait",
             "Picker util", "Makespan improvement vs pooled+dynamic+FCFS", "p", "Order-time improvement vs pooled+dynamic+FCFS", "p"], rows,
            "pooled = one shared batch queue, a picker delivers to its packing station and walks back to the order start station | dedicated = batch j is fixed to station (j mod S)\n"
            "dynamic = the next free picker takes the next batch of its queue | static = batches are given to pickers in advance (round robin)\n"
            "balanced = batches handed out longest first to the picker with the smallest estimated workload (LPT)\n"
            "fcfs = lowest batch number first | spt = batch with the shortest estimated tour first (TSP tour length in steps from\n"
            "the order start station through all stops to packing station 1; picking time, queueing and the walk back are not included).")

    keys = ["study", "split", "strategy", "batching", "assign", "seq"] + \
           [k for k in ctx.runs[0] if k not in ("study", "split", "strategy", "batching", "assign", "seq")]
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(ctx.runs)
    print(f"\nAll runs written to {a.out} ({len(ctx.runs)} runs, {(time.time() - t_start) / 60:.1f} min)")


if __name__ == "__main__":
    main()