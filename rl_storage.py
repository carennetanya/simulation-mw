# -*- coding: utf-8 -*-
"""
rl_storage.py - Reinforcement learning (tabular Q-learning) for SKU -> slot storage assignment.

Evaluated with the SAME manual picking simulation as the other storage methods (manual_sim.simulate).
RL is trained separately in every split on that split's TRAIN orders only (no test leakage).

MDP (one episode = one complete storage plan)
  - SKUs are placed one by one, most frequent first (fixed order, so an episode has 546 steps).
  - State  : (frequency bucket of the SKU, section of its strongest co-purchased SKU already placed or 'none')
  - Action : choose a section (91 sections, 6 slots each); a full section is masked out.
  - Reward : - [ lam_s * f_i * [walk(order start, section) + walk(section, packing)]  +  sum_j c_ij * walk(section, section_j) ]
             f_i = share of train orders containing SKU i, c_ij = share containing both i and j,
             j = SKUs already placed. The sum over an episode is the (train) walking-cost proxy of the plan.
  - Learning: Q-learning, epsilon-greedy, alpha=0.05, gamma=0.5 (tuned on the TRAIN objective only).
The final plan = greedy rollout of the learned Q-table. Inside a section SKUs are ordered by frequency.
"""
import argparse
import csv
import itertools
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from types import SimpleNamespace

import numpy as np

import manual_sim as M
import wh_sim as W


def build_stats(skus, train):
    n = len(train)
    idx = {s: i for i, s in enumerate(skus)}
    f = np.zeros(len(skus))
    co = defaultdict(dict)
    for o in train:
        ids = sorted({idx[s] for s in o if s in idx})
        for i in ids:
            f[i] += 1
        for i, j in itertools.combinations(ids, 2):
            co[i][j] = co[i].get(j, 0) + 1
            co[j][i] = co[j].get(i, 0) + 1
    f /= n
    nbr = []
    for i in range(len(skus)):
        items = sorted(co[i].items(), key=lambda kv: -kv[1])[:40]      # top-40 co-purchase partners
        nbr.append((np.array([j for j, _ in items], dtype=int), np.array([c / n for _, c in items])))
    return idx, f, nbr


def section_tables(ctx):
    secs, slots_of = [], {}
    for sl in ctx.slots:                                  # ctx.slots is ordered by distance to the station
        if sl.sec not in slots_of:
            slots_of[sl.sec] = []
            secs.append(sl.sec)
        slots_of[sl.sec].append(sl)
    acc = [slots_of[s][0].access for s in secs]
    dsec = np.array([[ctx.walk.d[(a, b)] for b in acc] for a in acc], dtype=float)
    dst = np.array([ctx.walk.d[(ctx.walk.start, a)] + ctx.walk.end_cost[a] for a in acc], dtype=float)   # every tour: order start -> section -> packing
    cap = np.array([len(slots_of[s]) for s in secs])
    return secs, slots_of, dsec, dst, cap


def train_rl(skus, train, ctx, episodes=3000, lam_s=1.0, alpha=0.05, gamma=0.5,
             eps0=0.3, eps1=0.02, n_fb=8, seed=0, verbose=True):
    rng = random.Random(seed)
    idx, f, nbr = build_stats(skus, train)
    secs, slots_of, dsec, dst, cap0 = section_tables(ctx)
    S, N = len(secs), len(skus)
    order = sorted(range(N), key=lambda i: (-f[i], i))
    fb = np.zeros(N, dtype=int)
    for r, i in enumerate(order):
        fb[i] = min(n_fb - 1, r * n_fb // N)
    Q = np.zeros((n_fb, S + 1, S))                         # optimistic init (rewards are <= 0)
    SCALE = 100.0
    curve = []

    def rollout(eps, learn):
        free = cap0.copy()
        sec_of = -np.ones(N, dtype=int)
        total = 0.0
        prev = None                                       # (state, action, reward) waiting for next state
        for t, i in enumerate(order):
            js, cs = nbr[i]
            placed = sec_of[js] >= 0
            anchor = S
            if placed.any():
                anchor = int(sec_of[js[placed][np.argmax(cs[placed])]])
            st = (fb[i], anchor)
            q = Q[st]
            mask = free > 0
            if learn and rng.random() < eps:
                a = rng.choice(np.flatnonzero(mask).tolist())
            else:
                a = int(np.argmax(np.where(mask, q, -1e18)))
            cost = lam_s * f[i] * dst[a]
            if placed.any():
                cost += float((cs[placed] * dsec[a, sec_of[js[placed]]]).sum())
            r = -cost * SCALE
            if learn and prev is not None:                # update previous step with max over this state
                ps, pa, pr = prev
                Q[ps][pa] += alpha * (pr + gamma * float(np.max(np.where(mask, q, -1e18))) - Q[ps][pa])
            prev = (st, a, r)
            sec_of[i] = a
            free[a] -= 1
            total += cost
        if learn and prev is not None:
            ps, pa, pr = prev
            Q[ps][pa] += alpha * (pr - Q[ps][pa])
        return sec_of, total

    t0 = time.time()
    for ep in range(episodes):
        eps = eps0 + (eps1 - eps0) * ep / max(1, episodes - 1)
        _, tot = rollout(eps, True)
        curve.append(tot)
        if verbose and (ep + 1) % max(1, episodes // 5) == 0:
            print(f"    episode {ep + 1}/{episodes}  cost(proxy)={statistics.mean(curve[-100:]):.4f}  eps={eps:.2f}  ({time.time() - t0:.0f}s)")
    sec_of, tot = rollout(0.0, False)
    # build assignment: section -> slots (already ordered level/basket), SKUs inside a section by frequency
    by_sec = defaultdict(list)
    for i in range(N):
        by_sec[int(sec_of[i])].append(i)
    asg = {}
    for s, members in by_sec.items():
        members.sort(key=lambda i: (-f[i], i))
        for i, sl in zip(members, slots_of[secs[s]]):
            asg[skus[i]] = sl
    assert len(asg) == N and len({sl.idx for sl in asg.values()}) == N
    return asg, curve, tot


def proxy_cost(asg, skus, train, ctx):
    """same cost proxy for ANY assignment (to compare RL with frequency / cluster on the training objective)."""
    idx, f, nbr = build_stats(skus, train)
    secs, slots_of, dsec, dst, _ = section_tables(ctx)
    sidx = {s: k for k, s in enumerate(secs)}
    sec_of = np.array([sidx[asg[s].sec] for s in skus])
    tot = float((f * dst[sec_of]).sum())
    for i in range(len(skus)):
        js, cs = nbr[i]
        tot += 0.5 * float((cs * dsec[sec_of[i], sec_of[js]]).sum())   # each pair counted twice
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="archive/order_products__train.csv")
    ap.add_argument("--products", default="archive/products.csv")
    ap.add_argument("--orders", type=int, default=100)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--lam-s", type=float, default=1.0)
    M.add_picking_args(ap)
    ap.add_argument("--out", default="rl_results.csv")
    ap.add_argument("--curve", default="rl_curve.csv")
    ap.add_argument("--export-dir", default=None, help="write the RL plan of every split as CSV (sku, slot_idx)")
    a = ap.parse_args()

    a.synthetic = False
    a.split_seed = 0
    data = M.load_data(a)
    mcfg = M.mcfg_from_args(a)
    ctx = M.Ctx(data, mcfg, SimpleNamespace(split_seed=0))
    per = defaultdict(list)
    curves = []
    for sp in range(a.splits):
        print(f"split {sp}")
        skus, train, test = ctx.split(a.orders, sp)
        asg, curve, tot = train_rl(skus, train, ctx, episodes=a.episodes, lam_s=a.lam_s, seed=sp)
        curves.append(curve)
        casg, _ = M.cluster_plan(ctx, skus, train, n_val=a.orders)
        fasg = W.assign_frequency(skus, ctx.slots, train)
        rr = ctx.rand_runs("rl", sp, skus, test, a.seeds)
        per["random"].append({m: M.mean_of(rr, m) for m, _, _ in M.METRICS})
        per["frequency"].append(ctx.run("rl", sp, "frequency", fasg, test))
        per["cluster"].append(ctx.run("rl", sp, "cluster", casg, test))
        per["rl"].append(ctx.run("rl", sp, "rl", asg, test))
        rnd = W.assign_random(skus, ctx.slots, 1000)
        per["proxy"].append({k: proxy_cost(v, skus, train, ctx) for k, v in
                             dict(random=rnd, frequency=fasg, cluster=casg, rl=asg).items()})
        if a.export_dir:
            os.makedirs(a.export_dir, exist_ok=True)
            with open(os.path.join(a.export_dir, f"rl_assignment_split{sp}.csv"), "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["sku", "slot_idx"])
                for s in skus:
                    w.writerow([s, asg[s].idx])

    rows = []
    for m, label, lower in M.METRICS:
        if not lower:
            continue
        rnd = [r[m] for r in per["random"]]
        row = [label, M.num(statistics.mean(rnd))]
        for k in ("frequency", "cluster", "rl"):
            v = [r[m] for r in per[k]]
            row.append(f"{M.num(statistics.mean(v))} ({statistics.mean(M.pct(x, y) for x, y in zip(rnd, v)):+.1f}%)")
        rows.append(row)
    M.table(f"Table 6. Storage: Random vs Frequency vs Cluster (data mining) vs RL (Q-learning) - {a.orders} orders, "
            f"{a.splits} splits, FCFS batching, dynamic assignment. In brackets: reduction vs Random.",
            ["Metric", "Random", "Frequency", "Cluster", "RL"], rows)
    prow = [[k, f"{statistics.mean(p[k] for p in per['proxy']):.4f}"] for k in ("random", "frequency", "cluster", "rl")]
    M.table("Table 6b. Training objective (expected walking-cost proxy on TRAIN orders, lower = better)", ["Plan", "proxy cost"], prow)
    wm = {k: [r["walk_m"] for r in per[k]] for k in ("random", "frequency", "cluster", "rl")}
    print("\nPaired t-test on total walking distance (over splits): "
          f"RL vs Random p={M.fmt_p(M.paired_p(wm['random'], wm['rl']))} | "
          f"RL vs Cluster p={M.fmt_p(M.paired_p(wm['cluster'], wm['rl']))} | "
          f"RL vs Frequency p={M.fmt_p(M.paired_p(wm['frequency'], wm['rl']))}")
    keys = sorted({k for r in ctx.runs for k in r})
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["study", "split", "strategy"] + [k for k in keys if k not in ("study", "split", "strategy")])
        w.writeheader()
        w.writerows(ctx.runs)
    with open(a.curve, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["episode"] + [f"split{s}" for s in range(len(curves))])
        for e in range(len(curves[0])):
            w.writerow([e + 1] + [c[e] for c in curves])


if __name__ == "__main__":
    main()