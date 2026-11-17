# -*- coding: utf-8 -*-
"""
lam_validation.py - how LAM (pair weight of the RL v2 cost) is chosen, using TRAIN data only.

Protocol (no test orders are ever used):
  1. Take the TRAIN orders of split 0.
  2. Randomly hold out `--orders` of them as VALIDATION orders (the rest = fitting orders). Repeat for `--partitions` partitions.
  3. For every LAM: learn the RL v2 plan on the fitting orders, then pick the validation orders in the manual simulation.
  4. The LAM with the smallest mean walking distance on the validation orders is chosen.
"""
import argparse
import random
import statistics
from types import SimpleNamespace

import numpy as np

import manual_sim as M
import rl2
import rl_storage as R1
import wh_sim as W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="archive/order_products__train.csv")
    ap.add_argument("--products", default="archive/products.csv")
    ap.add_argument("--orders", type=int, default=100, help="validation orders per wave")
    ap.add_argument("--partitions", type=int, default=3)
    M.add_picking_args(ap)
    ap.add_argument("--lams", default="0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8")
    a = ap.parse_args()
    a.synthetic, a.split_seed = False, 0
    lams = [float(x) for x in a.lams.split(",")]
    ctx = M.Ctx(M.load_data(a), M.mcfg_from_args(a), SimpleNamespace(split_seed=0))
    secs, slots_of, D, dst, _ = R1.section_tables(ctx)
    sidx = {s: k for k, s in enumerate(secs)}
    skus, train, _unused_test = ctx.split(100, 0)          # only `train` is used below
    rows = {l: [] for l in lams}
    freq = []
    for part in range(a.partitions):
        idx = list(range(len(train)))
        random.Random(part).shuffle(idx)
        val = [train[i] for i in idx[:a.orders]]
        fit = [train[i] for i in idx[a.orders:]]
        fasg = W.assign_frequency(skus, ctx.slots, fit)
        freq.append(M.simulate(ctx.walk, ctx.mcfg, fasg, val)["walk_m"])
        f, C = rl2.dense_stats(skus, fit)
        s0 = np.array([sidx[fasg[x].sec] for x in skus])
        for lam in lams:
            s, _, _ = rl2.train_improve(f, C, D, dst, s0, lam=lam, seed=part)
            asg = rl2.to_assignment(s, skus, f, secs, slots_of)
            rows[lam].append(M.simulate(ctx.walk, ctx.mcfg, asg, val)["walk_m"])
        print(f"partition {part + 1}/{a.partitions} done", flush=True)
    fm = statistics.mean(freq)
    out = [[f"{l}", M.num(statistics.mean(rows[l])),
            f"{statistics.mean(M.pct(b, x) for b, x in zip(freq, rows[l])):+.1f}%"] for l in lams]
    out.append(["frequency plan (no RL)", M.num(fm), "-"])
    M.table(f"LAM validation - walking distance of {a.orders} VALIDATION orders (held out from the TRAIN data of split 0), "
            f"{a.partitions} partitions", ["LAM", "Walking (m)", "Improvement vs frequency"], out)
    best = min(lams, key=lambda l: statistics.mean(rows[l]))
    print(f"\nSelected LAM = {best} (smallest mean walking distance on validation orders)")


if __name__ == "__main__":
    main()