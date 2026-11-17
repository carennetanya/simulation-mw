# -*- coding: utf-8 -*-
"""kmeans_plan.py - k-means as a comparison for the hierarchical (average linkage) clustering.

SKU features = row of the Jaccard co-purchase similarity matrix (TRAIN orders only), so every SKU is described by how
similar it is to all other SKUs. k-means (Euclidean, n_init = 10) clusters these rows; the clusters are placed with
the same rule as the report (wh_sim.pack_clusters). Needs wh_sim.py, manual_sim.py, instacart_patch.py in the folder.

Run:  python kmeans_plan.py --splits 5 --ks 5 20 65
Output: kmeans_compare.csv and a table with walking distance / makespan (mean over splits, improvement vs random).
"""
import argparse, csv, statistics, time
from types import SimpleNamespace
import numpy as np
from sklearn.cluster import KMeans
import wh_sim as W, manual_sim as M
from instacart_patch import load_instacart


def jaccard_rows(skus, train):
    idx = {s: i for i, s in enumerate(skus)}
    B = np.zeros((len(skus), len(train)), dtype=np.float32)
    for j, o in enumerate(train):
        for s in set(o):
            B[idx[s], j] = 1
    inter = B @ B.T
    cnt = B.sum(1)
    union = cnt[:, None] + cnt[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.where(union > 0, inter / union, 0.0)
    np.fill_diagonal(S, 1.0)
    return S


def kmeans_clusters(skus, train, k, seed=0):
    X = jaccard_rows(skus, train)
    lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    out = {}
    for s, l in zip(skus, lab):
        out.setdefault(int(l), []).append(s)
    return list(out.values())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="archive")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 20, 65])
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    orders = load_instacart(f"{a.folder}/products.csv", f"{a.folder}/order_products__train.csv", 546, 60000)
    ctx = M.Ctx(orders, M.MCfg(), SimpleNamespace(split_seed=0))
    res = {}
    for sp in range(a.splits):
        t0 = time.time()
        skus, train, test = ctx.split(100, sp)
        f = W.sku_frequency(train)
        plans = {"random": None, "frequency": W.assign_frequency(skus, ctx.slots, train)}
        ck, _ = W.cluster_skus_k(skus, train, a.ks)
        for k in a.ks:
            plans[f"hier k={k}"] = W.pack_clusters(ck[k], f, ctx.slots)
            km = kmeans_clusters(skus, train, k, sp)
            plans[f"kmeans k={k}"] = W.pack_clusters(km, f, ctx.slots)
            if sp == 0:
                print(f"k={k} sizes  hier: {sorted(map(len, ck[k]), reverse=True)[:8]}  kmeans: {sorted(map(len, km), reverse=True)[:8]}")
        for name, asg in plans.items():
            if name == "random":
                rr = [M.simulate(ctx.walk, ctx.mcfg, W.assign_random(skus, ctx.slots, 1000 + sd), test) for sd in range(a.seeds)]
                r = {m: statistics.mean(x[m] for x in rr) for m in ("walk_m", "makespan")}
            else:
                k_ = M.simulate(ctx.walk, ctx.mcfg, asg, test)
                r = {m: k_[m] for m in ("walk_m", "makespan")}
            res.setdefault(name, []).append(r)
        print(f"split {sp} done ({time.time() - t0:.0f}s)", flush=True)
    base = res["random"]
    rows = []
    for name, rs in res.items():
        w = statistics.mean(r["walk_m"] for r in rs); mk = statistics.mean(r["makespan"] for r in rs)
        iw = statistics.mean(100 * (b["walk_m"] - r["walk_m"]) / b["walk_m"] for b, r in zip(base, rs))
        im = statistics.mean(100 * (b["makespan"] - r["makespan"]) / b["makespan"] for b, r in zip(base, rs))
        rows.append((name, w, mk, iw, im))
    print(f"\n{'plan':16s} {'walk (m)':>9s} {'makespan':>9s} {'walk impr.':>11s} {'mksp impr.':>11s}")
    for r in rows:
        print(f"{r[0]:16s} {r[1]:9.1f} {r[2]:9.0f} {r[3]:10.1f}% {r[4]:10.1f}%")
    print("\nPaired t-test on walking distance over splits (p):")
    wk = {n: [r["walk_m"] for r in rs] for n, rs in res.items()}
    for k in a.ks:
        print(f"  k={k}: kmeans vs hier p={M.fmt_p(M.paired_p(wk[f'hier k={k}'], wk[f'kmeans k={k}']))} | "
              f"kmeans vs frequency p={M.fmt_p(M.paired_p(wk['frequency'], wk[f'kmeans k={k}']))} | "
              f"hier vs frequency p={M.fmt_p(M.paired_p(wk['frequency'], wk[f'hier k={k}']))}")
    with open("kmeans_compare.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["plan", "walk_m", "makespan", "walk_impr_pct", "makespan_impr_pct"]); w.writerows(rows)