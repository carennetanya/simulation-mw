# -*- coding: utf-8 -*-
"""
rl2.py - RL v2: Q-learning that LEARNS WHICH IMPROVEMENT MOVES TO MAKE on a storage plan (swap of two SKUs).

Environment
  plan s[i] = section of SKU i (91 sections x 6 slots; capacity is kept because moves are swaps).
  cost(plan) = sum_i f_i * [walk(order start, s_i) + walk(s_i, packing)]  +  LAM * sum_{i<j} c_ij * walk(s_i, s_j)      (TRAIN data only)
  Start of every episode: the frequency plan (most frequent SKUs nearest to the station).
Agent (tabular Q-learning, epsilon-greedy)
  step   : pick a SKU i at random (probability proportional to its frequency)
  state  : (frequency bucket of i [8], how much better the best section is than the current one [5 buckets])
  action : move i to its k-th best section by marginal cost (k = 1..5) or to a random section (6 actions);
           the SKU in that section that is cheapest to exchange is swapped back into i's old section
  reward : cost reduction of the swap (positive = better); a swap that does not reduce the cost is undone (reward 0)
  gamma 0.5, alpha 0.1, epsilon 0.3 -> 0.02
Final plan = greedy rollout of the learned Q-table, starting again from the frequency plan.
"""
import itertools
import time
import numpy as np


def dense_stats(skus, train):
    idx = {s: i for i, s in enumerate(skus)}
    N = len(skus)
    f = np.zeros(N)
    C = np.zeros((N, N))
    for o in train:
        ids = sorted({idx[s] for s in o if s in idx})
        for i in ids:
            f[i] += 1
        for i, j in itertools.combinations(ids, 2):
            C[i, j] += 1
            C[j, i] += 1
    n = len(train)
    return f / n, C / n


class Env:
    def __init__(self, f, C, D, dst, s0, lam):
        self.f, self.C, self.D, self.dst, self.lam = f, C, D, dst, lam
        self.N, self.S = len(f), len(dst)
        self.s = s0.copy()
        self.members = [list(np.flatnonzero(self.s == k)) for k in range(self.S)]
        self.Ds = D[self.s, :].copy()                              # Ds[k, y] = walk(s_k, y)
        self.T = f[:, None] * dst[None, :] + lam * (C @ self.Ds)   # T[i, y] = cost of i if it sat in section y
        self.cost = float((f * dst[self.s]).sum() + lam * 0.5 * (C * D[np.ix_(self.s, self.s)]).sum())

    def pair_delta(self, i, j):
        s, T = self.s, self.T
        si, sj = s[i], s[j]
        return (T[i, sj] - T[i, si] + T[j, si] - T[j, sj]) + 2 * self.lam * self.C[i, j] * self.D[si, sj]

    def swap(self, i, j):
        s, D, C, lam = self.s, self.D, self.C, self.lam
        si, sj = s[i], s[j]
        self.members[si].remove(i); self.members[si].append(j)
        self.members[sj].remove(j); self.members[sj].append(i)
        d_i, d_j = D[sj] - D[si], D[si] - D[sj]                    # change of rows Ds[i], Ds[j]
        self.T += lam * (np.outer(C[:, i], d_i) + np.outer(C[:, j], d_j))
        self.Ds[i], self.Ds[j] = D[sj], D[si]
        s[i], s[j] = sj, si

    def best_partner(self, i, y):
        mem = self.members[y]
        ds = [self.pair_delta(i, j) for j in mem]
        k = int(np.argmin(ds))
        return mem[k], ds[k]


def train_improve(f, C, D, dst, s0, lam=1.6, episodes=6, steps=6000, final_steps=20000,
                  alpha=0.1, gamma=0.5, eps0=0.3, eps1=0.02, n_fb=8, seed=0, verbose=False):
    rng = np.random.default_rng(seed)
    N, S = len(f), len(dst)
    rank = np.argsort(np.argsort(-f))
    fb = np.minimum(n_fb - 1, rank * n_fb // N)
    K = 5
    NA = K + 1
    Q = np.zeros((n_fb, 5, NA))
    p = f / f.sum()
    edges = np.array([1e-9, 0.002, 0.01, 0.03])                    # gap buckets (cost units)

    def run(env, nsteps, learn, eps):
        prev = None
        for _ in range(nsteps):
            i = int(rng.choice(N, p=p))
            row = env.T[i]
            order = [int(y) for y in np.argsort(row) if y != env.s[i]][:K]
            gap = row[env.s[i]] - row[order[0]]
            st = (fb[i], int(np.searchsorted(edges, gap)))
            a = int(rng.integers(NA)) if (learn and rng.random() < eps) else int(np.argmax(Q[st]))
            y = int(rng.integers(S)) if a == K else order[a]
            r = 0.0
            if y != env.s[i]:
                j, d = env.best_partner(i, y)
                if d < -1e-12:
                    env.swap(i, j)
                    env.cost += d
                    r = -d
            if learn:
                if prev is not None:
                    ps, pa, pr = prev
                    Q[ps][pa] += alpha * (pr * 1000 + gamma * Q[st].max() - Q[ps][pa])
                prev = (st, a, r)
        if learn and prev is not None:
            ps, pa, pr = prev
            Q[ps][pa] += alpha * (pr * 1000 - Q[ps][pa])

    t0 = time.time()
    for ep in range(episodes):
        env = Env(f, C, D, dst, s0, lam)
        eps = eps0 + (eps1 - eps0) * ep / max(1, episodes - 1)
        run(env, steps, True, eps)
        if verbose:
            print(f"    episode {ep + 1}/{episodes}  plan cost {env.cost:.4f}  eps={eps:.2f}  ({time.time() - t0:.0f}s)", flush=True)
    env = Env(f, C, D, dst, s0, lam)
    start = env.cost
    run(env, final_steps, False, 0.0)
    return env.s.copy(), start, env.cost


def to_assignment(s, skus, f, secs, slots_of):
    by = {}
    for i, k in enumerate(s):
        by.setdefault(int(k), []).append(i)
    asg = {}
    for k, m in by.items():
        m.sort(key=lambda i: (-f[i], i))
        for i, sl in zip(m, slots_of[secs[k]]):
            asg[skus[i]] = sl
    return asg