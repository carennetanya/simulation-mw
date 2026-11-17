# -*- coding: utf-8 -*-
"""
manual_sim.py - MANUAL (picker-to-parts) warehouse simulation for the storage assignment study.

Research problem (one problem only): where to store each SKU.
Picking operation is the same in every test; only the storage method changes.

Operation
  - P pickers walk in the aisles of the real warehouse layout (91 sections = 546 slots, SKU = slot; 15 racks R1..R15).
  - Pickers can walk both directions in every aisle and can pass each other (no collision model).
  - 2 stations: ORDER START (green) and PACKING (red). A picker takes a batch at the order start station, walks to every SKU of the
    batch in ONE tour, ends the tour at the packing station, sorts the items into the order boxes and hands them over
    (a packing station serves one picker at a time, first come first served by ARRIVAL time, so the pickers queue). Then the picker
    walks back to the order start station for the next batch; this walk counts in walking distance AND time. 1 step = 1 grid cell of walking.
  - Two designs for linking batches to packing stations (--station-mode); they only differ when --stations > 1 (with ONE packing station = default, pooled = dedicated):
      pooled    = ONE shared queue of batches; the next free picker takes the next batch and delivers it to its packing station;
      dedicated = every batch is fixed to a packing station beforehand (batch j -> station j mod S) and is picked by a picker
                  of that station.
  - A batch = 4 orders. Unique SKUs of the 4 orders are picked in ONE tour (batching saves walking).
  - Tour = TSP over the aisle stops (exact Held-Karp up to 9 stops, else nearest-neighbour + 2-opt + or-opt).
    Shortest paths between stops are computed on the aisle grid (BFS).
  - All orders are known at the start (a wave).

Decisions studied (professor's questions)
  Q1  number of clusters k for the SKUs    -> --study clusters  (k = number of clusters; sizes are free)
  Q2  number of orders to be picked        -> --study orders
  Q3  batching: FCFS vs optimization       -> --study batching  (fcfs | similarity | section)
  Q4  assign batches to pickers            -> --study assignment (dynamic | static)
  Q5  sequence batches of each picker      -> --study assignment (fcfs | spt)
  Main experiment: random vs frequency vs cluster storage with a fixed picking policy -> --study storage
"""
import argparse
import csv
import heapq
import math
import os
import random
import statistics
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, replace

import wh_sim as W


# ==========================================================================
# 1. CONFIGURATION
# ==========================================================================
def normalize_station_mode(mode):
    if mode is None:
        return "pooled"
    return "pooled" if mode == "shared" else mode


@dataclass(frozen=True)
class MCfg:
    n_pickers: int = 4           # EVEN, and the same number of pickers at every station (P is a multiple of S)
    n_stations: int = 1          # PACKING stations (red). The ORDER START station (green) is always one, see START_CELL
    batch_orders: int = 4
    pick_steps: int = 3          # time to pick one SKU VISIT (reach, take, put in the order box); independent of quantity
    station_base: int = 4        # fixed hand-over time at a station
    station_per_item: int = 1    # sorting time per item (one unit per order line in the Instacart data) at a station
    station_mode: str = "pooled"  # how batches are linked to stations: "pooled" | "dedicated" (see simulate())
    batch_items: int = 20  # capacity of a batch in ITEMS (order lines): orders are added to a batch while the total number of items fits; 0 = fixed batch_orders orders per batch
    arrival_rate: float = 0.0   # orders per step of a Poisson arrival process; 0 = all orders of the wave are present at time 0 (offline, default)
    window: int = 0             # online only: number of consecutive orders that are batched together; 0 = batch_orders (fixed number of orders per batch)
    level_extra: tuple = (0, 0, 0)  # extra pick steps per SKU visit by slot level (index = level - 1: level 1 = bottom, 2 = waist height, 3 = top); (0, 0, 0) = the level does not matter (default)


def add_picking_args(ap):
    """command-line options shared by all scripts that build the manual picking context."""
    ap.add_argument("--pickers", type=int, default=4, help="number of pickers; even, a multiple of --stations")
    ap.add_argument("--stations", type=int, default=1, choices=[1, 2, 3], help="number of packing stations (the order start station is always one)")
    ap.add_argument("--station-mode", default="pooled", choices=["pooled", "shared", "dedicated"],
                    help="pooled/shared = one shared batch queue, every picker takes the next batch and delivers it to its packing station | "
                         "dedicated = every batch is fixed to a station beforehand (batch j -> station j mod S)")
    ap.add_argument("--batch-items", type=int, default=20,
                    help="capacity of a batch in items (order lines): orders are added while the total fits; 0 = fixed number of orders (--batch-orders)")
    ap.add_argument("--batch-orders", type=int, default=4, help="number of orders k of a batch when --batch-items 0 (fixed number of orders, as in Tang and Chew 1997)")
    ap.add_argument("--arrival-rate", type=float, default=0.0,
                    help="online mode: orders arrive as a Poisson process with this rate (orders per step); 0 = offline, all orders present at time 0")
    ap.add_argument("--window", type=int, default=0,
                    help="online mode: orders are batched in windows of this many consecutive orders (0 = --batch-orders)")
    ap.add_argument("--level-extra", default="0,0,0",
                    help="extra pick steps per SKU visit for level 1,2,3 (1 = bottom, 2 = waist height, 3 = top), e.g. 1,0,2; 0,0,0 = level does not matter")


def parse_level_extra(txt):
    """'1,0,2' -> (1, 0, 2): extra pick steps per SKU visit for level 1, 2, 3."""
    v = tuple(int(x) for x in str(txt).replace(";", ",").split(","))
    if len(v) != 3 or min(v) < 0:
        raise SystemExit(f"--level-extra needs three non-negative integers (level 1,2,3), e.g. 1,0,2; got {txt!r}")
    return v


def mcfg_from_args(a):
    if a.pickers % a.stations:
        raise SystemExit(f"--pickers ({a.pickers}) must be a multiple of --stations ({a.stations}): same number of pickers per station")
    return MCfg(n_pickers=a.pickers, n_stations=a.stations, station_mode=normalize_station_mode(getattr(a, "station_mode", "pooled")),
                batch_items=getattr(a, "batch_items", 20), batch_orders=getattr(a, "batch_orders", 4),
                arrival_rate=getattr(a, "arrival_rate", 0.0), window=getattr(a, "window", 0),
                level_extra=parse_level_extra(getattr(a, "level_extra", "0,0,0")))


# ==========================================================================
# 2. WALKING GRAPH (undirected) + distances
# ==========================================================================
# --- ORDER START station (green in the layout figure) ---------------------------------------------------------------
# A picker takes a batch at the ORDER START station, walks through the aisles to the pick stops and ends the tour at the
# PACKING station (red = lay.svc, where the hand-over/sorting and the queue are). After packing the picker walks back to the
# order start station. The start station is reached by a dead-end path from the lower cross aisle (column c2), as drawn.
START_COL = 2
START_CELL = (START_COL, 0)
INBOUND_CELL = START_CELL                       # old name


def start_spur(lay):
    """cells of the dead-end path from the lower cross aisle (zB) down to the order start station."""
    return [(START_COL, z) for z in range(lay.zB, -1, -1)]


inbound_spur = start_spur                       # old name


def add_start_station(lay):
    """adds the order start station to the layout graph (idempotent); sets lay.start."""
    if getattr(lay, "start", None) is None:
        sp = start_spur(lay)
        for a_, b_ in zip(sp, sp[1:]):
            lay.edges[a_].append(b_)
            lay.edges[b_].append(a_)
        lay.start = START_CELL
    return lay.start


class Walk:
    """Shortest walking distance (steps and metres) between aisle stops and stations."""

    def __init__(self, lay):
        add_start_station(lay)
        nb = defaultdict(set)
        for a, ns in lay.edges.items():
            for n in ns:
                nb[a].add(n)
                nb[n].add(a)
        self.lay = lay
        self.nb = nb
        self._path = {}
        self.svc = list(lay.svc)                  # packing station(s)
        self.start = lay.start                    # order start station
        self.nodes = sorted({s.access for s in lay.slots} | set(self.svc) | {self.start})
        self.d, self.m = {}, {}
        for src in self.nodes:
            dist, met, q = {src: 0}, {src: 0.0}, deque([src])
            while q:
                c = q.popleft()
                for n in sorted(nb[c]):
                    if n not in dist:
                        dist[n] = dist[c] + 1
                        met[n] = met[c] + lay.step_m(c, n)
                        q.append(n)
            for t in self.nodes:
                self.d[(src, t)] = dist[t]
                self.m[(src, t)] = met[t]
        self.end_cost = {c: min(self.d[(c, s)] for s in self.svc) for c in self.nodes}

    def path(self, a, b):
        """cells from a to b (inclusive): one shortest path, same BFS order as the distance tables."""
        key = (a, b)
        if key not in self._path:
            par, q = {a: None}, deque([a])
            while q:
                c = q.popleft()
                if c == b:
                    break
                for n in sorted(self.nb[c]):
                    if n not in par:
                        par[n] = c
                        q.append(n)
            out, c = [], b
            while c is not None:
                out.append(c)
                c = par[c]
            self._path[key] = out[::-1]
        return self._path[key]


LEVEL_RANK = (2, 1, 3)    # preference of the levels inside a section: level 2 (waist height) first, then level 1 (bottom), then level 3 (top)


def ordered_slots(lay, walk, level_rank=LEVEL_RANK):
    """Slots grouped per section, sections ordered by the walking distance order start station -> section -> nearest packing station.
    Inside a section the slots are ordered by level_rank (waist level first, then bottom, then top), then by basket. Every storage plan
    (frequency, ABC, clusters, RL) fills the slots in this order, so the most popular SKUs get the easiest level of a section.
    The order inside a section does not change any walking distance."""
    secs = {}
    for sl in lay.slots:
        secs.setdefault(sl.sec, []).append(sl)
    key = lambda sec: (min(walk.d[(walk.start, secs[sec][0].access)] + walk.d[(secs[sec][0].access, st)] for st in walk.svc), sec)
    out = []
    for sec in sorted(secs, key=key):
        for sl in sorted(secs[sec], key=lambda s: (level_rank.index(s.level), s.basket)):
            out.append(replace(sl, idx=len(out)))
    return out


# ==========================================================================
# 3. TSP over aisle stops
# ==========================================================================
def route_total(walk, start, order, end=None):
    """steps from start through order and then to the nearest station (or to the fixed station `end`)."""
    if not order:
        return 0
    cur, tot = start, 0
    for c in order:
        tot += walk.d[(cur, c)]
        cur = c
    return tot + (walk.end_cost[cur] if end is None else walk.d[(cur, end)])


def _held_karp(walk, start, cells, end=None):
    n, d, INF = len(cells), walk.d, 10 ** 9
    ec = (lambda c: walk.end_cost[c]) if end is None else (lambda c: d[(c, end)])
    dp = [[INF] * n for _ in range(1 << n)]
    par = [[-1] * n for _ in range(1 << n)]
    for j, c in enumerate(cells):
        dp[1 << j][j] = d[(start, c)]
    for mask in range(1, 1 << n):
        for j in range(n):
            cur = dp[mask][j]
            if cur >= INF or not (mask >> j) & 1:
                continue
            for k in range(n):
                if (mask >> k) & 1:
                    continue
                nm, v = mask | (1 << k), cur + d[(cells[j], cells[k])]
                if v < dp[nm][k]:
                    dp[nm][k], par[nm][k] = v, j
    full = (1 << n) - 1
    j = min(range(n), key=lambda j: (dp[full][j] + ec(cells[j]), cells[j]))
    order, mask = [], full
    while j != -1:
        order.append(cells[j])
        pj = par[mask][j]
        mask ^= (1 << j)
        j = pj
    return order[::-1]


def _heuristic(walk, start, cells, end=None):
    d = walk.d
    rem, order, cur = list(cells), [], start
    while rem:
        nx = min(rem, key=lambda c: (d[(cur, c)], c))
        order.append(nx)
        rem.remove(nx)
        cur = nx
    best = route_total(walk, start, order, end)
    improved = True
    while improved:
        improved = False
        n = len(order)
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                c = route_total(walk, start, cand, end)
                if c < best:
                    order, best, improved = cand, c, True
        for i in range(len(order)):
            node, rest = order[i], order[:i] + order[i + 1:]
            for j in range(len(rest) + 1):
                if j == i:
                    continue
                cand = rest[:j] + [node] + rest[j:]
                c = route_total(walk, start, cand, end)
                if c < best:
                    order, best, improved = cand, c, True
                    break
            if improved:
                break
    return order


def tsp(walk, start, cells, end=None):
    """end=None: the tour ends at the nearest station (original behaviour); end=station: tour must end there."""
    cells = sorted(set(cells))
    if len(cells) <= 1:
        return cells
    if len(cells) <= 9:
        return _held_karp(walk, start, cells, end)
    return _heuristic(walk, start, cells, end)


# ==========================================================================
# 4. BATCHING (Q3)
# ==========================================================================
def make_batches(order_skus, slot_of, B, method, cap=0):
    """-> list of batches, each a list of order indices.
    cap = 0 : a batch holds at most B orders (fixed number of orders).
    cap > 0 : a batch holds as many orders as fit into the item capacity `cap` (total number of order lines, i.e. picks);
              B is ignored. An order that is larger than the capacity forms a batch of its own (orders are not split).
    fcfs       : orders in the order of arrival: a new batch is opened when the next order does not fit any more
    similarity : seed = order with most distinct SKUs, then add the order with the highest Jaccard
                 similarity of SKUs to the batch so far (item similarity idea of Qin et al. 2024); only orders that fit
    section    : same as similarity but on the aisle stops (sections) the orders need under the storage plan"""
    n = len(order_skus)
    items = [len(set(o)) for o in order_skus]
    if method == "fcfs":
        if cap <= 0:
            return [list(range(i, min(i + B, n))) for i in range(0, n, B)]
        out, cur, used = [], [], 0
        for i in range(n):
            if cur and used + items[i] > cap:
                out.append(cur)
                cur, used = [], 0
            cur.append(i)
            used += items[i]
        if cur:
            out.append(cur)
        return out
    if method == "similarity":
        sets = [set(o) for o in order_skus]
    elif method == "section":
        sets = [{slot_of[s].access for s in o} for o in order_skus]
    else:
        raise ValueError(method)
    pool, out = set(range(n)), []
    while pool:
        seed = max(pool, key=lambda i: (len(sets[i]), -i))
        batch, comp, used = [seed], set(sets[seed]), items[seed]
        pool.discard(seed)
        while pool:
            if cap > 0:
                cand = [i for i in pool if used + items[i] <= cap]
            else:
                cand = list(pool) if len(batch) < B else []
            if not cand:
                break
            nxt = max(cand, key=lambda i: (len(comp & sets[i]) / len(comp | sets[i]), -i))
            batch.append(nxt)
            used += items[nxt]
            comp |= sets[nxt]
            pool.discard(nxt)
        out.append(batch)
    return out


# ==========================================================================
# 5. SIMULATION (event driven)
# ==========================================================================
def make_arrivals(n, rate, seed=0):
    """arrival times of n orders: Poisson process with `rate` orders per step (exponential inter-arrival times); order 0 arrives at time 0,
    the order index is the arrival sequence. Same seed -> same arrivals (so storage plans are compared on the same arrivals)."""
    rng = random.Random(seed)
    t, out = 0.0, []
    for i in range(n):
        if i:
            t += rng.expovariate(rate)
        out.append(t)
    return out


def simulate(walk, cfg, slot_of, orders, batching="fcfs", assign="dynamic", seq="fcfs", trace=None, station_mode=None, arrivals=None):
    """One wave of orders, event driven.

    Stations: ONE order start station (green, walk.start) and S packing stations (red, walk.svc; default S = 1). Every tour starts at the
    order start station and ends at the packing station: order start -> all stops of the batch (exact / heuristic TSP, see tsp()) -> packing
    station. The picker hands the batch over and sorts it there; a packing station serves one picker at a time, first come first served by
    ARRIVAL time (event driven: a picker joins the packing queue when its tour ends). Afterwards the picker walks back to the order start
    station (distance AND time are counted; the last return of a picker, when no batch is left, is not walked). With S > 1 picker p
    delivers to packing station p mod S (P a multiple of S, so every packing station has P/S pickers).

    station_mode (how batches are linked to stations)
       pooled    : ONE shared queue of batches; whenever a picker is free it takes the next batch (according to `assign`)
                   and delivers it to its packing station.
       dedicated : every batch is fixed to a station beforehand (batch j -> station j mod S); only the pickers of that
                   station work on it.
    assign (which batch does a free picker take)
       dynamic  = the next batch of the queue (of the whole system in pooled mode, of the own station in dedicated mode)
       static   = batches are given to the pickers in advance, round robin (batch j -> picker j mod P in pooled mode,
                  inside a station's pickers in dedicated mode)
       balanced = batches are pre-assigned to balance the ESTIMATED workload (LPT rule, longest batch first)
    seq (order of the batches a picker works on)
       fcfs = lowest batch number first | spt = shortest ESTIMATED tour first.
    Online mode (cfg.arrival_rate > 0 or `arrivals` given; the model of Tang and Chew 1997): order i arrives at arrivals[i] (Poisson process);
    the orders are batched in windows of cfg.window consecutive orders (default cfg.window = cfg.batch_orders, i.e. fixed number of orders k per batch,
    FCFS in the sequence of arrival); a batch is RELEASED when the last order of its window has arrived, and a picker can only take released batches.
    Turnover time of an order = completion - arrival = W1 (waiting for the batch to be released) + W2 (waiting for a free picker) + S (tour, picking and
    packing station). Offline mode (default): all orders arrive at time 0, release = 0, and nothing of this changes.
    Tour estimate (used for spt and balanced): length in steps of the TSP tour from the order start station through all stops of the
    batch and then to packing station 1. Walking only: picking time, queueing and the walk back are NOT included, and
    it is always computed for packing station 1 (only matters with several packing stations). With one packing station the walk back is the same
    constant for every batch, so it does not change the SPT order; it IS included in the workload used by the balanced assignment."""
    station_mode = normalize_station_mode(station_mode or cfg.station_mode)
    P, stations = cfg.n_pickers, walk.svc
    S = len(stations)
    if station_mode not in ("pooled", "dedicated"):
        raise ValueError(station_mode)
    if P % S:
        raise ValueError("the number of pickers must be a multiple of the number of stations (same pickers per station)")
    home = [p % S for p in range(P)]
    n_ord = len(orders)
    online = cfg.arrival_rate > 0 or arrivals is not None
    if online and arrivals is None:
        arrivals = make_arrivals(n_ord, cfg.arrival_rate, 0)
    arr_t = list(arrivals) if online else [0.0] * n_ord
    if online:
        wsz = cfg.window if cfg.window > 0 else cfg.batch_orders
        groups, release = [], []
        for lo in range(0, n_ord, wsz):
            hi = min(lo + wsz, n_ord)
            for g in make_batches(orders[lo:hi], slot_of, cfg.batch_orders, batching, cap=cfg.batch_items):
                groups.append([lo + i for i in g])
                release.append(arr_t[hi - 1])             # released when the last order of the window has arrived
    else:
        groups = make_batches(orders, slot_of, cfg.batch_orders, batching, cap=cfg.batch_items)
        release = [0.0] * len(groups)
    batches = []
    for gi, g in enumerate(groups):
        skus = sorted({s for i in g for s in orders[i]})
        cells = sorted({slot_of[s].access for s in skus})
        est = route_total(walk, walk.start, tsp(walk, walk.start, cells, stations[0]), stations[0])
        pick_extra = sum(cfg.level_extra[slot_of[s].level - 1] for s in skus)
        batches.append(dict(release=release[gi], orders=g, skus=skus, cells=cells, items=sum(len(set(orders[i])) for i in g), est=est, pick_extra=pick_extra))
    nb = len(batches)
    key = (lambda b: (b, b)) if seq == "fcfs" else (lambda b: (batches[b]["est"], b))
    if assign not in ("dynamic", "static", "balanced"):
        raise ValueError(assign)
    load = [b["est"] + walk.d[(stations[0], walk.start)] + len(b["skus"]) * cfg.pick_steps + b["pick_extra"] + cfg.station_base + cfg.station_per_item * b["items"] for b in batches]

    def distribute(pool, pickers):
        """pre-assignment of the batches in `pool` to the `pickers` (static: round robin, balanced: LPT)."""
        own, tot = {p: [] for p in pickers}, {p: 0 for p in pickers}
        if assign == "static":
            for k, b in enumerate(pool):
                own[pickers[k % len(pickers)]].append(b)
        else:
            for b in sorted(pool, key=lambda b: (-load[b], b)):
                p = min(pickers, key=lambda p: (tot[p], p))
                own[p].append(b)
                tot[p] += load[b]
        return own

    if station_mode == "pooled":
        if assign == "dynamic":
            shared = list(range(nb))
            queue_of = lambda p: shared
        else:
            own = distribute(list(range(nb)), list(range(P)))
            queue_of = lambda p: own[p]
    else:                                              # dedicated: batch j -> station j mod S
        st_batches = [[b for b in range(nb) if b % S == s_] for s_ in range(S)]
        if assign == "dynamic":
            queue_of = lambda p: st_batches[home[p]]
        else:
            own = {}
            for s_ in range(S):
                own.update(distribute(st_batches[s_], [p for p in range(P) if home[p] == s_]))
            queue_of = lambda p: own[p]

    # Event driven, two event types (processed in time order; ties: arrivals first, then pickers, lowest picker id):
    #   kind 1 = picker is free  -> takes a batch, walks the tour and picks, and ARRIVES at its packing station at time `arr`
    #   kind 0 = picker arrives  -> joins the FIFO queue of its station: service starts at max(arr, station free time)
    # so every station serves its pickers in the order of ARRIVAL (first come, first served).
    station_free = [0] * S
    heap = [(0, 1, p, None) for p in range(P)]
    heapq.heapify(heap)
    steps_total, met_total, wait_total, busy, service_total = 0, 0.0, 0, 0, 0
    order_done, order_start = {}, {}
    batch_times = []
    while heap:
        t, kind, p, pl = heapq.heappop(heap)
        s = home[p]
        if kind == 1:
            q = queue_of(p)
            if not q:
                continue
            avail = [x for x in q if batches[x]["release"] <= t]
            if not avail:                              # online: no released batch yet -> wait until the first release
                heapq.heappush(heap, (min(batches[x]["release"] for x in q), 1, p, pl))
                continue
            b = min(avail, key=key)
            q.remove(b)
            bt = batches[b]
            st = stations[s]
            if pl == "back":                           # the picker walked from the packing station back to the order start station
                steps_total += walk.d[(st, walk.start)]
                met_total += walk.m[(st, walk.start)]
            order = tsp(walk, walk.start, bt["cells"], st)   # tour: order start station -> all stops -> packing station
            cur, to_last, to_last_m = walk.start, 0, 0.0
            for c in order:
                to_last += walk.d[(cur, c)]
                to_last_m += walk.m[(cur, c)]
                cur = c
            t_done = t + to_last + len(bt["skus"]) * cfg.pick_steps + bt["pick_extra"]
            arr = t_done + walk.d[(cur, st)]
            steps_total += to_last + walk.d[(cur, st)]
            met_total += to_last_m + walk.m[(cur, st)]
            heapq.heappush(heap, (arr, 0, p, (b, t, t_done, arr, order)))
        else:
            b, t0, t_done, arr, order = pl
            bt = batches[b]
            st_ = stations[s]
            start = max(arr, station_free[s])
            service = cfg.station_base + cfg.station_per_item * bt["items"]
            end = start + service
            station_free[s] = end
            wait_total += start - arr
            service_total += service
            busy += end - t0
            batch_times.append(end - t0)
            for i in bt["orders"]:
                order_done[i] = end
                order_start[i] = t0
            if trace is not None:    # timeline for the 3D visualisation (does not change any result)
                trace.append(dict(picker=p, batch=b, orders=list(bt["orders"]), skus=list(bt["skus"]), route=list(order),
                                  start_pos=walk.start, t0=t0, t_done=t_done, station=s, arr=arr, start=start, end=end))
            heapq.heappush(heap, (end + walk.d[(st_, walk.start)], 1, p, "back"))
    assert len(order_done) == len(orders), "not all orders were completed"
    makespan = max(order_done.values())
    n = len(orders)
    gi_of = {i: gi for gi, g in enumerate(groups) for i in g}
    return dict(
        batches=nb, steps=steps_total, walk_m=met_total, walk_m_per_order=met_total / n,
        makespan=makespan, avg_order_time=statistics.mean(order_done[i] - arr_t[i] for i in range(n)),
        avg_batch_wait=statistics.mean(release[gi_of[i]] - arr_t[i] for i in range(n)),            # W1: waiting until the batch is released
        avg_first_wait=statistics.mean(release[gi] - arr_t[min(g)] for gi, g in enumerate(groups)),  # W1 of the first order of every batch ((k-1)/rate)
        avg_pick_wait=statistics.mean(order_start[i] - release[gi_of[i]] for i in range(n)),        # W2: waiting for a free picker
        avg_service=statistics.mean(order_done[i] - order_start[i] for i in range(n)),              # S: tour, picking, packing station
        arrival_span=arr_t[-1] if n else 0.0,
        avg_batch_time=statistics.mean(batch_times), station_wait=wait_total,
        avg_stops=statistics.mean(len(b["cells"]) for b in batches),
        avg_skus=statistics.mean(len(b["skus"]) for b in batches),
        picker_util=busy / (P * makespan), station_util=service_total / (S * makespan))


# ==========================================================================
# 6. STATISTICS + TABLE PRINTING
# ==========================================================================
def _betacf(a, b, x, itmax=300, eps=3e-12):
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1e-300 if abs(d) < 1e-300 else d
        c = 1.0 + aa / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1e-300 if abs(d) < 1e-300 else d
        c = 1.0 + aa / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(a, b, x) / a
    return 1 - bt * _betacf(b, a, 1 - x) / b


def paired_p(a, b):
    """two-sided paired t-test p-value for lists a, b (same length)."""
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    if n < 2:
        return float("nan")
    sd = statistics.stdev(diffs)
    if sd == 0:
        return 0.0 if statistics.mean(diffs) != 0 else 1.0
    t = statistics.mean(diffs) / (sd / math.sqrt(n))
    df = n - 1
    return _betai(df / 2, 0.5, df / (df + t * t))


def fmt_p(p):
    return "n/a" if p != p else ("<0.001" if p < 0.001 else f"{p:.3f}")


def table(title, headers, rows, note=None):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = "-+-".join("-" * w for w in widths)
    print("\n" + title)
    print(" | ".join(str(h).ljust(widths[i]) if i == 0 else str(h).rjust(widths[i]) for i, h in enumerate(headers)))
    print(line)
    for r in rows:
        print(" | ".join(str(c).ljust(widths[i]) if i == 0 else str(c).rjust(widths[i]) for i, c in enumerate(r)))
    if note:
        print(note)


METRICS = [
    ("walk_m", "Total walking distance (m)", True),
    ("walk_m_per_order", "Walking distance per order (m)", True),
    ("makespan", "Makespan (steps)", True),
    ("avg_order_time", "Avg order completion time (steps)", True),
    ("avg_batch_time", "Avg batch cycle time (steps)", True),
    ("station_wait", "Total waiting for a station (steps)", False),
    ("avg_stops", "Avg aisle stops per batch", True),
    ("picker_util", "Picker utilization", False),
    ("station_util", "Station utilization", False),
]


def num(v):
    return f"{v:,.2f}" if abs(v) < 10 else f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.1f}"


def pct(base, new):
    return (base - new) / base * 100 if base else 0.0


# ==========================================================================
# 7. EXPERIMENT CONTEXT
# ==========================================================================
class Ctx:
    def __init__(self, orders_all, mcfg, args):
        self.orders_all, self.mcfg, self.args = orders_all, mcfg, args
        n_real = len({s for o in orders_all for s in o})
        n_sku = 546                                  # the layout ALWAYS has 91 sections x 6 slots = 546 slots
        assert n_real <= n_sku, f"{n_real} SKUs do not fit into {n_sku} slots"
        # fewer real SKUs than slots -> the remaining slots are EMPTY: modelled as zero-demand placeholder SKUs (never ordered)
        self.empty = [f"{W.EMPTY_PREFIX} {i + 1:03d}" for i in range(n_sku - n_real)]
        if self.empty:
            print(f"NOTE: {n_real} real SKUs, {n_sku} slots -> {len(self.empty)} slots stay EMPTY ({100 * len(self.empty) / n_sku:.0f}%)", flush=True)
        # The layout is built for 4 pickers always (this only fixes the length Q of the station area, which is NOT used by
        # manual picking). Otherwise the walking distance station -> racks would change with the number of pickers
        # and the picker-number study (2, 4, 6, 8) would be confounded.
        cfg = replace(W.Cfg(), n_robots=4, n_stations=mcfg.n_stations, batch_orders=mcfg.batch_orders)
        self.lay = W.make_layout(cfg, n_sku)
        self.walk = Walk(self.lay)
        self.slots = ordered_slots(self.lay, self.walk)
        assert len(self.slots) == n_sku, "number of slots must be 546"
        self.runs = []

    def split(self, n_orders, sp):
        skus, train, test = W.split_orders(self.orders_all, n_orders, self.args.split_seed + sp)
        return skus + self.empty, train, test        # + placeholders of the empty slots (if any)

    def run(self, study, sp, name, asg, test, pickers=None, **kw):
        """pickers=None -> self.mcfg.n_pickers. kw: batching, assign, seq, station_mode."""
        mc = self.mcfg if pickers is None else replace(self.mcfg, n_pickers=pickers)
        if "level_extra" in kw:                                   # sensitivity test: other pick-time extras per level
            mc = replace(mc, level_extra=tuple(kw.pop("level_extra")))
        if mc.arrival_rate > 0 and "arrivals" not in kw:          # online mode: the same arrivals for every storage plan of a split
            kw = dict(kw, arrivals=make_arrivals(len(test), mc.arrival_rate, 5000 + sp))
        k = simulate(self.walk, mc, asg, test, **kw)
        k.update(study=study, split=sp, strategy=name, pickers=mc.n_pickers, level_extra="-".join(map(str, mc.level_extra)),
                 **{x: kw.get(x, "") for x in ("batching", "assign", "seq")},
                 station_mode=kw.get("station_mode") or mc.station_mode)
        self.runs.append(k)
        return k

    def rand_runs(self, study, sp, skus, test, seeds, **kw):
        return [self.run(study, sp, "random", W.assign_random(skus, self.slots, 1000 + sd), test, **kw)
                for sd in range(seeds)]


def mean_of(runs, key):
    return statistics.mean(r[key] for r in runs)


# ----- choosing the number of clusters k (data mining, TRAIN data only) -----
K_GRID = [5, 10, 20, 40, 91, 100, 150, 250]       # 91 = number of sections (average cluster size 6)


def select_k(ctx, skus, train, n_val=100, ks=K_GRID, seed=0):
    """Choose k WITHOUT looking at the test orders: n_val orders of the TRAIN set are held out as validation orders, the
    clusters are built on the remaining (fitting) orders for every k, every plan is used to pick the validation orders in
    the manual simulation (FCFS batching, dynamic assignment, FCFS sequence) and the k with the smallest walking distance
    is selected.  The final plan is then rebuilt on ALL train orders with this k.   -> (k, {k: walking distance})"""
    import random
    idx = list(range(len(train)))
    random.Random(seed).shuffle(idx)
    val = [train[i] for i in idx[:n_val]]
    fit = [train[i] for i in idx[n_val:]]
    ck, f = W.cluster_skus_k(skus, fit, ks)
    res = {}
    for k in ks:
        asg = W.pack_clusters(ck[k], f, ctx.slots)
        res[k] = simulate(ctx.walk, ctx.mcfg, asg, val)["walk_m"]
    return min(ks, key=lambda k: (res[k], k)), res


def cluster_plan(ctx, skus, train, k=None, n_val=100):
    """Cluster (data mining) storage plan. k=None -> k is chosen on validation orders (select_k). -> (assignment, k)"""
    if k is None:
        k, _ = select_k(ctx, skus, train, n_val)
    asg, _ = W.assign_cluster_k(skus, ctx.slots, train, k)
    return asg, k


# ==========================================================================
# 8. STUDIES
# ==========================================================================
def study_storage(ctx):
    a, mc = ctx.args, ctx.mcfg
    per = defaultdict(list)                       # strategy -> list (per split) of metric dicts (means)
    cl_info, extra = [], {}
    for sp in range(a.splits):
        skus, train, test = ctx.split(a.orders, sp)
        rr = ctx.rand_runs("storage", sp, skus, test, a.seeds)
        per["random"].append({m: mean_of(rr, m) for m, _, _ in METRICS})
        fr = ctx.run("storage", sp, "frequency", W.assign_frequency(skus, ctx.slots, train), test)
        per["frequency"].append(fr)
        asg, kk = cluster_plan(ctx, skus, train)
        clusters = range(kk)
        cr = ctx.run("storage", sp, "cluster", asg, test)
        per["cluster"].append(cr)
        cl_info.append((sp, len(train), len(test), len(clusters), len(skus) / len(clusters)))
        if a.assignment_file:
            fa = load_assignment_file(a.assignment_file, ctx.slots, skus)
            per["file"].append(ctx.run("storage", sp, "file", fa, test))
    table("Table 1a. Data splits (a split = one random train/test partition of the orders; results are repeated over splits)",
          ["split", "train orders", "test orders", "clusters", "avg cluster size"],
          [[s, tr, te, k, f"{sz:.2f}"] for s, tr, te, k, sz in cl_info])
    hdr = ["Metric", f"Random (mean+-sd, {a.seeds} layouts x {a.splits} splits)", "Frequency", "Cluster",
           "Cluster vs Random", "p (paired t)", "Cluster vs Frequency"]
    if a.assignment_file:
        hdr.insert(4, "File")
    rows = []
    for m, label, lower in METRICS:
        rnd = [r[m] for r in per["random"]]
        fq = [r[m] for r in per["frequency"]]
        cl = [r[m] for r in per["cluster"]]
        sdr = statistics.pstdev(rnd) if len(rnd) > 1 else 0.0
        row = [label, f"{num(statistics.mean(rnd))} +- {num(sdr)}", num(statistics.mean(fq)), num(statistics.mean(cl))]
        if a.assignment_file:
            row.append(num(statistics.mean(r[m] for r in per["file"])))
        if lower:
            g = [pct(x, y) for x, y in zip(rnd, cl)]
            gf = [pct(x, y) for x, y in zip(fq, cl)]
            row += [f"{statistics.mean(g):.1f}% +- {statistics.pstdev(g):.1f}", fmt_p(paired_p(rnd, cl)),
                    f"{statistics.mean(gf):.1f}%"]
        else:
            row += ["-", "-", "-"]
        rows.append(row)
    table(f"Table 1. Storage strategies - picking policy fixed (FCFS batching of {mc.batch_orders} orders, dynamic assignment, "
          f"FCFS sequence) - {a.orders} orders, {mc.n_pickers} pickers, {mc.n_stations} stations",
          hdr, rows,
          "improvement (reduction) in %, positive = better than the comparison strategy. Utilization and waiting rows have no improvement.\n"
          "p = two-sided paired t-test of Cluster vs Random over the splits.")


def study_clusters(ctx):
    a = ctx.args
    acc = defaultdict(list)
    for sp in range(a.splits):
        skus, train, test = ctx.split(a.orders, sp)
        rr = ctx.rand_runs("clusters", sp, skus, test, a.cl_seeds)
        acc["random"].append({m: mean_of(rr, m) for m in ("walk_m", "makespan", "avg_order_time")})
        acc["frequency"].append(ctx.run("clusters", sp, "frequency", W.assign_frequency(skus, ctx.slots, train), test))
        ck, f = W.cluster_skus_k(skus, train, K_GRID)
        for k in K_GRID:
            acc[k].append(ctx.run("clusters", sp, f"cluster_k{k}", W.pack_clusters(ck[k], f, ctx.slots), test))
    base = acc["random"]
    rows = []
    for key, label in [("random", "Random (baseline)"), ("frequency", "Frequency (no clustering)")] + \
                      [(k, f"Cluster, k = {k}") for k in K_GRID]:
        rows.append([label, num(statistics.mean(r["walk_m"] for r in acc[key])),
                     num(statistics.mean(r["makespan"] for r in acc[key])),
                     f"{statistics.mean(pct(b['walk_m'], x['walk_m']) for b, x in zip(base, acc[key])):.1f}%",
                     f"{statistics.mean(pct(b['makespan'], x['makespan']) for b, x in zip(base, acc[key])):.1f}%"])
    table(f"Table 2. Q1 - number of clusters k (cluster sizes are free; 1 section = 6 slots) - {a.orders} orders",
          ["Storage", "Walking dist (m)", "Makespan (steps)", "Walking improvement vs Random", "Makespan improvement vs Random"], rows,
          "improvement = reduction in %, + = better. Reinforcement learning is not run here (see studies20.py).")


def study_batching(ctx):
    a = ctx.args
    methods = ["fcfs", "similarity", "section"]
    acc = defaultdict(list)
    for sp in range(a.splits):
        skus, train, test = ctx.split(a.orders, sp)
        casg, _ = cluster_plan(ctx, skus, train)
        for mth in methods:
            for sd in range(a.cl_seeds):
                acc[("random", mth)].append(ctx.run("batching", sp, "random", W.assign_random(skus, ctx.slots, 1000 + sd),
                                                    test, batching=mth))
            acc[("cluster", mth)].append(ctx.run("batching", sp, "cluster", casg, test, batching=mth))
    rows = []
    for st in ("random", "cluster"):
        b0 = acc[(st, "fcfs")]
        for mth in methods:
            r = acc[(st, mth)]
            rows.append([st, mth, num(mean_of(r, "walk_m")), num(mean_of(r, "avg_stops")), num(mean_of(r, "makespan")),
                         num(mean_of(r, "avg_order_time")),
                         f"{pct(mean_of(b0, 'walk_m'), mean_of(r, 'walk_m')):.1f}%",
                         f"{pct(mean_of(b0, 'makespan'), mean_of(r, 'makespan')):.1f}%"])
    table(f"Table 3. Q3 - batching method ({ctx.mcfg.batch_orders} orders per batch) - {a.orders} orders",
          ["Storage", "Batching", "Walking dist (m)", "Stops/batch", "Makespan (steps)", "Avg order time",
           "Walking improvement vs FCFS", "Makespan improvement vs FCFS"], rows,
          "fcfs = orders in arrival sequence | similarity = Jaccard similarity of SKUs | section = Jaccard similarity of the aisle stops.")


def study_assignment(ctx):
    a = ctx.args
    acc = defaultdict(list)
    combos = [(asg, sq) for asg in ("dynamic", "static") for sq in ("fcfs", "spt")]
    for sp in range(a.splits):
        skus, train, test = ctx.split(a.orders, sp)
        casg, _ = cluster_plan(ctx, skus, train)
        for asg, sq in combos:
            for sd in range(a.cl_seeds):
                acc[("random", asg, sq)].append(ctx.run("assignment", sp, "random", W.assign_random(skus, ctx.slots, 1000 + sd),
                                                        test, assign=asg, seq=sq))
            acc[("cluster", asg, sq)].append(ctx.run("assignment", sp, "cluster", casg, test, assign=asg, seq=sq))
    rows = []
    for st in ("random", "cluster"):
        for asg, sq in combos:
            r = acc[(st, asg, sq)]
            rows.append([st, asg, sq, num(mean_of(r, "makespan")), num(mean_of(r, "avg_order_time")),
                         num(mean_of(r, "walk_m")), num(mean_of(r, "station_wait")), f"{mean_of(r, 'picker_util'):.2f}"])
    table(f"Table 4. Q4 and Q5 - assignment of batches to pickers and sequence per picker ({ctx.mcfg.n_pickers} pickers) - {a.orders} orders",
          ["Storage", "Assignment", "Sequence", "Makespan (steps)", "Avg order time", "Walking dist (m)", "Station wait", "Picker util"],
          rows,
          "dynamic = next free picker takes the next batch | static = batch j goes to picker (j mod P) in advance\n"
          "fcfs = lowest batch number first | spt = batch with the shortest estimated tour first.")


def study_orders(ctx):
    a = ctx.args
    rows = []
    for n in (50, 100, 200, 300):
        g, gm, rm, cm, nb = [], [], [], [], 0
        for sp in range(a.splits):
            skus, train, test = ctx.split(n, sp)
            rr = ctx.rand_runs("orders", sp, skus, test, a.cl_seeds)
            asg, _ = cluster_plan(ctx, skus, train)
            cr = ctx.run("orders", sp, "cluster", asg, test)
            rw, rmk = mean_of(rr, "walk_m"), mean_of(rr, "makespan")
            g.append(pct(rw, cr["walk_m"]))
            gm.append(pct(rmk, cr["makespan"]))
            rm.append(rw / len(test))
            cm.append(cr["walk_m"] / len(test))
            nb = cr["batches"]
        rows.append([n, nb, num(statistics.mean(rm)), num(statistics.mean(cm)),
                     f"{statistics.mean(g):.1f}% +- {statistics.pstdev(g):.1f}",
                     f"{statistics.mean(gm):.1f}% +- {statistics.pstdev(gm):.1f}"])
    table("Table 5. Q2 - number of orders picked in one wave", ["Orders", "Batches", "Random walk/order (m)",
                                                                  "Cluster walk/order (m)", "Walking improvement", "Makespan improvement"], rows)


# ==========================================================================
# 9. DATA HOOKS
# ==========================================================================
def export_slots(ctx, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["slot_idx", "section_x", "section_z", "access_x", "access_z", "level", "basket", "steps_to_nearest_station", "steps_start_to_slot_to_packing"])
        for sl in ctx.slots:
            w.writerow([sl.idx, sl.sec[0], sl.sec[1], sl.access[0], sl.access[1], sl.level, sl.basket,
                        min(ctx.walk.d[(st, sl.access)] for st in ctx.walk.svc),
                        ctx.walk.d[(ctx.walk.start, sl.access)] + ctx.walk.end_cost[sl.access]])
    print("Slot table written to", path)


def load_assignment_file(path, slots, skus):
    """CSV with columns sku, slot_idx (slot_idx as in the exported slot table)."""
    m = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m[r["sku"]] = slots[int(r["slot_idx"])]
    assert set(m) == set(skus), "assignment file must contain every SKU exactly once"
    assert len({s.idx for s in m.values()}) == len(skus), "two SKUs share a slot"
    return m


def load_data(a):
    """Instacart files are detected automatically (header has order_id, product_id);
    products.csv is taken from --products or from the same folder as the dataset."""
    if a.synthetic:
        return W.synthetic_orders()
    if not os.path.exists(a.dataset):
        raise SystemExit(f"Dataset not found: {a.dataset}  (use --dataset path/to/order_products__train.csv)")
    with open(a.dataset, encoding="utf-8") as f:
        header = f.readline()
    if "order_id" in header and "product_id" in header:
        prod = a.products or os.path.join(os.path.dirname(a.dataset), "products.csv")
        if not os.path.exists(prod):
            raise SystemExit(f"products.csv not found: {prod}  (use --products path/to/products.csv)")
        from instacart_patch import load_instacart
        return load_instacart(prod, a.dataset, n_sku=546, max_orders=60000)
    return W.load_orders(a.dataset)


# ==========================================================================
# 10. MAIN
# ==========================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Manual picking simulation - storage assignment study")
    ap.add_argument("--dataset", default=os.path.join("archive", "order_products__train.csv"))
    ap.add_argument("--products", default=None, help="products.csv (Instacart format)")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--study", default="all", choices=["all", "storage", "clusters", "batching", "assignment", "orders"])
    ap.add_argument("--orders", type=int, default=100, help="orders picked in one wave")
    ap.add_argument("--splits", type=int, default=5, help="number of different train/test splits")
    ap.add_argument("--seeds", type=int, default=10, help="random layouts per split (main experiment)")
    ap.add_argument("--cl-seeds", type=int, default=3, help="random layouts per split (other studies)")
    ap.add_argument("--split-seed", type=int, default=0)
    add_picking_args(ap)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--pick-steps", type=int, default=3)
    ap.add_argument("--station-base", type=int, default=4)
    ap.add_argument("--station-per-item", type=int, default=1)
    ap.add_argument("--assignment-file", default=None, help="CSV (sku, slot_idx): evaluate an external storage plan, e.g. from RL")
    ap.add_argument("--export-slots", default=None, help="write the slot table (slot_idx, section, level, distance) and exit")
    ap.add_argument("--out", default="manual_results.csv")
    a = ap.parse_args()

    t0 = time.time()
    data = load_data(a)
    mcfg = replace(mcfg_from_args(a), batch_orders=a.batch, pick_steps=a.pick_steps, station_base=a.station_base, station_per_item=a.station_per_item)
    ctx = Ctx(data, mcfg, a)
    if a.export_slots:
        export_slots(ctx, a.export_slots)
        sys.exit(0)
    print(f"real SKU={len({s for o in data for s in o})} | slots={len(ctx.slots)} (empty={len(ctx.empty)}) | sections={ctx.lay.n_sections} | pick stops={len({s.access for s in ctx.slots})} | "
          f"layout=real warehouse grid {ctx.lay.W}x{ctx.lay.H} | pickers={mcfg.n_pickers} stations={mcfg.n_stations} | "
          f"batch={mcfg.batch_orders} orders | orders per wave={a.orders}")
    print("Time unit: 1 step = walking 1 grid cell. Picking = "
          f"{mcfg.pick_steps} steps per SKU; station = {mcfg.station_base} + {mcfg.station_per_item} x items steps per batch.")
    studies = {"storage": study_storage, "clusters": study_clusters, "batching": study_batching,
               "assignment": study_assignment, "orders": study_orders}
    for name, fn in studies.items():
        if a.study in ("all", name):
            t1 = time.time()
            fn(ctx)
            print(f"  ({name} study: {time.time() - t1:.0f} s)")
    keys = ["study", "split", "strategy", "batching", "assign", "seq"] + [k for k in ctx.runs[0] if k not in
            ("study", "split", "strategy", "batching", "assign", "seq")]
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(ctx.runs)
    print(f"\nAll runs written to {a.out} ({len(ctx.runs)} runs, {time.time() - t0:.0f} s)")