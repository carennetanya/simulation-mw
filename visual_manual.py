# -*- coding: utf-8 -*-
"""
visual_manual.py - 3D visualisation of the MANUAL picking simulation (manual_sim.py), all storage methods in one window
=====================================================================================================================
The engine is NOT changed: layout, shortest paths, TSP, batching, assignment and sequencing all come from manual_sim.py
(and wh_sim.py; 'optimized' batching from studies20.py; the storage plans from wh_sim.py, rl_storage.py and rl2.py).
manual_sim.simulate(trace=...) records a timeline of every batch; this file only plays it back, so the animation shows
exactly the numbers of the simulation (walking distance and makespan are checked against the engine for every method).

What you can do in the window
  * Storage method (menu): Random | Frequency | Data mining (clustering) | RL v1 | RL v2 (| plan from --assignment-file).
    All methods pick the SAME test orders with the SAME operating policy, so only the storage plan differs.
  * Carton colour (button): Frequency class  -> red = 20% most frequent SKUs, orange, yellow, green, blue = 20% least frequent
                            Co-purchase cluster -> one colour per cluster of SKUs that are often bought together
                            Plain
    With "Frequency class" you see at once where each method puts the popular SKUs (e.g. near the stations or spread out).
  * Traffic (button): the aisle floor is coloured by how often the pickers walk over each cell (grey = never, yellow, red = most);
    the colour scale is the same for all methods, so the pictures of different methods can be compared directly.
  * Comparison panel: walking distance, makespan and average order time of every method, and the reduction compared with random.
  * Labels (button): all | racks only (no "Going to ..." callouts, picker tags and section numbers) | none (clean pictures).
  * Save picture (button): downloads a PNG of the view, e.g. for the report (top view: key 1).

Run (same folder as manual_sim.py, wh_sim.py, instacart_patch.py, rl_storage.py, rl2.py; studies20.py and its imports
for --batching optimized):
    python visual_manual.py --products archive/products.csv --dataset archive/order_products__train.csv
    python visual_manual.py ... --batching optimized --seq spt          (configuration D of the report)
    python visual_manual.py ... --methods random,frequency,cluster,rl_v2 (leave out RL v1, which trains for a few minutes)
    python visual_manual.py ... --assignment-file rl_v2_plans/rl_v2_assignment_split0.csv   (adds a plan from a file)

The defaults are the settings of the report: 4 pickers, 1 order start station + 1 packing station, batch capacity 20 items,
extra pick steps per level 1,0,2, split 0, LAM = 6.4, 4,000 episodes for RL v1. Only offline waves (--arrival-rate 0) are shown.
The comparison panel shows ONE wave (one split, one random layout); the report averages 20 splits.

Controls: right-drag / Ctrl+left-drag = rotate | Shift+drag = pan | scroll = zoom | click a rack = its SKUs
          0 reset | 1 top | 2 front | 3 left | 4 right | Time slider = jump to any moment
"""
import argparse
import html
import time
from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace

from vpython import (canvas, box, cylinder, sphere, curve, label, vector, color,
                     compound, rate, button, slider, wtext, menu,
                     local_light, distant_light)

import manual_sim as M
import wh_sim as W

# ==========================================================================
# VISUAL CONSTANTS
# ==========================================================================
CANVAS_W, CANVAS_H = 1400, 760
LVL_Y0, LVL_H, CARTON_H, BASKET_GAP = 0.14, 0.62, 0.24, 0.03
POST = 0.045

PICKER_COLORS = [vector(0.20, 0.62, 1.00), vector(0.78, 0.30, 1.00), vector(1.00, 0.60, 0.12),
                 vector(0.25, 0.85, 0.45), vector(1.00, 0.35, 0.45), vector(0.95, 0.90, 0.25)]
PICKER_CSS = ["#33a0ff", "#c74dff", "#ff9a1f", "#40d973", "#ff5973", "#f2e640"]

STEEL_BLUE = vector(0.10, 0.27, 0.62)
BEAM_ORANGE = vector(0.95, 0.50, 0.10)
SHELF_GRAY = vector(0.42, 0.44, 0.48)
CARTON_PLAIN = vector(0.64, 0.44, 0.25)
TAPE_YELLOW = vector(0.95, 0.75, 0.12)
CLUSTER_PALETTE = [vector(0.90, 0.30, 0.30), vector(0.30, 0.65, 0.95), vector(0.35, 0.80, 0.40),
                   vector(0.95, 0.80, 0.25), vector(0.70, 0.40, 0.90), vector(0.95, 0.55, 0.20),
                   vector(0.25, 0.80, 0.80), vector(0.95, 0.45, 0.70), vector(0.60, 0.80, 0.25),
                   vector(0.55, 0.55, 0.95)]
# frequency classes: 0 = 20% most frequent SKUs ... 4 = 20% least frequent SKUs
FREQ_COLORS = [vector(0.90, 0.20, 0.20), vector(0.97, 0.55, 0.15), vector(0.95, 0.85, 0.25),
               vector(0.35, 0.80, 0.40), vector(0.30, 0.55, 0.95)]
FREQ_CSS = ["#e63333", "#f78c26", "#f2d940", "#59cc66", "#4d8cf2"]
FREQ_LABEL = ["20% most frequent SKUs", "next 20%", "middle 20%", "next 20%", "20% least frequent SKUs"]
TILE_COL = {"aisle": vector(0.31, 0.32, 0.36), "svc": vector(0.95, 0.62, 0.10), "start": vector(0.15, 0.70, 0.30)}

LED_COLORS = {"move": vector(0.2, 1.0, 0.4), "pick": vector(1.0, 1.0, 1.0),
              "service": vector(1.0, 0.85, 0.2), "wait": vector(1.0, 0.2, 0.2),
              "idle": vector(0.5, 0.5, 0.55)}
STATE_LED = {"walk": "move", "to_station": "move", "to_start": "move", "pick": "pick", "wait": "wait",
             "service": "service", "idle": "idle"}
STATE_LABEL = {"idle": "idle", "walk": "walking to rack", "pick": "picking", "to_station": "walking to packing station",
               "to_start": "walking back to order start", "wait": "waiting for the packing station",
               "service": "sorting at the packing station"}

PANEL_CSS = ("background:rgba(12,17,28,0.88);border:1px solid rgba(110,160,255,0.35);"
             "border-radius:9px;padding:9px 12px;color:#e8eefc;"
             "font:12px/1.45 'Segoe UI',Arial,sans-serif;box-shadow:0 2px 14px rgba(0,0,0,.5);")

METHOD_NAME = {"random": "Random", "frequency": "Frequency-based", "cluster": "Data mining (clustering)",
               "rl_v1": "RL v1 (constructive Q-learning)", "rl_v2": "RL v2 (improvement Q-learning)",
               "file": "Storage plan from file"}
METHOD_SHORT = {"random": "Random", "frequency": "Frequency", "cluster": "Data mining", "rl_v1": "RL v1",
                "rl_v2": "RL v2", "file": "File"}
COLOR_MODES = ["frequency", "cluster", "plain"]
LABEL_MODES = ["all", "racks", "none"]
LABEL_MODE_NAME = {"all": "all", "racks": "racks only", "none": "none"}
SLIDER_MAX = 1000          # time slider in 1/1000 of the makespan (a VPython slider cannot change its max after creation)
COLOR_MODE_NAME = {"frequency": "Frequency class", "cluster": "Co-purchase cluster", "plain": "Plain"}


def basket_y(level, basket):
    y_shelf = LVL_Y0 + (level - 1) * LVL_H
    return y_shelf + 0.015 + CARTON_H / 2 + (basket - 1) * (CARTON_H + BASKET_GAP)


LAY = None


def wpos(c, y=0.0):
    x, z = LAY.pt(c)
    return vector(x, y, z)


def wrap_panel(left, top, width, inner):
    return (f'<div style="position:absolute;left:{left}px;top:{top}px;width:{width}px;'
            f'z-index:50;pointer-events:none;{PANEL_CSS}">{inner}</div>')


def compress(cells):
    """keep corner cells only so that path lines stay light."""
    out, last, ld = [], None, None
    for c in cells:
        if last is None:
            out.append(c)
            last = c
            continue
        if c == last:
            continue
        d = (c[0] - last[0], c[1] - last[1])
        if ld is not None and d != ld:
            out.append(last)
        ld, last = d, c
    if last is not None and (not out or out[-1] != last):
        out.append(last)
    return out


def comp(parts, col):
    if parts:
        c = compound(parts)
        c.color = col
        return c
    return None


def heat(v):
    """0 -> aisle grey, 0.5 -> yellow, 1 -> red."""
    g, y, r = TILE_COL["aisle"], vector(0.98, 0.85, 0.20), vector(0.92, 0.15, 0.12)
    v = max(0.0, min(1.0, v))
    return g + (y - g) * (2 * v) if v < 0.5 else y + (r - y) * (2 * v - 1)


# ==========================================================================
# PICKER (simple person with a cart)
# ==========================================================================
class PickerVis:
    def __init__(self, pid, col):
        self.pid = pid
        self.body = cylinder(axis=vector(0, 0.30, 0), radius=0.075, color=col)
        self.head = sphere(radius=0.058, color=vector(0.95, 0.80, 0.65))
        self.cart = box(size=vector(0.20, 0.10, 0.14), color=CARTON_PLAIN, visible=False)
        self.led = sphere(radius=0.03, color=LED_COLORS["idle"], emissive=True)
        self.tag = label(text=f"P{pid + 1}", height=10, box=False, line=False, opacity=0,
                         color=color.white, yoffset=-18)
        self.place(vector(0, 0, 0))

    def place(self, p):
        y = 0.02
        self.body.pos = vector(p.x, y, p.z)
        self.head.pos = vector(p.x, y + 0.36, p.z)
        self.led.pos = vector(p.x, y + 0.45, p.z)
        self.cart.pos = vector(p.x + 0.13, y + 0.06, p.z)
        self.tag.pos = vector(p.x, y, p.z)

    def set_carry(self, on):
        self.cart.visible = on

    def set_state(self, state):
        self.led.color = LED_COLORS.get(STATE_LED.get(state, "idle"), LED_COLORS["idle"])


# ==========================================================================
# STORAGE PLANS (same functions and parameters as the experiments of the report)
# ==========================================================================
def build_plans(ctx, skus, train, args):
    """-> (plans {method: assignment}, clusters of the data-mining plan, k)"""
    want = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in want:
        if m not in ("random", "frequency", "cluster", "rl_v1", "rl_v2"):
            raise SystemExit(f"unknown method {m!r} in --methods")
    plans = {}
    t0 = time.time()
    fasg = W.assign_frequency(skus, ctx.slots, train)
    print("building the data-mining clusters (k chosen on validation orders unless --k is given) ...", flush=True)
    k = args.k if args.k is not None else M.select_k(ctx, skus, train, n_val=args.orders)[0]
    casg, clusters = W.assign_cluster_k(skus, ctx.slots, train, k)       # also used for the cluster colours
    for m in want:
        if m == "random":
            plans[m] = W.assign_random(skus, ctx.slots, args.seed)
        elif m == "frequency":
            plans[m] = fasg
        elif m == "cluster":
            plans[m] = casg
        elif m == "rl_v1":
            import rl_storage as R1
            print(f"training RL v1 ({args.v1_episodes} episodes) ...", flush=True)
            plans[m], _, _ = R1.train_rl(skus, train, ctx, episodes=args.v1_episodes, seed=args.split_seed, verbose=False)
        elif m == "rl_v2":
            import numpy as np
            import rl2
            import rl_storage as R1
            print(f"training RL v2 (LAM = {args.lam}) ...", flush=True)
            secs, slots_of, D, dst, _ = R1.section_tables(ctx)
            sidx = {s: i for i, s in enumerate(secs)}
            f, C = rl2.dense_stats(skus, train)
            s0 = np.array([sidx[fasg[x].sec] for x in skus])
            s, _, _ = rl2.train_improve(f, C, D, dst, s0, lam=args.lam, seed=args.split_seed)
            plans[m] = rl2.to_assignment(s, skus, f, secs, slots_of)
    if args.assignment_file:
        plans["file"] = M.load_assignment_file(args.assignment_file, ctx.slots, skus)
    print(f"storage plans ready: {', '.join(plans)} ({time.time() - t0:.0f} s)", flush=True)
    return plans, clusters, k


# ==========================================================================
# TIMELINE: replay of manual_sim.simulate (position of every picker at every step)
# ==========================================================================
def build_timelines(ctx, mc, trace, assignment):
    """Replays manual_sim.simulate(trace=...) step by step (one entry per time step and picker).
    trace entry: picker, batch, orders, skus, route, start_pos, t0, t_done, station, arr, start, end (see manual_sim.simulate).
    Tour of a batch: order start station -> route -> packing station; pick time per SKU = pick_steps + level_extra[level - 1].
    After a batch the picker walks back to the order start station, but only if it takes another batch (the engine counts the
    walk back when the next batch starts and does not walk the last return of a picker)."""
    walk, lay = ctx.walk, ctx.lay
    P, st_cells, start = mc.n_pickers, walk.svc, walk.start
    T_end = max(e["end"] for e in trace)          # = makespan: an order is finished when its batch leaves the packing station
    tl = dict(P=P, T_end=T_end, pos=[], st=[], bat=[], pi=[], cum=[], back_m=0.0, batches={}, order_t0={}, order_end={})
    by_p = defaultdict(list)
    for e in sorted(trace, key=lambda e: (e["t0"], e["picker"])):
        by_p[e["picker"]].append(e)
    for p in range(P):
        pos, st, bat, pi, cum = [start], ["idle"], [None], [0], [0.0]      # every picker starts at the ORDER START station

        def hold(n, state, b, i):
            for _ in range(n):
                pos.append(pos[-1]); st.append(state); bat.append(b); pi.append(i); cum.append(cum[-1])

        def move(c, state, b, i):
            m = lay.step_m(pos[-1], c)
            pos.append(c); st.append(state); bat.append(b); pi.append(i); cum.append(cum[-1] + m)

        first = True
        for e in by_p[p]:
            if not first:                              # walk back: packing station -> order start station (empty)
                m0 = cum[-1]
                for nxt in walk.path(pos[-1], start)[1:]:
                    move(nxt, "to_start", None, 0)
                tl["back_m"] += cum[-1] - m0
            first = False
            assert pos[-1] == e["start_pos"], "timeline mismatch (order start station)"
            assert len(pos) - 1 <= e["t0"], "timeline mismatch (walk back)"
            hold(e["t0"] - (len(pos) - 1), "idle", None, 0)
            b = e["batch"]
            fp = [pos[-1]]
            cell_skus = defaultdict(list)
            for s in e["skus"]:
                cell_skus[assignment[s].access].append(s)
            pick_time = {}
            for c in e["route"]:
                for nxt in walk.path(pos[-1], c)[1:]:
                    fp.append(nxt)
                    move(nxt, "walk", b, len(fp) - 1)
                for s in sorted(cell_skus[c]):
                    hold(mc.pick_steps + mc.level_extra[assignment[s].level - 1], "pick", b, len(fp) - 1)
                    pick_time[s] = len(pos) - 1
            assert len(pos) - 1 == e["t_done"], "timeline mismatch (picking)"
            for nxt in walk.path(pos[-1], st_cells[e["station"]])[1:]:
                fp.append(nxt)
                move(nxt, "to_station", b, len(fp) - 1)
            assert len(pos) - 1 == e["arr"], "timeline mismatch (arrival at the packing station)"
            hold(e["start"] - e["arr"], "wait", b, len(fp) - 1)
            hold(e["end"] - e["start"], "service", b, len(fp) - 1)
            assert len(pos) - 1 == e["end"], "timeline mismatch (packing station)"
            tl["batches"][b] = dict(picker=p, t0=e["t0"], end=e["end"], orders=e["orders"], skus=e["skus"],
                                    route=e["route"], station=e["station"], fp=fp, pick_time=pick_time,
                                    cell_skus=dict(cell_skus))
            for i in e["orders"]:
                tl["order_t0"][i], tl["order_end"][i] = e["t0"], e["end"]
        hold(T_end - (len(pos) - 1), "idle", None, 0)
        for key, arr in (("pos", pos), ("st", st), ("bat", bat), ("pi", pi), ("cum", cum)):
            tl[key].append(arr)
    return tl


def traffic_counts(tl):
    """number of times a picker steps onto each cell during the wave (walking only)."""
    cnt = defaultdict(int)
    for pos in tl["pos"]:
        for a, b in zip(pos, pos[1:]):
            if a != b:
                cnt[b] += 1
    return cnt


def next_target(bi, T):
    for c in bi["route"]:
        for s in bi["cell_skus"][c]:
            if bi["pick_time"][s] > T:
                return s
    return None


def orders_panel_html(TL, test, n_orders, T, limit=5, max_name=22):
    """orders that are in progress; struck-through item = already picked by the picker."""
    batch_of = {i: b for b, bi in TL["batches"].items() for i in bi["orders"]}
    active = sorted((i for i in range(n_orders) if TL["order_t0"][i] <= T < TL["order_end"][i]),
                    key=lambda i: (TL["order_t0"][i], i))
    done = sum(1 for i in range(n_orders) if TL["order_end"][i] <= T)

    def nm(x):
        x = x if len(x) <= max_name else x[:max_name - 1] + "\u2026"
        return html.escape(x)

    parts = [f'<div style="font-size:14px;font-weight:700;margin-bottom:2px">Current Orders</div>'
             f'<div style="color:#8fa0c8;font-size:11px;margin-bottom:4px">'
             f'{done}/{n_orders} done &nbsp;|&nbsp; struck = already picked</div>']
    for i in active[:limit]:
        b = batch_of[i]
        bi = TL["batches"][b]
        items = sorted(set(test[i]))
        got = [s for s in items if bi["pick_time"][s] <= T]
        txt = ", ".join((f'<span style="color:#6f7a99;text-decoration:line-through">{nm(s)}</span>'
                         if bi["pick_time"][s] <= T else f'<span style="color:#fff">{nm(s)}</span>') for s in items)
        parts.append(f'<div style="border-top:1px solid #2a3350;padding:3px 0;font-size:11px;line-height:1.35">'
                     f'<b style="color:#8cd4ff">Order #{i + 1}</b> '
                     f'<span style="color:#aab4d0">batch #{b + 1} &nbsp;|&nbsp; {len(got)}/{len(items)} picked</span>'
                     f'<br>{txt}</div>')
    if len(active) > limit:
        parts.append(f'<div style="color:#8fa0c8;font-size:11px;border-top:1px solid #2a3350;padding-top:3px">'
                     f'+ {len(active) - limit} more orders in progress</div>')
    elif not active:
        parts.append('<div style="color:#8fa0c8;font-size:11px">No active orders</div>')
    return '<div style="max-height:200px;overflow:hidden">' + "".join(parts) + '</div>'


# ==========================================================================
# MAIN
# ==========================================================================
def run_visual(args):
    # ---------------- 1) data + engine (identical to manual_sim) ----------------
    mc = replace(M.mcfg_from_args(args), pick_steps=args.pick_steps, station_base=args.station_base,
                 station_per_item=args.station_per_item)
    if mc.arrival_rate > 0:
        raise SystemExit("visual_manual.py shows offline waves only (all orders known at time 0): use --arrival-rate 0")
    data = M.load_data(args)
    ctx = M.Ctx(data, mc, SimpleNamespace(split_seed=args.split_seed))
    if args.batching == "optimized":
        import studies20 as S20                      # defines the 'optimized' batching method
        S20._WALK["w"] = ctx.walk
    global LAY
    lay = LAY = ctx.lay
    cfg = lay.cfg
    skus, train, test = ctx.split(args.orders, 0)
    n_orders = len(test)
    plans, clusters, k_sel = build_plans(ctx, skus, train, args)

    # every method: same orders, same operating policy -> simulate + replay
    runs = {}
    for m, asg in plans.items():
        trace = []
        met = M.simulate(ctx.walk, mc, asg, test, batching=args.batching, assign=args.assign, seq=args.seq, trace=trace)
        TL = build_timelines(ctx, mc, trace, asg)
        total = sum(TL["cum"][p][TL["T_end"]] for p in range(TL["P"]))
        assert abs(total - met["walk_m"]) < 1e-6, f"animation distance differs from the simulation ({m})"
        assert TL["T_end"] == met["makespan"], f"animation time differs from the simulation ({m})"
        runs[m] = dict(asg=asg, metrics=met, TL=TL, total_walk=total, slot_sku={sl.idx: s for s, sl in asg.items()},
                       traffic=None)
    methods = list(runs)
    base_m = "random" if "random" in runs else methods[0]
    P = mc.n_pickers

    # SKU properties used for the carton colours (from the TRAIN orders, like the plans)
    freq = W.sku_frequency(train)
    rank = sorted(skus, key=lambda s: (-freq[s], s))
    fclass = {s: min(4, r * 5 // len(skus)) for r, s in enumerate(rank)}
    pair = defaultdict(int)                          # co-purchase counts in the TRAIN orders (same data as the plans)
    for o in train:
        os_ = sorted(set(o))
        for a_i in range(len(os_)):
            for b_i in range(a_i + 1, len(os_)):
                pair[(os_[a_i], os_[b_i])] += 1
    partner = {}                                     # SKU -> (strongest co-purchase partner, orders together)
    for (a_, b_), c in pair.items():
        for x_, y_ in ((a_, b_), (b_, a_)):
            if c > partner.get(x_, (None, 0))[1] or (c == partner.get(x_, (None, 0))[1] and y_ < partner[x_][0]):
                partner[x_] = (y_, c)
    clusters_sorted = sorted(clusters, key=lambda c: (-len(c), sorted(c)[0]))
    sku_cluster = {}
    for i, c in enumerate(clusters_sorted):
        for s in c:
            sku_cluster[s] = i if len(c) > 1 else -1
    n_multi = sum(1 for c in clusters if len(c) > 1)
    sec_no = {s["sec"]: i + 1 for i, s in enumerate(lay.sections)}

    cap_txt = f"{mc.batch_items} items" if mc.batch_items > 0 else f"{mc.batch_orders} orders"
    lvl_txt = "-".join(map(str, mc.level_extra))
    print(f"\nSKU={len(skus)} slot={len(ctx.slots)} | sections={lay.n_sections} | pickers={P} | 1 order start + "
          f"{mc.n_stations} packing station(s) | orders={n_orders} | batch capacity={cap_txt} | level extra={lvl_txt} | "
          f"batching={args.batching} assignment={args.assign} sequence={args.seq} | data-mining k={k_sel} ({n_multi} clusters > 1 SKU)")
    bw, bm, bo = (runs[base_m]["metrics"][x] for x in ("walk_m", "makespan", "avg_order_time"))
    print(f"\n{'method':34s} {'walking (m)':>12s} {'makespan':>9s} {'avg order':>10s} {'walk vs ' + METHOD_SHORT[base_m]:>16s}")
    for m in methods:
        r = runs[m]["metrics"]
        print(f"{METHOD_NAME[m]:34s} {r['walk_m']:12.1f} {r['makespan']:9,} {r['avg_order_time']:10.1f} "
              f"{100 * (bw - r['walk_m']) / bw:+15.1f}%")

    st = dict(method=args.start_method if args.start_method in runs else base_m,
              color=args.color if args.color in COLOR_MODES else "frequency", traffic=False,
              labels=args.labels if args.labels in LABEL_MODES else "all")

    def R():
        return runs[st["method"]]

    # ---------------- 2) scene ----------------
    start_cells = sorted(set(ctx.walk.nb) - set(lay.cells))      # dead-end path to the ORDER START station (manual_sim.start_spur)
    all_cells = list(lay.cells) + start_cells
    allpts = [lay.pt(c) for c in all_cells]
    gx = [p[0] for p in allpts] + [lay.geom(q)["cx"] for q in lay.sections + lay.unbuilt_secs]
    gz = [p[1] for p in allpts] + [lay.geom(q)["cz"] for q in lay.sections + lay.unbuilt_secs]
    x_lo, x_hi, z_lo, z_hi = min(gx) - 0.6, max(gx) + 0.6, min(gz) - 1.2, max(gz) + 0.7
    cx0, cz0 = (x_lo + x_hi) / 2, (z_lo + z_hi) / 2 - 0.1
    VIEW_CENTER = vector(cx0, 0.6, cz0)
    VIEW_FWD = vector(0, -0.60, 0.80)
    VIEW_RANGE = max(x_hi - x_lo, z_hi - z_lo) * 0.55
    scene = canvas(title="", width=CANVAS_W, height=CANVAS_H, center=VIEW_CENTER,
                   background=vector(0.04, 0.045, 0.06))
    scene.append_to_caption("<style>html,body{margin:0!important;padding:0!important;"
                            "background:#0b0d12!important;color:#dde4f5!important;}</style>")
    scene.lights = []
    scene.ambient = color.gray(0.38)
    distant_light(direction=vector(0.4, -1, 0.5), color=color.gray(0.65))
    distant_light(direction=vector(-0.6, -0.5, -0.4), color=color.gray(0.25))
    local_light(pos=vector(cx0, 4.5, cz0), color=color.gray(0.35))
    scene.userspin = scene.userzoom = scene.userpan = True
    scene.autoscale = False

    def set_view(center, fwd, rng):
        scene.up = vector(0, 1, 0)
        scene.center, scene.forward, scene.range = center, fwd, rng

    set_view(VIEW_CENTER, VIEW_FWD, VIEW_RANGE)

    # ---------------- 3) floor, aisles (one tile per cell, for the traffic colours), stations ----------------
    fw, fd = x_hi - x_lo, z_hi - z_lo
    fcx, fcz = (x_lo + x_hi) / 2, (z_lo + z_hi) / 2
    box(pos=vector(fcx, -0.065, fcz), size=vector(fw, 0.11, fd), color=vector(0.07, 0.075, 0.09))
    box(pos=vector(fcx, -0.01, fcz), size=vector(fw - 0.3, 0.02, fd - 0.3), color=vector(0.25, 0.26, 0.29))

    def strip(x, z, sx, sz):
        return box(pos=vector(x, 0.004, z), size=vector(sx, 0.006, sz))

    def outline(cx, cz, w, d, m=0.0, t=0.03):
        Wd, D = w + 2 * m, d + 2 * m
        return [strip(cx, cz - D / 2, Wd, t), strip(cx, cz + D / 2, Wd, t),
                strip(cx - Wd / 2, cz, t, D), strip(cx + Wd / 2, cz, t, D)]

    svc_cells = set(lay.svc)
    cdim = lay.cell_dim()
    tile, tile_base = {}, {}
    for c in all_cells:
        key = "svc" if c in svc_cells else ("start" if c == ctx.walk.start else "aisle")
        tile_base[c] = TILE_COL[key]
        tile[c] = box(pos=wpos(c, 0.008), size=vector(cdim[0] * 0.92, 0.01, cdim[1] * 0.92), color=TILE_COL[key])

    tape = outline((x_lo + x_hi) / 2, (z_lo + z_hi) / 2, x_hi - x_lo - 0.3, z_hi - z_lo - 0.3, m=0.0)
    for q in lay.sections:
        g = lay.geom(q)
        tape += outline(g["cx"], g["cz"], g["sx"], g["sz"], m=0.02, t=0.02)
    for q in lay.unbuilt_secs:
        g = lay.geom(q)
        tape += outline(g["cx"], g["cz"], g["sx"], g["sz"], m=0.0, t=0.012)
    comp(tape, TAPE_YELLOW)

    station_labels, sec_labels, rack_labels = [], [], []      # kept so that they can be hidden (button "Labels")
    for i, sv in enumerate(lay.svc):                 # packing station: table + rollers + label
        ex = lay.exit[i]
        p1, p2 = lay.pt(sv), lay.pt(ex)
        mx, pz = (p1[0] + p2[0]) / 2, p1[1] - 0.50
        box(pos=vector(mx, 0.05, pz), size=vector(1.10, 0.10, 0.50), color=vector(0.20, 0.21, 0.25))
        rollers = [box(pos=vector(mx - 0.50 + 0.09 * k, 0.103, pz), size=vector(0.025, 0.012, 0.44))
                   for k in range(12)]
        comp(rollers, vector(0.55, 0.57, 0.62))
        box(pos=vector(mx - 0.25, 0.17, pz), size=vector(0.22, 0.14, 0.26), color=CARTON_PLAIN)
        name = "PACKING STATION" if len(lay.svc) == 1 else f"PACKING STATION {i + 1}"
        station_labels.append(label(pos=vector(mx, 0.35, pz - 0.40), text=name, height=13, box=True,
                                    color=color.white, background=vector(0.06, 0.07, 0.10), opacity=0.92))
    sx0, sz0 = lay.pt(ctx.walk.start)                # order start station (green)
    box(pos=vector(sx0, 0.05, sz0 - 0.50), size=vector(0.80, 0.10, 0.45), color=vector(0.20, 0.21, 0.25))
    box(pos=vector(sx0, 0.17, sz0 - 0.50), size=vector(0.22, 0.14, 0.26), color=CARTON_PLAIN)
    station_labels.append(label(pos=vector(sx0, 0.35, sz0 - 0.90), text="ORDER START", height=13, box=True,
                                color=color.white, background=vector(0.06, 0.07, 0.10), opacity=0.92))

    # ---------------- 4) racks (per section) and one carton per slot ----------------
    n_lv = cfg.levels
    total_h = LVL_Y0 + n_lv * LVL_H + 0.05
    posts, beams, shelves = [], [], []

    def build_frame(g):
        cx, cz, sx, sz = g["cx"], g["cz"], g["sx"], g["sz"]
        for ox in (-1, 1):
            for oz in (-1, 1):
                posts.append(box(pos=vector(cx + ox * (sx / 2 - POST / 2), total_h / 2,
                                            cz + oz * (sz / 2 - POST / 2)),
                                 size=vector(POST, total_h, POST)))
        for lv in range(n_lv + 1):
            y = LVL_Y0 + lv * LVL_H - 0.035
            for oz in (-1, 1):
                beams.append(box(pos=vector(cx, y, cz + oz * (sz / 2 - 0.02)),
                                 size=vector(sx, 0.06, 0.035)))
        for lv in range(n_lv):
            y0 = LVL_Y0 + lv * LVL_H
            shelves.append(box(pos=vector(cx, y0, cz), size=vector(sx - 0.02, 0.03, sz - 0.02)))
            shelves.append(box(pos=vector(cx, y0 + 0.015 + CARTON_H + BASKET_GAP / 2, cz),
                               size=vector(sx - 0.02, 0.012, sz - 0.02)))

    geoms = {}
    for q in lay.sections + lay.unbuilt_secs:
        geoms[q["sec"]] = lay.geom(q)
        build_frame(geoms[q["sec"]])
    comp(posts, STEEL_BLUE)
    comp(beams, BEAM_ORANGE)
    comp(shelves, SHELF_GRAY)
    carton = {}                                      # slot idx -> box (positions fixed, colour depends on the method)
    for sl in ctx.slots:
        g = geoms[sl.sec]
        yb = basket_y(sl.level, sl.basket) - CARTON_H / 2 + CARTON_H * 0.45
        carton[sl.idx] = box(pos=vector(g["cx"], yb, g["cz"]), size=vector(g["sx"] * 0.80, CARTON_H * 0.9, g["sz"] * 0.80),
                             color=CARTON_PLAIN)

    click_targets = []
    for q in lay.sections:                           # transparent click targets + section numbers
        g = geoms[q["sec"]]
        t = box(pos=vector(g["cx"], total_h / 2, g["cz"]),
                size=vector(g["sx"] + 0.04, total_h + 0.1, g["sz"] + 0.04), opacity=0.02, color=color.white)
        t.sec = q["sec"]
        click_targets.append(t)
        sec_labels.append(label(pos=vector(g["cx"], total_h + 0.12, g["cz"]), text=f"S{sec_no[q['sec']]:02d}", height=8,
                                box=False, line=False, color=vector(0.8, 0.85, 0.95), opacity=0))
    by = defaultdict(list)                           # rack labels R1..R15
    for q in lay.sections + lay.unbuilt_secs:
        by[q["rack"]].append(geoms[q["sec"]])
    for rid, gl in by.items():
        rack_labels.append(label(pos=vector(sum(g["cx"] for g in gl) / len(gl), total_h + 0.55, sum(g["cz"] for g in gl) / len(gl)),
                                 text=f"{rid} ({lay.rack_count[rid]}/{len(gl)})", height=13, box=True,
                                 color=color.white, background=vector(0.06, 0.07, 0.10), opacity=0.9,
                                 linecolor=vector(0.6, 0.65, 0.75), border=5))
    rack_of = {q["sec"]: q.get("rack", "") for q in lay.sections}

    def carton_color(sku):
        if st["color"] == "frequency":
            return FREQ_COLORS[fclass[sku]]
        if st["color"] == "cluster":
            c = sku_cluster.get(sku, -1)
            return CARTON_PLAIN if c < 0 else CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)]
        return CARTON_PLAIN

    def recolor_cartons():
        ss = R()["slot_sku"]
        for sl in ctx.slots:
            carton[sl.idx].color = carton_color(ss[sl.idx])

    def apply_tiles():
        if not st["traffic"]:
            for c, b in tile.items():
                b.color = tile_base[c]
            return
        for r_ in runs.values():                     # one common colour scale for all methods (fair comparison)
            if r_["traffic"] is None:
                r_["traffic"] = traffic_counts(r_["TL"])
        cnt = R()["traffic"]
        mx = max([r_["traffic"][c] for r_ in runs.values() for c in tile if tile_base[c] is TILE_COL["aisle"]] + [1])
        for c, b in tile.items():
            b.color = heat(cnt[c] / mx) if (tile_base[c] is TILE_COL["aisle"] and cnt[c] > 0) else tile_base[c]

    # ---------------- 5) pickers, path, target markers, callouts ----------------
    pickers, paths, callouts, markers, cur_batch = {}, {}, {}, {}, {}
    maxk = max(len(bi["skus"]) for r in runs.values() for bi in r["TL"]["batches"].values())
    hh = CARTON_H / 2 + 0.02
    for p in range(P):
        col = PICKER_COLORS[p % len(PICKER_COLORS)]
        pickers[p] = PickerVis(p, col)
        pickers[p].place(wpos(ctx.walk.start))
        paths[p] = curve(color=col, radius=0.014, emissive=True)
        callouts[p] = label(pos=vector(0, 0, 0), text="", height=11, box=True, line=True, linecolor=col,
                            color=color.white, background=vector(0.06, 0.08, 0.13), opacity=0.92,
                            xoffset=0, yoffset=60, border=6, visible=False)
        markers[p] = [dict(sku=None, glow=box(color=col, emissive=True, opacity=0.30, visible=False,
                                              size=vector(0.02, 0.1, 0.1))) for _ in range(maxk)]
        cur_batch[p] = "init"

    def loc_text(sku):
        sl = R()["asg"][sku]
        return f"S{sec_no[sl.sec]:02d}-L{sl.level}-B{sl.basket}"

    def item_geo(sku):
        sl = R()["asg"][sku]
        g = geoms[sl.sec]
        fx, fz = g["face"]
        front = vector(g["cx"] + fx * (g["sx"] / 2 + 0.012), basket_y(sl.level, sl.basket),
                       g["cz"] + fz * (g["sz"] / 2 + 0.012))
        hw = (g["sz"] if fx else g["sx"]) * 0.46
        return front, vector(fx, 0, fz), hw

    def setup_markers(p, b):
        sk = R()["TL"]["batches"][b]["skus"] if b is not None else []
        for k, m in enumerate(markers[p]):
            if k >= len(sk):
                m["sku"] = None
                m["glow"].visible = False
                continue
            m["sku"] = sk[k]
            f, face, hw = item_geo(sk[k])
            m["glow"].pos = f + face * 0.006
            m["glow"].size = vector(0.02, 2 * hh, 2 * hw) if face.x else vector(2 * hw, 2 * hh, 0.02)

    # ---------------- 6) panels ----------------
    scene.append_to_caption("\n")
    pick_w, kpi_w, legend_w, info_w, orders_w, cmp_w = (wtext(text="") for _ in range(6))
    ui = {"paused": False, "sec_open": False, "tps": float(args.speed), "tf": 0.0, "T": -1, "kpiT": -99, "printed": False}

    def pickers_html(T):
        TL = R()["TL"]
        parts = ['<div style="font-size:14px;font-weight:700;margin-bottom:6px">Pickers & Batches</div>']
        for p in range(P):
            col = PICKER_CSS[p % len(PICKER_CSS)]
            stt, b = TL["st"][p][T], TL["bat"][p][T]
            head = (f'<div style="border-top:1px solid #2a3350;padding:4px 0 2px"><b style="color:{col}">P{p + 1}</b> '
                    f'<span style="color:#aab4d0">{STATE_LABEL[stt]}')
            if b is not None:
                head += f' &nbsp;|&nbsp; batch #{b + 1}'
            parts.append(head + '</span></div>')
            if b is not None:
                bi = TL["batches"][b]
                got = sum(1 for s in bi["skus"] if bi["pick_time"][s] <= T)
                parts.append(f'<div style="color:#8fa0c8">picked {got}/{len(bi["skus"])} SKUs &nbsp;|&nbsp; '
                             f'walked {TL["cum"][p][T]:.1f} m</div>')
                nxt = next_target(bi, T)
                if nxt and stt in ("walk", "pick"):
                    parts.append(f'<div style="color:#fff">next: {html.escape(nxt)} '
                                 f'<span style="color:#8fa0c8">({loc_text(nxt)})</span></div>')
            else:
                parts.append(f'<div style="color:#8fa0c8">walked {TL["cum"][p][T]:.1f} m</div>')
        return wrap_panel(12, 12, 340, "".join(parts))

    def kpi_html(T):
        TL, r = R()["TL"], R()
        T_end = TL["T_end"]
        done = sum(1 for i in range(n_orders) if TL["order_end"][i] <= T)
        bdone = sum(1 for bi in TL["batches"].values() if bi["end"] <= T)
        walked = sum(TL["cum"][p][T] for p in range(P))
        waiting = sum(1 for p in range(P) if TL["st"][p][T] == "wait")
        head = f'<b style="font-size:14px">{METHOD_NAME[st["method"]]}</b>' + (" - FINISHED" if T >= T_end else "")
        inner = (f'<div style="text-align:center;line-height:1.6">{head}<br>'
                 f'{P} pickers &nbsp;|&nbsp; batch capacity {cap_txt} &nbsp;|&nbsp; extra steps level 1-2-3: {lvl_txt}<br>'
                 f'Batching: {args.batching} &nbsp;|&nbsp; Assignment: {args.assign} &nbsp;|&nbsp; Sequence: {args.seq}<br>'
                 f'Time: {T:,} / {T_end:,} steps<br>'
                 f'Orders done: {done}/{n_orders} &nbsp;|&nbsp; Batches done: {bdone}/{len(TL["batches"])}<br>'
                 f'Walking so far: {walked:,.0f} m (total {r["total_walk"]:,.0f} m)<br>'
                 f'Waiting for the packing station now: {waiting}</div>')
        return wrap_panel(CANVAS_W - 412, 12, 400, inner)

    def orders_html(T):
        if ui["sec_open"]:
            return ""
        return wrap_panel(CANVAS_W - 412, 205, 400, orders_panel_html(R()["TL"], test, n_orders, T))

    def compare_html():
        rows = []
        for m in methods:
            r = runs[m]["metrics"]
            on = m == st["method"]
            sty = "background:rgba(90,140,255,0.25);font-weight:700" if on else ""
            imp = 100 * (bw - r["walk_m"]) / bw
            rows.append(f'<tr style="{sty}"><td>{METHOD_SHORT[m]}</td><td align="right">{r["walk_m"]:,.0f}</td>'
                        f'<td align="right">{r["makespan"]:,}</td><td align="right">{r["avg_order_time"]:,.0f}</td>'
                        f'<td align="right">{imp:+.1f}%</td></tr>')
        inner = ('<div style="font-size:14px;font-weight:700;margin-bottom:4px">Storage methods (same orders)</div>'
                 '<table style="width:100%;border-collapse:collapse;font-size:11.5px">'
                 '<tr style="color:#8fa0c8"><td>method</td><td align="right">walk (m)</td><td align="right">makespan</td>'
                 f'<td align="right">order time</td><td align="right">walk vs {METHOD_SHORT[base_m].lower()}</td></tr>'
                 + "".join(rows) + '</table>'
                 f'<div style="color:#8fa0c8;font-size:10.5px;margin-top:4px">One wave of {n_orders} orders (split {args.split_seed}). '
                 'The report averages 20 splits.</div>')
        return wrap_panel(12, CANVAS_H - 205, 400, inner)

    def swatch(css, kind="dot"):
        if kind == "dot":
            return (f'<span style="display:inline-block;width:13px;height:13px;border-radius:50%;'
                    f'background:{css};margin-right:9px;vertical-align:middle"></span>')
        return (f'<span style="display:inline-block;width:18px;height:13px;background:{css};'
                f'margin-right:8px;vertical-align:middle"></span>')

    def legend_html():
        rows = [f'<div style="font-weight:700;margin-bottom:4px">Carton colour: {COLOR_MODE_NAME[st["color"]]}</div>']
        if st["color"] == "frequency":
            rows += [swatch(FREQ_CSS[i], "box") + FREQ_LABEL[i] for i in range(5)]
        elif st["color"] == "cluster":
            rows += [swatch("#e64d4d", "box") + swatch("#4da6f2", "box") + swatch("#59cc66", "box") + "one colour per cluster",
                     f'<span style="color:#8fa0c8">(k = {k_sel}; colours repeat after 10)</span>',
                     swatch("#a3703f", "box") + "SKU without a co-purchase partner"]
        else:
            rows += [swatch("#a3703f", "box") + "carton"]
        rows += ['<div style="border-top:1px solid #2a3350;margin:5px 0 3px"></div>',
                 swatch("#26b34d", "box") + "Order start station", swatch("#f0a01a", "box") + "Packing station",
                 swatch("#4cc9ff", "box") + "SKUs still to pick in a batch",
                 '<span style="margin-right:8px;color:#8cd4ff;font-weight:700">&#8594;</span>Planned route (TSP)']
        if st["traffic"]:
            rows += [swatch("linear-gradient(90deg,#50525c,#fad933,#eb261f)", "box") + "Aisle traffic: none - high",
                     '<span style="color:#8fa0c8">(same scale for all methods)</span>']
        return wrap_panel(CANVAS_W - 262, CANVAS_H - 300, 245, "".join(f'<div style="margin:2px 0">{r}</div>' for r in rows))

    def sec_info_html(sec):
        ss = R()["slot_sku"]
        items = sorted((sl.level, sl.basket, ss[sl.idx]) for sl in ctx.slots if sl.sec == sec)

        def with_txt(sku):
            if sku not in partner:
                return '<span style="color:#7e89a8">never bought together with another SKU</span>'
            y, c = partner[sku]
            return (f'<span style="color:#7e89a8">most often with </span>{html.escape(y[:26])}'
                    f'<span style="color:#7e89a8"> ({c} orders together, stored in {loc_text(y)})</span>')
        body = "".join(
            f'<div style="border-top:1px solid #2a3350;padding:2px 0">'
            f'<span style="display:inline-block;width:12px;height:10px;background:{FREQ_CSS[fclass[sku]]};margin-right:6px"></span>'
            f'<span style="color:#8fa0c8">L{lv}-B{bk}</span>&nbsp; {html.escape(sku)}'
            f'<span style="color:#7e89a8"> (rank {rank.index(sku) + 1} of {len(skus)}, in {freq[sku]} orders)</span>'
            f'<br><span style="font-size:11px">&nbsp;&nbsp;&nbsp;{with_txt(sku)}</span></div>' for lv, bk, sku in items)
        return wrap_panel(CANVAS_W - 412, 205, 400,
                          f'<div style="font-weight:700;margin-bottom:4px">Section S{sec_no[sec]:02d} '
                          f'{rack_of.get(sec, "")} - {METHOD_SHORT[st["method"]]}</div>{body}')

    html_cache = {}

    def setw(w, key, text):
        if html_cache.get(key) != text:        # only touch the page when the text really changed
            html_cache[key] = text
            w.text = text

    def render_T(T):
        """everything that depends on the integer time step T (stateless: works for jumping around)."""
        TL = R()["TL"]
        for p in range(P):
            stt, b = TL["st"][p][T], TL["bat"][p][T]
            pv = pickers[p]
            pv.set_state(stt)
            pv.set_carry(b is not None)
            if cur_batch[p] != b:
                setup_markers(p, b)
                cur_batch[p] = b
            bi = TL["batches"][b] if b is not None else None
            for m in markers[p]:
                m["glow"].visible = bool(bi is not None and m["sku"] is not None and bi["pick_time"][m["sku"]] > T)
            paths[p].clear()
            cal = callouts[p]
            cal.visible = False
            if bi is not None and stt in ("walk", "pick", "to_station"):
                pts = compress([TL["pos"][p][T]] + bi["fp"][TL["pi"][p][T] + 1:])
                if len(pts) >= 2:
                    for c in pts:
                        paths[p].append(wpos(c, 0.04))
            if bi is not None and stt in ("walk", "pick") and st["labels"] == "all":
                nxt = next_target(bi, T)
                if nxt:
                    f, _, _ = item_geo(nxt)
                    verb = "Picking" if stt == "pick" else "Going to"
                    txt = f"P{p + 1}\n{verb}: {nxt}\n({loc_text(nxt)})"
                    if cal.text != txt:
                        cal.text = txt
                    cal.pos = f + vector(0, hh, 0)
                    cal.visible = True
        setw(pick_w, "pick", pickers_html(T))
        if T % 4 == 0 or T >= TL["T_end"] or abs(T - ui["kpiT"]) > 4:
            ui["kpiT"] = T
            setw(kpi_w, "kpi", kpi_html(T))
        setw(orders_w, "orders", orders_html(T))

    # ---------------- 7) controls ----------------
    def back_to_overview(btn=None):
        set_view(VIEW_CENTER, VIEW_FWD, VIEW_RANGE)
        info_w.text = ""
        ui["sec_open"] = False
        setw(orders_w, "orders", orders_html(max(0, ui["T"])))

    def focus_section(sec):
        g = geoms[sec]
        fx, fz = g["face"]
        set_view(vector(g["cx"], 0.9, g["cz"]), vector(-fx, -0.25, -fz), 1.6)
        ui["sec_open"] = True
        setw(orders_w, "orders", "")
        info_w.text = sec_info_html(sec)

    def on_click(evt):
        picked_obj = scene.mouse.pick
        sec = getattr(picked_obj, "sec", None) if picked_obj is not None else None
        if sec:
            focus_section(sec)

    def on_key(evt):
        k = str(getattr(evt, "key", "")).lower()
        if k in ("0", "escape", "esc", "b"):
            back_to_overview()
        elif k == "1":
            set_view(VIEW_CENTER, vector(0, -1, 0.02), VIEW_RANGE)
        elif k == "2":
            set_view(VIEW_CENTER, vector(0, -0.12, 1), VIEW_RANGE)
        elif k == "3":
            set_view(VIEW_CENTER, vector(1, -0.15, 0), VIEW_RANGE)
        elif k == "4":
            set_view(VIEW_CENTER, vector(-1, -0.15, 0), VIEW_RANGE)

    def restart():
        ui["tf"], ui["T"], ui["kpiT"], ui["printed"] = 0.0, -1, -99, False

    def show_method():
        """switch the whole scene to the current storage method (cartons, traffic, pickers, panels, time)."""
        recolor_cartons()
        apply_tiles()
        for p in range(P):
            cur_batch[p] = "init"
            paths[p].clear()
            callouts[p].visible = False
            pickers[p].place(wpos(ctx.walk.start))
        try:
            tslider.value = 0
        except NameError:
            pass
        html_cache.clear()
        if ui["sec_open"]:
            back_to_overview()
        legend_w.text = legend_html()
        cmp_w.text = compare_html()
        restart()

    def on_method(mn):
        st["method"] = methods[mn.index]
        show_method()

    def on_color(b):
        st["color"] = COLOR_MODES[(COLOR_MODES.index(st["color"]) + 1) % len(COLOR_MODES)]
        b.text = f"Colour: {COLOR_MODE_NAME[st['color']]}"
        recolor_cartons()
        legend_w.text = legend_html()
        if ui["sec_open"]:
            back_to_overview()

    def on_traffic(b):
        st["traffic"] = not st["traffic"]
        b.text = "Traffic: ON" if st["traffic"] else "Traffic: OFF"
        apply_tiles()
        legend_w.text = legend_html()

    def apply_labels():
        """all = every label | racks = rack and station labels only | none = no labels (clean pictures)."""
        full, base = st["labels"] == "all", st["labels"] in ("all", "racks")
        for lb in sec_labels:
            lb.visible = full
        for p in range(P):
            pickers[p].tag.visible = full
            if not full:
                callouts[p].visible = False
        for lb in rack_labels + station_labels:
            lb.visible = base
        ui["T"] = -1                                # redraw the callouts of the current time step

    def on_labels(b):
        st["labels"] = LABEL_MODES[(LABEL_MODES.index(st["labels"]) + 1) % len(LABEL_MODES)]
        b.text = f"Labels: {LABEL_MODE_NAME[st['labels']]}"
        apply_labels()

    def on_capture(b):
        name = f"warehouse_{st['method']}_{st['color']}{'_traffic' if st['traffic'] else ''}"
        scene.capture(name)                         # the browser downloads name.png
        print(f"picture saved by the browser as {name}.png (Downloads folder)")

    def on_pause(b):
        ui["paused"] = not ui["paused"]
        b.text = "Resume" if ui["paused"] else "Pause"

    def on_restart(b):
        restart()

    speed_txt = wtext(text="")

    def on_speed(s):
        ui["tps"] = float(s.value)
        speed_txt.text = f" {int(s.value)} steps/s"

    def on_time(s):                                 # slider 0..1000 = share of the makespan of the method shown
        ui["tf"], ui["T"] = s.value / SLIDER_MAX * R()["TL"]["T_end"], -1

    scene.append_to_caption("\n  Storage method: ")
    menu(choices=[METHOD_NAME[m] for m in methods], index=methods.index(st["method"]), bind=on_method)
    scene.append_to_caption("   ")
    button(text=f"Colour: {COLOR_MODE_NAME[st['color']]}", bind=on_color)
    scene.append_to_caption("  ")
    button(text="Traffic: OFF", bind=on_traffic)
    scene.append_to_caption("  ")
    button(text=f"Labels: {LABEL_MODE_NAME[st['labels']]}", bind=on_labels)
    scene.append_to_caption("  ")
    button(text="Save picture", bind=on_capture)
    scene.append_to_caption("\n  ")
    button(text="Pause", bind=on_pause)
    scene.append_to_caption("  ")
    button(text="Restart", bind=on_restart)
    scene.append_to_caption("  ")
    button(text="Reset view", bind=back_to_overview)
    scene.append_to_caption("   Speed: ")
    slider(min=2, max=200, value=args.speed, step=1, length=220, bind=on_speed)
    speed_txt.text = f" {args.speed} steps/s"
    scene.append_to_caption("   Time: ")
    tslider = slider(min=0, max=SLIDER_MAX, value=0, step=1, length=300, bind=on_time)   # max cannot change later
    scene.append_to_caption(" ")
    scene.bind("keydown", on_key)
    scene.bind("click", on_click)
    show_method()
    apply_labels()

    def print_final():
        r, TL = R()["metrics"], R()["TL"]
        print("\n" + "=" * 70)
        print(f"FINISHED: {METHOD_NAME[st['method']]} | batching={args.batching} assignment={args.assign} sequence={args.seq}")
        print("=" * 70)
        labels = [("batches", "batches"), ("walk_m", "total walking distance (m)"),
                  ("walk_m_per_order", "walking distance per order (m)"), ("makespan", "makespan (steps)"),
                  ("avg_order_time", "avg order completion time (steps)"), ("avg_batch_time", "avg batch cycle time (steps)"),
                  ("station_wait", "total waiting for a station (steps)"),
                  ("avg_stops", "avg aisle stops per batch"), ("avg_skus", "avg SKUs per batch"),
                  ("picker_util", "picker utilization"), ("station_util", "station utilization")]
        for key, text in labels:
            v = r[key]
            print(f"{text:<40}: {v:,.2f}" if isinstance(v, float) else f"{text:<40}: {v:,}")
        print(f"{'  of which walk back to order start (m)':<40}: {TL['back_m']:,.2f}")
        print(f"{'check: animation distance (m)':<40}: {R()['total_walk']:,.2f}  (equals the simulation)")
        print("The window stays interactive (choose another storage method in the menu). Press Ctrl+C to exit.")

    # ---------------- 8) main loop (plays the timeline of the current method) ----------------
    last = time.perf_counter()
    while True:
        rate(60)
        TL = R()["TL"]
        T_end = TL["T_end"]
        now = time.perf_counter()
        dt = min(now - last, 0.1)
        last = now
        if not ui["paused"] and ui["tf"] < T_end:
            ui["tf"] = min(float(T_end), ui["tf"] + dt * ui["tps"])
            try:
                tslider.value = int(SLIDER_MAX * ui["tf"] / T_end)
            except Exception:
                pass
        T = min(int(ui["tf"]), T_end)
        frac = ui["tf"] - T
        if T != ui["T"]:
            ui["T"] = T
            render_T(T)
        T2 = min(T + 1, T_end)
        for p in range(P):
            a, b = wpos(TL["pos"][p][T]), wpos(TL["pos"][p][T2])
            q = a + (b - a) * frac
            if TL["st"][p][T] in ("wait", "service", "idle"):
                q = q + vector((p - (P - 1) / 2) * 0.10, 0, 0)        # keep people standing at one station apart
            pickers[p].place(q)
        if ui["tf"] >= T_end and not ui["printed"]:
            ui["printed"] = True
            print_final()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="3D visualisation of the manual picking simulation - all storage methods")
    ap.add_argument("--methods", default="random,frequency,cluster,rl_v1,rl_v2",
                    help="storage methods to compare (comma separated): random, frequency, cluster, rl_v1, rl_v2")
    ap.add_argument("--start-method", default="random", help="method shown first")
    ap.add_argument("--color", default="frequency", choices=COLOR_MODES, help="carton colour shown first")
    ap.add_argument("--labels", default="all", choices=LABEL_MODES,
                    help="labels shown first: all | racks (rack and station labels only) | none (clean pictures)")
    ap.add_argument("--assignment-file", default=None, help="CSV (sku, slot_idx): adds a storage plan from a file, e.g. RL")
    ap.add_argument("--dataset", default="archive/order_products__train.csv")
    ap.add_argument("--products", default=None, help="products.csv (Instacart format, found automatically if next to the dataset)")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--orders", type=int, default=100)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1000, help="seed of the random layout")
    ap.add_argument("--k", type=int, default=None, help="number of clusters of the data-mining plan (default: chosen on validation orders)")
    ap.add_argument("--lam", type=float, default=6.4, help="RL v2 pair weight (6.4 = value chosen by lam_validation.py)")
    ap.add_argument("--v1-episodes", type=int, default=4000, help="training episodes of RL v1")
    M.add_picking_args(ap)          # --pickers --stations --station-mode --batch-items --batch-orders --arrival-rate --window --level-extra
    ap.set_defaults(level_extra="1,0,2")   # extra pick steps of the main model of the report (level 1, 2, 3)
    ap.add_argument("--batching", choices=["fcfs", "similarity", "section", "optimized"], default="fcfs")
    ap.add_argument("--assign", choices=["dynamic", "static", "balanced"], default="dynamic")
    ap.add_argument("--seq", choices=["fcfs", "spt"], default="fcfs")
    ap.add_argument("--pick-steps", type=int, default=3)
    ap.add_argument("--station-base", type=int, default=4)
    ap.add_argument("--station-per-item", type=int, default=1)
    ap.add_argument("--speed", type=int, default=25, help="simulation steps per second")
    run_visual(ap.parse_args())