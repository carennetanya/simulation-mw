# -*- coding: utf-8 -*-
"""
wh_sim.py - Simulasi gudang robot (versi disederhanakan sesuai arahan dosen)
==========================================================================
MASALAH YANG DITELITI : STORAGE LOCATION ASSIGNMENT
    Operasi picking dibuat TETAP/SAMA. Yang dibandingkan hanya penempatan SKU:
        (1) Random   (2) Frequency (ablation)   (3) Data-mining / clustering

MODEL (sengaja sederhana):
    * Jumlah SKU = jumlah slot storage (dibuat otomatis dari dataset).
    * Layout grid 1 sel = 1 langkah. Lorong SATU ARAH (selang-seling) -> tidak ada tabrakan
      head-on; konflik sisanya diselesaikan per tick (satu sel hanya boleh satu robot).
    * Robot > station (default 6 robot, 1 station) -> antrean terbentuk alami di lane station.
    * Robot membawa sampai 4 tote per trip.
         A*  : jarak/jalur terpendek antar 2 sel (dipakai juga untuk membangun matriks jarak TSP)
         TSP : urutan kunjungan <=4 lokasi (eksak, brute force)
    * BATCHING (definisi eksplisit): ambil B order berurutan (FIFO, default B=4). Kebutuhan batch
      = gabungan SKU unik dari B order tsb. Satu tote SKU yang dibawa ke station melayani SEMUA
      order dalam batch yang butuh SKU itu. Lalu SKU unik dibagi ke trip (maks 4 tote) berdasarkan
      kedekatan lokasi. Aturan batching + routing IDENTIK untuk semua strategi storage.
    * Siklus trip: depot -> (TSP) ambil tote -> station (antre, layani) -> (TSP) kembalikan tote -> depot.

Pakai:
    python wh_sim.py --dataset archive/Groceries_dataset.csv --orders 300 --seeds 10 --splits 3
    python wh_sim.py --synthetic          # uji tanpa dataset (data sintetis)
"""
import argparse
import csv
import heapq
import itertools
import math
import os
import random
import statistics
from collections import defaultdict, deque, Counter
from dataclasses import dataclass, replace

INF = 10 ** 9


# ==========================================================================
# 1. KONFIGURASI
# ==========================================================================
@dataclass(frozen=True)
class Cfg:
    n_robots: int = 6
    n_stations: int = 1
    tote_cap: int = 4          # tote per trip
    batch_orders: int = 4      # jumlah order per batch
    levels: int = 3
    baskets: int = 2           # slot per section = levels * baskets
    blocks: int = 3            # jumlah blok rak ganda (harus ganjil -> lorong genap)
    lift_ticks: int = 2        # waktu ambil/taruh 1 tote
    service_ticks: int = 4     # waktu layani 1 tote di station
    putback: bool = True       # tote dikembalikan ke slot semula setelah dilayani
    check: bool = True         # assert tidak ada tabrakan tiap tick
    layout: str = "physical"  # physical = denah seperti simulasi1.py (R1-R15, 546 slot) | grid = blok rak generik


# ==========================================================================
# 2. LAYOUT (lorong satu arah) + A* + TSP
# ==========================================================================
@dataclass
class Slot:
    idx: int
    sec: tuple       # sel rak (x, z)
    access: tuple    # sel lorong tempat robot berhenti
    level: int
    basket: int


class Layout:
    """
    z=0..Q      : area station (lane antre turun di x_s, lane keluar naik di x_s+1, depot = stub di z=Q)
    z=Q+1 (zB)  : lintas depan, arah KANAN      z=Q+2 (zA): lintas depan, arah KIRI
    z=zA+1..    : rak bagian bawah (L1 baris), lalu LINTAS TENGAH 2 baris (kanan & kiri) sebagai jalan
                  pintas, lalu rak bagian atas (L2 baris)     z=zt : lintas atas, arah KANAN
    Lorong vertikal di x=0,3,6,..: indeks genap = NAIK, ganjil = TURUN. Rak di x=3b+1 dan 3b+2.
    """

    def __init__(self, cfg, n_skus):
        if cfg.blocks % 2 == 0:
            raise ValueError("cfg.blocks harus ganjil (jumlah lorong genap, arah selang-seling).")
        self.cfg = cfg
        S, R = cfg.n_stations, cfg.n_robots
        per_sec = cfg.levels * cfg.baskets
        self.n_sections = math.ceil(n_skus / per_sec)
        cols = 2 * cfg.blocks
        self.L = math.ceil(self.n_sections / cols)
        self.L1 = (self.L + 1) // 2 if self.L >= 2 else self.L
        self.L2 = self.L - self.L1
        self.W = 3 * cfg.blocks + 1
        self.Q = max(2, math.ceil(R / S))
        Q = self.Q
        self.zB, self.zA = Q + 1, Q + 2
        if self.L2 > 0:                                  # lintas tengah: zmR (kanan), zmL (kiri)
            self.zmR = self.zA + self.L1 + 1
            self.zmL = self.zmR + 1
            self.zt = self.zmL + self.L2 + 1
            self.rack_rows = (list(range(self.zA + 1, self.zA + self.L1 + 1))
                              + list(range(self.zmL + 1, self.zmL + self.L2 + 1)))
        else:
            self.zmR = self.zmL = None
            self.zt = self.zA + self.L1 + 1
            self.rack_rows = list(range(self.zA + 1, self.zA + self.L1 + 1))
        self.H = self.zt + 1
        W, zA, zB, zt = self.W, self.zA, self.zB, self.zt
        self.aisle_x = [3 * b for b in range(cfg.blocks + 1)]
        self.edges = defaultdict(list)

        def add(a, b):
            self.edges[a].append(b)

        for x in range(1, W):
            add((x, zA), (x - 1, zA))                    # lintas depan atas -> kiri
        for x in range(0, W - 1):
            add((x, zB), (x + 1, zB))                    # lintas depan bawah -> kanan
        add((0, zA), (0, zB))                            # putar balik kiri
        add((W - 1, zB), (W - 1, zA))                    # putar balik kanan
        for x in range(0, W - 1):
            add((x, zt), (x + 1, zt))                    # lintas atas -> kanan
        if self.L2 > 0:
            for x in range(1, W):
                add((x, self.zmL), (x - 1, self.zmL))    # lintas tengah -> kiri
            for x in range(0, W - 1):
                add((x, self.zmR), (x + 1, self.zmR))    # lintas tengah -> kanan
        for b, x in enumerate(self.aisle_x):
            if b % 2 == 0:                               # lorong NAIK
                for z in range(zA, zt):
                    add((x, z), (x, z + 1))
                if x > 0:
                    add((x, zB), (x, zA))
            else:                                        # lorong TURUN
                for z in range(zt, zA, -1):
                    add((x, z), (x, z - 1))

        # station
        spacing = (W - 1) // S
        if spacing < 3:
            raise ValueError("Gudang terlalu sempit untuk jumlah station; naikkan cfg.blocks.")
        self.station_x = [1 + i * spacing for i in range(S)]
        self.svc, self.exit, self.queue_cells, self.lane_cells = [], [], set(), []
        used_cols = set()
        for xs in self.station_x:
            used_cols |= {xs, xs + 1}
            self.svc.append((xs, 0))
            self.exit.append((xs + 1, 0))
            add((xs, zB), (xs, Q))
            for z in range(Q, 0, -1):
                add((xs, z), (xs, z - 1))
                self.queue_cells.add((xs, z))
            add((xs, 0), (xs + 1, 0))
            for z in range(0, Q):
                add((xs + 1, z), (xs + 1, z + 1))
            add((xs + 1, Q), (xs + 1, zB))
        # depot = stub 1 sel di bawah lintas zB
        free_cols = [x for x in range(W) if x not in used_cols]
        if len(free_cols) < R:
            raise ValueError(f"Hanya {len(free_cols)} depot muat; butuh {R}. Naikkan cfg.blocks (3 -> 5).")
        self.depots = [(x, Q) for x in free_cols[:R]]
        for d in self.depots:
            add((d[0], zB), d)
            add(d, (d[0], zB))

        self.cells = set(self.edges) | {n for v in self.edges.values() for n in v}
        self._path = {}
        self._check_connected()

        # section (sel rak) + sel akses
        secs = []
        for b in range(cfg.blocks):
            for col, ax in ((3 * b + 1, 3 * b), (3 * b + 2, 3 * b + 3)):
                for z in self.rack_rows:
                    secs.append(dict(sec=(col, z), access=(ax, z)))
        for s in secs:   # urut: dekat station = awal (jarak pulang-pergi rata-rata station)
            s["key"] = sum(self.dist(sv, s["access"]) + self.dist(s["access"], sv) for sv in self.svc) / S
        secs.sort(key=lambda s: (s["key"], s["sec"]))
        self.rack_cells = {s["sec"] for s in secs}
        self.sections = secs[:self.n_sections]                     # dibangun & dipakai
        self.unbuilt = {s["sec"] for s in secs[self.n_sections:]}  # sisa sel rak -> kosong
        self.unbuilt_secs = secs[self.n_sections:]
        self.slots = []
        for s in self.sections:
            for lv in range(1, cfg.levels + 1):
                for bk in range(1, cfg.baskets + 1):
                    if len(self.slots) < n_skus:
                        self.slots.append(Slot(len(self.slots), s["sec"], s["access"], lv, bk))
        assert len(self.slots) == n_skus, "Jumlah slot harus = jumlah SKU"

    # ---------- geometri (meter) untuk visual & KPI jarak ----------
    CELL_M = 0.7

    def pt(self, c):
        """sel grid -> posisi dunia (meter)."""
        return (c[0] * self.CELL_M, c[1] * self.CELL_M)

    def cell_dim(self):
        return (self.CELL_M, self.CELL_M)

    def step_m(self, a, b):
        """panjang satu langkah a->b (meter)."""
        return self.CELL_M

    def geom(self, s):
        """section -> dict(cx, cz, sx, sz, face) dalam meter; face = arah sisi akses."""
        w = self.CELL_M * 0.92
        cx, cz = self.pt(s["sec"])
        fx = -1 if s["access"][0] < s["sec"][0] else 1
        return dict(cx=cx, cz=cz, sx=w, sz=w, face=(fx, 0))

    # ---------- A* ----------
    def path(self, s, g):
        key = (s, g)
        p = self._path.get(key)
        if p is not None:
            return p
        if s == g:
            p = [s]
        else:
            h = lambda c: abs(c[0] - g[0]) + abs(c[1] - g[1])
            openq, cnt, best, came = [(h(s), 0, s)], 0, {s: 0}, {}
            p = None
            while openq:
                f, _, c = heapq.heappop(openq)
                if c == g:
                    p = [c]
                    while c in came:
                        c = came[c]
                        p.append(c)
                    p.reverse()
                    break
                if f - h(c) > best[c]:
                    continue
                for n in self.edges[c]:
                    ng = best[c] + 1
                    if ng < best.get(n, INF):
                        best[n], came[n] = ng, c
                        cnt += 1
                        heapq.heappush(openq, (ng + h(n), cnt, n))
            if p is None:
                raise RuntimeError(f"Tidak ada jalur {s}->{g}")
        self._path[key] = p
        return p

    def dist(self, a, b):
        return len(self.path(a, b)) - 1

    def sym(self, a, b):
        return self.dist(a, b) + self.dist(b, a)

    # ---------- TSP eksak untuk <=7 titik (A* memberi matriks jarak) ----------
    def tsp(self, start, stops, end):
        stops = list(stops)
        if len(stops) <= 1:
            return stops, self._tour_cost(start, stops, end)
        if len(stops) <= 7:
            best = min(itertools.permutations(stops), key=lambda p: self._tour_cost(start, p, end))
            return list(best), self._tour_cost(start, best, end)
        order, rem, cur = [], stops[:], start         # fallback NN + 2-opt
        while rem:
            nx = min(rem, key=lambda c: self.dist(cur, c))
            order.append(nx); rem.remove(nx); cur = nx
        best = self._tour_cost(start, order, end)
        improved = True
        while improved:
            improved = False
            for i in range(len(order) - 1):
                for j in range(i + 1, len(order)):
                    cand = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                    c = self._tour_cost(start, cand, end)
                    if c < best:
                        order, best, improved = cand, c, True
        return order, best

    def _tour_cost(self, start, seq, end):
        cur, tot = start, 0
        for c in seq:
            tot += self.dist(cur, c)
            cur = c
        return tot + self.dist(cur, end)

    def _check_connected(self):
        start = next(iter(self.cells))
        rev = defaultdict(list)
        for a, ns in self.edges.items():
            for n in ns:
                rev[n].append(a)
        for graph in (self.edges, rev):
            seen, q = {start}, deque([start])
            while q:
                c = q.popleft()
                for n in graph[c]:
                    if n not in seen:
                        seen.add(n); q.append(n)
            if seen != self.cells:
                raise RuntimeError("Graf lorong tidak strongly-connected.")


class PhysicalLayout(Layout):
    """
    Denah FISIK (ruang 7.2 m x ~7.6 m), LIMA BELAS rak R1..R15, nomor dari station ke atas, kiri ke kanan:
        R1/R2   baris paling bawah, diambil dari lorong zA                       (ungu)
        R3/R4   di atas R1/R2 (punggung-ke-punggung), diambil dari lorong zA+5   (oranye)
        R5/R6   di atas lorong zA+5, diambil dari lorong zA+5                    (ungu)  = konfigurasi sama dgn R1/R3
        R7/R8   di atas R5/R6 (punggung-ke-punggung), diambil dari lorong zA+10  (peach)
        R9/R10  di atas lorong zA+10, diambil dari lorong zA+10                  (hijau)
        R11/R12 di atas R9/R10 (punggung-ke-punggung), diambil dari lorong atas zA+15 (biru muda)
        R13     paling atas (15 section), diambil dari lorong atas zA+15         (biru)
        R14/R15 rak VERTIKAL di sisi kiri / kanan (8 section), diambil dari lorong c2 / c14 (ungu)
    Jadi blok tengah (R5-R8) menghadap DUA lorong (zA+5 dan zA+10) dan tiap lorong dipakai dua rak (satu di tiap sisinya).
    GEOMETRI SECTION: 0.38 m (searah lorong) x 0.50 m (kedalaman). Satu section = 3 level x 2 slot = 6 slot, SKU = slot.
    Jumlah section: R1..R10 50 + R11/R12 10 + R13 15 + R14/R15 16 = 91 section = 546 slot; SEMUA section terisi.

    Grid topologis (17 kolom): c0 lorong luar kiri | c1 rak R14 | c2 lorong | c3-7 blok kiri | c8 lorong tengah
    | c9-13 blok kanan | c14 lorong | c15 rak R15 | c16 lorong luar kanan.   1 sel = 1 tick.
    Lorong SATU ARAH: baris akses R1/R2 (zA) ke KIRI; baris akses zA+5 & zA+10 ke KANAN;
    baris akses atas (R11-R13) ke KANAN; kolom c0/c2/c8 NAIK; kolom c14/c16 TURUN.
    """
    # Catatan: saklar SHARED_AISLE lama TIDAK didukung lagi. Lorong zA+5 sekarang melayani R4/R5 (di atas) DAN R1/R8 (di bawah),
    # jadi sudah satu lorong untuk dua sisi; menghapusnya akan memutus akses R1/R8.
    SHARED_AISLE = False
    NC = 17
    PITCH = 0.38          # panjang 1 section SEARAH LORONG (m) - sama untuk rak horizontal dan vertikal
    RACK_D = 0.50         # kedalaman rak (m) - sama untuk rak horizontal dan vertikal
    ROW = 0.25            # jarak antar baris grid (m); rak horizontal tebal 0.50 m = 2 baris

    def __init__(self, cfg, n_skus):
        self.cfg = cfg
        S, R = cfg.n_stations, cfg.n_robots
        if S not in (1, 2, 3):
            raise ValueError("layout physical mendukung 1-3 station.")
        per_sec = cfg.levels * cfg.baskets
        self.n_sections = math.ceil(n_skus / per_sec)
        self.W = self.NC
        self.Q = max(2, math.ceil(R / S))
        Q = self.Q
        self.zB, self.zA = Q + 1, Q + 2
        zA, W = self.zA, self.W
        self.zt = zA + 15                      # baris akses atas (R11-R13)
        self.H = zA + 18
        top = self.zt
        self.edges = defaultdict(list)

        def add(a, b):
            self.edges[a].append(b)

        for x in range(W - 1):
            add((x, self.zB), (x + 1, self.zB))                # lintas bawah ->
        for x in range(1, W):
            add((x, zA), (x - 1, zA))                          # baris akses R6/R7 <-
        add((0, zA), (0, self.zB))
        add((W - 1, self.zB), (W - 1, zA))
        if os.environ.get("SHARED_AISLE", "0") == "1":
            raise ValueError("SHARED_AISLE is not supported with the current layout (R1/R8 are stacked above R6/R7 and use aisle zA+5).")
        self.rows_right = [zA + 5, zA + 10]                     # lorong zA+5 (R3/R4 + R5/R6) dan zA+10 (R7/R8 + R9/R10)
        for r in self.rows_right:
            for x in range(2, 14):
                add((x, r), (x + 1, r))
        for x in range(W - 1):
            add((x, top), (x + 1, top))                        # akses atas ->
        for r in range(zA, top):
            add((0, r), (0, r + 1))                            # c0 naik
            add((2, r), (2, r + 1))                            # c2 naik
            add((8, r), (8, r + 1))                            # c8 naik (lorong tengah)
        for r in range(top, zA, -1):
            add((14, r), (14, r - 1))                          # c14 turun
            add((16, r), (16, r - 1))                          # c16 turun
        add((8, self.zB), (8, zA))                             # c8 mulai dari lintas bawah

        # ---- station (antrean turun di xs, keluar naik di xs+1) ----
        st_cols = {1: [8], 2: [5, 11], 3: [3, 8, 12]}[S]
        self.station_x = st_cols
        self.svc, self.exit, self.queue_cells = [], [], set()
        used = set()
        for xs in st_cols:
            used |= {xs, xs + 1}
            self.svc.append((xs, 0))
            self.exit.append((xs + 1, 0))
            add((xs, self.zB), (xs, Q))
            for z in range(Q, 0, -1):
                add((xs, z), (xs, z - 1))
                self.queue_cells.add((xs, z))
            add((xs, 0), (xs + 1, 0))
            for z in range(0, Q):
                add((xs + 1, z), (xs + 1, z + 1))
            add((xs + 1, Q), (xs + 1, self.zB))
        free_cols = sorted((x for x in range(W) if x not in used), key=lambda x: (abs(x - 8.5), x))
        if len(free_cols) < R:
            raise ValueError(f"Hanya {len(free_cols)} depot muat; butuh {R} robot.")
        self.depots = [(x, Q) for x in free_cols[:R]]
        for d in self.depots:
            add((d[0], self.zB), d)
            add(d, (d[0], self.zB))

        self.cells = set(self.edges) | {n for v in self.edges.values() for n in v}
        self._path = {}
        self._check_connected()

        # ---- kandidat section: tepat 91 = 546 slot, semua dipakai ----
        cand = []

        def hrack(rid, cols, row0, arow):
            for k, col in enumerate(cols):
                cand.append(dict(rack=rid, k=k, sec=(col, row0), access=(col, arow), kind="H", row0=row0))

        def vrack(rid, col, acol, n=8):
            """rak vertikal 8 section (0.38 m searah lorong) di kolom c1 / c15, diambil dari lorong acol (c2 / c14)."""
            xc = (self.xm(col - 1) + self.xm(col + 1)) / 2
            zc = self.zm(zA + 7.5)
            for k in range(n):
                z = zc - n * self.PITCH / 2 + self.PITCH / 2 + k * self.PITCH
                r = zA + round((z - self.zm(zA)) / self.ROW)
                cand.append(dict(rack=rid, k=k, sec=(col, r), access=(acol, r), kind="V", row0=r, cx=xc, cz=z,
                                 face=(1 if acol > col else -1, 0)))

        a5, a10 = zA + 5, zA + 10                              # lorong tengah bawah / atas
        hrack("R1", range(3, 8), zA + 1, zA)
        hrack("R2", range(9, 14), zA + 1, zA)
        hrack("R3", range(3, 8), zA + 3, a5)
        hrack("R4", range(9, 14), zA + 3, a5)
        hrack("R5", range(3, 8), zA + 6, a5)                   # sama konfigurasi dgn R1/R3: punggung-ke-punggung dgn R7/R8
        hrack("R6", range(9, 14), zA + 6, a5)
        hrack("R7", range(3, 8), zA + 8, a10)
        hrack("R8", range(9, 14), zA + 8, a10)
        hrack("R9", range(3, 8), zA + 11, a10)
        hrack("R10", range(9, 14), zA + 11, a10)
        hrack("R11", range(3, 8), zA + 13, top)
        hrack("R12", range(9, 14), zA + 13, top)
        hrack("R13", range(1, 16), zA + 16, top)
        vrack("R14", 1, 2)
        vrack("R15", 15, 14)
        for c in cand:
            c["key"] = sum(self.dist(sv, c["access"]) + self.dist(c["access"], sv) for sv in self.svc) / S
        total = len(cand)
        if self.n_sections > total:
            raise ValueError(f"layout physical hanya punya {total} section ({total * per_sec} slot); "
                             f"SKU={n_skus} terlalu banyak. Pakai --layout grid.")
        # alokasi proporsional per rak (largest remainder), tiap rak dipakai dari sisi terdekat station
        byrack = defaultdict(list)
        for c in cand:
            byrack[c["rack"]].append(c)
        quota = {r: len(v) * self.n_sections / total for r, v in byrack.items()}
        cnt = {r: int(q) for r, q in quota.items()}
        for r in sorted(quota, key=lambda r: (-(quota[r] - cnt[r]), r)):
            if sum(cnt.values()) >= self.n_sections:
                break
            if cnt[r] < len(byrack[r]):
                cnt[r] += 1
        built, unbuilt = [], []
        for r, v in byrack.items():
            v = sorted(v, key=lambda c: (c["key"], c["sec"]))
            built += v[:cnt[r]]
            unbuilt += v[cnt[r]:]
        built.sort(key=lambda c: (c["key"], c["sec"]))
        self.sections, self.unbuilt_secs = built[:self.n_sections], unbuilt
        self.rack_cells = {c["sec"] for c in cand}
        self.unbuilt = {c["sec"] for c in unbuilt}
        self.slots = []
        for sc in self.sections:
            for lv in range(1, cfg.levels + 1):
                for bk in range(1, cfg.baskets + 1):
                    if len(self.slots) < n_skus:
                        self.slots.append(Slot(len(self.slots), sc["sec"], sc["access"], lv, bk))
        assert len(self.slots) == n_skus, "Jumlah slot harus = jumlah SKU"
        self.rack_ids = [f"R{i}" for i in range(1, 16)]
        self.rack_count = {r: sum(1 for c in self.sections if c["rack"] == r) for r in self.rack_ids}

    # ---------- koordinat meter ----------
    def xm(self, c):
        if c == 0:
            return 0.30
        if c == self.NC - 1:
            return 6.65
        return 0.65 + self.PITCH * (c - 0.5)

    def zm(self, r):
        return 1.975 - self.ROW * (self.zA - r)

    def pt(self, c):
        return (self.xm(c[0]), self.zm(c[1]))

    def cell_dim(self):
        return (self.PITCH, self.ROW)

    def step_m(self, a, b):
        pa, pb = self.pt(a), self.pt(b)
        return math.hypot(pa[0] - pb[0], pa[1] - pb[1])

    def geom(self, s):
        """R1..R13: horizontal, 0.38 m along the aisle x 0.50 m deep (two grid rows); R14/R15: vertical, 0.50 m (x) x 0.38 m (z)."""
        if s.get("kind") == "V":
            return dict(cx=s["cx"], cz=s["cz"], sx=self.RACK_D, sz=self.PITCH, face=s["face"])
        cx = self.xm(s["sec"][0])
        cz = (self.zm(s["row0"]) + self.zm(s["row0"] + 1)) / 2
        return dict(cx=cx, cz=cz, sx=self.PITCH, sz=self.RACK_D, face=(0, 1 if s["access"][1] > s["row0"] else -1))


def make_layout(cfg, n_skus):
    if cfg.layout == "physical":
        return PhysicalLayout(cfg, n_skus)
    if cfg.layout == "grid":
        return Layout(cfg, n_skus)
    raise ValueError(cfg.layout)


# ==========================================================================
# 3. STRATEGI STORAGE (satu-satunya variabel yang dibandingkan)
# ==========================================================================
def assign_random(skus, slots, seed):
    s = list(skus)
    random.Random(seed).shuffle(s)
    return {sku: slots[i] for i, sku in enumerate(s)}


def sku_frequency(train_orders):
    f = Counter()
    for o in train_orders:
        f.update(set(o))
    return f


def assign_frequency(skus, slots, train_orders):
    f = sku_frequency(train_orders)
    s = sorted(skus, key=lambda k: (-f[k], k))
    return {sku: slots[i] for i, sku in enumerate(s)}


def cluster_skus(skus, train_orders, cap, min_sim=0.0):
    """Hierarchical agglomerative clustering (average linkage) pada kemiripan Jaccard
    co-purchase, dengan batas ukuran cluster = cap (= slot per section)."""
    f = sku_frequency(train_orders)
    co = Counter()
    for o in train_orders:
        for a, b in itertools.combinations(sorted(set(o)), 2):
            co[(a, b)] += 1
    idx = {s: i for i, s in enumerate(sorted(skus))}
    sums = defaultdict(dict)                     # sums[a][b] = jumlah kemiripan antar anggota
    for (a, b), c in co.items():
        if a in idx and b in idx:
            sim = c / (f[a] + f[b] - c)
            i, j = idx[a], idx[b]
            sums[i][j] = sim
            sums[j][i] = sim
    members = {i: [s] for s, i in idx.items()}
    size = {i: 1 for i in members}
    nxt = len(members)
    while True:
        best = None
        for a, row in sums.items():
            for b, s in row.items():
                if a < b and size[a] + size[b] <= cap:
                    avg = s / (size[a] * size[b])
                    if avg > min_sim and (best is None or avg > best[0]):
                        best = (avg, a, b)
        if best is None:
            break
        _, a, b = best
        c = nxt; nxt += 1
        members[c] = members.pop(a) + members.pop(b)
        size[c] = size.pop(a) + size.pop(b)
        row = {}
        for x in (a, b):
            for k, v in sums.pop(x, {}).items():
                if k in (a, b):
                    continue
                row[k] = row.get(k, 0.0) + v
        for k, v in row.items():
            sums[k].pop(a, None); sums[k].pop(b, None)
            sums[k][c] = v
        sums[c] = row
    clusters = list(members.values())
    return clusters, f


def assign_cluster(skus, slots, train_orders, cap, min_sim=0.0):
    """Cluster co-purchase -> dikemas PER SECTION (bin packing first-fit).
    Cluster terpopuler diletakkan di section terdekat station; satu cluster tidak dipecah ke
    dua section kecuali terpaksa (tidak ada section dengan sisa ruang cukup)."""
    clusters, f = cluster_skus(skus, train_orders, cap, min_sim)
    clusters.sort(key=lambda c: (-sum(f[s] for s in c) / len(c), -len(c), c[0]))
    bins, order = {}, []                                   # section -> slot list (urut dekat station)
    for sl in slots:
        if sl.sec not in bins:
            bins[sl.sec] = []
            order.append(sl.sec)
        bins[sl.sec].append(sl)
    free = {sec: list(bins[sec]) for sec in order}
    assignment = {}
    for c in clusters:
        members = sorted(c, key=lambda s: (-f[s], s))
        target = next((sec for sec in order if len(free[sec]) >= len(members)), None)
        if target is not None:
            for sku in members:
                assignment[sku] = free[target].pop(0)
        else:                                              # fragmentasi: isi sisa ruang terdekat
            for sku in members:
                sec = next(x for x in order if free[x])
                assignment[sku] = free[sec].pop(0)
    return assignment, clusters


def cluster_skus_k(skus, train_orders, ks):
    """Hierarchical agglomerative clustering (average linkage) on the Jaccard co-purchase similarity of the SKUs.
    The number of clusters k is the INPUT (like k-means); cluster sizes are NOT limited and may differ.
    Merging goes on until k clusters are left; the dendrogram is cut at every k in `ks` in ONE pass.
    SKUs that were never bought together have similarity 0; when no similar pair is left the two smallest
    clusters are merged (so exactly k clusters are always obtained).   -> ({k: clusters}, frequency)"""
    ks = sorted(set(ks), reverse=True)
    f = sku_frequency(train_orders)
    co = Counter()
    for o in train_orders:
        for a, b in itertools.combinations(sorted(set(o)), 2):
            co[(a, b)] += 1
    idx = {s: i for i, s in enumerate(sorted(skus))}
    sums = defaultdict(dict)                     # sums[a][b] = sum of the SKU-pair similarities between clusters a and b
    for (a, b), c in co.items():
        if a in idx and b in idx:
            sim = c / (f[a] + f[b] - c)
            i, j = idx[a], idx[b]
            sums[i][j] = sim
            sums[j][i] = sim
    members = {i: [s] for s, i in idx.items()}
    size = {i: 1 for i in members}
    nxt = len(members)
    out = {}
    for k in ks:
        while len(members) > k:
            best = None
            for a, row in sums.items():
                for b, sv in row.items():
                    if a < b:
                        avg = sv / (size[a] * size[b])
                        if best is None or avg > best[0]:
                            best = (avg, a, b)
            if best is None:                     # no similar pair left: merge the two smallest clusters
                a, b = sorted(members, key=lambda c: (size[c], c))[:2]
            else:
                _, a, b = best
            c = nxt
            nxt += 1
            members[c] = members.pop(a) + members.pop(b)
            size[c] = size.pop(a) + size.pop(b)
            row = {}
            for x in (a, b):
                for kk, v in sums.pop(x, {}).items():
                    if kk in (a, b):
                        continue
                    row[kk] = row.get(kk, 0.0) + v
            for kk, v in row.items():
                sums[kk].pop(a, None)
                sums[kk].pop(b, None)
                sums[kk][c] = v
            sums[c] = row
        out[k] = [list(m) for m in members.values()]
    return out, f


def pack_clusters(clusters, f, slots):
    """Placement of clusters / SKUs (documented rule, used by every cluster plan):
      1. clusters are sorted by mean SKU frequency, most popular first;
      2. the members of a cluster are sorted by frequency, highest first;
      3. a cluster goes into the NEAREST section (sections are ordered by walking distance to the nearest station)
         that still has room for ALL its members (first-fit); the members fill the free slots of that section in order
         (level 1 / basket 1 first - level and basket do not influence the walking distance in this model);
      4. a cluster that fits in no single section (larger than a section, or only fragments are left) is split:
         its members fill the free slots of the nearest sections one after the other."""
    clusters = sorted(clusters, key=lambda c: (-sum(f[s] for s in c) / len(c), -len(c), c[0]))
    bins, order = {}, []
    for sl in slots:
        if sl.sec not in bins:
            bins[sl.sec] = []
            order.append(sl.sec)
        bins[sl.sec].append(sl)
    free = {sec: list(bins[sec]) for sec in order}
    assignment = {}
    for c in clusters:
        members = sorted(c, key=lambda s: (-f[s], s))
        target = next((sec for sec in order if len(free[sec]) >= len(members)), None)
        if target is not None:
            for sku in members:
                assignment[sku] = free[target].pop(0)
        else:
            for sku in members:
                sec = next(x for x in order if free[x])
                assignment[sku] = free[sec].pop(0)
    return assignment


def assign_cluster_k(skus, slots, train_orders, k, _cache=None):
    """Cluster plan with k clusters (data mining). -> (assignment, clusters)"""
    ck, f = _cache if _cache is not None else cluster_skus_k(skus, train_orders, [k])
    clusters = ck[k]
    return pack_clusters(clusters, f, slots), clusters


# ==========================================================================
# 4. ROBOT + SIMULATOR (operasi picking tetap)
# ==========================================================================
class Robot:
    def __init__(self, rid, depot):
        self.id, self.depot = rid, depot
        self.pos = self.prev = depot
        self.legs, self.leg, self.phase = deque(), None, 0
        self.path, self.dwell = deque(), 0
        self.state, self.trip, self.carry = "idle", None, 0
        self.dist = self.blocked = self.qwait = self.busy = 0
        self.meters = 0.0
        self.streak = 0


class Trip:
    _n = itertools.count(1)

    def __init__(self, batch, skus):
        self.id, self.batch, self.skus = next(Trip._n), batch, list(skus)
        self.station = None
        self.t_start = self.t_end = None


class Sim:
    def __init__(self, cfg, layout, assignment, orders):
        Trip._n = itertools.count(1)
        self.cfg, self.lay, self.assign = cfg, layout, assignment
        self.cell = {sku: slot.access for sku, slot in assignment.items()}
        self.orders = [list(dict.fromkeys(o)) for o in orders]
        for o in self.orders:
            for s in o:
                if s not in assignment:
                    raise KeyError(f"SKU {s} tidak punya slot")
        self.order_done = [None] * len(self.orders)
        self.order_need = [set(o) for o in self.orders]
        self.ptr, self.pending, self.batches = 0, deque(), []
        self.robots = [Robot(i, layout.depots[i]) for i in range(cfg.n_robots)]
        self.T = 0
        self.events = []
        self.assigned = [0] * cfg.n_stations
        self.station_busy = [0] * cfg.n_stations
        self.trips = []
        self.tote_visits = self.tote_demand = 0
        self.heat = Counter()

    # ----- batching -----
    def _form_batch(self):
        if self.ptr >= len(self.orders):
            return
        ids = list(range(self.ptr, min(self.ptr + self.cfg.batch_orders, len(self.orders))))
        self.ptr = ids[-1] + 1
        need = sorted({s for i in ids for s in self.orders[i]})
        self.tote_demand += sum(len(self.orders[i]) for i in ids)
        self.tote_visits += len(need)
        batch = dict(id=len(self.batches) + 1, orders=ids, skus=need)
        self.batches.append(batch)
        for grp in self._split(need):
            self.pending.append(Trip(batch, grp))

    def _split(self, skus):
        """SKU unik -> trip (<= tote_cap): seed = SKU terjauh dari station, isi dgn yg terdekat."""
        L, cap = self.lay, self.cfg.tote_cap
        rem, trips = sorted(skus), []
        ref = L.svc[0]
        while rem:
            seed = max(rem, key=lambda s: L.sym(ref, self.cell[s]))
            grp = [seed]; rem.remove(seed)
            while rem and len(grp) < cap:
                nx = min(rem, key=lambda s: min(L.sym(self.cell[g], self.cell[s]) for g in grp))
                grp.append(nx); rem.remove(nx)
            trips.append(grp)
        return trips

    # ----- rencana trip: TSP + A* -----
    def _plan(self, r, trip):
        L, cfg = self.lay, self.cfg
        st = min(range(cfg.n_stations), key=lambda s: (self.assigned[s], s))
        self.assigned[st] += 1
        trip.station, trip.t_start, r.trip = st, self.T, trip
        svc = L.svc[st]
        groups = defaultdict(list)
        for s in trip.skus:
            groups[self.cell[s]].append(s)
        order, _ = L.tsp(r.pos, list(groups), svc)
        r.legs = deque()
        for c in order:
            r.legs.append(("pick", c, cfg.lift_ticks * len(groups[c]), groups[c]))
        r.legs.append(("station", svc, cfg.service_ticks * len(trip.skus), trip.skus))
        if cfg.putback:
            back, _ = L.tsp(svc, list(groups), r.depot)
            for c in back:
                r.legs.append(("put", c, cfg.lift_ticks * len(groups[c]), groups[c]))
        r.legs.append(("park", r.depot, 0, None))
        r.leg, r.phase, r.state = None, 0, "move"
        self.trips.append(trip)
        self.events.append(("assign", r, trip))

    def _leg_done(self, r):
        kind, goal, _, data = r.leg
        if kind == "pick":
            r.carry += len(data)
            self.events.append(("pick", r, goal))
        elif kind == "put":
            r.carry -= len(data)
        elif kind == "station":
            self.assigned[r.trip.station] -= 1
            for i in r.trip.batch["orders"]:
                if self.order_done[i] is None:
                    self.order_need[i] -= set(data)
                    if not self.order_need[i]:
                        self.order_done[i] = self.T
                        self.events.append(("order_done", r, i))
            self.events.append(("served", r, r.trip))
        elif kind == "park":
            r.state, r.carry = "idle", 0
            r.trip.t_end = self.T
            self.events.append(("trip_done", r, r.trip))
            r.trip = None

    def _progress(self, r):
        while True:
            if r.leg is None:
                if not r.legs:
                    return
                r.leg, r.phase = r.legs.popleft(), 0
            kind, goal, dw, _ = r.leg
            if r.phase == 0:
                if r.pos != goal:
                    r.path = deque(self.lay.path(r.pos, goal)[1:])
                    r.phase = 1
                    return
                r.phase, r.dwell = 2, dw
            elif r.phase == 1:
                if r.path:
                    return
                r.phase, r.dwell = 2, dw
            if r.dwell > 0:
                r.dwell -= 1
                r.state = "service" if kind == "station" else "pick"
                if kind == "station":
                    self.station_busy[r.trip.station] += 1
                if r.dwell > 0:
                    return
            self._leg_done(r)
            r.leg = None

    # ----- penyelesaian konflik gerak (1 sel = 1 robot, tanpa swap) -----
    def _resolve(self, movers):
        occ = {r.pos: r for r in self.robots}
        want = {r.id: r.path[0] for r in movers}
        order = sorted(movers, key=lambda r: (-r.streak, r.id))
        ok, claimed = {}, set()
        changed = True
        while changed:
            changed = False
            for r in order:
                if r.id in ok:
                    continue
                c = want[r.id]
                if c in claimed:
                    continue
                o = occ.get(c)
                if o is None or (o.id in ok and ok[o.id] != r.pos):
                    ok[r.id] = c; claimed.add(c); changed = True
        for r in order:                       # rotasi siklik (A->B->C->A) boleh jalan serentak
            if r.id in ok:
                continue
            chain, cur, closed = [r], r, False
            while True:
                o = occ.get(want[cur.id])
                if o is None or o.id not in want or o.id in ok:
                    break
                if o is r:
                    closed = True
                    break
                if o in chain:
                    break
                chain.append(o); cur = o
            if closed and not any(want[x.id] in claimed for x in chain):
                for x in chain:
                    ok[x.id] = want[x.id]; claimed.add(want[x.id])
        return ok, want

    def tick(self):
        self.events = []
        for r in self.robots:                                   # dispatch
            if r.state == "idle" and r.leg is None and not r.legs:
                if not self.pending:
                    self._form_batch()
                if self.pending:
                    self._plan(r, self.pending.popleft())
        movers = []
        for r in self.robots:
            r.prev = r.pos
            self._progress(r)
            if r.path:
                movers.append(r)
        ok, want = self._resolve(movers) if movers else ({}, {})
        qset = self.lay.queue_cells | set(self.lay.svc)
        for r in movers:
            if r.id in ok:
                r.pos = ok[r.id]
                r.path.popleft()
                r.dist += 1
                r.meters += self.lay.step_m(r.prev, r.pos)
                r.streak = 0
                r.state = "move"
                self.heat[r.pos] += 1
            else:
                r.blocked += 1
                r.streak += 1
                r.state = "wait"
                if want[r.id] in qset:
                    r.qwait += 1
        for r in self.robots:
            if r.state != "idle":
                r.busy += 1
        if self.cfg.check:
            seen = {}
            for r in self.robots:
                assert r.pos not in seen, f"TABRAKAN di {r.pos} tick {self.T}"
                seen[r.pos] = r
            for r in self.robots:
                o = seen.get(r.prev)
                assert not (o is not None and o is not r and o.prev == r.pos and r.pos != r.prev), \
                    f"SWAP tick {self.T}"
        self.T += 1

    def finished(self):
        return (self.ptr >= len(self.orders) and not self.pending
                and all(r.state == "idle" and not r.legs and r.leg is None for r in self.robots))

    def run(self, max_ticks=3_000_000):
        while not self.finished():
            self.tick()
            if self.T > max_ticks:
                raise RuntimeError("Simulasi tidak selesai (deadlock?)")
        return self.kpi()

    def kpi(self):
        n = len(self.orders)
        done = [t for t in self.order_done if t is not None]
        makespan = max(done) + 1 if done else 0
        horizon = max(1, self.T)
        total = sum(r.dist for r in self.robots)
        trips = [t for t in self.trips if t.t_end is not None]
        return dict(
            orders=n, trips=len(self.trips),
            total_distance=total, dist_per_order=total / max(1, n),
            total_distance_m=sum(r.meters for r in self.robots),
            dist_m_per_order=sum(r.meters for r in self.robots) / max(1, n),
            makespan=makespan,
            avg_order_time=sum(done) / max(1, len(done)),
            avg_trip_time=sum(t.t_end - t.t_start for t in trips) / max(1, len(trips)),
            tote_visits=self.tote_visits, tote_saved_by_batching=self.tote_demand - self.tote_visits,
            blocked_ticks=sum(r.blocked for r in self.robots),
            queue_wait_ticks=sum(r.qwait for r in self.robots),
            station_util=sum(self.station_busy) / (self.cfg.n_stations * horizon),
            robot_util=sum(r.busy for r in self.robots) / (self.cfg.n_robots * horizon),
        )


# ==========================================================================
# 5. DATA + EKSPERIMEN
# ==========================================================================
def load_orders(path):
    """Groceries_dataset.csv (Member_number, Date, itemDescription): 1 transaksi = 1 order."""
    tx = defaultdict(set)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tx[(row["Member_number"], row["Date"])].add(row["itemDescription"].strip())
    return [sorted(v) for _, v in sorted(tx.items())]


def synthetic_orders(n=6000, n_sku=167, n_topics=30, seed=1):
    """Data uji (BUKAN data riil): SKU populer ala Zipf + topik belanja (ada struktur co-purchase)."""
    rng = random.Random(seed)
    skus = [f"SKU{i:03d}" for i in range(n_sku)]
    w = [1 / (i + 1) ** 0.8 for i in range(n_sku)]
    topics = [rng.sample(range(n_sku), 7) for _ in range(n_topics)]
    out = []
    for _ in range(n):
        o = set()
        if rng.random() < 0.7:
            o.update(skus[i] for i in rng.sample(rng.choice(topics), rng.randint(2, 4)))
        o.update(rng.choices(skus, w, k=rng.randint(1, 3)))
        out.append(sorted(o))
    return out


def split_orders(orders_all, n_orders, split_seed=0):
    """Hold-out: 80% transaksi untuk belajar (clustering), 20% untuk disimulasikan."""
    skus = sorted({s for o in orders_all for s in o})
    idx = list(range(len(orders_all)))
    random.Random(split_seed).shuffle(idx)
    n_train = int(0.8 * len(idx))
    train = [orders_all[i] for i in idx[:n_train]]
    test = [orders_all[i] for i in idx[n_train:]][:n_orders]
    return skus, train, test


def build_assignment(strategy, skus, lay, train, cfg, seed=1000, min_sim=0.0):
    """-> (assignment, clusters atau None). strategy: random | frequency | cluster"""
    if strategy == "random":
        return assign_random(skus, lay.slots, seed), None
    if strategy == "frequency":
        return assign_frequency(skus, lay.slots, train), None
    if strategy == "cluster":
        return assign_cluster(skus, lay.slots, train, cfg.levels * cfg.baskets, min_sim)
    raise ValueError(strategy)


def run_experiment(orders_all, cfg, n_orders=300, seeds=10, splits=3, split_seed=0,
                   min_sim=0.0, out_csv=None):
    """Untuk tiap split (train/test berbeda): random x `seeds` layout, frequency, cluster.
    Order test, batching, routing, robot IDENTIK antar strategi; hanya penempatan SKU yang beda."""
    lay = None
    runs = []
    for sp in range(splits):
        skus, train, test = split_orders(orders_all, n_orders, split_seed + sp)
        if lay is None:
            lay = make_layout(cfg, len(skus))
            print(f"SKU={len(skus)} | slot={len(lay.slots)} (harus sama) | section={lay.n_sections} | "
                  f"layout={cfg.layout} grid={lay.W}x{lay.H} | robot={cfg.n_robots} station={cfg.n_stations} | "
                  f"batch={cfg.batch_orders} order | tote/trip={cfg.tote_cap}")
            print(f"{splits} split train/test | {seeds} layout random per split\n")
        assert len(skus) == len(lay.slots)

        def go(name, assignment, seed=""):
            k = Sim(cfg, lay, assignment, test).run()
            k.update(strategy=name, seed=seed, split=sp)
            runs.append(k)

        for sd in range(seeds):
            go("random", assign_random(skus, lay.slots, 1000 + sd), sd)
        go("frequency", assign_frequency(skus, lay.slots, train))
        asg, clusters = build_assignment("cluster", skus, lay, train, cfg, min_sim=min_sim)
        go("cluster", asg)
        multi = [c for c in clusters if len(c) > 1]
        print(f"split {sp}: train={len(train)} test={len(test)} | cluster: {len(clusters)} total, "
              f"{len(multi)} berisi >1 SKU, rata-rata ukuran {len(skus)/len(clusters):.2f}")

    cols = ["total_distance", "total_distance_m", "dist_per_order", "dist_m_per_order", "makespan", "avg_order_time", "avg_trip_time",
            "queue_wait_ticks", "blocked_ticks", "robot_util", "station_util"]

    def mean_of(name, c, sp=None):
        return statistics.mean(r[c] for r in runs if r["strategy"] == name and (sp is None or r["split"] == sp))

    print(f"\n{'metrik':<18}{'random (mean+-sd)':>22}{'frequency':>14}{'cluster':>14}"
          f"{'gain cluster vs random':>26}")
    for c in cols:
        rnd = [r[c] for r in runs if r["strategy"] == "random"]
        m, sd = statistics.mean(rnd), (statistics.pstdev(rnd) if len(rnd) > 1 else 0.0)
        fq, cl = mean_of("frequency", c), mean_of("cluster", c)
        gains = [(mean_of("random", c, sp) - mean_of("cluster", c, sp)) / mean_of("random", c, sp) * 100
                 for sp in range(splits) if mean_of("random", c, sp)]
        g = statistics.mean(gains) if gains else 0.0
        gsd = statistics.pstdev(gains) if len(gains) > 1 else 0.0
        print(f"{c:<18}{f'{m:,.1f} +- {sd:,.1f}':>22}{fq:>14,.1f}{cl:>14,.1f}{f'{g:.1f} % +- {gsd:.1f}':>26}")
    print("\n(gain = pengurangan metrik vs rata-rata random pada split yang SAMA; rata-rata +- sd antar split."
          " Untuk robot_util/station_util angka gain bukan 'lebih baik' - baca sebagai perubahan saja.)")
    print("(tote_visits & tote_saved_by_batching sama untuk semua strategi dalam satu split:",
          runs[0]["tote_visits"], "&", runs[0]["tote_saved_by_batching"], "pada split 0)")
    if out_csv:
        keys = ["strategy", "split", "seed"] + [k for k in runs[0] if k not in ("strategy", "split", "seed")]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(runs)
        print("Hasil per-run ->", out_csv)
    return runs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.join("archive", "Groceries_dataset.csv"))
    ap.add_argument("--products", default=None, help="products.csv (format Instacart)")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--orders", type=int, default=300, help="jumlah order test yang disimulasikan")
    ap.add_argument("--seeds", type=int, default=10, help="jumlah layout random yang diulang per split")
    ap.add_argument("--splits", type=int, default=3, help="jumlah split train/test berbeda")
    ap.add_argument("--robots", type=int, default=6)
    ap.add_argument("--stations", type=int, default=1)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--blocks", type=int, default=3, help="hanya untuk --layout grid")
    ap.add_argument("--layout", choices=["physical", "grid"], default="physical",
                    help="physical = denah seperti simulasi1.py")
    ap.add_argument("--min-sim", type=float, default=0.0)
    ap.add_argument("--out", default="hasil_eksperimen.csv")
    a = ap.parse_args()
    cfg = replace(Cfg(), n_robots=a.robots, n_stations=a.stations, batch_orders=a.batch, blocks=a.blocks, layout=a.layout)
    if a.products:
        from instacart_patch import load_instacart
        data = load_instacart(a.products, a.dataset, n_sku=546, max_orders=60000)
    else:
        data = synthetic_orders() if a.synthetic else load_orders(a.dataset)
    run_experiment(data, cfg, a.orders, a.seeds, a.splits, min_sim=a.min_sim, out_csv=a.out)