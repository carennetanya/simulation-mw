# -*- coding: utf-8 -*-
"""Instacart loader for wh_sim.py / manual_sim.py -> exactly the n_sku most frequent products
(default 546 = number of storage slots of the physical layout: 91 sections x 6 slots)."""
import csv
from collections import Counter, defaultdict


def load_instacart(products_csv, order_products_csv, n_sku=546, max_orders=60000, min_items=1):
    """products.csv (product_id, product_name, ...) + order_products__train.csv / __prior.csv (order_id, product_id, ...).
    1) keep the n_sku most frequently ordered products
    2) an order = the set of kept products of one order_id (all other products are dropped)
    3) take the first max_orders orders (by order_id) that have at least min_items items."""
    name = {}
    with open(products_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name[row["product_id"]] = row["product_name"].strip()
    cnt = Counter()
    with open(order_products_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cnt[row["product_id"]] += 1
    top = {pid for pid, _ in cnt.most_common(n_sku)}
    if len(top) < n_sku:
        raise ValueError(f"Only {len(top)} products in the data; {n_sku} are needed.")
    label = {}
    for pid in sorted(top):                       # product names must be unique
        nm = name.get(pid, pid)
        label[pid] = nm if nm not in label.values() else f"{nm} #{pid}"
    tx = defaultdict(set)
    with open(order_products_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["product_id"] in top:
                tx[int(row["order_id"])].add(label[row["product_id"]])
    orders = [sorted(v) for _, v in sorted(tx.items()) if len(v) >= min_items]
    return orders[:max_orders]