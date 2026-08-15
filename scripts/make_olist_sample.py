#!/usr/bin/env python3
"""
Cut the full Olist download down to a sample small enough to commit.

The point is a working structured demo with no Kaggle account: `docker
compose up` then one load command, instead of register, download 43 MB,
unzip 126 MB. The full set is kept out of the repo -- it is large enough to
weigh on every clone, and half of it is a geolocation table the loader only
reads behind `--with-geolocation`.

Referential integrity is the whole difficulty. Sampling each file
independently gives orders whose customer is missing and items pointing at
products that were not kept, which produces a graph full of dangling ids and
queries that silently return nothing. So orders are sampled first and
everything else follows from them: the customers who placed them, the items,
payments and reviews belonging to them, and only the products and sellers
those items reference.

Selection is deterministic (every k-th order id, sorted) rather than random,
so re-running produces the same sample and the committed files do not churn.
Spreading across sorted ids also keeps the sample spread across time and
categories, which matters because the eval asks about 2017 orders and about
specific product categories.

Usage:
    python scripts/make_olist_sample.py --source ~/olist_csvs --out sample_data_to_test/structured/olist-sample
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Enough rows for every eval category to have real data -- multi-hop joins
# and per-state rankings need breadth, not just volume -- while keeping the
# committed sample small enough not to matter in a clone.
DEFAULT_ORDERS = 4000

ORDERS = "olist_orders_dataset.csv"
ITEMS = "olist_order_items_dataset.csv"
CUSTOMERS = "olist_customers_dataset.csv"
PAYMENTS = "olist_order_payments_dataset.csv"
REVIEWS = "olist_order_reviews_dataset.csv"
PRODUCTS = "olist_products_dataset.csv"
SELLERS = "olist_sellers_dataset.csv"
TRANSLATION = "product_category_name_translation.csv"


def read(path: Path) -> tuple[list[str], list[dict]]:
    # utf-8-sig: the translation file carries a BOM, and so may others
    # depending on how the download was unzipped.
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def write(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        # csv defaults to \r\n; force \n so the committed files match what
        # git normalises to and do not show as modified on every checkout.
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="directory holding the full olist_*.csv files")
    ap.add_argument("--out", required=True, help="directory to write the sample into")
    ap.add_argument("--orders", type=int, default=DEFAULT_ORDERS, help=f"orders to keep (default {DEFAULT_ORDERS})")
    args = ap.parse_args()

    src, out = Path(args.source).expanduser(), Path(args.out).expanduser()
    if not src.is_dir():
        sys.exit(f"not a directory: {src}")

    order_fields, orders = read(src / ORDERS)
    orders.sort(key=lambda r: r["order_id"])
    if args.orders >= len(orders):
        kept_orders = orders
    else:
        step = len(orders) / args.orders
        kept_orders = [orders[int(i * step)] for i in range(args.orders)]
    order_ids = {r["order_id"] for r in kept_orders}

    item_fields, items = read(src / ITEMS)
    kept_items = [r for r in items if r["order_id"] in order_ids]
    product_ids = {r["product_id"] for r in kept_items}
    seller_ids = {r["seller_id"] for r in kept_items}
    customer_ids = {r["customer_id"] for r in kept_orders}

    written: list[tuple[str, int]] = []
    for name, keep in (
        (ORDERS, None),
        (ITEMS, None),
        (CUSTOMERS, lambda r: r["customer_id"] in customer_ids),
        (PAYMENTS, lambda r: r["order_id"] in order_ids),
        (REVIEWS, lambda r: r["order_id"] in order_ids),
        (PRODUCTS, lambda r: r["product_id"] in product_ids),
        (SELLERS, lambda r: r["seller_id"] in seller_ids),
        (TRANSLATION, lambda r: True),  # tiny, and dropping rows would lose category names
    ):
        if name == ORDERS:
            fields, rows = order_fields, kept_orders
        elif name == ITEMS:
            fields, rows = item_fields, kept_items
        else:
            fields, all_rows = read(src / name)
            rows = [r for r in all_rows if keep(r)]
        write(out / name, fields, rows)
        written.append((name, len(rows)))

    total = sum((out / n).stat().st_size for n, _ in written)
    print(f"Sample written to {out}\n")
    for name, n in written:
        size = (out / name).stat().st_size
        print(f"  {name:<44}{n:>8,} rows  {size/1e6:6.2f} MB")
    print(f"\n  {'TOTAL':<44}{'':>8}       {total/1e6:6.2f} MB")
    print("\nGeolocation is not included: the loader only reads it behind --with-geolocation.")


if __name__ == "__main__":
    main()
