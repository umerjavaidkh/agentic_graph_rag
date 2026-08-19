#!/usr/bin/env python3
"""
Load the Olist Brazilian e-commerce dataset into Neo4j as the structured
demo graph.

Replaces the Northwind sample. Northwind is ~3k rows across a flat
order/product schema; Olist is ~100k real orders from 2016-2018 with a
genuine six-hop path (customer -> order -> item -> seller, and item ->
product -> category), five timestamps per order, and free-text reviews. That
depth is what the multistep structured strategy exists to demonstrate, and
Northwind could not reach it.

Graph model -- deliberately one node per real-world thing, so a Text-to-Cypher
model can guess the shape without being told:

    (:Customer)-[:PLACED]->(:Order)-[:CONTAINS]->(:OrderItem)-[:OF_PRODUCT]->(:Product)
                                                 (:OrderItem)-[:SOLD_BY]->(:Seller)
                            (:Order)-[:PAID_WITH]->(:Payment)
                            (:Order)-[:HAS_REVIEW]->(:Review)
                                                    (:Product)-[:IN_CATEGORY]->(:Category)

OrderItem is a node rather than a relationship because it carries its own
price and freight and points at BOTH a product and a seller -- collapsing it
would lose the per-line seller.

Geolocation (1,000,163 rows, 70% of the dataset) is skipped unless asked for:
it is a zip-prefix -> lat/lng lookup, not order data, and Customer/Seller
already carry city and state, so "revenue by state" works without it.

Usage:
    python scripts/load_olist.py --source /path/to/olist_csvs
    python scripts/load_olist.py --source ... --with-geolocation
    python scripts/load_olist.py --source ... --clear   # wipe structured data first
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.unstructured.graph.constants import SCHEMA_DOC_LABEL  # noqa: E402
from src.shared.neo4j.driver import get_neo4j_driver  # noqa: E402

BATCH = 5_000

# Labels this loader owns. Used by --clear so wiping the structured demo
# never touches an ingested document graph sharing the same database.
STRUCTURED_LABELS = [
    "Customer", "Order", "OrderItem", "Product", "Seller",
    "Payment", "Review", "Category", "Geo",
]
# Northwind-only labels. Northwind shares Customer/Order/Product/Category with
# the model above, so the two cannot coexist in one database -- --clear
# removes these as well, otherwise a retired Northwind leaves Suppliers and
# Addresses behind that belong to no schema and confuse Text-to-Cypher, which
# reads the live label list.
LEGACY_LABELS = ["Supplier", "Address"]

# Constraints that no longer match the model. `CREATE CONSTRAINT <name> IF NOT
# EXISTS` keys on the NAME, not the property, so re-pointing a key while
# keeping the name silently leaves the old constraint in place and creates no
# index for the new one. Moving Customer from customer_id to unique_id that way
# left every MERGE doing a full label scan -- a load that had taken about a
# minute was still running after twenty. Dropped explicitly, and the names
# below now carry the property so the same edit cannot go unnoticed again.
STALE_CONSTRAINTS = [
    "DROP CONSTRAINT olist_customer IF EXISTS",
    "DROP CONSTRAINT Customer_customerID IF EXISTS",
]

CONSTRAINTS = [
    "CREATE CONSTRAINT olist_customer_unique_id IF NOT EXISTS FOR (n:Customer) REQUIRE n.unique_id IS UNIQUE",
    "CREATE CONSTRAINT olist_order    IF NOT EXISTS FOR (n:Order)    REQUIRE n.order_id IS UNIQUE",
    "CREATE CONSTRAINT olist_item     IF NOT EXISTS FOR (n:OrderItem) REQUIRE n.item_key IS UNIQUE",
    "CREATE CONSTRAINT olist_product  IF NOT EXISTS FOR (n:Product)  REQUIRE n.product_id IS UNIQUE",
    "CREATE CONSTRAINT olist_seller   IF NOT EXISTS FOR (n:Seller)   REQUIRE n.seller_id IS UNIQUE",
    "CREATE CONSTRAINT olist_review   IF NOT EXISTS FOR (n:Review)   REQUIRE n.review_key IS UNIQUE",
    "CREATE CONSTRAINT olist_category IF NOT EXISTS FOR (n:Category) REQUIRE n.name IS UNIQUE",
]


# What each field MEANS, as opposed to what type it is. Written from the
# questions that got wrong answers: every note here corresponds to a real
# mistake the Cypher generator made when it had only names and types to go on.
FIELD_DOCS = [
    ("Customer", "The PERSON who buys, one node each. Olist issues a fresh "
     "customer_id per order, so this node is keyed on unique_id -- count these "
     "for 'how many customers', and group by them for repeat-purchase or "
     "orders-per-customer questions."),
    ("Customer.state", "State where the BUYER lives, two-letter Brazilian code. "
     "Use this for 'which state orders/spends/pays the most freight'. It is NOT "
     "the same as Seller.state, which is where the merchant is."),
    ("Customer.city", "City where the buyer lives. Lowercase, unnormalised."),
    ("Seller.state", "State where the MERCHANT is based. Use only when the "
     "question is about sellers; a question about where customers are asks for "
     "Customer.state."),
    ("Order", "One purchase. Carries status and the timestamps; the money is on "
     "its OrderItems."),
    ("Order.status", "Lifecycle stage. 'canceled' is spelled with one L."),
    ("Order.purchased_at", "DateTime. When the customer placed the order -- the "
     "START of any delivery-duration calculation. Elapsed days is "
     "duration.inDays(purchased_at, delivered_at).days; duration.between(...).days "
     "silently drops whole months and gives a smaller, wrongly-ranked answer."),
    ("Order.delivered_at", "When the customer received it. Null for ~3% of "
     "orders that were never delivered, so exclude nulls from delivery averages."),
    ("Order.estimated_delivery", "The date promised at checkout. 'Late' means "
     "delivered_at > estimated_delivery."),
    ("OrderItem", "One LINE of an order: a single product from a single seller. "
     "An order with three products has three of these."),
    ("OrderItem.price", "Double. Price of this ONE line, excluding freight. "
     "Revenue for any group is the SUM of its lines, never the average. "
     "'Average revenue per X' is a TWO-STAGE aggregate: sum the lines per X "
     "first, then average those sums -- `WITH x, sum(oi.price) AS rev RETURN "
     "avg(rev)`. Writing avg(oi.price) in one stage answers a different "
     "question (the typical line price) and returns one row per X rather than "
     "a single figure."),
    ("OrderItem.freight", "Double. Shipping charged on this ONE line, paid by "
     "the customer. Totals are the SUM of lines; 'average freight per state' "
     "averages over the customer's lines. Same two-stage rule as price when "
     "the question is 'average freight per order or per seller'."),
    ("Payment", "One payment instrument used on an order. Each Payment belongs "
     "to exactly ONE Order, and an order may have several. Never traverse from a "
     "Payment back out to a second Order -- that pattern requires one payment to "
     "have two orders and matches nothing."),
    ("Payment.value", "Amount settled by this instrument. It is not the order "
     "total unless the order had a single payment."),
    ("Review", "The customer's rating of a delivered order. Attached to Order, "
     "NOT to OrderItem -- a review covers the whole order."),
    ("Review.score", "1 to 5. 'Satisfaction' questions mean this."),
    ("Category.name", "Category in PORTUGUESE, which is what Product.category "
     "also holds. Match a Portuguese value here."),
    ("Category.name_english", "The same category in English. If the question "
     "names a category in English, match on THIS property."),
    ("Product", "A catalogue item. Anonymised: there is no product name "
     "anywhere in this data, only an id, a category and physical attributes."),
    ("Product.weight_g", "Shipping weight in grams. A physical measure -- never "
     "an answer to a question about money, cost or value."),
]


def write_field_docs(session) -> int:
    """Store FIELD_DOCS in the graph for SchemaProvider to read."""
    session.run(f"MATCH (d:{SCHEMA_DOC_LABEL}) DETACH DELETE d")
    session.run(
        f"UNWIND $rows AS r MERGE (d:{SCHEMA_DOC_LABEL} {{target: r.target}}) SET d.text = r.text",
        rows=[{"target": t, "text": x} for t, x in FIELD_DOCS],
    )
    return len(FIELD_DOCS)


def rows(path: Path) -> Iterator[dict[str, str]]:
    # utf-8-sig: the category-translation file carries a BOM, which would
    # otherwise become part of the first column's name.
    with path.open(newline="", encoding="utf-8-sig") as fh:
        yield from csv.DictReader(fh)


def num(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def load(session, label: str, path: Path, cypher: str, transform) -> int:
    """UNWIND a CSV into Neo4j in batches, returning the row count.

    Batched rather than row-at-a-time because these files run to six figures:
    one transaction per row would turn a seconds-long load into a very long
    one, and a single transaction for all of them would exhaust the heap.
    """
    batch: list[dict[str, Any]] = []
    total = 0
    for row in rows(path):
        record = transform(row)
        if record is None:
            continue
        batch.append(record)
        if len(batch) >= BATCH:
            session.run(cypher, rows=batch)
            total += len(batch)
            batch = []
    if batch:
        session.run(cypher, rows=batch)
        total += len(batch)
    print(f"  {label:<12} {total:>9,}")
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="directory holding the olist_*.csv files")
    ap.add_argument("--with-geolocation", action="store_true", help="also load the 1M-row zip->lat/lng lookup")
    ap.add_argument("--clear", action="store_true", help="delete existing structured nodes first")
    args = ap.parse_args()

    src = Path(args.source).expanduser()
    if not src.is_dir():
        sys.exit(f"not a directory: {src}")

    driver = get_neo4j_driver()
    with driver.session() as s:
        if args.clear:
            for label in STRUCTURED_LABELS + LEGACY_LABELS:
                # Detached in batches: a single DETACH DELETE over ~500k nodes
                # builds one enormous transaction and can exhaust the heap.
                while True:
                    n = s.run(
                        f"MATCH (n:{label}) WITH n LIMIT 10000 DETACH DELETE n RETURN count(n) AS n"
                    ).single()["n"]
                    if not n:
                        break
            print("  cleared existing structured graph")

        for c in STALE_CONSTRAINTS:
            s.run(c)
        for c in CONSTRAINTS:
            s.run(c)

        print(f"  field docs   {write_field_docs(s):>9,}")

        print("Loading Olist:")
        load(s, "categories", src / "product_category_name_translation.csv",
             "UNWIND $rows AS r MERGE (c:Category {name: r.name}) SET c.name_english = r.en",
             lambda r: {"name": r["product_category_name"], "en": r["product_category_name_english"]})

        # Keyed on unique_id, the PERSON -- not customer_id, which Olist
        # reissues for every order. Modelling a node per customer_id makes the
        # graph say there are 99,441 customers who each bought exactly once,
        # so "how many customers", "repeat purchase rate" and "orders per
        # customer" are all wrong and none of them look wrong. The real
        # figures are 96,096 people and a 3.12% repeat rate.
        # customer_id is deliberately NOT stored. A person has many of them,
        # so the node could only hold one arbitrary value -- and a property
        # that looks like an order reference but is not made a repeat-rate
        # query count DISTINCT order_ref per person, which is always 1, and
        # report that nobody buys twice. Orders join through PLACED.
        # customer_id -> unique_id, so orders can be attached to the person.
        # Held in memory rather than resolved in Cypher: the alternative is a
        # second lookup node per order purely to bridge the two ids.
        person_of = {
            r["customer_id"]: r["customer_unique_id"]
            for r in rows(src / "olist_customers_dataset.csv")
        }

        load(s, "customers", src / "olist_customers_dataset.csv",
             """UNWIND $rows AS r MERGE (c:Customer {unique_id: r.uid})
                SET c.zip_prefix = r.zip, c.city = r.city, c.state = r.state""",
             lambda r: {"id": r["customer_id"], "uid": r["customer_unique_id"],
                        "zip": r["customer_zip_code_prefix"], "city": r["customer_city"],
                        "state": r["customer_state"]})

        load(s, "sellers", src / "olist_sellers_dataset.csv",
             """UNWIND $rows AS r MERGE (x:Seller {seller_id: r.id})
                SET x.zip_prefix = r.zip, x.city = r.city, x.state = r.state""",
             lambda r: {"id": r["seller_id"], "zip": r["seller_zip_code_prefix"],
                        "city": r["seller_city"], "state": r["seller_state"]})

        load(s, "products", src / "olist_products_dataset.csv",
             """UNWIND $rows AS r MERGE (p:Product {product_id: r.id})
                SET p.category = r.cat, p.weight_g = r.w, p.photos = r.photos
                FOREACH (_ IN CASE WHEN r.cat IS NULL OR r.cat = '' THEN [] ELSE [1] END |
                  MERGE (c:Category {name: r.cat}) MERGE (p)-[:IN_CATEGORY]->(c))""",
             lambda r: {"id": r["product_id"], "cat": r["product_category_name"],
                        "w": num(r.get("product_weight_g")), "photos": num(r.get("product_photos_qty"))})

        load(s, "orders", src / "olist_orders_dataset.csv",
             """UNWIND $rows AS r MERGE (o:Order {order_id: r.id})
                SET o.status = r.status,
                    o.purchased_at = datetime(replace(r.purchased, ' ', 'T')),
                    o.delivered_at = CASE WHEN r.delivered = '' THEN null
                                          ELSE datetime(replace(r.delivered, ' ', 'T')) END,
                    o.estimated_delivery = CASE WHEN r.estimated = '' THEN null
                                                ELSE datetime(replace(r.estimated, ' ', 'T')) END
                WITH o, r MATCH (c:Customer {unique_id: r.customer}) MERGE (c)-[:PLACED]->(o)""",
             lambda r: {"id": r["order_id"], "customer": person_of.get(r["customer_id"]), "status": r["order_status"],
                        "purchased": r["order_purchase_timestamp"],
                        "delivered": r["order_delivered_customer_date"],
                        "estimated": r["order_estimated_delivery_date"]})

        load(s, "order items", src / "olist_order_items_dataset.csv",
             """UNWIND $rows AS r
                MERGE (i:OrderItem {item_key: r.key})
                SET i.price = r.price, i.freight = r.freight, i.line_no = r.line
                WITH i, r
                MATCH (o:Order {order_id: r.order}) MERGE (o)-[:CONTAINS]->(i)
                WITH i, r
                MATCH (p:Product {product_id: r.product}) MERGE (i)-[:OF_PRODUCT]->(p)
                WITH i, r
                MATCH (x:Seller {seller_id: r.seller}) MERGE (i)-[:SOLD_BY]->(x)""",
             lambda r: {"key": f"{r['order_id']}:{r['order_item_id']}", "order": r["order_id"],
                        "product": r["product_id"], "seller": r["seller_id"],
                        "line": num(r["order_item_id"]), "price": num(r["price"]),
                        "freight": num(r["freight_value"])})

        load(s, "payments", src / "olist_order_payments_dataset.csv",
             """UNWIND $rows AS r
                MATCH (o:Order {order_id: r.order})
                MERGE (o)-[:PAID_WITH]->(p:Payment {payment_key: r.key})
                SET p.type = r.type, p.installments = r.inst, p.value = r.value""",
             lambda r: {"key": f"{r['order_id']}:{r['payment_sequential']}", "order": r["order_id"],
                        "type": r["payment_type"], "inst": num(r["payment_installments"]),
                        "value": num(r["payment_value"])})

        load(s, "reviews", src / "olist_order_reviews_dataset.csv",
             """UNWIND $rows AS r
                MATCH (o:Order {order_id: r.order})
                MERGE (v:Review {review_key: r.key})
                SET v.score = r.score, v.title = r.title, v.comment = r.comment
                MERGE (o)-[:HAS_REVIEW]->(v)""",
             lambda r: {"key": f"{r['review_id']}:{r['order_id']}", "order": r["order_id"],
                        "score": num(r["review_score"]), "title": r.get("review_comment_title") or "",
                        "comment": r.get("review_comment_message") or ""})

        if args.with_geolocation:
            load(s, "geolocation", src / "olist_geolocation_dataset.csv",
                 """UNWIND $rows AS r MERGE (g:Geo {zip_prefix: r.zip, lat: r.lat, lng: r.lng})
                    SET g.city = r.city, g.state = r.state""",
                 lambda r: {"zip": r["geolocation_zip_code_prefix"], "lat": num(r["geolocation_lat"]),
                            "lng": num(r["geolocation_lng"]), "city": r["geolocation_city"],
                            "state": r["geolocation_state"]})

        print("\nGraph now holds:")
        for r in s.run(
            "MATCH (n) WHERE any(l IN labels(n) WHERE l IN $labels) "
            "UNWIND labels(n) AS l WITH l WHERE l IN $labels "
            "RETURN l AS label, count(*) AS n ORDER BY n DESC",
            labels=STRUCTURED_LABELS,
        ):
            print(f"  {r['label']:<12} {r['n']:>9,}")


if __name__ == "__main__":
    main()
