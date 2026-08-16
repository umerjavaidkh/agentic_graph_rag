# Olist graph — context for writing eval questions

Everything below is read from the live graph (full dataset: 99,441 orders), not from
the Kaggle documentation. Property names here are the **graph's** names, which differ
from the CSV column names — that difference has already caused one wrong answer while
writing this file, so check against this list rather than the CSV.

## Shape

```
(:Customer)-[:PLACED]->(:Order)-[:CONTAINS]->(:OrderItem)-[:OF_PRODUCT]->(:Product)-[:IN_CATEGORY]->(:Category)
                          |                       |
                          |                       +-[:SOLD_BY]->(:Seller)
                          +-[:PAID_WITH]->(:Payment)
                          +-[:HAS_REVIEW]->(:Review)
```

| Label | Nodes | Properties |
| --- | --- | --- |
| `:Customer` | 99,441 | `customer_id`, `unique_id`, `city`, `state`, `zip_prefix` |
| `:Order` | 99,441 | `order_id`, `status`, `purchased_at`, `delivered_at`, `estimated_delivery` (all DateTime) |
| `:OrderItem` | 112,650 | `item_key`, `line_no`, `price`, `freight` |
| `:Payment` | 103,886 | `payment_key`, `type`, `installments`, `value` |
| `:Review` | 99,224 | `review_key`, `score`, `title`, `comment` |
| `:Product` | 32,951 | `product_id`, `category`, `weight_g`, `photos` |
| `:Seller` | 3,095 | `seller_id`, `city`, `state`, `zip_prefix` |
| `:Category` | 73 | `name` (Portuguese), `name_english` |

Edge counts: `CONTAINS` 112,650 · `OF_PRODUCT` 112,650 · `SOLD_BY` 112,650 ·
`PAID_WITH` 103,886 · `PLACED` 99,441 · `HAS_REVIEW` 99,224 · `IN_CATEGORY` 32,341

## Values you can filter on

| Field | Values |
| --- | --- |
| `Order.purchased_at` | 2016-09-04 → 2018-10-17 |
| `Order.status` | delivered 96,478 · shipped 1,107 · canceled 625 · unavailable 609 · invoiced 314 · processing 301 · created 5 · approved 2 |
| `Payment.type` | credit_card 76,795 · boleto 19,784 · voucher 5,775 · debit_card 1,529 · not_defined 3 |
| `Review.score` | 1★ 11,424 · 2★ 3,151 · 3★ 8,179 · 4★ 19,142 · 5★ 57,328 |
| `OrderItem.price` | 0.85 – 6,735.00 (avg 120.65) |
| `OrderItem.freight` | 0.00 – 409.68 (avg 19.99) |
| `Payment.installments` | 0 – 24 (avg 2.85) |
| `Customer.state` | 27 states; largest SP, RJ, MG, RS, PR, SC |
| Top categories by revenue | beleza_saude, relogios_presentes, cama_mesa_banho, esporte_lazer, informatica_acessorios, moveis_decoracao |

## Traps — read before writing questions

These will produce a confidently wrong expected value if you miss them.

- **`unique_id` is the person; `customer_id` is per-order.** Olist issues a new
  `customer_id` for every order, so `count(DISTINCT customer_id)` equals the order
  count and any retention question grouped on it returns nonsense. Repeat-purchase
  rate on `unique_id` is **3.12%** (2,997 of 96,096).
- **Property names differ from the CSV.** `customer_unique_id` → `unique_id`,
  `customer_zip_code_prefix` → `zip_prefix`, `product_weight_g` → `weight_g`,
  `order_purchase_timestamp` → `purchased_at`, `freight_value` → `freight`. A
  reference to a name that does not exist returns **null, not an error**, so the query
  silently produces 0 or null.
- **Category names are Portuguese** in `Category.name`; English is in `name_english`.
  `Product.category` also holds the Portuguese string.
- **Products have no name.** The source data is anonymised — there is a
  `product_name_lenght` column upstream but no name, so a product can only be
  identified by id and category.
- **2,965 orders are never delivered** (`delivered_at IS NULL`), and 625 are cancelled.
  Any delivery-time average must decide whether to exclude them.
- **610 products have no category edge**, so category rollups do not cover every product.
- **58,247 reviews have a blank comment** — score is present, free text often is not.
- `Order.estimated_delivery` is a date at midnight; `delivered_at` has a real time.
  Comparing them directly is fine for late/on-time, but "days late" needs care.
- Duration argument order matters: `duration.inDays(earlier, later)`. Reversed gives a
  negative and reads as a data problem rather than a query bug.

## Question themes worth covering

Listed as coverage areas, **not** as questions known to work — the point of the suite is
to find what breaks, so write them before checking outcomes.

1. **Operational totals** — order counts, revenue, AOV, by period
2. **Delivery performance** — on-time rate, average delay, delay by state or category
3. **Satisfaction drivers** — review score against delivery lateness, price, category
4. **Seller analytics** — top sellers, revenue concentration, sellers per state,
   cross-state selling
5. **Category performance** — revenue, volume, average review, freight share
6. **Geography** — orders and revenue by state, freight versus distance proxy
7. **Payment behaviour** — instalment distribution, type mix, value by type
8. **Order funnel** — cancellation rate, status breakdown, unavailable orders
9. **Retention** — repeat-purchase rate, orders per customer (use `unique_id`)
10. **Absence** — things this graph genuinely does not hold (product names, profit,
    cost of goods, employees, warehouses, discounts, returns) — the answer must be a
    refusal, not a plausible number

## Ground truth already computed

Useful as fixed points; verify anything else yourself before trusting it.

| Question | Value |
| --- | --- |
| Average review, delivered late | **2.57** (n=7,701) |
| Average review, delivered on time | **4.29** (n=88,658) |
| Repeat-purchase rate (`unique_id`) | **3.12%** |
| Average order value (sum of item prices per order) | **137.75** |
| Average single order-item price | **120.65** |
| Freight as share of item value + freight | **14.21%** |
| Total freight | **2,251,910** |
| Average review score, all reviews | **4.09** |
| Orders in 2017 | **45,101** |
| Orders delivered later than estimated | **7,827** |
| Average days purchase → delivery | **12.09** |
