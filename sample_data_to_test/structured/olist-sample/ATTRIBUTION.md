# Olist sample — source and licence

These CSVs are a **subset** of the Brazilian E-Commerce Public Dataset published by
Olist on Kaggle:

<https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>

Regenerate it from a full download with:

```bash
python scripts/make_olist_sample.py --source /path/to/olist_csvs --out sample_data_to_test/structured/olist-sample
```

## What was kept

4,000 orders selected deterministically (every k-th order id, sorted), plus every
row reachable from them — the customers who placed them, their items, payments and
reviews, and only the products and sellers those items reference. Sampling each file
independently would have produced dangling ids and queries that silently return
nothing, so nothing here is sampled in isolation.

`olist_geolocation_dataset.csv` is **not** included. It is the largest file in the
download (61 MB) and the loader only reads it behind `--with-geolocation`.

## Licence — read before using commercially

The upstream dataset is published under **CC BY-NC-SA 4.0**, which is *not* the same
licence as this repository's MIT. In particular the **NonCommercial** term applies to
this data, and ShareAlike applies to derivatives of it.

This matters if you are evaluating the project for commercial use: the **code** is MIT
and unaffected, but **this folder is not**. Delete it and point `load_olist.py` at your
own download or your own data if that distinction is a problem for you.

Confirm the current terms on the Kaggle page above before redistributing — the licence
shown there is authoritative, not this note.
