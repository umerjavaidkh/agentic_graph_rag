"""Loading structured data straight from a database.

A live database is the only source that already knows its own schema. CSVs
force the loader to guess a relationship from a column called `customer_id`,
and it guesses wrongly on anything that merely looks like a key; reflection
reports what the schema actually declares.
"""
import sqlite3

import pytest

from src.structured.ingestion.tabular import (
    _as_sql_url,
    _rows_for,
    _safe_url,
    infer_schema,
    source_tag,
)

pytest.importorskip("sqlalchemy")


@pytest.fixture
def shop_db(tmp_path):
    """A database with a DECLARED foreign key, which is the point of the source."""
    path = tmp_path / "shop.db"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            total REAL,
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        );
        INSERT INTO customers VALUES (1, 'Ada'), (2, 'Grace');
        INSERT INTO orders VALUES (10, 1, 99.5), (11, 2, 12.0), (12, 1, 7.25);
        """
    )
    con.commit()
    con.close()
    return f"sqlite:///{path}"


def test_a_url_is_a_database_and_a_path_is_not(tmp_path):
    """Existing files win over URL-shaped strings: a relative path contains no
    '://' but is still a file, and a typo'd URL should fail connecting rather
    than be read as a missing file."""
    assert _as_sql_url("postgresql://u:p@host/db")
    assert _as_sql_url(tmp_path / "x.db") is None
    real = tmp_path / "real.db"
    real.write_text("")
    assert _as_sql_url(str(real)) is None


def test_declared_foreign_keys_become_relationships(shop_db):
    schema = infer_schema(shop_db)
    by_name = {t.name: t for t in schema.tables}
    assert set(by_name) == {"customers", "orders"}
    assert by_name["customers"].primary_key == "id"
    # the whole reason to read a database rather than its CSV export
    assert by_name["orders"].foreign_keys == {"customer_id": ("customers", "id")}
    assert all(t.source == "sql" for t in schema.tables)


def test_rows_stream_through_the_generator(shop_db):
    """Regression: _rows_for contains yields, so `return <iterator>` inside it
    sets a StopIteration value instead of handing back the rows -- every SQL
    table loaded as zero rows, with no error raised anywhere."""
    schema = infer_schema(shop_db)
    orders = next(t for t in schema.tables if t.name == "orders")
    rows = list(_rows_for(shop_db, orders))
    assert len(rows) == 3
    assert rows[0]["customer_id"] == 1


def test_the_provenance_stamp_never_carries_a_password():
    """`_source` is written onto every node this loader creates, so a raw URL
    would put the database password in the graph and in its backups."""
    secret = "postgresql://admin:s3cr3t@db.internal:5432/prod"
    assert "s3cr3t" not in _safe_url(secret)
    assert "s3cr3t" not in source_tag(secret)
    assert "db.internal" in source_tag(secret)      # still identifies the source


def test_a_malformed_url_still_does_not_leak():
    assert "s3cr3t" not in _safe_url("not-a-url://admin:s3cr3t@@@")


def test_composite_primary_key_is_reported_not_silently_dropped(tmp_path):
    """No single property to MERGE on, so rows would load without identity and
    duplicate on re-run. The caller has to be told before trusting the plan."""
    path = tmp_path / "composite.db"
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE pairs (a INTEGER, b INTEGER, v TEXT, PRIMARY KEY (a, b));"
    )
    con.commit()
    con.close()
    schema = infer_schema(f"sqlite:///{path}")
    assert schema.tables[0].primary_key is None
    assert any("composite primary key" in w for w in schema.warnings)
