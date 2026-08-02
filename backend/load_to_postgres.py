#!/usr/bin/env python3
"""
Load Redis live data into the PostgreSQL serving layer.
Creates serving tables and loads daily metrics + anomaly history from Redis.
Architecture: Redis (live cache) -> PostgreSQL (serving DB) -> dashboard / FastAPI.
Run:  python3 load_to_postgres.py
"""
import json
import redis
import psycopg2

PG = dict(host="localhost", port=5433, dbname="retail",
          user="retail", password="retail123")

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

conn = psycopg2.connect(**PG)
conn.autocommit = True
cur = conn.cursor()

# ----------------------------------------------------------------- schema
cur.execute("""
CREATE TABLE IF NOT EXISTS daily_metrics (
    metric_date DATE PRIMARY KEY,
    orders      INTEGER,
    revenue     NUMERIC(14,2)
);
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS anomalies (
    id          SERIAL PRIMARY KEY,
    order_id    TEXT,
    type        TEXT,
    score       REAL,
    details     TEXT,
    detected_at TIMESTAMP
);
""")

# --------------------------------------- load daily metrics from Redis agg
agg = r.hgetall("stream_aggregates") or {}
dates = set()
for k in agg:
    if ":" in k:
        dates.add(k.split(":", 1)[1])
for d in dates:
    orders = int(float(agg.get("orders:" + d, 0)))
    revenue = float(agg.get("revenue:" + d, 0))
    cur.execute(
        "INSERT INTO daily_metrics (metric_date, orders, revenue) "
        "VALUES (%s, %s, %s) "
        "ON CONFLICT (metric_date) DO UPDATE SET "
        "orders = EXCLUDED.orders, revenue = EXCLUDED.revenue;",
        (d, orders, revenue))

# ------------------------------------- load anomaly history from Redis list
cur.execute("TRUNCATE anomalies;")
raw = r.lrange("recent_anomalies", 0, -1)
for item in raw:
    try:
        a = json.loads(item)
    except Exception:
        continue
    cur.execute(
        "INSERT INTO anomalies (order_id, type, score, details, detected_at) "
        "VALUES (%s, %s, %s, %s, %s);",
        (a.get("order_id"), a.get("type"), a.get("score"),
         a.get("details"), a.get("detected_at")))

cur.execute("SELECT COUNT(*) FROM anomalies;")
n_anom = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM daily_metrics;")
n_days = cur.fetchone()[0]
print("Loaded %d anomalies and %d day(s) of metrics into PostgreSQL." % (n_anom, n_days))

cur.close()
conn.close()
