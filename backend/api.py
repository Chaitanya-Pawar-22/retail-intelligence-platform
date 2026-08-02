#!/usr/bin/env python3
"""Retail Intelligence Platform - FastAPI backend (PostgreSQL + Redis)."""
import json
from datetime import datetime
import redis
import psycopg2
import psycopg2.extras
from fastapi import FastAPI, Query

PG = dict(host="localhost", port=5433, dbname="retail",
          user="retail", password="retail123")

app = FastAPI(title="Retail Intelligence Platform API", version="1.0")

def get_pg_conn():
    return psycopg2.connect(**PG)

def get_redis_conn():
    return redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

def pg_query(sql, params=None):
    conn = get_pg_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(sql, params or ())
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(row) for row in rows]

@app.get("/")
def read_root():
    return {"status": "online",
            "message": "Retail Intelligence Platform API Running",
            "endpoints": ["/health", "/sales", "/metrics/live",
                          "/anomalies/live", "/anomalies/historical",
                          "/anomalies/summary", "/docs"]}

@app.get("/health")
def health():
    status = {"api": "ok"}
    try:
        pg_query("SELECT 1 AS ok;")
        status["postgres"] = "ok"
    except Exception as e:
        status["postgres"] = "error: " + str(e)
    try:
        get_redis_conn().ping()
        status["redis"] = "ok"
    except Exception as e:
        status["redis"] = "error: " + str(e)
    return status

@app.get("/sales")
def sales():
    return pg_query("SELECT metric_date, orders, revenue "
                    "FROM daily_metrics ORDER BY metric_date;")

@app.get("/metrics/live")
def metrics_live():
    r = get_redis_conn()
    today = datetime.now().strftime("%Y-%m-%d")
    agg = r.hgetall("stream_aggregates") or {}
    orders = int(float(agg.get("orders:" + today, 0)))
    revenue = float(agg.get("revenue:" + today, 0))
    aov = round(revenue / orders, 2) if orders else 0
    return {"date": today, "orders": orders,
            "revenue": round(revenue, 2), "avg_order_value": aov}

@app.get("/anomalies/live")
def anomalies_live(limit: int = Query(20, ge=1, le=200)):
    r = get_redis_conn()
    raw = r.lrange("recent_anomalies", 0, limit - 1)
    data = []
    for item in raw:
        try:
            data.append(json.loads(item))
        except Exception:
            pass
    return {"count": len(data), "data": data}

@app.get("/anomalies/historical")
def anomalies_historical(limit: int = Query(50, ge=1, le=500)):
    rows = pg_query(
        "SELECT id, order_id, type, score, details, detected_at "
        "FROM anomalies ORDER BY detected_at DESC NULLS LAST LIMIT %s;",
        (limit,))
    return {"count": len(rows), "data": rows}

@app.get("/anomalies/summary")
def anomalies_summary():
    return pg_query("SELECT type, COUNT(*) AS count FROM anomalies "
                    "GROUP BY type ORDER BY count DESC;")
