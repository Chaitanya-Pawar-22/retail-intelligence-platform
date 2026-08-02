#!/usr/bin/env python3
import os, json, redis
from datetime import datetime
import pyspark
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (StructType, StructField, StringType,
                               DoubleType, IntegerType, ArrayType)

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC_STREAM    = "test_olist_orders_stream"

BASE         = os.path.abspath("rip_output")
LOCAL_ORDERS = "file://" + os.path.join(BASE, "fact_order_items_streaming")
CHECKPOINT   = "file://" + os.path.join(BASE, "_checkpoint")

REDIS_HOST, REDIS_PORT = "localhost", 6379
REDIS_ALERT_KEY = "recent_anomalies"
REDIS_AGG_KEY   = "stream_aggregates"

GEO_TIME_HOURS         = 4
VELOCITY_LIMIT         = 5
FIRST_ORDER_HIGH_VALUE = 500
OFF_HOURS_START, OFF_HOURS_END = 1, 5
OFF_HOURS_HIGH_VALUE   = 1000
AMOUNT_HIGH_VALUE      = 3000
FREIGHT_LIMIT          = 100
FAR_PAIRS = {("SP","AM"),("SP","PA"),("RJ","AM"),("MG","AC"),
             ("SP","RR"),("PR","AP"),("RS","AM"),("SP","AC")}

KAFKA_PKG = f"org.apache.spark:spark-sql-kafka-0-10_2.12:{pyspark.__version__}"
spark = (SparkSession.builder
         .appName("Retail_Streaming_Rules")
         .master("local[*]")
         .config("spark.jars.packages", KAFKA_PKG)
         .config("spark.sql.shuffle.partitions", "4")
         .config("spark.driver.bindAddress", "127.0.0.1")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")
print(f"Spark {pyspark.__version__} ready. Kafka pkg: {KAFKA_PKG}")

items_schema = ArrayType(StructType([
    StructField("order_item_id", IntegerType()),
    StructField("product_id", StringType()),
    StructField("seller_id", StringType()),
    StructField("price", DoubleType()),
    StructField("freight_value", DoubleType()),
    StructField("product_category", StringType()),
]))
order_schema = StructType([
    StructField("order_id", StringType()),
    StructField("customer_id", StringType()),
    StructField("customer_unique_id", StringType()),
    StructField("customer_state", StringType()),
    StructField("customer_city", StringType()),
    StructField("order_status", StringType()),
    StructField("order_purchase_timestamp", StringType()),
    StructField("order_approved_at", StringType()),
    StructField("order_delivered_carrier_date", StringType()),
    StructField("order_delivered_customer_date", StringType()),
    StructField("order_estimated_delivery_date", StringType()),
    StructField("items", items_schema),
    StructField("payment_type", StringType()),
    StructField("payment_installments", IntegerType()),
    StructField("total_payment_value", DoubleType()),
    StructField("injected_anomaly", StructType([
        StructField("amount_anomaly", StringType()),
        StructField("geo_anomaly", StringType()),
    ])),
])

def push_alert(r, order_id, typ, detail):
    r.lpush(REDIS_ALERT_KEY, json.dumps({
        "order_id": order_id, "type": typ, "score": 1.0, "details": detail,
        "detected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }))

def process_batch(df, epoch_id):
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    if df.isEmpty():
        return
    parsed = (df.select(F.from_json(F.col("value").cast("string"), order_schema).alias("d"))
                .select("d.*"))
    valid = parsed.filter(F.col("order_id").isNotNull() & F.col("customer_id").isNotNull())
    if valid.isEmpty():
        return
    valid = (valid
             .withColumn("first_item_freight", F.col("items")[0]["freight_value"])
             .withColumn("purchase_hour",
                         F.hour(F.to_timestamp("order_purchase_timestamp", "dd-MM-yyyy HH:mm")))
             .withColumn("is_off_hours",
                         (F.col("purchase_hour") >= OFF_HOURS_START) &
                         (F.col("purchase_hour") < OFF_HOURS_END) &
                         (F.col("total_payment_value") > OFF_HOURS_HIGH_VALUE))
             .withColumn("is_amount", F.col("total_payment_value") > AMOUNT_HIGH_VALUE))

    alerts = 0
    rows = valid.select("order_id", "customer_unique_id", "customer_state",
                        "order_purchase_timestamp", "first_item_freight",
                        "total_payment_value").collect()
    for row in rows:
        cust, state = row["customer_unique_id"], row["customer_state"]
        first_freight = row["first_item_freight"] or 0.0
        total = row["total_payment_value"] or 0.0
        try:
            ts = datetime.strptime(row["order_purchase_timestamp"], "%d-%m-%Y %H:%M")
        except Exception:
            continue
        ls = r.hget(f"cust:{cust}", "last_state")
        lt = r.hget(f"cust:{cust}", "last_purchase_ts")
        if ls and lt:
            try:
                diff_h = abs((ts - datetime.strptime(lt, "%Y-%m-%d %H:%M:%S")).total_seconds()) / 3600.0
                if (diff_h < GEO_TIME_HOURS and state != ls and
                        ((state, ls) in FAR_PAIRS or (ls, state) in FAR_PAIRS)):
                    push_alert(r, row["order_id"], "geo_velocity",
                               f"State jump {ls}->{state} in {diff_h:.1f}h"); alerts += 1
            except Exception:
                pass
        r.hset(f"cust:{cust}", mapping={"last_state": state,
                                        "last_purchase_ts": ts.strftime("%Y-%m-%d %H:%M:%S")})
        if not r.sismember("cust_seen", cust):
            if total > FIRST_ORDER_HIGH_VALUE:
                push_alert(r, row["order_id"], "first_order_high_value",
                           f"First order {total:.0f} exceeds {FIRST_ORDER_HIGH_VALUE}"); alerts += 1
            r.sadd("cust_seen", cust)
        cnt = r.incr(f"cust:{cust}:order_count"); r.expire(f"cust:{cust}:order_count", 3600)
        if cnt > VELOCITY_LIMIT:
            push_alert(r, row["order_id"], "velocity", f"{cnt} orders in 1 hour"); alerts += 1
        if first_freight > FREIGHT_LIMIT:
            push_alert(r, row["order_id"], "freight_mismatch",
                       f"Freight {first_freight:.0f} unusually high"); alerts += 1

    flagged = (valid.filter(F.col("is_amount") | F.col("is_off_hours"))
                    .select("order_id", "is_amount", "is_off_hours",
                            "purchase_hour", "total_payment_value").collect())
    for row in flagged:
        if row["is_amount"]:
            push_alert(r, row["order_id"], "amount",
                       f"Value {row['total_payment_value']:.0f} exceeds {AMOUNT_HIGH_VALUE}"); alerts += 1
        if row["is_off_hours"]:
            push_alert(r, row["order_id"], "off_hours",
                       f"Order at {row['purchase_hour']}h value {row['total_payment_value']:.0f}"); alerts += 1

    r.ltrim(REDIS_ALERT_KEY, 0, 199)
    if alerts:
        print("  [batch %s] %s anomalies -> Redis" % (epoch_id, alerts))

    fact_rows = (valid.withColumn("item", F.explode("items")).select(
        F.col("order_id"),
        F.col("item.order_item_id").cast("int").alias("order_item_id"),
        F.col("item.product_id").alias("product_id"),
        F.col("item.seller_id").alias("seller_id"),
        F.col("customer_id"), F.col("customer_unique_id"), F.col("order_status"),
        F.to_timestamp("order_purchase_timestamp", "dd-MM-yyyy HH:mm").alias("order_purchase_timestamp"),
        F.col("item.price").cast("double").alias("price"),
        F.col("item.freight_value").cast("double").alias("freight_value"),
        F.col("item.product_category").alias("product_category_name"),
    ))
    fact_rows.write.format("parquet").mode("append").save(LOCAL_ORDERS)

    agg = fact_rows.agg(F.sum(F.col("price") + F.col("freight_value")).alias("rev"),
                        F.count("*").alias("cnt")).collect()[0]
    today = datetime.now().strftime("%Y-%m-%d")
    r.hincrbyfloat(REDIS_AGG_KEY, f"revenue:{today}", float(agg["rev"] or 0))
    r.hincrby(REDIS_AGG_KEY, f"orders:{today}", int(agg["cnt"] or 0))

df_kafka = (spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
            .option("subscribe", TOPIC_STREAM)
            .option("startingOffsets", "latest").load())

query = (df_kafka.writeStream
         .foreachBatch(process_batch)
         .outputMode("append")
         .trigger(processingTime="10 seconds")
         .option("checkpointLocation", CHECKPOINT).start())

print("Streaming pipeline started (6 rule-based anomalies). Waiting for orders...")
query.awaitTermination()
