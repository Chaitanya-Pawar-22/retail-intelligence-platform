from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, year, month, dayofmonth, dayofweek, hour, when

spark = SparkSession.builder \
    .appName("SilverLayerETL") \
    .getOrCreate()

BRONZE = "hdfs://master:9000/retail_project/data/bronze"
SILVER = "hdfs://master:9000/retail_project/data/silver"

# ---------- Load Bronze CSVs ----------
customers = spark.read.csv(f"{BRONZE}/olist_customers_dataset.csv", header=True, inferSchema=True)
orders = spark.read.csv(f"{BRONZE}/olist_orders_dataset.csv", header=True, inferSchema=True)
order_items = spark.read.csv(f"{BRONZE}/olist_order_items_dataset.csv", header=True, inferSchema=True)
payments = spark.read.csv(f"{BRONZE}/olist_order_payments_dataset.csv", header=True, inferSchema=True)
products = spark.read.csv(f"{BRONZE}/olist_products_dataset.csv", header=True, inferSchema=True)
sellers = spark.read.csv(f"{BRONZE}/olist_sellers_dataset.csv", header=True, inferSchema=True)
category_translation = spark.read.csv(f"{BRONZE}/product_category_name_translation.csv", header=True, inferSchema=True)
geolocation = spark.read.csv(f"{BRONZE}/olist_geolocation_dataset.csv", header=True, inferSchema=True)

print("Bronze row counts:")
print("customers:", customers.count())
print("orders:", orders.count())
print("order_items:", order_items.count())
print("payments:", payments.count())
print("products:", products.count())
print("sellers:", sellers.count())

# ---------- Clean: drop nulls in key columns ----------
orders_clean = orders.dropna(subset=["order_id", "customer_id", "order_purchase_timestamp"])
order_items_clean = order_items.dropna(subset=["order_id", "product_id", "seller_id", "price"])
customers_clean = customers.dropna(subset=["customer_id"])
products_clean = products.dropna(subset=["product_id"])

# ---------- dim_products: join with category translation ----------
dim_products = products_clean.join(
    category_translation, on="product_category_name", how="left"
)

# ---------- dim_customers ----------
dim_customers = customers_clean

# ---------- dim_sellers ----------
dim_sellers = sellers.dropna(subset=["seller_id"])

# ---------- fact_order_items: core fact table ----------
fact_order_items = order_items_clean \
    .join(orders_clean, on="order_id", how="inner") \
    .join(payments, on="order_id", how="left") \
    .withColumn("order_purchase_date", to_date(col("order_purchase_timestamp"))) \
    .withColumn("purchase_year", year(col("order_purchase_timestamp"))) \
    .withColumn("purchase_month", month(col("order_purchase_timestamp"))) \
    .withColumn("purchase_day", dayofmonth(col("order_purchase_timestamp"))) \
    .withColumn("purchase_weekday", dayofweek(col("order_purchase_timestamp"))) \
    .withColumn("purchase_hour", hour(col("order_purchase_timestamp"))) \
    .withColumn("is_weekend_order", when(dayofweek(col("order_purchase_timestamp")).isin([1, 7]), 1).otherwise(0))

print("Silver row counts:")
print("fact_order_items:", fact_order_items.count())
print("dim_customers:", dim_customers.count())
print("dim_products:", dim_products.count())
print("dim_sellers:", dim_sellers.count())

# ---------- Write to Silver (Parquet) ----------
fact_order_items.write.mode("overwrite").parquet(f"{SILVER}/fact_order_items")
dim_customers.write.mode("overwrite").parquet(f"{SILVER}/dim_customers")
dim_products.write.mode("overwrite").parquet(f"{SILVER}/dim_products")
dim_sellers.write.mode("overwrite").parquet(f"{SILVER}/dim_sellers")
geolocation.write.mode("overwrite").parquet(f"{SILVER}/dim_geolocation")

print("Silver layer written successfully!")
spark.stop()
