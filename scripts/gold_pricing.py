from pyspark.sql import SparkSession
from pyspark.sql.functions import col, datediff, round as _round

spark = SparkSession.builder.appName("GoldPricing").getOrCreate()

SILVER = "hdfs://master:9000/retail_project/data/silver"
GOLD = "hdfs://master:9000/retail_project/data/gold"

fact_order_items = spark.read.parquet(f"{SILVER}/fact_order_items")
dim_products = spark.read.parquet(f"{SILVER}/dim_products")
dim_customers = spark.read.parquet(f"{SILVER}/dim_customers")

base = fact_order_items \
    .join(dim_products.select("product_id", "product_weight_g", "product_length_cm",
                               "product_width_cm", "product_height_cm"), on="product_id", how="left") \
    .join(dim_customers.select("customer_id", "customer_state"), on="customer_id", how="left")

order_features_for_pricing = base \
    .withColumn("delivery_days", datediff(col("order_delivered_customer_date"), col("order_purchase_date"))) \
    .withColumn("estimated_vs_actual_delivery",
                datediff(col("order_estimated_delivery_date"), col("order_delivered_customer_date"))) \
    .withColumn("freight_ratio", _round(col("freight_value") / col("price"), 4)) \
    .select(
        "order_id", "product_id", "price", "freight_value", "freight_ratio",
        "delivery_days", "estimated_vs_actual_delivery",
        "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm",
        "customer_state", "purchase_hour", "purchase_weekday"
    ) \
    .withColumnRenamed("purchase_hour", "order_hour") \
    .withColumnRenamed("purchase_weekday", "order_dayofweek") \
    .filter(col("delivery_days").isNotNull() & (col("price") > 0))

order_features_for_pricing.write.mode("overwrite").parquet(f"{GOLD}/order_features_for_pricing")
print("DONE: order_features_for_pricing")
spark.stop()
