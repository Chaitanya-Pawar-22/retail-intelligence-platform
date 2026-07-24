from pyspark.sql import SparkSession
from pyspark.sql.functions import col, datediff, round as _round, when, count, sum as _sum, max as _max

spark = SparkSession.builder.appName("GoldMLFeatures").getOrCreate()

SILVER = "hdfs://master:9000/retail_project/data/silver"
GOLD = "hdfs://master:9000/retail_project/data/gold"

fact_order_items = spark.read.parquet(f"{SILVER}/fact_order_items")
dim_products = spark.read.parquet(f"{SILVER}/dim_products")
dim_customers = spark.read.parquet(f"{SILVER}/dim_customers")
dim_sellers = spark.read.parquet(f"{SILVER}/dim_sellers")

base = fact_order_items \
    .join(dim_products.select("product_id", "product_weight_g", "product_length_cm",
                               "product_width_cm", "product_height_cm", "product_photos_qty",
                               "product_name_lenght", "product_description_lenght"),
          on="product_id", how="left") \
    .join(dim_customers.select("customer_id", "customer_state", "customer_city"), on="customer_id", how="left") \
    .join(dim_sellers.select("seller_id", "seller_state", "seller_city"), on="seller_id", how="left")

order_level_agg = base.groupBy("order_id").agg(
    count("product_id").alias("total_items_in_order"),
    _sum("price").alias("total_order_value"),
    _max("payment_installments").alias("max_installments"),
    _sum("payment_value").alias("total_payment")
)

ml_feature_store_v2 = base \
    .withColumn("delay_vs_estimate_days",
                datediff(col("order_estimated_delivery_date"), col("order_delivered_customer_date"))) \
    .withColumn("estimated_delivery_days",
                datediff(col("order_estimated_delivery_date"), col("order_purchase_date"))) \
    .withColumn("is_same_state", when(col("customer_state") == col("seller_state"), 1).otherwise(0)) \
    .withColumn("price_per_item", col("price")) \
    .withColumn("freight_per_item", col("freight_value")) \
    .withColumn("freight_price_ratio", _round(col("freight_value") / col("price"), 4)) \
    .withColumn("weight_per_item", col("product_weight_g")) \
    .withColumn("product_volume",
                col("product_length_cm") * col("product_width_cm") * col("product_height_cm")) \
    .join(order_level_agg, on="order_id", how="left") \
    .select(
        "delay_vs_estimate_days", "estimated_delivery_days",
        "purchase_month", "purchase_day", "purchase_weekday", "purchase_hour", "is_weekend_order",
        "product_weight_g", "product_length_cm", "product_width_cm", "product_height_cm", "product_volume",
        "product_photos_qty", "product_name_lenght", "product_description_lenght",
        "customer_state", "customer_city", "seller_state", "seller_city", "is_same_state",
        "price_per_item", "freight_per_item", "total_items_in_order",
        "max_installments", "total_payment", "total_order_value",
        "freight_price_ratio", "weight_per_item"
    ) \
    .filter(col("delay_vs_estimate_days").isNotNull() & col("product_weight_g").isNotNull())

ml_feature_store_v2.write.mode("overwrite").partitionBy("purchase_month").parquet(f"{GOLD}/ml_feature_store_v2")
print("DONE: ml_feature_store_v2")
spark.stop()
