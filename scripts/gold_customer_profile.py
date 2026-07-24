from pyspark.sql import SparkSession
from pyspark.sql.functions import countDistinct, sum as _sum, avg

spark = SparkSession.builder.appName("GoldCustomerProfile").getOrCreate()

SILVER = "hdfs://master:9000/retail_project/data/silver"
GOLD = "hdfs://master:9000/retail_project/data/gold"

fact_order_items = spark.read.parquet(f"{SILVER}/fact_order_items")
dim_customers = spark.read.parquet(f"{SILVER}/dim_customers")

base = fact_order_items.join(
    dim_customers.select("customer_id", "customer_state", "customer_city"),
    on="customer_id", how="left"
)

customer_spending_profile = base.groupBy("customer_id", "customer_state", "customer_city").agg(
    countDistinct("order_id").alias("total_orders"),
    _sum("price").alias("total_spent"),
    avg("price").alias("avg_item_price"),
    _sum("freight_value").alias("total_freight_paid")
)

customer_spending_profile.write.mode("overwrite").parquet(f"{GOLD}/customer_spending_profile")
print("DONE: customer_spending_profile")
spark.stop()
