from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.fpm import FPGrowth

spark = SparkSession.builder.appName("ProductRecommendations").getOrCreate()

SILVER = "hdfs://master:9000/retail_project/data/silver"
GOLD = "hdfs://master:9000/retail_project/data/gold"

# Load fact table (has order_id, product_id per row)
fact_order_items = spark.read.parquet(f"{SILVER}/fact_order_items")

# Build "baskets": all product_ids bought together in one order
baskets = fact_order_items.groupBy("order_id").agg(
    F.collect_set("product_id").alias("items")
)

# Only keep baskets with more than 1 item (single-item orders give no co-purchase signal)
baskets = baskets.filter(F.size("items") > 1)

print("Number of multi-item baskets:", baskets.count())

# Train FP-Growth
fp = FPGrowth(itemsCol="items", minSupport=0.001, minConfidence=0.2)
model = fp.fit(baskets)

# Association rules: antecedent -> consequent, with confidence
rules = model.associationRules

# Flatten: explode antecedent (usually single item) and consequent into simple product_id -> recommended_product_id
recommendations = rules.select(
    F.element_at(F.col("antecedent"), 1).alias("product_id"),
    F.element_at(F.col("consequent"), 1).alias("recommended_product_id"),
    F.col("confidence")
).filter(F.col("product_id").isNotNull() & F.col("recommended_product_id").isNotNull())

print("Number of recommendation rules:", recommendations.count())
recommendations.show(10, truncate=False)

recommendations.write.mode("overwrite").parquet(f"{GOLD}/product_recommendations")
print("DONE: product_recommendations")
spark.stop()
