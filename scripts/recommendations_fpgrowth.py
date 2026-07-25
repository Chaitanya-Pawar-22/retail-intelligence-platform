from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.fpm import FPGrowth

spark = SparkSession.builder.appName("ProductRecommendations").getOrCreate()

SILVER = "hdfs://master:9000/retail_project/data/silver"
GOLD = "hdfs://master:9000/retail_project/data/gold"

fact_order_items = spark.read.parquet(f"{SILVER}/fact_order_items")

baskets = fact_order_items.groupBy("order_id").agg(
    F.collect_set("product_id").alias("items")
)
baskets = baskets.filter(F.size("items") > 1)

total_baskets = baskets.count()
print("Number of multi-item baskets:", total_baskets)

fp = FPGrowth(itemsCol="items", minSupport=0.001, minConfidence=0.2)
model = fp.fit(baskets)

# associationRules already contains: antecedent, consequent, confidence, lift, support
rules = model.associationRules

recommendations = rules.select(
    F.element_at(F.col("antecedent"), 1).alias("product_id"),
    F.element_at(F.col("consequent"), 1).alias("recommended_product_id"),
    F.col("confidence"),
    F.col("lift"),
    F.col("support")
).filter(F.col("product_id").isNotNull() & F.col("recommended_product_id").isNotNull())

n_rules = recommendations.count()
print("Number of recommendation rules:", n_rules)

# ---------- EVALUATION: summary quality metrics ----------
print("\n=== MODEL QUALITY METRICS ===")
metrics = recommendations.agg(
    F.avg("confidence").alias("avg_confidence"),
    F.avg("lift").alias("avg_lift"),
    F.avg("support").alias("avg_support"),
    F.max("confidence").alias("max_confidence"),
    F.max("lift").alias("max_lift")
).collect()[0]

print(f"Average Confidence : {metrics['avg_confidence']:.4f}  (higher = more reliable rules)")
print(f"Average Lift        : {metrics['avg_lift']:.4f}  (>1 means genuine pattern, not random chance)")
print(f"Average Support      : {metrics['avg_support']:.4f}  (how common the pattern is overall)")
print(f"Max Confidence found : {metrics['max_confidence']:.4f}")
print(f"Max Lift found        : {metrics['max_lift']:.4f}")

print("\nTop 10 rules by lift (strongest genuine associations):")
recommendations.orderBy(F.col("lift").desc()).show(10, truncate=False)

recommendations.write.mode("overwrite").parquet(f"{GOLD}/product_recommendations")
print("DONE: product_recommendations")
spark.stop()
