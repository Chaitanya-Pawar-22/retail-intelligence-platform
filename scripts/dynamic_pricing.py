from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("DynamicPricing").getOrCreate()

GOLD = "hdfs://master:9000/retail_project/data/gold"

df = spark.read.parquet(f"{GOLD}/order_features_for_pricing")
print("Total rows loaded:", df.count())

df = df.dropna(subset=[
    "price", "freight_value", "delivery_days", "product_weight_g",
    "product_length_cm", "product_width_cm", "product_height_cm"
])
print("Rows after dropping nulls:", df.count())

# ---------- Encode categorical ----------
indexer = StringIndexer(inputCol="customer_state", outputCol="customer_state_idx", handleInvalid="keep")

numeric_cols = [
    "freight_value", "freight_ratio", "delivery_days", "estimated_vs_actual_delivery",
    "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm",
    "order_hour", "order_dayofweek"
]
feature_cols = numeric_cols + ["customer_state_idx"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")

gbt = GBTRegressor(labelCol="price", featuresCol="features", maxIter=50, maxDepth=5)

pipeline = Pipeline(stages=[indexer, assembler, gbt])

# ---------- Train/Test split ----------
train, test = df.randomSplit([0.8, 0.2], seed=42)
print(f"Train rows: {train.count()}   Test rows: {test.count()}")

print("Training GBTRegressor...")
model = pipeline.fit(train)

print("Predicting on test set...")
predictions = model.transform(test)

# ---------- Evaluate ----------
evaluator_mae = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="mae")
evaluator_rmse = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="r2")

mae = evaluator_mae.evaluate(predictions)
rmse = evaluator_rmse.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

print("\n=== DYNAMIC PRICING MODEL PERFORMANCE ===")
print(f"MAE  (avg price off by, in R$) : {mae:.3f}")
print(f"RMSE                              : {rmse:.3f}")
print(f"R2 Score (0-1, higher better)    : {r2:.4f}")

# ---------- Pricing recommendations ----------
recommendations = predictions.withColumn(
    "adjustment_factor", F.round(F.col("prediction") / F.col("price"), 4)
).select(
    "product_id", "price", "prediction", "adjustment_factor",
    "customer_state", "delivery_days"
).withColumnRenamed("prediction", "predicted_price")

recommendations.show(10)

recommendations.write.mode("overwrite").parquet(f"{GOLD}/pricing_recommendations")
model.write().overwrite().save("hdfs://master:9000/retail_project/models/dynamic_pricing_gbt_model")

print("DONE: pricing_recommendations + model saved")
spark.stop()
