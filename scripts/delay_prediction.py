from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("DelayPrediction").getOrCreate()

GOLD = "hdfs://master:9000/retail_project/data/gold"

df = spark.read.parquet(f"{GOLD}/ml_feature_store_v2")
print("Total rows loaded:", df.count())

# Drop rows with nulls in target or key numeric features
df = df.dropna(subset=[
    "delay_vs_estimate_days", "estimated_delivery_days", "product_weight_g",
    "product_volume", "price_per_item", "freight_per_item", "total_order_value"
])
print("Rows after dropping nulls:", df.count())

# ---------- Encode categorical columns ----------
categorical_cols = ["customer_state", "seller_state"]
indexers = [
    StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep")
    for c in categorical_cols
]

numeric_cols = [
    "estimated_delivery_days", "purchase_month", "purchase_day", "purchase_weekday",
    "purchase_hour", "is_weekend_order", "product_weight_g", "product_length_cm",
    "product_width_cm", "product_height_cm", "product_volume", "product_photos_qty",
    "is_same_state", "price_per_item", "freight_per_item", "total_items_in_order",
    "max_installments", "total_payment", "total_order_value", "freight_price_ratio",
    "weight_per_item"
]

feature_cols = numeric_cols + [c + "_idx" for c in categorical_cols]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")

gbt = GBTRegressor(labelCol="delay_vs_estimate_days", featuresCol="features", maxIter=50, maxDepth=5)

pipeline = Pipeline(stages=indexers + [assembler, gbt])

# ---------- Train/Test split ----------
train, test = df.randomSplit([0.8, 0.2], seed=42)
print(f"Train rows: {train.count()}   Test rows: {test.count()}")

print("Training GBTRegressor...")
model = pipeline.fit(train)

print("Predicting on test set...")
predictions = model.transform(test)

# ---------- Evaluate ----------
evaluator_mae = RegressionEvaluator(labelCol="delay_vs_estimate_days", predictionCol="prediction", metricName="mae")
evaluator_rmse = RegressionEvaluator(labelCol="delay_vs_estimate_days", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="delay_vs_estimate_days", predictionCol="prediction", metricName="r2")

mae = evaluator_mae.evaluate(predictions)
rmse = evaluator_rmse.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

print("\n=== SHIPMENT DELAY MODEL PERFORMANCE ===")
print(f"MAE  (avg days off)  : {mae:.3f}")
print(f"RMSE                  : {rmse:.3f}")
print(f"R2 Score (0-1, higher better) : {r2:.4f}")

predictions.select("delay_vs_estimate_days", "prediction").show(10)

# ---------- Save predictions + model ----------
predictions.select(
    "delay_vs_estimate_days", "prediction", "customer_state", "seller_state",
    "total_order_value", "estimated_delivery_days"
).write.mode("overwrite").parquet(f"{GOLD}/shipment_delay_predictions")

model.write().overwrite().save("hdfs://master:9000/retail_project/models/shipment_delay_gbt_model")

print("DONE: shipment_delay_predictions + model saved")
spark.stop()
