from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType, LongType
import pandas as pd
import numpy as np

spark = SparkSession.builder.appName("DemandForecasting").getOrCreate()

GOLD = "hdfs://master:9000/retail_project/data/gold"

daily_sales = spark.read.parquet(f"{GOLD}/daily_sales_by_category") \
    .select("sale_date", "category_english", "num_orders") \
    .dropna()

forecast_schema = StructType([
    StructField("category_english", StringType()),
    StructField("ds", TimestampType()),
    StructField("yhat", DoubleType()),
    StructField("yhat_lower", DoubleType()),
    StructField("yhat_upper", DoubleType()),
])

eval_schema = StructType([
    StructField("category_english", StringType()),
    StructField("test_points", LongType()),
    StructField("mae", DoubleType()),
    StructField("rmse", DoubleType()),
])

TEST_WEEKS = 8  # hold out last 8 weeks as test set

def forecast_category(pdf: pd.DataFrame) -> pd.DataFrame:
    from prophet import Prophet
    category = pdf["category_english"].iloc[0]

    cat_df = pdf[["sale_date", "num_orders"]].rename(columns={"sale_date": "ds", "num_orders": "y"})
    cat_df["ds"] = pd.to_datetime(cat_df["ds"])
    cat_df = cat_df.groupby("ds", as_index=False)["y"].sum().sort_values("ds")

    if len(cat_df) < 30:
        return pd.DataFrame(columns=["category_english", "ds", "yhat", "yhat_lower", "yhat_upper"])

    m = Prophet()
    m.fit(cat_df)
    future = m.make_future_dataframe(periods=12, freq="W")
    forecast = m.predict(future)

    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    result["category_english"] = category
    return result[["category_english", "ds", "yhat", "yhat_lower", "yhat_upper"]]

def evaluate_category(pdf: pd.DataFrame) -> pd.DataFrame:
    """Backtest: train on all but last TEST_WEEKS, predict, compare to actuals."""
    from prophet import Prophet
    category = pdf["category_english"].iloc[0]

    cat_df = pdf[["sale_date", "num_orders"]].rename(columns={"sale_date": "ds", "num_orders": "y"})
    cat_df["ds"] = pd.to_datetime(cat_df["ds"])
    cat_df = cat_df.groupby("ds", as_index=False)["y"].sum().sort_values("ds")

    # Aggregate to weekly so it lines up with our weekly forecast horizon
    weekly = cat_df.set_index("ds").resample("W")["y"].sum().reset_index()

    if len(weekly) < (TEST_WEEKS + 15):  # need enough history to bother training
        return pd.DataFrame(columns=["category_english", "test_points", "mae", "rmse"])

    train = weekly.iloc[:-TEST_WEEKS]
    test = weekly.iloc[-TEST_WEEKS:]

    m = Prophet()
    m.fit(train)
    future = m.make_future_dataframe(periods=TEST_WEEKS, freq="W")
    forecast = m.predict(future)

    pred = forecast.set_index("ds").loc[test["ds"], "yhat"].values
    actual = test["y"].values

    mae = float(np.mean(np.abs(pred - actual)))
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))

    return pd.DataFrame([{
        "category_english": category,
        "test_points": len(test),
        "mae": mae,
        "rmse": rmse
    }])

# ---------- Full forecast (for actual future predictions) ----------
all_forecasts = daily_sales.groupBy("category_english").applyInPandas(
    forecast_category, schema=forecast_schema
)
print("Writing forecasts...")
all_forecasts.write.mode("overwrite").parquet(f"{GOLD}/forecast_weekly_demand")
print("DONE: forecast_weekly_demand")

# ---------- Evaluation (backtest on held-out weeks) ----------
print("\nRunning backtest evaluation (train/test split per category)...")
eval_results = daily_sales.groupBy("category_english").applyInPandas(
    evaluate_category, schema=eval_schema
)
eval_results.cache()

n_evaluated = eval_results.count()
print(f"Categories evaluated: {n_evaluated}")

if n_evaluated > 0:
    import pyspark.sql.functions as F
    summary = eval_results.agg(
        F.avg("mae").alias("avg_mae"),
        F.avg("rmse").alias("avg_rmse")
    ).collect()[0]
    print(f"\n=== FORECAST ACCURACY (averaged across categories) ===")
    print(f"Average MAE  (avg orders off by) : {summary['avg_mae']:.2f}")
    print(f"Average RMSE                        : {summary['avg_rmse']:.2f}")
    print("\nPer-category breakdown (worst 10 by MAE):")
    eval_results.orderBy(F.col("mae").desc()).show(10, truncate=False)

eval_results.write.mode("overwrite").parquet(f"{GOLD}/forecast_evaluation_metrics")
print("DONE: forecast_evaluation_metrics")

spark.stop()
