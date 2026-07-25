from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType
import pandas as pd

spark = SparkSession.builder.appName("DemandForecasting").getOrCreate()

GOLD = "hdfs://master:9000/retail_project/data/gold"

# Load daily sales, keep only what we need
daily_sales = spark.read.parquet(f"{GOLD}/daily_sales_by_category") \
    .select("sale_date", "category_english", "num_orders") \
    .dropna()

# Schema for what each parallel Prophet run will return:
# one category's forecast, for the next 12 weeks
forecast_schema = StructType([
    StructField("category_english", StringType()),
    StructField("ds", TimestampType()),
    StructField("yhat", DoubleType()),
    StructField("yhat_lower", DoubleType()),
    StructField("yhat_upper", DoubleType()),
])

def forecast_category(pdf: pd.DataFrame) -> pd.DataFrame:
    """Runs on a Spark worker: trains one Prophet model for ONE category."""
    from prophet import Prophet

    category = pdf["category_english"].iloc[0]

    cat_df = pdf[["sale_date", "num_orders"]].rename(columns={"sale_date": "ds", "num_orders": "y"})
    cat_df["ds"] = pd.to_datetime(cat_df["ds"])
    cat_df = cat_df.groupby("ds", as_index=False)["y"].sum()  # in case of dup dates

    # Skip categories with too little history to train a meaningful model
    if len(cat_df) < 30:
        return pd.DataFrame(columns=["category_english", "ds", "yhat", "yhat_lower", "yhat_upper"])

    m = Prophet()
    m.fit(cat_df)

    future = m.make_future_dataframe(periods=12, freq="W")
    forecast = m.predict(future)

    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    result["category_english"] = category
    return result[["category_english", "ds", "yhat", "yhat_lower", "yhat_upper"]]

# This is the key line: Spark groups by category and runs forecast_category()
# on EACH group IN PARALLEL across Master + Worker executors.
all_forecasts = daily_sales.groupBy("category_english").applyInPandas(
    forecast_category, schema=forecast_schema
)

print("Writing forecasts...")
all_forecasts.write.mode("overwrite").parquet(f"{GOLD}/forecast_weekly_demand")
print("DONE: forecast_weekly_demand")

spark.stop()
