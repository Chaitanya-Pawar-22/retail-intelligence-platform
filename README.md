# Retail Intelligence Platform

A Big Data pipeline built on a self-managed 2-node Hadoop/Spark cluster (VirtualBox), implementing a
medallion architecture (Bronze → Silver → Gold) on the Olist Brazilian E-commerce dataset.

## Architecture
- **Cluster**: 2-node Hadoop 2.7.3 cluster (Master: NameNode + ResourceManager, Worker: DataNode + NodeManager)
- **Processing**: Apache Spark 3.3.4 (PySpark), submitted via YARN
- **Storage**: HDFS, source data staged from AWS S3
- **Data flow**: S3 → Bronze (raw CSV) → Silver (cleaned Parquet fact/dim tables) → Gold (business/ML-ready Parquet)

## Pipeline Scripts (`scripts/`)
| Script | Purpose |
|---|---|
| `silver_layer.py` | Cleans Bronze CSVs, joins into fact_order_items + dimension tables |
| `gold_pricing.py` | Builds `order_features_for_pricing` gold table |
| `gold_ml_features.py` | Builds `ml_feature_store_v2` gold table (shipment delay features) |
| `gold_customer_profile.py` | Builds `customer_spending_profile` gold table |

## Gold Layer Tables
- `daily_sales_by_category` — demand forecasting input
- `order_features_for_pricing` — dynamic pricing input
- `ml_feature_store_v2` — shipment delay prediction input
- `customer_spending_profile` — customer analytics

## Status
- [x] Cluster setup
- [x] Bronze layer ingestion
- [x] Silver layer ETL
- [x] Gold layer ETL
- [ ] Demand forecasting + recommendations
- [ ] Shipment delay + dynamic pricing models
- [ ] Streamlit dashboard

## How to Run
```bash
spark-submit --master yarn scripts/silver_layer.py
spark-submit --master yarn --driver-memory 512m --executor-memory 512m --num-executors 1 scripts/gold_pricing.py
```
