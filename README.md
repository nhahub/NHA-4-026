# EGY Ecommerce — Data Engineering Project

Full batch data pipeline on a local Docker Compose cluster.

## Architecture
CSV Files → Bronze (HDFS Parquet) → Silver (cleaned Parquet) → Gold (PostgreSQL) → Power BI

## Stack
- Apache Spark 3.5.0
- HDFS (Hadoop)
- Apache Airflow
- PostgreSQL
- Power BI Desktop
- Docker Compose (12 containers)

## Structure
dags/                        ← Airflow DAG
shared/scripts/bronze/       ← Bronze ingestion script
shared/scripts/silver/       ← Silver transformation script
shared/scripts/gold/         ← Gold layer scripts (2 datamarts)
docker-compose.yaml          ← Full infrastructure

## Datamarts
- **Delivery Performance** — 6 tables (dim_date, dim_customer, dim_seller, dim_product, fct_order_delivery, fct_seller_fulfillment)
- **Customer Churn** — 5 tables (dim_date, dim_product, dim_customer_profile, fct_customer_orders, fct_churn_summary)

## How to Run
1. `docker-compose up -d`
2. Open Airflow at http://localhost:8089
3. Trigger DAG: `olist_batch_pipeline`