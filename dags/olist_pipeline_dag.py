from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner":           "olist_project",
    "depends_on_past": False,
    "start_date":      datetime(2024, 1, 1),
    "retries":         1,
    "retry_delay":     timedelta(minutes=5),
}

ENV_SETUP = "export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64"

SPARK_BASE = (
    "spark-submit "
    "--master spark://spark:7077 "
    "--conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000"
)

SPARK_GOLD = (
    "spark-submit "
    "--master spark://spark:7077 "
    "--conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 "
    "--jars /data/jars/postgresql-42.7.3.jar "
    "--conf spark.executor.extraClassPath=/data/jars/postgresql-42.7.3.jar "
    "--conf spark.driver.extraClassPath=/data/jars/postgresql-42.7.3.jar"
)

with DAG(
    dag_id="olist_batch_pipeline",
    default_args=default_args,
    description="Olist EG batch pipeline: Bronze -> Silver -> Gold",
    schedule_interval="@daily",
    catchup=False,
    tags=["olist", "batch", "delivery_performance"],
) as dag:

    bronze_task = BashOperator(
        task_id="bronze_ingestion",
        bash_command=f"{ENV_SETUP} && {SPARK_BASE} /data/scripts/bronze/bronze_ingestion.py",
        execution_timeout=timedelta(hours=1),
    )

    silver_task = BashOperator(
        task_id="silver_transformation",
        bash_command=f"{ENV_SETUP} && {SPARK_BASE} /data/scripts/silver/silver_transformation.py",
        execution_timeout=timedelta(hours=1),
    )

    gold_delivery_task = BashOperator(
        task_id="gold_delivery_performance",
        bash_command=f"{ENV_SETUP} && {SPARK_GOLD} /data/scripts/gold/gold_delivery_performance.py",
        execution_timeout=timedelta(hours=2),
    )

    gold_churn_task = BashOperator(
        task_id="gold_customer_churn",
        bash_command=f"{ENV_SETUP} && {SPARK_GOLD} /data/scripts/gold/gold_customer_churn.py",
        execution_timeout=timedelta(hours=2),
    )

    # Sequential: bronze → silver → delivery gold → churn gold
    bronze_task >> silver_task >> gold_delivery_task >> gold_churn_task