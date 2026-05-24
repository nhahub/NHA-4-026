from pyspark.sql import SparkSession
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Spark session
spark = SparkSession.builder \
    .appName("Olist_Bronze_Ingestion") \
    .master("spark://spark:7077") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

log.info("Spark session started successfully")

# File definitions

files = [
    ("eg_customers_dataset.csv",            "/olist/bronze/customers"),
    ("eg_geolocation_dataset.csv",          "/olist/bronze/geolocation"),
    ("eg_order_items_dataset.csv",          "/olist/bronze/order_items"),
    ("eg_order_payments_dataset.csv",       "/olist/bronze/order_payments"),
    ("eg_order_reviews_dataset.csv",        "/olist/bronze/order_reviews"),
    ("eg_orders_dataset.csv",               "/olist/bronze/orders"),
    ("eg_product_category_translation.csv", "/olist/bronze/product_category_translation"),
    ("eg_products_dataset.csv",             "/olist/bronze/products"),
    ("eg_sellers_dataset.csv",              "/olist/bronze/sellers"),
]

# Ingest each file
for csv_file, hdfs_path in files:
    
    local_path = f"file:///data/data/{csv_file}"
    log.info(f"Reading: {local_path}")

    df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .option("encoding", "UTF-8") \
        .csv(local_path)

    row_count = df.count()
    log.info(f"  Rows read    : {row_count}")
    log.info(f"  Columns      : {df.columns}")

    
    df.write \
        .mode("overwrite") \
        .parquet(hdfs_path)

    log.info(f"  Written to HDFS: {hdfs_path}")

log.info("Bronze ingestion completed for all 9 files")

# Stop Spark
spark.stop()