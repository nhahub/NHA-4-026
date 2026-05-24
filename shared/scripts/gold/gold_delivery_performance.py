from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, BooleanType
import logging
import subprocess

# Logging setup 
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

#  Spark session 
spark = SparkSession.builder \
    .appName("Olist_Gold_Delivery_Performance") \
    .master("spark://spark:7077") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
    .config("spark.jars", "/opt/spark/jars/postgresql-42.7.3.jar") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
log.info("Spark session started successfully")

#  PostgreSQL connection settings
PG_URL  = "jdbc:postgresql://postgres_general:5432/sessiondb"
PG_PROPS = {
    "user":     "admin",
    "password": "admin",
    "driver":   "org.postgresql.Driver"
}
SCHEMA = "delivery_performance"

#  Auto-truncate all tables before writing
log.info("Truncating all delivery_performance tables")
import psycopg2
_pg = psycopg2.connect(
    host="postgres_general", port=5432,
    dbname="sessiondb", user="admin", password="admin"
)
_pg.autocommit = True
_cur = _pg.cursor()
_cur.execute("TRUNCATE TABLE delivery_performance.fct_seller_fulfillment RESTART IDENTITY CASCADE")
_cur.execute("TRUNCATE TABLE delivery_performance.fct_order_delivery RESTART IDENTITY CASCADE")
_cur.execute("TRUNCATE TABLE delivery_performance.dim_product RESTART IDENTITY CASCADE")
_cur.execute("TRUNCATE TABLE delivery_performance.dim_seller RESTART IDENTITY CASCADE")
_cur.execute("TRUNCATE TABLE delivery_performance.dim_customer RESTART IDENTITY CASCADE")
_cur.execute("TRUNCATE TABLE delivery_performance.dim_date RESTART IDENTITY CASCADE")
_cur.close()
_pg.close()
log.info("All delivery_performance tables truncated successfully")

#  Fix HDFS permissions before writing
log.info("Fixing HDFS permissions")
hdfs_paths = [
    "/olist",
    "/olist/gold",
    "/olist/gold/dim_date",
    "/olist/gold/dim_customer",
    "/olist/gold/dim_seller",
    "/olist/gold/dim_product",
    "/olist/gold/fct_order_delivery",
    "/olist/gold/fct_seller_fulfillment",
]
for p in hdfs_paths:
    subprocess.run([
        "curl", "-s", "-X", "PUT",
        f"http://namenode:9870/webhdfs/v1{p}?op=SETPERMISSION&permission=777&user.name=root"
    ], capture_output=True)
log.info("HDFS permissions fixed")

#  Helpers
def read_silver(name):
    path = f"/olist/silver/{name}"
    log.info(f"Reading silver: {path}")
    return spark.read.parquet(path)

def write_pg(df, table):
    full_table = f"{SCHEMA}.{table}"
    log.info(f"Writing to PostgreSQL: {full_table}  rows={df.count()}")
    df.write.jdbc(url=PG_URL, table=full_table, mode="append", properties=PG_PROPS)
    log.info(f"Done: {full_table}")

def write_gold_hdfs(df, name):
    path = f"/olist/gold/{name}"
    df.write.mode("overwrite").parquet(path)
    log.info(f"Written to HDFS gold: {path}")


# READ ALL SILVER TABLES

customers   = read_silver("customers")
sellers     = read_silver("sellers")
products    = read_silver("products")
translation = read_silver("product_category_translation")
orders      = read_silver("orders")
order_items = read_silver("order_items")
geo         = read_silver("geolocation")

#  Geolocation average per zip
geo_avg = geo.groupBy("geolocation_zip_code_prefix").agg(
    F.round(F.avg("geolocation_lat"), 6).alias("latitude"),
    F.round(F.avg("geolocation_lng"), 6).alias("longitude")
)


# 1. dim_date

log.info("--- Building: dim_date ---")

date_bounds = orders.select(
    F.min(F.to_date("order_purchase_timestamp")).alias("min_date"),
    F.max(F.to_date("order_estimated_delivery_date")).alias("max_date")
).collect()[0]

min_date = str(date_bounds["min_date"])
max_date = str(date_bounds["max_date"])
log.info(f"Date spine: {min_date} to {max_date}")

date_df = spark.sql(f"""
    SELECT sequence(
        to_date('{min_date}'),
        to_date('{max_date}'),
        interval 1 day
    ) AS date_array
""").withColumn("full_date", F.explode(F.col("date_array"))).drop("date_array")

dim_date = date_df.select(
    F.col("full_date"),
    F.dayofmonth("full_date").alias("day_number"),
    F.date_format("full_date", "EEEE").alias("day_name"),
    F.weekofyear("full_date").alias("week_number"),
    F.month("full_date").alias("month_number"),
    F.date_format("full_date", "MMMM").alias("month_name"),
    F.quarter("full_date").alias("quarter_number"),
    F.year("full_date").alias("year_number"),
    F.when(F.dayofweek("full_date").isin(1, 7), True).otherwise(False).alias("is_weekend"),
    F.when(F.dayofmonth("full_date") == 1, True).otherwise(False).alias("is_month_start"),
    F.when(
        F.dayofmonth("full_date") == F.dayofmonth(F.last_day("full_date")),
        True
    ).otherwise(False).alias("is_month_end")
)

write_pg(dim_date, "dim_date")
write_gold_hdfs(dim_date, "dim_date")

dim_date_pg = spark.read.jdbc(
    url=PG_URL, table=f"{SCHEMA}.dim_date", properties=PG_PROPS
)


# 2. dim_customer

log.info("Building: dim_customer")

dim_customer = customers.join(
    geo_avg,
    customers["customer_zip_code_prefix"] == geo_avg["geolocation_zip_code_prefix"],
    how="left"
).select(
    F.col("customer_id"),
    F.col("customer_unique_id"),
    F.col("customer_zip_code_prefix"),
    F.col("customer_city"),
    F.col("customer_state"),
    F.col("customer_governorate").alias("customer_region"),
    F.col("latitude"),
    F.col("longitude"),
    F.current_timestamp().alias("created_at")
)

write_pg(dim_customer, "dim_customer")
write_gold_hdfs(dim_customer, "dim_customer")

dim_customer_pg = spark.read.jdbc(
    url=PG_URL, table=f"{SCHEMA}.dim_customer", properties=PG_PROPS
)


# 3. dim_seller

log.info("Building: dim_seller")

dim_seller = sellers.join(
    geo_avg,
    sellers["seller_zip_code_prefix"] == geo_avg["geolocation_zip_code_prefix"],
    how="left"
).select(
    F.col("seller_id"),
    F.col("seller_zip_code_prefix"),
    F.col("seller_city"),
    F.col("seller_state"),
    F.col("seller_governorate").alias("seller_region"),
    F.col("latitude"),
    F.col("longitude"),
    F.current_timestamp().alias("created_at")
)

write_pg(dim_seller, "dim_seller")
write_gold_hdfs(dim_seller, "dim_seller")

dim_seller_pg = spark.read.jdbc(
    url=PG_URL, table=f"{SCHEMA}.dim_seller", properties=PG_PROPS
)


# 4. dim_product

log.info("Building: dim_product")

dim_product = products.join(
    translation,
    products["product_category_name"] == translation["product_category_name"],
    how="left"
).select(
    products["product_id"],
    products["product_category_name"],
    translation["product_category_name_english"].alias("product_category_english"),
    products["product_weight_g"],
    products["product_length_cm"],
    products["product_height_cm"],
    products["product_width_cm"],
    products["product_volume_cm3"],
    products["product_photos_qty"].cast(IntegerType()),
    products["product_name_length"].cast(IntegerType()),
    products["product_description_length"].cast(IntegerType()),
    F.when(F.col("product_volume_cm3") <= 1000,  "small")
     .when(F.col("product_volume_cm3") <= 10000, "medium")
     .when(F.col("product_volume_cm3") <= 50000, "large")
     .otherwise("extra_large").alias("logistics_size_category"),
    F.when(F.col("product_weight_g") <= 500,   "light")
     .when(F.col("product_weight_g") <= 2000,  "medium")
     .when(F.col("product_weight_g") <= 10000, "heavy")
     .otherwise("very_heavy").alias("logistics_weight_category"),
    F.current_timestamp().alias("created_at")
)

write_pg(dim_product, "dim_product")
write_gold_hdfs(dim_product, "dim_product")

dim_product_pg = spark.read.jdbc(
    url=PG_URL, table=f"{SCHEMA}.dim_product", properties=PG_PROPS
)

#  Date SK lookup helper
def get_date_sk(df, date_col, sk_alias):
    return df.join(
        dim_date_pg.select(
            F.col("date_sk").alias(sk_alias),
            F.col("full_date").alias(f"_match_{sk_alias}")
        ),
        F.to_date(F.col(date_col)) == F.col(f"_match_{sk_alias}"),
        how="left"
    ).drop(f"_match_{sk_alias}")

# Haversine distance
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1_r = F.radians(lat1)
    lat2_r = F.radians(lat2)
    dlat   = F.radians(lat2 - lat1)
    dlon   = F.radians(lon2 - lon1)
    a = F.sin(dlat / 2) ** 2 + F.cos(lat1_r) * F.cos(lat2_r) * F.sin(dlon / 2) ** 2
    return F.round(R * 2 * F.asin(F.sqrt(a)), 2)


# 5. fct_order_delivery

log.info("Building: fct_order_delivery")

items_agg = order_items.groupBy("order_id").agg(
    F.round(F.sum("freight_value_egp"), 2).alias("freight_total_value"),
    F.count("order_item_id").alias("total_items_count"),
    F.countDistinct("seller_id").alias("seller_count"),
    F.first("seller_id").alias("seller_id")
)

fct_delivery = orders.join(items_agg, on="order_id", how="inner")

fct_delivery = fct_delivery.join(
    dim_customer_pg.select("customer_sk", "customer_id", "latitude", "longitude"),
    on="customer_id", how="left"
).withColumnRenamed("customer_sk", "customer_sk_fk") \
 .withColumnRenamed("latitude",    "customer_lat") \
 .withColumnRenamed("longitude",   "customer_lon")

fct_delivery = fct_delivery.join(
    dim_seller_pg.select("seller_sk", "seller_id", "latitude", "longitude"),
    on="seller_id", how="left"
).withColumnRenamed("seller_sk",  "seller_sk_fk") \
 .withColumnRenamed("latitude",   "seller_lat") \
 .withColumnRenamed("longitude",  "seller_lon")

fct_delivery = get_date_sk(fct_delivery, "order_purchase_timestamp",     "purchase_date_sk")
fct_delivery = get_date_sk(fct_delivery, "order_estimated_delivery_date", "estimated_delivery_date_sk")
fct_delivery = get_date_sk(fct_delivery, "order_delivered_customer_date", "actual_delivery_date_sk")

fct_delivery = fct_delivery \
    .withColumn("delivery_duration_days",
        F.when(F.col("order_delivered_customer_date").isNotNull(),
            F.datediff("order_delivered_customer_date", "order_purchase_timestamp")
        ).otherwise(None).cast(IntegerType())) \
    .withColumn("buffer_days",
        F.when(F.col("order_delivered_customer_date").isNotNull(),
            F.datediff("order_estimated_delivery_date", "order_delivered_customer_date")
        ).otherwise(None).cast(IntegerType())) \
    .withColumn("delay_days",
        F.when(F.col("order_delivered_customer_date").isNotNull(),
            F.datediff("order_delivered_customer_date", "order_estimated_delivery_date")
        ).otherwise(None).cast(IntegerType())) \
    .withColumn("on_time_flag",
        F.when(F.col("delay_days") <= 0, True)
         .when(F.col("delay_days").isNull(), None)
         .otherwise(False).cast(BooleanType())) \
    .withColumn("delivery_status",
        F.when(F.col("delay_days") <= 0,  "on_time")
         .when(F.col("delay_days") <= 7,  "late")
         .when(F.col("delay_days").isNull(), "not_delivered")
         .otherwise("very_late")) \
    .withColumn("is_multi_seller_order",
        F.when(F.col("seller_count") > 1, True).otherwise(False).cast(BooleanType())) \
    .withColumn("distance_km",
        haversine_km(
            F.col("seller_lat"), F.col("seller_lon"),
            F.col("customer_lat"), F.col("customer_lon")
        )) \
    .withColumn("distance_bucket",
        F.when(F.col("distance_km").isNull(),  "unknown")
         .when(F.col("distance_km") <= 50,     "0-50km")
         .when(F.col("distance_km") <= 200,    "50-200km")
         .when(F.col("distance_km") <= 500,    "200-500km")
         .otherwise("500km+"))

fct_delivery_final = fct_delivery.select(
    F.col("order_id"),
    F.col("customer_sk_fk"),
    F.col("seller_sk_fk"),
    F.col("purchase_date_sk"),
    F.col("estimated_delivery_date_sk"),
    F.col("actual_delivery_date_sk"),
    F.col("order_status"),
    F.col("order_purchase_timestamp").alias("purchase_timestamp"),
    F.col("order_estimated_delivery_date").alias("estimated_delivery_date"),
    F.col("order_delivered_customer_date").alias("actual_delivery_date"),
    F.col("order_delivered_carrier_date").alias("carrier_handoff_date"),
    F.col("delivery_duration_days"),
    F.col("buffer_days"),
    F.col("delay_days"),
    F.col("delivery_status"),
    F.col("distance_bucket"),
    F.col("freight_total_value"),
    F.col("total_items_count"),
    F.col("seller_count"),
    F.col("is_multi_seller_order"),
    F.col("on_time_flag"),
    F.current_timestamp().alias("created_at")
)

write_pg(fct_delivery_final, "fct_order_delivery")
write_gold_hdfs(fct_delivery_final, "fct_order_delivery")


# 6. fct_seller_fulfillment

log.info("Building: fct_seller_fulfillment")

fct_fulfillment = order_items.join(
    orders.select("order_id", "customer_id", "order_purchase_timestamp"),
    on="order_id", how="inner"
)

fct_fulfillment = fct_fulfillment.join(
    dim_seller_pg.select("seller_sk", "seller_id",
                         F.col("latitude").alias("seller_lat"),
                         F.col("longitude").alias("seller_lon")),
    on="seller_id", how="left"
).withColumnRenamed("seller_sk", "seller_sk_fk")

fct_fulfillment = fct_fulfillment.join(
    dim_product_pg.select("product_sk", "product_id",
                          F.col("product_weight_g").alias("prod_weight"),
                          F.col("product_volume_cm3").alias("prod_volume")),
    on="product_id", how="left"
).withColumnRenamed("product_sk", "product_sk_fk")

fct_fulfillment = fct_fulfillment.join(
    dim_customer_pg.select("customer_sk", "customer_id",
                           F.col("latitude").alias("customer_lat"),
                           F.col("longitude").alias("customer_lon")),
    on="customer_id", how="left"
).withColumnRenamed("customer_sk", "customer_sk_fk")

fct_fulfillment = get_date_sk(fct_fulfillment, "order_purchase_timestamp", "purchase_date_sk")
fct_fulfillment = get_date_sk(fct_fulfillment, "shipping_limit_date",      "shipping_limit_date_sk")

fct_fulfillment = fct_fulfillment \
    .withColumn("seller_preparation_days",
        F.datediff("shipping_limit_date", "order_purchase_timestamp").cast(IntegerType())) \
    .withColumn("shipping_deadline_gap_days",
        F.datediff("shipping_limit_date", "order_purchase_timestamp").cast(IntegerType())) \
    .withColumn("freight_ratio",
        F.when(F.col("price_egp") > 0,
            F.round(F.col("freight_value_egp") / F.col("price_egp"), 4)
        ).otherwise(None)) \
    .withColumn("heavy_product_flag",
        F.when(F.col("prod_weight") > 10000, True).otherwise(False).cast(BooleanType())) \
    .withColumn("oversized_product_flag",
        F.when(F.col("prod_volume") > 50000, True).otherwise(False).cast(BooleanType())) \
    .withColumn("seller_to_customer_distance_km",
        haversine_km(
            F.col("seller_lat"), F.col("seller_lon"),
            F.col("customer_lat"), F.col("customer_lon")
        ))

fct_fulfillment_final = fct_fulfillment.select(
    F.col("order_id"),
    F.col("order_item_id"),
    F.col("seller_sk_fk"),
    F.col("product_sk_fk"),
    F.col("customer_sk_fk"),
    F.col("purchase_date_sk"),
    F.col("shipping_limit_date_sk"),
    F.col("seller_id"),
    F.col("product_id"),
    F.col("shipping_limit_date"),
    F.col("freight_value_egp").alias("freight_value"),
    F.col("price_egp").alias("item_price"),
    F.col("prod_weight").alias("product_weight_g"),
    F.col("prod_volume").alias("product_volume_cm3"),
    F.col("seller_preparation_days"),
    F.col("shipping_deadline_gap_days"),
    F.col("freight_ratio"),
    F.col("heavy_product_flag"),
    F.col("oversized_product_flag"),
    F.col("seller_to_customer_distance_km"),
    F.current_timestamp().alias("created_at")
)

write_pg(fct_fulfillment_final, "fct_seller_fulfillment")
write_gold_hdfs(fct_fulfillment_final, "fct_seller_fulfillment")


log.info("Gold layer completed — all 6 tables written to PostgreSQL and HDFS")
spark.stop()