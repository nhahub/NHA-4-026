from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Spark session
spark = SparkSession.builder \
    .appName("Olist_Silver_Transformation") \
    .master("spark://spark:7077") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
log.info("Spark session started successfully")


# HELPER: read from Bronze HDFS

def read_bronze(name):
    path = f"/olist/bronze/{name}"
    log.info(f"Reading bronze: {path}")
    return spark.read.parquet(path)


# HELPER: write to Silver HDFS

def write_silver(df, name):
    path = f"/olist/silver/{name}"
    df.write.mode("overwrite").parquet(path)
    log.info(f"Written to silver: {path}  rows={df.count()}")


# 1. CUSTOMERS

log.info("Cleaning: customers")
customers = read_bronze("customers")

customers = (
    customers
    # Standardize city names: trim whitespace, lowercase
    .withColumn("customer_city",        F.lower(F.trim(F.col("customer_city"))))
    .withColumn("customer_state",       F.upper(F.trim(F.col("customer_state"))))
    .withColumn("customer_governorate", F.initcap(F.trim(F.col("customer_governorate"))))
    # Drop any fully duplicate rows
    .dropDuplicates()
    # Drop rows where primary key is null
    .filter(F.col("customer_id").isNotNull())
    .filter(F.col("customer_unique_id").isNotNull())
)

write_silver(customers, "customers")


# 2. GEOLOCATION

log.info("Cleaning: geolocation")
geo = read_bronze("geolocation")

geo = (
    geo
    .withColumn("geolocation_city",  F.lower(F.trim(F.col("geolocation_city"))))
    .withColumn("geolocation_state", F.upper(F.trim(F.col("geolocation_state"))))
    # Remove rows where lat or lng is null
    .filter(F.col("geolocation_lat").isNotNull())
    .filter(F.col("geolocation_lng").isNotNull())
    # Remove duplicate zip+lat+lng combinations
    .dropDuplicates(["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"])
)

write_silver(geo, "geolocation")


# 3. SELLERS

log.info("Cleaning: sellers")
sellers = read_bronze("sellers")

sellers = (
    sellers
    .withColumn("seller_city",       F.lower(F.trim(F.col("seller_city"))))
    .withColumn("seller_state",      F.upper(F.trim(F.col("seller_state"))))
    .withColumn("seller_governorate",F.initcap(F.trim(F.col("seller_governorate"))))
    .dropDuplicates()
    .filter(F.col("seller_id").isNotNull())
)

write_silver(sellers, "sellers")


# 4. PRODUCTS

log.info("Cleaning: products")
products = read_bronze("products")

# Drop rows where ALL product attributes are null
products = products.filter(F.col("product_category_name").isNotNull())

# Fix typos in column names: lenght -> length
products = (
    products
    .withColumnRenamed("product_name_lenght",        "product_name_length")
    .withColumnRenamed("product_description_lenght", "product_description_length")
)

# Compute median values for numeric columns to fill the 2 remaining nulls
median_weight = products.approxQuantile("product_weight_g",   [0.5], 0.01)[0]
median_length = products.approxQuantile("product_length_cm",  [0.5], 0.01)[0]
median_height = products.approxQuantile("product_height_cm",  [0.5], 0.01)[0]
median_width  = products.approxQuantile("product_width_cm",   [0.5], 0.01)[0]

products = (
    products
    .withColumn("product_weight_g",  F.when(F.col("product_weight_g").isNull(),  median_weight).otherwise(F.col("product_weight_g")))
    .withColumn("product_length_cm", F.when(F.col("product_length_cm").isNull(), median_length).otherwise(F.col("product_length_cm")))
    .withColumn("product_height_cm", F.when(F.col("product_height_cm").isNull(), median_height).otherwise(F.col("product_height_cm")))
    .withColumn("product_width_cm",  F.when(F.col("product_width_cm").isNull(),  median_width ).otherwise(F.col("product_width_cm")))
    # Compute volume
    .withColumn("product_volume_cm3",
        F.round(F.col("product_length_cm") * F.col("product_height_cm") * F.col("product_width_cm"), 2))
    .withColumn("product_category_name", F.lower(F.trim(F.col("product_category_name"))))
    .dropDuplicates()
    .filter(F.col("product_id").isNotNull())
)

write_silver(products, "products")


# 5. PRODUCT CATEGORY TRANSLATION

log.info("Cleaning: product_category_translation")
translation = read_bronze("product_category_translation")

# Add the 2 missing categories using pure Spark SQL
spark.sql("""
    CREATE OR REPLACE TEMP VIEW missing_cats AS
    SELECT 'pc_gamer' AS product_category_name,
           'pc_gamer' AS product_category_name_english,
           'العاب الكمبيوتر' AS product_category_name_arabic
    UNION ALL
    SELECT 'portateis_cozinha_e_preparadores_de_alimentos',
           'portable_kitchen',
           'ادوات المطبخ المحمولة'
""")

missing_rows = spark.sql("SELECT * FROM missing_cats")
translation = translation.union(missing_rows)

translation = (
    translation
    .withColumn("product_category_name",         F.lower(F.trim(F.col("product_category_name"))))
    .withColumn("product_category_name_english",  F.lower(F.trim(F.col("product_category_name_english"))))
    .dropDuplicates()
)

write_silver(translation, "product_category_translation")


# 6. ORDERS

log.info("Cleaning: orders")
orders = read_bronze("orders")

# Cast all timestamp columns to proper TimestampType
ts_cols = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]
for c in ts_cols:
    orders = orders.withColumn(c, F.to_timestamp(F.col(c)))

# Remove rows where purchase timestamp is null (essential anchor)
orders = orders.filter(F.col("order_purchase_timestamp").isNotNull())

# Fix chronological violations:
# Rule 1: carrier date must be >= approved_at  → null out the bad carrier date
orders = orders.withColumn(
    "order_delivered_carrier_date",
    F.when(
        F.col("order_delivered_carrier_date") < F.col("order_approved_at"),
        None
    ).otherwise(F.col("order_delivered_carrier_date"))
)

# Rule 2: customer delivery must be >= carrier date → null out the bad customer date
orders = orders.withColumn(
    "order_delivered_customer_date",
    F.when(
        F.col("order_delivered_customer_date") < F.col("order_delivered_carrier_date"),
        None
    ).otherwise(F.col("order_delivered_customer_date"))
)

# Standardize order_status
orders = orders.withColumn("order_status", F.lower(F.trim(F.col("order_status"))))

orders = orders.dropDuplicates().filter(F.col("order_id").isNotNull())

write_silver(orders, "orders")


# 7. ORDER ITEMS

log.info("Cleaning: order_items")
order_items = read_bronze("order_items")

order_items = (
    order_items
    .withColumn("shipping_limit_date", F.to_timestamp(F.col("shipping_limit_date")))
    # Price and freight must be positive
    .filter(F.col("price_egp") > 0)
    .filter(F.col("freight_value_egp") >= 0)
    .dropDuplicates()
    .filter(F.col("order_id").isNotNull())
    .filter(F.col("product_id").isNotNull())
    .filter(F.col("seller_id").isNotNull())
)

write_silver(order_items, "order_items")


# 8. ORDER PAYMENTS

log.info("Cleaning: order_payments")
payments = read_bronze("order_payments")

payments = (
    payments
    .withColumn("payment_type", F.lower(F.trim(F.col("payment_type"))))
    # Payment value must be positive
    .filter(F.col("payment_value_egp") > 0)
    .dropDuplicates()
    .filter(F.col("order_id").isNotNull())
)

write_silver(payments, "order_payments")

# 9. ORDER REVIEWS

log.info("Cleaning: order_reviews")
reviews = read_bronze("order_reviews")

reviews = (
    reviews
    .withColumn("review_creation_date",    F.to_timestamp(F.col("review_creation_date")))
    .withColumn("review_answer_timestamp", F.to_timestamp(F.col("review_answer_timestamp")))
    # Fill nulls with readable placeholders
    .withColumn("review_comment_title",
        F.when(F.col("review_comment_title").isNull(),   "No Title")
         .otherwise(F.trim(F.col("review_comment_title"))))
    .withColumn("review_comment_message",
        F.when(F.col("review_comment_message").isNull(), "No Comment")
         .otherwise(F.trim(F.col("review_comment_message"))))
    # Score must be between 1 and 5
    .filter(F.col("review_score").between(1, 5))
    .dropDuplicates()
    .filter(F.col("review_id").isNotNull())
    .filter(F.col("order_id").isNotNull())
)

write_silver(reviews, "order_reviews")


log.info("Silver transformation completed for all 9 files")
spark.stop()