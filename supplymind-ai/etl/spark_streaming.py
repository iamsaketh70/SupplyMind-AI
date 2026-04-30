# ──────────────────────────────────────────────────────────────────────────────
# HOW TO RUN ON DATABRICKS COMMUNITY EDITION
# ──────────────────────────────────────────────────────────────────────────────
# 1. Create a new notebook in Databricks Community Edition (Runtime 13.x LTS).
# 2. Copy this entire file into a single cell (or split into cells as you like).
# 3. Replace KAFKA_BROKER with your Confluent Cloud or external Kafka endpoint;
#    Databricks Community Edition cannot reach localhost:9092.
# 4. Delta Lake is built in on Databricks — remove the delta-spark config lines.
# 5. Change DELTA_PATHS to "/mnt/delta/silver_*" or DBFS paths.
# 6. Comment out the local SparkSession builder and use the existing `spark`
#    variable that Databricks provides automatically.
# 7. Run the notebook — each readStream/writeStream pair runs as a separate
#    streaming query visible in the Spark UI under the Structured Streaming tab.
# ──────────────────────────────────────────────────────────────────────────────

"""PySpark Structured Streaming consumer — reads Kafka topics and writes to Delta Lake."""

import os
import sys
import re

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, from_json, lower, trim, regexp_replace,
    current_timestamp, length,
)
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import KAFKA_BROKER, TOPICS, DELTA_PATHS, CHECKPOINT_DIR


# ── Spark Session ────────────────────────────────────────────────────────────

def get_spark():
    """Build a local SparkSession with Delta Lake and Kafka support."""
    return (
        SparkSession.builder
        .appName("SupplyMind-AI-Streaming")
        .master("local[*]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.0,"
            "io.delta:delta-spark_2.13:4.0.0",
        )
        .getOrCreate()
    )


# ── Schemas ──────────────────────────────────────────────────────────────────

NEWS_SCHEMA = StructType([
    StructField("text", StringType(), True),
    StructField("source", StringType(), True),
    StructField("timestamp", DoubleType(), True),
    StructField("file", StringType(), True),
])

SOCIAL_SCHEMA = StructType([
    StructField("text", StringType(), True),
    StructField("source", StringType(), True),
    StructField("timestamp", DoubleType(), True),
    StructField("file", StringType(), True),
])

SUPPLY_SCHEMA = StructType([
    StructField("text", StringType(), True),
    StructField("source", StringType(), True),
    StructField("timestamp", DoubleType(), True),
    StructField("file", StringType(), True),
])


# ── Text cleaning ────────────────────────────────────────────────────────────

def clean_text_df(df):
    """Lowercase, strip URLs, remove special chars, trim whitespace, drop empties."""
    cleaned = (
        df
        .withColumn("text", lower(col("text")))
        .withColumn("text", regexp_replace(col("text"), r"http\S+", ""))
        .withColumn("text", regexp_replace(col("text"), r"[^a-z0-9\s.,!?'-]", " "))
        .withColumn("text", trim(col("text")))
        .filter(col("text").isNotNull())
        .filter(length(col("text")) > 2)
        .withColumn("ingested_at", current_timestamp())
    )
    return cleaned


# ── Stream builder ───────────────────────────────────────────────────────────

def start_stream(spark, topic, schema, delta_path, query_name):
    """Subscribe to a Kafka topic, clean data, and write to Delta Lake."""
    raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BROKER)
        .option("subscribe", topic)
        .option("startingOffsets", "earliest")
        .option("failOnDataLoss", "false")
        .load()
    )

    parsed = (
        raw
        .selectExpr("CAST(value AS STRING) as json_str")
        .select(from_json(col("json_str"), schema).alias("data"))
        .select("data.*")
    )

    cleaned = clean_text_df(parsed)

    checkpoint = os.path.join(CHECKPOINT_DIR, query_name)
    query = (
        cleaned.writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", checkpoint)
        .option("path", delta_path)
        .queryName(query_name)
        .start()
    )
    print(f"[ETL] Stream '{query_name}' started — {topic} → {delta_path}")
    return query


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    """Start all three Kafka → Delta Lake streaming queries."""
    print("=" * 60)
    print("  SupplyMind AI — PySpark Structured Streaming ETL")
    print("=" * 60)

    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")
    print("[ETL] SparkSession ready.")

    queries = []
    queries.append(start_stream(
        spark, TOPICS["news"], NEWS_SCHEMA,
        DELTA_PATHS["silver_news"], "silver_news_stream",
    ))
    queries.append(start_stream(
        spark, TOPICS["social"], SOCIAL_SCHEMA,
        DELTA_PATHS["silver_social"], "silver_social_stream",
    ))
    queries.append(start_stream(
        spark, TOPICS["supply"], SUPPLY_SCHEMA,
        DELTA_PATHS["silver_supply"], "silver_supply_stream",
    ))

    print("[ETL] All 3 streams running. Press Ctrl+C to stop.")
    try:
        spark.streams.awaitAnyTermination()
    except KeyboardInterrupt:
        print("[ETL] Shutting down streams ...")
        for q in queries:
            q.stop()
        spark.stop()
        print("[ETL] Stopped.")


if __name__ == "__main__":
    main()
