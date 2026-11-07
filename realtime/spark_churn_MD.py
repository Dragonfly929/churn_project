"""SPARK STRUCTURED STREAMING (lesson 5: spark_trio_MD.py -> spark_churn_MD.py).

Reads bank.events and writes windowed aggregates to the console and to CSV ("the book of records"):
  1. processing-time windows of 10 seconds: events per source (how fast data arrives)
  2. event-time windows of ONE BANK DAY with a watermark: transactions, money in/out, messages
     per source and day; the watermark tells Spark when a day is complete (late events after it
     are dropped), which is exactly the moment the scorer's clock path rescores silent clients.
Output: realtime/output/spark/proc_10s/*.csv and realtime/output/spark/bank_day/*.csv (append mode)

  python realtime/spark_churn_MD.py                    # Kafka (downloads the spark-sql-kafka connector once)
  python realtime/spark_churn_MD.py --bus file         # reads the file-bus folder instead
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from realtime.bus import BUS_DIR, add_common_args  # noqa: E402

OUT = config.RT_OUT / "spark"


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--watermark", default="1 day", help="event-time lateness allowed")
    a = ap.parse_args()
    import pyspark
    from pyspark.sql import SparkSession, functions as F, types as T

    b = SparkSession.builder.appName("spark_churn_MD").master("local[1]") \
        .config("spark.sql.shuffle.partitions", "2").config("spark.ui.showConsoleProgress", "false") \
        .config("spark.sql.session.timeZone", "UTC")
    if a.bus == "kafka":
        b = b.config("spark.jars.packages", f"org.apache.spark:spark-sql-kafka-0-10_2.13:{pyspark.__version__}")
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    schema = T.StructType([T.StructField(n, t) for n, t in [
        ("type", T.StringType()), ("source", T.StringType()), ("client_id", T.LongType()), ("ts", T.StringType()),
        ("amount", T.DoubleType()), ("direction", T.StringType()), ("channel", T.StringType()),
        ("is_complaint", T.LongType()), ("day", T.StringType()), ("run_id", T.StringType()),
        ("sent_at", T.DoubleType())]])
    if a.bus == "kafka":
        raw = (spark.readStream.format("kafka").option("kafka.bootstrap.servers", a.bootstrap)
               .option("subscribe", "bank.events").option("startingOffsets", "latest").load())
        ev = raw.select(F.from_json(F.col("value").cast("string"), schema).alias("e")).select("e.*")
    else:
        ev = spark.readStream.schema(schema).option("maxFilesPerTrigger", 50).json(str(BUS_DIR / "bank.events"))
    ev = ev.withColumn("event_time", F.to_timestamp("ts")).withColumn("proc_time", F.current_timestamp())

    # 1. processing-time windows (10 s)
    proc = (ev.withWatermark("proc_time", "10 seconds")
            .groupBy(F.window("proc_time", "10 seconds"), "source").count()
            .select(F.col("window.start").alias("win_start"), F.col("window.end").alias("win_end"), "source", "count"))
    # 2. event-time windows (1 bank day) with watermark
    data = ev.filter(F.col("type") != "day_end")
    day = (data.withWatermark("event_time", a.watermark)
           .groupBy(F.window("event_time", "1 day"), "source")
           .agg(F.sum(F.when(F.col("type") == "transaction", 1).otherwise(0)).alias("transactions"),
                F.sum(F.when(F.col("direction") == "in", F.col("amount")).otherwise(0)).alias("amount_in"),
                F.sum(F.when(F.col("direction") == "out", F.col("amount")).otherwise(0)).alias("amount_out"),
                F.sum(F.when(F.col("type") == "message", 1).otherwise(0)).alias("messages"),
                F.sum(F.coalesce(F.col("is_complaint"), F.lit(0))).alias("complaints"),
                F.approx_count_distinct("client_id").alias("active_clients"))
           .select(F.date_format("window.start", "yyyy-MM-dd").alias("bank_day"), "source", "transactions",
                   F.round("amount_in", 2).alias("amount_in"), F.round("amount_out", 2).alias("amount_out"),
                   "messages", "complaints", "active_clients"))
    q = []
    for name, df in [("proc_10s", proc), ("bank_day", day)]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
        q.append(df.writeStream.outputMode("append").format("csv").option("header", True)
                 .option("path", str(OUT / name)).option("checkpointLocation", str(OUT / f"_chk_{name}"))
                 .trigger(processingTime="10 seconds").start())
    q.append(day.writeStream.outputMode("append").format("console").option("truncate", False)
             .trigger(processingTime="10 seconds").start())
    print("[spark] streaming queries started; CSV ->", OUT, flush=True)
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
