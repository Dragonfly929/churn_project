"""Stage 1 (optional) — the SAME extraction in Spark SQL ("the query in Spark").

PostgreSQL DWH : Spark reads every star-schema table over JDBC
                 (the driver is downloaded automatically through spark.jars.packages).
SQLite DWH     : the tables are read with pandas and registered as Spark views
                 (no JDBC driver needed).

  python stage1/spark_query.py                      # writes data/datasets/bank_spark.csv
  CHURN_DB=postgres python stage1/spark_query.py
Then it compares the Spark result with the SQL result (parity).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from stage1 import db  # noqa: E402

TABLES = ["dim_client", "dim_date", "dim_channel", "fact_account_month", "fact_transaction",
          "fact_feedback", "fact_survey", "fact_client_exit"]


def main(source: str = "bank"):
    from pyspark.sql import SparkSession

    def builder():
        return SparkSession.builder.appName("churn_stage1_query").master("local[*]") \
            .config("spark.sql.session.timeZone", "UTC").config("spark.ui.showConsoleProgress", "false")
    use_jdbc = config.DB_BACKEND == "postgres"
    spark = None
    if use_jdbc:
        try:
            spark = builder().config("spark.jars.packages", "org.postgresql:postgresql:42.7.4").getOrCreate()
        except Exception as e:  # noqa: BLE001  (no internet for Maven -> read the tables with pandas instead)
            print(f"JDBC driver download failed ({str(e)[:80]}...) -> reading tables through pandas")
            use_jdbc = False
    if spark is None:
        b = builder()
        b._options.pop("spark.jars.packages", None)  # the builder is shared: drop the failed package option
        spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    for t in TABLES:
        if use_jdbc:
            url = config.PG_URL.replace("postgresql+psycopg://", "")
            creds, hostdb = url.split("@")
            user, pwd = creds.split(":")
            sdf = (spark.read.format("jdbc")
                   .option("url", f"jdbc:postgresql://{hostdb}?currentSchema={source}")
                   .option("dbtable", t).option("user", user).option("password", pwd)
                   .option("driver", "org.postgresql.Driver").load())
        else:
            pdf = db.read_sql(f"SELECT * FROM {t}", source=source)
            for c in ("full_date", "birth_date", "join_date"):
                if c in pdf.columns:
                    pdf[c] = pd.to_datetime(pdf[c]).dt.date
            for c in ("tx_ts", "msg_ts"):
                if c in pdf.columns:
                    pdf[c] = pd.to_datetime(pdf[c])
            pdf = pdf.astype({c: "object" for c in pdf.columns if pdf[c].dtype == "string"})
            sdf = spark.createDataFrame(pdf.where(pdf.notna(), None))
        sdf.createOrReplaceTempView(t)
        print(f"  view {t}")

    q = (config.SQL_DIR / "query_features_spark.sql").read_text(encoding="utf-8")
    q = "\n".join(l for l in q.splitlines() if not l.strip().startswith("--"))
    args = {"cutoff": config.CUTOFF, "label_cutoff": config.LABEL_CUTOFF,
            "feature_cutoff": config.CUTOFF, "denoise": 1}
    out = spark.sql(q, args=args).toPandas()
    out.to_csv(config.DATASETS / f"{source}_spark.csv", index=False)
    print(f"Spark dataset: {out.shape}, churn share {out['churn'].mean():.3f}")

    ref = pd.read_csv(config.DATASETS / f"{source}_{config.CUTOFF}.csv")
    bad = []
    for c in ref.columns:
        a, b = ref[c], out[c]
        if pd.api.types.is_numeric_dtype(a):
            if (~np.isclose(a.fillna(-999).astype(float), pd.to_numeric(b).fillna(-999).astype(float), atol=1e-3)).sum():
                bad.append(c)
        elif (a.astype(str) != b.astype(str)).sum():
            bad.append(c)
    print("PARITY Spark vs SQL:", "OK - identical" if not bad else f"differences in {bad}")
    spark.stop()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bank")
