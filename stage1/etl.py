"""Steps 1.5 / 2.0 — Source systems -> STAGING -> ETL/cleaning -> DWH (star schema).

  python stage1/etl.py                  # main bank   (data/raw)        -> source 'bank'
  python stage1/etl.py --source branch  # external branch (data/raw_branch) -> source 'branch'

Engine: CHURN_DB=sqlite (default) or CHURN_DB=postgres
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from ds.cleaning import clean_sources  # noqa: E402
from stage1 import db  # noqa: E402

SOURCES = ["core_clients", "core_accounts_monthly", "core_exits", "card_transactions",
           "contact_messages", "survey_responses"]
CHANNELS = [  # key, name, group, digital
    (1, "mobile_app", "payments", 1), (2, "web", "payments", 1), (3, "card_pos", "payments", 0),
    (4, "atm", "payments", 0), (5, "branch", "payments", 0), (6, "transfer", "payments", 0),
    (7, "call", "contact", 0), (8, "chat", "contact", 1), (9, "email", "contact", 1),
    (10, "app_review", "contact", 1)]


def date_key(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.strftime("%Y%m%d").astype("Int64")


def load_staging(raw_dir: Path, source: str) -> dict[str, pd.DataFrame]:
    """Raw loads, as they arrive: every column TEXT, nothing fixed."""
    raw = {}
    for name in SOURCES:
        df = pd.read_csv(raw_dir / f"{name}.csv", dtype=str, keep_default_na=False)
        df = df.replace({"": None})
        raw[name] = df
        cols = ", ".join(f"{c} TEXT" for c in df.columns)
        db.run_script(f"DROP TABLE IF EXISTS stg_{name}; CREATE TABLE stg_{name} ({cols})", source)
        db.bulk_insert(df, f"stg_{name}", source)
        print(f"  staging  stg_{name:24s} {len(df):>9,d} rows")
    return raw


def build_dwh(clean: dict[str, pd.DataFrame], source: str) -> None:
    db.run_script(db.sql_file("schema"), source)
    def iso(s):
        return pd.to_datetime(s).dt.strftime("%Y-%m-%d")

    # dim_date
    dates = pd.date_range("2010-01-01", "2026-12-31", freq="D")
    dd = pd.DataFrame({"date_key": dates.strftime("%Y%m%d").astype(int),
                       "full_date": dates.strftime("%Y-%m-%d"), "year": dates.year,
                       "quarter": dates.quarter, "month": dates.month, "day": dates.day,
                       "day_of_week": dates.dayofweek, "is_month_end": dates.is_month_end.astype(int)})
    db.bulk_insert(dd, "dim_date", source)
    ch = pd.DataFrame(CHANNELS, columns=["channel_key", "channel_name", "channel_group", "is_digital"])
    db.bulk_insert(ch, "dim_channel", source)
    chmap = dict(zip(ch.channel_name, ch.channel_key))

    c = clean["clients"].dropna(subset=["client_id", "join_date"]).copy()
    c = c[c["join_date"] >= "2010-01-01"]
    dc = pd.DataFrame({"client_key": c["client_id"].astype(int), "client_id": c["client_id"].astype(int),
                       "birth_date": pd.to_datetime(c["birth_date"]).dt.strftime("%Y-%m-%d"),
                       "gender": c["gender"], "country": c["country"],
                       "join_date": iso(c["join_date"]), "marketing_opt_out": c["marketing_opt_out"]})
    db.bulk_insert(dc, "dim_client", source)
    known = set(dc["client_key"])

    b = clean["balances"]
    b = b[b["client_id"].isin(known)]
    fa = pd.DataFrame({"client_key": b["client_id"].astype(int), "date_key": date_key(b["month_end"]),
                       "balance": b["balance"].round(2), "num_products": b["num_products"],
                       "credit_score": b["credit_score"].round(1), "has_card": b["has_card"],
                       "is_active_member": b["is_active_member"], "est_salary": b["est_salary"],
                       "status": b["status"]})
    db.bulk_insert(fa, "fact_account_month", source)

    t = clean["transactions"]
    t = t[t["client_id"].isin(known) & t["ts"].notna()]
    ft = pd.DataFrame({"tx_id": t["tx_id"].astype(int), "client_key": t["client_id"].astype(int),
                       "date_key": date_key(t["ts"]), "channel_key": t["channel"].map(chmap),
                       "tx_ts": t["ts"].dt.strftime("%Y-%m-%d %H:%M:%S"), "amount": t["amount"].round(2),
                       "direction": t["direction"]})
    db.bulk_insert(ft, "fact_transaction", source)

    m = clean["messages"]
    m = m[m["client_id"].isin(known) & m["ts"].notna()]
    ff = pd.DataFrame({"msg_id": m["msg_id"].astype(int), "client_key": m["client_id"].astype(int),
                       "date_key": date_key(m["ts"]), "channel_key": m["channel"].map(chmap),
                       "msg_ts": m["ts"].dt.strftime("%Y-%m-%d %H:%M:%S"), "text_raw": m["text"],
                       "text_clean": m["text_clean"], "sentiment_raw": m["sentiment_raw"],
                       "sentiment_score": m["sentiment_score"], "is_negative": m["is_negative"],
                       "is_complaint": m["complaint_flag"], "is_empty": m["is_empty"]})
    db.bulk_insert(ff, "fact_feedback", source)

    s = clean["survey"]
    s = s[s["client_id"].isin(known)]
    fs = pd.DataFrame({"survey_id": s["survey_id"].astype(int), "client_key": s["client_id"].astype(int),
                       "sent_date_key": date_key(s["sent_date"]),
                       "answered_date_key": date_key(s["answered_date"]),
                       "nps_score": s["nps_score"].astype("Int64")})
    db.bulk_insert(fs, "fact_survey", source)

    e = clean["exits"]
    e = e[e["client_id"].isin(known)]
    fe = pd.DataFrame({"client_key": e["client_id"].astype(int), "exit_date_key": date_key(e["exit_date"]),
                       "exit_channel_key": e["exit_channel"].map(chmap).astype("Int64")})
    db.bulk_insert(fe, "fact_client_exit", source)
    for name, df in [("dim_date", dd), ("dim_channel", ch), ("dim_client", dc), ("fact_account_month", fa),
                     ("fact_transaction", ft), ("fact_feedback", ff), ("fact_survey", fs),
                     ("fact_client_exit", fe)]:
        print(f"  dwh      {name:28s} {len(df):>9,d} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="bank", choices=["bank", "branch"])
    args = ap.parse_args()
    raw_dir = config.RAW if args.source == "bank" else config.RAW_BRANCH
    t0 = time.time()
    print(f"[ETL] engine={config.DB_BACKEND} source={args.source}")
    raw = load_staging(raw_dir, args.source)
    clean, log = clean_sources(raw, config.CUTOFF)
    log.to_csv(config.TABLES / f"cleaning_report_{args.source}.csv", index=False)
    print(log.to_string(index=False))
    build_dwh(clean, args.source)
    print(f"[ETL] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
