"""Step 2.0 — Data cleaning rules, written as code so they are applied
identically to the main bank AND to the external branch export (step 4, pass 2).

Each rule logs how many rows it touched -> reports/tables/cleaning_report_<source>.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ds.textproc import denoise, is_garbled, raw_sentiment, sentiment

COUNTRY_MAP = {"md": "MD", "moldova": "MD", "r. moldova": "MD", "republica moldova": "MD",
               "ro": "RO", "romania": "RO", "românia": "RO",
               "ua": "UA", "ukraine": "UA", "ucraina": "UA"}
GENDER_MAP = {"f": "F", "female": "F", "m": "M", "male": "M"}


def parse_dates(s: pd.Series) -> pd.Series:
    """ISO dates and dd.mm.yyyy (another branch's format) -> datetime."""
    s = s.astype("string")
    iso = pd.to_datetime(s, errors="coerce", format="ISO8601")
    dot = s.str.contains(r"^\d{2}\.\d{2}\.\d{4}", na=False)
    if dot.any():
        iso[dot] = pd.to_datetime(s[dot], errors="coerce", dayfirst=True, format="mixed")
    return iso


def to_number(s: pd.Series) -> pd.Series:
    """'12,50' (comma decimal) -> 12.5."""
    return pd.to_numeric(s.astype("string").str.replace(",", ".", regex=False).str.strip(),
                         errors="coerce")


class Log:
    def __init__(self):
        self.rows = []

    def add(self, table, rule, n, decision):
        self.rows.append({"table": table, "rule": rule, "rows_affected": int(n), "decision": decision})

    def df(self):
        return pd.DataFrame(self.rows)


def clean_sources(raw: dict[str, pd.DataFrame], ref_date: str) -> tuple[dict, pd.DataFrame]:
    log = Log()
    out = {}

    # ---------------- clients
    c = raw["core_clients"].copy()
    n0 = len(c)
    c = c.drop_duplicates("client_id")
    log.add("core_clients", "duplicate client_id", n0 - len(c), "keep first row")
    c["client_id"] = to_number(c["client_id"]).astype("Int64")
    before = c["country"].copy()
    c["country"] = c["country"].astype("string").str.strip().str.lower().map(COUNTRY_MAP)
    log.add("core_clients", "country name variants", (before.astype(str) != c["country"].astype(str)).sum(),
            "map to ISO-2 code (MD/RO/UA)")
    c["gender"] = c["gender"].astype("string").str.strip().str.lower().map(GENDER_MAP)
    c["birth_date"] = parse_dates(c["birth_date"])
    c["join_date"] = parse_dates(c["join_date"])
    age = (pd.Timestamp(ref_date) - c["birth_date"]).dt.days / 365.25
    bad = (age < 18) | (age > 100)
    c.loc[bad, "birth_date"] = pd.NaT
    log.add("core_clients", "impossible age (<18 or >100)", bad.sum(), "set birth_date NULL (age missing)")
    c["marketing_opt_out"] = to_number(c["marketing_opt_out"]).fillna(0).astype(int)
    c = c.drop(columns=["full_name"])  # GDPR: no names in the analytical store
    log.add("core_clients", "drop full_name (GDPR data minimisation)", len(c), "column removed")
    out["clients"] = c

    # ---------------- monthly snapshots
    b = raw["core_accounts_monthly"].copy()
    b["client_id"] = to_number(b["client_id"]).astype("Int64")
    b["month_end"] = parse_dates(b["month_end"])
    commas = b["balance"].astype("string").str.contains(",", na=False).sum()
    for col in ["balance", "num_products", "credit_score", "has_card", "is_active_member", "est_salary"]:
        b[col] = to_number(b[col])
    log.add("core_accounts_monthly", "comma decimal separator", commas, "convert to dot decimal")
    huge = b["balance"] > 1e8
    b.loc[huge, "balance"] = np.nan
    log.add("core_accounts_monthly", "impossible balance (>1e8)", huge.sum(), "set NULL")
    n0 = len(b)
    b = b.drop_duplicates(["client_id", "month_end"])
    log.add("core_accounts_monthly", "duplicate (client, month)", n0 - len(b), "keep first")
    log.add("core_accounts_monthly", "missing credit_score", b["credit_score"].isna().sum(),
            "keep NULL; imputed later inside the model pipeline (median)")
    out["balances"] = b

    # ---------------- exits
    e = raw["core_exits"].copy()
    e["client_id"] = to_number(e["client_id"]).astype("Int64")
    e["exit_date"] = parse_dates(e["exit_date"])
    out["exits"] = e.drop_duplicates("client_id")

    # ---------------- transactions
    t = raw["card_transactions"].copy()
    n0 = len(t)
    t = t.drop_duplicates("tx_id")
    log.add("card_transactions", "duplicate tx_id", n0 - len(t), "keep first (same transaction twice)")
    t["tx_id"] = to_number(t["tx_id"]).astype("Int64")
    t["client_id"] = to_number(t["client_id"]).astype("Int64")
    dots = t["ts"].astype("string").str.contains(r"^\d{2}\.", na=False).sum()
    t["ts"] = parse_dates(t["ts"])
    log.add("card_transactions", "date as dd.mm.yyyy text", dots, "parse with dayfirst")
    t["amount"] = to_number(t["amount"])
    badamt = t["amount"].isna() | (t["amount"] <= 0)
    t = t[~badamt]
    log.add("card_transactions", "amount missing or <= 0", badamt.sum(), "drop row")
    out["transactions"] = t

    # ---------------- messages (text noise)
    m = raw["contact_messages"].copy()
    m["msg_id"] = to_number(m["msg_id"]).astype("Int64")
    m["client_id"] = to_number(m["client_id"]).astype("Int64")
    m["ts"] = parse_dates(m["ts"])
    m["text"] = m["text"].fillna("").astype(str)
    garbled = m["text"].map(is_garbled)
    empty = m["text"].str.strip().eq("") | garbled
    log.add("contact_messages", "empty message", m["text"].str.strip().eq("").sum(),
            "keep the row (contact happened), flag is_empty, no sentiment")
    log.add("contact_messages", "garbled text (encoding errors)", garbled.sum(), "treat as empty")
    m["text_clean"] = m["text"].map(denoise)
    m["sentiment_raw"] = m["text"].map(raw_sentiment)
    m["sentiment_score"] = m["text_clean"].map(sentiment)
    m["is_empty"] = empty.astype(int)
    m["is_negative"] = ((m["sentiment_score"] <= -0.2) & ~empty).astype(int)
    m["complaint_flag"] = to_number(m["complaint_flag"]).fillna(0).astype(int)
    log.add("contact_messages", "text denoising (typos, diacritics, sarcasm)", len(m),
            "text_clean + sentiment_score; raw kept as sentiment_raw for comparison")
    out["messages"] = m

    # ---------------- survey
    s = raw["survey_responses"].copy()
    s["survey_id"] = to_number(s["survey_id"]).astype("Int64")
    s["client_id"] = to_number(s["client_id"]).astype("Int64")
    s["sent_date"] = parse_dates(s["sent_date"])
    s["answered_date"] = parse_dates(s["answered_date"])
    s["nps_score"] = to_number(s["nps_score"])
    log.add("survey_responses", "survey not answered (NULL score)", s["nps_score"].isna().sum(),
            "keep NULL: missing NOT at random -> feature nps_missing (silence)")
    out["survey"] = s
    return out, log.df()
