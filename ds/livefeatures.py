"""The Stage 1 features computed in Python from raw events, for the REAL-TIME path.

Same definitions as stage1/sql/query_features_*.sql (windows, rounding, defaults),
so the live features equal the SQL features: proved by realtime/parity_test.py.

LiveStore keeps the history needed by the windows (365 days of transactions and
messages, monthly snapshots, surveys) and accepts new events from the stream.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from stage1 import db

ROUND4 = ["balance_change_3m", "tx_trend", "digital_share_90d", "avg_sentiment_180d"]


def _r(x, n):
    """SQL ROUND (half away from zero), not numpy's banker's rounding."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.floor(np.abs(x) * 10 ** n + 0.5) / 10 ** n


class LiveStore:
    def __init__(self, source: str = "bank", since: str = "2024-12-01"):
        self.source = source
        q = lambda s: db.read_sql(s, source=source)  # noqa: E731
        self.profile = q("""SELECT c.client_id, c.birth_date, c.gender, c.country, c.join_date,
                                   c.marketing_opt_out, d.full_date AS exit_date
                            FROM dim_client c LEFT JOIN fact_client_exit e ON e.client_key = c.client_key
                            LEFT JOIN dim_date d ON d.date_key = e.exit_date_key""")
        for c in ("birth_date", "join_date", "exit_date"):
            self.profile[c] = pd.to_datetime(self.profile[c])
        self.profile = self.profile.set_index("client_id")
        self.tx = q(f"""SELECT t.client_key AS client_id, d.full_date AS day, t.amount, t.direction, ch.is_digital
                        FROM fact_transaction t JOIN dim_date d ON d.date_key = t.date_key
                        JOIN dim_channel ch ON ch.channel_key = t.channel_key
                        WHERE d.full_date >= '{since}'""")
        self.msg = q(f"""SELECT f.client_key AS client_id, d.full_date AS day, f.is_complaint, f.is_negative,
                                f.sentiment_score, f.is_empty
                         FROM fact_feedback f JOIN dim_date d ON d.date_key = f.date_key
                         WHERE d.full_date >= '{since}'""")
        self.snap = q("""SELECT f.client_key AS client_id, d.full_date AS day, f.balance, f.num_products,
                                f.credit_score, f.has_card, f.is_active_member, f.est_salary
                         FROM fact_account_month f JOIN dim_date d ON d.date_key = f.date_key""")
        self.survey = q("""SELECT s.client_key AS client_id, ds.full_date AS sent, da.full_date AS answered, s.nps_score
                           FROM fact_survey s JOIN dim_date ds ON ds.date_key = s.sent_date_key
                           LEFT JOIN dim_date da ON da.date_key = s.answered_date_key""")
        for df, cols in [(self.tx, ["day"]), (self.msg, ["day"]), (self.snap, ["day"]), (self.survey, ["sent", "answered"])]:
            for c in cols:
                df[c] = pd.to_datetime(df[c])
        self.tx["amount"] = self.tx["amount"].astype(float)
        self.snap[["balance", "credit_score", "est_salary"]] = self.snap[["balance", "credit_score", "est_salary"]].astype(float)
        self._new_tx, self._new_msg = [], []

    # -------------------------------------------------------- stream input
    def restrict_until(self, day: str):
        """Keep only what was known at `day` (the replay then streams the rest)."""
        d = pd.Timestamp(day)
        self.tx = self.tx[self.tx.day <= d].reset_index(drop=True)
        self.msg = self.msg[self.msg.day <= d].reset_index(drop=True)
        self.survey.loc[self.survey.answered > d, ["answered", "nps_score"]] = [pd.NaT, np.nan]

    def add_event(self, ev: dict):
        t = ev.get("type")
        day = pd.Timestamp(str(ev["ts"])[:10])
        if t == "transaction":
            self._new_tx.append({"client_id": int(ev["client_id"]), "day": day, "amount": float(ev["amount"]),
                                 "direction": ev["direction"], "is_digital": int(ev.get("is_digital", 0))})
        elif t == "message":
            self._new_msg.append({"client_id": int(ev["client_id"]), "day": day, "is_complaint": int(ev.get("is_complaint", 0)),
                                  "is_negative": int(ev.get("is_negative", 0)), "sentiment_score": float(ev.get("sentiment", 0)),
                                  "is_empty": int(ev.get("is_empty", 0))})
        elif t == "survey_answer":
            m = (self.survey.client_id == int(ev["client_id"])) & (self.survey.sent == pd.Timestamp(ev["sent"]))
            self.survey.loc[m, ["answered", "nps_score"]] = [day, float(ev["nps_score"])]

    def flush(self):
        if self._new_tx:
            self.tx = pd.concat([self.tx, pd.DataFrame(self._new_tx)], ignore_index=True)
            self._new_tx = []
        if self._new_msg:
            self.msg = pd.concat([self.msg, pd.DataFrame(self._new_msg)], ignore_index=True)
            self._new_msg = []

    def active_clients(self, day) -> np.ndarray:
        d = pd.Timestamp(day)
        p = self.profile
        return p.index[(p.join_date <= d) & (p.exit_date.isna() | (p.exit_date > d))].values

    # -------------------------------------------------------- the features (= the SQL query)
    def features(self, as_of, clients=None, eligibility_cutoff=None) -> pd.DataFrame:
        self.flush()
        f = pd.Timestamp(as_of)
        cut = pd.Timestamp(eligibility_cutoff or as_of)
        ids = np.asarray(clients) if clients is not None else self.active_clients(cut)
        P = self.profile.loc[ids]
        out = pd.DataFrame(index=pd.Index(ids, name="client_id"))
        out["age"] = np.floor((cut - P["birth_date"]).dt.days / 365.25)
        out["gender"] = P["gender"].values
        out["country"] = P["country"].values
        out["tenure_months"] = np.floor((cut - P["join_date"]).dt.days / 30.44)
        sn = self.snap[(self.snap.day <= f) & self.snap.client_id.isin(ids)].sort_values(["client_id", "day"], ascending=[True, False])
        sn["rn"] = sn.groupby("client_id").cumcount() + 1
        s1 = sn[sn.rn == 1].set_index("client_id")
        s4 = sn[sn.rn == 4].set_index("client_id")["balance"]
        for c in ["credit_score", "num_products", "has_card", "is_active_member", "est_salary", "balance"]:
            out[c] = s1[c].reindex(ids).values
        b4 = s4.reindex(ids).values
        with np.errstate(divide="ignore", invalid="ignore"):
            out["balance_change_3m"] = np.where(b4 > 0, _r((out["balance"].values - b4) / b4, 4), 0.0)
        tx = self.tx[(self.tx.day <= f) & (self.tx.day > f - pd.Timedelta(days=365)) & self.tx.client_id.isin(ids)]
        w90 = tx.day > f - pd.Timedelta(days=90)
        wp = (tx.day <= f - pd.Timedelta(days=90)) & (tx.day > f - pd.Timedelta(days=180))
        g = pd.DataFrame({"client_id": tx.client_id, "c90": w90.astype(int), "cp": wp.astype(int),
                          "in90": np.where(w90 & (tx.direction == "in"), tx.amount, 0.0),
                          "out90": np.where(w90 & (tx.direction == "out"), tx.amount, 0.0),
                          "dig": (w90 & (tx.is_digital == 1)).astype(int), "day": tx.day}).groupby("client_id").agg(
            c90=("c90", "sum"), cp=("cp", "sum"), in90=("in90", "sum"), out90=("out90", "sum"), dig=("dig", "sum"),
            last=("day", "max")).reindex(ids)
        c90, cp = g["c90"].fillna(0).values, g["cp"].fillna(0).values
        out["tx_count_90d"] = c90
        out["tx_count_prev_90d"] = cp
        out["tx_trend"] = _r((c90 + 1.0) / (cp + 1.0), 4)
        out["amount_in_90d"] = _r(g["in90"].fillna(0).values, 2)
        out["amount_out_90d"] = _r(g["out90"].fillna(0).values, 2)
        out["net_flow_90d"] = _r(g["in90"].fillna(0).values - g["out90"].fillna(0).values, 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["digital_share_90d"] = np.where(c90 > 0, _r(g["dig"].fillna(0).values / np.maximum(c90, 1), 4), 0.0)
        m = self.msg[(self.msg.day <= f) & (self.msg.day > f - pd.Timedelta(days=365)) & self.msg.client_id.isin(ids)]
        w180 = m.day > f - pd.Timedelta(days=180)
        mg = pd.DataFrame({"client_id": m.client_id, "n": w180.astype(int),
                           "comp": np.where(w180, m.is_complaint, 0), "neg": np.where(w180, m.is_negative, 0),
                           "sent": np.where(w180 & (m.is_empty == 0), m.sentiment_score, np.nan), "day": m.day}
                          ).groupby("client_id").agg(n=("n", "sum"), comp=("comp", "sum"), neg=("neg", "sum"),
                                                     sent=("sent", "mean"), last=("day", "max")).reindex(ids)
        n180 = mg["n"].fillna(0).values
        out["msg_count_180d"] = n180
        out["complaints_180d"] = mg["comp"].fillna(0).values
        out["neg_msg_180d"] = mg["neg"].fillna(0).values
        out["avg_sentiment_180d"] = _r(mg["sent"].fillna(0).values, 4)
        out["days_since_last_tx"] = np.where(g["last"].isna(), 365, (f - g["last"]).dt.days)
        out["days_since_last_msg"] = np.where(mg["last"].isna(), 365, (f - mg["last"]).dt.days)
        out["no_feedback_6m"] = (n180 == 0).astype(int)
        sv = self.survey[(self.survey.sent <= f - pd.Timedelta(days=15)) & (self.survey.sent > f - pd.Timedelta(days=120))
                         & self.survey.client_id.isin(ids)].sort_values(["client_id", "sent"], ascending=[True, False])
        sv = sv.drop_duplicates("client_id").set_index("client_id")
        nps = np.where(sv["answered"].notna() & (sv["answered"] <= f), sv["nps_score"], np.nan)
        out["nps_score"] = pd.Series(nps, index=sv.index).reindex(ids).values
        out["nps_missing"] = pd.isna(out["nps_score"]).astype(int)
        out["silent_decline"] = (((c90 + 1.0) / (cp + 1.0) < 0.5) & (n180 == 0)).astype(int)
        out["marketing_opt_out"] = P["marketing_opt_out"].values
        return out.reset_index()
