"""The live scorer shared by the offline simulation (ds/streaming_sim.py) and the
Kafka service (realtime/alert_churn_MD.py).

Two scoring paths (the answer to the research question):
  event path : a new transaction/message rescores THAT client immediately (latency in ms)
  clock path : silence sends no messages -> when a bank day closes, EVERY active client is
               rescored; days without activity grow and the risk rises with no event at all.
Alert rule  : EWMA of daily scores >= threshold (from the cost matrix), within the daily
              capacity of the retention team, highest risk first. Each alert carries the SHAP
              reason and a RAG recommendation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

import config
from ds import rag
from ds.livefeatures import LiveStore
from ds.xai import reason_code, shap_frame


def load_champion(path=None) -> dict:
    return joblib.load(path or (config.ARTIFACTS / "champion.joblib"))


@dataclass
class Scorer:
    champion: dict
    store: LiveStore
    alpha: float = 0.3                 # EWMA smoothing
    capacity: int = 40                 # alerts per bank day
    use_ewma: bool = True
    challengers: dict = field(default_factory=dict)   # name -> {"pipe", "features", "threshold"} (shadow mode)
    ewma: dict = field(default_factory=dict)
    alerted: dict = field(default_factory=dict)
    last_scores: dict = field(default_factory=dict)
    shadow_alerted: dict = field(default_factory=dict)
    latencies: list = field(default_factory=list)

    @property
    def threshold(self):
        return float(self.champion["threshold"])

    def _score(self, X: pd.DataFrame, model=None) -> np.ndarray:
        m = model or self.champion
        return m["pipe"].predict_proba(X[m["features"]])[:, 1]

    # ---------------------------------------------------------------- event path
    def on_event(self, ev: dict) -> dict | None:
        t0 = time.perf_counter()
        self.store.add_event(ev)
        if ev.get("type") not in ("transaction", "message", "survey_answer"):
            return None
        cid = int(ev["client_id"])
        if cid not in self.store.profile.index:
            return None
        day = str(ev["ts"])[:10]
        X = self.store.features(day, clients=[cid], eligibility_cutoff=day)
        p = float(self._score(X)[0])
        self.last_scores[cid] = p
        ms = (time.perf_counter() - t0) * 1000
        self.latencies.append(ms)
        return {"client_id": cid, "score": p, "latency_ms": ms}

    # ---------------------------------------------------------------- clock path
    def close_day(self, day: str, explain: bool = True) -> tuple[list[dict], pd.DataFrame]:
        ids = self.store.active_clients(day)
        X = self.store.features(day, clients=ids, eligibility_cutoff=day)
        p = self._score(X)
        sm = []
        for cid, pi in zip(X["client_id"].values, p):
            prev = self.ewma.get(cid, pi)
            e = self.alpha * pi + (1 - self.alpha) * prev if self.use_ewma else pi
            self.ewma[cid] = e
            sm.append(e)
        X = X.assign(score=p, ewma=sm)
        for name, ch in self.challengers.items():  # shadow mode: scored, never acted upon
            ps = self._score(X, ch)
            for cid, v in zip(X["client_id"].values, ps):
                if v >= ch["threshold"] and cid not in self.shadow_alerted.setdefault(name, {}):
                    self.shadow_alerted[name][cid] = day
        cand = X[(X["ewma"] >= self.threshold) & (~X["client_id"].isin(list(self.alerted)))]
        cand = cand.sort_values("ewma", ascending=False).head(self.capacity)
        alerts = []
        if len(cand):
            reasons = {}
            if explain:
                sdf, _ = shap_frame(self.champion["pipe"], cand.set_index("client_id"), self.champion["features"],
                                    max_rows=len(cand) + 1)
                for cid, row in sdf.iterrows():
                    r, _ = reason_code(row)
                    reasons[cid] = (r, row.sort_values(ascending=False).index[:3].tolist())
            for _, row in cand.iterrows():
                cid = int(row["client_id"])
                r, top = reasons.get(cid, ("unknown", []))
                rec = rag.recommend(r, top, float(row["ewma"]), int(row.get("marketing_opt_out", 0) or 0))
                a = {"client_id": cid, "alert_day": day, "risk": float(row["ewma"]), "score_today": float(row["score"]),
                     "reason": r, "top_features": top, "offer_suggested": rec["offer"], "rag_text": rec["text"],
                     "gender": row["gender"], "country": row["country"],
                     "marketing_opt_out": int(row.get("marketing_opt_out", 0) or 0)}
                self.alerted[cid] = day
                alerts.append(a)
        return alerts, X
