"""Streaming fundamental (offline replay, no broker needed):
replay the label window day by day, score with the clock path + EWMA, and compare the
lead time with monthly batch scoring. Also measures what EWMA smoothing changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from ds import plots
from ds.livefeatures import LiveStore
from ds.scoring import Scorer


def truth_exits(source="bank") -> pd.DataFrame:
    t = pd.read_csv((config.RAW if source == "bank" else config.RAW_BRANCH) / "sim_truth.csv", parse_dates=["exit_date"])
    return t


def replay(champion: dict, days: int = config.STREAM_DAYS, start=config.STREAM_START, capacity=40,
           use_ewma=True, explain=True, store=None, progress=None):
    store = store or LiveStore()
    sc = Scorer(champion, store, capacity=capacity, use_ewma=use_ewma)
    alerts = []
    for i, d in enumerate(pd.date_range(start, periods=days, freq="D")):
        a, _ = sc.close_day(d.strftime("%Y-%m-%d"), explain=explain)
        alerts += a
        if progress:
            progress(i / days)
    return pd.DataFrame(alerts), sc


def batch_monthly(champion: dict, store: LiveStore, dates=("2025-12-31", "2026-01-31", "2026-02-28"),
                  capacity=40, start=config.STREAM_START, days=config.STREAM_DAYS):
    """Monthly batch scoring. The ranked list is worked through by the same team capacity per day,
    so the alert date = the day the team actually reaches the client (fair comparison)."""
    rows, contacted = [], set()
    queue = pd.DataFrame(columns=["client_id", "score"])
    sched = {pd.Timestamp(d) for d in dates}
    for d in pd.date_range(pd.Timestamp(dates[0]), pd.Timestamp(start) + pd.Timedelta(days=days - 1)):
        if d in sched:  # new monthly scoring replaces the queue
            ids = store.active_clients(d)
            X = store.features(d, clients=ids, eligibility_cutoff=d)
            p = champion["pipe"].predict_proba(X[champion["features"]])[:, 1]
            X = X.assign(score=p)
            queue = X[(X.score >= champion["threshold"]) & ~X.client_id.isin(contacted)][["client_id", "score"]] \
                .sort_values("score", ascending=False)
        if d < pd.Timestamp(start):
            continue
        active = set(store.active_clients(d))
        today = queue[queue.client_id.isin(active)].head(capacity)
        queue = queue[~queue.client_id.isin(today.client_id)]
        for cid, s_ in zip(today.client_id, today.score):
            contacted.add(cid)
            rows.append({"client_id": cid, "alert_day": d.strftime("%Y-%m-%d"), "risk": s_})
    return pd.DataFrame(rows)


def evaluate(alerts: pd.DataFrame, truth: pd.DataFrame, start=config.STREAM_START, days=config.STREAM_DAYS, label="",
             subset=None):
    s = pd.Timestamp(start)
    e = s + pd.Timedelta(days=days - 1)
    leavers = truth[(truth.exit_date >= s) & (truth.exit_date <= e)]
    if subset is not None:
        leavers = leavers[leavers.client_id.isin(subset)]
    a = alerts.copy()
    a["alert_day"] = pd.to_datetime(a["alert_day"])
    m = leavers.merge(a[["client_id", "alert_day"]], on="client_id", how="left")
    m["caught"] = m["alert_day"].notna() & (m["alert_day"] < m["exit_date"])
    m["lead_days"] = np.where(m["caught"], (m["exit_date"] - m["alert_day"]).dt.days, np.nan)
    ex = truth.set_index("client_id")["exit_date"]
    a["will_leave"] = a["client_id"].map(ex).notna() & (a["client_id"].map(ex) > a["alert_day"])
    return {"mode": label, "alerts": len(a), "leavers_in_window": len(leavers), "caught_before_exit": int(m["caught"].sum()),
            "recall_before_exit": m["caught"].mean(), "median_lead_days": float(np.nanmedian(m["lead_days"])),
            "share_lead>=30d": float((m["lead_days"] >= 30).sum() / max(1, len(m))),
            "false_alarms (never leave)": int((~a["will_leave"]).sum()),
            "precision": float(a["will_leave"].mean()) if len(a) else np.nan}, m


def run_streaming_fundamental(champion: dict, days=config.STREAM_DAYS, capacity=40):
    truth = truth_exits()
    store = LiveStore()
    al_ewma, sc = replay(champion, days, capacity=capacity, use_ewma=True, store=store)
    al_raw, _ = replay(champion, days, capacity=capacity, use_ewma=False, explain=False, store=store)
    al_batch = batch_monthly(champion, store, capacity=capacity, days=days)
    # leavers that the model could NOT see at the cutoff (risk appeared later): where streaming should matter
    X0 = store.features(config.CUTOFF, clients=store.active_clients(config.CUTOFF), eligibility_cutoff=config.CUTOFF)
    p0 = champion["pipe"].predict_proba(X0[champion["features"]])[:, 1]
    late = set(X0.client_id[p0 < champion["threshold"]])
    rows, details = [], {}
    for lab, a in [("streaming daily + EWMA", al_ewma), ("monthly batch (same daily capacity)", al_batch)]:
        r, _ = evaluate(a, truth, label=lab + " | only leavers NOT flagged at the cutoff", subset=late, days=days)
        rows.append(r)
    for lab, a in [("streaming daily + EWMA", al_ewma), ("streaming daily, no smoothing", al_raw), ("monthly batch (same daily capacity)", al_batch)]:
        r, m = evaluate(a, truth, label=lab, days=days)
        rows.insert(len(details), r)
        details[lab] = m
    t = pd.DataFrame(rows)
    f, ax = plots.fig(8, 3.6)
    for i, lab in enumerate(["streaming daily + EWMA", "monthly batch (same daily capacity)"]):
        v = details[lab]["lead_days"].dropna()
        ax.hist(v, bins=range(0, 95, 5), alpha=0.6, color=plots.PALETTE[i], label=f"{lab} (median {v.median():.0f} d)")
    ax.set(title="Lead time: days between alert and departure", xlabel="days", ylabel="leavers caught")
    ax.legend()
    plots.save(f, "f5_streaming_lead_time")
    if len(al_ewma):
        al_ewma["will_leave"] = al_ewma["client_id"].map(truth.set_index("client_id")["exit_date"]).pipe(
            lambda s: s.notna() & (s > pd.to_datetime(al_ewma["alert_day"])))
    measures = [
        "S-a introdus calea „ceas” (rescorarea tuturor clienților activi la închiderea fiecărei zile bancare), "
        "deoarece tăcerea nu trimite evenimente: fără ea, un client tăcut nu ar fi rescorat niciodată.",
        "S-a efectuat netezirea alertelor cu EWMA, deoarece scorurile zilnice sunt zgomotoase și ar genera alarme "
        "false și oferte inutile.",
        "S-a limitat numărul de alerte la capacitatea zilnică a echipei, cel mai mare risc primul, deoarece o alertă "
        "care nu poate fi tratată nu are valoare.",
    ]
    return {"table": t, "alerts": al_ewma, "measures": measures, "scorer": sc, "truth": truth}
