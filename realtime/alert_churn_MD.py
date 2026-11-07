"""SCORER + ALERT (lesson 5: alert_trio_MD.py -> alert_churn_MD.py).

Consumes bank.events (group 'scorer'), keeps per-client state (365 days of history, loaded from
the DWH up to the cutoff), scores with the trained champion and publishes alerts to bank.alerts.

  event path : each micro-batch of new transactions/messages rescores the affected clients;
               end-to-end latency (producer send -> score) is measured per event
  clock path : when every source has closed a bank day (day_end), ALL active clients are rescored
               (silence raises risk without any event), EWMA, capacity, SHAP reason, RAG text
  shadow mode: L1 / L2 / ElasticNet challengers score the same clients; never acted upon

  python realtime/alert_churn_MD.py                 # Kafka
  python realtime/alert_churn_MD.py --bus file      # fallback
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from ds.livefeatures import LiveStore  # noqa: E402
from ds.scoring import Scorer, load_champion  # noqa: E402
from ds.textproc import denoise, is_garbled, sentiment  # noqa: E402
from realtime.bus import add_common_args, get_bus, read_state, run_info, write_state  # noqa: E402

OUT = config.RT_OUT


def enrich_message(ev):
    """Denoise the raw text on arrival (Noise fundamental, live)."""
    txt = ev.get("text", "") or ""
    empty = (not txt.strip()) or is_garbled(txt)
    s = 0.0 if empty else sentiment(denoise(txt))
    ev.update(is_empty=int(empty), sentiment=s, is_negative=int((s <= -0.2) and not empty))
    return ev


def append_csv(rows, name):
    if rows:
        p = OUT / name
        pd.DataFrame(rows).to_csv(p, mode="a", header=not p.exists(), index=False)


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--sources", type=int, default=3, help="number of producers that send day_end")
    ap.add_argument("--capacity", type=int, default=40)
    a = ap.parse_args()
    info = run_info()
    run_id = info.get("run_id", "manual")
    spd = info.get("seconds_per_day", config.SECONDS_PER_BANK_DAY)
    bus = get_bus(a.bus, a.bootstrap)
    print(f"[scorer] loading DWH state up to the cutoff {config.CUTOFF} ...", flush=True)
    store = LiveStore()
    store.restrict_until(config.CUTOFF)
    champ = load_champion()
    ch_path = config.ARTIFACTS / "challengers.joblib"
    challengers = joblib.load(ch_path) if ch_path.exists() else {}
    sc = Scorer(champ, store, capacity=a.capacity, challengers=challengers)
    cons = bus.consumer("bank.events", f"scorer-{run_id}")
    print(f"[scorer] ready: champion={champ.get('name')} threshold={sc.threshold:.2f}, bus={bus.kind}", flush=True)

    ends: dict[str, set] = {}
    next_day = pd.Timestamp(info.get("start_day", config.STREAM_START))
    lat_e2e, lat_compute = [], []
    n_events, n_days = 0, 0
    last_status = 0
    while True:
        batch = cons.poll(300)
        ctrl = read_state("control.json", {}) or {}
        sc.capacity = int(ctrl.get("capacity", sc.capacity))
        affected, sent_at = set(), []
        for ev in batch:
            if ev.get("run_id", run_id) != run_id:
                continue
            t = ev.get("type")
            if t == "day_end":
                ends.setdefault(ev["day"], set()).add(ev["source"])
                continue
            if t == "message":
                ev = enrich_message(ev)
            store.add_event(ev)
            if t in ("transaction", "message", "survey_answer"):
                affected.add(int(ev["client_id"]))
                sent_at.append(float(ev.get("sent_at", time.time())))
                n_events += 1
        # ---- event path (micro-batch): rescore only the clients that just did something
        if affected:
            t0 = time.perf_counter()
            ids = [c for c in affected if c in store.profile.index]
            day = str(next_day.date())
            X = store.features(day, clients=ids, eligibility_cutoff=day)
            p = sc._score(X)
            for cid, v in zip(X["client_id"], p):
                sc.last_scores[int(cid)] = float(v)
            comp = (time.perf_counter() - t0) * 1000
            now = time.time()
            lat_compute.append(comp)
            lat_e2e += [(now - s) * 1000 for s in sent_at]
            lat_e2e = lat_e2e[-20000:]
        # ---- clock path: close every bank day that all sources have finished (or that timed out)
        while True:
            d = str(next_day.date())
            due = info.get("t0", 0) + spd * ((next_day - pd.Timestamp(info.get("start_day", config.STREAM_START))).days + 1)
            if len(ends.get(d, set())) >= a.sources or (info.get("t0") and time.time() > due + 3 * spd and ends.get(d)):
                t1 = time.perf_counter()
                alerts, X = sc.close_day(d)
                ms = (time.perf_counter() - t1) * 1000
                for al in alerts:
                    al["run_id"] = run_id
                    al["top_features"] = ",".join(al["top_features"])
                    bus.send("bank.alerts", al, key=al["client_id"])
                bus.flush()
                append_csv(alerts, "alerts.csv")
                append_csv([{"day": d, "active_clients": len(X), "mean_score": float(X["score"].mean()),
                             "clients_over_threshold": int((X["ewma"] >= sc.threshold).sum()), "alerts": len(alerts),
                             "close_day_ms": ms, "silent_clients": int((X["no_feedback_6m"] == 1).sum())}], "daily.csv")
                shadow = [{"challenger": k, "client_id": c, "alert_day": dd} for k, v in sc.shadow_alerted.items()
                          for c, dd in v.items() if dd == d]
                append_csv(shadow, "shadow_alerts.csv")
                n_days += 1
                next_day += pd.Timedelta(days=1)
                print(f"[scorer] closed {d}: {len(X)} clients rescored in {ms:.0f} ms, {len(alerts)} alerts", flush=True)
            else:
                break
        cons.commit()  # commit AFTER processing
        if time.time() - last_status > 1.0:
            arr = np.array(lat_e2e[-5000:]) if lat_e2e else np.array([np.nan])
            write_state("scorer.json", {"events": n_events, "days_closed": n_days, "next_day": str(next_day.date()),
                                        "latency_e2e_p50_ms": float(np.nanpercentile(arr, 50)),
                                        "latency_e2e_p95_ms": float(np.nanpercentile(arr, 95)),
                                        "compute_ms_last": lat_compute[-1] if lat_compute else None,
                                        "alerts_total": len(sc.alerted), "threshold": sc.threshold,
                                        "capacity": sc.capacity, "updated": time.time(), "bus": bus.kind,
                                        "top_risk_now": sorted(sc.last_scores.items(), key=lambda kv: -kv[1])[:10]})
            last_status = time.time()
        if n_days >= info.get("days", config.STREAM_DAYS):
            print("[scorer] all bank days closed - done")
            break


if __name__ == "__main__":
    main()
