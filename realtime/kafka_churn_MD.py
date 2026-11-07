"""PRODUCER (lesson 5: kafka_trio_MD.py -> kafka_churn_MD.py).

Each bank system is a separate producer process, able to run on another laptop:
  --source mobile_app      transactions in the mobile app / web
  --source card_branch     card payments, ATM, branch, salary transfers
  --source contact_center  messages (raw text) and survey answers

Events are replayed in EVENT TIME, accelerated (default 2 s = 1 bank day), from the source system
exports (data/raw/*.csv) for the 90 days after the cutoff. All producers share one bank clock
(t0 in realtime/state/run.json), so they stay on the same day. Key = client_id (per-client order).
After each bank day the producer sends a 'day_end' heartbeat: silence sends nothing, so the
scorer needs the clock to rescore silent clients.

  python realtime/kafka_churn_MD.py --source mobile_app                       # Kafka on localhost
  python realtime/kafka_churn_MD.py --source card_branch --bootstrap 192.168.1.20:9092 --t0 1730800000 --run-id X
  python realtime/kafka_churn_MD.py --source contact_center --bus file        # no-Kafka fallback
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from realtime.bus import add_common_args, get_bus, run_info, write_state  # noqa: E402

DIGITAL = {"mobile_app", "web"}
SOURCE_CHANNELS = {"mobile_app": {"mobile_app", "web"}, "card_branch": {"card_pos", "atm", "branch", "transfer"}}


def load_events(source: str, start: str, days: int, raw_dir=config.RAW) -> pd.DataFrame:
    s, e = pd.Timestamp(start), pd.Timestamp(start) + pd.Timedelta(days=days)
    if source in SOURCE_CHANNELS:
        t = pd.read_csv(raw_dir / "card_transactions.csv", parse_dates=["ts"])
        t = t[(t.ts >= s) & (t.ts < e) & t.channel.isin(SOURCE_CHANNELS[source])].copy()
        t["type"] = "transaction"
        t["is_digital"] = t.channel.isin(DIGITAL).astype(int)
        return t[["type", "client_id", "ts", "amount", "direction", "channel", "is_digital"]].sort_values("ts")
    m = pd.read_csv(raw_dir / "contact_messages.csv", parse_dates=["ts"])
    m = m[(m.ts >= s) & (m.ts < e)].rename(columns={"complaint_flag": "is_complaint"})
    m["type"] = "message"
    m["text"] = m["text"].fillna("")
    sv = pd.read_csv(raw_dir / "survey_responses.csv", parse_dates=["sent_date", "answered_date"])
    sv = sv[(sv.answered_date >= s) & (sv.answered_date < e)].copy()
    sv = sv.assign(type="survey_answer", ts=sv.answered_date + pd.Timedelta(hours=12),
                   sent=sv.sent_date.dt.strftime("%Y-%m-%d"))[["type", "client_id", "ts", "sent", "nps_score"]]
    ev = pd.concat([m[["type", "client_id", "ts", "channel", "text", "is_complaint"]], sv], ignore_index=True)
    return ev.sort_values("ts")


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--source", required=True, choices=["mobile_app", "card_branch", "contact_center"])
    ap.add_argument("--t0", type=float, default=None, help="shared start (epoch seconds); default from run.json")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--seconds-per-day", type=float, default=None)
    ap.add_argument("--days", type=int, default=None)
    a = ap.parse_args()
    info = run_info()
    t0 = a.t0 or info.get("t0") or time.time() + 3
    run_id = a.run_id or info.get("run_id", "manual")
    spd = a.seconds_per_day or info.get("seconds_per_day", config.SECONDS_PER_BANK_DAY)
    days = a.days or info.get("days", config.STREAM_DAYS)
    start = pd.Timestamp(info.get("start_day", config.STREAM_START))

    bus = get_bus(a.bus, a.bootstrap)
    ev = load_events(a.source, str(start.date()), days)
    print(f"[{a.source}] {len(ev):,} events for {days} bank days, {spd}s per day, bus={bus.kind}, run={run_id}")
    ev["day"] = (ev.ts.dt.normalize() - start).dt.days
    ev["frac"] = (ev.ts - ev.ts.dt.normalize()).dt.total_seconds() / 86400
    sent = 0
    for d in range(days):
        g = ev[ev.day == d]
        for sl in range(10):  # 10 slices per bank day: realistic pacing without one sleep per event
            due = t0 + spd * (d + sl / 10)
            if due > time.time():
                time.sleep(due - time.time())
            part = g[(g.frac >= sl / 10) & (g.frac < (sl + 1) / 10)]
            now = time.time()
            for r in part.itertuples(index=False):
                v = {k: (str(x) if k == "ts" else (x.item() if hasattr(x, "item") else x))
                     for k, x in r._asdict().items() if k not in ("day", "frac") and not (isinstance(x, float) and pd.isna(x))}
                v.update(source=a.source, run_id=run_id, sent_at=now)
                bus.send("bank.events", v, key=int(r.client_id))
                sent += 1
            bus.flush()
        day = (start + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        bus.send("bank.events", {"type": "day_end", "source": a.source, "day": day, "run_id": run_id,
                                 "sent_at": time.time(), "client_id": 0}, key=0)
        bus.flush()
        write_state(f"producer_{a.source}.json", {"source": a.source, "bank_day": day, "events_sent": sent,
                                                  "events_today": len(g), "updated": time.time(), "bus": bus.kind})
        print(f"[{a.source}] {day}: {len(g):5d} events (total {sent:,})", flush=True)
    print(f"[{a.source}] done: {sent:,} events")


if __name__ == "__main__":
    main()
