"""ONLINE LEARNING next to the static model — the idea of the teacher's app_river.py applied to churn.

app_river.py scores each new PC-metrics message with River's HalfSpaceTrees (score_one, THEN
learn_one: incremental, never retrained from scratch). Here the same detector reads the bank's
event stream: at every closed bank day each client's day becomes one observation
  x = {tx: transactions, out: money out, msgs: messages, neg: complaints}   (scaled to [0,1])
and the detector flags clients whose day is unusual compared with what it has learned.
Clients with NO activity are scored too (silence = an all-zero day).

  python realtime/river_churn_MD.py [--bus file]       (pip install river)
Output: realtime/output/river_anomalies.csv (top anomalies per day) and state/river.json
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from realtime.bus import add_common_args, get_bus, run_info, write_state  # noqa: E402


def main():
    from river import anomaly
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--sources", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--warmup-days", type=int, default=5)
    a = ap.parse_args()
    info = run_info()
    run_id = info.get("run_id", "manual")
    bus = get_bus(a.bus, a.bootstrap)
    cons = bus.consumer("bank.events", f"river-{run_id}")
    model = anomaly.HalfSpaceTrees(n_trees=10, height=6, window_size=250, seed=config.SEED)  # lighter than app_river (25x8): ~4000 clients per bank day
    truth = pd.read_csv(config.RAW / "sim_truth.csv", parse_dates=["exit_date"]).set_index("client_id")["exit_date"]
    clients = pd.read_csv(config.DATASETS / f"bank_{config.CUTOFF}.csv", usecols=["client_id"])["client_id"].tolist()
    day_acc: dict[str, dict] = defaultdict(lambda: defaultdict(lambda: [0, 0.0, 0, 0]))
    ends: dict[str, set] = defaultdict(set)
    done_days, flagged_leavers, flagged = 0, 0, 0
    print(f"[river] HalfSpaceTrees online detector ready (bus={bus.kind})", flush=True)
    while True:
        for ev in cons.poll(300):
            if ev.get("run_id", run_id) != run_id:
                continue
            if ev["type"] == "day_end":
                ends[ev["day"]].add(ev["source"])
                continue
            d = str(ev["ts"])[:10]
            acc = day_acc[d][int(ev["client_id"])]
            if ev["type"] == "transaction":
                acc[0] += 1
                acc[1] += float(ev["amount"]) if ev.get("direction") == "out" else 0.0
            elif ev["type"] == "message":
                acc[2] += 1
                acc[3] += int(ev.get("is_complaint", 0))
        cons.commit()
        for d in sorted(k for k, v in ends.items() if len(v) >= a.sources):
            rows = []
            for cid in clients:
                ex = truth.get(cid)
                if pd.notna(ex) and ex <= pd.Timestamp(d):
                    continue  # account closed
                tx, out, msgs, neg = day_acc[d].get(cid, [0, 0.0, 0, 0])
                x = {"tx": min(1.0, tx / 5), "out": min(1.0, out / 3000), "msgs": min(1.0, msgs / 2), "neg": min(1.0, neg / 2)}
                s = float(model.score_one(x))   # score first ...
                model.learn_one(x)              # ... then learn (incremental)
                rows.append({"day": d, "client_id": cid, "score": s,
                             "future_leaver": bool(pd.notna(ex) and ex > pd.Timestamp(d)), **x})
            done_days += 1
            df = pd.DataFrame(rows)
            if done_days > a.warmup_days:
                top = df[df.score >= a.threshold]
                flagged += len(top)
                flagged_leavers += int(top.future_leaver.sum())
                p = config.RT_OUT / "river_anomalies.csv"
                top.nlargest(30, "score").to_csv(p, mode="a", header=not p.exists(), index=False)
            write_state("river.json", {"days": done_days, "flagged": flagged,
                                       "share_future_leavers_among_flagged": flagged_leavers / flagged if flagged else None,
                                       "base_rate_future_leavers": float(df.future_leaver.mean()) if len(df) else None,
                                       "warmup_days": a.warmup_days, "updated": time.time()})
            del ends[d]
            day_acc.pop(d, None)
            print(f"[river] {d}: scored {len(df)} client-days", flush=True)


if __name__ == "__main__":
    main()
