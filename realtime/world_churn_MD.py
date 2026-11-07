"""WORLD service (simulation only): plays the clients.

Consumes bank.offers, waits some bank days, then publishes the outcome to bank.outcomes:
did the client stay? Its response rates per archetype x offer are HIDDEN from the RTAP policy
(in a real bank this is what you observe, ideally against a control group = uplift).

  python realtime/world_churn_MD.py [--bus file] [--delay-days 7]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from ds.rtap import World  # noqa: E402
from realtime.bus import add_common_args, get_bus, run_info  # noqa: E402


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--delay-days", type=float, default=7)
    a = ap.parse_args()
    info = run_info()
    run_id = info.get("run_id", "manual")
    spd = info.get("seconds_per_day", config.SECONDS_PER_BANK_DAY)
    truth = pd.read_csv(config.RAW / "sim_truth.csv", parse_dates=["exit_date"])
    exits = truth.set_index("client_id")["exit_date"]
    world = World(truth)
    bus = get_bus(a.bus, a.bootstrap)
    cons = bus.consumer("bank.offers", f"world-{run_id}")
    pending = []
    print(f"[world] ready (bus={bus.kind}); outcomes after {a.delay_days} bank days", flush=True)
    while True:
        for o in cons.poll(200):
            if o.get("run_id", run_id) != run_id:
                continue
            ex = exits.get(int(o["client_id"]))
            will_leave = bool(pd.notna(ex) and ex > pd.Timestamp(o["alert_day"]))
            stays = world.outcome(int(o["client_id"]), o["offer"], will_leave)
            pending.append((time.time() + a.delay_days * spd, {**o, "will_leave": will_leave, "stays": stays,
                                                               "sent_at": None}))
        cons.commit()
        now = time.time()
        due = [p for p in pending if p[0] <= now]
        pending = [p for p in pending if p[0] > now]
        for _, out in due:
            out["sent_at"] = now
            bus.send("bank.outcomes", out, key=out["client_id"])
        if due:
            bus.flush()
            p = config.RT_OUT / "outcomes.csv"
            pd.DataFrame([d for _, d in due]).to_csv(p, mode="a", header=not p.exists(), index=False)


if __name__ == "__main__":
    main()
