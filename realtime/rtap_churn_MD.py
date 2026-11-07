"""RTAP service: alert -> choose an action -> act -> observe -> learn -> measure.

Consumes bank.alerts (group 'rtap') and bank.outcomes; publishes bank.offers.
Policy = Thompson Sampling per (reason x offer); constraints = daily capacity and budget from the
control panel; ethical rule = the offer depends only on the reason (never gender/age/country);
marketing opt-out = service call only; human in the loop = overrides from the control panel.

  python realtime/rtap_churn_MD.py [--bus file]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from ds.rtap import OFFERS, ThompsonSampler  # noqa: E402
from realtime.bus import add_common_args, get_bus, read_state, run_info, write_state  # noqa: E402


def main():
    ap = add_common_args(argparse.ArgumentParser())
    a = ap.parse_args()
    info = run_info()
    run_id = info.get("run_id", "manual")
    bus = get_bus(a.bus, a.bootstrap)
    c_alerts = bus.consumer("bank.alerts", f"rtap-{run_id}")
    c_out = bus.consumer("bank.outcomes", f"rtap-{run_id}")
    ts = ThompsonSampler()
    per_day: dict[str, int] = {}
    spent = saved = lost = offers = no_capacity = 0
    log = []
    print(f"[rtap] ready (bus={bus.kind})", flush=True)
    last = 0
    while True:
        ctrl = read_state("control.json", {}) or {}
        cap = int(ctrl.get("capacity", 40))
        budget = float(ctrl.get("budget", 1e9))
        overrides = {int(k): v for k, v in (ctrl.get("overrides") or {}).items()}
        for al in c_alerts.poll(200):
            if al.get("run_id", run_id) != run_id:
                continue
            d = al["alert_day"]
            per_day[d] = per_day.get(d, 0)
            if per_day[d] >= cap or spent >= budget or ctrl.get("pause"):
                no_capacity += 1
                continue
            allowed = ["personal_call"] if al.get("marketing_opt_out") else list(OFFERS)
            offer = ts.choose(al["reason"], allowed)          # context = reason ONLY (ethical rule)
            human = overrides.get(int(al["client_id"]))
            if human:
                offer = human if human in OFFERS else None   # human in the loop: change or cancel
            if not offer:
                continue
            per_day[d] += 1
            spent += OFFERS[offer]
            offers += 1
            msg = {"client_id": al["client_id"], "offer": offer, "cost": OFFERS[offer], "reason": al["reason"],
                   "alert_day": d, "risk": al["risk"], "human_override": bool(human), "run_id": run_id,
                   "sent_at": time.time()}
            bus.send("bank.offers", msg, key=al["client_id"])
            log.append({**msg, "gender": al.get("gender"), "country": al.get("country")})
        bus.flush()
        c_alerts.commit()
        for o in c_out.poll(200):
            if o.get("run_id", run_id) != run_id:
                continue
            if o["will_leave"]:  # a false alarm says nothing about the offer's effect
                ts.update(o["reason"], o["offer"], bool(o["stays"]))
                saved += int(o["stays"])
                lost += int(not o["stays"])
        c_out.commit()
        if time.time() - last > 1:
            value = saved * config.COST_FN
            write_state("rtap.json", {"offers": offers, "spent": spent, "saved": saved, "lost_after_offer": lost,
                                      "value_retained": value, "ROI": value / spent if spent else None,
                                      "skipped_capacity_or_budget": no_capacity, "posteriors": ts.table().to_dict("records"),
                                      "best": ts.best().to_dict("records"), "updated": time.time()})
            if log:
                p = config.RT_OUT / "offers.csv"
                pd.DataFrame(log).to_csv(p, mode="a", header=not p.exists(), index=False)
                log = []
            last = time.time()


if __name__ == "__main__":
    main()
