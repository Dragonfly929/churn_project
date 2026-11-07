"""RTAP — Real-Time Analytical Processing: alert -> action -> outcome -> learning.

ThompsonSampler : one Beta(successes+1, failures+1) belief per (reason, offer).
World           : plays the clients (simulation). Its response rates are HIDDEN from the policy.
compare_policies: no action / same offer for all / random / expert rules / Thompson Sampling,
                  with daily capacity, budget, the ethical rule and human override.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

OFFERS = {"personal_call": 25.0, "fee_waiver": 40.0, "card_cashback": 50.0, "deposit_bonus": 80.0}
REASONS = ["silence", "dissatisfaction", "money_withdrawal", "disengagement", "profile"]
EXPERT_RULES = {"silence": "personal_call", "dissatisfaction": "fee_waiver", "money_withdrawal": "deposit_bonus",
                "disengagement": "card_cashback", "profile": "fee_waiver"}

# hidden truth of the simulated world: P(a true leaver stays | offer, archetype)
RESPONSE = {
    "silent_exit": {"personal_call": 0.45, "fee_waiver": 0.08, "card_cashback": 0.10, "deposit_bonus": 0.15},
    "vocal": {"personal_call": 0.30, "fee_waiver": 0.50, "card_cashback": 0.10, "deposit_bonus": 0.15},
    "money_drain": {"personal_call": 0.15, "fee_waiver": 0.10, "card_cashback": 0.10, "deposit_bonus": 0.50},
    "gradual_fade": {"personal_call": 0.20, "fee_waiver": 0.12, "card_cashback": 0.40, "deposit_bonus": 0.15},
}


class ThompsonSampler:
    """Ethical by construction: the context is ONLY the alert reason (never gender, age, country)."""

    def __init__(self, value_saved=config.COST_FN, seed=config.SEED):
        self.a = {(r, o): 1.0 for r in REASONS for o in OFFERS}
        self.b = {(r, o): 1.0 for r in REASONS for o in OFFERS}
        self.value = value_saved
        self.rng = np.random.default_rng(seed)

    def choose(self, reason: str, allowed=None) -> str:
        allowed = allowed or list(OFFERS)
        draws = {o: self.rng.beta(self.a[(reason, o)], self.b[(reason, o)]) * self.value - OFFERS[o] for o in allowed}
        return max(draws, key=draws.get)

    def update(self, reason: str, offer: str, success: bool):
        if success:
            self.a[(reason, offer)] += 1
        else:
            self.b[(reason, offer)] += 1

    def table(self) -> pd.DataFrame:
        rows = [{"reason": r, "offer": o, "posterior_mean": self.a[(r, o)] / (self.a[(r, o)] + self.b[(r, o)]),
                 "trials": int(self.a[(r, o)] + self.b[(r, o)] - 2)} for r in REASONS for o in OFFERS]
        return pd.DataFrame(rows)

    def best(self) -> pd.DataFrame:
        t = self.table()
        t = t[t["trials"] > 0] if (t["trials"] > 0).any() else t
        t["expected_gain"] = t["posterior_mean"] * self.value - t["offer"].map(OFFERS)
        return t.loc[t.groupby("reason")["expected_gain"].idxmax()].reset_index(drop=True)

    def to_json(self):
        return {"a": {f"{r}|{o}": v for (r, o), v in self.a.items()}, "b": {f"{r}|{o}": v for (r, o), v in self.b.items()}}

    def load_json(self, d):
        for k, v in d.get("a", {}).items():
            self.a[tuple(k.split("|"))] = v
        for k, v in d.get("b", {}).items():
            self.b[tuple(k.split("|"))] = v


class World:
    """Simulated clients' reactions (unknown to the policy)."""

    def __init__(self, truth: pd.DataFrame, seed=config.SEED):
        self.arche = dict(zip(truth.client_id, truth.archetype))
        self.rng = np.random.default_rng(seed + 7)

    def outcome(self, client_id: int, offer: str | None, will_leave: bool) -> bool:
        """True = the client stays."""
        if not will_leave:
            return True
        a = self.arche.get(int(client_id), "stayer")
        if offer is None or offer not in OFFERS or a not in RESPONSE:
            return False
        return bool(self.rng.random() < RESPONSE[a][offer])


def compare_policies(alerts: pd.DataFrame, truth: pd.DataFrame, capacity: int = 40, budget: float = 1e9,
                     seed=config.SEED, overrides: dict | None = None) -> tuple[pd.DataFrame, ThompsonSampler, pd.DataFrame]:
    """alerts: client_id, alert_day, reason, risk, will_leave, gender, country, marketing_opt_out."""
    overrides = overrides or {}
    if "will_leave" not in alerts.columns:
        ex = alerts["client_id"].map(truth.set_index("client_id")["exit_date"])
        alerts = alerts.assign(will_leave=ex.notna() & (pd.to_datetime(ex) > pd.to_datetime(alerts["alert_day"])))
    rows, logs = [], []
    ts_final = None
    for pol in ["no action", "same offer for all", "random", "expert rules", "Thompson Sampling"]:
        rng = np.random.default_rng(seed)
        world = World(truth, seed)
        ts = ThompsonSampler(seed=seed)
        spent, saved, n_offers, lost = 0.0, 0, 0, 0
        for day, g in alerts.sort_values(["alert_day", "risk"], ascending=[True, False]).groupby("alert_day", sort=True):
            for i, (_, a) in enumerate(g.iterrows()):
                offer = None
                if pol != "no action" and i < capacity and spent < budget:
                    allowed = list(OFFERS)
                    if a.get("marketing_opt_out", 0):
                        allowed = ["personal_call"]  # service call only (GDPR Art. 21)
                    if pol == "same offer for all":
                        offer = "fee_waiver" if "fee_waiver" in allowed else allowed[0]
                    elif pol == "random":
                        offer = rng.choice(allowed)
                    elif pol == "expert rules":
                        offer = EXPERT_RULES[a["reason"]] if EXPERT_RULES[a["reason"]] in allowed else allowed[0]
                    else:
                        offer = ts.choose(a["reason"], allowed)
                    offer = overrides.get(int(a["client_id"]), offer)  # human in the loop
                stays = world.outcome(a["client_id"], offer, bool(a["will_leave"]))
                if offer:
                    spent += OFFERS[offer]
                    n_offers += 1
                    if pol == "Thompson Sampling" and a["will_leave"]:
                        ts.update(a["reason"], offer, stays)
                    elif pol == "Thompson Sampling":
                        pass  # a false alarm teaches nothing about the offer (would have stayed anyway)
                if a["will_leave"]:
                    saved += int(stays)
                    lost += int(not stays)
                if pol == "Thompson Sampling":
                    logs.append({"day": day, "client_id": a["client_id"], "reason": a["reason"], "offer": offer,
                                 "gender": a.get("gender"), "country": a.get("country"), "stays": stays,
                                 "will_leave": a["will_leave"]})
        value = saved * config.COST_FN
        rows.append({"policy": pol, "offers_sent": n_offers, "clients_saved": saved, "leavers_lost": lost,
                     "offer_cost": spent, "value_retained": value, "net_gain": value - spent,
                     "ROI": value / spent if spent else np.nan})
        if pol == "Thompson Sampling":
            ts_final = ts
    return pd.DataFrame(rows), ts_final, pd.DataFrame(logs)


def offer_fairness(log: pd.DataFrame) -> pd.DataFrame:
    """Share of alerted clients receiving each offer, by gender and country (ethical check)."""
    out = []
    for col in ("gender", "country"):
        t = pd.crosstab(log[col], log["offer"].fillna("none"), normalize="index")
        t.index = [f"{col}={i}" for i in t.index]
        out.append(t)
    return pd.concat(out).round(3)


MEASURES = [
    "S-a introdus o politică adaptivă (Thompson Sampling pe motiv × ofertă), deoarece ratele de răspuns la "
    "oferte nu sunt cunoscute dinainte și trebuie învățate din rezultate.",
    "S-a impus regula etică (oferta depinde doar de motivul alertei, nu de gen/vârstă/țară) și opt-out-ul de "
    "marketing, deoarece decizia automată nu trebuie să discrimineze (GDPR art. 21–22).",
    "S-au limitat ofertele la capacitatea zilnică și la buget și s-a păstrat suprascrierea umană, deoarece "
    "modelul doar propune, omul decide.",
]


def rtap_fundamental(alerts: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """The RTAP fundamental: the five policies compared on the alerts of the streaming replay."""
    pol, ts, log = compare_policies(alerts, truth)
    return {"table": pol, "learned": ts.best(), "fair": offer_fairness(log), "measures": MEASURES}
