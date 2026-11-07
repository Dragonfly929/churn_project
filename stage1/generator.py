"""Step 1.5 — Simulate the bank's SOURCE SYSTEMS from the data contract.

Real bank data is not available to students, so the operational systems are
simulated. Each system writes its own CSV export, and they share client_id:

  core banking    -> core_clients.csv, core_accounts_monthly.csv, core_exits.csv
  cards/payments  -> card_transactions.csv   (channels: mobile_app, web, card_pos, atm, branch, transfer)
  contact centre  -> contact_messages.csv    (call, chat, email, app_review)
  survey (NPS)    -> survey_responses.csv
  simulation only -> sim_truth.csv, sim_truth_messages.csv  (NEVER read by the query/model)

Ways of leaving that are simulated (the research question):
  silent_exit   - activity stops abruptly, salary moves away, no messages, no survey
  gradual_fade  - transactions decline over 3-5 months
  money_drain   - big outgoing transfers, balance drops to ~0
  vocal         - complaints and negative messages, then leaves
  false_alarm   - a STAYER with a temporary dip (vacation, illness) -> the model must not overreact
  stayer        - normal behaviour

Usage:
  python stage1/generator.py                 # main bank  -> data/raw/
  python stage1/generator.py --branch        # external branch, messy -> data/raw_branch/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from stage1.textgen import make_message  # noqa: E402

FIRST = ["Ion", "Maria", "Andrei", "Elena", "Mihai", "Ana", "Victor", "Natalia", "Sergiu", "Olga",
         "Alexandru", "Cristina", "Vasile", "Irina", "Dumitru", "Tatiana"]
LAST = ["Popescu", "Rusu", "Ceban", "Lungu", "Moraru", "Cojocaru", "Turcan", "Rotaru", "Munteanu",
        "Bivol", "Ciobanu", "Gutu"]
CHANNELS_DIGITAL = ["mobile_app", "web"]
CHANNELS_PHYSICAL = ["card_pos", "atm", "branch"]
SURVEY_DATES = ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30"]


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def simulate(n: int, seed: int, shift: dict | None = None) -> dict[str, pd.DataFrame]:
    """Simulate n clients. `shift` changes the population (external branch)."""
    shift = shift or {}
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp(config.SIM_START), pd.Timestamp(config.SIM_END)
    days = pd.date_range(start, end, freq="D")
    nd = len(days)
    ref = pd.Timestamp(config.CUTOFF)

    # ------------------------------------------------------------ profile
    age = np.clip(rng.normal(41 + shift.get("age", 0), 13, n), 18, 85).round().astype(int)
    birth = [ref - pd.DateOffset(years=int(a)) - pd.Timedelta(days=int(rng.integers(0, 365)))
             for a in age]
    gender = rng.choice(["F", "M"], n)
    cp = shift.get("country_p", [0.6, 0.25, 0.15])
    country = rng.choice(["MD", "RO", "UA"], n, p=cp)
    tenure_days = np.where(rng.random(n) < 0.1, -rng.integers(30, 300, n),  # joined during 2025
                           rng.integers(30, 3650, n))
    join = np.array([start - pd.Timedelta(days=int(t)) for t in tenure_days])
    credit = np.clip(rng.normal(650, 95, n), 350, 850).round()
    nprod = rng.choice([1, 2, 3, 4], n, p=[0.5, 0.42, 0.06, 0.02])
    has_card = (rng.random(n) < 0.7).astype(int)
    salary = np.round(np.exp(rng.normal(10.2, 0.6, n)), 0)  # yearly EUR
    employed = rng.random(n) < 0.72
    opt_out = (rng.random(n) < 0.08).astype(int)
    digital = _sigmoid(2.0 - 0.07 * (age - 35) + shift.get("digital", 0) + rng.normal(0, 0.6, n))
    lam = np.exp(rng.normal(np.log(0.22), 0.45, n))  # card transactions / day
    bal0 = np.where(rng.random(n) < 0.33, rng.uniform(0, 300, n), np.exp(rng.normal(8.9, 1.0, n)))

    # ------------------------------------------------------------ who leaves, how, when
    z = (-0.95 + 0.035 * np.clip(age - 40, -15, 25) + 0.45 * (country == "RO")
         + np.select([nprod == 1, nprod == 2], [0.35, -0.65], 1.3)
         + 0.3 * (tenure_days < 365) + rng.normal(0, 0.4, n))
    leaver = rng.random(n) < _sigmoid(z)
    p_silent = np.where(age < 32, 0.5, 0.3)
    arche = np.full(n, "stayer", dtype=object)
    for i in np.where(leaver)[0]:
        ps = p_silent[i]
        rest = (1 - ps) / 3
        arche[i] = rng.choice(["silent_exit", "gradual_fade", "money_drain", "vocal"],
                              p=[ps, rest * 1.0, rest * 0.75, rest * 1.25])
    fa = (~leaver) & (rng.random(n) < 0.12)
    arche[fa] = "false_alarm"

    q_edges = [("2025-04-01", "2025-09-30"), ("2025-10-01", "2025-12-31"),
               ("2026-01-01", "2026-03-31"), ("2026-04-01", "2026-06-30")]
    exit_date = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    for i in np.where(leaver)[0]:
        a, b = q_edges[rng.choice(4, p=[0.15, 0.30, 0.40, 0.15])]
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        exit_date[i] = a + pd.Timedelta(days=int(rng.integers(0, (b - a).days + 1)))
    # a client cannot leave before joining + 60 days
    for i in np.where(leaver)[0]:
        if pd.Timestamp(exit_date[i]) < pd.Timestamp(join[i]) + pd.Timedelta(days=60):
            exit_date[i] = np.datetime64(pd.Timestamp(join[i]) + pd.Timedelta(days=90))

    # ------------------------------------------------------------ behaviour, day by day
    tx_parts, msg_rows, bal_rows, truth_msgs = [], [], [], []
    day_idx = np.arange(nd)
    month_ends = pd.date_range(start, end, freq="ME")
    msg_id = 0
    for i in range(n):
        ji = max(0, (pd.Timestamp(join[i]) - start).days)
        ei = nd if pd.isna(exit_date[i]) else min(nd, (pd.Timestamp(exit_date[i]) - start).days)
        alive = (day_idx >= ji) & (day_idx < ei)
        d_to_exit = (ei - day_idx).astype(float) if ei < nd else np.full(nd, 9999.0)
        m = np.ones(nd)
        m *= np.where(days.month == 8, 0.85, 1.0)  # summer seasonality
        a = arche[i]
        salary_stop = nd
        if a == "gradual_fade":
            onset = rng.integers(90, 160)
            m *= np.clip(d_to_exit / onset, 0.08, 1.0)
        elif a == "silent_exit":
            drop = rng.integers(35, 75)
            level = 0.12 if rng.random() < 0.75 else 0.6  # 25% are hard to see
            m *= np.where(d_to_exit < drop, level, 1.0)
            salary_stop = ei - drop
        elif a == "vocal":
            m *= np.where(d_to_exit < 50, 0.8, 1.0)
        elif a == "false_alarm":
            s = rng.integers(200, nd - 80)
            dur = rng.integers(40, 80)
            m *= np.where((day_idx >= s) & (day_idx < s + dur), 0.3, 1.0)
        cnt = rng.poisson(lam[i] * m * alive)
        k = cnt.sum()
        if k:
            di = np.repeat(day_idx, cnt)
            dig = rng.random(k) < digital[i]
            ch = np.where(dig, rng.choice(CHANNELS_DIGITAL, k, p=[0.8, 0.2]),
                          rng.choice(CHANNELS_PHYSICAL, k, p=[0.6, 0.3, 0.1]))
            direction = np.where(rng.random(k) < 0.12, "in", "out")
            amt = np.round(np.exp(rng.normal(3.4, 0.9, k)), 2)
            tx_parts.append(pd.DataFrame({"client_id": i + 1, "di": di, "amount": amt,
                                          "direction": direction, "channel": ch}))
        # salary credits on day 5 of each month
        if employed[i]:
            sal_days = day_idx[(days.day == 5) & alive & (day_idx < salary_stop)]
            if len(sal_days):
                tx_parts.append(pd.DataFrame({
                    "client_id": i + 1, "di": sal_days,
                    "amount": np.round(salary[i] / 12 * rng.uniform(0.9, 1.1, len(sal_days)), 2),
                    "direction": "in", "channel": "transfer"}))
        # money drain: large outgoing transfers before exit
        drain_total = 0.0
        if a == "money_drain" and ei < nd:
            kk = int(rng.integers(2, 5))
            dd = np.clip(ei - rng.integers(5, 45, kk), ji, nd - 1)
            amts = np.round(bal0[i] * rng.uniform(0.15, 0.35, kk) + rng.uniform(200, 2000, kk), 2)
            drain_total = amts.sum()
            tx_parts.append(pd.DataFrame({"client_id": i + 1, "di": dd, "amount": amts,
                                          "direction": "out", "channel": "mobile_app"}))
        # ---------------- messages
        base = 0.0045 if a != "silent_exit" else 0.002
        rate = np.full(nd, base)
        p_neg = np.full(nd, 0.2)
        if a == "vocal":
            rate = np.where(d_to_exit < 120, 0.045, base)
            p_neg = np.where(d_to_exit < 120, 0.75, 0.25)
        elif a == "gradual_fade":
            p_neg = np.where(d_to_exit < 150, 0.4, 0.2)
        elif a == "silent_exit":
            rate = np.where(d_to_exit < 200, 0.0, base)
        elif a == "false_alarm":
            rate = rate * 1.6
            p_neg = np.full(nd, 0.35)
        elif a == "money_drain":
            p_neg = np.where(d_to_exit < 60, 0.35, 0.2)
        mc = rng.poisson(rate * alive)
        for dix in np.repeat(day_idx, mc):
            r = rng.random()
            s = "neg" if r < p_neg[dix] else ("neu" if r < p_neg[dix] + 0.45 else "pos")
            text, noise = make_message(s, rng)
            msg_id += 1
            msg_rows.append((msg_id, i + 1, int(dix), rng.choice(["call", "chat", "email", "app_review"],
                             p=[0.35, 0.35, 0.15, 0.15]), text, int(s == "neg" and rng.random() < 0.7)))
            truth_msgs.append((msg_id, s, noise))
        # ---------------- monthly snapshot (core banking)
        bal = bal0[i]
        for me in month_ends:
            mi = (me - start).days
            if mi < ji:
                continue
            closed = ei < nd and mi >= ei
            if closed:
                bal_rows.append((i + 1, me, 0.0, nprod[i], credit[i], has_card[i], 0, salary[i], "closed"))
                break
            mstart = max(ji, mi - me.day + 1)
            act = int(cnt[mstart:mi + 1].sum())
            bal = max(0.0, bal * rng.normal(1.0, 0.04) + (salary[i] / 12 * 0.08 if employed[i] else 0))
            if a == "money_drain" and ei < nd and ei - mi < 60:
                bal *= 0.35
            if a == "silent_exit" and ei < nd and ei - mi < 75:
                bal *= 0.7
            cs = credit[i] + rng.normal(0, 6)
            if rng.random() < 0.03:
                cs = np.nan  # missing completely at random (CRM glitch)
            bal_rows.append((i + 1, me, round(bal, 2), nprod[i], cs, has_card[i], int(act >= 4),
                             salary[i], "active"))
        _ = drain_total

    tx = pd.concat(tx_parts, ignore_index=True)
    secs = rng.integers(7 * 3600, 22 * 3600, len(tx))
    tx["ts"] = days.values[tx["di"].values] + pd.to_timedelta(secs, unit="s")
    tx = tx.sort_values("ts").reset_index(drop=True)
    tx.insert(0, "tx_id", np.arange(1, len(tx) + 1))
    tx = tx[["tx_id", "client_id", "ts", "amount", "direction", "channel"]]

    msgs = pd.DataFrame(msg_rows, columns=["msg_id", "client_id", "di", "channel", "text", "complaint_flag"])
    msgs["ts"] = days.values[msgs["di"].values] + pd.to_timedelta(
        rng.integers(8 * 3600, 20 * 3600, len(msgs)), unit="s")
    msgs = msgs[["msg_id", "client_id", "ts", "channel", "text", "complaint_flag"]]

    # ------------------------------------------------------------ survey (NPS)
    srows, sid = [], 0
    for sd in SURVEY_DATES:
        sdt = pd.Timestamp(sd)
        for i in range(n):
            if pd.Timestamp(join[i]) > sdt or (not pd.isna(exit_date[i]) and pd.Timestamp(exit_date[i]) <= sdt):
                continue
            a = arche[i]
            d = (pd.Timestamp(exit_date[i]) - sdt).days if not pd.isna(exit_date[i]) else 9999
            p = {"stayer": 0.45, "false_alarm": 0.35, "vocal": 0.5, "gradual_fade": 0.25,
                 "money_drain": 0.3}.get(a, 0.03 if d < 200 else 0.3)
            sid += 1
            if rng.random() < p:
                lo, hi = {"vocal": (0, 5), "gradual_fade": (3, 8), "silent_exit": (5, 9),
                          "money_drain": (3, 8)}.get(a, (6, 10)) if d < 200 else (5, 10)
                srows.append((sid, i + 1, sdt, sdt + pd.Timedelta(days=int(rng.integers(1, 15))),
                              int(rng.integers(lo, hi + 1))))
            else:
                srows.append((sid, i + 1, sdt, pd.NaT, np.nan))
    survey = pd.DataFrame(srows, columns=["survey_id", "client_id", "sent_date", "answered_date", "nps_score"])

    names = [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(n)]
    clients = pd.DataFrame({
        "client_id": np.arange(1, n + 1), "full_name": names,
        "birth_date": pd.to_datetime(birth).date, "gender": gender, "country": country,
        "join_date": pd.to_datetime(join).date, "marketing_opt_out": opt_out})
    balances = pd.DataFrame(bal_rows, columns=["client_id", "month_end", "balance", "num_products",
                                               "credit_score", "has_card", "is_active_member",
                                               "est_salary", "status"])
    balances["month_end"] = pd.to_datetime(balances["month_end"]).dt.date
    exits = pd.DataFrame({"client_id": np.where(leaver)[0] + 1,
                          "exit_date": pd.to_datetime(exit_date[leaver]).date,
                          "exit_channel": rng.choice(["branch", "mobile_app", "call"], leaver.sum())})
    truth = pd.DataFrame({"client_id": np.arange(1, n + 1), "archetype": arche,
                          "exit_date": pd.to_datetime(exit_date).date})
    truth_m = pd.DataFrame(truth_msgs, columns=["msg_id", "true_sentiment", "noise_type"])
    return {"core_clients": clients, "core_accounts_monthly": balances, "core_exits": exits,
            "card_transactions": tx, "contact_messages": msgs, "survey_responses": survey,
            "sim_truth": truth, "sim_truth_messages": truth_m}


def make_messy(dfs: dict[str, pd.DataFrame], seed: int) -> dict[str, pd.DataFrame]:
    """Another branch's raw export: duplicates, wrong formats, impossible values, garbled text."""
    rng = np.random.default_rng(seed)
    out = {k: v.copy() for k, v in dfs.items()}
    c = out["core_clients"]
    c["country"] = c["country"].map(lambda x: rng.choice({
        "MD": ["MD", "Moldova", "moldova ", "R. Moldova"], "RO": ["RO", "Romania", "România"],
        "UA": ["UA", "Ukraine", "Ucraina"]}[x]))
    c["gender"] = c["gender"].map(lambda g: rng.choice([g, g.lower(), {"F": "Female", "M": "Male"}[g]]))
    c["birth_date"] = [pd.Timestamp(b).strftime("%d.%m.%Y") if rng.random() < 0.5 else str(b)
                       for b in c["birth_date"]]
    bad = rng.choice(len(c), max(1, len(c) // 200), replace=False)
    c.loc[bad, "birth_date"] = "01.01.2090"  # impossible (negative age)
    out["core_clients"] = pd.concat([c, c.sample(frac=0.01, random_state=seed)], ignore_index=True)
    t = out["card_transactions"]
    t["amount"] = [str(a).replace(".", ",") if rng.random() < 0.3 else str(a) for a in t["amount"]]
    t["ts"] = [pd.Timestamp(x).strftime("%d.%m.%Y %H:%M:%S") if rng.random() < 0.3 else str(x)
               for x in t["ts"]]
    out["card_transactions"] = pd.concat([t, t.sample(frac=0.02, random_state=seed)], ignore_index=True)
    mm = out["contact_messages"]
    g = rng.random(len(mm)) < 0.05
    mm.loc[g, "text"] = ["".join(rng.choice(list("#@%&*xqz?"), 12)) for _ in range(g.sum())]
    b = out["core_accounts_monthly"]
    b["balance"] = [str(x).replace(".", ",") if rng.random() < 0.2 else x for x in b["balance"]]
    huge = rng.choice(len(b), 5, replace=False)
    b.loc[huge, "balance"] = 9.9e12  # impossible value
    out["core_accounts_monthly"] = b
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", action="store_true", help="generate the external messy branch export")
    args = ap.parse_args()
    if args.branch:
        dfs = simulate(config.N_CLIENTS_BRANCH, config.SEED + 1000,
                       shift={"age": 6, "country_p": [0.35, 0.25, 0.40], "digital": -0.5})
        dfs = make_messy(dfs, config.SEED + 1000)
        out = config.RAW_BRANCH
    else:
        dfs = simulate(config.N_CLIENTS, config.SEED)
        out = config.RAW
    for name, df in dfs.items():
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"{name:26s} {len(df):>9,d} rows -> {out / (name + '.csv')}")


if __name__ == "__main__":
    main()
