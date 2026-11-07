"""LIVE DASHBOARD (lesson 5: rtap_app.py) — 5 windows + Control:
Producer | Kafka | Spark | Alert | Verdict | Control

  streamlit run realtime/rtap_app.py
Reads only the files the services write (state/*.json, output/*.csv) and the broker statistics,
so it can be opened, closed and reopened at any time without disturbing the stream.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from realtime.bus import get_bus, read_state, run_info, write_state  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from app import ui  # noqa: E402
from ds import viz  # noqa: E402

st.set_page_config(page_title="RTAP churn — live", layout="wide",
                   page_icon=str(ui.ASSETS / "emergency.png") if (ui.ASSETS / "emergency.png").exists() else "🚨")
ui.setup("emergency.png")
OUT = config.RT_OUT


def csv(name):
    p = OUT / name
    try:
        return pd.read_csv(p) if p.exists() else pd.DataFrame()
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


@st.cache_data
def truth():
    return pd.read_csv(config.RAW / "sim_truth.csv", parse_dates=["exit_date"])


def evaluate_alerts(al: pd.DataFrame, upto: str) -> dict:
    t = truth()
    s = pd.Timestamp(run_info().get("start_day", config.STREAM_START))
    e = pd.Timestamp(upto)
    lv = t[(t.exit_date >= s) & (t.exit_date <= e)]
    if al.empty or lv.empty:
        return {"leavers_so_far": len(lv), "caught": 0, "recall": np.nan, "median_lead": np.nan, "precision": np.nan}
    al = al.assign(alert_day=pd.to_datetime(al["alert_day"]))
    first = al.groupby("client_id")["alert_day"].min()
    m = lv.assign(alert=lv.client_id.map(first))
    caught = m.alert.notna() & (m.alert < m.exit_date)
    ex = t.set_index("client_id")["exit_date"]
    will = al.client_id.map(ex).notna() & (al.client_id.map(ex) > al.alert_day)
    return {"leavers_so_far": len(lv), "caught": int(caught.sum()), "recall": caught.mean(),
            "median_lead": float(((m.exit_date - m.alert).dt.days[caught]).median()) if caught.any() else np.nan,
            "precision": float(will.mean())}


info = run_info()
bus_kind = (read_state("scorer.json", {}) or {}).get("bus", "file")
ui.hero("RTAP live — Loialitate, tăcere și plecare",
        f"run {info.get('run_id', '-')} · start {info.get('start_day', '-')} · {info.get('seconds_per_day', '-')} s per bank day · "
        f"bus = {bus_kind}", icon="emergency.png")
ui.credits()
tabs = st.tabs(["1 · Producer", "2 · Kafka", "3 · Spark", "4 · Alert", "5 · Verdict", "Control"])


@st.fragment(run_every=2)
def live():
    sc = read_state("scorer.json", {}) or {}
    with tabs[0]:
        st.subheader("Sources (one producer per bank system)")
        rows = [read_state(f"producer_{s}.json", {"source": s}) for s in ("mobile_app", "card_branch", "contact_center")]
        dfp = pd.DataFrame(rows)
        if "updated" in dfp:
            dfp["seconds_ago"] = (time.time() - dfp["updated"]).round(1)
            dfp = dfp.drop(columns=["updated"])
        ui.cards([(r.get("source", "-"), f"{int(r.get('events_sent', 0)):,}",
                   f"bank day {r.get('bank_day', '–')} · {r.get('events_today', 0)} today") for r in rows])
        st.dataframe(dfp, width="stretch", hide_index=True)
        st.caption("Each producer can run on another laptop (--bootstrap <IP>:9092). day_end heartbeats keep the bank clock.")
    with tabs[1]:
        st.subheader("Topics, offsets, consumer groups and lag")
        rid = info.get("run_id", "manual")
        groups = {"bank.events": [f"scorer-{rid}", f"river-{rid}"], "bank.alerts": [f"rtap-{rid}"],
                  "bank.offers": [f"world-{rid}"], "bank.outcomes": [f"rtap-out-{rid}"]}
        try:
            st.dataframe(pd.DataFrame(get_bus(bus_kind).stats(groups)), width="stretch", hide_index=True)
        except Exception as e:  # noqa: BLE001
            st.warning(f"broker not reachable: {e}")
        ui.cards([("Events scored", f"{sc.get('events', 0):,}", f"bank days closed: {sc.get('days_closed', 0)}"),
                  ("Latency p50", f"{sc.get('latency_e2e_p50_ms', float('nan')):.0f} ms", "producer → score"),
                  ("Latency p95", f"{sc.get('latency_e2e_p95_ms', float('nan')):.0f} ms",
                   f"contract < {config.CONTRACT['latency_p95_ms_max']:.0f} ms")])
    with tabs[2]:
        st.subheader("Spark Structured Streaming windows (CSV 'book of records')")
        for name, title in [("bank_day", "event-time window = 1 bank day (with watermark)"),
                            ("proc_10s", "processing-time window = 10 s")]:
            files = sorted((OUT / "spark" / name).glob("*.csv"))
            parts = []
            for f in files[-300:]:
                try:
                    parts.append(pd.read_csv(f))
                except Exception:  # noqa: BLE001
                    pass
            st.markdown(f"**{title}**")
            if parts:
                d = pd.concat(parts)
                st.dataframe(d.tail(20), width="stretch", hide_index=True)
                if name == "bank_day" and "transactions" in d:
                    g = d.groupby("bank_day")[["transactions", "messages", "complaints"]].sum().reset_index().sort_values("bank_day")
                    ui.chart(viz.lines(g, "bank_day", ["transactions", "messages", "complaints"], "Per bank day (event-time windows)"),
                             key=f"spark_{name}")
            else:
                st.caption("no window closed yet (Spark writes a window when the watermark passes it)")
        daily = csv("daily.csv")
        if not daily.empty:
            st.markdown("**Scorer clock path per closed day** (all active clients rescored, even silent ones)")
            ui.chart(viz.lines(daily, "day", ["clients_over_threshold", "alerts"], "Clients over the threshold vs alerts sent (capacity)"),
                     key="daily")
    with tabs[3]:
        st.subheader("Alerts (EWMA ≥ threshold from the cost matrix, within daily capacity)")
        al = csv("alerts.csv")
        ui.cards([("Bank day", str(sc.get("next_day", "-")), "currently open"), ("Alerts total", f"{len(al):,}", "EWMA ≥ threshold"),
                  ("Threshold", f"{sc.get('threshold', float('nan')):.2f}", "from the cost matrix")])
        if not al.empty:
            st.dataframe(al[["alert_day", "client_id", "risk", "reason", "top_features", "offer_suggested"]].tail(25)
                         .iloc[::-1], width="stretch", hide_index=True)
            pick = st.selectbox("open one alert", al["client_id"].tail(50).iloc[::-1].tolist(), key="alert_pick")
            row = al[al.client_id == pick].iloc[-1]
            a_, b_ = st.columns([1, 2])
            with a_:
                ui.cards([(f"Client {pick}", f"{row['risk']:.0%}", f"reason: {row['reason']}")])
                st.caption(f"top SHAP features: {row['top_features']}")
            b_.info(row["rag_text"])
            ui.chart(viz.hbar(al["reason"].value_counts().astype(float), "Alerts by reason", color="#7469B6", fmt=".0f"), key="reasons")
        riv = read_state("river.json")
        if riv:
            st.markdown("**Online learning (River HalfSpaceTrees, idea of app_river.py):** score each client-day, then "
                        "learn from it. Do the unusual days point to future leavers?")
            base, share = riv.get("base_rate_future_leavers"), riv.get("share_future_leavers_among_flagged")
            ui.cards([("Days learned", f"{riv.get('days', 0)}", f"warm-up {riv.get('warmup_days', '–')} days"),
                      ("Clients flagged", f"{riv.get('flagged', 0):,}", "unusual day, incl. silence"),
                      ("Future leavers among flagged", f"{share:.0%}" if share is not None else "–",
                       f"vs {base:.0%} among all clients" if base is not None else "")])
    with tabs[4]:
        st.subheader("Verdict against the contract matrix")
        al = csv("alerts.csv")
        upto = sc.get("next_day", info.get("start_day", config.STREAM_START))
        ev = evaluate_alerts(al, upto)
        rt = read_state("rtap.json", {}) or {}
        offers = csv("offers.csv")
        rows = [("M1 recall of leavers (alert before exit)", ">= 0.70", ev["recall"], ev["recall"] >= 0.70 if ev["recall"] == ev["recall"] else False),
                ("B3 median lead time (days)", ">= 30", ev["median_lead"], ev["median_lead"] >= 30 if ev["median_lead"] == ev["median_lead"] else False),
                ("Streaming latency p95 (ms)", f"< {config.CONTRACT['latency_p95_ms_max']:.0f}", sc.get("latency_e2e_p95_ms", np.nan),
                 (sc.get("latency_e2e_p95_ms") or 1e9) < config.CONTRACT["latency_p95_ms_max"]),
                ("B4 ROI of offers", "> 1", rt.get("ROI"), (rt.get("ROI") or 0) > 1)]
        if not offers.empty:
            sh = pd.crosstab(offers["gender"], offers["offer"], normalize="index")
            gap = float((sh.max() - sh.min()).max()) if len(sh) > 1 else 0.0
            rows.append(("Offer mix difference between genders (ethical rule)", "<= 0.10", gap, gap <= 0.10))
        v = pd.DataFrame(rows, columns=["requirement", "threshold", "measured", "pass"])
        v["verdict"] = np.where(v["pass"], "PASS", "FAIL")
        st.markdown(f"### Overall {ui.verdict_pill(bool(v['pass'].all()))}", unsafe_allow_html=True)
        v["measured"] = pd.to_numeric(v["measured"], errors="coerce")
        sty = v.drop(columns=["pass"]).style.format({"measured": "{:.2f}"}, na_rep="–").map(lambda x: "background-color:#DDF3EA;color:#1E6E4C;font-weight:700" if x == "PASS"
                                                 else ("background-color:#FFE1E8;color:#B0123F;font-weight:700" if x == "FAIL" else ""),
                                                 subset=["verdict"])
        st.dataframe(sty, width="stretch", hide_index=True)
        st.markdown(f"{'' if v['pass'].all() else 'FAIL → back to box 1 of the cycle · '}"
                    f"leavers so far {ev['leavers_so_far']}, caught {ev['caught']}, alert precision {ev['precision']:.2f}"
                    if ev['precision'] == ev['precision'] else "waiting for data")
        st.markdown("**RTAP — what Thompson Sampling learned (posterior success rate per reason × offer)**")
        if rt.get("posteriors"):
            post = pd.DataFrame(rt["posteriors"]).pivot(index="reason", columns="offer", values="posterior_mean")
            import plotly.graph_objects as go
            ui.chart(go.Figure(go.Heatmap(z=post.values, x=post.columns, y=post.index, zmin=0, zmax=1, texttemplate="%{z:.2f}",
                                          colorscale=[[0, "#F5EFF8"], [1, "#8A2A78"]], textfont=dict(size=15)))
                     .update_layout(title="P(client stays | reason, offer) learned so far", height=380), key="ts")
            roi = rt.get("ROI")
            ui.cards([("Offers sent", f"{rt.get('offers', 0):,}", f"{rt.get('skipped_capacity_or_budget', 0)} skipped (capacity / budget)"),
                      ("Spent", f"{rt.get('spent', 0):,.0f} €", "offer cost"),
                      ("Clients saved", f"{rt.get('saved', 0):,}", f"{rt.get('lost_after_offer', 0):,} left anyway"),
                      ("ROI", f"{roi:.2f}" if roi is not None else "–", "value kept / cost")])
        sha = csv("shadow_alerts.csv")
        if not sha.empty:
            st.markdown("**Champion vs challengers in shadow mode (L1 / L2 / ElasticNet)**")
            comp = [{"model": "champion", **evaluate_alerts(al, upto)}]
            for ch, g in sha.groupby("challenger"):
                comp.append({"model": ch, **evaluate_alerts(g, upto)})
            st.dataframe(pd.DataFrame(comp), width="stretch", hide_index=True)


live()

with tabs[5]:
    st.subheader("Control (read by the scorer and RTAP within a second)")
    ctrl = read_state("control.json", {}) or {}
    cap = st.slider("retention team capacity (alerts / offers per bank day)", 5, 200, int(ctrl.get("capacity", 40)))
    budget = st.number_input("offer budget (EUR)", 0, 1_000_000, int(ctrl.get("budget", 200000)), step=5000)
    pause = st.checkbox("pause automatic offers (human review only)", bool(ctrl.get("pause", False)))
    ov = st.text_area("human override: client_id=offer per line (offer 'none' cancels)",
                      "\n".join(f"{k}={v}" for k, v in (ctrl.get("overrides") or {}).items()))
    if st.button("Apply", type="primary"):
        overrides = {}
        for line in ov.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                overrides[k.strip()] = None if v.strip() == "none" else v.strip()
        write_state("control.json", {"capacity": cap, "budget": budget, "pause": pause, "overrides": overrides})
        st.success("saved")