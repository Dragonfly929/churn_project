"""THE CYCLE APP — one page per box of the board's cycle, each with its result AND its logic/code.

  streamlit run app/streamlit_app.py        (run it from the project folder: the theme is in .streamlit/)

Run `python run_all.py` once before (it builds the DWH, the datasets and the first iterations).
Charts are interactive (Plotly): hover for values, drag to zoom, double-click to reset, click a legend item to hide it.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))
import config  # noqa: E402
from app import ui  # noqa: E402
from ds import catalog, cycle, eda, modeling as M, viz, xai  # noqa: E402
from ds.catalog import ALL_FEATURES  # noqa: E402

st.set_page_config(page_title="Loialitate, tăcere și plecare", layout="wide",
                   page_icon=str(ui.ASSETS / "churn.png") if (ui.ASSETS / "churn.png").exists() else "🏦")
ui.setup("churn.png")
chart = ui.chart

PAGES = ["0 · The cycle", "1 · Problem & KPIs", "2 · Data source (DWH + query)", "3 · EDA", "4 · Contract matrix",
         "5 · Features", "6 · Model training", "7 · Decision boundaries", "8 · XAI (SHAP, LIME)",
         "9 · External raw data", "10 · Streaming / RTAP", "11 · RAG", "12 · Verdict & iterations",
         "★ Six fundamentals", "★ Lab experiments", "★ Monitoring & model card"]


def code(*objs, title="Logic & code (how we got here)"):
    with st.expander(title):
        for o in objs:
            if isinstance(o, str):
                st.code(o, language="sql" if "SELECT" in o else "python")
            else:
                st.code(inspect.getsource(o), language="python")


def need_data():
    if not (config.DATASETS / f"bank_{config.CUTOFF}.csv").exists():
        st.error("No dataset yet. Run once in a terminal:  `python run_all.py --fast`  (builds the DWH and the datasets).")
        st.stop()


def show_contract(cm: pd.DataFrame):
    ok = bool(cm["pass"].all())
    st.markdown(f"**Verdict:** {ui.verdict_pill(ok)}  &nbsp; {int(cm['pass'].sum())} of {len(cm)} rows pass",
                unsafe_allow_html=True)
    sty = cm.drop(columns=["pass"]).style.format({"measured": "{:.3f}"}).map(
        lambda v: "background-color:#DDF3EA;color:#1E6E4C;font-weight:700" if v == "PASS"
        else ("background-color:#FFE1E8;color:#B0123F;font-weight:700" if v == "FAIL" else ""), subset=["verdict"])
    st.dataframe(sty, hide_index=True, width="stretch")


# ------------------------------------------------------------------ state
ss = st.session_state
ss.setdefault("contract", dict(config.CONTRACT))
ss.setdefault("costs", {"fn": config.COST_FN, "fp": config.COST_FP})
ss.setdefault("features", [f for f in ALL_FEATURES])
ss.setdefault("cfg", {"model": "gb", "params": {}, "balancing": "none", "threshold_mode": "cost",
                      "manual_threshold": 0.5, "per_group_thresholds": False, "group_col": "age_group",
                      "noise_x_sigma": 0.0, "noise_y_flip": 0.0, "bias_y_silent": 0.0, "bias_x_group": "",
                      "bias_x_share": 0.0})

# ------------------------------------------------------------------ sidebar
st.sidebar.markdown("## Loialitate, tăcere și plecare")
page = st.sidebar.radio("Box of the cycle", PAGES)
from stage1 import github_load  # noqa: E402
SRC_BANK, SRC_EXT, SRC_KAG = "simulated bank (DWH)", "external branch (raw export)", "github (external validation)"
sources = [SRC_BANK, SRC_EXT] + ([SRC_KAG] if github_load.available() else [])
src = st.sidebar.selectbox("Data source", sources, help="see page 2 for what each source is and why it exists")
cutoff = st.sidebar.date_input("Cutoff date (features only up to it)", pd.Timestamp(config.CUTOFF)).strftime("%Y-%m-%d")
leaky = st.sidebar.checkbox("⚠ wrong cutoff (leakage demo)", False)
st.sidebar.caption(f"Engine: **{config.DB_BACKEND}** · label = leaves within 90 days after the cutoff")
ui.credits()


@st.cache_data(show_spinner="Running the Stage 1 query on the DWH ...")
def get_dataset(src: str, cutoff: str, leaky: bool) -> pd.DataFrame:
    from stage1.extract_dataset import run_query
    if src.startswith("github"):
        k = pd.read_csv(config.DATASETS / "github.csv") if (config.DATASETS / "github.csv").exists() else github_load.load()
        k["country"] = k["geography"].astype(str)
        k["age_group"] = M.age_group(k["age"])
        k["age_country"] = k["age_group"] + "|" + k["country"]
        k["silent"] = 0
        return k
    source = "branch" if src.startswith("external") else "bank"
    label = (pd.Timestamp(cutoff) + pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    fpath = config.DATASETS / f"{source}_{cutoff}.csv"
    if not leaky and fpath.exists():
        return M.add_groups(pd.read_csv(fpath))
    return M.add_groups(run_query(cutoff, label, config.LEAKY_FEATURE_CUTOFF if leaky else cutoff, 1, source))


need_data()
df = get_dataset(src, cutoff, leaky)
avail = [f for f in ss["features"] if f in df.columns]
NUM = [f for f in ALL_FEATURES if f in df.columns and f not in config.CATEGORICAL]


def current_cfg() -> M.RunConfig:
    c = ss["cfg"]
    return M.RunConfig(model=c["model"], params=c["params"], balancing=c["balancing"], features=avail,
                       threshold_mode=c["threshold_mode"], manual_threshold=c["manual_threshold"],
                       cost_fn=ss["costs"]["fn"], cost_fp=ss["costs"]["fp"],
                       per_group_thresholds=c["per_group_thresholds"], group_col=c["group_col"],
                       noise_x_sigma=c["noise_x_sigma"], noise_y_flip=c["noise_y_flip"],
                       bias_y_silent=c["bias_y_silent"], bias_x_group=c["bias_x_group"], bias_x_share=c["bias_x_share"],
                       min_recall=ss["contract"]["M1_recall_min"] + 0.04)


@st.cache_resource(show_spinner="Training (cross-validation inside the training part) ...", max_entries=12)
def train(cfg_json: str, src: str, cutoff: str, leaky: bool):
    cfg = M.RunConfig(**json.loads(cfg_json))
    return M.run(cfg, get_dataset(src, cutoff, leaky))


def trained():
    cfg = current_cfg()
    if src.startswith("github"):
        st.warning("github has no behaviour/silence features; only the profile features are used.")
    return train(json.dumps(cfg.to_dict(), sort_keys=True, default=str), src, cutoff, leaky)


def model_badge(r):
    m = r.metrics
    ui.cards([("ROC-AUC", f"{m['roc_auc']:.3f}", f"CV {r.cv_auc:.3f}"),
              ("Recall", f"{m['recall']:.0%}", "leavers caught"),
              ("Precision", f"{m['precision']:.0%}", "alerts that are right"),
              ("Threshold", f"{r.threshold:.2f}", r.cfg.threshold_mode),
              ("Expected cost", f"{m['expected_cost'] / 1000:,.0f} k€", f"vs {m['cost_no_model'] / 1000:,.0f} k€ without model")])


# ================================================================== pages
title = page.split("·")[-1].strip()
if not page.startswith("0"):
    st.title(title)

# ------------------------------------------------------------------ 0
if page.startswith("0"):
    ui.hero("Loialitate, tăcere și plecare",
            f"Analiza churn-ului clienților unei bănci · {catalog.PROBLEM['question']}", icon="churn.png")
    leavers = df[df.churn == 1] if "churn" in df else df
    if src == SRC_KAG:  # a static public file: no dates, no messages, so no 90-day window and no silence
        churn_card = ("Left the bank", f"{df['churn'].mean():.1%}", f"{int(df['churn'].sum()):,} clients · no time window")
        silent_card = ("Silent leavers", "n/a", "no messages or dates in this file")
    else:
        churn_card = ("Leave within 90 days", f"{df['churn'].mean():.1%}", f"{int(df['churn'].sum()):,} clients")
        silent_card = ("Silent leavers", f"{leavers['silent'].mean():.0%}" if "silent" in leavers and len(leavers) else "–",
                       "no messages 6 mo · no survey")
    ui.cards([("Clients", f"{len(df):,}", f"{src}"), churn_card, silent_card,
              ("Iterations saved", f"{len(cycle.load_log())}", "page 12")])
    st.write("")
    ui.cycle_diagram()
    st.markdown("Every page shows the result **and** the logic behind it. Change anything on pages 2–6, look at the "
                "verdict on page 12, save the iteration with a reason, and go back to box 1.  \n"
                "Lifecycle CV_ȘAD: **I** planning (1–2) · **II** cleaning + EDA (2–3) · **III** features, modelling, "
                "evaluation (4–9) · **IV** implementation + monitoring (10–★).")
    log = cycle.load_log()
    if log:
        chart(viz.iterations_chart(log))

# ------------------------------------------------------------------ 1
elif page.startswith("1 "):
    P = catalog.PROBLEM
    c1, c2 = st.columns([3, 2])
    c1.markdown(f"### Problem\n{P['problem']}\n\n### Goal\n{P['goal']}")
    c2.markdown("### Objectives\n" + "\n".join(f"- {o}" for o in P["objectives"]))
    st.markdown("### Hypotheses\n" + "\n".join(f"- {h}" for h in P["hypotheses"]))
    if not src.startswith("github"):
        k = eda.kpis_and_hypotheses(df)
        ui.cards([("B1 · churn rate (90 days)", f"{k['B1 churn rate (90d)']:.1%}", "leaves after the cutoff"),
                  ("B2 · silent leavers", f"{k['B2 share of silent leavers']:.0%}", "of all leavers"),
                  ("Silent clients overall", f"{k['share of silent clients overall']:.0%}", "of active clients")])
        st.info(k["H1 churn rate silent vs not"])
    st.subheader("KPIs — 2 for the model (M1, M2), 2 for the features (F1, F2), plus business KPIs")
    st.dataframe(catalog.KPIS, hide_index=True, width="stretch")
    code(catalog)

# ------------------------------------------------------------------ 2
elif page.startswith("2"):
    from stage1 import db
    source = "branch" if src.startswith("external") else ("github" if src.startswith("github") else "bank")
    st.subheader("Where the data comes from")
    st.markdown(
        '<div class="src"><b>1 · Simulated bank — the DWH (main source)</b><br>'
        "Five source systems (core banking, cards & payments, mobile app, contact centre, NPS survey) simulated from the "
        "data contract → staging → cleaning → star schema → <b>the query</b> with a cutoff date. Every model, chart and "
        "the live stream use this. It is a <b>snapshot</b> at the cutoff (static); the 90 days after it are replayed as "
        "events by the streaming part.</div>"
        '<div class="src ext"><b>2 · External branch — raw export of ANOTHER branch (not github)</b><br>'
        "A second, different population (older, more clients from UA, less digital) delivered as a messy file: duplicates, "
        "dd.mm.yyyy dates, comma decimals, the country spelled five ways, impossible values, garbled text. It goes through "
        "the same cleaning, the same query and the same model. Purpose: XAI pass 2 and robustness — does the model and its "
        "explanation survive data from outside the warehouse? (drift / PSI, KPI F2).</div>"
        '<div class="src opt"><b>3 · github — real public data, external validation only</b><br>'
        "Appears when <code>Churn_Modelling.csv</code> is in <code>data/external/</code>. It goes through staging and the "
        "same query, restricted to the profile features it shares with the bank; used for the data-sources comparison "
        "(step 6.4), never as the starting point. It has no messages or dates, so silence cannot be measured on it.</div>",
        unsafe_allow_html=True)
    st.subheader("Star schema (Kimball): facts around dimensions")
    st.graphviz_chart("""digraph S { rankdir=LR; bgcolor="transparent"; node [shape=record, fontname=Helvetica, fontsize=11, color="#8A2A78", fontcolor="#252A34"];
      dc [label="{dim_client|PK client_key\\lclient_id\\lbirth_date\\lgender\\lcountry\\ljoin_date\\l}", style=filled, fillcolor="#F5EFF8"];
      dd [label="{dim_date|PK date_key\\lfull_date\\lyear, quarter, month\\l}", style=filled, fillcolor="#F5EFF8"];
      ch [label="{dim_channel|PK channel_key\\lchannel_name\\lis_digital\\l}", style=filled, fillcolor="#F5EFF8"];
      fa [label="{fact_account_month|FK client_key, date_key\\lbalance, num_products\\lcredit_score, status\\l}", style=filled, fillcolor="#E6F7F9", color="#1A8FB0"];
      ft [label="{fact_transaction|PK tx_id\\lFK client, date, channel\\lamount, direction\\l}", style=filled, fillcolor="#E6F7F9", color="#1A8FB0"];
      ff [label="{fact_feedback|PK msg_id\\lFK client, date, channel\\ltext_raw, text_clean\\lsentiment, complaint\\l}", style=filled, fillcolor="#E6F7F9", color="#1A8FB0"];
      fs [label="{fact_survey|PK survey_id\\lFK client, sent/answered date\\lnps_score\\l}", style=filled, fillcolor="#E6F7F9", color="#1A8FB0"];
      fe [label="{fact_client_exit|PK/FK client_key\\lFK exit_date_key\\l}", style=filled, fillcolor="#FFE1E8", color="#FF2E63"];
      edge [color="#7469B6"]; fa->dc; fa->dd; ft->dc; ft->dd; ft->ch; ff->dc; ff->dd; ff->ch; fs->dc; fs->dd; fe->dc; fe->dd; }""")
    st.caption("Why a star schema: made for analytical queries with time windows (OLAP). A normalised OLTP schema is "
               "made for updates; a snowflake adds joins for little gain; MongoDB is good for raw events but weak at joins and windows.")
    if source != "github":
        tabs_ = ["dim_client", "dim_date", "dim_channel", "fact_account_month", "fact_transaction", "fact_feedback",
                 "fact_survey", "fact_client_exit"]
        counts = {t: int(db.read_sql(f"SELECT COUNT(*) AS n FROM {t}", source=source)["n"][0]) for t in tabs_}
        chart(viz.table_counts(counts))
        t = st.selectbox("Look at raw rows of a table", tabs_, index=4)
        st.dataframe(db.read_sql(f"SELECT * FROM {t} LIMIT 20", source=source), width="stretch", hide_index=True)
    st.subheader("The query (Stage 1.6)")
    st.caption(":cutoff = eligibility and last date of features · :label_cutoff = end of the 90-day target window · "
               ":feature_cutoff = :cutoff (the leakage demo moves it after the departures) · :denoise")
    with st.expander("Show the SQL", expanded=False):
        st.code(db.sql_file("query_features") if source != "github" else github_load.QUERY, language="sql")
    st.subheader("The dataset it produced")
    ui.cards([("Rows (clients)", f"{len(df):,}", f"cutoff {cutoff}"),
              ("Features", f"{len([c for c in df.columns if c in ALL_FEATURES])}", "profile · behaviour · voice · silence"),
              ("Churn share", f"{df['churn'].mean():.1%}", "leave within 90 days")])
    st.write("")
    e1, e2 = st.columns(2)
    fx = e1.selectbox("Explore a feature", NUM, index=NUM.index("tx_trend") if "tx_trend" in NUM else 0)
    with e1:
        chart(viz.hist_by_churn(df, fx))
    with e2:
        ax = st.selectbox("x axis", NUM, index=NUM.index("balance_change_3m") if "balance_change_3m" in NUM else 0)
        ay = st.selectbox("y axis", NUM, index=NUM.index("days_since_last_msg") if "days_since_last_msg" in NUM else 1)
        chart(viz.scatter_by_churn(df, ax, ay))
    with st.expander("Raw rows of the dataset"):
        st.dataframe(df.head(200), width="stretch", hide_index=True)
    if source != "github":
        st.subheader("Data contract check (input validity)")
        st.dataframe(catalog.validate_contract(df), hide_index=True)
    st.subheader("Leakage trap")
    if src == SRC_KAG:
        st.info("The public dataset has no dates, so there is no cutoff to get wrong: choose the simulated bank or the "
                "external branch in the sidebar to run the leakage demo.")
    elif st.button("Run the leakage demo (same model, correct vs WRONG cutoff)", type="primary"):
        lk = get_dataset(src, cutoff, True)
        res = eda.leakage_demo(get_dataset(src, cutoff, False), lk)
        chart(viz.grouped_bars(res.reset_index().rename(columns={"index": "setup"}), "setup", ["roc_auc", "recall", "top_feature_share"],
                               "Correct vs wrong cutoff", pct=True))
        st.dataframe(res, width="stretch")
        st.warning("With the wrong cutoff the AUC is ~1.0 and one feature (the closed account's balance) carries the "
                   "model: features must use data ≤ cutoff, the target only data after it.")
    p = config.TABLES / f"cleaning_report_{source}.csv"
    if p.exists():
        st.subheader("Cleaning rules applied (step 2.0)")
        st.dataframe(pd.read_csv(p), hide_index=True, width="stretch")
    code(get_dataset)

# ------------------------------------------------------------------ 3
elif page.startswith("3"):
    if src.startswith("github"):
        st.info("EDA is designed for the DWH sources."); st.stop()
    st.markdown(f"**Step 0 — unique identifier:** `client_id`, {df['client_id'].nunique():,} unique of {len(df):,} rows "
                f"→ {'one row per client ✔' if df['client_id'].is_unique else 'DUPLICATES ✘'}")
    t1, t2, t3, t4, t5, t6 = st.tabs(["1 · Missing", "2 · Types", "3 · Shape", "4 · Correlations", "5 · Outliers", "KPIs & hypotheses"])
    with t1:
        mi = eda.step1_missing(df)
        if len(mi):
            chart(viz.hbar(mi["share"], "Share of missing values (and why — table below)", fmt=".1%", height=260))
        st.dataframe(mi, width="stretch")
    with t2:
        st.dataframe(eda.step2_types(df), width="stretch", hide_index=True)
    with t3:
        f1 = st.selectbox("feature", NUM, index=NUM.index("tx_trend") if "tx_trend" in NUM else 0, key="eda_f")
        a, b = st.columns(2)
        with a:
            chart(viz.hist_by_churn(df, f1))
        with b:
            chart(viz.box_by_churn(df, f1))
        st.markdown("**Churn rate by segment** (dashed line = overall)")
        segs = st.multiselect("segments", ["age_group", "country", "gender", "num_products", "silent", "is_active_member", "has_card"],
                              default=["age_group", "country", "num_products", "silent"])
        cols = st.columns(2)
        for i, s_ in enumerate(segs):
            with cols[i % 2]:
                chart(viz.rate_by_segment(df, s_, df.churn.mean()), key=f"seg_{s_}")
        _, low = eda.step3_shape(df)
        st.caption(f"near-zero-variance features: {low or 'none'}")
    with t4:
        tgt, corr, prox = eda.step4_correlations(df)
        a, b = st.columns([2, 3])
        with a:
            chart(viz.hbar(tgt, "Spearman ρ with churn (pink = more churn)", signed=True, fmt=".2f"))
        with b:
            chart(viz.corr_heatmap(corr))
        if len(prox):
            st.warning("Possible proxies for a sensitive attribute:"); st.dataframe(prox, hide_index=True)
    with t5:
        st.dataframe(eda.step5_outliers(df), width="stretch", hide_index=True)
    with t6:
        st.json(eda.kpis_and_hypotheses(df))
    code(eda)

# ------------------------------------------------------------------ 4
elif page.startswith("4"):
    st.markdown("The contract matrix is the acceptance test, agreed **before** training. Move the thresholds: "
                "pages 6, 8, 9 and 12 re-check every row.")
    c = ss["contract"]
    c1, c2, c3 = st.columns(3)
    c["M1_recall_min"] = c1.slider("M1 recall of leavers ≥", 0.3, 0.95, float(c["M1_recall_min"]), 0.01)
    c["M2_auc_min"] = c1.slider("M2 ROC-AUC ≥", 0.5, 0.95, float(c["M2_auc_min"]), 0.01)
    c["F1_jaccard_min"] = c2.slider("F1 SHAP top-5 Jaccard ≥", 0.2, 1.0, float(c["F1_jaccard_min"]), 0.05)
    c["F2_rankcorr_min"] = c2.slider("F2 external SHAP rank corr ≥", 0.3, 1.0, float(c["F2_rankcorr_min"]), 0.05)
    c["F2_psi_max"] = c2.slider("F2 max PSI <", 0.05, 0.5, float(c["F2_psi_max"]), 0.01)
    c["fairness_gap_max"] = c3.slider("Fairness: recall gap ≤", 0.02, 0.4, float(c["fairness_gap_max"]), 0.01)
    c["leakage_auc_max"] = c3.slider("Leakage alarm: AUC <", 0.9, 1.0, float(c["leakage_auc_max"]), 0.005)
    c["latency_p95_ms_max"] = c3.slider("Streaming latency p95 < (ms)", 100.0, 5000.0, float(c["latency_p95_ms_max"]), 100.0)
    st.subheader("Cost matrix (the threshold is chosen from it, not left at 0.5)")
    k1, k2 = st.columns(2)
    ss["costs"]["fn"] = k1.number_input("cost of a lost client (FN), €", 100.0, 10000.0, float(ss["costs"]["fn"]), 50.0)
    ss["costs"]["fp"] = k2.number_input("cost of a useless offer (FP), €", 1.0, 1000.0, float(ss["costs"]["fp"]), 5.0)
    fn_, fp_ = ss["costs"]["fn"], ss["costs"]["fp"]
    tp_ = fp_ + (1 - config.OFFER_SAVE_RATE) * fn_
    import plotly.graph_objects as go
    fig = go.Figure(go.Heatmap(z=[[0, fp_], [fn_, tp_]], x=["predicted: stays", "predicted: leaves"],
                               y=["actually stays", "actually leaves"], colorscale=[[0, "#F5EFF8"], [1, "#FF2E63"]],
                               showscale=False, texttemplate="<b>%{z:,.0f} €</b>", textfont=dict(size=20), hoverinfo="skip"))
    fig.update_layout(title=f"Cost per client (an offer keeps {config.OFFER_SAVE_RATE:.0%} of true leavers)", height=320,
                      yaxis_autorange="reversed")
    chart(fig)
    st.subheader("Data contract (part 1 of the matrix)")
    st.dataframe(catalog.DATA_CONTRACT, hide_index=True, width="stretch")
    code(M.contract_matrix, M.expected_cost)

# ------------------------------------------------------------------ 5
elif page.startswith("5"):
    st.markdown("Turn features on/off (inner loop of the cycle). Sensitive attributes are marked; "
                "`digital_share_90d` is a **proxy for age** (EDA step 4).")
    fdf = catalog.FEATURES_DF.copy()
    fdf["use"] = fdf["feature"].isin(ss["features"])
    fdf["sensitive"] = fdf["feature"].isin(config.SENSITIVE)
    ed = st.data_editor(fdf[["use", "feature", "group", "type", "sensitive", "formula", "source", "why"]],
                        disabled=["feature", "group", "type", "sensitive", "formula", "source", "why"],
                        hide_index=True, width="stretch", height=560)
    c1, c2, c3 = st.columns(3)
    if c1.button("Apply selection", type="primary"):
        ss["features"] = ed.loc[ed["use"], "feature"].tolist(); st.rerun()
    if c2.button("Drop sensitive attributes + proxy"):
        ss["features"] = [f for f in ALL_FEATURES if f not in config.SENSITIVE + ["digital_share_90d"]]; st.rerun()
    if c3.button("All features"):
        ss["features"] = list(ALL_FEATURES); st.rerun()
    g = catalog.FEATURES_DF.assign(used=catalog.FEATURES_DF.feature.isin(avail)).groupby("group")["used"].agg(["sum", "count"])
    chart(viz.grouped_bars(g.reset_index().rename(columns={"sum": "in use", "count": "available"}), "group",
                           ["available", "in use"], f"{len(avail)} features in use, by group"))

# ------------------------------------------------------------------ 6
elif page.startswith("6"):
    c = ss["cfg"]
    with st.container(border=True):
        a1, a2, a3 = st.columns(3)
        a1.markdown("#### Model")
        c["model"] = a1.selectbox("Model", list(M.MODELS), index=list(M.MODELS).index(c["model"]), format_func=M.MODELS.get)
        p = dict(M.DEFAULT_PARAMS[c["model"]])
        if c["model"] == "logreg":
            p["penalty"] = a1.selectbox("penalty", ["l2", "l1", "elasticnet"])
            p["C"] = a1.select_slider("C (inverse strength)", [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0], 1.0)
            p["l1_ratio"] = a1.slider("α (l1_ratio, ElasticNet)", 0.0, 1.0, 0.5)
        elif c["model"] == "rf":
            p["n_estimators"] = a1.slider("trees", 50, 500, 300, 50); p["max_depth"] = a1.slider("max_depth", 2, 20, 8)
            p["min_samples_leaf"] = a1.slider("min_samples_leaf", 1, 50, 5)
        elif c["model"] == "gb":
            p["learning_rate"] = a1.select_slider("learning rate", [0.01, 0.02, 0.05, 0.1, 0.2], 0.05)
            p["max_depth"] = a1.slider("max_depth", 2, 10, 4); p["max_iter"] = a1.slider("max boosting iterations", 50, 800, 300, 50)
        else:
            p["hidden"] = a1.text_input("hidden layers", "64,32"); p["alpha"] = a1.select_slider("L2 alpha", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1], 1e-3)
        c["params"] = p
        a2.markdown("#### Decision")
        c["balancing"] = a2.selectbox("Class balancing (SMOTE fundamental)", M.BALANCING, index=M.BALANCING.index(c["balancing"]))
        c["threshold_mode"] = a2.selectbox("Decision threshold", ["cost", "cost+recall", "0.5", "manual"],
                                           index=["cost", "cost+recall", "0.5", "manual"].index(c["threshold_mode"]))
        if c["threshold_mode"] == "manual":
            c["manual_threshold"] = a2.slider("manual threshold", 0.01, 0.99, float(c["manual_threshold"]))
        c["per_group_thresholds"] = a2.toggle("Per-group thresholds (bias mitigation)", c["per_group_thresholds"])
        c["group_col"] = a2.selectbox("group for thresholds", ["age_group", "age_country"], index=["age_group", "age_country"].index(c["group_col"]))
        a3.markdown("#### Perturb the TRAINING data")
        c["noise_x_sigma"] = a3.slider("noise on X (σ, std units)", 0.0, 2.0, float(c["noise_x_sigma"]), 0.05)
        c["noise_y_flip"] = a3.slider("noise on Y (% labels flipped)", 0.0, 0.4, float(c["noise_y_flip"]), 0.01)
        c["bias_y_silent"] = a3.slider("bias on Y: silent leavers labelled 'stayed'", 0.0, 1.0, float(c["bias_y_silent"]), 0.05)
        grp = a3.selectbox("bias on X: remove part of a group", ["", "age_group=60+", "age_group=18-29", "country=UA", "gender=F"])
        c["bias_x_group"] = grp
        c["bias_x_share"] = a3.slider("share of that group removed", 0.0, 1.0, float(c["bias_x_share"]), 0.05) if grp else 0.0
    r = trained()
    m = r.metrics
    model_badge(r)
    if m["roc_auc"] >= ss["contract"]["leakage_auc_max"]:
        st.error("AUC suspiciously high (97%?) — check for leakage!")
    b1, b2 = st.columns(2)
    with b1:
        chart(viz.confusion(m["tn"], m["fp"], m["fn"], m["tp"], "Confusion matrix (test set)"))
    with b2:
        chart(viz.fairness_bars(r.fairness))
    st.subheader("Contract matrix for this run")
    show_contract(M.contract_matrix(r, {"data_contract": bool(catalog.validate_contract(df)["pass"].all())}
                                    if not src.startswith("github") else {}, ss["contract"]))
    with st.expander("Fairness table (out-of-sample predictions, all clients)"):
        st.dataframe(r.fairness.round(3), hide_index=True, width="stretch")
    code(M.make_pipeline, M.run, M.perturb_training)

# ------------------------------------------------------------------ 7
elif page.startswith("7"):
    r = trained()
    y, p = r.test["churn"].values, r.p_test
    t = st.slider("Move the threshold — every chart below follows it", 0.01, 0.99, float(r.threshold), 0.01)
    mm = M.metrics(y, p, t, ss["costs"]["fn"], ss["costs"]["fp"])
    ui.cards([("Recall", f"{mm['recall']:.0%}", "leavers caught"), ("Precision", f"{mm['precision']:.0%}", "alerts that are right"),
              ("F1", f"{mm['f1']:.3f}", ""), ("Expected cost", f"{mm['expected_cost'] / 1000:,.1f} k€", f"optimum at {r.threshold:.2f}")])
    chart(viz.roc_pr(y, p, t))
    cc = M.cost_curve(r.train["churn"].values, r.p_oof, ss["costs"]["fn"], ss["costs"]["fp"])
    a, b, c_ = st.columns(3)
    with a:
        chart(viz.confusion(mm["tn"], mm["fp"], mm["fn"], mm["tp"], f"Confusion at {t:.2f}"))
    with b:
        chart(viz.cost_curve(cc, r.threshold, t))
    with c_:
        chart(viz.threshold_tradeoff(cc))
    d = pd.DataFrame({"p": p, "y": y}); d["decile"] = pd.qcut(d.p.rank(method="first"), 10, labels=range(10, 0, -1))
    dec = d.groupby("decile", observed=True).agg(clients=("y", "size"), leavers=("y", "sum"), churn_rate=("y", "mean"),
                                                 min_score=("p", "min")).sort_index(ascending=False)
    chart(viz.deciles(dec))
    st.subheader("2D decision boundaries of different models on two features")
    num = [f for f in avail if f not in config.CATEGORICAL]
    f1_ = st.selectbox("feature x", num, index=num.index("balance_change_3m") if "balance_change_3m" in num else 0)
    f2_ = st.selectbox("feature y", num, index=num.index("nps_score") if "nps_score" in num else 1)
    sig = st.slider("add noise to the training points (σ)", 0.0, 2.0, 0.0, 0.1)
    if st.button("Draw boundaries", type="primary"):
        from ds.fundamentals import boundaries
        dd = df.copy()
        if sig:
            rng = np.random.default_rng(0)
            for ff in (f1_, f2_):
                dd[ff] = dd[ff] + rng.normal(0, sig * dd[ff].std(), len(dd))
        with st.spinner("training 4 models x 2 features ..."):
            out = boundaries(dd, M.RunConfig(**{**r.cfg.to_dict(), "features": avail}), top2=(f1_, f2_))
        st.image(str(config.FIGURES / "f3_boundaries_2d.png"))
        chart(viz.lines(out["complexity"], "max_depth", ["train_auc", "valid_auc"],
                        "Complexity curve: training vs validation AUC (gap = overfitting)"))
        chart(viz.grouped_bars(out["table"], "model", ["roc_auc", "recall", "precision"], "Models compared (test set)", pct=True))
    code(M.cost_curve, M.best_threshold)

# ------------------------------------------------------------------ 8
elif page.startswith("8"):
    r = trained()
    sdf, Xs = xai.shap_frame(r.pipe, r.test, r.cfg.features, background=r.train, max_rows=500)
    imp = xai.global_importance(sdf)
    tab1, tab2, tabm, tab3 = st.tabs(["Global (all clients)", "Local (one client)", "Client map: SHAP vs LIME",
                                      "Stability (KPI F1)"])
    with tab1:
        a, b = st.columns([2, 3])
        with a:
            chart(viz.hbar(imp.head(15), "Mean |SHAP| — what the model relies on", color="#8A2A78"))
            rc = pd.Series([xai.reason_code(row)[0] for _, row in sdf.iterrows()]).value_counts()
            chart(viz.hbar(rc.astype(float), "Reason codes over the test clients", color="#7469B6", fmt=".0f"))
        with b:
            chart(viz.shap_beeswarm(sdf, Xs))
    with tab2:
        cl = xai.pick_contrasting_clients(r.test, r.p_test)
        opts = {f"{k} (client {r.test.loc[v, 'client_id']})": v for k, v in cl.items()}
        hi = r.test.assign(p=r.p_test).sort_values("p", ascending=False)
        opts.update({f"client {hi.loc[i, 'client_id']} · risk {hi.loc[i, 'p']:.2f}": i for i in hi.index[:200]})
        idx = opts[st.selectbox("client", list(opts))]
        s1, _ = xai.shap_frame(r.pipe, r.test.loc[[idx]], r.cfg.features)
        row = s1.iloc[0]
        reason, _ = xai.reason_code(row)
        p_i = float(r.pipe.predict_proba(r.test.loc[[idx], r.cfg.features])[:, 1][0])
        ui.cards([("Risk", f"{p_i:.0%}", f"threshold {r.threshold:.2f}"),
                  ("Actually", "left" if r.test.loc[idx, "churn"] else "stayed", "ground truth"),
                  ("Reason code", reason, "drives the retention offer")])
        a, b = st.columns(2)
        with a:
            chart(viz.shap_local(row))
        with b:
            lime_df = xai.lime_explain(r.pipe, r.train, r.test.loc[idx], r.cfg.features)
            chart(viz.hbar(lime_df.set_index("condition")["weight"], "LIME (same client)", signed=True))
        agree = xai.shap_lime_agreement(row, lime_df)
        st.markdown(f"**SHAP vs LIME:** rank correlation {agree['rank_corr']:.2f}, same direction for "
                    f"{agree['sign_agreement']:.0%} of the shared features ({agree['n']}).")
        st.dataframe(r.test.loc[[idx], r.cfg.features], width="stretch")
    with tabm:
        st.markdown("Each point is a test client, placed with **PCA** on 2 axes so similar clients sit close together. "
                    "The model itself uses all features; the 2D map is only for looking. A client is **FLAGGED** when "
                    "SHAP and LIME disagree about *why* the model scored them: low correlation between the two "
                    "explanations, or a large difference on one feature. **Click a point** to compare the two methods "
                    "below; to select several, pick **Box select** or **Lasso select** in the chart's toolbar.")
        c1, c2, c3 = st.columns(3)
        n_map = c1.select_slider("clients on the map", [100, 150, 200], 150)
        t_corr = c2.slider("flag if SHAP–LIME correlation is below", 0.0, 1.0, 0.5, 0.05)
        t_diff = c3.slider("flag if one feature differs by more than", 0.0, 1.0, 0.8, 0.05,
                           help="Contributions are normalised to [-1, 1] per method. With 28 features the largest "
                                "single difference is naturally bigger than with 4, hence 0.8 by default.")
        map_key = (id(r.pipe), n_map)
        if ss.get("xai_map_key") != map_key:
            with st.spinner(f"SHAP and LIME for {n_map} clients ..."):
                ss["xai_map"] = xai.shap_lime_map(r.pipe, r.train, r.test, r.cfg.features, n=n_map)
                ss["xai_map_key"] = map_key
        mp = ss["xai_map"]
        pts = mp["points"].copy()
        diff = (mp["shap"] - mp["lime"]).abs()
        pts["status"] = np.where((pts["corr"].fillna(0) < t_corr) | (diff.max(axis=1) > t_diff), "FLAGGED", "OK")
        pts["model says"] = np.where(pts["risk"] >= r.threshold, "alert (leaves)", "no alert (stays)")
        ui.cards([("Clients flagged", f"{(pts.status == 'FLAGGED').mean():.0%}", f"{(pts.status == 'FLAGGED').sum()} of {len(pts)}"),
                  ("Median correlation", f"{pts['corr'].median():.2f}", "SHAP vs LIME, per client"),
                  ("PCA keeps", f"{mp['explained_variance']:.0%}", "of the variance on 2 axes")])
        ev = ui.chart(viz.xai_map(pts, r.threshold), key="xai_map_chart", select=True)
        picked = viz.selected_rows(ev, pts)
        if not picked:
            worst = pts.sort_values("corr").index[:1].tolist()
            st.info("Nothing selected yet: showing the client where SHAP and LIME agree least. Select points on the map to change it.")
            picked = worst
        first = picked[0]
        st.subheader(f"Client {pts.loc[first, 'client_id']} · risk {pts.loc[first, 'risk']:.0%} · "
                     f"{'left' if pts.loc[first, 'left'] else 'stayed'} · {pts.loc[first, 'status']}")
        a, b = st.columns(2)
        with a:
            chart(viz.contrib_bars(mp["shap"].loc[first], "SHAP (normalised)"))
        with b:
            chart(viz.contrib_bars(mp["lime"].loc[first], "LIME (normalised)"))
        st.caption("Bars to the right push towards **leaving**, bars to the left towards **staying**. "
                   "Compare which features matter and in which direction.")
        rows = []
        for i in picked:
            for f in r.cfg.features:
                d = float(diff.loc[i, f])
                rows.append({"client": pts.loc[i, "client_id"], "feature": f, "value": mp["values"].loc[i, f],
                             "SHAP": round(float(mp["shap"].loc[i, f]), 3), "LIME": round(float(mp["lime"].loc[i, f]), 3),
                             "abs. difference": round(d, 3), "interpretation": xai.interpret_difference(d),
                             "feature flag": "LARGE DIFFERENCE" if d > t_diff else "OK"})
        detail = pd.DataFrame(rows)
        detail["value"] = [("missing" if pd.isna(v) else str(v)) for v in detail["value"]]
        summary = pd.DataFrame([{"client": pts.loc[i, "client_id"], "risk": round(float(pts.loc[i, "risk"]), 3),
                                 "correlation": round(float(pts.loc[i, "corr"]), 3),
                                 "client flag": "LOW AGREEMENT" if pts.loc[i, "corr"] < t_corr else "OK",
                                 "features flagged": ", ".join(diff.columns[diff.loc[i] > t_diff]) or "none"}
                                for i in picked])
        st.markdown(f"**Selected clients: summary** ({len(picked)})")
        st.dataframe(summary, hide_index=True, width="stretch")
        with st.expander("Per-feature table (SHAP vs LIME for every selected client)", expanded=len(picked) == 1):
            st.dataframe(detail, hide_index=True, width="stretch")
        d1, d2 = st.columns(2)
        d1.download_button("Download per-feature table (CSV)", detail.to_csv(index=False), "xai_shap_lime_selected.csv", "text/csv")
        d2.download_button("Download summary (CSV)", summary.to_csv(index=False), "xai_flag_summary.csv", "text/csv")
    with tab3:
        if st.button("Measure stability: retrain 5× on bootstrap samples", type="primary"):
            with st.spinner("retraining ..."):
                j, tops = xai.stability(r.cfg, r.train, n=5)
            ui.cards([("Mean Jaccard of top-5", f"{j:.2f}", "PASS" if j >= ss["contract"]["F1_jaccard_min"] else "FAIL")])
            st.dataframe(pd.DataFrame(tops, index=[f"retrain {i + 1}" for i in range(len(tops))]), width="stretch")
    code(xai.shap_frame, xai.reason_code, xai.lime_explain, xai.shap_lime_map, xai.stability)

# ------------------------------------------------------------------ 9
elif page.startswith("9"):
    st.markdown("Pass 2 of XAI: data from **outside** the DWH — another branch's raw export (duplicates, dd.mm.yyyy dates, "
                "comma decimals, country spelled 5 ways, impossible values, garbled text). Same cleaning code, same query, "
                "same model.")
    bank = get_dataset(SRC_BANK, cutoff, False)
    ext = get_dataset(SRC_EXT, cutoff, False)
    r = train(json.dumps(M.RunConfig(**{**current_cfg().to_dict()}).to_dict(), sort_keys=True, default=str), SRC_BANK, cutoff, False)
    from sklearn.metrics import recall_score, roc_auc_score
    pe = r.pipe.predict_proba(ext[r.cfg.features])[:, 1]
    ui.cards([("AUC internal (test)", f"{r.metrics['roc_auc']:.3f}", f"{len(bank):,} clients"),
              ("AUC external branch", f"{roc_auc_score(ext['churn'], pe):.3f}", f"{len(ext):,} clients"),
              ("Recall external", f"{recall_score(ext['churn'], (pe >= r.threshold).astype(int)):.0%}", f"at threshold {r.threshold:.2f}")])
    ps = xai.psi_table(r.train, ext, r.cfg.features)
    si, _ = xai.shap_frame(r.pipe, r.test, r.cfg.features, max_rows=400)
    se, _ = xai.shap_frame(r.pipe, ext, r.cfg.features, max_rows=400)
    cmp = xai.compare_importance(xai.global_importance(si), xai.global_importance(se))
    a, b = st.columns(2)
    with a:
        chart(viz.psi_bars(ps))
    with b:
        chart(viz.importance_compare(xai.global_importance(si), xai.global_importance(se)))
    st.markdown(f"**KPI F2 — SHAP rank correlation internal vs external: {cmp['rank_corr']:.2f}** · top-5 overlap "
                f"(Jaccard) {cmp['jaccard_top5']:.2f}")
    if ps["psi"].max() >= ss["contract"]["F2_psi_max"]:
        st.warning(f"Feature `{ps.iloc[0]['feature']}` drifts on external data → flag it and go back to the features (inner loop).")
    p = config.TABLES / "cleaning_report_branch.csv"
    if p.exists():
        with st.expander("Cleaning rules applied to the external export"):
            st.dataframe(pd.read_csv(p), hide_index=True, width="stretch")
    fx = st.selectbox("Compare a feature's distribution: our bank vs the external branch", r.cfg.features)
    if fx not in config.CATEGORICAL:
        import plotly.graph_objects as go
        fig = go.Figure()
        for nm, d_, col in [("our bank", bank, viz.CAT[0]), ("external branch", ext, viz.CAT[2])]:
            fig.add_histogram(x=d_[fx], name=nm, histnorm="percent", opacity=0.7, marker_color=col, nbinsx=40)
        fig.update_layout(barmode="overlay", title=f"{fx}: our bank vs external branch", height=380)
        chart(fig)

# ------------------------------------------------------------------ 10
elif page.startswith("10"):
    st.markdown("**Live system:** `bash realtime/run_realtime.sh` (Kafka) or `--bus file` (fallback), then "
                "`streamlit run realtime/rtap_app.py` for the 5 windows. Below: the same scorer replayed offline.")
    if not (config.ARTIFACTS / "champion.joblib").exists():
        st.warning("No champion yet: save an iteration as champion on page 12 or run run_all.py."); st.stop()
    from ds import rtap, streaming_sim
    from ds.scoring import load_champion
    c1, c2 = st.columns(2)
    days = c1.slider("bank days to replay", 10, 90, 30)
    cap = c2.slider("retention capacity per day", 10, 200, 40)
    if st.button("Replay: streaming (clock path + EWMA) vs monthly batch", type="primary"):
        champ = load_champion()
        prog = st.progress(0.0)
        from ds.livefeatures import LiveStore
        store = LiveStore()
        al, _ = streaming_sim.replay(champ, days, capacity=cap, store=store, progress=prog.progress)
        ab = streaming_sim.batch_monthly(champ, store, capacity=cap, days=days)
        truth = streaming_sim.truth_exits()
        r1, m1 = streaming_sim.evaluate(al, truth, days=days, label="streaming + EWMA")
        r2, m2 = streaming_sim.evaluate(ab, truth, days=days, label="monthly batch")
        ss["replay"] = {"rows": [r1, r2], "details": {"streaming + EWMA": m1, "monthly batch": m2}, "alerts": al}
    if "replay" in ss:
        R = ss["replay"]
        t = pd.DataFrame(R["rows"])
        ui.cards([(f"{r['mode']}", f"{r['recall_before_exit']:.0%}", f"caught before exit · median lead {r['median_lead_days']:.0f} d")
                  for r in R["rows"]])
        chart(viz.lead_time_hist(R["details"]))
        st.dataframe(t, hide_index=True, width="stretch")
        al = R["alerts"]
        if len(al):
            a, b = st.columns([1, 2])
            with a:
                chart(viz.hbar(al["reason"].value_counts().astype(float), "Alerts by reason", color="#7469B6", fmt=".0f"))
            with b:
                st.markdown("**Alerts with SHAP reason and RAG recommendation**")
                st.dataframe(al[["alert_day", "client_id", "risk", "reason", "offer_suggested", "rag_text"]].head(50),
                             hide_index=True, height=360)
            st.subheader("RTAP: policies compared on these alerts")
            pol, ts, log = rtap.compare_policies(al, streaming_sim.truth_exits(), capacity=cap)
            chart(viz.grouped_bars(pol, "policy", ["clients_saved", "offers_sent"], "Clients saved vs offers sent"))
            chart(viz.grouped_bars(pol, "policy", ["ROI"], "Return on the offers (value kept / cost)"))
            a, b = st.columns(2)
            a.markdown("**What Thompson Sampling learned**"); a.dataframe(ts.best(), hide_index=True)
            b.markdown("**Offer mix by gender / country** (ethical rule: rows should look alike)"); b.dataframe(rtap.offer_fairness(log))
    code(__import__("ds.scoring", fromlist=["Scorer"]).Scorer, rtap.ThompsonSampler)

# ------------------------------------------------------------------ 11
elif page.startswith("11"):
    from ds import rag
    r = trained()
    st.markdown("RAG = retrieve passages from the bank's documents (`kb/*.md`), then write the recommendation **only** "
                "from them, citing the sources. Optional local LLM: set `OLLAMA_MODEL=llama3.2` and run Ollama.")
    cl = xai.pick_contrasting_clients(r.test, r.p_test)
    hi = r.test.assign(p=r.p_test).sort_values("p", ascending=False)
    opts = {f"{k} (client {r.test.loc[v, 'client_id']})": v for k, v in cl.items()}
    opts.update({f"client {hi.loc[i, 'client_id']} · risk {hi.loc[i, 'p']:.2f}": i for i in hi.index[:30]})
    idx = opts[st.selectbox("client", list(opts))]
    s1, _ = xai.shap_frame(r.pipe, r.test.loc[[idx]], r.cfg.features)
    reason, _ = xai.reason_code(s1.iloc[0])
    top = s1.iloc[0].sort_values(ascending=False).index[:3].tolist()
    p_i = float(r.pipe.predict_proba(r.test.loc[[idx], r.cfg.features])[:, 1][0])
    rec = rag.recommend(reason, top, p_i, int(r.test.loc[idx].get("marketing_opt_out", 0) or 0),
                        use_llm=st.toggle("use local LLM (Ollama)", False))
    ui.cards([("Risk", f"{p_i:.0%}", ""), ("Reason", reason, "from SHAP"), ("Offer", rec["offer"], "from the playbook")])
    st.markdown(f"**Query sent to the retriever:** `{rec['query']}`")
    st.success(rec["text"])
    st.dataframe(pd.DataFrame(rec["sources"]), hide_index=True, width="stretch")
    q = st.text_input("Ask the knowledge base anything", "what do we do with a silent client who stopped paying with the card?")
    st.dataframe(pd.DataFrame(rag.kb().retrieve(q, 4)), hide_index=True, width="stretch")
    code(rag.KnowledgeBase, rag.recommend)

# ------------------------------------------------------------------ 12
elif page.startswith("12"):
    if not src.startswith("simulated"):
        st.info("Iterations are saved on the main source (simulated bank)."); st.stop()
    cfg = current_cfg()
    st.markdown(f"Current settings: model **{cfg.model}**, balancing **{cfg.balancing}**, threshold **{cfg.threshold_mode}**, "
                f"{len(cfg.features)} features, per-group thresholds **{cfg.per_group_thresholds}**")
    reason = st.text_input("Reason for this iteration (what changed and why)", "")
    c1, c2 = st.columns(2)
    if c1.button("Evaluate full contract (incl. stability + external data) and SAVE → back to box 1", type="primary"):
        ext = get_dataset(SRC_EXT, cutoff, False)
        with st.spinner("training, XAI, stability, external check ..."):
            it = cycle.evaluate_iteration(cfg, df, ext, thresholds=ss["contract"])
        e = cycle.save_iteration(it, reason or "(no reason given)")
        ss["last_it"] = it
        st.success(f"saved iteration #{e['id']}: {e['verdict']}")
    if c2.button("Make the last evaluated iteration the CHAMPION (used by streaming)") and "last_it" in ss:
        cycle.save_champion(ss["last_it"], "app iteration"); st.success("champion saved to artifacts/champion.joblib")
    if "last_it" in ss:
        st.subheader("Verdict of the last evaluation")
        show_contract(ss["last_it"]["contract"])
        if ss["last_it"]["verdict"] == "FAIL":
            st.error("FAIL → go back in the cycle (features, data, balancing, threshold) and try again.")
    log = cycle.load_log()
    if log:
        st.subheader("Compare iterations side by side")
        names = [f"#{e['id']} {e['name']}" for e in log]
        pick = st.multiselect("iterations", names, default=names[-3:])
        sel = [e for e in log if f"#{e['id']} {e['name']}" in pick]
        if sel:
            chart(viz.iterations_chart(sel))
            st.dataframe(cycle.compare_table(sel), width="stretch")
        if len(sel) >= 2:
            st.write("What changed between the first two:", cycle.features_entered_left(sel[0], sel[1]))
        if st.button("Clear the iteration log"):
            cycle.LOG.unlink(); st.rerun()
    code(cycle.evaluate_iteration, cycle.save_iteration)

# ------------------------------------------------------------------ ★ six fundamentals
elif page.startswith("★ Six"):
    if src == SRC_KAG:  # these experiments need the behaviour, text and silence features the public file lacks
        st.info("These experiments use the bank's behaviour, text and silence features, which the public dataset does "
                "not have, so they run on the simulated bank (DWH).")
        df = get_dataset(SRC_BANK, cutoff, False)
        avail = [f for f in ss["features"] if f in df.columns]
    from ds import fundamentals as F, fundamentals_text as FT
    base = current_cfg()
    key = st.segmented_control("Fundamental (report order 2.2.1–2.2.6)", FT.ORDER, default="bias",
                               format_func=FT.TITLES.get) or "bias"
    st.subheader("1 · How it appears in this problem")
    st.markdown(FT.APPEARS[key])
    st.subheader("2 · Measured: experiment, before and after")
    if st.button(f"Run the {FT.TITLES[key]} experiment", type="primary"):
        with st.spinner("running ..."):
            all_feats = M.RunConfig(**{**base.to_dict(), "features": [f for f in ALL_FEATURES if f in df]})
            if key == "bias":
                o = F.bias(df, all_feats)
            elif key == "noise":
                raw = M.load_dataset("bank_rawtext") if (config.DATASETS / "bank_rawtext.csv").exists() else df
                o = F.noise(df, raw, base, reps=1)
            elif key == "smote":
                o = F.smote(df, all_feats)
            elif key == "boundaries":
                o = F.boundaries(df, base)
            else:
                from ds import rtap, streaming_sim
                from ds.scoring import load_champion
                o = streaming_sim.run_streaming_fundamental(load_champion())
                if key == "rtap":
                    o = rtap.rtap_fundamental(o["alerts"], o["truth"])
            ss[f"fund_{key}"] = o
    o = ss.get(f"fund_{key}")
    if o is None:
        st.caption("Press the button to run the experiment on the current dataset. Until then, the figure from the "
                   "last `python run_all.py` is shown.")
        for fname in {"bias": ["f1_bias_recall_by_age"], "noise": ["f2_noise_x_vs_y"], "smote": ["f4_smote_variants"],
                      "boundaries": ["f3_roc_pr", "f3_boundaries_2d"], "streaming": ["f5_streaming_lead_time"],
                      "rtap": []}[key]:
            p = config.FIGURES / f"{fname}.png"
            if p.exists():
                st.image(str(p), caption=f"{fname}.png (last run_all.py)")
    else:
        if key == "bias":
            t = o["table"].copy(); t["variant"] = t["variant"].str.slice(0, 34)
            chart(viz.grouped_bars(t, "variant", ["recall", "gap_age_group", "gap_country", "gap_gender"],
                                   "Recall and fairness gaps per variant (lower gaps = fairer)", pct=True))
            a, b = st.columns(2)
            with a:
                chart(viz.fairness_bars(o["fairness_full"]), key="ff")
                st.caption("before: all features")
            with b:
                chart(viz.fairness_bars(o["fairness_mitigated"]), key="fm")
                st.caption("after: sensitive attributes and the age proxy removed")
        elif key == "noise":
            tt = o["text_table"].reset_index()
            chart(viz.grouped_bars(tt, "noise_type", ["accuracy_raw", "accuracy_denoised"],
                                   "Text: sentiment accuracy by noise type, raw vs denoised", pct=True))
            a, b = st.columns(2)
            for col, (k, g) in zip((a, b), o["table"].groupby("noise", sort=False)):
                with col:
                    chart(viz.lines(g, "level", ["auc_mean", "f1_mean"], k, err={"auc_mean": "auc_std"}), key=k)
            st.dataframe(o["text_model"], hide_index=True, width="stretch")
        elif key == "smote":
            t = o["table"].dropna(subset=["roc_auc"])
            chart(viz.grouped_bars(t, "variant", ["recall", "precision", "f1"], "Balancing variants (test set)", pct=True))
            chart(viz.grouped_bars(o["table"], "variant", ["cv_auc"], "Validation AUC: note the inflated 'SMOTE before the split'", pct=True))
            a, b = st.columns(2)
            a.dataframe(o["realism"], hide_index=True, width="stretch")
            b.dataframe(o["subset_noise"], hide_index=True, width="stretch")
        elif key == "boundaries":
            chart(viz.grouped_bars(o["table"], "model", ["roc_auc", "recall", "precision", "f1"], "Models compared", pct=True))
            a, b = st.columns(2)
            with a:
                chart(viz.lines(o["complexity"], "max_depth", ["train_auc", "valid_auc"], "Complexity curve"))
            with b:
                chart(viz.deciles(o["deciles"]))
            st.dataframe(o["confusion"], hide_index=True, width="stretch")
        elif key == "streaming":
            t = o["table"].copy(); t["mode"] = t["mode"].str.replace(" | only leavers NOT flagged at the cutoff", " (late risk)")
            chart(viz.grouped_bars(t, "mode", ["recall_before_exit", "share_lead>=30d"], "Streaming vs batch", pct=True))
        elif key == "rtap":
            a, b = st.columns(2)
            with a:
                chart(viz.grouped_bars(o["table"], "policy", ["clients_saved", "offers_sent"], "Clients saved vs offers sent"))
            with b:
                chart(viz.grouped_bars(o["table"].dropna(subset=["ROI"]), "policy", ["ROI"], "ROI per policy"))
            st.dataframe(o["learned"], hide_index=True, width="stretch")
            st.markdown("Offer mix per group (ethical check: it should not depend on gender or country)")
            st.dataframe(o["fair"], width="stretch")
        st.dataframe(o["table"], hide_index=True, width="stretch")
        st.subheader("3 · Observed")
        st.markdown("\n".join(f"- {x}" for x in FT.observed(key, o)))
        st.subheader("4 · Mitigation „S-a efectuat …, deoarece …”")
        st.markdown("\n".join(f"- {m}" for m in o["measures"]))
    st.subheader("5 · Răspunsuri (lab questions)")
    for q, ans in FT.ANSWERS[key]:
        with st.expander(q):
            st.write(ans)

# ------------------------------------------------------------------ ★ lab experiments
elif page.startswith("★ Lab"):
    if src == SRC_KAG:  # these experiments need the behaviour, text and silence features the public file lacks
        st.info("These experiments use the bank's behaviour, text and silence features, which the public dataset does "
                "not have, so they run on the simulated bank (DWH).")
        df = get_dataset(SRC_BANK, cutoff, False)
        avail = [f for f in ss["features"] if f in df.columns]
    from ds import experiments as E
    which = st.segmented_control("Experiment", ["Experiments 1–2 (X vs Y)", "PCA representations", "Regularization L1/L2/ElasticNet",
                                                "Experiment A (toy, known truth)", "Experiment B + data sources compared"],
                                 default="Experiments 1–2 (X vs Y)")
    reps = st.number_input("repetitions for Experiments 1–2", 5, 100, 20)
    if st.button("Run", type="primary"):
        with st.spinner("running ..."):
            if which.startswith("Experiments 1"):
                t = E.experiment_1_2(df, reps=reps)
                w = t.pivot(index="level", columns="scenario", values="f1_mean").reset_index()
                sd = t.pivot(index="level", columns="scenario", values="f1_std").add_suffix("_sd").reset_index(drop=True)
                w = pd.concat([w, sd], axis=1)
                a, b = st.columns(2)
                with a:
                    chart(viz.lines(w, "level", ["noise_X", "noise_Y"], "Experiment 1: noise on X vs Y (F1, ± std)",
                                    err={"noise_X": "noise_X_sd", "noise_Y": "noise_Y_sd"}))
                with b:
                    chart(viz.lines(w, "level", ["bias_X", "bias_Y"], "Experiment 2: bias on X vs Y (F1, ± std)",
                                    err={"bias_X": "bias_X_sd", "bias_Y": "bias_Y_sd"}))
                st.image(str(config.FIGURES / "e12_boundary_moves.png"))
                st.dataframe(t, hide_index=True)
            elif which.startswith("PCA"):
                st.write(E.pca_text()); st.image(str(config.FIGURES / "e62_text_pca.png"))
                st.write(E.pca_mlp_hidden(df)); st.image(str(config.FIGURES / "e62_mlp_hidden_pca.png"))
            elif which.startswith("Regular"):
                o = E.regularization(df)
                st.markdown(f"**Set to 0 by L1:** {', '.join(o['zeros_L1']) or 'none'}")
                sw = o["sweep"]
                a, b = st.columns(2)
                with a:
                    w = sw.pivot(index="C", columns="l1_ratio", values="f1").reset_index(); w.columns = [str(c) for c in w.columns]
                    chart(viz.lines(w, "C", [c for c in w.columns if c != "C"], "F1 vs C, one line per α (l1_ratio)", logx=True))
                with b:
                    w = sw.pivot(index="C", columns="l1_ratio", values="nonzero_coefs").reset_index(); w.columns = [str(c) for c in w.columns]
                    chart(viz.lines(w, "C", [c for c in w.columns if c != "C"], "Non-zero coefficients vs C", logx=True))
                import plotly.graph_objects as go
                hm = o["heatmap"]
                lim = float(np.abs(hm.values).max())
                chart(go.Figure(go.Heatmap(z=hm.values, x=hm.columns, y=hm.index, colorscale=viz.DIV, zmin=-lim, zmax=lim))
                      .update_layout(title="L1 coefficients vs C (white = 0)", height=650, xaxis_title="C"))
                st.dataframe(o["shap_top5"])
            elif which.startswith("Experiment A"):
                t = E.experiment_a()
                chart(viz.lines(t[t.model == "L1"], "C", ["noise_features_zeroed", "informative_zeroed"],
                                "Experiment A: what L1 removes as C grows (known truth)", logx=True))
                st.dataframe(t, hide_index=True)
            else:
                bank = get_dataset(SRC_BANK, cutoff, False)
                ext = get_dataset(SRC_EXT, cutoff, False)
                srcs = {"simulated bank": bank[["client_id"] + github_load.COMMON + ["churn"]],
                        "external branch": ext[["client_id"] + github_load.COMMON + ["churn"]]}
                real = srcs["simulated bank"]
                if github_load.available():
                    k = pd.read_csv(config.DATASETS / "github.csv") if (config.DATASETS / "github.csv").exists() else github_load.load()
                    srcs["github (real)"] = k; real = k.drop(columns=["geography"])
                res, syn = E.experiment_b(real, "github" if github_load.available() else "simulated bank")
                srcs["learned synthetic"] = syn
                ui.cards([("Train real → test real", f"{res['AUC train-real/test-real (TRTR)']:.3f}", "AUC"),
                          ("Train synthetic → test real", f"{res['AUC train-synthetic/test-real (TSTR)']:.3f}", "AUC"),
                          ("Mean KS distance", f"{res['mean_KS']:.3f}", "0 = identical distributions")])
                mat, keep, common = E.sources_matrix(srcs)
                import plotly.graph_objects as go
                chart(go.Figure(go.Heatmap(z=mat.values, x=mat.columns, y=mat.index, colorscale=[[0, "#F5EFF8"], [1, "#1A8FB0"]],
                                           texttemplate="%{z:.3f}", textfont=dict(size=16)))
                      .update_layout(title="ROC-AUC: trained on (rows) × tested on (columns)", height=420, yaxis_autorange="reversed"))
                st.dataframe(keep); st.caption(f"common features only: {common}"); st.write(res)

# ------------------------------------------------------------------ ★ monitoring
elif page.startswith("★ Monitoring"):
    from ds import monitoring
    from ds.scoring import load_champion
    if not (config.ARTIFACTS / "champion.joblib").exists():
        st.warning("No champion yet."); st.stop()
    champ = load_champion()
    if st.button("Run monthly monitoring (PSI, performance, SHAP drift, fairness)", type="primary"):
        with st.spinner("running the query at 9 month-ends ..."):
            o = monitoring.monitor(champ, M.load_dataset(f"bank_{config.CUTOFF}"))
        t = o["table"]
        a, b = st.columns(2)
        with a:
            chart(viz.lines(t, "month_cutoff", ["roc_auc", "recall"], "Performance by month"))
        with b:
            chart(viz.lines(t, "month_cutoff", ["max_psi", "recall_gap_age", "shap_top5_jaccard_vs_training"], "Drift and fairness by month"))
        st.dataframe(t, hide_index=True, width="stretch")
        st.caption("Note: the 2025-12-31 row is partly in-sample (the champion was trained on part of it).")
    p = config.REPORTS / "model_card.md"
    if p.exists():
        with st.container(border=True):
            st.markdown(p.read_text(encoding="utf-8"))
