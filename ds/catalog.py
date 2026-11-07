"""Stage 1.2-1.4 as code: KPIs, feature catalog and data contract.

The app, the report tables and the contract checks all read from here,
so there is one definition of every KPI and feature.
"""
from __future__ import annotations

import pandas as pd

PROBLEM = {
    "problem": ("Clienții pleacă din bancă în tăcere: își reduc treptat activitatea, nu se plâng și nu "
                "răspund la sondaje, iar banca observă plecarea abia când contul este închis. "
                "Fiecare client pierdut costă marja anuală (~1 000 €)."),
    "goal": "Reducerea costului churn-ului prin detectarea timpurie a riscului și intervenții de retenție rentabile și corecte.",
    "objectives": [
        "O1. Detectarea riscului de plecare cu cel puțin 30 de zile înainte de plecare (lead time).",
        "O2. Identificarea a cel puțin 70% dintre clienții care pleacă în următoarele 90 de zile (recall).",
        "O3. Costul campaniei de retenție mai mic decât valoarea clienților salvați (ROI > 1).",
        "O4. Detectarea clienților tăcuți (fără feedback, fără răspuns la sondaj) – întrebarea de cercetare.",
        "O5. Diferența de recall între grupuri (vârstă, gen, țară) de cel mult 0.10.",
        "O6. Explicații stabile și verificabile pentru fiecare alertă (SHAP/LIME).",
    ],
    "question": "„Ce înseamnă să pleci fără să spui nimic? Cum putem asculta tăcerea unui client?”",
    "hypotheses": [
        "H1: clienții tăcuți (fără feedback 6 luni și fără răspuns la sondaj) pleacă mai des.",
        "H2: scorarea în flux (streaming) avertizează mai devreme decât scorarea lunară (batch).",
        "H3: bucla (iterația 2) reduce diferența de fairness fără pierderi mari de recall.",
    ],
}

KPIS = pd.DataFrame([
    ("M1", "model", "Recall-ul clienților care pleacă (la pragul ales)", "TP / (TP + FN)", ">= 0.70", "set de test"),
    ("M2", "model", "Cost așteptat vs. fără model; ROC-AUC",
     "FN·1000€ + FP·60€ + TP·(60€ + 0.65·1000€)", "cost < fără model; AUC >= 0.80", "set de test"),
    ("F1", "features", "Stabilitatea explicațiilor (top-5 SHAP)", "Jaccard mediu al top-5 între reantrenări", ">= 0.60", "5 reantrenări (seed/bootstrap)"),
    ("F2", "features", "Robustețea pe date externe", "corelația Spearman a importanțelor SHAP intern vs extern; PSI per feature",
     "rho >= 0.80; PSI < 0.20", "filiala externă"),
    ("B1", "business", "Rata churn", "plecați în 90 zile / clienți activi la cutoff", "monitorizare", "query Stage 1"),
    ("B2", "business", "Ponderea plecărilor tăcute", "plecați tăcuți / toți plecații", "monitorizare", "query Stage 1"),
    ("B3", "business", "Lead time", "zile între alertă și plecare (mediană)", ">= 30 zile", "streaming"),
    ("B4", "business", "ROI campanie", "valoare reținută / cost oferte", "> 1", "RTAP"),
    ("B5", "business", "Fairness gap", "max - min recall între grupuri", "<= 0.10", "set de test"),
], columns=["id", "judges", "KPI", "formula", "target", "source"])

# name, group, type, formula, source (table.column), why it relates to churn
FEATURES = [
    ("age", "profile", "discrete", "floor((cutoff - birth_date)/365.25)", "dim_client.birth_date", "life stage; SENSITIVE attribute"),
    ("gender", "profile", "nominal", "as recorded", "dim_client.gender", "SENSITIVE attribute (fairness check only)"),
    ("country", "profile", "nominal", "as recorded", "dim_client.country", "SENSITIVE attribute; regional competition"),
    ("tenure_months", "profile", "discrete", "floor((cutoff - join_date)/30.44)", "dim_client.join_date", "new clients leave more easily"),
    ("credit_score", "profile", "continuous", "last monthly snapshot", "fact_account_month.credit_score", "financial health"),
    ("num_products", "profile", "discrete", "last monthly snapshot", "fact_account_month.num_products", "1 product = weak tie; 3-4 = over-selling"),
    ("has_card", "profile", "binary", "last monthly snapshot", "fact_account_month.has_card", "card = daily touchpoint"),
    ("is_active_member", "profile", "binary", ">=4 transactions in the month", "fact_account_month.is_active_member", "engagement"),
    ("est_salary", "profile", "continuous", "last monthly snapshot", "fact_account_month.est_salary", "client value"),
    ("balance", "profile", "continuous", "last monthly snapshot", "fact_account_month.balance", "money kept at the bank"),
    ("balance_change_3m", "behaviour", "continuous", "(bal_t - bal_t-3)/bal_t-3", "fact_account_month.balance", "money being moved out"),
    ("tx_count_90d", "behaviour", "discrete", "count tx in (cutoff-90, cutoff]", "fact_transaction", "current activity"),
    ("tx_count_prev_90d", "behaviour", "discrete", "count tx in (cutoff-180, cutoff-90]", "fact_transaction", "previous activity (baseline)"),
    ("tx_trend", "behaviour", "continuous", "(tx_90d+1)/(tx_prev_90d+1)", "fact_transaction", "fading activity (gradual fade)"),
    ("amount_in_90d", "behaviour", "continuous", "sum in-amounts 90d", "fact_transaction.amount", "salary moved to another bank"),
    ("amount_out_90d", "behaviour", "continuous", "sum out-amounts 90d", "fact_transaction.amount", "money drain"),
    ("net_flow_90d", "behaviour", "continuous", "in - out 90d", "fact_transaction.amount", "money drain"),
    ("digital_share_90d", "behaviour", "continuous", "digital tx / all tx 90d", "fact_transaction + dim_channel.is_digital", "channel habit; PROXY for age"),
    ("msg_count_180d", "voice", "discrete", "messages in 180d", "fact_feedback", "the client still talks to the bank"),
    ("complaints_180d", "voice", "discrete", "complaints in 180d", "fact_feedback.is_complaint", "dissatisfaction (vocal leaver)"),
    ("neg_msg_180d", "voice", "discrete", "negative messages (denoised sentiment)", "fact_feedback.is_negative", "dissatisfaction"),
    ("avg_sentiment_180d", "voice", "continuous", "mean sentiment of non-empty messages", "fact_feedback.sentiment_score", "tone of the relationship"),
    ("days_since_last_tx", "silence", "discrete", "cutoff - last tx date (365 if none)", "fact_transaction", "silence in payments"),
    ("days_since_last_msg", "silence", "discrete", "cutoff - last message date (365 if none)", "fact_feedback", "silence in contact"),
    ("no_feedback_6m", "silence", "binary", "msg_count_180d = 0", "fact_feedback", "the client stopped talking"),
    ("nps_score", "silence", "ordinal", "last answered NPS (0-10), NULL if not answered", "fact_survey.nps_score", "satisfaction when spoken"),
    ("nps_missing", "silence", "binary", "last survey not answered", "fact_survey.answered_date_key", "missing NOT at random = silence"),
    ("silent_decline", "silence", "binary", "tx_trend < 0.5 AND no_feedback_6m", "derived", "silent exit pattern"),
]
FEATURES_DF = pd.DataFrame(FEATURES, columns=["feature", "group", "type", "formula", "source", "why"])
ALL_FEATURES = FEATURES_DF["feature"].tolist()
GROUP_OF = dict(zip(FEATURES_DF.feature, FEATURES_DF.group))

REASONS = {  # reason codes (step 4.5) -> drive the retention offer
    "dissatisfaction": ["complaints_180d", "neg_msg_180d", "avg_sentiment_180d", "nps_score"],
    "disengagement": ["tx_count_90d", "tx_count_prev_90d", "tx_trend", "is_active_member", "digital_share_90d", "msg_count_180d"],
    "money_withdrawal": ["amount_out_90d", "net_flow_90d", "balance_change_3m", "balance", "amount_in_90d"],
    "silence": ["days_since_last_tx", "days_since_last_msg", "no_feedback_6m", "nps_missing", "silent_decline"],
    "profile": ["age", "gender", "country", "tenure_months", "credit_score", "num_products", "has_card", "est_salary"],
}
REASON_OF = {f: r for r, fs in REASONS.items() for f in fs}

DATA_CONTRACT = pd.DataFrame([
    ("core banking", "core_clients", "client_id", "integer", "unique, > 0", "once", "2010"),
    ("core banking", "core_clients", "birth_date", "date", "age 18-100 at cutoff", "once", "2010"),
    ("core banking", "core_clients", "gender", "char(1)", "F / M", "once", "2010"),
    ("core banking", "core_clients", "country", "char(2)", "MD / RO / UA", "once", "2010"),
    ("core banking", "core_clients", "join_date", "date", "<= today", "once", "2010"),
    ("core banking", "core_accounts_monthly", "balance", "numeric(16,2)", "0 - 1e8", "monthly", "2025-01"),
    ("core banking", "core_accounts_monthly", "num_products", "smallint", "1 - 4", "monthly", "2025-01"),
    ("core banking", "core_accounts_monthly", "credit_score", "numeric(6,1)", "300 - 900, NULL allowed", "monthly", "2025-01"),
    ("core banking", "core_exits", "exit_date", "date", "> join_date", "event", "2025-01"),
    ("cards/payments", "card_transactions", "amount", "numeric(14,2)", "> 0", "real time", "2025-01"),
    ("cards/payments", "card_transactions", "direction", "text", "in / out", "real time", "2025-01"),
    ("cards/payments", "card_transactions", "channel", "text", "mobile_app/web/card_pos/atm/branch/transfer", "real time", "2025-01"),
    ("contact centre", "contact_messages", "text", "text", "free text, may be empty", "real time", "2025-01"),
    ("contact centre", "contact_messages", "complaint_flag", "smallint", "0 / 1", "real time", "2025-01"),
    ("survey", "survey_responses", "nps_score", "smallint", "0 - 10, NULL = not answered", "quarterly", "2025-03"),
], columns=["source_system", "table", "column", "type", "allowed_values", "update_frequency", "exists_since"])


def validate_contract(df: pd.DataFrame) -> pd.DataFrame:
    """Data-contract check on a dataset produced by the query (part 1 of the contract matrix)."""
    checks = [
        ("client_id unique", df["client_id"].is_unique),
        ("age in 18-100", df["age"].dropna().between(18, 100).all()),
        ("gender in {F,M}", df["gender"].dropna().isin(["F", "M"]).all()),
        ("country in {MD,RO,UA}", df["country"].dropna().isin(["MD", "RO", "UA"]).all()),
        ("num_products in 1-4", df["num_products"].dropna().between(1, 4).all()),
        ("balance >= 0", (df["balance"].dropna() >= 0).all()),
        ("nps_score in 0-10", df["nps_score"].dropna().between(0, 10).all()),
        ("missing share per feature < 40% (except nps_score)",
         (df.drop(columns=["nps_score"]).isna().mean() < 0.4).all()),
        ("churn is 0/1", df["churn"].isin([0, 1]).all()),
    ]
    return pd.DataFrame(checks, columns=["rule", "pass"])
