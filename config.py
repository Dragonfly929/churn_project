"""Central configuration: paths, dates, costs, seeds.

Every script imports from here, so one change propagates everywhere
(reproducibility rule from Step 9.2).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"                 # source systems (CSV exports)
RAW_BRANCH = DATA / "raw_branch"   # external, messy export of another branch
EXTERNAL = DATA / "external"       # Kaggle "Churn_Modelling.csv" goes here (optional)
DATASETS = DATA / "datasets"       # results of the Stage 1 query
ARTIFACTS = ROOT / "artifacts"     # trained models, iteration log
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"
SQL_DIR = ROOT / "stage1" / "sql"
KB_DIR = ROOT / "kb"               # RAG knowledge base
RT_STATE = ROOT / "realtime" / "state"
RT_OUT = ROOT / "realtime" / "output"

for _p in (DATA, RAW, RAW_BRANCH, EXTERNAL, DATASETS, ARTIFACTS, REPORTS, FIGURES,
           TABLES, RT_STATE, RT_OUT):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42

# ---------------------------------------------------------------- database
# CHURN_DB=sqlite (default, nothing to install) or CHURN_DB=postgres
DB_BACKEND = os.environ.get("CHURN_DB", "sqlite").lower()
PG_URL = os.environ.get(
    "CHURN_PG_URL", "postgresql+psycopg://churn:churn@localhost:5432/bank_dwh")

# ---------------------------------------------------------------- simulation
N_CLIENTS = int(os.environ.get("CHURN_N_CLIENTS", 5000))
N_CLIENTS_BRANCH = int(os.environ.get("CHURN_N_CLIENTS_BRANCH", 1500))
SIM_START = "2025-01-01"
SIM_END = "2026-06-30"

# ---------------------------------------------------------------- Stage 1 query
CUTOFF = "2025-12-31"          # features only from data <= cutoff
LABEL_CUTOFF = "2026-03-31"    # target: leaves in (cutoff, label_cutoff]  (90 days)
OOT_CUTOFF = "2025-09-30"      # out-of-time check: train earlier ...
OOT_LABEL_CUTOFF = "2025-12-29"
LEAKY_FEATURE_CUTOFF = "2026-03-31"  # the WRONG cutoff used in the leakage demo

# ---------------------------------------------------------------- cost matrix
COST_FN = 1000.0     # a lost client = yearly margin
COST_FP = 60.0       # a useless retention offer
OFFER_SAVE_RATE = 0.35  # share of true leavers an offer keeps (used for TP cost)

# ---------------------------------------------------------------- contract matrix defaults
CONTRACT = {
    "M1_recall_min": 0.70,
    "M2_auc_min": 0.80,
    "F1_jaccard_min": 0.60,
    "F2_rankcorr_min": 0.80,
    "F2_psi_max": 0.20,
    "fairness_gap_max": 0.10,
    "leakage_auc_max": 0.97,
    "latency_p95_ms_max": 1000.0,
}

SENSITIVE = ["age", "gender", "country"]
CATEGORICAL = ["gender", "country"]

# ---------------------------------------------------------------- real time
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
TOPICS = ["bank.events", "bank.alerts", "bank.offers", "bank.outcomes"]
SECONDS_PER_BANK_DAY = float(os.environ.get("CHURN_SECONDS_PER_DAY", 2.0))
STREAM_START = "2026-01-01"    # replay = the label window, after the cutoff
STREAM_DAYS = 90
