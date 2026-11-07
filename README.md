# „Loialitate, tăcere și plecare” — bank churn analytics (ȘAD, UTM FCIM)

> *„Ce înseamnă să pleci fără să spui nimic? Cum putem asculta tăcerea unui client?”*

A complete, runnable project that follows the teacher's research cycle:
**problem → KPIs → data contract → DWH (star schema) → the query → dataset → EDA → contract matrix →
ML/DL black box → decision boundaries → XAI (SHAP + LIME) → external raw data → streaming (Kafka → Spark →
Alert → RTAP → Verdict) → RAG → verdict → back to the start.**

It does **not** start from a downloaded dataset. It starts from the problem, simulates the bank's source
systems from a data contract, loads them into a data warehouse, and the dataset is the result of a SQL
query with a cutoff date (Stage 1). Github is used only for external validation and the data-sources
comparison of step 6.4 (optional).

---

## Contents

1. [What is in the box](#1-what-is-in-the-box)
2. [Tools: what the teacher uses, what we use on Ubuntu, and why](#2-tools)
3. [Step 0 — install everything on Ubuntu (step by step)](#3-step-0--install-on-ubuntu)
4. [Step 1 — build everything with one command](#4-build-everything)
5. [Step 2 — the Streamlit cycle app](#5-the-cycle-app)
6. [Step 3 — the real-time system (Kafka, Spark, RTAP)](#6-the-real-time-system)
7. [Show the DWH like SSMS (pgAdmin / DBeaver)](#7-show-the-dwh-like-ssms)
8. [Where each requirement is implemented](#8-where-each-requirement-is)
9. [Writing the report from the results](#9-writing-the-report)
10. [The 5-minute live demo](#10-live-demo)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. What is in the box

```
churn_project/
├── config.py                  dates (cutoff 2025-12-31, label to 2026-03-31), costs, seeds, contract thresholds
├── run_all.py                 ONE command rebuilds the DWH, datasets, models, figures, results summary
├── requirements.txt
├── docker-compose.yml         OPTIONAL alternative to the native install: Kafka + Kafka UI + PostgreSQL
├── stage1/                    ── STAGE 1 (−2 points if missing) ──
│   ├── generator.py           simulates the bank's source systems from the data contract (5 ways of leaving)
│   ├── textgen.py             client messages with typos, sarcasm, empty/contradictory text (noise)
│   ├── etl.py                 source CSV → staging (stg_*) → cleaning → DWH star schema
│   ├── extract_dataset.py     runs THE QUERY with :cutoff / :label_cutoff (and the leakage trap)
│   ├── spark_query.py         the same query in Spark SQL + parity check
│   ├── github_load.py         github file → staging → same path (external validation, step 6.4 only)
│   ├── db.py                  SQLite or PostgreSQL (CHURN_DB=postgres)
│   └── sql/                   schema_*.sql (PK/FK/indexes) and query_features_{sqlite,postgres,spark}.sql
├── ds/                        ── data science ──
│   ├── catalog.py             problem, objectives, 4 core KPIs (M1,M2,F1,F2) + business KPIs, features, data contract
│   ├── cleaning.py            step 2.0 rules as code (reused on the external branch)
│   ├── textproc.py            text denoising + sentiment
│   ├── eda.py                 EDA in the course's order (unique id + 5 steps), hypotheses, leakage demo
│   ├── modeling.py            LR / RF / Gradient Boosting / MLP, balancing, cost threshold, fairness, contract matrix
│   ├── xai.py                 SHAP global/local, LIME, SHAP-vs-LIME client map, stability (F1), reason codes, PSI (F2)
│   ├── fundamentals.py        Bias, Noise, Decision Boundaries, SMOTE: the experiments (before / after)
│   ├── fundamentals_text.py   the five parts of each fundamental: appears, observed (from the numbers), lab answers
│   ├── streaming_sim.py       Streaming fundamental: daily clock path + EWMA vs monthly batch, lead time
│   ├── rtap.py                Thompson Sampling, simulated world, 5 policies compared, ethical rule
│   ├── rag.py                 retrieval over kb/*.md + recommendation with cited sources (optional Ollama)
│   ├── experiments.py         Experiments 1–2, PCA, L1/L2/ElasticNet, Experiment A, Experiment B, sources matrix
│   ├── livefeatures.py        the Stage 1 features in Python for the live path (= SQL, parity tested)
│   ├── scoring.py             live scorer: event path + clock path + EWMA + capacity + SHAP + RAG
│   ├── cycle.py               iterations: evaluate → contract → verdict → log → compare side by side
│   └── monitoring.py          PSI drift, decay, SHAP drift, fairness re-check, retrain rule, model card
├── kb/                        RAG knowledge base: retention playbook, ethics policy, products, methodology
├── app/streamlit_app.py       THE CYCLE APP: one page per box, result + logic + code, all controls
├── app/ui.py                  shared look of both apps (cards, PASS/FAIL, icon credits); theme in .streamlit/config.toml
├── realtime/                  ── lesson 5 structure, teacher's file names ──
│   ├── kafka/                 install / init (format once) / start / create topics / status / reconnect (MODUL_1)
│   ├── kafka_churn_MD.py      producer (one per source: mobile_app, card_branch, contact_center)
│   ├── spark_churn_MD.py      Spark Structured Streaming: 10 s windows + 1 bank-day windows with watermark → CSV
│   ├── alert_churn_MD.py      scorer: event path, clock path, EWMA, SHAP reason, RAG text, challengers (shadow)
│   ├── rtap_churn_MD.py       RTAP: Thompson Sampling offers, capacity, budget, opt-out, human override
│   ├── world_churn_MD.py      the simulated clients' reactions (hidden response rates)
│   ├── river_churn_MD.py      online learning with River (the app_river.py idea applied to churn)
│   ├── consumer_churn_MD.py   plain consumer to watch any topic
│   ├── rtap_app.py            live dashboard: Producer | Kafka | Spark | Alert | Verdict | Control
│   ├── bus.py                 Kafka backend + FILE FALLBACK (no-Kafka mode, same semantics)
│   ├── run_realtime.sh / stop_realtime.sh
│   ├── parity_test.py         live features == SQL features
│   └── teacher/               the teacher's main_KAFKAPRODUSER.py, main_KAFKACONSUMER.py, app_river.py (run once)
├── data/                      (generated) raw source exports, raw_branch, DWH files, datasets
├── artifacts/                 (generated) champion model, challengers, iterations.json
└── reports/                   (generated) results_summary.md, model_card.md, figures/*.png, tables/*.csv
```

---

## 2. Tools

The teacher demonstrates on Windows (SQL Server + SSMS, `.bat` scripts, PyCharm). Everything below works
on Ubuntu; where a better or more practical alternative exists it is **added**, not substituted, and the
reason is given.

| Need | Teacher's tool | Used here on Ubuntu | Why |
|---|---|---|---|
| Data warehouse | SQL Server + SSMS | **PostgreSQL 16** (main) **or SQLite** (zero install, default) | SSMS is Windows-only. PostgreSQL is a real client-server DWH: window functions, real DATE types, Spark can read it over JDBC, other laptops can connect, roles for GDPR. SQLite = one file, nothing to install, ideal fallback. Both run the **same** star schema and query (only date functions differ). |
| DWH GUI | SSMS | **pgAdmin 4** or **DBeaver CE** | same things as SSMS: tree of tables, rows, query tool, ER diagram, explain plan. DBeaver also opens the SQLite file. |
| "Query in Spark" | Spark | **PySpark** (`stage1/spark_query.py`) | Spark comes inside the pip package; the same query runs in Spark SQL and matches the SQL result row by row. |
| MongoDB | allowed | not used | events are relational here (joins + time windows are the core of the query). Say so in the report: Mongo is good for raw events, weaker for joins/windows. |
| Streaming broker | Kafka 4.3.1 KRaft | **Kafka 4.3.1 KRaft** (same version) + **file fallback** | same steps as the teacher's guide (`.sh` instead of `.bat`). The fallback (`--bus file`) keeps offsets, consumer groups and replay on disk, so the class demo cannot fail. |
| Stream processing | Spark Structured Streaming | **same** | processing-time and event-time windows with watermark, CSV output ("book of records"). |
| Online learning | River (`app_river.py`) | **River** (`river_churn_MD.py`) | the teacher's idea (score_one then learn_one) applied to the bank's events. |
| App | Streamlit | **Streamlit** (2 apps) | cycle app + live 5-window dashboard. |
| Editor | PyCharm | PyCharm Community or VS Code | either works; open the folder, choose the `.venv` interpreter. |
| Containers | – | Docker Engine (optional) | the same Kafka and PostgreSQL in one command, plus a Kafka web UI. The main path and the demo stay native (terminal). Not Docker Desktop. |
| LLM for RAG | – | template (default) or **Ollama** (optional) | RAG works offline with a template; a local LLM can rewrite the text, only from the retrieved passages. |

Not needed: SQL Server, Docker Desktop, a separate Spark download, ZooKeeper (removed in Kafka 4), a GPU.

---

## 3. Step 0 — install on Ubuntu

Works on Ubuntu 22.04 / 24.04 / 26.04, x86_64. System commands run from any folder; project commands
must run **inside `~/churn_project` with the venv active**.

### 3.1 Base system, Git, Java (needed by Kafka and Spark)

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl wget gpg build-essential python3 python3-venv python3-pip python3-dev openjdk-17-jdk
# if openjdk-17-jdk is not found on your Ubuntu:  sudo apt install -y openjdk-21-jdk
java -version        # must say 17 or 21
python3 --version    # 3.10 – 3.13 are fine
```

### 3.2 Put the project in your home folder and create the Python environment

```bash
cd ~
unzip churn_project.zip          # -> ~/churn_project   (no spaces in the path, not on a USB stick)
cd ~/churn_project
python3 -m venv .venv
source .venv/bin/activate        # the prompt now starts with (.venv)  — do this in EVERY new terminal
pip install --upgrade pip
pip install -r requirements.txt  # ~3-5 minutes (pyspark is ~300 MB)
```

If a library fails to build on a very new Python (e.g. 3.14), create the venv with Python 3.12 via `uv`:
```bash
pip install --user uv && uv venv --python 3.12 .venv && source .venv/bin/activate && uv pip install -r requirements.txt
```

### 3.3 PostgreSQL + the project database (recommended; skip to use SQLite only)

```bash
sudo apt install -y postgresql postgresql-contrib
sudo systemctl enable --now postgresql
systemctl is-active postgresql                     # -> active
sudo -u postgres psql -c "CREATE USER churn WITH PASSWORD 'churn';"
sudo -u postgres psql -c "CREATE DATABASE bank_dwh OWNER churn;"
```
The project uses PostgreSQL when you set `CHURN_DB=postgres` (otherwise SQLite):
```bash
echo 'export CHURN_DB=postgres' >> ~/.bashrc && source ~/.bashrc
```

### 3.4 A database GUI (to show the DWH like the teacher does in SSMS)

DBeaver (simplest, also opens SQLite): `sudo snap install dbeaver-ce`

or pgAdmin 4 desktop (official repository, from pgadmin.org):
```bash
curl -fsS https://www.pgadmin.org/static/packages_pgadmin_org.pub | sudo gpg --dearmor -o /usr/share/keyrings/packages-pgadmin-org.gpg
sudo sh -c 'echo "deb [signed-by=/usr/share/keyrings/packages-pgadmin-org.gpg] https://ftp.postgresql.org/pub/pgadmin/pgadmin4/apt/$(lsb_release -cs) pgadmin4 main" > /etc/apt/sources.list.d/pgadmin4.list'
sudo apt update && sudo apt install -y pgadmin4-desktop
```
(if your Ubuntu release is too new for the pgAdmin repository, use DBeaver.)
Connect: host `localhost`, port `5432`, database `bank_dwh`, user `churn`, password `churn`.

### 3.5 Kafka 4.3.1 (KRaft) — same version as the teacher's guide

```bash
cd ~/churn_project
bash realtime/kafka/install_kafka.sh     # downloads into ~/kafka/kafka_2.13-4.3.1
bash realtime/kafka/init_kafka.sh        # ONCE: random-uuid + format --standalone (golden rule 1)
```
Every session afterwards, terminal 1: `bash realtime/kafka/start_kafka.sh` (keep it open), then once:
`bash realtime/kafka/create_topics.sh` (golden rule 2: creates the 4 topics only if they do not exist).

Alternative with Docker Engine (optional): `docker compose up -d kafka kafka-ui` → Kafka on `localhost:9092`,
web UI on http://localhost:8080. Docker Engine install: Docker's official apt repository (docs.docker.com);
if it does not support your release yet: `sudo apt install -y docker.io docker-compose-v2`, then
`sudo usermod -aG docker $USER` and log out/in; check with `docker run hello-world`.

### 3.6 Editor

`sudo snap install pycharm-community --classic` (as in the teacher's materials) or `sudo snap install code --classic`.
Open `~/churn_project`, set the interpreter to `~/churn_project/.venv/bin/python`.

### 3.7 Optional extras

| Extra | Install | Used by |
|---|---|---|
| SDV (learned synthetic data) | `pip install sdv` | Experiment B (otherwise a built-in Gaussian copula is used) |
| sentence embeddings | `pip install sentence-transformers` | text-embedding PCA (otherwise TF-IDF + PCA) |
| Ollama local LLM for RAG | see ollama.com, then `ollama pull llama3.2` and `export OLLAMA_MODEL=llama3.2` | RAG text |
| github "Churn Modelling" | download `Churn_Modelling.csv` from github into `data/external/` | external validation, Experiment B, sources matrix (step 6.4) |

### 3.8 On Windows

**Recommended: WSL2 (Ubuntu inside Windows).** Everything in this guide then works unchanged, including Kafka,
Spark and the `.sh` scripts.
1. PowerShell as administrator: `wsl --install -d Ubuntu`, restart, open "Ubuntu" from the Start menu and create a user.
2. Copy the project into the Linux home (faster than `/mnt/c/...`): `cp /mnt/c/Users/<you>/Downloads/churn_project.zip ~ && cd ~ && unzip churn_project.zip`.
3. Follow sections 3.1–3.5 as written. The apps open in the Windows browser at http://localhost:8501 as usual.

**Without WSL (analysis part only).** Python 3.12 from python.org, then in PowerShell, inside the project folder:
```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
$env:PYTHONUTF8 = "1"            # Romanian diacritics in the reports
pip install -r requirements.txt
python run_all.py --fast
streamlit run app/streamlit_app.py
```
SQLite needs nothing else. The real-time part relies on `.sh` scripts, and Spark needs extra Hadoop files on
native Windows, so run it under WSL2.

---

## 4. Build everything

```bash
cd ~/churn_project && source .venv/bin/activate
python run_all.py --fast        # ~4-5 min: everything with fewer repetitions (use this first)
python run_all.py               # ~6-10 min: Experiments 1–2 with 100 repetitions (for the report)
```
What it does, in the order of the cycle (each step can also be run alone):

| # | Command (alone) | Output |
|---|---|---|
| 1 | `python stage1/generator.py` and `--branch` | `data/raw/*.csv` (5 source systems) and the messy external branch |
| 2 | `python stage1/etl.py` and `--source branch` | staging `stg_*` → cleaning report → star schema with PK/FK/indexes |
| 3 | `python stage1/extract_dataset.py` | **the dataset**: 4 213 clients × 28 features, churn 16.3 % |
| 4 | `python stage1/extract_dataset.py --leaky` | the leakage trap (wrong cutoff) |
| 5 | `python stage1/spark_query.py` | the same query in Spark SQL + parity check |
| 6 | `python realtime/parity_test.py` | live Python features == SQL features |
| 7 | (in run_all) | EDA, iterations 1→2→3 with contract matrix, models, XAI, 6 fundamentals, lab experiments, monitoring |

Results: **`reports/results_summary.md`** (every number, measured), `reports/figures/*.png`,
`reports/tables/*.csv`, `reports/model_card.md`, `artifacts/iterations.json`.

To use PostgreSQL: `CHURN_DB=postgres python run_all.py --fast` (identical numbers — tested).
Fixed seed (42) → same data and results every time.

---

## 5. The cycle app

```bash
cd ~/churn_project                       # run from the project folder: the theme lives in .streamlit/config.toml
streamlit run app/streamlit_app.py       # opens http://localhost:8501
```
**Look and feel.** Theme in `.streamlit/config.toml` (bigger base font 17 px, plum/teal palette, dark-plum sidebar),
extra styling in `app/ui.py` (hero banner, KPI cards, PASS/FAIL pills), charts in `ds/viz.py` (Plotly: hover for
values, drag to zoom, double-click to reset, click a legend item to hide a series). Chart colours are fixed per
meaning — **teal = stays, pink = leaves** everywhere — and were checked for colour-blind separation and contrast.
To change the palette, edit `CAT` in `ds/viz.py` and `chartCategoricalColors` in `.streamlit/config.toml`.

**Logos.** `app/assets/churn.png` (cycle app) and `app/assets/emergency.png` (live dashboard), Flaticon icons by
iconsitee and ViconsDesign; the apps show a short clickable credit in the sidebar (`app/assets/README.md`).

One page per box of the board, each with the result **and** an expander "Logic & code". Sidebar: data source
(simulated bank / external branch / github), cutoff date, leakage switch.

| Control (step 8.3) | Where | What visibly changes |
|---|---|---|
| data source and cutoff, wrong cutoff | sidebar, page 2 | dataset; leaky → AUC ≈ 1.0 and one dominant feature |
| contract thresholds, cost values | page 4 | which rows pass/fail (pages 6, 12) |
| model + hyperparameters | page 6 | metrics, boundaries, explanations |
| features on/off, drop sensitive + proxy | page 5 | metrics, fairness gaps, SHAP ranking |
| balancing none / class weights / ROS / SMOTE / SMOTE-NC | page 6 | recall, precision |
| threshold from cost / cost+recall / 0.5 / manual | page 6, slider on page 7 | confusion matrix, cost, recall |
| per-group thresholds (experiment only) | page 6 | fairness gaps vs overall recall |
| noise on X (σ), noise on Y (% flipped) | page 6 | metric drop (Y hurts more) |
| bias on Y (silent leavers labelled "stayed"), bias on X (remove part of a group) | page 6 | recall of that group collapses |
| client picker | pages 8, 11 | SHAP + LIME + reason code + RAG text |
| client map: click / box / lasso clients, two agreement thresholds | page 8, tab "Client map" | SHAP vs LIME per feature, OK / FLAGGED, CSV export |
| "save and go back to the start" | page 12 | new row in the iteration log; compare iterations side by side |

---

## 6. The real-time system

Architecture (lesson 5, cameras replaced by the bank's systems):
```
mobile_app ─┐                       ┌─> alert_churn_MD (scorer: event path + clock path, EWMA, SHAP, RAG) ─> bank.alerts ─> rtap_churn_MD ─> bank.offers ─> world_churn_MD
card_branch ├─> Kafka bank.events ──┼─> spark_churn_MD (10 s + 1 bank-day windows, watermark) ─> CSV                         ^                          │
contact_ctr ┘   (key = client_id)   └─> river_churn_MD (online HalfSpaceTrees)                              bank.outcomes <────────┴──────────────────────────┘
                                                        rtap_app.py: Producer | Kafka | Spark | Alert | Verdict | Control
```

### Order of terminals (golden rule 3)

```bash
# T1  broker
bash realtime/kafka/start_kafka.sh              # (or: docker compose up -d kafka kafka-ui)
# T2  everything else in the right order (scorer/RTAP/Spark before the producers), logs in realtime/logs/
bash realtime/run_realtime.sh                   # 2 s = 1 bank day, 90 days ≈ 3 minutes
# T3  watch messages (optional)
python realtime/consumer_churn_MD.py --topic bank.alerts
# T4  dashboard
streamlit run realtime/rtap_app.py --server.port 8502
# stop
bash realtime/stop_realtime.sh
```
Prefer one terminal per service for the demo? Open them yourself in this order:
`alert_churn_MD.py`, `rtap_churn_MD.py`, `world_churn_MD.py`, `spark_churn_MD.py`, `river_churn_MD.py`, then the three
`kafka_churn_MD.py --source …` (first create the shared clock: `python -c "from realtime.bus import new_run; new_run()"`).

**No-Kafka fallback (always works):** `bash realtime/run_realtime.sh --bus file` — same scripts, same topics,
offsets and consumer groups, kept as files under `realtime/bus/`. Spark reads that folder as a file stream.
Low RAM? add `--no-spark`. Faster demo: `SECONDS_PER_DAY=1 DAYS=30 bash realtime/run_realtime.sh --bus file`.

**Several laptops (distributed demo):** on the broker laptop `bash realtime/kafka/start_kafka.sh --lan`
(it prints its IP; open the port with `sudo ufw allow 9092/tcp`). On each other laptop (same Wi-Fi, project
copied, `python stage1/generator.py` run once — same seed gives the same source exports):
```bash
ping 192.168.1.20
python realtime/kafka_churn_MD.py --source contact_center --bootstrap 192.168.1.20:9092 --t0 <t0> --run-id <run_id>
```
(`t0` and `run_id` are in `realtime/state/run.json` on the main laptop; clocks must be in sync — Ubuntu uses NTP by default).

**From Teacher's shared materials basic round trip (run once):** `bash realtime/kafka/teacher_demo.sh` and follow the printed two commands.
**Reconnect after a disconnect (MODUL_1):** `bash realtime/kafka/reconnect_kafka.sh` — never reformat; consumers
resume from their committed offsets. **Topics/offsets/lag:** `bash realtime/kafka/status_kafka.sh`.

What to explain: *partitions* (key = client_id keeps each client's events in order), *offsets* (position in the
log), *consumer groups* (scorer, RTAP, River each read the full stream independently), *commit after processing*
(at-least-once), *watermark* (when Spark considers a bank day complete), and **how silence is detected with no
events**: the clock path rescores every active client when a bank day closes, so days without activity grow and
the risk rises without any message.

---

## 7. Show the DWH like SSMS

| SSMS | pgAdmin 4 / DBeaver |
|---|---|
| databases → tables → columns | tree on the left (schema `bank`, `branch`; `github` if loaded) |
| open a table | right-click → View/Edit Data (e.g. `bank.fact_transaction`) |
| new query | Query Tool / SQL Editor: paste `stage1/sql/query_features_postgres.sql`, replace `:cutoff` by `DATE '2025-12-31'`, `:label_cutoff` by `DATE '2026-03-31'`, `:feature_cutoff` by `DATE '2025-12-31'`, `:denoise` by `1` |
| database diagram | ERD Tool (pgAdmin) / ER Diagram tab (DBeaver) — drawn from the PK/FK |
| execution plan | Explain / Explain Analyze |
| leakage re-run | same query with `:feature_cutoff` = `DATE '2026-03-31'` |

With SQLite: DBeaver → New connection → SQLite → `~/churn_project/data/dwh_bank.sqlite`.

---

## 8. Where each requirement is

| Requirement | Implemented in | Shown in the app |
|---|---|---|
| Stage 1: problem, goal, objectives, hypotheses | `ds/catalog.py` | page 1 |
| 4 core KPIs (M1, M2, F1, F2) + business KPIs | `ds/catalog.py`, measured in `run_all.py` | pages 1, 4, 12 |
| features with formula, source, reason (incl. silence) | `ds/catalog.py` | page 5 |
| data contract | `ds/catalog.py` (+ `validate_contract`) | pages 2, 4 |
| source systems → staging → ETL → star schema (PK/FK/indexes) | `stage1/generator.py`, `etl.py`, `sql/schema_*.sql` | page 2 |
| the query with :cutoff / :label_cutoff, CTE per group, LEFT JOINs, eligible population | `stage1/sql/query_features_*.sql` | page 2 |
| Spark version of the query | `stage1/spark_query.py` | — |
| leakage trap | `extract_dataset.py --leaky`, `eda.leakage_demo` | page 2, sidebar |
| cleaning as code, reused externally | `ds/cleaning.py` | pages 2, 9 |
| EDA (unique id + 5 steps), hypotheses | `ds/eda.py` | page 3 |
| contract matrix + cost matrix + threshold from cost | `ds/modeling.py` | pages 4, 6, 7, 12 |
| LR, RF, Gradient Boosting, MLP; CV; tuning; out-of-time; complexity curve | `ds/modeling.py`, `fundamentals.boundaries`, `run_all.py` | pages 6, 7 |
| XAI pass 1 (SHAP global/local, LIME, SHAP-vs-LIME client map, reason codes, stability) | `ds/xai.py` | page 8 |
| XAI pass 2 on external raw data (PSI, rank correlation) | `ds/xai.py`, `ds/cycle.py` | page 9 |
| six fundamentals, five parts each (appears, measured, observed, mitigation, lab answers) | `ds/fundamentals.py`, `fundamentals_text.py`, `streaming_sim.py`, `rtap.py` | page ★ Six fundamentals |
| Experiments 1–2, PCA, regularization, Exp. A, Exp. B, sources matrix | `ds/experiments.py` | page ★ Lab experiments |
| Kafka → Spark → Alert → RTAP → Verdict, 3 producers, fallback, parity test | `realtime/` | page 10 + `rtap_app.py` |
| RAG with cited sources | `ds/rag.py`, `kb/` | page 11, every alert |
| the loop: iterations, side-by-side comparison | `ds/cycle.py` | page 12 |
| monitoring, retrain rule, model card | `ds/monitoring.py` | page ★ Monitoring |
| reproducibility (one command, seeds) | `run_all.py`, `config.py` | — |

---

## 9. Writing the report

`reports/results_summary.md` follows the report's structure and contains **only measured numbers**. For each
fundamental it has the five parts in the template's order (2.2.1 Bias … 2.2.6 RTAP): how it appears, the
before/after tables, what was observed (sentences computed from those tables), the mitigation sentences
(„S-a efectuat …, deoarece …”) and the answers to the lab questions.

Notes on the results:

* **What is simulated and why:** the bank, the clients, the messages and the reactions to offers (real bank data
  is not available to students). The sentiment lexicon was built for the simulator's message templates, so
  denoising looks better here than it would on real text, which is stated as a limit.
* **The loop:** iteration 1 (all features) fails on fairness and on external drift (`digital_share_90d` drifts on
  the branch and is a proxy for age); iteration 2 removes the sensitive attributes and the proxy → fairness and
  drift pass, recall drops below 0.70; iteration 3 keeps one threshold for all clients, the cheapest one whose
  out-of-fold recall is at least 0.74 (contract 0.70 + safety margin) → PASS. No decision depends on age.
  Side-by-side table: `iterations_side_by_side`.
* **Streaming vs batch:** with the same team capacity, overall recall is similar (the team is the bottleneck), but
  for leavers whose risk appears *after* the cutoff streaming catches clearly more, earlier: the value of
  listening to silence daily. EWMA cuts false alarms.
* **RTAP:** Thompson Sampling learns which offer works per reason without knowing response rates; expert rules are a
  strong baseline when the expert is right; the offer mix is the same across genders/countries (ethical rule).
* Fairness is measured on out-of-sample predictions for all clients (CV + test), because test groups alone are
  too small for stable recall estimates.

---

## 10. Live demo

1. DBeaver/pgAdmin: tables, rows of `fact_transaction`, ER diagram, run the query, then the leakage re-run.
2. `streamlit run app/streamlit_app.py`: pages 1 → 2 → 4 → 6 (move noise on Y) → 8 (silent vs vocal client) → 12
   (iterations side by side).
3. Start Kafka + `bash realtime/run_realtime.sh`, open `rtap_app.py`: walk the 5 windows, open one alert (SHAP reason
   + RAG text), show what RTAP learned and the verdict. If anything fails: `bash realtime/run_realtime.sh --bus file`.

---

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `No dataset yet` in the app | `python run_all.py --fast` |
| `psycopg.OperationalError` | PostgreSQL not running (`sudo systemctl start postgresql`) or user/db not created (3.3); or use SQLite: `unset CHURN_DB` |
| Kafka `NoBrokersAvailable` | start the broker (T1) and wait for "Kafka Server started"; check `ss -ltn | grep 9092` |
| Kafka will not start after a crash | `bash realtime/kafka/reconnect_kafka.sh` (do **not** run init again) |
| Spark Kafka connector download fails (no internet) | use `--bus file`, or run Spark once with internet so it caches the connector in `~/.ivy2` |
| Java errors from Spark | `java -version` must be 17 or 21; `sudo update-alternatives --config java` |
| Machine slow during the stream | `--no-spark`, or `SECONDS_PER_DAY=3`; Spark needs ~1 GB RAM |
| Other laptop cannot connect | broker started with `--lan`, `sudo ufw allow 9092/tcp`, same Wi-Fi, `ping <IP>` |
| pyspark warns about pandas ≥ 3 | harmless here; or `pip install "pandas<3"` |
| start over completely | `rm -rf data artifacts reports realtime/output realtime/state realtime/bus && python run_all.py --fast` |

---

### About `reference_results/`
A copy of `reports/` from a full run (SQLite, Python 3.13, seed 42; the same run on PostgreSQL and on
Python 3.12 + pandas 2 gives the same verdicts), so the results can be read without running anything.
A new run writes to `reports/`.
