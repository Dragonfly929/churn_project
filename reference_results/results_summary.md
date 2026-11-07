# Results summary — „Loialitate, tăcere și plecare”

Engine: **sqlite**, cutoff 2025-12-31, label window to 2026-03-31, seed 42. All numbers below were MEASURED by run_all.py — use these in the report, never example numbers.


## 1.1-1.2 Problem, goal, objectives, KPIs (defined before any data)

Clienții pleacă din bancă în tăcere: își reduc treptat activitatea, nu se plâng și nu răspund la sondaje, iar banca observă plecarea abia când contul este închis. Fiecare client pierdut costă marja anuală (~1 000 €).

**Goal:** Reducerea costului churn-ului prin detectarea timpurie a riscului și intervenții de retenție rentabile și corecte.

- O1. Detectarea riscului de plecare cu cel puțin 30 de zile înainte de plecare (lead time).
- O2. Identificarea a cel puțin 70% dintre clienții care pleacă în următoarele 90 de zile (recall).
- O3. Costul campaniei de retenție mai mic decât valoarea clienților salvați (ROI > 1).
- O4. Detectarea clienților tăcuți (fără feedback, fără răspuns la sondaj) – întrebarea de cercetare.
- O5. Diferența de recall între grupuri (vârstă, gen, țară) de cel mult 0.10.
- O6. Explicații stabile și verificabile pentru fiecare alertă (SHAP/LIME).

*Table: kpis*

| id   | judges   | KPI                                               | formula                                                                   | target                         | source                         |
|:-----|:---------|:--------------------------------------------------|:--------------------------------------------------------------------------|:-------------------------------|:-------------------------------|
| M1   | model    | Recall-ul clienților care pleacă (la pragul ales) | TP / (TP + FN)                                                            | >= 0.70                        | set de test                    |
| M2   | model    | Cost așteptat vs. fără model; ROC-AUC             | FN·1000€ + FP·60€ + TP·(60€ + 0.65·1000€)                                 | cost < fără model; AUC >= 0.80 | set de test                    |
| F1   | features | Stabilitatea explicațiilor (top-5 SHAP)           | Jaccard mediu al top-5 între reantrenări                                  | >= 0.60                        | 5 reantrenări (seed/bootstrap) |
| F2   | features | Robustețea pe date externe                        | corelația Spearman a importanțelor SHAP intern vs extern; PSI per feature | rho >= 0.80; PSI < 0.20        | filiala externă                |
| B1   | business | Rata churn                                        | plecați în 90 zile / clienți activi la cutoff                             | monitorizare                   | query Stage 1                  |
| B2   | business | Ponderea plecărilor tăcute                        | plecați tăcuți / toți plecații                                            | monitorizare                   | query Stage 1                  |
| B3   | business | Lead time                                         | zile între alertă și plecare (mediană)                                    | >= 30 zile                     | streaming                      |
| B4   | business | ROI campanie                                      | valoare reținută / cost oferte                                            | > 1                            | RTAP                           |
| B5   | business | Fairness gap                                      | max - min recall între grupuri                                            | <= 0.10                        | set de test                    |


## 1.3-1.4 Features and data contract

*Table: features_catalog*

| feature             | group     | type       | formula                                        | source                                    | why                                       |
|:--------------------|:----------|:-----------|:-----------------------------------------------|:------------------------------------------|:------------------------------------------|
| age                 | profile   | discrete   | floor((cutoff - birth_date)/365.25)            | dim_client.birth_date                     | life stage; SENSITIVE attribute           |
| gender              | profile   | nominal    | as recorded                                    | dim_client.gender                         | SENSITIVE attribute (fairness check only) |
| country             | profile   | nominal    | as recorded                                    | dim_client.country                        | SENSITIVE attribute; regional competition |
| tenure_months       | profile   | discrete   | floor((cutoff - join_date)/30.44)              | dim_client.join_date                      | new clients leave more easily             |
| credit_score        | profile   | continuous | last monthly snapshot                          | fact_account_month.credit_score           | financial health                          |
| num_products        | profile   | discrete   | last monthly snapshot                          | fact_account_month.num_products           | 1 product = weak tie; 3-4 = over-selling  |
| has_card            | profile   | binary     | last monthly snapshot                          | fact_account_month.has_card               | card = daily touchpoint                   |
| is_active_member    | profile   | binary     | >=4 transactions in the month                  | fact_account_month.is_active_member       | engagement                                |
| est_salary          | profile   | continuous | last monthly snapshot                          | fact_account_month.est_salary             | client value                              |
| balance             | profile   | continuous | last monthly snapshot                          | fact_account_month.balance                | money kept at the bank                    |
| balance_change_3m   | behaviour | continuous | (bal_t - bal_t-3)/bal_t-3                      | fact_account_month.balance                | money being moved out                     |
| tx_count_90d        | behaviour | discrete   | count tx in (cutoff-90, cutoff]                | fact_transaction                          | current activity                          |
| tx_count_prev_90d   | behaviour | discrete   | count tx in (cutoff-180, cutoff-90]            | fact_transaction                          | previous activity (baseline)              |
| tx_trend            | behaviour | continuous | (tx_90d+1)/(tx_prev_90d+1)                     | fact_transaction                          | fading activity (gradual fade)            |
| amount_in_90d       | behaviour | continuous | sum in-amounts 90d                             | fact_transaction.amount                   | salary moved to another bank              |
| amount_out_90d      | behaviour | continuous | sum out-amounts 90d                            | fact_transaction.amount                   | money drain                               |
| net_flow_90d        | behaviour | continuous | in - out 90d                                   | fact_transaction.amount                   | money drain                               |
| digital_share_90d   | behaviour | continuous | digital tx / all tx 90d                        | fact_transaction + dim_channel.is_digital | channel habit; PROXY for age              |
| msg_count_180d      | voice     | discrete   | messages in 180d                               | fact_feedback                             | the client still talks to the bank        |
| complaints_180d     | voice     | discrete   | complaints in 180d                             | fact_feedback.is_complaint                | dissatisfaction (vocal leaver)            |
| neg_msg_180d        | voice     | discrete   | negative messages (denoised sentiment)         | fact_feedback.is_negative                 | dissatisfaction                           |
| avg_sentiment_180d  | voice     | continuous | mean sentiment of non-empty messages           | fact_feedback.sentiment_score             | tone of the relationship                  |
| days_since_last_tx  | silence   | discrete   | cutoff - last tx date (365 if none)            | fact_transaction                          | silence in payments                       |
| days_since_last_msg | silence   | discrete   | cutoff - last message date (365 if none)       | fact_feedback                             | silence in contact                        |
| no_feedback_6m      | silence   | binary     | msg_count_180d = 0                             | fact_feedback                             | the client stopped talking                |
| nps_score           | silence   | ordinal    | last answered NPS (0-10), NULL if not answered | fact_survey.nps_score                     | satisfaction when spoken                  |
| nps_missing         | silence   | binary     | last survey not answered                       | fact_survey.answered_date_key             | missing NOT at random = silence           |
| silent_decline      | silence   | binary     | tx_trend < 0.5 AND no_feedback_6m              | derived                                   | silent exit pattern                       |

*Table: data_contract*

| source_system   | table                 | column         | type          | allowed_values                              | update_frequency   | exists_since   |
|:----------------|:----------------------|:---------------|:--------------|:--------------------------------------------|:-------------------|:---------------|
| core banking    | core_clients          | client_id      | integer       | unique, > 0                                 | once               | 2010           |
| core banking    | core_clients          | birth_date     | date          | age 18-100 at cutoff                        | once               | 2010           |
| core banking    | core_clients          | gender         | char(1)       | F / M                                       | once               | 2010           |
| core banking    | core_clients          | country        | char(2)       | MD / RO / UA                                | once               | 2010           |
| core banking    | core_clients          | join_date      | date          | <= today                                    | once               | 2010           |
| core banking    | core_accounts_monthly | balance        | numeric(16,2) | 0 - 1e8                                     | monthly            | 2025-01        |
| core banking    | core_accounts_monthly | num_products   | smallint      | 1 - 4                                       | monthly            | 2025-01        |
| core banking    | core_accounts_monthly | credit_score   | numeric(6,1)  | 300 - 900, NULL allowed                     | monthly            | 2025-01        |
| core banking    | core_exits            | exit_date      | date          | > join_date                                 | event              | 2025-01        |
| cards/payments  | card_transactions     | amount         | numeric(14,2) | > 0                                         | real time          | 2025-01        |
| cards/payments  | card_transactions     | direction      | text          | in / out                                    | real time          | 2025-01        |
| cards/payments  | card_transactions     | channel        | text          | mobile_app/web/card_pos/atm/branch/transfer | real time          | 2025-01        |
| contact centre  | contact_messages      | text           | text          | free text, may be empty                     | real time          | 2025-01        |
| contact centre  | contact_messages      | complaint_flag | smallint      | 0 / 1                                       | real time          | 2025-01        |
| survey          | survey_responses      | nps_score      | smallint      | 0 - 10, NULL = not answered                 | quarterly          | 2025-03        |


## 1.5 Source systems -> staging -> ETL -> DWH

*Table: cleaning_report_bank*

| table                 | rule                                        |   rows_affected | decision                                                               |
|:----------------------|:--------------------------------------------|----------------:|:-----------------------------------------------------------------------|
| core_clients          | duplicate client_id                         |               0 | keep first row                                                         |
| core_clients          | country name variants                       |               0 | map to ISO-2 code (MD/RO/UA)                                           |
| core_clients          | impossible age (<18 or >100)                |               0 | set birth_date NULL (age missing)                                      |
| core_clients          | drop full_name (GDPR data minimisation)     |            5000 | column removed                                                         |
| core_accounts_monthly | comma decimal separator                     |               0 | convert to dot decimal                                                 |
| core_accounts_monthly | impossible balance (>1e8)                   |               0 | set NULL                                                               |
| core_accounts_monthly | duplicate (client, month)                   |               0 | keep first                                                             |
| core_accounts_monthly | missing credit_score                        |            2301 | keep NULL; imputed later inside the model pipeline (median)            |
| card_transactions     | duplicate tx_id                             |               0 | keep first (same transaction twice)                                    |
| card_transactions     | date as dd.mm.yyyy text                     |               0 | parse with dayfirst                                                    |
| card_transactions     | amount missing or <= 0                      |               0 | drop row                                                               |
| contact_messages      | empty message                               |             491 | keep the row (contact happened), flag is_empty, no sentiment           |
| contact_messages      | garbled text (encoding errors)              |               0 | treat as empty                                                         |
| contact_messages      | text denoising (typos, diacritics, sarcasm) |           12802 | text_clean + sentiment_score; raw kept as sentiment_raw for comparison |
| survey_responses      | survey not answered (NULL score)            |           14806 | keep NULL: missing NOT at random -> feature nps_missing (silence)      |

*Table: cleaning_report_branch*

| table                 | rule                                        |   rows_affected | decision                                                               |
|:----------------------|:--------------------------------------------|----------------:|:-----------------------------------------------------------------------|
| core_clients          | duplicate client_id                         |              15 | keep first row                                                         |
| core_clients          | country name variants                       |            1068 | map to ISO-2 code (MD/RO/UA)                                           |
| core_clients          | impossible age (<18 or >100)                |               7 | set birth_date NULL (age missing)                                      |
| core_clients          | drop full_name (GDPR data minimisation)     |            1500 | column removed                                                         |
| core_accounts_monthly | comma decimal separator                     |            4611 | convert to dot decimal                                                 |
| core_accounts_monthly | impossible balance (>1e8)                   |               5 | set NULL                                                               |
| core_accounts_monthly | duplicate (client, month)                   |               0 | keep first                                                             |
| core_accounts_monthly | missing credit_score                        |             681 | keep NULL; imputed later inside the model pipeline (median)            |
| card_transactions     | duplicate tx_id                             |            3564 | keep first (same transaction twice)                                    |
| card_transactions     | date as dd.mm.yyyy text                     |           53508 | parse with dayfirst                                                    |
| card_transactions     | amount missing or <= 0                      |               0 | drop row                                                               |
| contact_messages      | empty message                               |             141 | keep the row (contact happened), flag is_empty, no sentiment           |
| contact_messages      | garbled text (encoding errors)              |             190 | treat as empty                                                         |
| contact_messages      | text denoising (typos, diacritics, sarcasm) |            3775 | text_clean + sentiment_score; raw kept as sentiment_raw for comparison |
| survey_responses      | survey not answered (NULL score)            |            4418 | keep NULL: missing NOT at random -> feature nps_missing (silence)      |


## 1.7 The dataset (output of the query)

Shape: **4213 clients × 29 features**, churn share **0.163**. External branch: 1245 clients, churn share 0.196.

*Table: data_contract_check*

| rule                                               | pass   |
|:---------------------------------------------------|:-------|
| client_id unique                                   | True   |
| age in 18-100                                      | True   |
| gender in {F,M}                                    | True   |
| country in {MD,RO,UA}                              | True   |
| num_products in 1-4                                | True   |
| balance >= 0                                       | True   |
| nps_score in 0-10                                  | True   |
| missing share per feature < 40% (except nps_score) | True   |
| churn is 0/1                                       | True   |


## 2. EDA (course steps)

Step 0 — unique identifier: {'identifier': 'client_id (one row per client at the cutoff)', 'rows': 4213, 'unique': 4213, 'is_unique': True}

*Table: eda_missing*

|              |   missing |   share | why / decision                                                                     |
|:-------------|----------:|--------:|:-----------------------------------------------------------------------------------|
| nps_score    |      2554 |   0.606 | NOT at random: silent clients do not answer the survey -> kept as nps_missing flag |
| credit_score |       128 |   0.030 | completely at random (CRM glitch) -> median imputation inside the pipeline         |

*Table: eda_types*

| feature             | group     | type       |      mean |   n_unique |
|:--------------------|:----------|:-----------|----------:|-----------:|
| age                 | profile   | discrete   |    40.426 |         65 |
| gender              | profile   | nominal    |   nan     |          2 |
| country             | profile   | nominal    |   nan     |          3 |
| tenure_months       | profile   | discrete   |    64.885 |        129 |
| credit_score        | profile   | continuous |   650.641 |       2462 |
| num_products        | profile   | discrete   |     1.590 |          4 |
| has_card            | profile   | binary     |     0.692 |          2 |
| is_active_member    | profile   | binary     |     0.787 |          2 |
| est_salary          | profile   | continuous | 31546.978 |       4049 |
| balance             | profile   | continuous |  9644.229 |       4204 |
| balance_change_3m   | behaviour | continuous |     0.083 |       2939 |
| tx_count_90d        | behaviour | discrete   |    23.360 |         80 |
| tx_count_prev_90d   | behaviour | discrete   |    22.179 |         77 |
| tx_trend            | behaviour | continuous |     1.429 |        813 |
| amount_in_90d       | behaviour | continuous |  5802.892 |       4044 |
| amount_out_90d      | behaviour | continuous |   887.858 |       4163 |
| net_flow_90d        | behaviour | continuous |  4915.033 |       4205 |
| digital_share_90d   | behaviour | continuous |     0.705 |        497 |
| msg_count_180d      | voice     | discrete   |     0.944 |         12 |
| complaints_180d     | voice     | discrete   |     0.204 |          9 |
| neg_msg_180d        | voice     | discrete   |     0.283 |          9 |
| avg_sentiment_180d  | voice     | continuous |     0.030 |         72 |
| days_since_last_tx  | silence   | discrete   |     5.237 |         48 |
| days_since_last_msg | silence   | discrete   |   180.671 |        364 |
| no_feedback_6m      | silence   | binary     |     0.448 |          2 |
| nps_score           | silence   | ordinal    |     7.096 |         11 |
| nps_missing         | silence   | binary     |     0.606 |          2 |
| silent_decline      | silence   | binary     |     0.009 |          2 |

*Table: eda_churn_by_age_group*

| age_group   |   churn_rate |   count |
|:------------|-------------:|--------:|
| 18-29       |        0.093 |     851 |
| 30-44       |        0.154 |    1803 |
| 45-59       |        0.201 |    1261 |
| 60+         |        0.262 |     298 |

*Table: eda_churn_by_country*

| country   |   churn_rate |   count |
|:----------|-------------:|--------:|
| MD        |        0.150 |    2547 |
| RO        |        0.207 |    1054 |
| UA        |        0.144 |     612 |

*Table: eda_churn_by_gender*

| gender   |   churn_rate |   count |
|:---------|-------------:|--------:|
| F        |        0.164 |    2139 |
| M        |        0.162 |    2074 |

*Table: eda_churn_by_num_products*

|   num_products |   churn_rate |    count |
|---------------:|-------------:|---------:|
|          1.000 |        0.201 | 2055.000 |
|          2.000 |        0.091 | 1894.000 |
|          3.000 |        0.393 |  201.000 |
|          4.000 |        0.349 |   63.000 |

*Table: eda_churn_by_silent*

|   silent |   churn_rate |    count |
|---------:|-------------:|---------:|
|    0.000 |        0.116 | 3020.000 |
|    1.000 |        0.282 | 1193.000 |

Near-zero-variance features: ['silent_decline']

*Table: eda_corr_with_churn*

| feature             |   rho_churn |
|:--------------------|------------:|
| nps_score           |      -0.450 |
| balance_change_3m   |      -0.219 |
| avg_sentiment_180d  |      -0.174 |
| complaints_180d     |       0.154 |
| is_active_member    |      -0.141 |
| neg_msg_180d        |       0.140 |
| days_since_last_tx  |       0.129 |
| nps_missing         |       0.129 |
| age                 |       0.125 |
| silent_decline      |       0.109 |
| digital_share_90d   |      -0.103 |
| tx_trend            |      -0.098 |
| no_feedback_6m      |       0.086 |
| tx_count_90d        |      -0.069 |
| balance             |      -0.063 |
| num_products        |      -0.055 |
| days_since_last_msg |       0.036 |
| amount_out_90d      |      -0.021 |
| net_flow_90d        |      -0.018 |
| msg_count_180d      |      -0.018 |
| est_salary          |      -0.010 |
| amount_in_90d       |      -0.010 |
| tx_count_prev_90d   |       0.007 |
| has_card            |      -0.006 |
| credit_score        |      -0.004 |
| tenure_months       |      -0.002 |

*Table: eda_proxies*

| feature           |   spearman_with_age | note                                           |
|:------------------|--------------------:|:-----------------------------------------------|
| digital_share_90d |              -0.608 | possible PROXY for the sensitive attribute age |

*Table: eda_outliers*

| feature             |   outliers_3IQR |   outliers_z4 |   churn_rate_among_outliers | decision                                               |
|:--------------------|----------------:|--------------:|----------------------------:|:-------------------------------------------------------|
| avg_sentiment_180d  |            1457 |             0 |                       0.163 | keep; robust models + scaling                          |
| neg_msg_180d        |             818 |            47 |                       0.249 | KEEP: outliers concentrate leavers = signal, not error |
| complaints_180d     |             615 |            53 |                       0.286 | KEEP: outliers concentrate leavers = signal, not error |
| balance             |             131 |            52 |                       0.122 | keep; robust models + scaling                          |
| tx_trend            |             123 |            54 |                       0.138 | keep; robust models + scaling                          |
| days_since_last_tx  |             106 |            33 |                       0.472 | KEEP: outliers concentrate leavers = signal, not error |
| msg_count_180d      |              83 |            43 |                       0.723 | KEEP: outliers concentrate leavers = signal, not error |
| balance_change_3m   |              63 |            36 |                       0.730 | KEEP: outliers concentrate leavers = signal, not error |
| amount_out_90d      |              37 |            20 |                       0.703 | KEEP: outliers concentrate leavers = signal, not error |
| est_salary          |              31 |            26 |                       0.258 | KEEP: outliers concentrate leavers = signal, not error |
| tx_count_prev_90d   |              14 |            14 |                       0.214 | keep; robust models + scaling                          |
| tx_count_90d        |              14 |            20 |                       0.143 | keep; robust models + scaling                          |
| net_flow_90d        |               6 |            18 |                       0.667 | KEEP: outliers concentrate leavers = signal, not error |
| amount_in_90d       |               5 |            19 |                       0.600 | KEEP: outliers concentrate leavers = signal, not error |
| credit_score        |               0 |             0 |                     nan     | keep; robust models + scaling                          |
| tenure_months       |               0 |             0 |                     nan     | keep; robust models + scaling                          |
| age                 |               0 |             0 |                     nan     | keep; robust models + scaling                          |
| num_products        |               0 |             0 |                     nan     | keep; robust models + scaling                          |
| digital_share_90d   |               0 |             0 |                     nan     | keep; robust models + scaling                          |
| days_since_last_msg |               0 |             0 |                     nan     | keep; robust models + scaling                          |
| nps_score           |               0 |             0 |                     nan     | keep; robust models + scaling                          |

*Table: eda_kpis_hypotheses*

| KPI / hypothesis                             | value                                            |
|:---------------------------------------------|:-------------------------------------------------|
| B1 churn rate (90d)                          | 0.16330405886541657                              |
| B2 share of silent leavers                   | 0.4898255813953488                               |
| share of silent clients overall              | 0.2831711369570377                               |
| H1 churn rate silent vs not                  | 0.282 vs 0.116 (chi2 p=3.01e-39) -> H1 SUPPORTED |
| tx_trend leavers vs stayers (Mann-Whitney p) | 2.06e-10                                         |


## 2. Leakage trap (wrong cutoff)

*Table: leakage_demo*

| index                  |   roc_auc |   recall | top_feature       |   top_feature_share | top3                                      |
|:-----------------------|----------:|---------:|:------------------|--------------------:|:------------------------------------------|
| correct cutoff         |     0.893 |    0.754 | balance_change_3m |               0.195 | balance_change_3m, nps_score, nps_missing |
| WRONG cutoff (leakage) |     1.000 |    1.000 | balance           |               1.000 | balance, age, tenure_months               |


## 3.1 Contract matrix: cost matrix

*Table: cost_matrix*

| index           | Predicted: stays     | Predicted: leaves         |
|:----------------|:---------------------|:--------------------------|
| Actually stays  | 0 €                  | 60 € (useless offer)      |
| Actually leaves | 1000 € (lost client) | 60 € + 65%·1000 € = 710 € |


## 3.5 / 8 The loop: iterations

*Table: iterations_side_by_side*

| item                                                     | #1 iteration 1                                                            | #2 iteration 2                                                                                                                                                                                                                              | #3 iteration 3                                                                                                                                                          |
|:---------------------------------------------------------|:--------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| reason for change                                        | baseline: all features incl. sensitive attributes, cost-optimal threshold | iteration 1 failed ['Fairness: max recall gap (age/gender/country)', 'F2c max PSI (drift) external']: removed age/gender/country (non-discrimination) and digital_share_90d (proxy for age AND the feature drifting on the external branch) | iteration 2 failed ['M1 recall of leavers']: one threshold for all clients = the cheapest one whose out-of-fold recall is at least 0.74 (contract 0.70 + safety margin) |
| model                                                    | gb                                                                        | gb                                                                                                                                                                                                                                          | gb                                                                                                                                                                      |
| balancing                                                | none                                                                      | none                                                                                                                                                                                                                                        | none                                                                                                                                                                    |
| n features                                               | 28                                                                        | 24                                                                                                                                                                                                                                          | 24                                                                                                                                                                      |
| sensitive attributes used                                | age, gender, country                                                      | none                                                                                                                                                                                                                                        | none                                                                                                                                                                    |
| per-group thresholds                                     | False                                                                     | False                                                                                                                                                                                                                                       | False                                                                                                                                                                   |
| threshold                                                | 0.19                                                                      | 0.2                                                                                                                                                                                                                                         | 0.15                                                                                                                                                                    |
| recall                                                   | 0.703                                                                     | 0.638                                                                                                                                                                                                                                       | 0.725                                                                                                                                                                   |
| precision                                                | 0.664                                                                     | 0.672                                                                                                                                                                                                                                       | 0.535                                                                                                                                                                   |
| roc_auc                                                  | 0.893                                                                     | 0.888                                                                                                                                                                                                                                       | 0.888                                                                                                                                                                   |
| expected_cost                                            | 112810.0                                                                  | 115060.0                                                                                                                                                                                                                                    | 114220.0                                                                                                                                                                |
| recall gap age_group                                     | 0.136                                                                     | 0.022                                                                                                                                                                                                                                       | 0.016                                                                                                                                                                   |
| recall gap country                                       | 0.036                                                                     | 0.057                                                                                                                                                                                                                                       | 0.08                                                                                                                                                                    |
| recall gap gender                                        | 0.029                                                                     | 0.049                                                                                                                                                                                                                                       | 0.016                                                                                                                                                                   |
| Jaccard top-5 (F1)                                       | 0.722                                                                     | 1.0                                                                                                                                                                                                                                         | 1.0                                                                                                                                                                     |
| external AUC                                             | 0.904                                                                     | 0.906                                                                                                                                                                                                                                       | 0.906                                                                                                                                                                   |
| external SHAP rank corr (F2)                             | 0.993                                                                     | 0.999                                                                                                                                                                                                                                       | 0.999                                                                                                                                                                   |
| SHAP top-5                                               | balance_change_3m, nps_score, nps_missing, num_products, age              | balance_change_3m, nps_score, nps_missing, num_products, days_since_last_msg                                                                                                                                                                | balance_change_3m, nps_score, nps_missing, num_products, days_since_last_msg                                                                                            |
| [contract] M1 recall of leavers                          | PASS                                                                      | FAIL                                                                                                                                                                                                                                        | PASS                                                                                                                                                                    |
| [contract] M2a ROC-AUC                                   | PASS                                                                      | PASS                                                                                                                                                                                                                                        | PASS                                                                                                                                                                    |
| [contract] M2b expected cost vs no model (EUR)           | PASS                                                                      | PASS                                                                                                                                                                                                                                        | PASS                                                                                                                                                                    |
| [contract] Fairness: max recall gap (age/gender/country) | FAIL                                                                      | PASS                                                                                                                                                                                                                                        | PASS                                                                                                                                                                    |
| [contract] Leakage alarm: AUC suspiciously high          | PASS                                                                      | PASS                                                                                                                                                                                                                                        | PASS                                                                                                                                                                    |
| [contract] F1 XAI stability (Jaccard top-5)              | PASS                                                                      | PASS                                                                                                                                                                                                                                        | PASS                                                                                                                                                                    |
| [contract] F2a external SHAP rank correlation            | PASS                                                                      | PASS                                                                                                                                                                                                                                        | PASS                                                                                                                                                                    |
| [contract] F2b external ROC-AUC                          | PASS                                                                      | PASS                                                                                                                                                                                                                                        | PASS                                                                                                                                                                    |
| [contract] F2c max PSI (drift) external                  | FAIL                                                                      | PASS                                                                                                                                                                                                                                        | PASS                                                                                                                                                                    |
| [contract] Data contract (input validity)                | PASS                                                                      | PASS                                                                                                                                                                                                                                        | PASS                                                                                                                                                                    |
| VERDICT                                                  | FAIL                                                                      | FAIL                                                                                                                                                                                                                                        | PASS                                                                                                                                                                    |

Feature changes 1 -> 2: {'features removed': ['age', 'country', 'digital_share_90d', 'gender'], 'features added': [], 'entered SHAP top-5': ['days_since_last_msg'], 'left SHAP top-5': ['age']}

*Table: contract_matrix_iteration_1*

| requirement                                   | threshold      |   measured | pass   | verdict   |
|:----------------------------------------------|:---------------|-----------:|:-------|:----------|
| M1 recall of leavers                          | >= 0.70        |      0.703 | True   | PASS      |
| M2a ROC-AUC                                   | >= 0.80        |      0.893 | True   | PASS      |
| M2b expected cost vs no model (EUR)           | < 138000       | 112810.000 | True   | PASS      |
| Fairness: max recall gap (age/gender/country) | <= 0.10        |      0.136 | False  | FAIL      |
| Leakage alarm: AUC suspiciously high          | < 0.97         |      0.893 | True   | PASS      |
| F1 XAI stability (Jaccard top-5)              | >= 0.60        |      0.722 | True   | PASS      |
| F2a external SHAP rank correlation            | >= 0.80        |      0.993 | True   | PASS      |
| F2b external ROC-AUC                          | >= 0.80        |      0.904 | True   | PASS      |
| F2c max PSI (drift) external                  | < 0.20         |      0.433 | False  | FAIL      |
| Data contract (input validity)                | all rules pass |      1.000 | True   | PASS      |

*Table: contract_matrix_iteration_2*

| requirement                                   | threshold      |   measured | pass   | verdict   |
|:----------------------------------------------|:---------------|-----------:|:-------|:----------|
| M1 recall of leavers                          | >= 0.70        |      0.638 | False  | FAIL      |
| M2a ROC-AUC                                   | >= 0.80        |      0.888 | True   | PASS      |
| M2b expected cost vs no model (EUR)           | < 138000       | 115060.000 | True   | PASS      |
| Fairness: max recall gap (age/gender/country) | <= 0.10        |      0.057 | True   | PASS      |
| Leakage alarm: AUC suspiciously high          | < 0.97         |      0.888 | True   | PASS      |
| F1 XAI stability (Jaccard top-5)              | >= 0.60        |      1.000 | True   | PASS      |
| F2a external SHAP rank correlation            | >= 0.80        |      0.999 | True   | PASS      |
| F2b external ROC-AUC                          | >= 0.80        |      0.906 | True   | PASS      |
| F2c max PSI (drift) external                  | < 0.20         |      0.023 | True   | PASS      |
| Data contract (input validity)                | all rules pass |      1.000 | True   | PASS      |

*Table: contract_matrix_iteration_3*

| requirement                                   | threshold      |   measured | pass   | verdict   |
|:----------------------------------------------|:---------------|-----------:|:-------|:----------|
| M1 recall of leavers                          | >= 0.70        |      0.725 | True   | PASS      |
| M2a ROC-AUC                                   | >= 0.80        |      0.888 | True   | PASS      |
| M2b expected cost vs no model (EUR)           | < 138000       | 114220.000 | True   | PASS      |
| Fairness: max recall gap (age/gender/country) | <= 0.10        |      0.080 | True   | PASS      |
| Leakage alarm: AUC suspiciously high          | < 0.97         |      0.888 | True   | PASS      |
| F1 XAI stability (Jaccard top-5)              | >= 0.60        |      1.000 | True   | PASS      |
| F2a external SHAP rank correlation            | >= 0.80        |      0.999 | True   | PASS      |
| F2b external ROC-AUC                          | >= 0.80        |      0.906 | True   | PASS      |
| F2c max PSI (drift) external                  | < 0.20         |      0.023 | True   | PASS      |
| Data contract (input validity)                | all rules pass |      1.000 | True   | PASS      |

Champion = iteration 3, verdict **PASS**


## 3.2-3.3 Models compared, tuning, out-of-time

*Table: models_comparison*

| model   |   roc_auc |   pr_auc |   accuracy |   precision |   recall |    f1 |   lift_top10 |   threshold |   expected_cost |   cv_auc |   fit_s |   latency_ms |
|:--------|----------:|---------:|-----------:|------------:|---------:|------:|-------------:|------------:|----------------:|---------:|--------:|-------------:|
| logreg  |     0.865 |    0.712 |      0.816 |       0.460 |    0.710 | 0.558 |        5.091 |       0.170 |      116480.000 |    0.879 |   0.026 |        6.462 |
| rf      |     0.880 |    0.747 |      0.859 |       0.558 |    0.667 | 0.607 |        5.454 |       0.160 |      115700.000 |    0.896 |   0.902 |       62.465 |
| gb      |     0.893 |    0.770 |      0.855 |       0.542 |    0.754 | 0.630 |        5.381 |       0.150 |      113120.000 |    0.898 |   0.202 |        7.639 |
| mlp     |     0.863 |    0.716 |      0.811 |       0.451 |    0.703 | 0.550 |        5.163 |       0.180 |      116950.000 |    0.878 |   0.176 |        6.581 |

Hyperparameter tuning (CV on training only): best {'clf__max_iter': 300, 'clf__max_depth': 3, 'clf__learning_rate': 0.02, 'clf__l2_regularization': 2.0}, CV AUC 0.902

*Table: out_of_time*

| setup                                            |   roc_auc |   recall |   precision |
|:-------------------------------------------------|----------:|---------:|------------:|
| random split (main cutoff)                       |     0.888 |    0.725 |       0.535 |
| out-of-time: train 2025-09-30 -> test 2025-12-31 |     0.878 |    0.651 |       0.713 |

*Table: complexity_curve*

| model   |   max_depth |   train_auc |   valid_auc |
|:--------|------------:|------------:|------------:|
| rf      |           1 |       0.884 |       0.869 |
| rf      |           2 |       0.903 |       0.886 |
| rf      |           3 |       0.912 |       0.893 |
| rf      |           4 |       0.923 |       0.894 |
| rf      |           6 |       0.948 |       0.898 |
| rf      |           8 |       0.971 |       0.899 |
| rf      |          12 |       0.998 |       0.898 |
| rf      |          16 |       1.000 |       0.890 |


## 4. XAI pass 1 (our data) and pass 2 (external branch)

*Table: shap_global*

| feature             |   mean_abs_shap |
|:--------------------|----------------:|
| balance_change_3m   |           0.731 |
| nps_score           |           0.691 |
| nps_missing         |           0.401 |
| num_products        |           0.247 |
| days_since_last_msg |           0.213 |
| neg_msg_180d        |           0.182 |
| is_active_member    |           0.132 |
| balance             |           0.121 |
| tx_count_prev_90d   |           0.116 |
| avg_sentiment_180d  |           0.097 |
| days_since_last_tx  |           0.095 |
| tx_trend            |           0.082 |
| amount_in_90d       |           0.071 |
| tenure_months       |           0.069 |
| est_salary          |           0.064 |
| tx_count_90d        |           0.040 |
| amount_out_90d      |           0.033 |
| complaints_180d     |           0.025 |
| credit_score        |           0.024 |
| net_flow_90d        |           0.021 |
| msg_count_180d      |           0.011 |
| silent_decline      |           0.004 |
| has_card            |           0.001 |
| no_feedback_6m      |           0.000 |

**silent leaver** (client 629, risk 1.00) — reason code: **money_withdrawal**; SHAP–LIME agreement: {'rank_corr': 0.19047619047619052, 'sign_agreement': 1.0, 'n': 8}

*Table: shap_local_silent_leaver*

| feature             |   shap |
|:--------------------|-------:|
| balance_change_3m   |  5.876 |
| tx_count_prev_90d   |  0.419 |
| is_active_member    |  0.362 |
| days_since_last_tx  |  0.310 |
| days_since_last_msg |  0.310 |
| balance             |  0.235 |
| tx_trend            |  0.229 |
| nps_score           |  0.223 |

*Table: lime_silent_leaver*

| condition                    |   weight | feature             |
|:-----------------------------|---------:|:--------------------|
| balance_change_3m <= -0.01   |    0.484 | balance_change_3m   |
| nps_score <= 7.00            |    0.116 | nps_score           |
| neg_msg_180d <= 0.00         |   -0.097 | neg_msg_180d        |
| 0.00 < nps_missing <= 1.00   |    0.048 | nps_missing         |
| silent_decline <= 0.00       |   -0.044 | silent_decline      |
| days_since_last_msg > 334.75 |    0.036 | days_since_last_msg |
| days_since_last_tx > 7.00    |    0.035 | days_since_last_tx  |
| num_products <= 1.00         |    0.025 | num_products        |

**vocal leaver** (client 4234, risk 1.00) — reason code: **money_withdrawal**; SHAP–LIME agreement: {'rank_corr': 0.5714285714285715, 'sign_agreement': 0.875, 'n': 8}

*Table: shap_local_vocal_leaver*

| feature           |   shap |
|:------------------|-------:|
| balance_change_3m |  4.704 |
| nps_score         |  2.560 |
| tenure_months     |  0.200 |
| num_products      |  0.186 |
| nps_missing       | -0.159 |
| balance           |  0.139 |
| is_active_member  |  0.126 |
| tx_count_prev_90d |  0.107 |

*Table: lime_vocal_leaver*

| condition                  |   weight | feature            |
|:---------------------------|---------:|:-------------------|
| balance_change_3m <= -0.01 |    0.484 | balance_change_3m  |
| nps_score <= 7.00          |    0.117 | nps_score          |
| neg_msg_180d > 0.00        |    0.098 | neg_msg_180d       |
| nps_missing <= 0.00        |   -0.047 | nps_missing        |
| silent_decline <= 0.00     |   -0.042 | silent_decline     |
| days_since_last_tx > 7.00  |    0.036 | days_since_last_tx |
| num_products <= 1.00       |    0.025 | num_products       |
| avg_sentiment_180d <= 0.00 |    0.021 | avg_sentiment_180d |

Stability (KPI F1): mean Jaccard of top-5 = **1.000**; tops = [['balance_change_3m', 'nps_score', 'nps_missing', 'num_products', 'days_since_last_msg'], ['balance_change_3m', 'nps_missing', 'nps_score', 'days_since_last_msg', 'num_products'], ['nps_score', 'balance_change_3m', 'nps_missing', 'days_since_last_msg', 'num_products'], ['nps_score', 'balance_change_3m', 'nps_missing', 'days_since_last_msg', 'num_products']]

External branch (KPI F2): AUC 0.906, recall 0.766, SHAP rank correlation 0.999, top-5 external ['balance_change_3m', 'nps_score', 'nps_missing', 'num_products', 'days_since_last_msg']

*Table: psi_external*

| feature             |   psi | drift   |
|:--------------------|------:|:--------|
| balance             | 0.023 | low     |
| nps_score           | 0.018 | low     |
| amount_in_90d       | 0.015 | low     |
| tx_count_prev_90d   | 0.014 | low     |
| net_flow_90d        | 0.013 | low     |
| balance_change_3m   | 0.011 | low     |
| days_since_last_msg | 0.007 | low     |
| tx_count_90d        | 0.007 | low     |
| tx_trend            | 0.006 | low     |
| amount_out_90d      | 0.006 | low     |
| days_since_last_tx  | 0.005 | low     |
| credit_score        | 0.005 | low     |
| est_salary          | 0.004 | low     |
| tenure_months       | 0.004 | low     |
| has_card            | 0.003 | low     |
| nps_missing         | 0.001 | low     |
| is_active_member    | 0.001 | low     |
| complaints_180d     | 0.001 | low     |
| silent_decline      | 0.001 | low     |
| num_products        | 0.001 | low     |
| neg_msg_180d        | 0.000 | low     |
| avg_sentiment_180d  | 0.000 | low     |
| msg_count_180d      | 0.000 | low     |
| no_feedback_6m      | 0.000 | low     |


## 5. Six fundamentals

### Bias

**1. How it appears.** Clienții nu sunt tratați la fel pe grupuri de vârstă, gen și țară. Vârsta are și un proxy: ponderea operațiunilor digitale (digital_share_90d), puternic corelată cu vârsta. Un eșantion de antrenare în care un grup este subreprezentat (de exemplu 60+) produce un model care îl recunoaște mai slab.

**2. Measured.**

*Table: fund_bias*

| variant                                           |   roc_auc |   recall |   precision |   expected_cost |   gap_age_group |   gap_country |   gap_gender |
|:--------------------------------------------------|----------:|---------:|------------:|----------------:|----------------:|--------------:|-------------:|
| 1. all features (incl. age, gender, country)      |     0.893 |    0.761 |       0.479 |      114390.000 |           0.162 |         0.036 |        0.011 |
| 2. BIASED training sample (80% of 60+ removed)    |     0.887 |    0.732 |       0.546 |      113750.000 |           0.127 |         0.053 |        0.014 |
| 3. sensitive attributes removed                   |     0.896 |    0.725 |       0.529 |      114340.000 |           0.067 |         0.062 |        0.018 |
| 4. sensitive + proxy (digital_share) removed      |     0.888 |    0.725 |       0.535 |      114220.000 |           0.016 |         0.080 |        0.016 |
| 5. as 4 + per-age-group thresholds                |     0.888 |    0.725 |       0.543 |      114040.000 |           0.060 |         0.080 |        0.030 |
| 6. fairlearn ThresholdOptimizer (TPR parity, age) |     0.888 |    0.652 |       0.570 |      115980.000 |           0.213 |         0.237 |        0.124 |

**3. Observed.**
- Cu toate caracteristicile, diferența de recall între grupurile de vârstă este 0.162; după eliminarea atributelor sensibile și a proxy-ului scade la 0.016, cu AUC 0.893 → 0.888.
- Eșantionul biasat (60+ subreprezentat) dă diferența 0.127 și recall 0.732: AUC-ul (0.887) nu arată problema.
- Pragurile pe grup de vârstă schimbă recall-ul 0.725 → 0.725 și diferența de vârstă 0.016 → 0.060, deci nu aduc un câștig de corectitudine: după eliminarea proxy-ului grupurile sunt deja apropiate, iar un prag unic evită o decizie care depinde de vârstă.

**4. Mitigation.**
- S-a eliminat vârsta, genul și țara din modelul de decizie, deoarece acestea sunt atribute sensibile și decizia de retenție nu trebuie să depindă de ele (GDPR, nediscriminare).
- S-a verificat proxy-ul digital_share_90d (corelat cu vârsta), deoarece eliminarea atributului sensibil nu ajută dacă o altă variabilă îl înlocuiește.
- S-a păstrat un singur prag pentru toți clienții (fără praguri pe grup de vârstă), deoarece după eliminarea proxy-ului diferența de recall era deja sub limita din matricea-contract, iar o decizie care depinde de vârstă ar contrazice regula de nediscriminare; recall-ul țintă se atinge prin alegerea pragului unic.

**5. Răspunsuri.**
- *Ce tip de bias ați observat în datele inițiale?* Bias de reprezentare (grupul 60+ subreprezentat în varianta 2), bias prin atribute sensibile (vârstă, gen, țară) și bias prin proxy (digital_share_90d înlocuiește vârsta după eliminarea ei).
- *Cum a afectat bias-ul performanța modelului?* AUC-ul global se schimbă puțin, dar recall-ul diferă între grupurile de vârstă: un AUC bun poate ascunde un model nedrept. Diferența de recall este metrica relevantă, nu acuratețea.
- *Ce diferențe apar între modelul biasat și cel corectat?* Modelul corectat (varianta 4: fără atribute sensibile și fără proxy) are diferența de recall pe vârstă mult mai mică, cu un AUC practic egal; recall-ul pierdut se recuperează prin alegerea pragului unic (iterația 3), nu prin praguri diferite pe grupuri.
- *Ce metode alternative de corectare ați putea explora?* Reponderarea eșantioanelor, ThresholdOptimizer din fairlearn (varianta 6), constrângeri în antrenare (ExponentiatedGradient), colectarea de date suplimentare pentru grupul subreprezentat.
- *Ce diferențe apar în metricile de fairness? Selection rate și FPR?* Selection rate arată cui i se trimit oferte; FPR arată cine este deranjat fără motiv. Un grup cu FPR mare primește mai multe contactări inutile, unul cu recall mic este ignorat când pleacă.
- *Ce alte caracteristici ar putea fi sensibile?* Orice variabilă corelată cu vârsta, genul sau țara: canalul digital, tipul de card, produsele deținute, limba mesajelor.
- *Legătura cu corectitudinea și responsabilitatea în IA; integrarea în dezvoltare* Fairness este un rând în matricea-contract (diferența maximă de recall ≤ 0,10), verificat la fiecare iterație ca orice altă cerință; decizia automată este doar o propunere (GDPR art. 22).
- *Ce este mai rău, bias-ul sau noise-ul? Pot fi „bune”?* Bias-ul este mai periculos: este sistematic și nu dispare cu mai multe date, pe când noise-ul se mai mediază. Noise-ul controlat poate fi util (augmentare, regularizare), iar un bias asumat (de exemplu prudența față de pierderea unui client) poate fi o alegere de business documentată.

### Noise

**1. How it appears.** Feedback-ul textual are greșeli de scriere, sarcasm, mesaje contradictorii, în altă limbă sau goale; valorile numerice au erori de măsurare, iar eticheta poate fi greșită (data ieșirii înregistrată târziu). Comportamentul zilnic al clienților variază natural.

**2. Measured.**

*Table: fund_noise*

| noise                                |   level |   auc_mean |   auc_std |   f1_mean |
|:-------------------------------------|--------:|-----------:|----------:|----------:|
| X: gaussian noise (sigma, std units) |   0.000 |      0.889 |     0.001 |     0.577 |
| X: gaussian noise (sigma, std units) |   0.250 |      0.888 |     0.010 |     0.563 |
| X: gaussian noise (sigma, std units) |   0.500 |      0.889 |     0.001 |     0.578 |
| X: gaussian noise (sigma, std units) |   1.000 |      0.868 |     0.015 |     0.512 |
| X: gaussian noise (sigma, std units) |   2.000 |      0.820 |     0.011 |     0.467 |
| Y: labels flipped (share)            |   0.000 |      0.889 |     0.001 |     0.577 |
| Y: labels flipped (share)            |   0.050 |      0.866 |     0.017 |     0.467 |
| Y: labels flipped (share)            |   0.100 |      0.861 |     0.011 |     0.390 |
| Y: labels flipped (share)            |   0.200 |      0.844 |     0.007 |     0.285 |
| Y: labels flipped (share)            |   0.300 |      0.778 |     0.008 |     0.281 |

*Table: fund_noise_text_table*

| noise_type          |         n |   accuracy_raw |   accuracy_denoised |
|:--------------------|----------:|---------------:|--------------------:|
| ALL                 | 12311.000 |          0.899 |               0.994 |
| clean               |  8409.000 |          1.000 |               1.000 |
| typos               |  3101.000 |          0.809 |               0.975 |
| sarcasm             |   303.000 |          0.000 |               1.000 |
| contradictory       |   196.000 |          0.638 |               1.000 |
| sarcasm+typos       |   108.000 |          0.056 |               1.000 |
| english             |    83.000 |          0.000 |               1.000 |
| contradictory+typos |    77.000 |          0.286 |               1.000 |
| english+typos       |    34.000 |          0.000 |               1.000 |

*Table: fund_noise_text_model*

| variant                      |   roc_auc |   recall |   precision |   expected_cost |   gap_age_group |   gap_country |   gap_gender |
|:-----------------------------|----------:|---------:|------------:|----------------:|----------------:|--------------:|-------------:|
| sentiment from DENOISED text |     0.888 |    0.725 |       0.535 |      114220.000 |           0.016 |         0.080 |        0.016 |
| sentiment from RAW text      |     0.893 |    0.696 |       0.568 |      114540.000 |           0.022 |         0.043 |        0.045 |

**3. Observed.**
- Zgomot pe X (σ până la 2): AUC 0.889 → 0.820, F1 0.577 → 0.467.
- Etichete greșite pe Y (până la 30%): AUC 0.889 → 0.778, F1 0.577 → 0.281: zgomotul pe etichetă strică mai mult decât cel pe intrări.
- Sentimentul din text: acuratețe 0.899 pe textul brut, 0.994 după denoising; sarcasmul și mesajele în engleză erau citite greșit aproape complet.

**4. Mitigation.**
- S-a efectuat denoising-ul textului (normalizare, diacritice, corectarea greșelilor, tratarea sarcasmului și a construcțiilor cu „dar”), deoarece feedback-ul brut inversa sau anula sentimentul.
- S-au tratat mesajele goale/corupte ca „fără conținut” (is_empty), deoarece un mesaj gol nu este un mesaj pozitiv; contactul în sine rămâne numărat.
- S-a folosit un model robust (boosting cu early stopping, regularizare L2) și netezirea alertelor cu EWMA în streaming, deoarece zgomotul numeric și zilnic produce alarme false.
- S-a verificat definiția etichetei (data ieșirii, fereastra de 90 de zile), deoarece zgomotul pe Y afectează modelul mult mai mult decât zgomotul pe X.

**5. Răspunsuri.**
- *Ce model „înțelege” mai bine forma reală și de ce?* Boosting-ul și pădurea aleatoare urmăresc relații neliniare (de exemplu tăcerea combinată cu scăderea soldului); regresia logistică vede doar direcții liniare, dar este mai stabilă la zgomot.
- *Cum reacționează fiecare model la noise? Unde se pierde semnalul și unde se păstrează?* Zgomotul pe X scade AUC-ul treptat; zgomotul pe etichete (Y) scade F1 mult mai repede, pentru că modelul învață direct exemple greșite. Semnalul se păstrează în tendințe agregate (90 de zile), se pierde în valorile zilnice izolate și în textul brut.
- *Ce înseamnă „adevăr” într-un model care învață din zgomot? Ce se pierde, ce se păstrează?* Adevărul este tiparul stabil care rămâne după mediere și curățare. Se pierd cazurile rare și nuanțele (sarcasmul fără denoising), se păstrează semnalele repetate la mulți clienți.

### SMOTE

**1. How it appears.** Clienții care pleacă sunt clasa minoritară (aproximativ 16%). Fără echilibrare modelul învață mai ales clasa majoritară; SMOTE creează clienți sintetici prin interpolare între clienți reali care pleacă.

**2. Measured.**

*Table: fund_smote*

| variant                        |   cv_auc |   roc_auc |   pr_auc |   recall |   precision |      f1 |   expected_cost |   threshold |
|:-------------------------------|---------:|----------:|---------:|---------:|------------:|--------:|----------------:|------------:|
| none                           |    0.899 |     0.893 |    0.770 |    0.761 |       0.479 |   0.588 |      114390.000 |       0.130 |
| class_weight                   |    0.898 |     0.902 |    0.777 |    0.754 |       0.523 |   0.617 |      113540.000 |       0.420 |
| ros                            |    0.893 |     0.888 |    0.757 |    0.732 |       0.495 |   0.591 |      114890.000 |       0.350 |
| smote                          |    0.899 |     0.897 |    0.763 |    0.710 |       0.573 |   0.634 |      113960.000 |       0.230 |
| smotenc                        |    0.898 |     0.901 |    0.770 |    0.768 |       0.535 |   0.631 |      112780.000 |       0.190 |
| SMOTE BEFORE the split (WRONG) |    0.984 |   nan     |  nan     |  nan     |     nan     | nan     |         nan     |     nan     |

*Table: fund_smote_realism*

| check                                                             |   value |
|:------------------------------------------------------------------|--------:|
| num_products: share of synthetic values that are not integers     |   0.579 |
| has_card: share of synthetic values that are not integers         |   0.419 |
| is_active_member: share of synthetic values that are not integers |   0.411 |
| no_feedback_6m: share of synthetic values that are not integers   |   0.467 |
| nps_missing: share of synthetic values that are not integers      |   0.385 |
| silent_decline: share of synthetic values that are not integers   |   0.057 |
| complaints_180d: share of synthetic values that are not integers  |   0.442 |
| tx_count_90d: share of synthetic values that are not integers     |   0.968 |
| median distance synthetic -> nearest real leaver (std units)      |   1.821 |
| median distance real leaver -> nearest other real leaver          |   3.758 |

*Table: fund_smote_subset_noise*

| variant                        |   roc_auc |   recall |   precision |    f1 |
|:-------------------------------|----------:|---------:|------------:|------:|
| SMOTE, all features            |     0.897 |    0.746 |       0.545 | 0.630 |
| SMOTE, 10 selected features    |     0.899 |    0.761 |       0.507 | 0.609 |
| SMOTE + label noise 10%        |     0.834 |    0.891 |       0.239 | 0.377 |
| no balancing + label noise 10% |     0.873 |    0.993 |       0.206 | 0.341 |

**3. Observed.**
- Fără echilibrare: recall 0.761, precizie 0.479; varianta cu costul cel mai mic este „smotenc” (recall 0.768, precizie 0.535, cost 112,780 €).
- SMOTE aplicat înainte de împărțire raportează un AUC de validare de 0.984 față de 0.898 aplicat corect: un rezultat fals, din scurgerea de date.
- Cu 10% etichete greșite, SMOTE scade precizia la 0.239: multiplică exemplele greșite.

**4. Mitigation.**
- S-au comparat variantele de echilibrare, deoarece clienții care pleacă sunt minoritari (~16%); „smotenc” are costul așteptat minim (112,780 €), iar fără echilibrare costul este 114,390 €. Diferența este mică, deoarece pragul ales din matricea de cost compensează deja dezechilibrul; modelul final rămâne fără echilibrare (mai simplu, probabilități nedistorsionate), cu SMOTE-NC ca alternativă documentată.
- S-a aplicat SMOTE numai în interiorul fold-urilor de antrenare (imblearn Pipeline), deoarece SMOTE înainte de împărțire copiază clienți din test în antrenare și umflă artificial scorul.
- S-a folosit SMOTE-NC pentru variabilele categoriale, deoarece SMOTE simplu interpolează coduri de țară/gen fără sens; s-a verificat realismul clienților sintetici (valori ne-întregi, distanța față de clienții reali).

**5. Răspunsuri.**
- *SMOTE aplicat pe un subset de features păstrează performanța? De ce?* Da, aproape la fel (tabelul subset): caracteristicile irelevante nu adaugă semnal, iar interpolarea pe mai puține dimensiuni creează clienți sintetici mai plauzibili.
- *Noise-ul afectează eficiența SMOTE? Cum?* Da: cu 10% etichete greșite SMOTE multiplică exact exemplele greșite, iar precizia scade puternic.
- *În ce scenarii este SMOTE recomandat și în care nu?* Recomandat când minoritatea este mică și cu caracteristici numerice continue; nu este recomandat cu etichete zgomotoase, cu variabile categoriale fără SMOTE-NC sau când ponderile claselor obțin același rezultat.
- *Cum validăm că modelul nu învață pe artefacte sintetice?* SMOTE numai în fold-urile de antrenare, evaluare doar pe clienți reali, verificarea realismului (valori ne-întregi, distanța față de clienții reali); varianta greșită (SMOTE înainte de împărțire) umflă AUC-ul din validare.
- *Riscuri cu SMOTE pe toate features și cum alegem subsetul? Ce metrici sunt sensibile?* Pe toate caracteristicile se interpolează și zgomot sau coduri fără sens; subsetul se alege după importanța SHAP și cunoașterea domeniului. Cele mai sensibile sunt precizia și F1, AUC-ul se schimbă puțin.
- *SMOTE înainte de noise; intensitatea noise-ului* Zgomotul adăugat după SMOTE afectează și clienții sintetici, deci efectul se dublează; intensitatea se crește treptat (0,05 … 0,3) și se oprește unde F1 scade vizibil.
- *Augmentări necontrolate, optimizări obsesive; legătura cu under/overfitting* Augmentarea necontrolată inventează clienți nerealiști (overfitting pe artefacte); optimizarea obsesivă a unei metrici pe aceleași date duce la overfitting pe setul de validare. Augmentarea poate fi și pe caracteristici (zgomot controlat, variante de text).

### Decision boundaries

**1. How it appears.** Granița dintre „pleacă” și „rămâne” este o alegere de business: un client pierdut costă 1000 €, o ofertă inutilă 60 €. Pragul 0,5 ignoră această asimetrie; modele diferite trasează granițe diferite.

**2. Measured.**

*Table: fund_boundaries*

| model   |   roc_auc |   pr_auc |   accuracy |   precision |   recall |    f1 |   lift_top10 |   threshold |   expected_cost |   cv_auc |   fit_s |   latency_ms |
|:--------|----------:|---------:|-----------:|------------:|---------:|------:|-------------:|------------:|----------------:|---------:|--------:|-------------:|
| logreg  |     0.865 |    0.712 |      0.816 |       0.460 |    0.710 | 0.558 |        5.091 |       0.170 |      116480.000 |    0.879 |   0.026 |        6.462 |
| rf      |     0.880 |    0.747 |      0.859 |       0.558 |    0.667 | 0.607 |        5.454 |       0.160 |      115700.000 |    0.896 |   0.902 |       62.465 |
| gb      |     0.893 |    0.770 |      0.855 |       0.542 |    0.754 | 0.630 |        5.381 |       0.150 |      113120.000 |    0.898 |   0.202 |        7.639 |
| mlp     |     0.863 |    0.716 |      0.811 |       0.451 |    0.703 | 0.550 |        5.163 |       0.180 |      116950.000 |    0.878 |   0.176 |        6.581 |

*Table: fund_boundaries_confusion*

|   threshold |      TP |      FP |     FN |      TN |   recall |   precision |       cost |
|------------:|--------:|--------:|-------:|--------:|---------:|------------:|-----------:|
|       0.100 | 116.000 | 178.000 | 22.000 | 527.000 |    0.841 |       0.395 | 115040.000 |
|       0.150 | 104.000 |  88.000 | 34.000 | 617.000 |    0.754 |       0.542 | 113120.000 |
|       0.200 |  94.000 |  45.000 | 44.000 | 660.000 |    0.681 |       0.676 | 113440.000 |
|       0.500 |  70.000 |   8.000 | 68.000 | 697.000 |    0.507 |       0.897 | 118180.000 |

*Table: fund_boundaries_deciles*

|   risk_decile |   clients |   leavers |   churn_rate |   min_score |
|--------------:|----------:|----------:|-------------:|------------:|
|         1.000 |    85.000 |    74.000 |        0.871 |       0.413 |
|         2.000 |    84.000 |    27.000 |        0.321 |       0.168 |
|         3.000 |    84.000 |     7.000 |        0.083 |       0.117 |
|         4.000 |    84.000 |    11.000 |        0.131 |       0.087 |
|         5.000 |    84.000 |     7.000 |        0.083 |       0.068 |
|         6.000 |    85.000 |     5.000 |        0.059 |       0.050 |
|         7.000 |    84.000 |     2.000 |        0.024 |       0.035 |
|         8.000 |    84.000 |     3.000 |        0.036 |       0.022 |
|         9.000 |    84.000 |     2.000 |        0.024 |       0.005 |
|        10.000 |    85.000 |     0.000 |        0.000 |       0.001 |

**3. Observed.**
- Cel mai mic cost așteptat îl are gb (AUC 0.893, recall 0.754 la pragul 0.15); diferențele de AUC dintre modele sunt mici.
- La pragul 0.50 recall-ul este 0.507, la pragul de cost 0.15 este 0.754, cu costul 118,180 € → 113,120 €.
- Decila de risc de sus are o rată de plecare de 87%, cea de jos 0%: scorul ordonează bine clienții.

**4. Mitigation.**
- S-a ales pragul de decizie din matricea de cost (minimul costului așteptat pe predicții out-of-fold), deoarece un client pierdut costă mult mai mult decât o ofertă inutilă; pragul 0.5 ignoră această asimetrie.
- S-a segmentat scorul în decile de risc, deoarece echipa de retenție are capacitate limitată și contactează întâi decilele de sus.
- S-a ales adâncimea modelului din curba de complexitate (validare), deoarece peste acest punct modelul învață zgomotul (overfitting), iar sub el ignoră tipare reale (underfitting).

**5. Răspunsuri.**
- *Ce diferențe observați între frontiere? Ce a învățat fiecare model?* Regresia logistică trasează o graniță dreaptă (o direcție de risc), arborii trasează trepte, boosting-ul o graniță flexibilă, rețeaua neuronală o curbă netedă.
- *Cum ați trata incertitudinea dacă ați fi un algoritm?* Clienții de lângă prag sunt incerți: îi trimit omului (decila 2–3) în loc să decid automat, iar pragul îl aleg după cost.
- *Ce observați când creșteți zgomotul? Cum se schimbă decizia?* Granițele devin neregulate și instabile; modelele flexibile urmăresc zgomotul (overfitting), modelul liniar se schimbă puțin.
- *„A învăța prea puțin” sau „prea mult”? Prudent sau atent la tot?* Curba de complexitate arată ambele: adâncime mică = underfitting, adâncime mare = AUC de antrenare mare și AUC de validare în scădere. Pentru retenție este preferabil modelul ușor prudent, cu prag ales din cost.
- *Ce am învățat? Ce întrebări am? Ce aleg ca model?* Pragul contează mai mult decât diferențele mici dintre modele. Rămâne întrebarea cât de stabil este costul unei oferte. Aleg gradient boosting cu prag din matricea de cost.

### Streaming

**1. How it appears.** Semnalele de plecare apar în timp (retrageri, reclamații, tăcere). Un scor lunar observă prea târziu clienții al căror risc crește între două rulări; tăcerea nu trimite evenimente.

**2. Measured.**

*Table: fund_streaming*

| mode                                                                         |   alerts |   leavers_in_window |   caught_before_exit |   recall_before_exit |   median_lead_days |   share_lead>=30d |   false_alarms (never leave) |   precision |
|:-----------------------------------------------------------------------------|---------:|--------------------:|---------------------:|---------------------:|-------------------:|------------------:|-----------------------------:|------------:|
| streaming daily + EWMA                                                       |     1645 |                 688 |                  611 |                0.888 |             39.000 |             0.549 |                          807 |       0.509 |
| streaming daily, no smoothing                                                |     1812 |                 688 |                  612 |                0.890 |             39.000 |             0.555 |                          969 |       0.465 |
| monthly batch (same daily capacity)                                          |     1460 |                 688 |                  607 |                0.882 |             40.000 |             0.555 |                          666 |       0.544 |
| streaming daily + EWMA | only leavers NOT flagged at the cutoff              |     1645 |                 104 |                   87 |                0.837 |             44.000 |             0.596 |                          807 |       0.509 |
| monthly batch (same daily capacity) | only leavers NOT flagged at the cutoff |     1460 |                 104 |                   76 |                0.731 |             41.500 |             0.519 |                          666 |       0.544 |

**3. Observed.**
- Pe toți cei care pleacă, streaming și batch sunt apropiate (recall înainte de ieșire 0.888 vs 0.882).
- Diferența apare la clienții al căror risc crește după data de referință: 0.837 cu streaming vs 0.731 cu batch lunar.
- EWMA reduce alarmele false de la 969 la 807 fără să piardă clienți prinși.

**4. Mitigation.**
- S-a introdus calea „ceas” (rescorarea tuturor clienților activi la închiderea fiecărei zile bancare), deoarece tăcerea nu trimite evenimente: fără ea, un client tăcut nu ar fi rescorat niciodată.
- S-a efectuat netezirea alertelor cu EWMA, deoarece scorurile zilnice sunt zgomotoase și ar genera alarme false și oferte inutile.
- S-a limitat numărul de alerte la capacitatea zilnică a echipei, cel mai mare risc primul, deoarece o alertă care nu poate fi tratată nu are valoare.

**5. Răspunsuri.**
- *De ce în timp real și nu lunar?* Riscul crește între două rulări lunare; scorarea zilnică (evenimente + ceasul de închidere a zilei) prinde mai devreme clienții al căror risc apare după data de referință.
- *Cum tratăm tăcerea și alarmele false?* Calea „ceas” rescorează zilnic și clienții fără evenimente; EWMA netezește scorul ca o singură zi zgomotoasă să nu genereze o ofertă.

### RTAP

**1. How it appears.** O alertă nu valorează nimic fără o acțiune. Ofertele costă, capacitatea echipei este limitată, iar rata de răspuns a fiecărei oferte pentru fiecare motiv de plecare nu este cunoscută dinainte.

**2. Measured.**

*Table: fund_rtap*

| policy             |   offers_sent |   clients_saved |   leavers_lost |   offer_cost |   value_retained |   net_gain |     ROI |
|:-------------------|--------------:|----------------:|---------------:|-------------:|-----------------:|-----------:|--------:|
| no action          |             0 |               0 |            838 |        0.000 |            0.000 |      0.000 | nan     |
| same offer for all |          1645 |             196 |            642 |    63775.000 |       196000.000 | 132225.000 |   3.073 |
| random             |          1645 |             189 |            649 |    77520.000 |       189000.000 | 111480.000 |   2.438 |
| expert rules       |          1645 |             293 |            545 |    70375.000 |       293000.000 | 222625.000 |   4.163 |
| Thompson Sampling  |          1645 |             263 |            575 |    60830.000 |       263000.000 | 202170.000 |   4.324 |

*Table: fund_rtap_learned*

| reason           | offer         |   posterior_mean |   trials |   expected_gain |
|:-----------------|:--------------|-----------------:|---------:|----------------:|
| disengagement    | card_cashback |            0.269 |       24 |         219.231 |
| dissatisfaction  | fee_waiver    |            0.410 |      198 |         370.000 |
| money_withdrawal | personal_call |            0.351 |      246 |         325.806 |
| silence          | personal_call |            0.367 |      107 |         341.972 |

*Table: fund_rtap_fair*

| index      |   card_cashback |   deposit_bonus |   fee_waiver |   personal_call |
|:-----------|----------------:|----------------:|-------------:|----------------:|
| gender=F   |           0.131 |           0.087 |        0.287 |           0.495 |
| gender=M   |           0.121 |           0.081 |        0.273 |           0.525 |
| country=MD |           0.144 |           0.080 |        0.277 |           0.499 |
| country=RO |           0.103 |           0.080 |        0.262 |           0.554 |
| country=UA |           0.094 |           0.107 |        0.329 |           0.470 |

**3. Observed.**
- Thompson Sampling: ROI 4.32, 263 clienți salvați; regulile expertului: ROI 4.16, 293 salvați.
- Aceeași ofertă pentru toți: ROI 3.07: potrivirea ofertei cu motivul plecării contează.
- Politica adaptivă pornește fără nicio cunoaștere și învață din rezultate; regulile expertului presupun că cineva cunoaște deja răspunsurile.

**4. Mitigation.**
- S-a introdus o politică adaptivă (Thompson Sampling pe motiv × ofertă), deoarece ratele de răspuns la oferte nu sunt cunoscute dinainte și trebuie învățate din rezultate.
- S-a impus regula etică (oferta depinde doar de motivul alertei, nu de gen/vârstă/țară) și opt-out-ul de marketing, deoarece decizia automată nu trebuie să discrimineze (GDPR art. 21–22).
- S-au limitat ofertele la capacitatea zilnică și la buget și s-a păstrat suprascrierea umană, deoarece modelul doar propune, omul decide.

**5. Răspunsuri.**
- *De ce o politică adaptivă?* Rata de succes a fiecărei oferte pe fiecare motiv nu este cunoscută; Thompson Sampling o învață din rezultate și echilibrează explorarea cu exploatarea.
- *Ce limite are acțiunea automată?* Buget și capacitate zilnică, opt-out de marketing, oferta depinde doar de motiv (nu de gen, vârstă, țară), iar omul poate suprascrie orice decizie.
- *Ce face RAG? Unde caută, cine definește sursele, de ce e mai aproape de adevăr?* Primește alerta și motivul, caută pasaje în baza de cunoștințe a băncii (playbook de retenție, produse, etică, metodologie), definită și întreținută de bancă, și formulează recomandarea doar din ele, cu sursele citate; nu inventează produse sau reguli.


## 6. Lab experiments

*Table: exp12_noise_bias_xy*

| scenario   |   level |   auc_mean |   auc_std |   f1_mean |   f1_std |   reps |
|:-----------|--------:|-----------:|----------:|----------:|---------:|-------:|
| noise_X    |   0.000 |      0.824 |     0.000 |     0.573 |    0.000 |    100 |
| noise_X    |   0.100 |      0.822 |     0.002 |     0.591 |    0.009 |    100 |
| noise_X    |   0.200 |      0.820 |     0.004 |     0.574 |    0.014 |    100 |
| noise_X    |   0.300 |      0.815 |     0.005 |     0.534 |    0.014 |    100 |
| noise_X    |   0.400 |      0.809 |     0.007 |     0.444 |    0.024 |    100 |
| noise_Y    |   0.000 |      0.824 |     0.000 |     0.573 |    0.000 |    100 |
| noise_Y    |   0.100 |      0.815 |     0.005 |     0.499 |    0.014 |    100 |
| noise_Y    |   0.200 |      0.810 |     0.008 |     0.364 |    0.013 |    100 |
| noise_Y    |   0.300 |      0.796 |     0.016 |     0.288 |    0.004 |    100 |
| noise_Y    |   0.400 |      0.737 |     0.035 |     0.281 |    0.001 |    100 |
| bias_X     |   0.000 |      0.824 |     0.000 |     0.573 |    0.000 |    100 |
| bias_X     |   0.100 |      0.823 |     0.000 |     0.530 |    0.081 |    100 |
| bias_X     |   0.200 |      0.823 |     0.000 |     0.473 |    0.119 |    100 |
| bias_X     |   0.300 |      0.823 |     0.000 |     0.390 |    0.133 |    100 |
| bias_X     |   0.400 |      0.823 |     0.000 |     0.355 |    0.135 |    100 |
| bias_Y     |   0.000 |      0.824 |     0.000 |     0.573 |    0.000 |    100 |
| bias_Y     |   0.100 |      0.821 |     0.002 |     0.600 |    0.007 |    100 |
| bias_Y     |   0.200 |      0.808 |     0.004 |     0.575 |    0.013 |    100 |
| bias_Y     |   0.300 |      0.759 |     0.009 |     0.459 |    0.027 |    100 |
| bias_Y     |   0.400 |      0.581 |     0.000 |     0.326 |    0.000 |    100 |

Text embeddings + PCA: {'method': 'TF-IDF + PCA (TruncatedSVD)', 'variance_kept': 0.08220001245875001}

MLP hidden layer + PCA: {'raw_variance': 0.26082524735764445, 'hidden_variance': 0.24688857635238154, 'raw_silhouette': 0.21639698540439414, 'hidden_silhouette': 0.18957639870825346}

*Table: reg_coefficients*

| feature             |   L1 (Lasso) |   L2 (Ridge) |   ElasticNet a=0.5 |
|:--------------------|-------------:|-------------:|-------------------:|
| tenure_months       |        0.000 |        0.015 |              0.000 |
| credit_score        |        0.000 |        0.019 |              0.000 |
| num_products        |       -0.004 |       -0.037 |             -0.020 |
| has_card            |        0.000 |       -0.039 |             -0.018 |
| is_active_member    |       -0.207 |       -0.215 |             -0.208 |
| est_salary          |       -0.012 |       -0.132 |             -0.073 |
| balance             |       -0.180 |       -0.227 |             -0.203 |
| balance_change_3m   |       -0.914 |       -0.927 |             -0.917 |
| tx_count_90d        |        0.000 |       -0.233 |             -0.109 |
| tx_count_prev_90d   |        0.110 |        0.237 |              0.172 |
| tx_trend            |        0.000 |        0.011 |              0.000 |
| amount_in_90d       |        0.261 |        0.226 |              0.266 |
| amount_out_90d      |        0.030 |        0.274 |              0.132 |
| net_flow_90d        |        0.000 |        0.176 |              0.064 |
| digital_share_90d   |       -0.267 |       -0.274 |             -0.270 |
| msg_count_180d      |        0.000 |        0.101 |              0.037 |
| complaints_180d     |        0.197 |        0.263 |              0.231 |
| neg_msg_180d        |        0.581 |        0.563 |              0.567 |
| avg_sentiment_180d  |        0.000 |        0.024 |              0.000 |
| days_since_last_tx  |        0.222 |        0.237 |              0.229 |
| days_since_last_msg |        0.158 |        0.241 |              0.199 |
| no_feedback_6m      |        0.295 |        0.326 |              0.301 |
| nps_score           |       -1.286 |       -1.289 |             -1.284 |
| nps_missing         |        0.867 |        0.874 |              0.870 |
| silent_decline      |        0.045 |        0.058 |              0.052 |

Features set to 0 by L1 (C=0.05): ['tenure_months', 'credit_score', 'has_card', 'tx_count_90d', 'tx_trend', 'net_flow_90d', 'msg_count_180d', 'avg_sentiment_180d']

*Table: reg_shap_top5*

| L1 (Lasso)        | L2 (Ridge)        | ElasticNet a=0.5   |
|:------------------|:------------------|:-------------------|
| nps_missing       | nps_missing       | nps_missing        |
| balance_change_3m | balance_change_3m | balance_change_3m  |
| nps_score         | nps_score         | nps_score          |
| neg_msg_180d      | neg_msg_180d      | neg_msg_180d       |
| no_feedback_6m    | no_feedback_6m    | no_feedback_6m     |

*Table: reg_sweep*

|   l1_ratio |      C |    f1 |   auc |   fit_ms |   latency_us_per_client |   nonzero_coefs |
|-----------:|-------:|------:|------:|---------:|------------------------:|----------------:|
|      0.000 |  0.001 | 0.575 | 0.857 |   18.400 |                   4.887 |          25.000 |
|      0.000 |  0.003 | 0.509 | 0.862 |   16.406 |                   4.817 |          25.000 |
|      0.000 |  0.010 | 0.577 | 0.863 |   17.147 |                   4.858 |          25.000 |
|      0.000 |  0.032 | 0.592 | 0.863 |   17.221 |                   4.676 |          25.000 |
|      0.000 |  0.100 | 0.593 | 0.863 |   21.191 |                   5.347 |          25.000 |
|      0.000 |  0.316 | 0.597 | 0.863 |   18.072 |                   4.740 |          25.000 |
|      0.000 |  1.000 | 0.597 | 0.863 |   18.361 |                   5.015 |          25.000 |
|      0.000 |  3.162 | 0.597 | 0.863 |   18.560 |                   4.950 |          25.000 |
|      0.000 | 10.000 | 0.596 | 0.863 |   18.764 |                   4.751 |          25.000 |
|      0.250 |  0.001 | 0.583 | 0.818 |   25.814 |                   4.849 |           7.000 |
|      0.250 |  0.003 | 0.626 | 0.861 |   32.340 |                   4.950 |          15.000 |
|      0.250 |  0.010 | 0.548 | 0.864 |   37.499 |                   5.058 |          17.000 |
|      0.250 |  0.032 | 0.594 | 0.865 |   52.896 |                   5.214 |          22.000 |
|      0.250 |  0.100 | 0.598 | 0.863 |   74.291 |                   5.300 |          25.000 |
|      0.250 |  0.316 | 0.597 | 0.863 |   63.452 |                   5.362 |          25.000 |
|      0.250 |  1.000 | 0.596 | 0.863 |   64.542 |                   5.179 |          25.000 |
|      0.250 |  3.162 | 0.596 | 0.863 |   79.259 |                   5.759 |          25.000 |
|      0.250 | 10.000 | 0.596 | 0.863 |   72.913 |                   6.552 |          25.000 |
|      0.500 |  0.001 | 0.524 | 0.809 |   30.982 |                   6.327 |           3.000 |
|      0.500 |  0.003 | 0.618 | 0.845 |   25.744 |                   5.200 |          10.000 |
|      0.500 |  0.010 | 0.544 | 0.864 |   48.163 |                   7.511 |          15.000 |
|      0.500 |  0.032 | 0.565 | 0.866 |   84.820 |                   6.012 |          20.000 |
|      0.500 |  0.100 | 0.596 | 0.864 |  204.863 |                   6.170 |          24.000 |
|      0.500 |  0.316 | 0.597 | 0.863 |  218.286 |                   5.650 |          25.000 |
|      0.500 |  1.000 | 0.596 | 0.863 |   66.122 |                   5.432 |          25.000 |
|      0.500 |  3.162 | 0.596 | 0.863 |   68.526 |                   5.517 |          25.000 |
|      0.500 | 10.000 | 0.596 | 0.863 |   71.003 |                   5.493 |          25.000 |
|      0.750 |  0.001 | 0.367 | 0.712 |   33.106 |                   5.591 |           1.000 |
|      0.750 |  0.003 | 0.567 | 0.823 |   28.697 |                   5.122 |           7.000 |
|      0.750 |  0.010 | 0.577 | 0.864 |   33.868 |                   4.892 |          13.000 |
|      0.750 |  0.032 | 0.561 | 0.866 |   39.618 |                   5.129 |          15.000 |
|      0.750 |  0.100 | 0.585 | 0.865 |  163.770 |                   8.493 |          23.000 |
|      0.750 |  0.316 | 0.590 | 0.863 |  497.855 |                   5.777 |          24.000 |
|      0.750 |  1.000 | 0.596 | 0.863 |  242.260 |                   6.197 |          25.000 |
|      0.750 |  3.162 | 0.597 | 0.863 |   67.963 |                   5.389 |          25.000 |
|      0.750 | 10.000 | 0.596 | 0.863 |   70.229 |                   5.290 |          25.000 |
|      1.000 |  0.001 | 0.000 | 0.500 |   19.960 |                   5.423 |           0.000 |
|      1.000 |  0.003 | 0.526 | 0.809 |   26.639 |                   5.257 |           4.000 |
|      1.000 |  0.010 | 0.629 | 0.861 |   39.058 |                   5.321 |          11.000 |
|      1.000 |  0.032 | 0.549 | 0.866 |   39.166 |                   4.927 |          15.000 |
|      1.000 |  0.100 | 0.601 | 0.865 |   69.637 |                   5.079 |          22.000 |
|      1.000 |  0.316 | 0.592 | 0.863 |  204.996 |                   5.671 |          24.000 |
|      1.000 |  1.000 | 0.593 | 0.863 |  615.401 |                   5.719 |          24.000 |
|      1.000 |  3.162 | 0.597 | 0.863 |   66.736 |                   5.259 |          25.000 |
|      1.000 | 10.000 | 0.596 | 0.863 |   68.272 |                   5.421 |          25.000 |

*Table: expA_toy*

| model      |      C |    f1 |   fit_ms | zeros                                                                                      |   noise_features_zeroed |   informative_zeroed |
|:-----------|-------:|------:|---------:|:-------------------------------------------------------------------------------------------|------------------------:|---------------------:|
| L1         |  0.001 | 0.000 |    2.596 | inf_0, inf_1, inf_2, inf_3, inf_4, inf_5, red_0, red_1, noise_0, noise_1, noise_2, noise_3 |                       4 |                    6 |
| ElasticNet |  0.001 | 0.818 |    4.886 | inf_0, inf_1, inf_3, inf_4, inf_5, red_0, red_1, noise_0, noise_1, noise_2, noise_3        |                       4 |                    5 |
| L2         |  0.001 | 0.873 |    2.522 |                                                                                            |                       0 |                    0 |
| L1         |  0.003 | 0.830 |    4.861 | inf_0, inf_1, inf_3, inf_5, red_0, red_1, noise_0, noise_1, noise_2, noise_3               |                       4 |                    4 |
| ElasticNet |  0.003 | 0.861 |    4.570 | inf_1, inf_3, inf_5, red_0, noise_0, noise_1, noise_2, noise_3                             |                       4 |                    3 |
| L2         |  0.003 | 0.882 |    2.186 |                                                                                            |                       0 |                    0 |
| L1         |  0.010 | 0.879 |    4.817 | inf_3, inf_5, red_0, noise_0, noise_1, noise_2, noise_3                                    |                       4 |                    2 |
| ElasticNet |  0.010 | 0.884 |    4.916 | inf_5, red_0, noise_0, noise_1, noise_2, noise_3                                           |                       4 |                    1 |
| L2         |  0.010 | 0.891 |    2.320 |                                                                                            |                       0 |                    0 |
| L1         |  0.032 | 0.890 |    5.113 | red_0, red_1, noise_0, noise_1, noise_2, noise_3                                           |                       4 |                    0 |
| ElasticNet |  0.032 | 0.894 |    5.253 | red_0, noise_0, noise_2, noise_3                                                           |                       3 |                    0 |
| L2         |  0.032 | 0.895 |    2.435 |                                                                                            |                       0 |                    0 |
| L1         |  0.100 | 0.894 |    9.345 | red_0, red_1, noise_0, noise_2                                                             |                       2 |                    0 |
| ElasticNet |  0.100 | 0.895 |   10.657 | noise_2                                                                                    |                       1 |                    0 |
| L2         |  0.100 | 0.893 |    2.579 |                                                                                            |                       0 |                    0 |
| L1         |  0.316 | 0.894 |   25.450 | red_0, red_1                                                                               |                       0 |                    0 |
| ElasticNet |  0.316 | 0.894 |   20.043 |                                                                                            |                       0 |                    0 |
| L2         |  0.316 | 0.894 |    2.618 |                                                                                            |                       0 |                    0 |
| L1         |  1.000 | 0.892 |   78.704 | red_0, red_1                                                                               |                       0 |                    0 |
| ElasticNet |  1.000 | 0.893 |   39.693 |                                                                                            |                       0 |                    0 |
| L2         |  1.000 | 0.893 |    2.817 |                                                                                            |                       0 |                    0 |
| L1         |  3.162 | 0.892 |  365.589 | red_0, red_1                                                                               |                       0 |                    0 |
| ElasticNet |  3.162 | 0.892 |   92.297 |                                                                                            |                       0 |                    0 |
| L2         |  3.162 | 0.893 |    2.911 |                                                                                            |                       0 |                    0 |
| L1         | 10.000 | 0.892 |  807.609 | red_0, red_1                                                                               |                       0 |                    0 |
| ElasticNet | 10.000 | 0.892 |   60.510 |                                                                                            |                       0 |                    0 |
| L2         | 10.000 | 0.892 |    3.036 |                                                                                            |                       0 |                    0 |

github file not found in data/external/ -> Experiment B runs on the simulated bank's profile features.

*Table: expB_synthetic*

| source         | method                                                    |   mean_KS |   mean_abs_corr_diff |   AUC train-real/test-real (TRTR) |   AUC train-synthetic/test-real (TSTR) |   median DCR synthetic->real |   median real->real NN distance |   share synthetic closer than 5th pct real NN |
|:---------------|:----------------------------------------------------------|----------:|---------------------:|----------------------------------:|---------------------------------------:|-----------------------------:|--------------------------------:|----------------------------------------------:|
| simulated bank | own Gaussian copula (pip install sdv for the SDV version) |     0.014 |                0.018 |                             0.648 |                                  0.505 |                        0.648 |                           0.663 |                                         0.057 |

*Table: sources_matrix*

| trained on \ tested on   |   simulated bank |   external branch |   learned synthetic |
|:-------------------------|-----------------:|------------------:|--------------------:|
| simulated bank           |            0.648 |             0.668 |               0.560 |
| external branch          |            0.609 |             0.666 |               0.521 |
| learned synthetic        |            0.488 |             0.562 |               0.537 |

*Table: sources_features*

| source            | L1 keeps (C=0.02)              | SHAP top-5                                               |
|:------------------|:-------------------------------|:---------------------------------------------------------|
| simulated bank    | age, balance, is_active_member | num_products, age, balance, is_active_member, est_salary |
| external branch   |                                | balance, age, num_products, est_salary, credit_score     |
| learned synthetic | age, is_active_member          | age, balance, credit_score, est_salary, tenure_months    |

Common features used across sources: ['age', 'balance', 'credit_score', 'est_salary', 'gender', 'has_card', 'is_active_member', 'num_products', 'tenure_months']


## 9. Monitoring and model card

*Table: monitoring*

| month_cutoff   |   clients |   churn_rate |   roc_auc |   recall |   max_psi | feature_max_psi     | shap_top5                                                                    |   recall_gap_age |   shap_top5_jaccard_vs_training | RETRAIN?   |
|:---------------|----------:|-------------:|----------:|---------:|----------:|:--------------------|:-----------------------------------------------------------------------------|-----------------:|--------------------------------:|:-----------|
| 2025-07-31     |      4646 |        0.055 |     0.817 |    0.594 |     0.384 | days_since_last_msg | nps_score, balance_change_3m, nps_missing, num_products, days_since_last_msg |            0.167 |                           1.000 | yes        |
| 2025-08-31     |      4673 |        0.082 |     0.864 |    0.724 |     0.194 | balance_change_3m   | nps_score, balance_change_3m, nps_missing, num_products, days_since_last_msg |            0.166 |                           1.000 | no         |
| 2025-09-30     |      4682 |        0.110 |     0.886 |    0.762 |     0.145 | balance_change_3m   | nps_score, balance_change_3m, nps_missing, num_products, days_since_last_msg |            0.143 |                           1.000 | no         |
| 2025-10-31     |      4559 |        0.124 |     0.882 |    0.764 |     0.084 | balance_change_3m   | nps_score, balance_change_3m, nps_missing, num_products, days_since_last_msg |            0.145 |                           1.000 | no         |
| 2025-11-30     |      4402 |        0.139 |     0.905 |    0.775 |     0.027 | balance_change_3m   | nps_score, balance_change_3m, nps_missing, num_products, days_since_last_msg |            0.129 |                           1.000 | no         |
| 2025-12-31     |      4213 |        0.163 |     0.950 |    0.849 |     0.001 | tx_count_90d        | nps_score, balance_change_3m, nps_missing, num_products, days_since_last_msg |            0.060 |                           1.000 | no         |
| 2026-01-31     |      3979 |        0.134 |     0.918 |    0.828 |     0.012 | balance_change_3m   | balance_change_3m, nps_score, nps_missing, num_products, days_since_last_msg |            0.104 |                           1.000 | no         |
| 2026-02-28     |      3789 |        0.113 |     0.929 |    0.837 |     0.037 | tx_trend            | balance_change_3m, nps_score, nps_missing, num_products, days_since_last_msg |            0.106 |                           1.000 | no         |
| 2026-03-31     |      3525 |        0.076 |     0.901 |    0.705 |     0.100 | balance_change_3m   | nps_score, balance_change_3m, nps_missing, num_products, days_since_last_msg |            0.070 |                           1.000 | no         |

Model card written to reports/model_card.md
