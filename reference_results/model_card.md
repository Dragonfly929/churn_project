# Model card — churn early warning (iteration 3)
- Date: 2026-11-07   - Owner: student (ȘAD)
- Purpose: rank active clients by risk of leaving in the next 90 days and propose a retention action. The model only PROPOSES; an employee decides (GDPR Art. 22).
- Data: simulated bank DWH (star schema), Stage 1 query at cutoff 2025-12-31, label window to 2026-03-31. Synthetic data: no real client is processed.
- Model: gb with balancing 'none', 24 features: tenure_months, credit_score, num_products, has_card, is_active_member, est_salary, balance, balance_change_3m, tx_count_90d, tx_count_prev_90d, tx_trend, amount_in_90d, amount_out_90d, net_flow_90d, msg_count_180d, complaints_180d, neg_msg_180d, avg_sentiment_180d, days_since_last_tx, days_since_last_msg, no_feedback_6m, nps_score, nps_missing, silent_decline
- Decision threshold (from the cost matrix FN=1000 €, FP=60 €): 0.15; per-group thresholds: none
- Test metrics: ROC-AUC 0.888, recall 0.725, precision 0.535, expected cost 114220 € vs 138000 € without model
- Contract matrix:

| requirement                                   | threshold      |   measured | verdict   |
|:----------------------------------------------|:---------------|-----------:|:----------|
| M1 recall of leavers                          | >= 0.70        |      0.725 | PASS      |
| M2a ROC-AUC                                   | >= 0.80        |      0.888 | PASS      |
| M2b expected cost vs no model (EUR)           | < 138000       | 114220.000 | PASS      |
| Fairness: max recall gap (age/gender/country) | <= 0.10        |      0.080 | PASS      |
| Leakage alarm: AUC suspiciously high          | < 0.97         |      0.888 | PASS      |
| F1 XAI stability (Jaccard top-5)              | >= 0.60        |      1.000 | PASS      |
| F2a external SHAP rank correlation            | >= 0.80        |      0.999 | PASS      |
| F2b external ROC-AUC                          | >= 0.80        |      0.906 | PASS      |
| F2c max PSI (drift) external                  | < 0.20         |      0.023 | PASS      |
| Data contract (input validity)                | all rules pass |      1.000 | PASS      |

- Limits: simulated clients and simulated reactions to offers; the sentiment lexicon was built for the message templates of the simulator; fairness measured only for age group, gender, country.
- Monitoring: monthly PSI per feature (alert >= 0.2), recall and lead time when exits are known, SHAP top-5 drift, recall gap by group; retrain when any crosses its threshold, then champion vs challenger by the contract matrix.