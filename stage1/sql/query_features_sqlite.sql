-- =====================================================================
-- Stage 1.6 — THE QUERY: DWH  ->  one row per client (SQLite dialect)
-- Parameters
--   :cutoff          eligibility date; features may only use data <= cutoff
--   :label_cutoff    target = client leaves in (cutoff, label_cutoff]  (90 days)
--   :feature_cutoff  normally = :cutoff. The leakage demo sets it AFTER the departures.
--   :denoise         1 = sentiment on denoised text, 0 = raw text (Noise fundamental)
-- One CTE per feature group, LEFT JOINs so clients with no activity still appear.
-- =====================================================================
WITH params AS (
    SELECT :cutoff AS cutoff, :label_cutoff AS label_cutoff,
           :feature_cutoff AS fcut, CAST(:denoise AS INTEGER) AS denoise
),
-- eligible population: joined by the cutoff and still a client at the cutoff
eligible AS (
    SELECT c.client_key, c.client_id, c.gender, c.country, c.marketing_opt_out,
           CAST((julianday(p.cutoff) - julianday(c.birth_date)) / 365.25 AS INTEGER) AS age,
           CAST((julianday(p.cutoff) - julianday(c.join_date)) / 30.44 AS INTEGER) AS tenure_months
    FROM dim_client c
    CROSS JOIN params p
    LEFT JOIN fact_client_exit e ON e.client_key = c.client_key
    LEFT JOIN dim_date de        ON de.date_key = e.exit_date_key
    WHERE c.join_date <= p.cutoff
      AND (de.full_date IS NULL OR de.full_date > p.cutoff)
),
-- PROFILE: last monthly snapshot at or before the feature cutoff (rn=1) and 3 months earlier (rn=4)
snap AS (
    SELECT f.client_key, f.balance, f.num_products, f.credit_score, f.has_card,
           f.is_active_member, f.est_salary,
           ROW_NUMBER() OVER (PARTITION BY f.client_key ORDER BY d.full_date DESC) AS rn
    FROM fact_account_month f
    JOIN dim_date d ON d.date_key = f.date_key
    CROSS JOIN params p
    WHERE d.full_date <= p.fcut
),
-- FINANCIAL BEHAVIOUR: last 90 days vs the 90 days before
tx AS (
    SELECT t.client_key,
        SUM(CASE WHEN d.full_date >  date(p.fcut, '-90 day') THEN 1 ELSE 0 END)                                         AS tx_count_90d,
        SUM(CASE WHEN d.full_date <= date(p.fcut, '-90 day') AND d.full_date > date(p.fcut, '-180 day') THEN 1 ELSE 0 END) AS tx_count_prev_90d,
        SUM(CASE WHEN d.full_date >  date(p.fcut, '-90 day') AND t.direction = 'in'  THEN t.amount ELSE 0 END)          AS amount_in_90d,
        SUM(CASE WHEN d.full_date >  date(p.fcut, '-90 day') AND t.direction = 'out' THEN t.amount ELSE 0 END)          AS amount_out_90d,
        SUM(CASE WHEN d.full_date >  date(p.fcut, '-90 day') AND ch.is_digital = 1 THEN 1 ELSE 0 END)                    AS digital_tx_90d,
        MAX(d.full_date) AS last_tx_date
    FROM fact_transaction t
    JOIN dim_date d     ON d.date_key = t.date_key
    JOIN dim_channel ch ON ch.channel_key = t.channel_key
    CROSS JOIN params p
    WHERE d.full_date <= p.fcut AND d.full_date > date(p.fcut, '-365 day')
    GROUP BY t.client_key
),
-- VOICE OF THE CLIENT: messages in the last 180 days
fb AS (
    SELECT f.client_key,
        SUM(CASE WHEN d.full_date > date(p.fcut, '-180 day') THEN 1 ELSE 0 END)                  AS msg_count_180d,
        SUM(CASE WHEN d.full_date > date(p.fcut, '-180 day') THEN f.is_complaint ELSE 0 END)     AS complaints_180d,
        SUM(CASE WHEN d.full_date > date(p.fcut, '-180 day') THEN f.is_negative ELSE 0 END)      AS neg_msg_180d,
        AVG(CASE WHEN d.full_date > date(p.fcut, '-180 day') AND f.is_empty = 0
                 THEN (CASE WHEN p.denoise = 1 THEN f.sentiment_score ELSE f.sentiment_raw END) END) AS avg_sentiment_180d,
        MAX(d.full_date) AS last_msg_date
    FROM fact_feedback f
    JOIN dim_date d ON d.date_key = f.date_key
    CROSS JOIN params p
    WHERE d.full_date <= p.fcut AND d.full_date > date(p.fcut, '-365 day')
    GROUP BY f.client_key
),
-- SILENCE (survey): the last survey whose 14-day answer window closed before the cutoff;
-- answered only if answered by the cutoff (a missing answer = silence, NOT random)
sv AS (
    SELECT s.client_key,
           CASE WHEN da.full_date IS NOT NULL AND da.full_date <= p.fcut THEN s.nps_score END AS nps_score,
           ROW_NUMBER() OVER (PARTITION BY s.client_key ORDER BY ds.full_date DESC) AS rn
    FROM fact_survey s
    JOIN dim_date ds      ON ds.date_key = s.sent_date_key
    LEFT JOIN dim_date da ON da.date_key = s.answered_date_key
    CROSS JOIN params p
    WHERE ds.full_date <= date(p.fcut, '-15 day') AND ds.full_date > date(p.fcut, '-120 day')
),
-- TARGET: leaves in the 90 days AFTER the cutoff (never read before it)
label AS (
    SELECT e.client_key, 1 AS churn
    FROM fact_client_exit e
    JOIN dim_date d ON d.date_key = e.exit_date_key
    CROSS JOIN params p
    WHERE d.full_date > p.cutoff AND d.full_date <= p.label_cutoff
)
SELECT
    el.client_id,
    -- profile
    el.age, el.gender, el.country, el.tenure_months,
    s1.credit_score, s1.num_products, s1.has_card, s1.is_active_member, s1.est_salary,
    s1.balance,
    CASE WHEN s4.balance > 0 THEN ROUND((s1.balance - s4.balance) * 1.0 / s4.balance, 4) ELSE 0 END AS balance_change_3m,
    -- financial behaviour
    COALESCE(tx.tx_count_90d, 0)      AS tx_count_90d,
    COALESCE(tx.tx_count_prev_90d, 0) AS tx_count_prev_90d,
    ROUND((COALESCE(tx.tx_count_90d, 0) + 1.0) / (COALESCE(tx.tx_count_prev_90d, 0) + 1.0), 4) AS tx_trend,
    ROUND(COALESCE(tx.amount_in_90d, 0), 2)  AS amount_in_90d,
    ROUND(COALESCE(tx.amount_out_90d, 0), 2) AS amount_out_90d,
    ROUND(COALESCE(tx.amount_in_90d, 0) - COALESCE(tx.amount_out_90d, 0), 2) AS net_flow_90d,
    ROUND(CASE WHEN COALESCE(tx.tx_count_90d, 0) > 0 THEN 1.0 * tx.digital_tx_90d / tx.tx_count_90d ELSE 0 END, 4) AS digital_share_90d,
    -- voice of the client
    COALESCE(fb.msg_count_180d, 0)  AS msg_count_180d,
    COALESCE(fb.complaints_180d, 0) AS complaints_180d,
    COALESCE(fb.neg_msg_180d, 0)    AS neg_msg_180d,
    ROUND(COALESCE(fb.avg_sentiment_180d, 0), 4) AS avg_sentiment_180d,
    -- silence
    CAST(COALESCE(julianday(p.fcut) - julianday(tx.last_tx_date), 365) AS INTEGER)  AS days_since_last_tx,
    CAST(COALESCE(julianday(p.fcut) - julianday(fb.last_msg_date), 365) AS INTEGER) AS days_since_last_msg,
    CASE WHEN COALESCE(fb.msg_count_180d, 0) = 0 THEN 1 ELSE 0 END AS no_feedback_6m,
    sv.nps_score,
    CASE WHEN sv.nps_score IS NULL THEN 1 ELSE 0 END AS nps_missing,
    CASE WHEN (COALESCE(tx.tx_count_90d, 0) + 1.0) / (COALESCE(tx.tx_count_prev_90d, 0) + 1.0) < 0.5
          AND COALESCE(fb.msg_count_180d, 0) = 0 THEN 1 ELSE 0 END AS silent_decline,
    el.marketing_opt_out,
    -- target
    COALESCE(lb.churn, 0) AS churn
FROM eligible el
CROSS JOIN params p
LEFT JOIN snap s1  ON s1.client_key = el.client_key AND s1.rn = 1
LEFT JOIN snap s4  ON s4.client_key = el.client_key AND s4.rn = 4
LEFT JOIN tx       ON tx.client_key = el.client_key
LEFT JOIN fb       ON fb.client_key = el.client_key
LEFT JOIN sv       ON sv.client_key = el.client_key AND sv.rn = 1
LEFT JOIN label lb ON lb.client_key = el.client_key
ORDER BY el.client_id
