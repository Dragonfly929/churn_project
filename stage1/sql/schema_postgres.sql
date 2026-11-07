-- =====================================================================
-- Stage 1 — DWH star schema (PostgreSQL)
-- Dimensions: dim_client, dim_date, dim_channel
-- Facts: fact_account_month (monthly snapshot), fact_transaction,
--        fact_feedback, fact_survey, fact_client_exit (the label source)
-- Primary + foreign keys are declared so pgAdmin / DBeaver draw the ER diagram.
-- =====================================================================
DROP TABLE IF EXISTS fact_client_exit CASCADE;
DROP TABLE IF EXISTS fact_survey CASCADE;
DROP TABLE IF EXISTS fact_feedback CASCADE;
DROP TABLE IF EXISTS fact_transaction CASCADE;
DROP TABLE IF EXISTS fact_account_month CASCADE;
DROP TABLE IF EXISTS dim_channel CASCADE;
DROP TABLE IF EXISTS dim_date CASCADE;
DROP TABLE IF EXISTS dim_client CASCADE;

CREATE TABLE dim_client (
    client_key        INTEGER PRIMARY KEY,
    client_id         INTEGER NOT NULL UNIQUE,
    birth_date        DATE,
    gender            CHAR(1)  CHECK (gender IN ('F','M')),
    country           CHAR(2)  CHECK (country IN ('MD','RO','UA')),
    join_date         DATE NOT NULL,
    marketing_opt_out SMALLINT NOT NULL DEFAULT 0
);

CREATE TABLE dim_date (
    date_key     INTEGER PRIMARY KEY,          -- yyyymmdd
    full_date    DATE NOT NULL UNIQUE,
    year         SMALLINT, quarter SMALLINT, month SMALLINT, day SMALLINT,
    day_of_week  SMALLINT, is_month_end SMALLINT
);

CREATE TABLE dim_channel (
    channel_key   SMALLINT PRIMARY KEY,
    channel_name  VARCHAR(20) NOT NULL UNIQUE,
    channel_group VARCHAR(20) NOT NULL,        -- payments / contact
    is_digital    SMALLINT NOT NULL
);

CREATE TABLE fact_account_month (
    client_key       INTEGER NOT NULL REFERENCES dim_client(client_key),
    date_key         INTEGER NOT NULL REFERENCES dim_date(date_key),
    balance          NUMERIC(16,2),
    num_products     SMALLINT,
    credit_score     NUMERIC(6,1),
    has_card         SMALLINT,
    is_active_member SMALLINT,
    est_salary       NUMERIC(14,2),
    status           VARCHAR(10) NOT NULL,
    PRIMARY KEY (client_key, date_key)
);

CREATE TABLE fact_transaction (
    tx_id       BIGINT PRIMARY KEY,
    client_key  INTEGER NOT NULL REFERENCES dim_client(client_key),
    date_key    INTEGER NOT NULL REFERENCES dim_date(date_key),
    channel_key SMALLINT NOT NULL REFERENCES dim_channel(channel_key),
    tx_ts       TIMESTAMP NOT NULL,
    amount      NUMERIC(14,2) NOT NULL CHECK (amount > 0),
    direction   VARCHAR(3) NOT NULL CHECK (direction IN ('in','out'))
);

CREATE TABLE fact_feedback (
    msg_id          BIGINT PRIMARY KEY,
    client_key      INTEGER NOT NULL REFERENCES dim_client(client_key),
    date_key        INTEGER NOT NULL REFERENCES dim_date(date_key),
    channel_key     SMALLINT NOT NULL REFERENCES dim_channel(channel_key),
    msg_ts          TIMESTAMP NOT NULL,
    text_raw        TEXT,
    text_clean      TEXT,
    sentiment_raw   REAL,
    sentiment_score REAL,
    is_negative     SMALLINT,
    is_complaint    SMALLINT,
    is_empty        SMALLINT
);

CREATE TABLE fact_survey (
    survey_id         BIGINT PRIMARY KEY,
    client_key        INTEGER NOT NULL REFERENCES dim_client(client_key),
    sent_date_key     INTEGER NOT NULL REFERENCES dim_date(date_key),
    answered_date_key INTEGER REFERENCES dim_date(date_key),
    nps_score         SMALLINT CHECK (nps_score BETWEEN 0 AND 10)
);

CREATE TABLE fact_client_exit (
    client_key       INTEGER PRIMARY KEY REFERENCES dim_client(client_key),
    exit_date_key    INTEGER NOT NULL REFERENCES dim_date(date_key),
    exit_channel_key SMALLINT REFERENCES dim_channel(channel_key)
);

CREATE INDEX ix_tx_client_date   ON fact_transaction (client_key, date_key);
CREATE INDEX ix_tx_date          ON fact_transaction (date_key);
CREATE INDEX ix_fb_client_date   ON fact_feedback (client_key, date_key);
CREATE INDEX ix_sv_client_date   ON fact_survey (client_key, sent_date_key);
CREATE INDEX ix_acc_date         ON fact_account_month (date_key);
